"""爬虫包：来源注册、并发抓取、关键字过滤、去重、薪资统计、入库。

对外只暴露一个入口：``crawl_for_profile(profile)``。
它把「用户档案 → 真实岗位 → 薪资行情 → 落库」整条链路串起来。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config import get_settings

from ..db import repository, session_scope
from ..salary import SalaryRange, parse_salary, summarize_salaries
from . import generic_render, public_api, render_api
from .base import (
    DEFAULT_SOURCES,
    BaseCrawler,
    CrawlBudget,
    CrawlBudgetExceeded,
    JobItem,
    SourceResult,
    build_job_key,
    load_sources,
    normalize_title,
    normalize_url,
    random_user_agent,
    save_sources,
    sources_file_path,
)
from .matching import (
    city_matches,
    dedupe_by_title,
    expand_keywords,
    normalize_city,
    search_terms,
    split_keywords,
    title_matches,
)

logger = logging.getLogger("ai_resume_helper.crawlers")

__all__ = [
    "CrawlReport",
    "DEFAULT_SOURCES",
    "JobItem",
    "SourceResult",
    "build_crawler",
    "is_crawlable",
    "crawl_for_profile",
    "load_sources",
    "save_sources",
    "sources_file_path",
    "test_source",
]

# 每个来源单独的时间预算（秒）
PER_SOURCE_BUDGET = 90.0


def build_crawler(source: dict[str, Any]) -> BaseCrawler:
    """按 ``kind`` 实例化对应的爬虫。"""
    kind = str(source.get("kind") or "generic_render")
    if kind == "public_api":
        crawler = public_api.build(source)
        if crawler is None:
            adapter = (source.get("config") or {}).get("adapter") or source.get("key")
            raise ValueError(f"没有名为 {adapter!r} 的公开接口适配器")
        return crawler
    if kind == "generic_render":
        return generic_render.GenericRenderCrawler(source)
    if kind == "render_api":
        return render_api.PlaywrightApiCrawler(source)
    raise ValueError(f"未知的爬虫类型：{kind}")


#: 不需要用户填 ``list_url`` 就能工作的爬虫类型（自带检索入口）
_SELF_CONTAINED_KINDS = {"public_api", "render_api"}


def is_crawlable(source: dict[str, Any]) -> bool:
    """判断一个来源是否具备抓取条件（避免「已启用但没填地址」的空转一次）。"""
    if not source.get("enabled", True):
        return False
    if str(source.get("kind") or "generic_render") in _SELF_CONTAINED_KINDS:
        return True
    config = source.get("config") or {}
    return bool(config.get("list_url") or source.get("careers_url"))


@dataclass
class CrawlReport:
    """一次抓取的完整结果（成功失败都要如实呈现）。"""

    role: str
    city: str
    keywords: list[str] = field(default_factory=list)
    keyword_expansion: dict[str, list[str]] = field(default_factory=dict)
    jobs: list[JobItem] = field(default_factory=list)
    sources: list[SourceResult] = field(default_factory=list)
    salary: dict[str, Any] = field(default_factory=dict)
    saved: int = 0
    deduped: int = 0
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, job_limit: int = 60) -> dict[str, Any]:
        jobs = [job.to_dict() for job in self.jobs[:job_limit]]
        return {
            "role": self.role,
            "city": self.city,
            "keywords": self.keywords,
            "keyword_expansion": self.keyword_expansion,
            "total_jobs": len(self.jobs),
            "with_salary": sum(1 for job in self.jobs if job.salary_mid_k is not None),
            "saved": self.saved,
            "deduped": self.deduped,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "salary": self.salary,
            "sources": [item.to_dict() for item in self.sources],
            "jobs": jobs,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------- #
# 单来源抓取（同步，跑在线程里）
# ---------------------------------------------------------------------- #
def _run_source(
    source: dict[str, Any], keywords: list[str], city: str, budget_seconds: float
) -> SourceResult:
    started = time.perf_counter()
    label = str(source.get("label") or source.get("key") or "未知来源")
    key = str(source.get("key") or label)
    try:
        crawler = build_crawler(source)
        # 把过滤条件透传给爬虫：
        #   _keywords       通用渲染爬虫靠它决定要不要补抓详情页
        #   _search_terms   公开接口类爬虫用它逐级放宽地做**服务端检索**
        #                   （完整方向名常常搜不到东西，见 matching.search_terms）
        #   _search_keyword 只给不支持多词检索的适配器兜底
        #   _city           服务端/本地城市过滤
        terms = search_terms(keywords)
        crawler.config["_keywords"] = keywords
        crawler.config["_search_terms"] = terms
        crawler.config["_search_keyword"] = terms[0] if terms else ""
        crawler.config["_city"] = city
        with CrawlBudget(budget_seconds) as budget:
            jobs = crawler.fetch(budget)
        return SourceResult(
            source=key,
            label=label,
            ok=True,
            jobs=jobs,
            fetched=len(jobs),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    except CrawlBudgetExceeded as exc:
        return SourceResult(
            source=key, label=label, ok=False, error=f"超时：{exc}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # noqa: BLE001 - 单个来源失败不能拖垮整体
        logger.warning("来源 %s 抓取失败：%s", label, exc)
        return SourceResult(
            source=key,
            label=label,
            ok=False,
            error=f"{type(exc).__name__}: {exc}"[:300],
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


async def _crawl_sources(
    sources: list[dict[str, Any]], keywords: list[str], city: str, concurrency: int
) -> list[SourceResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(source: dict[str, Any]) -> SourceResult:
        async with semaphore:
            return await asyncio.to_thread(_run_source, source, keywords, city, PER_SOURCE_BUDGET)

    return list(await asyncio.gather(*(run(source) for source in sources)))


# ---------------------------------------------------------------------- #
# 主入口
# ---------------------------------------------------------------------- #
async def crawl_for_profile(
    profile: Any,
    *,
    source_keys: list[str] | None = None,
    extra_keywords: list[str] | None = None,
    save: bool = True,
) -> CrawlReport:
    """按用户档案抓取真实岗位并统计薪资。

    参数
        profile       : ``backend.profile.UserProfileData``
        source_keys   : 只抓这些来源；None 表示全部启用的来源
        extra_keywords: 额外关键字（前端可让用户补充）
        save          : 是否写入数据库
    """
    settings = get_settings()
    started = time.perf_counter()

    role = profile.target_role
    city = profile.expect_city
    keywords = list(profile.search_keywords())
    for extra in extra_keywords or []:
        token = str(extra).strip().lower()
        if token and token not in keywords:
            keywords.append(token)

    match_terms, expansion = expand_keywords(keywords)

    all_sources = load_sources()
    picked = [
        source
        for source in all_sources
        if is_crawlable(source)
        and (not source_keys or str(source.get("key")) in source_keys)
    ]

    report = CrawlReport(
        role=role,
        city=city,
        keywords=keywords,
        keyword_expansion=expansion,
    )

    if not picked:
        report.warnings.append(
            "没有可用的岗位来源。请在「岗位来源」里启用至少一个，或添加自定义官网。"
        )
        report.salary = summarize_salaries([], profile.expect_salary_min, profile.expect_salary_max)
        report.elapsed_ms = (time.perf_counter() - started) * 1000
        return report
    if not keywords:
        report.warnings.append("求职方向为空，无法按职位名匹配岗位。")

    results = await _crawl_sources(picked, keywords, city, settings.crawler_concurrency)
    report.sources = results

    # ---- 合并、按职位名过滤、按城市过滤、去重 ----
    #
    # 去重分两层，缺一不可：
    #
    #   1. job_key（来源+公司+标题+规范化URL 的哈希）—— 同一岗位重复抓到
    #   2. identity（标题+公司+城市）—— 同一岗位被以**不同 job_id 重复发布**。
    #      招聘站上很常见（不同 HR 各发一遍），只按 job_key 去重拦不住，
    #      结果会把薪资统计算歪（同一条样本被计两次）。
    #
    # identity 冲突时保留「有薪资 / 有 JD 正文」的那条，信息更全。
    by_key: dict[str, JobItem] = {}
    by_identity: dict[str, JobItem] = {}
    max_per_title = max(4, int(settings.crawler_max_jobs) // 8)

    def _identity(job: JobItem) -> str:
        return "|".join(
            (
                normalize_title(job.title),
                (job.company or "").strip().casefold(),
                normalize_city(job.city, city).casefold(),
            )
        )

    def _richness(job: JobItem) -> tuple[int, int, int]:
        return (
            1 if job.salary_mid_k is not None else 0,
            1 if job.has_detail else 0,
            len(job.jd_text or ""),
        )

    title_counts: dict[str, int] = {}
    for result in results:
        if not result.ok:
            continue
        for raw_job in result.jobs:
            hits = title_matches(raw_job.title, match_terms)
            if match_terms and not hits:
                continue
            raw_job.matched_keywords = hits
            if city and not city_matches(f"{raw_job.city} {raw_job.title}", city):
                continue

            raw_job.city = normalize_city(raw_job.city, city)
            raw_job.url = normalize_url(raw_job.url) or raw_job.url
            salary = parse_salary(raw_job.salary_text)
            if salary is not None:
                raw_job.salary_low_k = salary.low_k
                raw_job.salary_high_k = salary.high_k
                raw_job.salary_mid_k = salary.mid_k
                raw_job.salary_months = salary.months

            key = build_job_key(raw_job.source, raw_job.company, raw_job.title, raw_job.url)
            if key in by_key:
                report.deduped += 1
                continue

            identity = _identity(raw_job)
            existing = by_identity.get(identity)
            if existing is not None:
                report.deduped += 1
                # 已有条目信息更少时，用新的替换（但不要重复计数）
                if _richness(raw_job) > _richness(existing):
                    by_key.pop(
                        build_job_key(
                            existing.source, existing.company, existing.title, existing.url
                        ),
                        None,
                    )
                    by_identity[identity] = raw_job
                    by_key[key] = raw_job
                continue

            title_key = normalize_title(raw_job.title)
            if title_counts.get(title_key, 0) >= max_per_title:
                report.deduped += 1
                continue
            title_counts[title_key] = title_counts.get(title_key, 0) + 1
            by_key[key] = raw_job
            by_identity[identity] = raw_job

    merged = sorted(
        by_key.values(), key=lambda job: (-len(job.matched_keywords), job.title)
    )[: int(settings.crawler_max_jobs)]
    report.jobs = merged

    # ---- 薪资统计 ----
    ranges = [
        SalaryRange(
            low_k=job.salary_low_k, high_k=job.salary_high_k,
            months=job.salary_months, raw=job.salary_text,
        )
        for job in merged
        if job.salary_low_k is not None and job.salary_high_k is not None
    ]
    report.salary = summarize_salaries(
        ranges, profile.expect_salary_min, profile.expect_salary_max
    )
    if not ranges:
        report.warnings.append(
            "抓到的岗位都没有公开薪资（校招官网通常不挂薪资），"
            "因此无法算出市场平均薪资。可以启用带薪资的社招来源，或在档案里补充期望薪资作为参考。"
        )
    elif report.salary.get("low_confidence"):
        report.warnings.append(
            f"薪资样本只有 {report.salary['sample_size']} 条，统计结果参考价值有限。"
        )

    # ---- 落库 ----
    if save and merged:
        try:
            async with session_scope() as db:
                # 同一「方向+城市」的旧结果清掉，避免新旧混算
                await repository.clear_crawled_jobs(db, query_role=role, query_city=city)
                records = []
                for job in merged:
                    record = job.to_dict()
                    record["query_role"] = role
                    record["query_city"] = city
                    record["source_url"] = job.url
                    records.append(record)
                report.saved = await repository.upsert_crawled_jobs(db, records)
        except Exception as exc:  # noqa: BLE001
            logger.exception("岗位入库失败")
            report.warnings.append(f"岗位入库失败：{exc}")

    report.elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "抓取完成：方向=%s 城市=%s 来源=%d 岗位=%d 有薪资=%d 耗时=%.1fs",
        role, city, len(results), len(merged), len(ranges), report.elapsed_ms / 1000,
    )
    return report


async def test_source(source_key: str) -> dict[str, Any]:
    """单独试抓一个来源，用于前端排查配置是否正确。"""
    sources = load_sources()
    source = next((item for item in sources if str(item.get("key")) == source_key), None)
    if source is None:
        raise ValueError(f"没有这个来源：{source_key}")

    keywords = split_keywords(str((source.get("config") or {}).get("_test_keywords") or ""))
    match_terms, _ = expand_keywords(keywords) if keywords else ([], {})
    result = await asyncio.to_thread(
        _run_source, source, keywords or [], "", PER_SOURCE_BUDGET
    )
    payload = result.to_dict()
    payload["sample"] = [
        {"title": job.title, "city": job.city, "salary": job.salary_text, "url": job.url}
        for job in result.jobs[:5]
    ]
    return payload
