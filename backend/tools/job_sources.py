"""岗位数据源适配器：从多个**真实**招聘数据源抓取岗位，并按职位名称匹配关键字。

为什么要多源？
    * 免费公开的招聘接口没有中文源（拉勾/BOSS 等无公开 API），
      所以内置的是 4 个真实但以英文远程岗位为主的数据源；
    * 本地 MySQL 里的种子数据只作为「离线兜底 / 演示」，绝不是主要数据源；
    * 用户自己的数据源通过 ``JOB_API_BASE`` 接入（优先级最高）。

匹配规则（按用户要求）：
    **关键词只匹配「职位名称」**，不做全字段模糊匹配。
    多个关键字之间是「或」的关系：职位名包含任意一个关键字即命中。

关于缓存：
    所有适配器都**只抓取、不过滤**，过滤统一在 ``fetch_jobs`` 里按关键字做。
    这样 TTL 缓存可以只按数据源名做 key——否则「换了关键字却拿到上一次过滤结果」
    这种 bug 几乎必然出现（早期版本就踩过）。
    代价是首次抓取会拉全量列表（部分源单次响应可达 2MB），所以 TTL 缓存是必需的。

中文关键字处理：
    英文数据源的职位名不会出现「python后端」这种写法，
    因此内置一张很小的中英同义词表做扩展（``python后端`` → ``python backend``、
    ``backend engineer`` 等）。这是有损的启发式手段，返回结果里会标注
    ``keyword_expansion``，让调用方知道实际用了哪些英文词。
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

from config import get_settings

logger = logging.getLogger("ai_resume_helper.tools.jobs.sources")

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str | None, limit: int = 0) -> str:
    """把 HTML 描述转成纯文本。"""
    if not text:
        return ""
    plain = html.unescape(_HTML_TAG_RE.sub(" ", text))
    plain = _WS_RE.sub(" ", plain).strip()
    return plain[:limit] if limit else plain


# ---------------------------------------------------------------------- #
# 中文关键字 -> 英文同义词（用于英文数据源的职位名匹配）
# ---------------------------------------------------------------------- #
KEYWORD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "python后端": ("python backend", "backend engineer", "backend developer", "python engineer"),
    "后端": ("backend", "back-end", "back end"),
    "前端": ("frontend", "front-end", "front end", "react", "vue"),
    "全栈": ("full stack", "fullstack"),
    "算法": ("algorithm", "machine learning", "ml engineer", "data scientist"),
    "数据分析": ("data analyst", "data analytics", "business intelligence"),
    "数据": ("data",),
    "大模型": ("llm", "large language model", "generative ai", "genai"),
    "ai": ("ai", "artificial intelligence", "machine learning"),
    "人工智能": ("ai", "artificial intelligence", "machine learning"),
    "agent": ("agent", "agentic", "llm"),
    "智能体": ("agent", "agentic"),
    "测试": ("qa", "test", "sdet", "quality assurance"),
    "运维": ("devops", "sre", "site reliability", "platform engineer"),
    "产品经理": ("product manager", "product owner"),
    "运营": ("operations", "growth", "community"),
    "市场": ("marketing", "growth"),
    "销售": ("sales", "account executive"),
    "设计": ("designer", "design", "ux", "ui"),
    "安全": ("security", "appsec", "infosec"),
    "嵌入式": ("embedded", "firmware"),
    "移动端": ("mobile", "android", "ios"),
    "java": ("java",),
    "go": ("golang", "go engineer"),
    "golang": ("golang",),
    "c++": ("c++", "cpp"),
    "rust": ("rust",),
    "react": ("react",),
    "vue": ("vue",),
    "mysql": ("mysql", "database", "dba"),
    "kubernetes": ("kubernetes", "k8s"),
    "运维开发": ("devops", "sre", "platform"),
}


def split_keywords(raw: str | Sequence[str] | None) -> list[str]:
    """把用户输入拆成关键字列表，支持逗号/顿号/空格/竖线分隔。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[,，、|/\s]+", raw)
    else:
        parts = []
        for item in raw:
            parts.extend(re.split(r"[,，、|/\s]+", str(item)))
    seen: list[str] = []
    for part in parts:
        token = part.strip().lower()
        if token and token not in seen:
            seen.append(token)
    return seen


def expand_keywords(keywords: Sequence[str]) -> tuple[list[str], dict[str, list[str]]]:
    """为英文数据源扩展同义词，返回 (全部匹配词, 每个原关键字的扩展结果)。"""
    expanded: dict[str, list[str]] = {}
    pool: list[str] = []
    for keyword in keywords:
        variants = [keyword]
        for cn, en_list in KEYWORD_SYNONYMS.items():
            if cn in keyword:
                variants.extend(en_list)
        # 去掉「python后端」这类中文尾缀，保留纯技术名词（python 本身就能匹配）
        ascii_only = " ".join(re.findall(r"[a-zA-Z0-9+#.]+", keyword))
        if ascii_only:
            variants.append(ascii_only)
        deduped: list[str] = []
        for variant in variants:
            variant = variant.strip().lower()
            if variant and variant not in deduped:
                deduped.append(variant)
        expanded[keyword] = deduped
        for variant in deduped:
            if variant not in pool:
                pool.append(variant)
    return pool, expanded


def title_matches(title: str, match_terms: Sequence[str]) -> list[str]:
    """返回职位名命中的关键字（空列表表示不匹配）。"""
    lowered = (title or "").lower()
    return [term for term in match_terms if term in lowered]


# ---------------------------------------------------------------------- #
# 统一的岗位结构
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class JobItem:
    id: str
    source: str
    title: str
    company: str = ""
    location: str = ""
    salary: str = ""
    tags: list[str] = field(default_factory=list)
    url: str = ""
    posted_at: str = ""
    description: str = ""
    job_type: str = ""
    matched_keywords: list[str] = field(default_factory=list)

    def to_dict(self, description_chars: int = 400) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "salary": self.salary or "未提供",
            "tags": self.tags[:15],
            "url": self.url,
            "posted_at": self.posted_at,
            "job_type": self.job_type,
            "matched_keywords": self.matched_keywords,
            "description": self.description[:description_chars],
        }


@dataclass(slots=True)
class SourceOutcome:
    """单个数据源的抓取结果，用于如实上报成功/失败原因。"""

    source: str
    ok: bool
    fetched: int = 0
    matched: int = 0
    error: str = ""
    elapsed_ms: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ok": self.ok,
            "fetched": self.fetched,
            "matched": self.matched,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "note": self.note,
        }


# ---------------------------------------------------------------------- #
# 数据源实现
# ---------------------------------------------------------------------- #
async def _get_json(client: Any, url: str, params: dict[str, Any] | None = None) -> Any:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def fetch_remotive(client: Any, limit: int) -> tuple[list[JobItem], int]:
    """Remotive：一次返回全量岗位列表（不做服务端关键字过滤，过滤交给上层）。"""
    payload = await _get_json(client, "https://remotive.com/api/remote-jobs")
    raw = (payload or {}).get("jobs") or []
    items: list[JobItem] = []
    for job in raw:
        title = str(job.get("title") or "")
        if not title:
            continue
        items.append(
            JobItem(
                id=f"remotive-{job.get('id')}",
                source="Remotive",
                title=title,
                company=str(job.get("company_name") or ""),
                location=str(job.get("candidate_required_location") or "Remote"),
                salary=str(job.get("salary") or ""),
                tags=[str(t) for t in (job.get("tags") or [])],
                url=str(job.get("url") or ""),
                posted_at=str(job.get("publication_date") or "")[:10],
                description=strip_html(job.get("description"), 1200),
                job_type=str(job.get("job_type") or ""),
            )
        )
    return items, len(raw)


async def fetch_arbeitnow(client: Any, limit: int) -> tuple[list[JobItem], int]:
    payload = await _get_json(client, "https://www.arbeitnow.com/api/job-board-api")
    raw = (payload or {}).get("data") or []
    items: list[JobItem] = []
    for job in raw:
        title = str(job.get("title") or "")
        if not title:
            continue
        items.append(
            JobItem(
                id=f"arbeitnow-{job.get('slug')}",
                source="Arbeitnow",
                title=title,
                company=str(job.get("company_name") or ""),
                location=str(job.get("location") or "") + ("（远程）" if job.get("remote") else ""),
                tags=[str(t) for t in (job.get("tags") or [])],
                url=str(job.get("url") or ""),
                posted_at=str(job.get("created_at") or "")[:10],
                description=strip_html(job.get("description"), 1200),
                job_type="、".join(str(t) for t in (job.get("job_types") or [])),
            )
        )
    return items, len(raw)


async def fetch_jobicy(client: Any, limit: int) -> tuple[list[JobItem], int]:
    # Jobicy 无关键字参数时返回最新一批（上限 100）
    payload = await _get_json(client, "https://jobicy.com/api/v2/remote-jobs", {"count": 100})
    raw = (payload or {}).get("jobs") or []
    items: list[JobItem] = []
    for job in raw:
        title = str(job.get("jobTitle") or "")
        if not title:
            continue
        salary = ""
        low, high = job.get("annualSalaryMin"), job.get("annualSalaryMax")
        if low or high:
            currency = job.get("salaryCurrency") or "USD"
            salary = f"{low or '?'}-{high or '?'} {currency}/年"
        items.append(
            JobItem(
                id=f"jobicy-{job.get('id')}",
                source="Jobicy",
                title=title,
                company=str(job.get("companyName") or ""),
                location=str(job.get("jobGeo") or "Remote"),
                salary=salary,
                tags=[str(t) for t in (job.get("jobIndustry") or [])],
                url=str(job.get("url") or ""),
                posted_at=str(job.get("pubDate") or "")[:10],
                description=strip_html(job.get("jobExcerpt") or job.get("jobDescription"), 1200),
                job_type="、".join(str(t) for t in (job.get("jobType") or [])),
            )
        )
    return items, len(raw)


async def fetch_remoteok(client: Any, limit: int) -> tuple[list[JobItem], int]:
    payload = await _get_json(client, "https://remoteok.com/api")
    raw = [item for item in (payload or []) if isinstance(item, dict) and item.get("position")]
    items: list[JobItem] = []
    for job in raw:
        title = str(job.get("position") or "")
        if not title:
            continue
        salary = ""
        low, high = job.get("salary_min"), job.get("salary_max")
        if low or high:
            salary = f"${low or '?'}-{high or '?'}/年"
        items.append(
            JobItem(
                id=f"remoteok-{job.get('id')}",
                source="RemoteOK",
                title=title,
                company=str(job.get("company") or ""),
                location=str(job.get("location") or "Remote"),
                salary=salary,
                tags=[str(t) for t in (job.get("tags") or [])],
                url=str(job.get("url") or job.get("apply_url") or ""),
                posted_at=str(job.get("date") or "")[:10],
                description=strip_html(job.get("description"), 1200),
            )
        )
    return items, len(raw)


async def fetch_custom(client: Any, limit: int) -> tuple[list[JobItem], int]:
    """用户自建/自购的岗位 API（``JOB_API_BASE``）。

    约定：``GET {JOB_API_BASE}/jobs?limit=...`` 返回
    ``{"jobs": [{"id","title","company","city","salary","url","tags","posted_at","description"}]}``
    """
    settings = get_settings()
    headers = {"Accept": "application/json"}
    if settings.job_api_key:
        headers["Authorization"] = f"Bearer {settings.job_api_key}"

    response = await client.get(
        settings.job_api_base.rstrip("/") + "/jobs", params={"limit": 500}, headers=headers
    )
    response.raise_for_status()
    payload = response.json()

    raw = payload.get("jobs") if isinstance(payload, dict) else payload
    raw = raw or []
    items: list[JobItem] = []
    for job in raw:
        title = str(job.get("title") or "")
        if not title:
            continue
        items.append(
            JobItem(
                id=f"custom-{job.get('id', len(items))}",
                source="自建数据源",
                title=title,
                company=str(job.get("company") or ""),
                location=str(job.get("city") or job.get("location") or ""),
                salary=str(job.get("salary") or ""),
                tags=[str(t) for t in (job.get("tags") or job.get("skills") or [])],
                url=str(job.get("url") or ""),
                posted_at=str(job.get("posted_at") or ""),
                description=str(job.get("description") or ""),
                job_type=str(job.get("employment_type") or ""),
            )
        )
    return items, len(raw)


async def fetch_local(client: Any, limit: int) -> tuple[list[JobItem], int]:
    """本地 MySQL 种子数据（离线兜底，仅作演示，不是实时岗位）。"""
    from ..db import repository, session_scope

    async with session_scope() as db:
        rows = await repository.search_job_postings(db, limit=500)
        total = len(rows)
        items: list[JobItem] = []
        for row in rows:
            data = repository.job_to_dict(row)
            items.append(
                JobItem(
                    id=data["id"],
                    source="本地示例库",
                    title=data["title"],
                    company=data["company"],
                    location=data["city"],
                    salary=data["salary"],
                    tags=data["skills"],
                    url=data["url"],
                    posted_at=data["posted_at"],
                    description=data["description"] or "、".join(data["responsibilities"][:3]),
                    job_type=data["employment_type"],
                )
            )
    return items, total


Adapter = Callable[[Any, int], Awaitable[tuple[list[JobItem], int]]]

ADAPTERS: dict[str, Adapter] = {
    "remotive": fetch_remotive,
    "arbeitnow": fetch_arbeitnow,
    "jobicy": fetch_jobicy,
    "remoteok": fetch_remoteok,
    "custom": fetch_custom,
    "local": fetch_local,
}

# 默认数据源顺序：真实公开源优先，本地兜底放最后
DEFAULT_SOURCES: tuple[str, ...] = ("jobicy", "remotive", "remoteok", "arbeitnow", "local")

SOURCE_LABELS = {
    "jobicy": "Jobicy（公开 API，真实岗位）",
    "remotive": "Remotive（公开 API，真实岗位）",
    "remoteok": "RemoteOK（公开 API，真实岗位）",
    "arbeitnow": "Arbeitnow（公开 API，真实岗位）",
    "custom": "自建数据源（JOB_API_BASE）",
    "local": "本地示例库（非实时，仅供离线兜底）",
}


def available_sources() -> list[str]:
    """按当前配置返回可用的数据源。"""
    settings = get_settings()
    sources = ["jobicy", "remotive", "remoteok", "arbeitnow"]
    if settings.job_api_base:
        sources.insert(0, "custom")
    sources.append("local")
    return sources


def resolve_sources(requested: Sequence[str] | None) -> list[str]:
    available = available_sources()
    if not requested or "auto" in requested:
        return available
    picked = [name for name in requested if name in available]
    return picked or available


# ---------------------------------------------------------------------- #
# 抓取入口（TTL 缓存 + 并发 + 单源失败不影响整体）
# ---------------------------------------------------------------------- #
_cache: dict[str, tuple[float, list[JobItem], int]] = {}
_cache_lock = asyncio.Lock()


async def _fetch_source_raw(source: str, timeout: float) -> tuple[list[JobItem], int, str]:
    """抓取单个数据源的**全量**岗位（带 TTL 缓存）。

    缓存只按数据源名做 key，因为这里返回的是不过滤的原始结果；
    关键字过滤在 ``fetch_jobs`` 里做，两者解耦，避免缓存串味。
    """
    import httpx

    adapter = ADAPTERS.get(source)
    if adapter is None:
        return [], 0, f"未知数据源：{source}"

    async with _cache_lock:
        cached = _cache.get(source)
        if cached and time.time() - cached[0] < get_settings().job_cache_ttl:
            return cached[1], cached[2], ""

    headers = {"User-Agent": "ai-resume-helper/2.0 (+local demo)", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        items, fetched = await adapter(client, 0)

    async with _cache_lock:
        _cache[source] = (time.time(), items, fetched)
    return items, fetched, ""


async def fetch_jobs(
    keywords: Sequence[str],
    sources: Sequence[str] | None = None,
    limit_per_source: int = 100,
    total_limit: int = 20,
) -> tuple[list[JobItem], list[SourceOutcome], list[str], dict[str, list[str]]]:
    """并发抓取多个数据源，按职位名称匹配关键字。"""
    settings = get_settings()
    picked = resolve_sources(sources)
    match_terms, expansion = expand_keywords(keywords)

    async def run(source: str) -> tuple[str, list[JobItem], int, str, float]:
        started = time.perf_counter()
        try:
            raw_items, fetched, error = await _fetch_source_raw(source, settings.tool_timeout)
            # 关键字过滤在这里做：只匹配职位名称
            matched: list[JobItem] = []
            for item in raw_items:
                hits = title_matches(item.title, match_terms)
                if not hits:
                    continue
                item.matched_keywords = hits
                matched.append(item)
                if len(matched) >= limit_per_source:
                    break
            return source, matched, fetched, error, (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - 单个数据源失败不能影响整体
            logger.warning("岗位数据源 %s 抓取失败：%s", source, exc)
            return source, [], 0, f"{type(exc).__name__}: {exc}", (time.perf_counter() - started) * 1000

    results = await asyncio.gather(*(run(source) for source in picked))

    outcomes: list[SourceOutcome] = []
    merged: list[JobItem] = []
    seen_titles: set[str] = set()

    for source, items, fetched, error, elapsed in results:
        outcome = SourceOutcome(
            source=source,
            ok=not error,
            fetched=fetched,
            matched=len(items),
            error=error,
            elapsed_ms=elapsed,
        )
        if error:
            outcome.note = "该数据源本次不可用，已跳过"
        elif not items:
            outcome.note = "抓取成功，但没有职位名命中关键字"
        outcomes.append(outcome)

        for item in items:
            # 简单去重：公司 + 职位名
            dedupe_key = f"{item.company}|{item.title}".lower()
            if dedupe_key in seen_titles:
                continue
            seen_titles.add(dedupe_key)
            merged.append(item)

    # 命中关键字多的排前面，其次按来源优先级
    source_order = {name: i for i, name in enumerate(picked)}
    merged.sort(
        key=lambda job: (-len(job.matched_keywords), source_order.get(job.source.lower(), 99))
    )
    return merged[:total_limit], outcomes, match_terms, expansion
