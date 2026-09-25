"""爬虫基础设施：统一岗位结构、去重身份、请求重试、浏览器启动。

设计参考了成熟招聘爬虫工程的几条经验：

1. **统一输出契约**：所有爬虫都产出同一种 ``JobItem``，
   上层（薪资统计、知识库、去重入库）只认这个结构，不关心来源差异。
2. **岗位身份去重**：招聘站的同一个岗位，每次抓到的 URL 往往带不同的
   跟踪参数（``utm_*`` / ``spm`` / ``recommendCode`` …），位置参数也可能变化。
   必须先归一化标题与 URL 再做哈希，否则数据库里会堆满重复岗位。
3. **两级抓取**：先抓列表拿到标题，**按职位名匹配关键字**筛出候选，
   再只对候选补抓详情页 JD。既省流量，也正好符合「关键字匹配职位名称」的需求。
4. **浏览器用系统自带的**：Playwright 驱动本机 Edge/Chrome（``channel``），
   不打包 Chromium，exe 体积能小 150MB。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger("ai_resume_helper.crawlers")

# 常见的 URL 跟踪参数，参与身份哈希前必须剔除
_TRACKING_KEYS = {
    "from", "ref", "referrer", "recommendcode", "recommend_code", "shareid",
    "share_id", "source", "spm", "trackid", "track_id", "channel", "ch",
    "sessionid", "session_id", "token", "utm_source", "utm_medium",
    "utm_campaign", "utm_term", "utm_content", "clickid", "gclid", "fbclid",
}

_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
)


def random_user_agent() -> str:
    return random.choice(_USER_AGENTS)


# ---------------------------------------------------------------------- #
# 归一化与去重身份
# ---------------------------------------------------------------------- #
def normalize_title(value: object) -> str:
    """职位名归一化：全角转半角、大小写折叠、去掉所有符号与空白。"""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w\u3400-\u9fff]+", "", text)


def normalize_url(value: object) -> str:
    """URL 归一化：去 www、去跟踪参数、query 排序、保留 SPA 的 hash 路由。"""
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return ""
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return ""

    host = parsed.hostname.casefold()
    if host.startswith("www."):
        host = host[4:]
    try:
        port = parsed.port
    except ValueError:
        return ""
    if port and not (
        (parsed.scheme.casefold() == "https" and port == 443)
        or (parsed.scheme.casefold() == "http" and port == 80)
    ):
        host = f"{host}:{port}"

    def _clean_query(raw: str) -> str:
        pairs = [
            (key.casefold(), item)
            for key, item in parse_qsl(raw, keep_blank_values=True)
            if key.casefold() not in _TRACKING_KEYS
        ]
        return urlencode(sorted(pairs))

    path = (parsed.path or "/").rstrip("/") or "/"
    fragment = parsed.fragment
    # 很多 SPA 把真正的岗位 ID 放在 hash 里，例如 #/details?id=123
    if "?" in fragment:
        route, frag_query = fragment.split("?", 1)
        cleaned = _clean_query(frag_query)
        fragment = route + (f"?{cleaned}" if cleaned else "")

    return urlunsplit(
        (parsed.scheme.casefold(), host, path, _clean_query(parsed.query), fragment)
    )


def build_job_key(source: str, company: str, title: str, url: str) -> str:
    """生成稳定的岗位身份哈希。

    优先用「来源 + 公司 + 归一化标题 + 归一化URL」；
    URL 缺失时退化为「来源 + 公司 + 标题」，保证同一岗位不会重复入库。
    """
    normalized_url = normalize_url(url)
    parts = ["v1", (source or "").strip().casefold(), (company or "").strip().casefold()]
    parts.append(normalize_title(title))
    if normalized_url:
        parts.append(normalized_url)
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]


# ---------------------------------------------------------------------- #
# 统一岗位结构
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class JobItem:
    """所有爬虫的统一输出。"""

    source: str
    title: str
    company: str = ""
    city: str = ""
    salary_text: str = ""
    job_type: str = ""
    url: str = ""
    jd_text: str = ""
    published_at: str = ""
    tags: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    # 薪资解析结果由上层（salary.py）填，爬虫不负责
    salary_low_k: float | None = None
    salary_high_k: float | None = None
    salary_mid_k: float | None = None
    salary_months: int = 12

    @property
    def job_key(self) -> str:
        return build_job_key(self.source, self.company, self.title, self.url)

    @property
    def has_detail(self) -> bool:
        """是否已经抓到 JD 正文（用于判断要不要补抓详情）。"""
        return len(self.jd_text.strip()) >= 40

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_key": self.job_key,
            "source": self.source,
            "title": self.title,
            "company": self.company,
            "city": self.city,
            "salary_text": self.salary_text,
            "job_type": self.job_type,
            "url": self.url,
            "jd_text": self.jd_text,
            "published_at": self.published_at,
            "tags": self.tags,
            "matched_keywords": self.matched_keywords,
            "salary_low_k": self.salary_low_k,
            "salary_high_k": self.salary_high_k,
            "salary_mid_k": self.salary_mid_k,
            "salary_months": self.salary_months,
        }


@dataclass(slots=True)
class SourceResult:
    """单个来源的抓取结果（成功/失败都要如实上报）。"""

    source: str
    label: str
    ok: bool
    jobs: list[JobItem] = field(default_factory=list)
    fetched: int = 0
    error: str = ""
    elapsed_ms: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "ok": self.ok,
            "jobs": len(self.jobs),
            "fetched": self.fetched,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "note": self.note,
        }


# ---------------------------------------------------------------------- #
# 抓取预算（避免某个慢站点拖死整个流程）
# ---------------------------------------------------------------------- #
class CrawlBudgetExceeded(RuntimeError):
    """超出本次抓取的时间预算。"""


@dataclass
class CrawlBudget:
    """单次抓取的时间预算（同时也能当上下文管理器用）。"""

    seconds: float
    started: float = field(default_factory=time.monotonic)

    def remaining(self) -> float:
        return self.seconds - (time.monotonic() - self.started)

    def check(self) -> None:
        if self.remaining() <= 0:
            raise CrawlBudgetExceeded(f"抓取超出预算 {self.seconds:g}s")

    def __enter__(self) -> "CrawlBudget":
        self.started = time.monotonic()
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def crawl_budget(seconds: float) -> CrawlBudget:
    """便捷构造（保留函数形式，语义更直观）。"""
    return CrawlBudget(seconds=max(1.0, float(seconds)))


# ---------------------------------------------------------------------- #
# 基类
# ---------------------------------------------------------------------- #
def _decode_response(response: Any) -> Any:
    """按 UTF-8 解析 JSON；失败时回退 GBK（国内不少站点仍返回 GBK）。"""
    raw = response.content
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    # 最后再让 httpx 自己猜一次
    try:
        return response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"响应不是合法 JSON（前 200 字节：{raw[:200]!r}）") from exc


class BaseCrawler:
    """所有爬虫的基类。

    子类只需要实现 ``fetch(budget) -> list[JobItem]``。
    """

    kind = "base"
    #: 请求失败重试次数
    REQUEST_ATTEMPTS = 3

    def __init__(self, source: dict[str, Any]) -> None:
        self.source = source
        self.key = str(source.get("key") or "")
        self.label = str(source.get("label") or self.key)
        self.careers_url = str(source.get("careers_url") or "")
        self.config: dict[str, Any] = dict(source.get("config") or {})

    # ---------------- 网络请求 ----------------
    def http_get(
        self, url: str, *, params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None, timeout: float = 20.0,
    ) -> Any:
        import httpx

        merged = {
            "User-Agent": random_user_agent(),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        }
        merged.update(headers or {})
        last_error = ""
        for attempt in range(1, self.REQUEST_ATTEMPTS + 1):
            try:
                response = httpx.get(
                    url, params=params, headers=merged, timeout=timeout, follow_redirects=True
                )
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.REQUEST_ATTEMPTS:
                    time.sleep(0.6 * attempt)
        raise RuntimeError(f"请求失败（已重试 {self.REQUEST_ATTEMPTS} 次）：{last_error}")

    def http_post_json(
        self, url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None,
        timeout: float = 20.0,
    ) -> Any:
        import httpx

        merged = {
            "User-Agent": random_user_agent(),
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Content-Type": "application/json",
            "Referer": self.careers_url or url,
        }
        merged.update(headers or {})
        last_error = ""
        for attempt in range(1, self.REQUEST_ATTEMPTS + 1):
            try:
                response = httpx.post(
                    url, json=payload, headers=merged, timeout=timeout, follow_redirects=True
                )
                response.raise_for_status()
                return _decode_response(response)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.REQUEST_ATTEMPTS:
                    time.sleep(0.6 * attempt)
        raise RuntimeError(f"请求失败（已重试 {self.REQUEST_ATTEMPTS} 次）：{last_error}")

    # ---------------- 浏览器 ----------------
    @staticmethod
    def launch_browser(playwright: Any, *, channel: str, headless: bool) -> Any:
        """启动**系统自带**的浏览器，不依赖打包的 Chromium。

        依次尝试 Edge → Chrome → 打包的 Chromium，任何一个成功即可。
        """
        last_error: Exception | None = None
        for candidate in (channel, "msedge", "chrome"):
            if not candidate:
                continue
            try:
                return playwright.chromium.launch(channel=candidate, headless=headless)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        try:
            # 实在没有系统浏览器，才退回 Playwright 自带内核（需 playwright install）
            return playwright.chromium.launch(headless=headless)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "无法启动浏览器：请确认已安装 Microsoft Edge 或 Chrome。"
                f"（最后一次错误：{type(last_error).__name__ if last_error else 'N/A'} / {exc}）"
            ) from exc

    # ---------------- 子类实现 ----------------
    def fetch(self, budget: CrawlBudget) -> list[JobItem]:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def make_job(self, **kwargs: Any) -> JobItem:
        kwargs.setdefault("source", self.label)
        kwargs.setdefault("company", self.config.get("company") or self.label)
        return JobItem(**kwargs)


# ---------------------------------------------------------------------- #
# 来源注册表（用户可编辑的 sources.json）
# ---------------------------------------------------------------------- #
DEFAULT_SOURCES: list[dict[str, Any]] = [
    {
        "key": "nowcoder_campus",
        "label": "牛客网 · 校招（含薪资）",
        "kind": "public_api",
        "enabled": True,
        "careers_url": "https://www.nowcoder.com/jobs/fulltime/center",
        "config": {"adapter": "nowcoder", "recruit_type": 1, "max_pages": 4},
    },
    {
        "key": "nowcoder_social",
        "label": "牛客网 · 社招（含薪资）",
        "kind": "public_api",
        "enabled": True,
        "careers_url": "https://www.nowcoder.com/jobs/fulltime/center",
        "config": {"adapter": "nowcoder", "recruit_type": 3, "max_pages": 4},
    },
    {
        "key": "jd_campus",
        "label": "京东校招（无薪资字段）",
        "kind": "public_api",
        "enabled": True,
        "careers_url": "https://campus.jd.com/",
        "config": {"adapter": "jd", "company": "京东", "max_pages": 6},
    },
    # ------------------------------------------------------------------ #
    # 51job：岗位量最大的来源，也是**主力薪资来源**
    #
    # 阿里云 WAF 挡住了直接请求，所以走 render_api（真浏览器 + 拦截 XHR）。
    # 它的接口自带数字薪资字段 jobSalaryMin / jobSalaryMax（元/月），
    # 比解析「6-8.5千」这种文案可靠得多。
    #
    # city_codes 里每一个代码都是**实测验证过**的：用该代码搜索后检查返回
    # 岗位的 jobAreaString 是否真的落在目标城市。凭记忆猜的 8 个代码
    # （南通/南昌/哈尔滨/长春/南宁/石家庄/太原）实测都会返回**别的城市**，
    # 已全部剔除——猜错会静默给出错误城市的岗位，最难发现。
    # 表中没有的城市会自动退化成「全国检索 + 本地按城市过滤」，结果依然正确。
    # ------------------------------------------------------------------ #
    {
        "key": "51job",
        "label": "前程无忧 51job（社招，带薪资）",
        "kind": "render_api",
        "enabled": True,
        "careers_url": "https://we.51job.com/pc/search",
        "config": {
            # 注意：这里**不拼 pageNum**。实测把 pageNum=2 放进搜索页 URL 不会真的
            # 翻页（仍返回第 1 页），翻页必须靠点「下一页」按钮，见 next_selector。
            "page_url": (
                "https://we.51job.com/pc/search?keyword={keyword}&jobArea={city_code}"
            ),
            "intercept": ["/api/job/search-pc"],
            "items_path": "resultbody.job.items",
            "total_path": "resultbody.job.totalCount",
            # 点它翻页，实测第 2 页会返回 20 条**不同**的岗位
            "next_selector": ".btn-next",
            "pages_per_term": 2,
            "max_terms": 2,
            "max_jobs": 200,
            # 详情链接带每次检索都不同的 req= 校验哈希，不剥掉会产生重复岗位
            "url_strip_query": True,
            "city_codes": {
                "北京": "010000", "上海": "020000", "广州": "030200",
                "深圳": "040000", "杭州": "080200", "成都": "090200",
                "武汉": "180200", "南京": "070200", "西安": "200200",
                "天津": "050000", "重庆": "060000", "苏州": "070300",
                "无锡": "070400", "常州": "070500", "宁波": "080300",
                "温州": "080400", "佛山": "030600", "东莞": "030800",
                "珠海": "030500", "厦门": "110300", "福州": "110200",
                "济南": "120200", "青岛": "120300", "烟台": "120400",
                "合肥": "150200", "郑州": "170200", "长沙": "190200",
                "沈阳": "230200", "大连": "230300", "昆明": "250200",
                "贵阳": "260200",
            },
            "field_map": {
                "title": "jobName",
                # fullCompanyName 是全称，比简称更适合做去重身份
                "company": "fullCompanyName",
                "city": "jobAreaString",
                "url": "jobHref",
                "jd_text": "jobDescribe",
                "salary_text": "provideSalaryString",
                "salary_min_yuan": "jobSalaryMin",
                "salary_max_yuan": "jobSalaryMax",
                "published_at": "issueDateString",
                "tags": "jobTags",
                "job_type": "termStr",
                "degree": "degreeString",
                "experience": "workYearString",
                "company_size": "companySizeString",
                "company_type": "companyTypeString",
            },
        },
    },
    # ------------------------------------------------------------------ #
    # 公司官网：权威一手 JD。用 json_api 适配器，换公司只改配置。
    #
    # 注意：公司官网基本都**不公开薪资**，所以它们能丰富岗位和 JD，
    # 但薪资样本仍然主要来自牛客和 51job。
    # ------------------------------------------------------------------ #
    {
        "key": "tencent",
        "label": "腾讯招聘官网（无薪资）",
        "kind": "public_api",
        "enabled": True,
        "careers_url": "https://careers.tencent.com/",
        "config": {
            "adapter": "json_api",
            "method": "GET",
            "endpoint": "https://careers.tencent.com/tencentcareer/api/post/Query",
            "params": {
                "keyword": "{keyword}",
                "pageIndex": "{page}",
                "pageSize": "50",
                "language": "zh-cn",
                "area": "cn",
                "timestamp": "{timestamp}",
            },
            "items_path": "Data.Posts",
            "pages_per_term": 3,
            "page_size": 50,
            "max_terms": 3,
            "field_map": {
                "title": "RecruitPostName",
                "city": "LocationName",
                "company": "ComName",
                "jd_text": "Responsibility",
                "url": "PostURL",
                "published_at": "LastUpdateTime",
                "tags": "BGName",
                "experience": "RequireWorkYearsName",
            },
        },
    },
    {
        "key": "netease",
        "label": "网易招聘官网（无薪资）",
        "kind": "public_api",
        "enabled": True,
        "careers_url": "https://hr.163.com/job-list.html",
        "config": {
            "adapter": "json_api",
            "method": "POST",
            "endpoint": "https://hr.163.com/api/hr163/position/queryPage",
            "body": {"currentPage": "{page}", "pageSize": 50, "keyword": "{keyword}"},
            "items_path": "data.list",
            "pages_per_term": 3,
            "page_size": 50,
            "max_terms": 3,
            # 列表接口不给详情链接（beeUrl 恒为 null），用模板拼出来。
            # 该路由已实测：打开后正文里确实出现对应职位名。
            "company": "网易",
            "field_map": {
                "title": "name",
                "city": "workPlaceNameList",
                "url": "https://hr.163.com/job-detail.html?id={id}",
                "jd_text": ["requirement", "description"],
                "published_at_ms": "updateTime",
                "job_type": "firstPostTypeName",
                "degree": "reqEducationName",
                "experience": "reqWorkYearsName",
            },
        },
    },
    {
        "key": "custom_1",
        "label": "自定义公司官网（改成你自己的目标公司）",
        "kind": "generic_render",
        "enabled": False,
        "careers_url": "",
        "config": {
            "list_url": "",
            "item_selector": "",
            "title_selector": "",
            "city_selector": "",
            "salary_selector": "",
            "link_selector": "a",
            "max_pages": 3,
            "fetch_detail": True,
            "detail_limit": 10,
        },
    },
]


def sources_file_path() -> Path:
    from paths import DATA_HOME

    return DATA_HOME / "sources.json"


def _merge_missing_defaults(user_sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """把用户文件里**缺少的内置来源**补回去，返回 (合并结果, 新增的 key)。

    为什么要合并：``sources.json`` 是「首次运行时生成」的，所以老用户升级后
    文件里没有新加的内置来源（比如 51job），结果就是「代码里加了新来源但用户
    永远用不到」。这里只补**键完全不存在**的条目，用户已有的（哪怕被禁用、
    被改过配置）一律保留不动，不会覆盖用户的修改。
    """
    known = {str(item.get("key")) for item in user_sources if isinstance(item, dict)}
    added: list[str] = []
    merged = list(user_sources)
    for default in DEFAULT_SOURCES:
        key = str(default.get("key"))
        if key and key not in known:
            merged.append(json.loads(json.dumps(default)))  # 深拷贝，避免共享引用
            added.append(key)
    return merged, added


def load_sources() -> list[dict[str, Any]]:
    """读取用户可编辑的来源清单；文件不存在时写入默认值。"""
    path = sources_file_path()
    if not path.exists():
        save_sources(DEFAULT_SOURCES)
        return [dict(item) for item in DEFAULT_SOURCES]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("sources"), list):
            data = data["sources"]
        if isinstance(data, list):
            merged, added = _merge_missing_defaults(data)
            if added:
                logger.info("来源清单补入内置来源：%s", "、".join(added))
                save_sources(merged)
            return merged
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 %s 失败，使用默认来源：%s", path.name, exc)
    return [dict(item) for item in DEFAULT_SOURCES]


def save_sources(sources: list[dict[str, Any]]) -> Path:
    # 统一走 sources_file_path()，保证「读」和「写」永远是同一个文件
    path = sources_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "sources": sources}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
