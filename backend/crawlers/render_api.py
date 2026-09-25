"""「过 WAF + 拦截 XHR」型爬虫。

有些招聘站（如 51job）用阿里云 WAF 挡住了直接请求：httpx 拿到的是一段
JS 挑战页而不是数据。但**用真实浏览器访问时 WAF 是放行的**。

于是有了这个适配器：

    用 Playwright 打开搜索页 → 监听页面发出的 XHR → 拦截其中的 JSON 响应
    → 按配置的 JSON 路径取出岗位数组 → 转成统一的 JobItem

相比「渲染后抓 DOM」，好处是：

* **拿到的是结构化数据**（薪资甚至有数字字段），不用去猜 DOM 结构和文本格式
* 站点改版只要接口没变就还能用
* 配置写在 ``sources.json`` 里，加一个新站不用改代码

职责边界：本类**只负责把结构化数据取回来**。按职位名过滤、按城市过滤、
薪资文本解析、两级去重都由 ``crawlers/__init__.py`` 的 ``crawl_for_profile``
统一处理，这里不重复实现（否则同一套规则会有两份，早晚不一致）。

配置示例（``sources.json`` 里的一条）：:

    {
      "key": "51job",
      "kind": "render_api",
      "careers_url": "https://we.51job.com/pc/search",
      "config": {
        "page_url": "https://we.51job.com/pc/search?keyword={keyword}&jobArea={city_code}&pageNum={page}",
        "intercept": ["/api/job/search-pc"],
        "items_path": "resultbody.job.items",
        "pages_per_term": 2,
        "max_terms": 2,
        "city_codes": {"北京": "010000", "上海": "020000"},
        "field_map": {
          "title": "jobName",
          "company": "companyName",
          "city": "jobAreaString",
          "url": "jobHref",
          "salary_text": "provideSalaryString",
          "salary_min_yuan": "jobSalaryMin",
          "salary_max_yuan": "jobSalaryMax",
          "published_at": "issueDateString",
          "tags": "jobTags"
        }
      }
    }

``page_url`` 里可用的占位符：``{keyword}`` ``{page}`` ``{city_code}`` ``{city}``。
某个占位符取不到值时，连同它所在的查询参数一起删掉（见 ``_build_url``）——
所以「城市没有配置代码」不会拼出一个 ``jobArea=`` 的空参数，而是退化成全国检索，
再由上层按城市过滤，结果依然正确。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import quote

from .base import BaseCrawler, CrawlBudget, JobItem
from .fieldmap import build_job_from_mapping, json_path_get

logger = logging.getLogger("ai_resume_helper.crawlers.render_api")


#: 要彻底删掉整个参数的占位符（值为空时）
_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")

#: 出现这些字样说明撞上了人机验证页（而不是「站点改版了」）
_CAPTCHA_MARKERS = ("滑动验证", "访问验证", "安全验证", "验证页面", "captcha", "robot")
_CAPTCHA_SELECTORS = (
    ".nc_container", "#nc_1_wrapper", "#nc_1_n1z",
    "[class*=captcha]", "[id*=captcha]", "[class*=slider]",
)


class CaptchaEncountered(RuntimeError):
    """撞上了站点的滑动验证/人机验证，本次拿不到数据。

    这是**需要如实上报**的情况，不能悄悄算成「0 个岗位」——
    否则用户只会看到「这个来源怎么没数据」，无从判断是站点改版、网络问题，
    还是被拦了。实际原因就是被拦了。
    """


class PlaywrightApiCrawler(BaseCrawler):
    """用 Playwright 过 WAF，拦截站点的 JSON 接口。"""

    kind = "render_api"

    # ---------------- 配置读取 ----------------
    @property
    def page_url_template(self) -> str:
        return str(self.config.get("page_url") or self.careers_url or "")

    @property
    def intercept_patterns(self) -> list[str]:
        raw = self.config.get("intercept") or []
        if isinstance(raw, str):
            raw = [raw]
        return [str(item) for item in raw if str(item).strip()]

    @property
    def items_path(self) -> str:
        return str(self.config.get("items_path") or "")

    @property
    def field_map(self) -> dict[str, Any]:
        return dict(self.config.get("field_map") or {})

    @property
    def pages_per_term(self) -> int:
        return max(1, int(self.config.get("pages_per_term") or 1))

    @property
    def max_terms(self) -> int:
        return max(1, int(self.config.get("max_terms") or 2))

    @property
    def max_jobs(self) -> int:
        return max(1, int(self.config.get("max_jobs") or 200))

    def _search_terms(self) -> list[str]:
        terms = [
            str(t).strip()
            for t in (self.config.get("_search_terms") or [])
            if str(t).strip()
        ]
        if terms:
            return terms[: self.max_terms]
        single = str(self.config.get("_search_keyword") or "").strip()
        return [single] if single else [""]

    def _city_code(self) -> str:
        """城市 → 站点区域代码。没配置就返回空串，退化为全国检索。"""
        city = str(self.config.get("_city") or "").strip()
        codes = self.config.get("city_codes") or {}
        if not city or not isinstance(codes, dict):
            return ""
        if city in codes:
            return str(codes[city])
        # 允许「北京·朝阳区」这类带后缀的城市名
        for name, code in codes.items():
            if name and name in city:
                return str(code)
        return ""

    def _build_url(self, keyword: str, page: int) -> str:
        """填占位符，并把「填不出值」的参数整段删掉。"""
        values = {
            "keyword": quote(str(keyword)),
            "page": str(page),
            "city": quote(str(self.config.get("_city") or "")),
            "city_code": self._city_code(),
            "timestamp": str(int(time.time() * 1000)),
        }
        template = self.page_url_template

        # 先把「值为空的占位符」记下来，用于整段删除它所在的参数
        empty_names = {
            name for name, value in values.items() if not value and f"{{{name}}}" in template
        }
        if empty_names:
            # 注意字符类里**不能排除 `=`**：参数名和值之间的 `=` 必须能匹配到，
            # 否则 `&jobArea={city_code}` 这种「名=值」的参数永远匹配不上。
            pattern = re.compile(
                r"[?&][^?&]*(?:" + "|".join(re.escape(f"{{{n}}}") for n in empty_names) + r")[^&]*"
            )
            template = pattern.sub("", template)
            if "?" not in template and "&" in template:
                template = template.replace("&", "?", 1)

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            return values.get(name, match.group(0))

        return _PLACEHOLDER_RE.sub(_replace, template)

    # ---------------- 拦截 ----------------
    def _collect(
        self,
        context: Any,
        url: str,
        budget: CrawlBudget,
        pages: int = 1,
        wait_seconds: float = 25.0,
    ) -> list[Any]:
        """打开页面并收集目标接口的 JSON 响应（可翻页）。

        翻页方式的坑：把 ``pageNum=2`` 拼进搜索页 URL **不会真的翻页**，
        站点还是返回第 1 页（表现为「拦截到 2 个响应，但只有 20 个唯一岗位」）。
        所以翻页要靠**点真实的「下一页」按钮**，再拦截随之发出的新请求。

        另外试过「录下浏览器的 XHR 请求+Cookie，用 httpx 改 pageNum 重放」，
        被 WAF 挡住了（返回非 JSON）。所以只能留在浏览器里点。
        """
        payloads: list[Any] = []
        page = context.new_page()

        def on_response(response: Any) -> None:
            if not any(pattern in response.url for pattern in self.intercept_patterns):
                return
            try:
                data = response.json()
            except Exception:  # noqa: BLE001 - 不是 JSON 就跳过
                return
            if json_path_get(data, self.items_path) is not None:
                payloads.append(data)

        page.on("response", on_response)
        try:
            page.goto(url, wait_until="domcontentloaded")
            deadline = time.monotonic() + wait_seconds
            checks = 0
            while not payloads and time.monotonic() < deadline:
                page.wait_for_timeout(400)
                checks += 1
                if budget.remaining() <= 0:
                    break
                # 撞上验证页就立刻放弃，不用白等到超时
                if checks >= 2 and self._looks_like_captcha(page):
                    raise CaptchaEncountered(
                        "触发了人机验证（滑动验证），本次跳过。"
                        "该来源靠真实浏览器访问，短时间内请求过多会被要求验证；"
                        "稍后重试即可，也可在「岗位来源」里先关掉它。"
                    )

            next_selector = str(self.config.get("next_selector") or "")
            for _ in range(max(0, pages - 1)):
                if not next_selector or not payloads or budget.remaining() <= 0:
                    break
                before = len(payloads)
                try:
                    page.click(next_selector, timeout=6000)
                except Exception as exc:  # noqa: BLE001
                    logger.info(
                        "[%s] 点「下一页」失败，停止翻页：%s", self.label, type(exc).__name__
                    )
                    break
                page_deadline = time.monotonic() + 20.0
                while len(payloads) == before and time.monotonic() < page_deadline:
                    page.wait_for_timeout(400)
                    if budget.remaining() <= 0:
                        break
                if len(payloads) == before:
                    logger.info("[%s] 第 %d 页没有新数据，停止翻页", self.label, before + 1)
                    break
        finally:
            page.close()
        return payloads

    @staticmethod
    def _looks_like_captcha(page: Any) -> bool:
        """判断当前页面是不是人机验证页。"""
        try:
            title = (page.title() or "").lower()
        except Exception:  # noqa: BLE001
            return False
        if any(marker in title for marker in _CAPTCHA_MARKERS):
            return True
        try:
            for selector in _CAPTCHA_SELECTORS:
                if page.query_selector(selector) is not None:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    # ---------------- 字段转换 ----------------
    def _to_job(self, item: dict[str, Any]) -> JobItem | None:
        return build_job_from_mapping(
            self.make_job,
            item,
            self.field_map,
            url_strip_query=bool(self.config.get("url_strip_query")),
        )

    # ---------------- 主流程 ----------------
    def fetch(self, budget: CrawlBudget) -> list[JobItem]:
        from playwright.sync_api import sync_playwright

        from config import get_settings

        if not self.page_url_template:
            raise ValueError("未配置 page_url（搜索页地址）")
        if not self.items_path:
            raise ValueError("未配置 items_path（岗位数组的 JSON 路径）")
        if not self.intercept_patterns:
            raise ValueError("未配置 intercept（要拦截的接口 URL 片段）")

        settings = get_settings()
        jobs: list[JobItem] = []
        seen: set[str] = set()
        captured_pages = 0

        with sync_playwright() as playwright:
            browser = self.launch_browser(
                playwright,
                channel=settings.crawler_browser_channel,
                headless=settings.crawler_headless,
            )
            try:
                context = browser.new_context(
                    locale="zh-CN",
                    viewport={"width": 1500, "height": 950},
                )
                context.set_default_timeout(settings.crawler_navigate_timeout_ms)

                for term in self._search_terms():
                    if budget.remaining() <= 0 or len(jobs) >= self.max_jobs:
                        break
                    url = self._build_url(term, 1)
                    payloads = self._collect(
                        context, url, budget, pages=self.pages_per_term
                    )
                    if not payloads:
                        logger.warning(
                            "[%s] 未拦截到接口响应（词=%r）——可能触发了验证码",
                            self.label,
                            term,
                        )
                        continue

                    for payload in payloads:
                        captured_pages += 1
                        items = json_path_get(payload, self.items_path) or []
                        if not isinstance(items, list) or not items:
                            continue
                        for raw in items:
                            if isinstance(raw, dict):
                                job = self._to_job(raw)
                                if job is not None and job.job_key not in seen:
                                    seen.add(job.job_key)
                                    jobs.append(job)

                context.close()
            finally:
                browser.close()

        with_salary = sum(1 for job in jobs if job.salary_text)
        logger.info(
            "[%s] 拦截 %d 个接口响应，解析出 %d 个岗位（含薪资文本 %d 条）",
            self.label,
            captured_pages,
            len(jobs),
            with_salary,
        )
        return jobs


__all__ = ["PlaywrightApiCrawler", "json_path_get"]
