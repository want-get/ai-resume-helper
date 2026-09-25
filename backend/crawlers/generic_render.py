"""通用渲染爬虫：用 Playwright 驱动**系统自带的 Edge/Chrome**抓取任意招聘页。

为什么需要它？
    国内公司的官网绝大多数是 JS 渲染的 SPA，直接请求 HTML 拿不到岗位列表；
    给每家公司写一个适配器维护成本太高。这个爬虫用「配置 + 自动识别」
    覆盖长尾站点：用户只要填一个列表页 URL 就能抓。

两级抓取（这是关键）：
    1. 渲染列表页，抽取所有岗位的「标题 + 链接 + 城市 + 薪资」
    2. **只对关键字命中的候选**再打开详情页补全 JD 正文

自动识别策略（用户没填选择器时）：
    扫描 DOM，找出「重复出现多次、内部文本较短、且包含链接」的容器，
    选出现次数最多的那个作为岗位卡片选择器。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from .base import BaseCrawler, CrawlBudget, JobItem
from .matching import city_matches, title_matches
logger = logging.getLogger("ai_resume_helper.crawlers.render")

# 判断一段文本像不像薪资（用于自动识别薪资字段）
_SALARY_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*[-~至]\s*\d+(?:\.\d+)?\s*[kK千万]|\d+\s*[kK]\b|面议|\d+(?:\.\d+)?\s*万)",
)

# 判断一段文本像不像城市
_CITY_HINT_RE = re.compile(r"(北京|上海|广州|深圳|杭州|成都|武汉|南京|西安|苏州|天津|重庆|长沙|郑州|青岛|合肥|厦门|福州|济南|大连|宁波|无锡|佛山|东莞|珠海|香港|远程)")

# 职位卡片里的「非标题」行：经验、学历、类型、规模等
_NOISE_LINE_RE = re.compile(
    r"^(\d+\s*[-~至]?\s*\d*\s*年|应届|在校|不限|本科|硕士|博士|大专|学历不限|"
    r"全职|兼职|实习|校招|社招|远程|面议|\d+\s*人|\d+-\d+人|已上市|未融资|"
    r"[A-ZＢＣＤ]轮|不需要融资|天使轮|国企|外企|民营|上市公司)",
    re.I,
)


class GenericRenderCrawler(BaseCrawler):
    """配置驱动的通用招聘页爬虫。"""

    kind = "generic_render"

    # ---------------- 配置读取 ----------------
    @property
    def list_url(self) -> str:
        return str(self.config.get("list_url") or self.careers_url or "")

    @property
    def item_selector(self) -> str:
        return str(self.config.get("item_selector") or "")

    @property
    def title_selector(self) -> str:
        return str(self.config.get("title_selector") or "")

    @property
    def link_selector(self) -> str:
        return str(self.config.get("link_selector") or "a")

    @property
    def max_pages(self) -> int:
        return int(self.config.get("max_pages") or 1)

    @property
    def next_selector(self) -> str:
        """下一页按钮选择器（可选）。"""
        return str(self.config.get("next_selector") or "")

    # ---------------- 自动识别岗位卡片 ----------------
    @staticmethod
    def _autodetect_item_selector(page: Any) -> str:
        """找出最像「岗位卡片列表」的容器选择器。

        思路：统计每个 class 出现的次数，挑出
        「出现 >= 4 次、内部含有链接、文本长度适中」的那个。
        """
        script = """
        () => {
          const stats = {};
          document.querySelectorAll('li, div, article, tr').forEach((el) => {
            if (el.querySelectorAll('a').length === 0) return;
            const text = (el.innerText || '').trim();
            if (text.length < 4 || text.length > 400) return;
            const cls = (el.className && typeof el.className === 'string')
              ? el.className.trim().split(/\\s+/).filter(Boolean).slice(0, 2).join('.')
              : '';
            const key = el.tagName.toLowerCase() + (cls ? '.' + cls : '');
            stats[key] = (stats[key] || 0) + 1;
          });
          const best = Object.entries(stats)
            .filter(([, count]) => count >= 4)
            .sort((a, b) => b[1] - a[1]);
          return best.length ? best[0][0] : '';
        }
        """
        try:
            return str(page.evaluate(script) or "")
        except Exception as exc:  # noqa: BLE001
            logger.debug("自动识别岗位卡片失败：%s", exc)
            return ""

    # ---------------- 列表页抽取 ----------------
    def _extract_list(self, page: Any, selector: str) -> list[dict[str, str]]:
        raw = page.eval_on_selector_all(
            selector,
            """
            (nodes) => nodes.map((node) => {
              const link = node.querySelector('a[href]');
              const text = (node.innerText || '').trim().replace(/\\s+/g, ' ');
              const lines = (node.innerText || '')
                .split('\\n').map((s) => s.trim()).filter(Boolean);
              return {
                text: text.slice(0, 400),
                lines: lines.slice(0, 8),
                href: link ? link.href : '',
                title: link ? (link.innerText || '').trim().slice(0, 120) : '',
              };
            })
            """,
        )
        return [item for item in (raw or []) if item.get("text")]

    @staticmethod
    def _guess_field(item: dict[str, Any], field: str) -> str:
        """在卡片文本里猜城市 / 薪资。"""
        lines: list[str] = item.get("lines") or []
        text: str = item.get("text") or ""
        if field == "city":
            for line in lines:
                if _CITY_HINT_RE.search(line) and len(line) <= 40:
                    return line.strip()
            match = _CITY_HINT_RE.search(text)
            return match.group(1) if match else ""
        if field == "salary":
            match = _SALARY_RE.search(text)
            return match.group(1).strip() if match else ""
        return ""

    def _extract_title(self, item: dict[str, Any]) -> str:
        """从卡片里挑出真正的职位名。

        很多站点的 ``<a>`` 包住**整张卡片**，``link.innerText`` 会是好几行，
        直接拿来当标题就会出现「ai产品经理 17-27K 深圳 1-3年…」这种脏数据。
        所以：单行的链接文本最可信；否则逐行排除薪资/城市/年限这些噪声行。
        """
        link_text = str(item.get("title") or "").strip()
        # 1) 链接文本是单行且长度合适 —— 最可信
        if link_text and "\n" not in link_text and 2 <= len(link_text) <= 90:
            return link_text

        # 2) 逐行挑第一个「不像元信息」的短行
        for line in item.get("lines") or []:
            text = str(line).strip()
            if not 2 <= len(text) <= 90:
                continue
            if _SALARY_RE.search(text):
                continue
            if _CITY_HINT_RE.fullmatch(text):
                continue
            if _NOISE_LINE_RE.match(text):
                continue
            return text

        # 3) 兜底：链接文本（或整卡文本）的第一行
        source = link_text or str(item.get("text") or "")
        first_line = source.split("\n")[0].strip()
        return first_line[:120]

    # ---------------- 详情页补全 ----------------
    def _extract_detail(self, page: Any) -> str:
        """打开详情页后抽取 JD 正文。

        优先按配置的 ``detail_selector``，否则取正文里最长的一段文本块。
        """
        selector = str(self.config.get("detail_selector") or "")
        try:
            if selector:
                text = page.inner_text(selector)
                if text and len(text.strip()) >= 40:
                    return text.strip()[:12000]
        except Exception:  # noqa: BLE001
            pass

        try:
            body = page.inner_text("body")
        except Exception:  # noqa: BLE001
            return ""
        # 只保留含 JD 信号词的段落，避免把导航/页脚一起塞进去
        signal = re.compile(
            r"岗位职责|工作职责|职位描述|任职要求|岗位要求|任职资格|工作内容|"
            r"responsibilit|requirement|qualification",
            re.I,
        )
        chunks = [chunk.strip() for chunk in re.split(r"\n{2,}", body) if len(chunk.strip()) >= 30]
        picked = [chunk for chunk in chunks if signal.search(chunk)]
        if picked:
            return "\n".join(picked)[:12000]
        return ("\n".join(chunks)[:4000]) if chunks else ""

    # ---------------- 主流程 ----------------
    def fetch(self, budget: CrawlBudget) -> list[JobItem]:
        from playwright.sync_api import sync_playwright

        from config import get_settings

        settings = get_settings()
        url = self.list_url
        if not url:
            raise ValueError("未配置 list_url（招聘列表页地址）")

        jobs: list[JobItem] = []
        with sync_playwright() as playwright:
            browser = self.launch_browser(
                playwright,
                channel=settings.crawler_browser_channel,
                headless=settings.crawler_headless,
            )
            try:
                context = browser.new_context(
                    locale="zh-CN",
                    user_agent=None,
                    viewport={"width": 1440, "height": 900},
                )
                page = context.new_page()
                page.set_default_timeout(settings.crawler_navigate_timeout_ms)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(settings.crawler_render_wait_ms)

                selector = self.item_selector or self._autodetect_item_selector(page)
                if not selector:
                    raise RuntimeError(
                        "没能自动识别岗位列表。请在该来源的配置里填写 item_selector"
                        "（岗位卡片的 CSS 选择器）。"
                    )
                logger.info("[%s] 使用岗位卡片选择器：%s", self.label, selector)

                collected: dict[str, dict[str, str]] = {}
                for page_index in range(max(1, self.max_pages)):
                    budget.check()
                    for item in self._extract_list(page, selector):
                        title = self._extract_title(item)
                        if not title:
                            continue
                        key = f"{title}|{item.get('href', '')}"
                        if key not in collected:
                            collected[key] = {**item, "title": title}

                    if page_index + 1 >= self.max_pages or not self.next_selector:
                        break
                    try:
                        page.click(self.next_selector, timeout=5000)
                        page.wait_for_timeout(settings.crawler_render_wait_ms)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("[%s] 翻页结束：%s", self.label, exc)
                        break

                # ---- 按标题关键字筛候选，再补详情 ----
                keywords = [str(k) for k in (self.config.get("_keywords") or [])]
                city_filter = str(self.config.get("_city") or "")
                candidates: list[dict[str, str]] = []
                for item in collected.values():
                    title = item["title"]
                    hits = title_matches(title, keywords) if keywords else []
                    if keywords and not hits:
                        continue
                    item["_hits"] = ",".join(hits)
                    candidates.append(item)

                detail_limit = int(self.config.get("detail_limit") or 12)
                want_detail = bool(self.config.get("fetch_detail", True))

                for index, item in enumerate(candidates):
                    budget.check()
                    city = self._guess_field(item, "city")
                    if city_filter and not city_matches(city or item.get("text", ""), city_filter):
                        continue

                    jd_text = ""
                    href = item.get("href") or ""
                    if want_detail and href and index < detail_limit:
                        try:
                            detail_page = context.new_page()
                            detail_page.set_default_timeout(
                                settings.crawler_navigate_timeout_ms
                            )
                            detail_page.goto(href, wait_until="domcontentloaded")
                            detail_page.wait_for_timeout(600)
                            jd_text = self._extract_detail(detail_page)
                            detail_page.close()
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("[%s] 详情页失败 %s：%s", self.label, href, exc)

                    jobs.append(
                        self.make_job(
                            title=item["title"],
                            city=city,
                            salary_text=self._guess_field(item, "salary"),
                            job_type=str(self.config.get("job_type") or ""),
                            url=href,
                            jd_text=jd_text or item.get("text", ""),
                            matched_keywords=[h for h in item.get("_hits", "").split(",") if h],
                        )
                    )

                context.close()
            finally:
                browser.close()

        logger.info("[%s] 抓到 %d 个岗位", self.label, len(jobs))
        return jobs


__all__ = ["GenericRenderCrawler"]
