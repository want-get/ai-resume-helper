"""校验 51job 城市代码表：逐个确认内置的代码真的对应目标城市。

为什么需要这个脚本：区域代码错了**不会报错**，只会安静地返回别的城市的岗位。
开发时就踩过——凭记忆写的映射表有 8 个是错的（南通→常熟、南昌→石家庄、
哈尔滨→长春…）。所以内置的代码表必须能随时重新验证。

判定方式：用该代码搜索一个常见职位，检查返回岗位的 ``jobAreaString``
是否真的落在目标城市。

用法：
    python scripts/check_51job_city_codes.py                  # 校验全部内置城市
    python scripts/check_51job_city_codes.py 北京 上海          # 只校验指定城市
    python scripts/check_51job_city_codes.py --delay 20       # 放慢请求间隔

⚠️ 请**克制使用**：连续快速请求会触发滑动验证（本脚本已内置间隔，
但整体仍是个消耗配额的操作）。被拦截时会明确提示，不会给出错误结论。
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.crawlers.base import DEFAULT_SOURCES, CrawlBudget  # noqa: E402
from backend.crawlers.render_api import (  # noqa: E402
    CaptchaEncountered,
    PlaywrightApiCrawler,
    json_path_get,
)

#: 用一个常见职位来探测，保证每个城市都能搜出结果
PROBE_KEYWORD = "会计"
#: 两次探测之间的间隔（秒）。请求太密会被要求滑动验证
DEFAULT_DELAY = 12.0


def shipped_city_codes() -> dict[str, str]:
    source = next(s for s in DEFAULT_SOURCES if s["key"] == "51job")
    return dict(source["config"].get("city_codes") or {})


def make_crawler(city: str, code: str) -> PlaywrightApiCrawler:
    crawler = PlaywrightApiCrawler({
        "key": "probe",
        "label": f"51job·{city}",
        "kind": "render_api",
        "careers_url": "https://we.51job.com/pc/search",
        "config": {
            "page_url": (
                "https://we.51job.com/pc/search?keyword={keyword}&jobArea={city_code}"
            ),
            "intercept": ["/api/job/search-pc"],
            "items_path": "resultbody.job.items",
            "city_codes": {city: code},
            "field_map": {"title": "jobName", "city": "jobAreaString"},
        },
    })
    crawler.config["_search_terms"] = [PROBE_KEYWORD]
    crawler.config["_search_keyword"] = PROBE_KEYWORD
    crawler.config["_city"] = city
    return crawler


def parse_args() -> tuple[list[str], float]:
    args = sys.argv[1:]
    delay = DEFAULT_DELAY
    cities: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--delay" and index + 1 < len(args):
            delay = float(args[index + 1])
            index += 2
            continue
        cities.append(args[index])
        index += 1
    return cities, delay


def main() -> int:
    wanted, delay = parse_args()
    codes = shipped_city_codes()
    if wanted:
        codes = {city: code for city, code in codes.items() if city in wanted}

    print("=" * 74)
    print(f"51job 城市代码校验（探测职位：{PROBE_KEYWORD}，共 {len(codes)} 个城市）")
    print("=" * 74)

    ok: list[str] = []
    bad: list[str] = []
    blocked: list[str] = []

    # 复用**同一个浏览器**跑完所有城市：每个城市各开一个浏览器会瞬间打满配额，
    # 反而更容易被要求验证（这正是本脚本第一版的毛病）
    from config import get_settings
    from playwright.sync_api import sync_playwright

    settings = get_settings()
    with sync_playwright() as playwright:
        browser = PlaywrightApiCrawler.launch_browser(
            playwright,
            channel=settings.crawler_browser_channel,
            headless=settings.crawler_headless,
        )
        context = None
        try:
            context = browser.new_context(locale="zh-CN")
            context.set_default_timeout(settings.crawler_navigate_timeout_ms)

            for position, (city, code) in enumerate(codes.items()):
                if position:
                    time.sleep(delay)
                crawler = make_crawler(city, code)
                try:
                    payloads = crawler._collect(
                        context, crawler._build_url(PROBE_KEYWORD, 1), CrawlBudget(60.0)
                    )
                except CaptchaEncountered:
                    print(f"⚠️  {city:<4} code={code}  被滑动验证拦截（已停止后续校验）")
                    blocked.append(city)
                    break  # 已被拦，继续跑只会更糟
                except Exception as exc:  # noqa: BLE001
                    print(f"⚠️  {city:<4} code={code}  异常：{type(exc).__name__}: {exc}")
                    continue

                if not payloads:
                    print(f"⚠️  {city:<4} code={code}  未拿到数据")
                    continue

                payload = payloads[0]
                total = json_path_get(payload, "resultbody.job.totalCount") or 0
                items = json_path_get(payload, "resultbody.job.items") or []
                areas: Counter = Counter()
                for item in items:
                    name = str(item.get("jobAreaString") or "").strip()
                    areas[name.split("·")[0].split(" ")[0]] += 1

                good = any(city in name for name in areas)
                top = "、".join(f"{n}×{c}" for n, c in areas.most_common(4))
                print(
                    f"{'✅' if good else '❌'} {city:<4} code={code}  "
                    f"职位总数={total:<6} 首页: {top or '（空）'}"
                )
                (ok if good else bad).append(city)
        finally:
            if context is not None:
                context.close()
            browser.close()

    print("-" * 74)
    print(f"通过 {len(ok)} 个，未通过 {len(bad)} 个，被拦截未校验 {len(blocked)} 个")
    if bad:
        print(f"有问题的城市（代码可能已变更，请重新采集）：{'、'.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
