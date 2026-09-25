"""验证通用渲染爬虫能否真的从真实招聘站抓到岗位。

    python scripts/check_render_crawler.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.crawlers.base import CrawlBudget  # noqa: E402
from backend.crawlers.generic_render import GenericRenderCrawler  # noqa: E402

TARGETS = [
    ("智联招聘·杭州AI", "https://sou.zhaopin.com/?kw=AI&jl=653"),
    ("牛客·职位广场", "https://www.nowcoder.com/jobs/fulltime/center?recruitType=1&keyword=AI"),
]


def main() -> int:
    ok_count = 0
    for label, url in TARGETS:
        print("=" * 78)
        print(f"  目标：{label}")
        print(f"  URL ：{url}")
        print("=" * 78)

        source = {
            "key": "probe",
            "label": label,
            "kind": "generic_render",
            "careers_url": url,
            "config": {
                "list_url": url,
                "max_pages": 1,
                "fetch_detail": False,
                "_keywords": ["ai"],
                "_city": "",
            },
        }
        crawler = GenericRenderCrawler(source)
        started = time.time()
        try:
            with CrawlBudget(70) as budget:
                jobs = crawler.fetch(budget)
        except Exception as exc:  # noqa: BLE001
            print(f"  失败：{type(exc).__name__}: {str(exc)[:180]}")
            print()
            continue

        elapsed = time.time() - started
        print(f"  抓取到 {len(jobs)} 条，用时 {elapsed:.1f}s")
        for job in jobs[:6]:
            print(f"    · {job.title[:44]:46} | 城市={job.city[:10]:10} | 薪资={job.salary_text or '—'}")
            if job.url:
                print(f"      {job.url[:100]}")
        if jobs:
            ok_count += 1
            with_url = sum(1 for j in jobs if j.url)
            with_salary = sum(1 for j in jobs if j.salary_text)
            print(f"  统计：带链接 {with_url}/{len(jobs)}，带薪资 {with_salary}/{len(jobs)}")
        print()

    print("=" * 78)
    if ok_count:
        print(f"  通用渲染爬虫可用：{ok_count}/{len(TARGETS)} 个目标成功抓到岗位")
    else:
        print("  通用渲染爬虫未能从这些目标抓到岗位（需要为对应站点配置 item_selector）")
    print("=" * 78)
    return 0 if ok_count else 1


if __name__ == "__main__":
    sys.exit(main())
