"""真实岗位抓取自检：按用户档案抓取岗位 + 统计薪资。

    python scripts/check_crawl.py                    # 用默认档案（AI Agent 开发 / 北京）
    python scripts/check_crawl.py "Python后端" 杭州
    python scripts/check_crawl.py --list             # 只看已配置的来源
    python scripts/check_crawl.py --source jd_campus # 只抓一个来源

会真实联网请求招聘网站；抓下来的岗位会写入本地数据库。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.crawlers import crawl_for_profile, load_sources, test_source  # noqa: E402
from backend.db import dispose_database, init_database  # noqa: E402
from backend.profile import UserProfileData, derive_keywords  # noqa: E402


async def main(args: argparse.Namespace) -> int:
    if args.list:
        sources = load_sources()
        print("已配置的岗位来源：")
        for item in sources:
            flag = "启用" if item.get("enabled", True) else "停用"
            print(f"  [{flag}] {item.get('key'):14} {item.get('kind'):16} "
                  f"{item.get('label')}  ->  {item.get('careers_url') or '(未填URL)'}")
        return 0

    await init_database()

    if args.source:
        print(f"试抓单个来源：{args.source}")
        result = await test_source(args.source)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        await dispose_database()
        return 0 if result.get("ok") else 1

    role = args.role
    city = args.city
    profile = UserProfileData(
        target_role=role,
        expect_city=city,
        expect_salary_min=args.salary_min,
        expect_salary_max=args.salary_max,
        target_keywords=[],
    )

    print("=" * 78)
    print("  开始抓取真实岗位")
    print("=" * 78)
    print(f"  求职方向：{role}")
    print(f"  期望城市：{city}")
    print(f"  期望薪资：{args.salary_min:g}~{args.salary_max:g}K")
    print(f"  匹配关键字：{profile.search_keywords()}")
    print("=" * 78)

    report = await crawl_for_profile(profile)

    print("\n各来源结果：")
    for item in report.sources:
        mark = "✅" if item.ok else "❌"
        line = f"  {mark} {item.label:16} 抓取 {item.fetched:4d} 条  耗时 {item.elapsed_ms:7.0f}ms"
        if item.error:
            line += f"  错误：{item.error[:80]}"
        print(line)

    print(f"\n关键字扩展：")
    for original, variants in report.keyword_expansion.items():
        print(f"  {original!r} -> {variants}")

    print(f"\n去重后岗位：{len(report.jobs)} 条（去重丢弃 {report.deduped} 条），"
          f"其中 {sum(1 for j in report.jobs if j.salary_mid_k is not None)} 条有薪资")

    if report.jobs:
        print("\n岗位样例（前 12 条）：")
        for job in report.jobs[:12]:
            salary = job.salary_text or "—"
            print(f"  · {job.title[:46]:48} | {job.city[:14]:14} | {salary:12} | {job.source}")
            if job.matched_keywords:
                print(f"      命中：{'、'.join(job.matched_keywords)}")

    summary = report.salary
    print("\n" + "-" * 78)
    print("  薪资行情统计")
    print("-" * 78)
    if summary.get("sample_size"):
        monthly = summary["monthly"]
        print(f"  样本量：{summary['sample_size']} 条（人民币）")
        print(f"  月薪　：平均 {monthly['mean']}K | 中位 {monthly['median']}K | "
              f"P25 {monthly['p25']}K | P75 {monthly['p75']}K")
        print(f"  区间　：{monthly['min']}K ~ {monthly['max']}K")
        print(f"  年薪中位：{summary['annual']['median']}K")
        if summary.get("low_confidence"):
            print("  ⚠️  样本偏少，参考价值有限")
        expectation = summary.get("expectation")
        if expectation:
            print(f"\n  你的期望：{expectation['expected_mid_k']}K（中位）")
            print(f"  市场对比：{expectation['verdict']}"
                  f"（差 {expectation['gap_k']:+}K / {expectation['gap_ratio']:+}%）")
            print(f"  建议　　：{expectation['advice']}")
    else:
        print(f"  没有可统计的薪资样本。（解析到的样本数：{summary.get('parsed_size', 0)}）")

    for warning in report.warnings:
        print(f"\n  ⚠️  {warning}")

    print(f"\n入库 {report.saved} 条，总耗时 {report.elapsed_ms / 1000:.1f}s")
    await dispose_database()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实岗位抓取自检")
    parser.add_argument("role", nargs="?", default="AI Agent 开发工程师")
    parser.add_argument("city", nargs="?", default="北京")
    parser.add_argument("--salary-min", type=float, default=25)
    parser.add_argument("--salary-max", type=float, default=40)
    parser.add_argument("--list", action="store_true", help="只看已配置来源")
    parser.add_argument("--source", default="", help="只试抓一个来源")
    sys.exit(asyncio.run(main(parser.parse_args())))
