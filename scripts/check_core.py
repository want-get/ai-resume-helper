"""核心纯函数自检：薪资解析、关键字推导、大模型配置（不需要网络与数据库）。

    python scripts/check_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from backend.llm_config import get_llm_config  # noqa: E402
from backend.llm_providers import provider_list  # noqa: E402
from backend.profile import derive_keywords  # noqa: E402
from backend.salary import parse_salary, summarize_salaries  # noqa: E402
from paths import describe as describe_paths  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  {'OK ' if condition else 'FAIL'} {name}" + (f"  ->  {detail}" if detail else ""))


SALARY_CASES: list[tuple[str, tuple[float, float] | None, int]] = [
    ("25-40K·14薪", (25, 40), 14),
    ("20-35k", (20, 35), 12),
    ("2-3万/月", (20, 30), 12),
    ("20-40万/年", (16.67, 33.33), 12),
    ("8000-12000元/月", (8, 12), 12),
    ("15k-25k·13薪", (15, 25), 13),
    ("1.5-2万", (15, 20), 12),
    ("30K", (30, 30), 12),
    ("薪资面议", None, 12),
    ("面议", None, 12),
    ("", None, 12),
]


def check_salary() -> None:
    print("=" * 74)
    print("[1] 薪资文本解析")
    print("=" * 74)
    for text, expected, months in SALARY_CASES:
        result = parse_salary(text)
        if expected is None:
            check(f"{text!r} 应解析为 None", result is None, str(result))
            continue
        if result is None:
            check(f"{text!r} 解析失败", False, "返回 None")
            continue
        ok = (
            abs(result.low_k - expected[0]) < 0.05
            and abs(result.high_k - expected[1]) < 0.05
            and result.months == months
        )
        check(
            f"{text!r}",
            ok,
            f"{result.low_k:g}~{result.high_k:g}K x{result.months}",
        )

    print()
    print("-" * 74)
    print("[2] 外币薪资不与人民币混算")
    print("-" * 74)
    usd = parse_salary("$120,000 - $180,000 a year")
    check("美元年薪可解析", usd is not None and usd.currency == "USD", str(usd.to_dict() if usd else None))
    mixed = summarize_salaries(
        [parse_salary("25-40K"), parse_salary("$120,000 a year"), parse_salary("30-40K")]
    )
    check(
        "统计只算人民币样本",
        mixed["sample_size"] == 2 and mixed["other_currency_samples"] == 1,
        f"cny={mixed['sample_size']} other={mixed['other_currency_samples']}",
    )

    print()
    print("-" * 74)
    print("[3] 市场统计与期望对比")
    print("-" * 74)
    texts = ["25-40K·14薪", "20-35K", "30-50K·15薪", "18-28K", "面议", "40-60K·16薪"]
    ranges = [r for r in (parse_salary(t) for t in texts) if r]
    summary = summarize_salaries(ranges, expected_min_k=25, expected_max_k=35)
    check("样本数正确（面议被剔除）", summary["sample_size"] == 5, str(summary["sample_size"]))
    mids = sorted(r.mid_k for r in ranges)
    check(
        "中位数与手算一致",
        abs(summary["monthly"]["median"] - mids[len(mids) // 2]) < 0.2,
        f"median={summary['monthly']['median']} 手算={mids[len(mids)//2]}",
    )
    check("含期望对比", "expectation" in summary, summary.get("expectation", {}).get("verdict", ""))
    check("样本少时标记低置信", summary["low_confidence"] is False or summary["sample_size"] < 5)
    print(f"       月度统计：{json.dumps(summary['monthly'], ensure_ascii=False)}")
    print(f"       年度统计：{json.dumps(summary['annual'], ensure_ascii=False)}")


def check_keywords() -> None:
    print()
    print("=" * 74)
    print("[4] 求职方向 -> 检索关键字")
    print("=" * 74)
    cases = [
        ("AI Agent 开发工程师", ["ai", "agent"]),
        ("Python后端开发", ["python"]),
        ("大模型应用开发", ["大模型"]),
        ("数据分析师", ["数据"]),
    ]
    for role, expected_any in cases:
        words = derive_keywords(role)
        check(
            f"{role!r}",
            all(any(exp in w for w in words) for exp in expected_any),
            str(words),
        )
    words = derive_keywords("AI Agent 开发工程师", ["Python", "FastAPI"])
    check("技能会被并入关键字", "python" in words and "fastapi" in words, str(words))
    check("去掉了「工程师/开发」这类噪声词", "工程师" not in words, str(words))


def check_llm() -> None:
    print()
    print("=" * 74)
    print("[5] 大模型配置")
    print("=" * 74)
    providers = provider_list()
    check("厂商预设齐全", len(providers) >= 4, "、".join(p["key"] for p in providers))
    for required in ("deepseek", "kimi", "dashscope", "openai_compatible"):
        check(f"包含 {required}", any(p["key"] == required for p in providers))

    config = get_llm_config()
    check("配置可读取", config is not None)
    check("不泄露明文 Key", "api_key" not in config.to_public_dict())
    print(f"       厂商={config.provider} 模型={config.resolved_model()} "
          f"已配置={config.configured} 掩码={config.masked_key()!r}")


if __name__ == "__main__":
    print()
    for key, value in describe_paths().items():
        print(f"  {key:16} {value}")
    print()
    check_salary()
    check_keywords()
    check_llm()
    print()
    print("=" * 74)
    print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for item in FAILED:
            print(f"    - {item}")
    print("=" * 74)
    sys.exit(1 if FAILED else 0)
