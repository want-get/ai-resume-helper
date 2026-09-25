"""薪资文本解析与市场统计。

招聘网站上的薪资写法极其杂乱，必须先归一化才能算平均值：

    25-40K·14薪       → 月薪 25~40K，发 14 个月
    20-35k            → 月薪 20~35K
    2-3万/月           → 月薪 20~30K
    20-40万/年         → 月薪 16.7~33.3K（按 12 个月折算）
    8000-12000元/月   → 月薪 8~12K
    15k-25k·13薪      → 月薪 15~25K，发 13 个月
    $120,000-$180,000 a year  → 美元年薪，单独统计不与人民币混算
    薪资面议 / Negotiable     → 无法解析，样本丢弃（但计数保留）

统一口径：**月薪，单位 K（千元）**；年薪 = 月薪中位数 × 发薪月数。
解析不出来的样本不会瞎猜，直接标记为 unknown 并从统计中剔除。
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# 明确表示「没有薪资信息」的写法
_UNKNOWN_PATTERNS = (
    "面议", "面谈", "待定", "不限", "negotiable", "competitive", "doe",
    "薪资面议", "待遇面议", "详谈", "see description",
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_MONTHS_RE = re.compile(r"(\d{1,2})\s*薪")
_USD_HINT_RE = re.compile(r"usd|\$|dollar|美元", re.I)
_ANNUAL_HINT_RE = re.compile(r"/\s*(?:年|yr|year|annum|annually)|年薪|per\s+year|a\s+year", re.I)
_MONTHLY_HINT_RE = re.compile(r"/\s*(?:月|mo|month)|月薪|per\s+month|a\s+month", re.I)
_HOURLY_HINT_RE = re.compile(r"/\s*(?:时|小时|hr|hour)|时薪|per\s+hour", re.I)
_DAY_HINT_RE = re.compile(r"/\s*(?:天|日|day)|日薪|per\s+day", re.I)
_WAN_RE = re.compile(r"万|w\b", re.I)
_K_RE = re.compile(r"k\b|千")


def _normalize(text: str) -> str:
    """全角转半角、去空白、统一分隔符。"""
    out: list[str] = []
    for ch in text or "":
        code = ord(ch)
        if code == 0x3000:
            ch = " "
        elif 0xFF01 <= code <= 0xFF5E:
            ch = chr(code - 0xFEE0)
        out.append(ch)
    normalized = "".join(out).lower()
    normalized = normalized.replace("—", "-").replace("–", "-").replace("~", "-").replace("至", "-")
    return re.sub(r"\s+", "", normalized)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


@dataclass(slots=True)
class SalaryRange:
    """归一化后的薪资区间（月薪，单位 K）。"""

    low_k: float
    high_k: float
    months: int = 12
    currency: str = "CNY"
    raw: str = ""

    @property
    def mid_k(self) -> float:
        return (self.low_k + self.high_k) / 2

    @property
    def annual_k(self) -> float:
        return self.mid_k * self.months

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_k": round(self.low_k, 1),
            "high_k": round(self.high_k, 1),
            "mid_k": round(self.mid_k, 1),
            "months": self.months,
            "annual_k": round(self.annual_k, 1),
            "currency": self.currency,
            "raw": self.raw,
        }


def parse_salary(text: str | None) -> SalaryRange | None:
    """把一条薪资文本解析成统一口径；解析不出来返回 None。"""
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    normalized = _normalize(raw)
    if not normalized or any(pattern in normalized for pattern in _UNKNOWN_PATTERNS):
        return None

    currency = "USD" if _USD_HINT_RE.search(normalized) else "CNY"

    # 发薪月数：14薪 / *14薪 / 13薪
    months = 12
    months_match = _MONTHS_RE.search(normalized)
    if months_match:
        try:
            months = int(months_match.group(1))
        except ValueError:
            months = 12
        if not 12 <= months <= 24:
            months = 12
        normalized = normalized[: months_match.start()] + normalized[months_match.end() :]

    numbers = [_to_float(m) for m in _NUMBER_RE.findall(normalized)]
    if not numbers:
        return None
    if len(numbers) == 1:
        low = high = numbers[0]
    else:
        low, high = numbers[0], numbers[1]
        if low > high:
            low, high = high, low

    annual = bool(_ANNUAL_HINT_RE.search(normalized))
    monthly = bool(_MONTHLY_HINT_RE.search(normalized))
    hourly = bool(_HOURLY_HINT_RE.search(normalized))
    daily = bool(_DAY_HINT_RE.search(normalized))
    is_wan = bool(_WAN_RE.search(normalized))
    is_k = bool(_K_RE.search(normalized))

    # ---- 折算成「月薪 K」----
    if currency == "USD":
        # 英文岗位常见：$120,000 - $180,000 a year / $50 an hour
        if hourly:
            low_k, high_k = low * 160 / 1000, high * 160 / 1000
        elif daily:
            low_k, high_k = low * 21 / 1000, high * 21 / 1000
        elif annual or not monthly:
            low_k, high_k = low / 12 / 1000, high / 12 / 1000
        else:
            low_k, high_k = low / 1000, high / 1000
    elif is_wan:
        # 「万」：显式 /年 或数值较大 → 年薪；否则按 万/月
        if annual or (not monthly and high > 10):
            low_k, high_k = low * 10 / 12, high * 10 / 12
        else:
            low_k, high_k = low * 10, high * 10
    elif is_k:
        low_k, high_k = low, high
    elif hourly:
        low_k, high_k = low * 160 / 1000, high * 160 / 1000
    elif daily:
        low_k, high_k = low * 21 / 1000, high * 21 / 1000
    elif annual:
        low_k, high_k = low / 12 / 1000, high / 12 / 1000
    elif high >= 1000:
        # 「8000-12000」这种不带单位的写法，按 元/月
        low_k, high_k = low / 1000, high / 1000
    else:
        # 极简写法（如 "25-40"），按 K/月
        low_k, high_k = low, high

    # 合理性过滤：月薪 0.5K ~ 2000K 之外的当噪声丢掉
    if not (0.5 <= low_k <= 2000) or not (0.5 <= high_k <= 2000) or high_k < low_k:
        return None

    return SalaryRange(
        low_k=round(low_k, 2),
        high_k=round(high_k, 2),
        months=months,
        currency=currency,
        raw=raw,
    )


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def summarize_salaries(
    ranges: Iterable[SalaryRange],
    expected_min_k: float | None = None,
    expected_max_k: float | None = None,
) -> dict[str, Any]:
    """对一批薪资区间做统计，并与用户期望对比。

    统计口径：以每条岗位薪资区间的**中位数**作为该岗位的代表值，
    这样「20-40K」和「25-30K」不会因为区间宽窄被不公平地加权。
    """
    items = [item for item in ranges if item is not None]
    cny = [item for item in items if item.currency == "CNY"]
    other = [item for item in items if item.currency != "CNY"]

    if not cny:
        return {
            "sample_size": 0,
            "parsed_size": len(items),
            "currency": "CNY",
            "note": "没有解析到可统计的人民币薪资样本",
            "other_currency_samples": len(other),
        }

    mids = [item.mid_k for item in cny]
    annuals = [item.annual_k for item in cny]
    monthly_equiv = [item.mid_k * item.months / 12 for item in cny]

    result: dict[str, Any] = {
        "sample_size": len(cny),
        "parsed_size": len(items),
        "currency": "CNY",
        "unit": "K/月",
        "monthly": {
            "mean": round(statistics.fmean(mids), 1),
            "median": round(statistics.median(mids), 1),
            "p25": round(_percentile(mids, 0.25), 1),
            "p75": round(_percentile(mids, 0.75), 1),
            "min": round(min(mids), 1),
            "max": round(max(mids), 1),
        },
        "annual_equiv_monthly": {
            "mean": round(statistics.fmean(monthly_equiv), 1),
            "median": round(statistics.median(monthly_equiv), 1),
        },
        "annual": {
            "mean": round(statistics.fmean(annuals), 1),
            "median": round(statistics.median(annuals), 1),
        },
        "other_currency_samples": len(other),
        "low_confidence": len(cny) < 5,
    }

    if expected_min_k or expected_max_k:
        median_mid = statistics.median(mids)
        expected_min = float(expected_min_k or 0)
        expected_max = float(expected_max_k or 0)
        if expected_min and expected_max:
            expected_mid = (expected_min + expected_max) / 2
        else:
            expected_mid = expected_min or expected_max

        gap = median_mid - expected_mid
        gap_ratio = (gap / expected_mid * 100) if expected_mid else 0.0
        if gap_ratio >= 10:
            verdict = "高于期望"
            advice = "市场上这个岗位的薪资普遍高于你的期望，可以适当抬高报价。"
        elif gap_ratio <= -10:
            verdict = "低于期望"
            advice = "市场薪资低于你的期望，建议放宽城市/方向，或补充技能与项目经验后再谈薪。"
        else:
            verdict = "基本吻合"
            advice = "你的期望与市场水平基本吻合，报价时可以按中位数上浮 10%~20% 争取。"

        result["expectation"] = {
            "expected_min_k": expected_min or None,
            "expected_max_k": expected_max or None,
            "expected_mid_k": round(expected_mid, 1),
            "market_median_k": round(median_mid, 1),
            "gap_k": round(gap, 1),
            "gap_ratio": round(gap_ratio, 1),
            "verdict": verdict,
            "advice": advice,
        }
    return result


__all__ = ["SalaryRange", "parse_salary", "summarize_salaries"]
