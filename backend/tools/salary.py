"""薪资查询工具（Function Calling）。

数据源优先级：
1. 配置了 ``SALARY_API_BASE`` → 调用**外部真实薪酬 API**（httpx 异步请求）
2. 未配置或调用失败 → 回退到本地 MySQL/SQLite 的 ``salary_records`` 表

无论走哪条路，返回值都会带上 ``source`` 与统计口径（样本量、更新时间），
让模型只能引用真实存在的数据。
"""

from __future__ import annotations

import logging
from typing import Any

from config import get_settings

from ..db import repository, session_scope
from .registry import ToolExecution, ToolRegistry, ToolSpec

logger = logging.getLogger("ai_resume_helper.tools.salary")

SALARY_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {
            "type": "string",
            "description": "岗位名称，例如：Python后端开发工程师、产品经理、算法工程师",
        },
        "city": {
            "type": "string",
            "description": "城市，例如：北京、上海、深圳、杭州、成都。不确定可不填",
        },
        "level": {
            "type": "string",
            "enum": ["初级", "中级", "高级"],
            "description": "职级：初级约 0-2 年，中级约 3-5 年，高级约 5-8 年",
        },
        "years": {
            "type": "number",
            "description": "工作年限，例如 3.5。用于在未指定职级时推断职级",
        },
    },
    "required": ["role"],
    "additionalProperties": False,
}


def _level_from_years(years: float | None) -> str | None:
    if years is None:
        return None
    if years < 3:
        return "初级"
    if years < 5:
        return "中级"
    return "高级"


def _role_similarity(query: str, candidate: str) -> float:
    """字符 bigram 的 Dice 相似度，用于把「Python后端」匹配到「Python后端开发工程师」。"""
    query = "".join(query.split()).lower()
    candidate = "".join(candidate.split()).lower()
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    if query in candidate or candidate in query:
        return 0.9

    def bigrams(text: str) -> set[str]:
        if len(text) < 2:
            return {text}
        return {text[i : i + 2] for i in range(len(text) - 1)}

    left, right = bigrams(query), bigrams(candidate)
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def _to_dict(record: Any) -> dict[str, Any]:
    return repository.salary_to_dict(record)


async def _query_external(role: str, city: str | None, level: str | None) -> dict[str, Any] | None:
    """调用外部薪酬 API。未配置或失败时返回 None。"""
    settings = get_settings()
    if not settings.salary_api_base:
        return None
    try:
        import httpx

        headers = {"Accept": "application/json"}
        if settings.salary_api_key:
            headers["Authorization"] = f"Bearer {settings.salary_api_key}"
        params = {"role": role}
        if city:
            params["city"] = city
        if level:
            params["level"] = level

        async with httpx.AsyncClient(timeout=settings.tool_timeout) as client:
            response = await client.get(
                settings.salary_api_base.rstrip("/") + "/salary", params=params, headers=headers
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload.setdefault("source", f"外部薪酬API（{settings.salary_api_base}）")
                payload["data_origin"] = "external_api"
                return payload
    except Exception as exc:  # noqa: BLE001 - 外部接口不可靠，失败就回退本地
        logger.warning("外部薪资 API 调用失败，回退本地数据：%s", exc)
    return None


async def query_salary(
    role: str,
    city: str | None = None,
    level: str | None = None,
    years: float | None = None,
) -> dict[str, Any]:
    """查询岗位薪资分位数据。"""
    role = (role or "").strip()
    if not role:
        return {"found": False, "error": "role 不能为空", "source": "薪资查询工具"}

    level = level or _level_from_years(years)

    external = await _query_external(role, city, level)
    if external is not None:
        return external

    async with session_scope() as db:
        records = await repository.list_salary_records(db, city=city, level=level)
        if not records:
            records = await repository.list_salary_records(db, city=city)
        if not records:
            records = await repository.list_salary_records(db)
        rows = [_to_dict(record) for record in records]

    if not rows:
        return {
            "found": False,
            "error": "薪资库暂无数据",
            "source": "本地薪酬样本库",
            "hint": "可运行 python scripts/init_db.py 导入种子数据",
        }

    scored = sorted(
        ((_role_similarity(role, row["role"]), row) for row in rows),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best = scored[0]

    if best_score < 0.3:
        candidates = sorted({row["role"] for _, row in scored[:8]})
        return {
            "found": False,
            "query_role": role,
            "city": city,
            "level": level,
            "error": f"薪资库中没有与「{role}」匹配的岗位",
            "available_roles": candidates,
            "source": "本地薪酬样本库",
        }

    alternatives = [
        {
            "role": row["role"],
            "city": row["city"],
            "level": row["level"],
            "p50": row["p50"],
            "unit": row["unit"],
        }
        for score, row in scored[1:4]
        if score > 0.5
    ]

    notes: list[str] = []
    if best_score < 0.9:
        notes.append(f"按相似度匹配到最接近的岗位：{best['role']}")
    if city and city not in best["city"]:
        notes.append(f"未找到 {city} 的数据，返回的是 {best['city']} 的样本")
    if level and level != best["level"]:
        notes.append(f"未找到{level}数据，返回的是{best['level']}的样本")

    return {
        "found": True,
        "query_role": role,
        "matched_role": best["role"],
        "match_score": round(best_score, 3),
        "city": best["city"],
        "level": best["level"],
        "years": f"{int(best['years_min'])}-{int(best['years_max'])}年",
        "percentiles": {
            "p25": best["p25"],
            "p50": best["p50"],
            "p75": best["p75"],
            "p90": best["p90"],
        },
        "unit": best["unit"],
        "currency": best["currency"],
        "sample_size": best["sample_size"],
        "updated_at": best["updated_at"],
        "source": "本地薪酬样本库（示例数据，非实时行情）",
        "data_origin": "local_dataset",
        "notes": notes,
        "alternatives": alternatives,
    }


async def query_salary_execution(
    role: str, city: str | None = None, level: str | None = None, years: float | None = None
) -> ToolExecution:
    """便捷函数：直接拿到可展示的 ToolExecution 包装。"""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="query_salary",
            description="查询岗位薪资分位数据",
            parameters=SALARY_PARAMETERS,
            handler=query_salary,
        )
    )
    return await registry.execute(
        "query_salary", {"role": role, "city": city, "level": level, "years": years}
    )


def register(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="query_salary",
            description=(
                "查询指定岗位在指定城市的薪资分位数据（P25/P50/P75/P90、样本量、统计口径）。"
                "当用户询问薪资、工资、薪酬、多少钱、涨薪空间时必须调用此工具，不要凭记忆回答。"
            ),
            parameters=SALARY_PARAMETERS,
            handler=query_salary,
            tags=("salary", "realtime"),
        )
    )
