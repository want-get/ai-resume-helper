"""岗位搜索工具（Function Calling）。

数据来源（按优先级）：
    1. 用户自建数据源 ``JOB_API_BASE``（如自购/自建的招聘 API）
    2. **真实公开招聘接口**：Jobicy、Remotive、RemoteOK、Arbeitnow（免 Key）
    3. 本地示例库（仅离线兜底，会在结果里明确标注「非实时」）

匹配规则：**关键字只匹配职位名称**；多个关键字是「或」的关系。
调用方（大模型或用户）给出的关键字可以是中文，英文数据源会自动做同义词扩展。

主页说明：所有免费公开接口都不提供中文岗位，因此中文关键字主要靠本地兜底；
真实中文岗位请通过 ``JOB_API_BASE`` 接入自己的数据源。
"""

from __future__ import annotations

import logging
from typing import Any

from config import get_settings

from .job_sources import (
    SOURCE_LABELS,
    available_sources,
    expand_keywords,
    fetch_jobs,
    split_keywords,
)
from .registry import ToolRegistry, ToolSpec

logger = logging.getLogger("ai_resume_helper.tools.jobs")

JOB_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "string",
            "description": (
                "搜索关键字，只匹配职位名称。多个关键字用空格或逗号分隔，之间是「或」的关系。"
                "例如：python后端、AI、Agent。可以用中文，系统会自动扩展英文同义词。"
            ),
        },
        "city": {
            "type": "string",
            "description": "城市/地区过滤（对本地数据源生效，公开远程数据源通常没有城市概念）",
        },
        "sources": {
            "type": "string",
            "description": (
                "数据源，逗号分隔。可选：jobicy, remotive, remoteok, arbeitnow, local, custom。"
                "留空表示用全部可用数据源。local 是本地示例库（非实时），一般不要单独使用。"
            ),
        },
        "limit": {"type": "integer", "description": "返回条数，默认 8，最大 30"},
        "with_description": {
            "type": "boolean",
            "description": "是否返回职位描述正文（默认 true）；只要标题列表时可设 false 以节省 token",
        },
    },
    "required": ["keywords"],
    "additionalProperties": False,
}


async def search_jobs(
    keywords: str,
    city: str | None = None,
    sources: str | None = None,
    limit: int = 8,
    with_description: bool = True,
) -> dict[str, Any]:
    """按关键字检索真实在招岗位（只匹配职位名称）。"""
    settings = get_settings()
    limit = max(1, min(int(limit or 8), 30))

    keyword_list = split_keywords(keywords)
    if not keyword_list:
        return {
            "found": False,
            "error": "keywords 不能为空，请给出至少一个关键字，例如：python后端、AI、Agent",
            "source": "岗位聚合检索",
        }

    requested = split_keywords(sources) if sources else None
    jobs, outcomes, match_terms, expansion = await fetch_jobs(
        keyword_list,
        sources=requested,
        limit_per_source=100,
        total_limit=limit * 3 if city else limit,
    )

    # 城市过滤只对能提供城市的来源有意义（本地库/自建源）
    if city:
        filtered = [job for job in jobs if city.lower() in (job.location or "").lower()]
        if filtered:
            jobs = filtered
        elif any(job.source == "本地示例库" for job in jobs):
            pass  # 保留原结果，在 notes 里说明

    jobs = jobs[:limit]

    notes: list[str] = []
    if city and jobs and not any(city.lower() in (j.location or "").lower() for j in jobs):
        notes.append(f"公开数据源多为远程岗位，没有按「{city}」过滤出结果，已返回全部命中项")
    if keyword_list and all(any("\u4e00" <= ch <= "\u9fff" for ch in kw) for kw in keyword_list):
        notes.append(
            "免费公开接口没有中文岗位库，中文关键字主要靠同义词映射与本地兜底；"
            "如需真实中文岗位，请配置 JOB_API_BASE 接入自己的数据源"
        )
    failed = [o.source for o in outcomes if not o.ok]
    if failed:
        notes.append(f"以下数据源本次不可用（已跳过）：{', '.join(failed)}")

    return {
        "found": bool(jobs),
        "query": {"keywords": keyword_list, "city": city, "limit": limit},
        "keyword_expansion": expansion,
        "match_terms_used": match_terms,
        "returned": len(jobs),
        "jobs": [job.to_dict(description_chars=500 if with_description else 0) for job in jobs],
        "sources": outcomes_dict(outcomes),
        "data_origin": "public_api" if any(o.source != "local" and o.ok for o in outcomes) else "local_dataset",
        "source": "岗位聚合检索（" + "、".join(SOURCE_LABELS.get(o.source, o.source) for o in outcomes if o.ok) + "）",
        "notes": notes,
        "hint": (
            "结果里每条岗位都带 url 与 description；"
            "如果用户想针对某个岗位准备面试，可以把该岗位的 title + description 作为目标岗位。"
        ),
    }


def outcomes_dict(outcomes: Any) -> list[dict[str, Any]]:
    return [outcome.to_dict() for outcome in outcomes]


async def describe_sources() -> dict[str, Any]:
    """给前端/接口用的数据源清单（说明每个源到底是什么）。"""
    settings = get_settings()
    return {
        "available": [
            {"key": name, "label": SOURCE_LABELS.get(name, name)} for name in available_sources()
        ],
        "default_order": available_sources(),
        "custom_api_configured": bool(settings.job_api_base),
        "cache_ttl_seconds": settings.job_cache_ttl,
        "note": (
            "内置数据源均为免 Key 的公开招聘接口，实时但以英文远程岗位为主；"
            "本地示例库只用于离线兜底，不是实时数据。"
        ),
    }


def register(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="search_jobs",
            description=(
                "按关键字检索真实在招岗位。关键字只匹配「职位名称」，多个关键字用空格/逗号分隔（或的关系），"
                "例如 python后端、AI、Agent。数据来自公开招聘接口，返回职位名、公司、地点、薪资、标签、"
                "原文链接与职位描述。用户问「有哪些岗位」「招不招 XX」「岗位要求是什么」时必须调用，"
                "不得凭记忆回答。"
            ),
            parameters=JOB_PARAMETERS,
            handler=search_jobs,
            tags=("job", "realtime"),
        )
    )
