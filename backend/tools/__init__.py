"""Function Calling 工具集。

默认注册三个工具：

===============  ================================================
工具名            作用
===============  ================================================
query_salary     薪资查询：外部薪酬 API 优先，回退本地样本库
search_jobs      岗位搜索：按关键字匹配**职位名称**，抓取真实公开招聘接口
                 （Jobicy / Remotive / RemoteOK / Arbeitnow，免 Key），
                 本地示例库仅作离线兜底
search_knowledge_base  知识库混合检索（把 RAG 也暴露为工具）
===============  ================================================

``execute_tool_calls`` 会把模型一轮返回的多个 tool_call **并行执行**，
再把结果按 ``tool_call_id`` 一一配对回传，这是支撑低延迟多工具调用的关键。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..ai_client import ToolCallRequest
from . import job_search, kb_search, salary
from .registry import ToolExecution, ToolRegistry, ToolSpec

logger = logging.getLogger("ai_resume_helper.tools")

__all__ = [
    "ToolExecution",
    "ToolRegistry",
    "ToolSpec",
    "build_default_registry",
    "get_tool_registry",
    "reset_tool_registry",
    "execute_tool_calls",
    "tool_schemas",
]


def build_default_registry() -> ToolRegistry:
    """构建包含全部内置工具的注册表。"""
    registry = ToolRegistry()
    salary.register(registry)
    job_search.register(registry)
    kb_search.register(registry)
    return registry


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def reset_tool_registry() -> None:
    global _registry
    _registry = None


def tool_schemas(strict: bool = False, names: Sequence[str] | None = None) -> list[dict[str, Any]]:
    return get_tool_registry().schemas(names=names, strict=strict)


async def execute_tool_calls(
    calls: Sequence[ToolCallRequest],
    timeout: float | None = None,
) -> list[tuple[ToolCallRequest, ToolExecution]]:
    """并行执行模型发起的工具调用，保持与入参一致的顺序。"""
    if not calls:
        return []

    from config import get_settings

    settings = get_settings()
    timeout = timeout or settings.tool_timeout
    registry = get_tool_registry()

    tasks = [
        registry.execute(call.name, call.parsed_arguments(), timeout=timeout) for call in calls
    ]
    import asyncio

    executions = await asyncio.gather(*tasks)
    return list(zip(calls, executions))
