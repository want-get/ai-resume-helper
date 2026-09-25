"""Function Calling 工具注册表。

一个「工具」需要三部分：
1. JSON Schema 描述（给模型看，决定何时调用、参数怎么填）
2. 异步处理函数（真正干活的代码）
3. 执行结果包装（成功/失败、耗时、原始参数），统一回传给模型

注册表负责：
* 生成 ``tools=[...]`` 参数（可选 ``strict`` 严格模式）
* 校验模型给的函数名与参数
* 带超时地执行，**任何异常都转成可读文本回传**，绝不把异常抛进对话流程——
  这样模型能基于「工具失败」这个事实给出诚实的回答，而不是编造数据。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

logger = logging.getLogger("ai_resume_helper.tools")

ToolHandler = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    """单个工具的完整定义。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    strict_safe: bool = True  # 参数结构是否满足 OpenAI strict 模式要求
    tags: tuple[str, ...] = ()

    def schema(self, strict: bool = False) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if strict and self.strict_safe:
            function["strict"] = True
        return {"type": "function", "function": function}


@dataclass(slots=True)
class ToolExecution:
    """一次工具调用的结果。"""

    name: str
    ok: bool
    result: Any = None
    error: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    source: str = ""

    def to_content(self) -> str:
        """转成回传给模型的 ``tool`` 消息内容。"""
        import json

        if not self.ok:
            return json.dumps(
                {"ok": False, "tool": self.name, "error": self.error}, ensure_ascii=False
            )
        return json.dumps({"ok": True, "tool": self.name, "data": self.result}, ensure_ascii=False, default=str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "source": self.source,
        }


class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------ #
    def register(self, spec: ToolSpec, override: bool = False) -> None:
        if spec.name in self._tools and not override:
            raise ValueError(f"工具 {spec.name} 已注册")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, names: Sequence[str] | None = None, strict: bool = False) -> list[dict[str, Any]]:
        selected = self._tools.values() if names is None else [
            self._tools[n] for n in names if n in self._tools
        ]
        return [spec.schema(strict=strict) for spec in selected]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "tags": list(spec.tags),
                "parameters": spec.parameters,
            }
            for spec in self._tools.values()
        ]

    # ------------------------------------------------------------------ #
    async def execute(
        self, name: str, arguments: dict[str, Any], timeout: float = 10.0
    ) -> ToolExecution:
        started = time.perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            return ToolExecution(
                name=name,
                ok=False,
                error=f"未知工具：{name}",
                arguments=arguments,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            result = await asyncio.wait_for(spec.handler(**arguments), timeout=timeout)
            execution = ToolExecution(
                name=name,
                ok=True,
                result=result,
                arguments=arguments,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except asyncio.TimeoutError:
            execution = ToolExecution(
                name=name,
                ok=False,
                error=f"工具执行超时（>{timeout}s）",
                arguments=arguments,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except TypeError as exc:
            execution = ToolExecution(
                name=name,
                ok=False,
                error=f"参数不合法：{exc}",
                arguments=arguments,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - 工具异常必须被吞掉并如实上报
            logger.exception("工具 %s 执行失败", name)
            execution = ToolExecution(
                name=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                arguments=arguments,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        if isinstance(execution.result, dict):
            execution.source = str(execution.result.get("source", ""))
        return execution

    async def execute_many(
        self,
        calls: Sequence[tuple[str, str, dict[str, Any]]],
        timeout: float = 10.0,
    ) -> list[ToolExecution]:
        """并行执行多个工具调用（对应模型的并行 tool_calls）。"""
        tasks = [self.execute(name, args, timeout) for _, name, args in calls]
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))
