"""异步 DeepSeek 客户端。

* 基于 ``AsyncOpenAI`` + ``httpx``，全程异步，不阻塞事件循环。
* 内置 ``asyncio.Semaphore`` 并发闸门：限制同时在飞的大模型请求数，
  防止上游限流击穿后端的 100+ 并发能力。
* 支持 Function Calling（含并行工具调用）与 ``strict`` 严格 Schema 模式。
* 支持 **离线 Mock 模式**：未配置 API Key 时（或显式开启）用确定性桩响应
  跑通全链路，便于本地自测与并发压测，不会伪造真实模型能力。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from config import Settings, get_settings

from .llm_providers import DEFAULT_PROVIDER

logger = logging.getLogger("ai_resume_helper.ai")


class AIClientError(RuntimeError):
    """大模型调用异常基类。"""


class AIConfigError(AIClientError):
    """未配置 API Key。"""


class AIUpstreamError(AIClientError):
    """上游返回错误 / 超时。"""


@dataclass(slots=True)
class ToolCallRequest:
    """归一化后的工具调用请求（真实响应与 Mock 响应共用同一结构）。"""

    id: str
    name: str
    arguments: str  # JSON 字符串

    def parsed_arguments(self) -> dict[str, Any]:
        if not self.arguments:
            return {}
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def to_message_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.parsed_arguments()}


@dataclass(slots=True)
class LLMResult:
    """一次大模型调用的结果。"""

    content: str
    model: str = ""
    finish_reason: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    mock: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_ms": round(self.latency_ms, 2),
            "mock": self.mock,
            "error": self.error,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


# ---------------------------------------------------------------------- #
# Mock 桩响应
# ---------------------------------------------------------------------- #
_MOCK_TOOL_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("薪资", "工资", "薪酬", "salary", "多少钱", "月薪"), "query_salary"),
    (("岗位", "职位", "招聘", "jd", "在招", "job"), "search_jobs"),
    (("面试题", "题库", "知识库", "考察点"), "search_knowledge_base"),
)


class _MockFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _MockToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.id = f"mock-{uuid.uuid4().hex[:12]}"
        self.function = _MockFunction(name, arguments)


class _MockMessage:
    def __init__(self, content: str | None, tool_calls: list[_MockToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _MockChoice:
    def __init__(self, message: _MockMessage) -> None:
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class _MockResponse:
    def __init__(self, message: _MockMessage) -> None:
        self.choices = [_MockChoice(message)]
        self.model = "mock-deepseek-chat"


# ---------------------------------------------------------------------- #
# 客户端
# ---------------------------------------------------------------------- #
class AsyncAIClient:
    """异步大模型客户端。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._clients: dict[str, Any] = {}
        self._semaphore = asyncio.Semaphore(max(1, self.settings.max_concurrent_llm))
        self._lock = asyncio.Lock()
        self.stats: dict[str, Any] = {
            "calls": 0,
            "errors": 0,
            "mock_calls": 0,
            "in_flight": 0,
            "peak_in_flight": 0,
            "total_latency_ms": 0.0,
        }

    # ------------------------------------------------------------------ #
    # 连接参数统一取自 ``llm_config``（用户在界面上保存的那份配置）
    #
    # ⚠️ 这里曾经踩过一个很隐蔽的坑：本类原先读的是 ``self.settings.deepseek_*``
    # （即 .env / 环境变量），而界面上保存的 Key 写的是 ``settings.json``
    # （由 ``llm_config`` 管理）。两处各自为政，导致：
    #   * 源码里跑（有 .env）一切正常
    #   * **打包成 exe 后**（没有 .env）用户在界面上填了 Key，
    #     「测试连接」显示成功，但真正出题/优化时仍然走 Mock 桩，
    #     返回「未配置 DEEPSEEK_API_KEY」——用户看到的是假结果。
    # 现在统一以 ``get_llm_config()`` 为准，环境变量只作为它的默认值来源。
    # ------------------------------------------------------------------ #
    @property
    def llm_config(self) -> Any:
        from .llm_config import get_llm_config

        return get_llm_config()

    @property
    def configured(self) -> bool:
        return self.llm_config.configured

    @property
    def mock_mode(self) -> bool:
        return self.settings.llm_mock_mode or not self.configured

    async def _get_client(self, beta: bool = False) -> Any:
        """按 (base_url, api_key) 缓存客户端（strict 模式走 DeepSeek beta 端点）。

        缓存键必须带上 api_key：用户换 Key 后如果仍复用旧客户端，会继续用旧 Key 请求。
        """
        config = self.llm_config
        base_url = config.resolved_base_url()
        if beta and config.provider == DEFAULT_PROVIDER:
            beta_url = (self.settings.deepseek_beta_base_url or "").strip()
            if beta_url:
                base_url = beta_url.rstrip("/")

        cache_key = (base_url, config.api_key)
        client = self._clients.get(cache_key)
        if client is None:
            async with self._lock:
                client = self._clients.get(cache_key)
                if client is None:
                    from openai import AsyncOpenAI

                    client = AsyncOpenAI(
                        api_key=config.api_key,
                        base_url=base_url,
                        timeout=config.timeout,
                        max_retries=0,  # 重试逻辑自己控制
                    )
                    self._clients[cache_key] = client
        return client

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:  # pragma: no cover - 关闭失败不影响退出
                pass
        self._clients.clear()

    # ------------------------------------------------------------------ #
    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        strict: bool = False,
    ) -> LLMResult:
        """发起一次对话补全（可带工具）。"""
        config = self.llm_config
        temperature = config.temperature if temperature is None else temperature
        max_tokens = max_tokens or config.max_tokens
        model = model or config.resolved_model()
        started = time.perf_counter()

        if self.mock_mode:
            await asyncio.sleep(0)  # 让出控制权，模拟异步
            result = self._mock_completion(messages, tools, temperature)
            self._record(result, started)
            return result

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        async with self._semaphore:
            self.stats["in_flight"] += 1
            self.stats["peak_in_flight"] = max(
                self.stats["peak_in_flight"], self.stats["in_flight"]
            )
            try:
                return await self._call_with_retry(strict, kwargs, model, started)
            finally:
                self.stats["in_flight"] -= 1
                self.stats["calls"] += 1

    async def _call_with_retry(
        self, beta: bool, kwargs: dict[str, Any], model: str, started: float
    ) -> LLMResult:
        config = self.llm_config
        client = await self._get_client(beta=beta)

        last_error: str = ""
        for attempt in range(config.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs), timeout=config.timeout
                )
                message = response.choices[0].message
                tool_calls: list[ToolCallRequest] = []
                for call in getattr(message, "tool_calls", None) or []:
                    tool_calls.append(
                        ToolCallRequest(
                            id=call.id,
                            name=call.function.name,
                            arguments=call.function.arguments or "{}",
                        )
                    )
                usage = {}
                if getattr(response, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }
                result = LLMResult(
                    content=(message.content or "").strip(),
                    model=getattr(response, "model", model),
                    finish_reason=response.choices[0].finish_reason or "",
                    tool_calls=tool_calls,
                    usage=usage,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                self._record(result, started)
                return result
            except asyncio.TimeoutError:
                last_error = f"调用超时（>{settings.llm_timeout}s）"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                message = str(exc)
                # 4xx（除 429）不重试，重试也没用
                if any(code in message for code in ("401", "403", "400", "404")):
                    break
            if attempt < settings.llm_max_retries:
                await asyncio.sleep(0.6 * (2**attempt))

        self.stats["errors"] += 1
        result = LLMResult(
            content="",
            model=model,
            error=last_error,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return result

    # ------------------------------------------------------------------ #
    def _record(self, result: LLMResult, started: float) -> None:
        self.stats["total_latency_ms"] += (time.perf_counter() - started) * 1000
        if result.mock:
            self.stats["mock_calls"] += 1

    # ------------------------------------------------------------------ #
    # Mock 实现
    # ------------------------------------------------------------------ #
    def _mock_completion(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
        temperature: float,
    ) -> LLMResult:
        """确定性的桩响应：优先演示工具调用，其次基于上下文作答。"""
        tool_names = {
            (tool.get("function") or {}).get("name", "") for tool in (tools or [])
        }
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content") or "")
                break

        already_called = {
            message.get("name")
            for message in messages
            if message.get("role") == "tool"
        }
        tool_results = [m for m in messages if m.get("role") == "tool"]

        # 第一轮：判断是否需要调用工具
        if tools and not tool_results:
            lowered = last_user.lower()
            for keywords, name in _MOCK_TOOL_HINTS:
                if name not in tool_names or name in already_called:
                    continue
                if any(keyword in lowered for keyword in keywords):
                    arguments = self._mock_arguments(name, last_user)
                    call = _MockToolCall(name, json.dumps(arguments, ensure_ascii=False))
                    return LLMResult(
                        content="",
                        model="mock-deepseek-chat",
                        finish_reason="tool_calls",
                        tool_calls=[
                            ToolCallRequest(
                                id=call.id, name=name, arguments=call.function.arguments
                            )
                        ],
                        mock=True,
                    )

        # 已有工具结果 → 汇总
        if tool_results:
            lines = ["根据工具返回的实时数据："]
            for message in tool_results:
                lines.append(f"- {message.get('content', '')}")
            lines.append("（以上为工具返回结果，未做额外推测。）")
            return LLMResult(
                content="\n".join(lines), model="mock-deepseek-chat", mock=True
            )

        # 无工具：基于 system 中的参考资料作答
        system_text = "\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        context = ""
        marker = "【参考资料】"
        if marker in system_text:
            context = system_text.split(marker, 1)[1].split("【输出要求】", 1)[0].strip()
        elif "【工具返回数据】" in system_text:
            context = system_text.split("【工具返回数据】", 1)[1].strip()

        if context:
            body = context[:1200]
            return LLMResult(
                content=(
                    "【Mock 模式回答】未配置 DEEPSEEK_API_KEY，以下内容直接摘自检索到的资料：\n\n"
                    f"{body}\n\n"
                    "（引用编号与来源见响应中的 sources 字段。）"
                ),
                model="mock-deepseek-chat",
                mock=True,
            )

        return LLMResult(
            content=(
                "【Mock 模式回答】未配置 DEEPSEEK_API_KEY，当前为离线桩响应。"
                f"收到的问题：{last_user[:200]}"
            ),
            model="mock-deepseek-chat",
            mock=True,
        )

    @staticmethod
    def _mock_arguments(name: str, question: str) -> dict[str, Any]:
        if name == "query_salary":
            return {"role": "Python后端开发工程师", "city": "杭州", "level": "中级"}
        if name == "search_jobs":
            # 从提问里挑关键字，避免桩响应看起来像"写死只搜 Python 后端"
            import re

            tokens = [
                token
                for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{1,20}", question)
                if token.lower() not in {"the", "and", "for", "job", "jobs"}
            ]
            chinese = re.findall(r"[\u4e00-\u9fff]{2,8}(?:岗位|工程师|开发|运营|产品|算法|测试)", question)
            keywords = " ".join(dict.fromkeys(tokens[:4] + chinese[:3])) or question[:30]
            return {"keywords": keywords, "limit": 5}
        if name == "search_knowledge_base":
            return {"query": question[:60], "top_k": 3}
        return {}

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        calls = max(1, self.stats["calls"])
        return {
            "configured": self.configured,
            "mock_mode": self.mock_mode,
            "model": self.llm_config.resolved_model(),
            **self.stats,
            "avg_latency_ms": round(self.stats["total_latency_ms"] / calls, 2),
            "concurrency_limit": self.settings.max_concurrent_llm,
        }


_ai_client: AsyncAIClient | None = None


def get_ai_client() -> AsyncAIClient:
    """进程内单例。"""
    global _ai_client
    if _ai_client is None:
        _ai_client = AsyncAIClient()
    return _ai_client


async def reset_ai_client() -> None:
    global _ai_client
    if _ai_client is not None:
        await _ai_client.close()
    _ai_client = None
