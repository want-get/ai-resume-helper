"""长上下文机制：滑动窗口 + 滚动摘要。

问题：多轮对话（尤其是模拟面试）很容易超出模型的上下文窗口；
全部塞进去既贵又会稀释注意力，粗暴截断又会「失忆」。

做法（对应项目要求「滑动窗口 + 早期对话摘要」）：

    会话历史:  m1  m2  m3  m4  m5  m6  m7  m8  m9  m10
               └──────── 已摘要 ────────┘ └─ 原文窗口 ─┘
               压缩进 session.summary        最近 N 轮原文

* **滑动窗口**：从最新一条往回取，直到满足 ``memory_window_messages`` 条
  且不超过 ``memory_window_max_chars`` 字符预算。窗口内保留**原文**，
  保证最近的细节（用户刚说的项目名、数字）不失真。
* **滚动摘要**：窗口之外的消息，当累积字符数超过
  ``memory_summary_trigger_chars`` 时，调用模型把「旧摘要 + 新增对话」
  合并成一份新摘要，并记录 ``summary_upto_seq``；已被摘要的消息不再以
  原文发送。摘要是有状态的、增量更新的，不会每次重算全量。
* 最终送给模型的消息 = system + 摘要(system 角色) + 窗口原文。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from config import Settings, get_settings
from prompts.system import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT

from .ai_client import AsyncAIClient

logger = logging.getLogger("ai_resume_helper.memory")


@dataclass(slots=True)
class MemoryStats:
    """长上下文的可观测指标，随响应返回，便于前端展示与调优。"""

    total_messages: int = 0
    window_messages: int = 0
    summarized_messages: int = 0
    summary_chars: int = 0
    window_chars: int = 0
    history_chars: int = 0
    compressed_ratio: float = 0.0
    summary_updated: bool = False
    summary_upto_seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "window_messages": self.window_messages,
            "summarized_messages": self.summarized_messages,
            "summary_chars": self.summary_chars,
            "window_chars": self.window_chars,
            "history_chars": self.history_chars,
            "compressed_ratio": round(self.compressed_ratio, 3),
            "summary_updated": self.summary_updated,
            "summary_upto_seq": self.summary_upto_seq,
        }


@dataclass(slots=True)
class BuiltContext:
    """构建好的上下文。"""

    messages: list[dict[str, Any]]
    summary: str
    summary_upto_seq: int
    stats: MemoryStats
    window: list[dict[str, Any]] = field(default_factory=list)


class ConversationMemory:
    """滑动窗口 + 滚动摘要的上下文构建器。"""

    def __init__(
        self, ai_client: AsyncAIClient | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.ai_client = ai_client

    # ------------------------------------------------------------------ #
    def select_window(self, history: Sequence[dict[str, Any]]) -> int:
        """返回窗口起始下标（history[:index] 属于旧消息）。"""
        cfg = self.settings
        max_messages = max(1, cfg.memory_window_messages)
        max_chars = max(200, cfg.memory_window_max_chars)

        index = len(history)
        chars = 0
        count = 0
        while index > 0:
            candidate = history[index - 1]
            length = len(str(candidate.get("content") or ""))
            if count >= max_messages:
                break
            if count > 0 and chars + length > max_chars:
                break
            chars += length
            count += 1
            index -= 1
        return index

    # ------------------------------------------------------------------ #
    async def build(
        self,
        history: Sequence[dict[str, Any]],
        system_prompt: str,
        summary: str = "",
        summary_upto_seq: int = 0,
    ) -> BuiltContext:
        """构建最终发给模型的消息列表（必要时刷新摘要）。"""
        cfg = self.settings
        stats = MemoryStats(
            total_messages=len(history),
            history_chars=sum(len(str(m.get("content") or "")) for m in history),
            summary_upto_seq=summary_upto_seq,
        )

        if not history:
            stats.summary_chars = len(summary)
            return BuiltContext(
                messages=[{"role": "system", "content": system_prompt}],
                summary=summary,
                summary_upto_seq=summary_upto_seq,
                stats=stats,
            )

        window_start = self.select_window(history)
        window = list(history[window_start:])
        older = list(history[:window_start])
        pending = [m for m in older if int(m.get("seq") or 0) > summary_upto_seq]

        pending_chars = sum(len(str(m.get("content") or "")) for m in pending)
        if (
            cfg.memory_enable_summary
            and pending
            and pending_chars >= cfg.memory_summary_trigger_chars
        ):
            try:
                summary = await self.summarize(summary, pending)
                summary_upto_seq = int(pending[-1].get("seq") or summary_upto_seq)
                stats.summary_updated = True
                # 摘要变短后，窗口可以向后多留一些原文
                if len(summary) < pending_chars * 0.6:
                    window_start = self.select_window(history)
                    window = list(history[window_start:])
            except Exception as exc:  # noqa: BLE001 - 摘要失败不能影响主流程
                logger.warning("滚动摘要失败，保留原摘要：%s", exc)

        stats.window_messages = len(window)
        stats.summary_chars = len(summary)
        stats.window_chars = sum(len(str(m.get("content") or "")) for m in window)
        stats.summarized_messages = sum(
            1 for m in history if int(m.get("seq") or 0) <= summary_upto_seq
        )
        stats.summary_upto_seq = summary_upto_seq
        if stats.history_chars:
            stats.compressed_ratio = 1.0 - (stats.window_chars + stats.summary_chars) / stats.history_chars

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "【更早对话的滚动摘要】（这部分原文已被压缩，若与摘要冲突以最近对话为准）\n"
                        f"{summary}"
                    ),
                }
            )
        for message in window:
            role = message.get("role")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": str(message.get("content") or "")})

        return BuiltContext(
            messages=messages,
            summary=summary,
            summary_upto_seq=summary_upto_seq,
            stats=stats,
            window=window,
        )

    # ------------------------------------------------------------------ #
    async def summarize(self, previous_summary: str, messages: Sequence[dict[str, Any]]) -> str:
        """把「旧摘要 + 新增对话」合并成新摘要。"""
        cfg = self.settings
        new_text = "\n".join(
            f"{'候选人' if m.get('role') == 'user' else '面试官'}：{m.get('content')}"
            for m in messages
        )

        if self.ai_client is None or self.ai_client.mock_mode:
            return self._extractive_summary(previous_summary, messages, cfg.memory_summary_max_chars)

        system_prompt = SUMMARY_SYSTEM_PROMPT.format(max_chars=cfg.memory_summary_max_chars)
        user_prompt = SUMMARY_USER_PROMPT.format(
            summary=previous_summary or "（无）", new_messages=new_text
        )
        result = await self.ai_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # 摘要要稳定，温度压到最低
            max_tokens=800,
        )
        if not result.ok or not result.content:
            logger.warning("摘要调用失败：%s", result.error)
            return self._extractive_summary(previous_summary, messages, cfg.memory_summary_max_chars)
        return result.content[: cfg.memory_summary_max_chars * 2]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extractive_summary(
        previous_summary: str, messages: Sequence[dict[str, Any]], max_chars: int
    ) -> str:
        """离线兜底摘要：抽取式压缩，不做任何生成，绝不会编造内容。"""
        lines: list[str] = []
        if previous_summary:
            lines.append(previous_summary.strip())
        per_message = max(40, max_chars // max(1, len(messages)) // 2)
        for message in messages:
            role = "候选人" if message.get("role") == "user" else "面试官"
            content = " ".join(str(message.get("content") or "").split())
            if not content:
                continue
            snippet = content[:per_message]
            if len(content) > per_message:
                snippet += "…"
            lines.append(f"- {role}：{snippet}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    # ------------------------------------------------------------------ #
    @staticmethod
    def history_to_dicts(rows: Sequence[Any]) -> list[dict[str, Any]]:
        """把 ORM 消息行转成 memory 需要的轻量结构。"""
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "seq": getattr(row, "seq", 0),
                    "role": getattr(row, "role", "user"),
                    "content": getattr(row, "content", ""),
                }
            )
        return out


def context_preview(built: BuiltContext, max_chars: int = 600) -> dict[str, Any]:
    """给 /context 接口用的上下文预览。"""
    preview = []
    for message in built.messages:
        content = str(message.get("content") or "")
        preview.append(
            {
                "role": message.get("role"),
                "chars": len(content),
                "preview": content[:max_chars],
            }
        )
    return {
        "stats": built.stats.to_dict(),
        "summary": built.summary,
        "messages": preview,
    }


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
