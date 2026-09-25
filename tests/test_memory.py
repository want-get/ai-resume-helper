"""长上下文（滑动窗口 + 滚动摘要）单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from backend.memory import ConversationMemory
from config import Settings

SYSTEM = "你是一个面试官。"


@pytest.fixture()
def memory() -> ConversationMemory:
    """每个测试用独立的 Settings 实例，避免污染全局单例配置。"""
    return ConversationMemory(settings=Settings())


def make_history(count: int, chars: int = 100) -> list[dict[str, Any]]:
    history = []
    for i in range(1, count + 1):
        history.append(
            {
                "seq": i,
                "role": "user" if i % 2 else "assistant",
                "content": f"第{i}轮" + "内容" * (chars // 2),
            }
        )
    return history


def test_select_window_limits_message_count(memory: ConversationMemory) -> None:
    memory.settings.memory_window_messages = 4
    history = make_history(20)
    start = memory.select_window(history)
    assert len(history) - start == 4


def test_select_window_limits_chars(memory: ConversationMemory) -> None:
    memory.settings.memory_window_messages = 50
    memory.settings.memory_window_max_chars = 300
    history = make_history(20, chars=200)
    start = memory.select_window(history)
    window_chars = sum(len(m["content"]) for m in history[start:])
    # 至少保留 1 条，且不会远超预算
    assert 1 <= len(history) - start < 20
    assert window_chars <= 200 * 2


async def test_build_keeps_recent_messages_verbatim(memory: ConversationMemory) -> None:
    memory.settings.memory_window_messages = 3
    memory.settings.memory_enable_summary = False
    built = await memory.build(make_history(10), SYSTEM)
    assert built.messages[0]["role"] == "system"
    # 窗口内 3 条原文 + 1 条 system
    assert len(built.messages) == 4
    assert built.stats.window_messages == 3
    assert built.stats.total_messages == 10


async def test_build_triggers_summary_and_compresses(memory: ConversationMemory) -> None:
    memory.settings.memory_window_messages = 4
    memory.settings.memory_window_max_chars = 1_000_000
    memory.settings.memory_summary_trigger_chars = 500
    memory.settings.memory_enable_summary = True

    built = await memory.build(make_history(20, chars=200), SYSTEM)
    assert built.stats.summary_updated is True
    assert built.summary
    assert built.stats.summarized_messages > 0
    assert built.stats.summary_upto_seq > 0
    # 摘要以 system 角色注入，位于窗口原文之前
    assert built.messages[1]["role"] == "system"
    assert "滚动摘要" in built.messages[1]["content"]


async def test_summary_not_retriggered_without_new_messages(memory: ConversationMemory) -> None:
    memory.settings.memory_window_messages = 4
    memory.settings.memory_window_max_chars = 1_000_000
    memory.settings.memory_summary_trigger_chars = 500

    history = make_history(20, chars=200)
    first = await memory.build(history, SYSTEM)
    second = await memory.build(
        history, SYSTEM, summary=first.summary, summary_upto_seq=first.summary_upto_seq
    )
    assert second.stats.summary_updated is False
    assert second.summary == first.summary


async def test_extractive_summary_never_invents_content(memory: ConversationMemory) -> None:
    messages = [
        {"seq": 1, "role": "user", "content": "我负责订单中台，QPS 提升了 5 倍"},
        {"seq": 2, "role": "assistant", "content": "很好，请说明具体做法"},
    ]
    summary = memory._extractive_summary("", messages, max_chars=1000)
    assert "订单中台" in summary
    assert "QPS" in summary
    assert len(summary) <= 1000


async def test_empty_history_returns_system_only(memory: ConversationMemory) -> None:
    built = await memory.build([], SYSTEM)
    assert built.messages == [{"role": "system", "content": SYSTEM}]
    assert built.stats.total_messages == 0


def test_history_to_dicts_converts_orm_like_rows() -> None:
    class Row:
        def __init__(self, seq: int, role: str, content: str) -> None:
            self.seq, self.role, self.content = seq, role, content

    out = ConversationMemory.history_to_dicts([Row(1, "user", "你好")])
    assert out == [{"seq": 1, "role": "user", "content": "你好"}]
