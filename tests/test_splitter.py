"""递归切分 + 重叠策略的单元测试。"""

from __future__ import annotations

from rag.splitter import RecursiveSplitter, split_text

SAMPLE = (
    "第一段：Python 的 GIL 是解释器级别的互斥锁，它限制了多线程的并行能力。\n\n"
    "第二段：IO 密集型任务在等待时会释放 GIL，所以多线程依然有效；"
    "CPU 密集型任务应该使用多进程。\n\n"
    "第三段：asyncio 是单线程事件循环，通过 await 主动让出控制权，"
    "切换成本远低于线程。"
)


def test_recursive_split_produces_multiple_chunks() -> None:
    chunks = split_text(SAMPLE, chunk_size=60, chunk_overlap=0)
    assert len(chunks) > 1
    assert "".join(chunks).replace(" ", "") != ""  # 内容非空


def test_chunks_respect_size_limit() -> None:
    splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=0)
    for chunk in splitter.split(SAMPLE):
        # 允许等于上限；不应显著超出
        assert len(chunk.text) <= 50


def test_overlap_between_neighbours() -> None:
    splitter = RecursiveSplitter(chunk_size=60, chunk_overlap=20)
    chunks = splitter.split(SAMPLE)
    assert len(chunks) >= 2
    for i in range(1, len(chunks)):
        # 相邻块在原文中的区间必须相交，且重叠不超过配置值
        overlap = chunks[i - 1].end - chunks[i].start
        assert overlap > 0, "相邻块之间必须有重叠"
        assert overlap <= 20


def test_offsets_match_original_text() -> None:
    splitter = RecursiveSplitter(chunk_size=60, chunk_overlap=20)
    for chunk in splitter.split(SAMPLE):
        assert SAMPLE[chunk.start : chunk.end].strip() == chunk.text


def test_separator_priority_prefers_paragraph() -> None:
    text = "A" * 40 + "\n\n" + "B" * 40
    chunks = split_text(text, chunk_size=60, chunk_overlap=0)
    assert len(chunks) == 2


def test_empty_and_whitespace_input() -> None:
    assert split_text("", chunk_size=50) == []
    assert split_text("   \n\n  ", chunk_size=50) == []


def test_hard_split_without_separators() -> None:
    text = "x" * 250
    chunks = split_text(text, chunk_size=50, chunk_overlap=0)
    assert len(chunks) == 5
    assert all(len(c) == 50 for c in chunks)


def test_overlap_clamped_below_chunk_size() -> None:
    splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=999)
    assert splitter.chunk_overlap == 49


def test_split_documents_assigns_global_index() -> None:
    splitter = RecursiveSplitter(chunk_size=60, chunk_overlap=10)
    chunks = splitter.split_documents(
        [("段落一。" * 20, {"source": "a"}), ("段落二。" * 20, {"source": "b"})]
    )
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata.get("source") in {"a", "b"} for c in chunks)
