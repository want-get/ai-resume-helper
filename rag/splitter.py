"""递归字符切分器（Recursive Character Text Splitter）+ 重叠窗口。

切分策略（对应项目要求「递归切分 + 带重叠策略」）：

1. **递归切分**：按分隔符优先级从「粗」到「细」递归下降。
   先用 ``\\n\\n`` 切段落，段落仍然过大就改用 ``\\n``、再到 ``。``/``！``/``？``、
   最后退化到逗号、空格乃至单字符。这样切出来的块天然贴近语义边界，
   而不是把句子从中间劈开。
2. **带重叠**：相邻块之间保留 ``chunk_overlap`` 个字符的重叠（overlap 直接取自
   原文的连续片段），避免关键信息正好落在切分点被截断而检索不到。
3. **可追溯**：每个块都带 ``start`` / ``end`` 字符偏移，供「引用来源」精确定位。

纯标准库实现，不依赖 LangChain。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# 分隔符按优先级从粗到细排列，"" 表示最终退化为按字符切
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",      # 段落
    "\n",        # 换行
    "。", "！", "？", "；", "……",   # 中文句末
    ". ", "! ", "? ", "; ",          # 英文句末
    "，", ", ",                      # 从句
    "、", " ",                       # 词
    "",                              # 兜底：逐字符
)


@dataclass(slots=True)
class Chunk:
    """一个切分后的文本块。"""

    text: str
    index: int
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """拍平成可写入 Chroma 的元数据（值必须是 str/int/float/bool）。"""
        meta: dict[str, Any] = {
            "chunk_index": self.index,
            "char_start": self.start,
            "char_end": self.end,
        }
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta[key] = value
            elif value is None:
                continue
            else:
                meta[key] = str(value)
        return meta


class RecursiveSplitter:
    """递归 + 重叠的文本切分器。"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Sequence[str] | None = None,
        min_chunk_chars: int = 0,
        keep_separator: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负数")
        # overlap 必须小于 chunk_size，否则会退化为原地踏步
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(0, chunk_size - 1))
        self.separators = tuple(separators) if separators else DEFAULT_SEPARATORS
        self.min_chunk_chars = max(0, min_chunk_chars)
        self.keep_separator = keep_separator

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """把一段文本切成带重叠的块。"""
        if not text or not text.strip():
            return []

        pieces = self._recursive_pieces(text, 0, self.separators)
        if not pieces:
            return []

        spans = self._merge_pieces(pieces)
        spans = self._apply_overlap(spans)
        spans = self._absorb_short_tail(spans)

        chunks: list[Chunk] = []
        for idx, (start, end) in enumerate(spans):
            piece = text[start:end]
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    index=len(chunks),
                    start=start,
                    end=end,
                    metadata=dict(metadata or {}),
                )
            )
        return chunks

    def split_documents(self, documents: Iterable[Any]) -> list[Chunk]:
        """切分多篇文档。

        ``documents`` 中每一项要么是 ``(text, metadata)`` 元组，
        要么是带 ``.text`` / ``.metadata`` 属性的对象（如 rag.loader.Document）。
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            if isinstance(doc, tuple):
                text, meta = doc[0], dict(doc[1] or {})
            else:
                text, meta = doc.text, dict(getattr(doc, "metadata", {}) or {})
            for chunk in self.split(text, meta):
                chunk.index = len(all_chunks)
                all_chunks.append(chunk)
        return all_chunks

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _recursive_pieces(
        self, text: str, offset: int, separators: Sequence[str]
    ) -> list[tuple[str, int]]:
        """递归下降，返回 (片段文本, 在原文中的起始下标) 列表。"""
        if len(text) <= self.chunk_size:
            return [(text, offset)] if text else []

        if not separators:
            return self._hard_split(text, offset)

        sep = separators[0]
        rest = separators[1:]

        if sep == "":
            return self._hard_split(text, offset)

        if sep not in text:
            # 当前分隔符不存在，换下一个更细的分隔符
            return self._recursive_pieces(text, offset, rest)

        parts = text.split(sep)
        pieces: list[tuple[str, int]] = []
        cursor = offset
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            piece = part if (is_last or not self.keep_separator) else part + sep
            if piece:
                pieces.append((piece, cursor))
            cursor += len(piece)

        result: list[tuple[str, int]] = []
        for piece, start in pieces:
            if len(piece) > self.chunk_size:
                result.extend(self._recursive_pieces(piece, start, rest))
            else:
                result.append((piece, start))
        return result

    def _hard_split(self, text: str, offset: int) -> list[tuple[str, int]]:
        """兜底：按固定长度硬切（用于没有任何分隔符的超长串）。"""
        return [
            (text[i : i + self.chunk_size], offset + i)
            for i in range(0, len(text), self.chunk_size)
        ]

    def _merge_pieces(self, pieces: Sequence[tuple[str, int]]) -> list[tuple[int, int]]:
        """把细碎片段贪心合并成不超过 chunk_size 的块，返回字符区间。"""
        spans: list[tuple[int, int]] = []
        cur_start: int | None = None
        cur_len = 0

        for piece, start in pieces:
            plen = len(piece)
            if cur_start is not None and cur_len + plen > self.chunk_size:
                spans.append((cur_start, start))
                cur_start = None
                cur_len = 0
            if cur_start is None:
                cur_start = start
            cur_len += plen

        if cur_start is not None and pieces:
            last_end = pieces[-1][1] + len(pieces[-1][0])
            spans.append((cur_start, last_end))
        return spans

    def _apply_overlap(self, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """给相邻块加上重叠：把本块起点向前回退 overlap 个字符。"""
        if self.chunk_overlap <= 0 or len(spans) <= 1:
            return list(spans)

        overlapped: list[tuple[int, int]] = [spans[0]]
        for i in range(1, len(spans)):
            start, end = spans[i]
            prev_start, prev_end = spans[i - 1]
            # 回退量不能越过上一块的起点，也不能越过本块自身长度
            lower_bound = max(prev_start, prev_end - self.chunk_overlap)
            new_start = min(start, max(0, start - self.chunk_overlap))
            new_start = max(new_start, lower_bound)
            overlapped.append((min(new_start, start), end))
        return overlapped

    def _absorb_short_tail(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """把过短的尾块并入前一块，避免产生大量无意义的碎片。"""
        if self.min_chunk_chars <= 0 or len(spans) <= 1:
            return spans
        start, end = spans[-1]
        if end - start < self.min_chunk_chars:
            prev_start, _ = spans[-2]
            spans = spans[:-2] + [(prev_start, end)]
        return spans


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """便捷函数：只返回文本块字符串列表。"""
    splitter = RecursiveSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    return [chunk.text for chunk in splitter.split(text)]


def default_splitter() -> RecursiveSplitter:
    """按全局配置构造一个切分器。"""
    from config import get_settings

    cfg = get_settings()
    return RecursiveSplitter(
        chunk_size=cfg.rag_chunk_size,
        chunk_overlap=cfg.rag_chunk_overlap,
        min_chunk_chars=cfg.rag_min_chunk_chars,
    )
