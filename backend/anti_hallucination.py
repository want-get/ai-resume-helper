"""防幻觉机制。

四道防线（对应项目要求「Prompt 约束 + 低 temperature 参数 + 引用来源」）：

1. **Prompt 约束**：``prompts/system.py`` 里的强制事实性规则，
   要求模型只依据检索到的片段作答，资料不足必须说明。
2. **低 temperature**：由 ``config.llm_temperature`` 统一控制（默认 0.2），
   减少发散与编造。业务层不单独调高，除非是创意类任务。
3. **引用来源**：把检索片段编号后写进 Prompt，要求模型用 ``[1]`` 标注，
   响应里同时原样返回带相似度、来源标签的片段列表，前端可展开核对。
4. **引用校验**：模型返回后，用正则提取引用编号，检查编号是否真实存在；
   不存在的编号会被剔除并记录，引用覆盖率会作为可信度指标返回。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from rag import RetrievedChunk, estimate_tokens

# 匹配 [1]、[2]、[1][3]、[1,2]、[1，2] 等写法
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")


@dataclass(slots=True)
class SourceRef:
    """一条可展示的引用来源。"""

    index: int
    chunk_id: str
    collection: str
    label: str
    text: str
    similarity: float | None = None
    bm25_score: float | None = None
    retrieval: str = "vector"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, preview_chars: int = 400) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "collection": self.collection,
            "label": self.label,
            "text": self.text,
            "preview": self.text[:preview_chars],
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 4),
            "retrieval": self.retrieval,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CitationReport:
    """引用校验结果。"""

    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    available: int = 0
    has_citation: bool = False
    citation_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cited": self.cited,
            "invalid": self.invalid,
            "available": self.available,
            "has_citation": self.has_citation,
            "citation_rate": round(self.citation_rate, 3),
        }


@dataclass(slots=True)
class GuardedAnswer:
    """经过防幻觉后处理后的最终回答。"""

    content: str
    sources: list[SourceRef]
    report: CitationReport
    confidence: float
    refused: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "citation_report": self.report.to_dict(),
            "confidence": round(self.confidence, 3),
            "refused": self.refused,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------- #
# 上下文拼装
# ---------------------------------------------------------------------- #
def build_context_block(
    chunks: Sequence[RetrievedChunk], max_chars: int = 6000
) -> tuple[str, list[SourceRef]]:
    """把检索片段编号后拼成 Prompt 里的「参考资料」区块。"""
    sources: list[SourceRef] = []
    blocks: list[str] = []
    used = 0

    for i, chunk in enumerate(chunks, start=1):
        text = (chunk.text or "").strip()
        if not text:
            continue
        block = f"[{i}] 来源：{chunk.citation_label}\n{text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
        sources.append(
            SourceRef(
                index=i,
                chunk_id=chunk.id,
                collection=chunk.collection,
                label=chunk.citation_label,
                text=text,
                similarity=chunk.similarity,
                bm25_score=chunk.bm25_score,
                retrieval=chunk.source,
                metadata=chunk.metadata,
            )
        )

    return "\n\n".join(blocks), sources


def build_tool_context(results: Iterable[dict[str, Any]]) -> str:
    """把工具调用结果渲染成 Prompt 里的「工具返回数据」区块。"""
    blocks: list[str] = []
    for i, item in enumerate(results, start=1):
        name = item.get("name", "tool")
        ok = item.get("ok", True)
        payload = item.get("result")
        if isinstance(payload, (dict, list)):
            import json

            payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            payload_text = str(payload)
        status = "成功" if ok else "失败"
        blocks.append(f"[工具{i}] {name}（{status}）\n{payload_text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------- #
# 引用校验
# ---------------------------------------------------------------------- #
def extract_citations(text: str) -> list[int]:
    """提取回答中出现的引用编号（去重、保序）。"""
    found: list[int] = []
    for match in _CITATION_RE.finditer(text or ""):
        for part in re.split(r"[,，]", match.group(1)):
            part = part.strip()
            if part.isdigit():
                number = int(part)
                if number not in found:
                    found.append(number)
    return found


def validate_citations(answer: str, sources: Sequence[SourceRef]) -> CitationReport:
    """校验引用编号是否都在真实来源范围内。"""
    available = len(sources)
    cited = extract_citations(answer)
    valid = [n for n in cited if 1 <= n <= available]
    invalid = [n for n in cited if n < 1 or n > available]
    rate = (len(valid) / len(cited)) if cited else 0.0
    return CitationReport(
        cited=valid,
        invalid=invalid,
        available=available,
        has_citation=bool(cited),
        citation_rate=rate,
    )


def sanitize_answer(answer: str, report: CitationReport) -> str:
    """剔除指向不存在来源的引用编号，避免误导用户。"""
    if not report.invalid:
        return answer
    for number in report.invalid:
        answer = answer.replace(f"[{number}]", "")
    return re.sub(r"[ \t]{2,}", " ", answer).strip()


def compute_confidence(sources: Sequence[SourceRef], report: CitationReport) -> float:
    """可信度打分（0~1），用于前端提示「本次回答的可靠程度」。

    组成：
      * 检索质量（最高相似度）占 45%
      * 引用覆盖率占 35%
      * 命中来源数量占 20%
    """
    if not sources:
        return 0.0

    similarities = [s.similarity for s in sources if s.similarity is not None]
    top_sim = max(similarities) if similarities else 0.0
    # 哈希 TF-IDF 的相似度天然偏低，做个温和的归一化
    quality = min(1.0, top_sim / 0.45)

    coverage = report.citation_rate if report.has_citation else 0.0
    breadth = min(1.0, len(sources) / 3.0)
    return max(0.0, min(1.0, 0.45 * quality + 0.35 * coverage + 0.20 * breadth))


def guard_answer(
    answer: str,
    sources: Sequence[SourceRef],
    refused: bool = False,
    extra_warnings: Sequence[str] | None = None,
) -> GuardedAnswer:
    """对模型输出做一次完整的后置校验。"""
    report = validate_citations(answer, sources)
    warnings: list[str] = list(extra_warnings or [])

    if sources:
        if not report.has_citation:
            warnings.append("回答未标注任何引用编号，请谨慎采信")
        elif report.invalid:
            warnings.append(f"已剔除无效引用编号：{report.invalid}")
    content = sanitize_answer(answer, report)
    confidence = compute_confidence(sources, report)

    if not sources:
        confidence = 0.0 if refused else min(confidence, 0.2)

    return GuardedAnswer(
        content=content,
        sources=list(sources),
        report=report,
        confidence=confidence,
        refused=refused,
        warnings=warnings,
    )


def estimate_context_tokens(sources: Sequence[SourceRef]) -> int:
    return sum(estimate_tokens(s.text) for s in sources)
