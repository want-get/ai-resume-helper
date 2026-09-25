"""防幻觉（引用校验）单元测试。"""

from __future__ import annotations

from backend.anti_hallucination import (
    SourceRef,
    build_context_block,
    build_tool_context,
    compute_confidence,
    extract_citations,
    guard_answer,
    sanitize_answer,
    validate_citations,
)
from rag import RetrievedChunk


def make_chunk(index: int, text: str, similarity: float = 0.4) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"chunk-{index}",
        text=text,
        metadata={"source": "面试题库", "role": "Python后端开发", "chunk_index": index},
        score=1.0 / index,
        similarity=similarity,
        collection="interview_questions",
        source="hybrid",
    )


def test_build_context_block_numbers_sources() -> None:
    context, sources = build_context_block([make_chunk(1, "答案一"), make_chunk(2, "答案二")])
    assert "[1]" in context and "[2]" in context
    assert [s.index for s in sources] == [1, 2]
    assert sources[0].label.startswith("面试题库")


def test_build_context_block_respects_char_budget() -> None:
    chunks = [make_chunk(i, "内容" * 200) for i in range(1, 6)]
    context, sources = build_context_block(chunks, max_chars=500)
    assert len(context) <= 900
    assert len(sources) < len(chunks)


def test_extract_citations_supports_multiple_forms() -> None:
    assert extract_citations("结论 [1] 与 [2] 以及 [3][4]") == [1, 2, 3, 4]
    assert extract_citations("合并写法 [1,2] 和 [3，4]") == [1, 2, 3, 4]
    assert extract_citations("没有引用") == []


def test_validate_citations_flags_out_of_range() -> None:
    _, sources = build_context_block([make_chunk(1, "a"), make_chunk(2, "b")])
    report = validate_citations("结论 [1] 和 [7]", sources)
    assert report.cited == [1]
    assert report.invalid == [7]
    assert report.available == 2
    assert report.citation_rate == 0.5


def test_sanitize_removes_invalid_citations() -> None:
    _, sources = build_context_block([make_chunk(1, "a")])
    report = validate_citations("好的 [1] 以及 [9]", sources)
    cleaned = sanitize_answer("好的 [1] 以及 [9]", report)
    assert "[9]" not in cleaned
    assert "[1]" in cleaned


def test_guard_answer_warns_when_no_citation() -> None:
    _, sources = build_context_block([make_chunk(1, "a")])
    guarded = guard_answer("这是一段没有任何引用的回答", sources)
    assert guarded.report.has_citation is False
    assert any("未标注任何引用编号" in w for w in guarded.warnings)
    assert guarded.confidence < 1.0


def test_guard_answer_flags_invalid_citations() -> None:
    _, sources = build_context_block([make_chunk(1, "a")])
    guarded = guard_answer("回答 [1] 与 [42]", sources)
    assert guarded.report.invalid == [42]
    assert any("无效引用编号" in w for w in guarded.warnings)


def test_confidence_zero_without_sources() -> None:
    assert compute_confidence([], validate_citations("文本", [])) == 0.0


def test_confidence_increases_with_quality() -> None:
    _, low = build_context_block([make_chunk(1, "a", similarity=0.05)])
    _, high = build_context_block(
        [make_chunk(1, "a", similarity=0.5), make_chunk(2, "b", similarity=0.45)]
    )
    low_report = validate_citations("引用 [1]", low)
    high_report = validate_citations("引用 [1][2]", high)
    assert compute_confidence(high, high_report) > compute_confidence(low, low_report)


def test_guarded_answer_to_dict_shape() -> None:
    _, sources = build_context_block([make_chunk(1, "内容")])
    payload = guard_answer("回答 [1]", sources).to_dict()
    assert set(payload) >= {"content", "sources", "citation_report", "confidence", "refused"}
    assert payload["sources"][0]["index"] == 1


def test_build_tool_context_renders_results() -> None:
    text = build_tool_context(
        [{"name": "query_salary", "ok": True, "result": {"p50": 26}}, {"name": "search_jobs", "ok": False, "error": "超时"}]
    )
    assert "query_salary" in text and "成功" in text
    assert "search_jobs" in text and "失败" in text
