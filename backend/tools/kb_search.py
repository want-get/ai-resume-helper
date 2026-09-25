"""知识库检索工具（把 RAG 也做成模型可自主调用的工具）。

这样模型在自由对话中也能主动「去查一下知识库」，
而不是只能被动接受后端预先检索好的片段。
"""

from __future__ import annotations

from typing import Any

from config import get_settings

from .registry import ToolRegistry, ToolSpec

KB_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "检索关键词或问题，例如：GIL 多线程、缓存穿透"},
        "collection": {
            "type": "string",
            "enum": ["interview_questions", "job_descriptions"],
            "description": "检索范围：interview_questions=面试题库，job_descriptions=岗位JD库",
        },
        "top_k": {"type": "integer", "description": "返回条数，默认 4，最大 10"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def search_knowledge_base(
    query: str, collection: str | None = None, top_k: int = 4
) -> dict[str, Any]:
    """在面试题库 / 岗位 JD 库中做混合检索。"""
    from rag import get_knowledge_base

    cfg = get_settings()
    top_k = max(1, min(int(top_k or 4), 10))
    collections = [collection] if collection in cfg.kb_collections else None

    kb = get_knowledge_base()
    result = await kb.asearch(query, collections=collections, top_k=top_k)

    if not result.chunks:
        return {
            "found": False,
            "query": query,
            "chunks": [],
            "source": "本地知识库（Chroma + BM25 混合检索）",
            "hint": "知识库中未收录相关内容，可以换关键词再试",
        }

    return {
        "found": True,
        "query": query,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "retrieval": {
            "dense_hits": result.dense_hits,
            "bm25_hits": result.bm25_hits,
            "strategy": "向量召回 + BM25 关键词召回 + RRF 融合",
        },
        "chunks": [
            {
                "index": i,
                "label": chunk.citation_label,
                "collection": chunk.collection,
                "text": chunk.text,
                "similarity": None if chunk.similarity is None else round(chunk.similarity, 4),
            }
            for i, chunk in enumerate(result.chunks, start=1)
        ],
        "source": "本地知识库（Chroma + BM25 混合检索）",
    }


def register(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            description=(
                "在面试题库与岗位 JD 知识库中做语义 + 关键词混合检索，"
                "返回带来源标签的原文片段。当用户询问面试题、考察点、岗位要求时必须调用。"
            ),
            parameters=KB_SEARCH_PARAMETERS,
            handler=search_knowledge_base,
            tags=("rag", "knowledge"),
        )
    )
