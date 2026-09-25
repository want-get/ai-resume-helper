"""RAG 知识库包。

对外主要接口::

    from rag import get_knowledge_base, default_splitter, build_embedder

    kb = get_knowledge_base()
    kb.build("interview_questions", documents)          # 建库
    result = kb.search("Python GIL 是什么", top_k=5)     # 混合检索
    for chunk in result.chunks:
        print(chunk.citation_label, chunk.similarity)
"""

from .bm25 import BM25Index
from .embeddings import HashingTfidfEmbedding, build_embedder, estimate_tokens
from .kb import (
    BuildResult,
    KnowledgeBase,
    RetrievedChunk,
    SearchResult,
    get_knowledge_base,
)
from .loader import (
    Document,
    load_any,
    load_jsonl,
    load_pdf,
    load_seed_documents,
    load_text_file,
    render_interview_question,
    render_job_description,
)
from .splitter import Chunk, RecursiveSplitter, default_splitter, split_text
from .store import VectorStore

__all__ = [
    "BM25Index",
    "BuildResult",
    "Chunk",
    "Document",
    "HashingTfidfEmbedding",
    "KnowledgeBase",
    "RecursiveSplitter",
    "RetrievedChunk",
    "SearchResult",
    "VectorStore",
    "build_embedder",
    "default_splitter",
    "estimate_tokens",
    "get_knowledge_base",
    "load_any",
    "load_jsonl",
    "load_pdf",
    "load_seed_documents",
    "load_text_file",
    "render_interview_question",
    "render_job_description",
    "split_text",
]
