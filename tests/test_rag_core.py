"""离线向量化与 BM25 的单元测试。"""

from __future__ import annotations

import math

import pytest

from config import get_settings
from rag.bm25 import BM25Index
from rag.embeddings import HashingTfidfEmbedding, analyze, estimate_tokens, normalize_text

DOCS = [
    "Python 的 GIL 是解释器级别的互斥锁，限制多线程并行。",
    "MySQL 索引在列上做函数运算会失效。",
    "用户运营要提升新用户 7 日留存。",
]


@pytest.fixture()
def embedder() -> HashingTfidfEmbedding:
    cfg = get_settings()
    return HashingTfidfEmbedding(
        dim=1024, ngram_min=cfg.rag_ngram_min, ngram_max=cfg.rag_ngram_max
    ).fit(DOCS)


def test_analyze_returns_features() -> None:
    features = analyze("Python GIL 多线程")
    assert any(f.startswith("w:python") for f in features)
    assert any(f.startswith("c:") for f in features)


def test_normalize_text_fullwidth_and_case() -> None:
    assert normalize_text("Ｐｙｔｈｏｎ　测试") == "python 测试"


def test_embedding_is_normalized(embedder: HashingTfidfEmbedding) -> None:
    vector = embedder.transform_one("GIL 多线程")
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-4


def test_embedding_dimension(embedder: HashingTfidfEmbedding) -> None:
    assert len(embedder.transform_one("任意文本")) == 1024


def test_relevant_document_scores_higher(embedder: HashingTfidfEmbedding) -> None:
    query, _mysql, _ops = embedder.transform(["GIL 多线程并行", DOCS[1], DOCS[2]])
    python_doc = embedder.transform([DOCS[0]])[0]

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    assert dot(query, python_doc) > dot(query, _mysql)
    assert dot(query, python_doc) > dot(query, _ops)


def test_empty_text_gives_zero_vector(embedder: HashingTfidfEmbedding) -> None:
    assert all(v == 0.0 for v in embedder.transform_one(""))


def test_embedder_state_roundtrip(state_dir) -> None:
    path = state_dir / "idf.json"
    embedder = HashingTfidfEmbedding(dim=512, state_path=path).fit(DOCS)
    embedder.save()
    reloaded = HashingTfidfEmbedding(dim=512, state_path=path).load()
    assert reloaded.n_docs == embedder.n_docs
    assert reloaded.df == embedder.df
    assert reloaded.transform_one("GIL") == pytest.approx(embedder.transform_one("GIL"))


def test_embedder_chroma_protocol(embedder: HashingTfidfEmbedding) -> None:
    """Chroma 要求实现 name / get_config / build_from_config。"""
    assert HashingTfidfEmbedding.name() == "hashing-tfidf"
    config = embedder.get_config()
    rebuilt = HashingTfidfEmbedding.build_from_config(config)
    assert rebuilt.dim == embedder.dim


def test_bm25_matched_idf_separates_relevance() -> None:
    index = BM25Index(dim=1024).fit([f"doc{i}" for i in range(len(DOCS))], DOCS)

    relevant = index.search("GIL 多线程")
    irrelevant = index.search("今天天气怎么样适合穿什么衣服")

    assert relevant, "相关查询应当有召回"
    assert relevant[0].matched_idf > (irrelevant[0].matched_idf if irrelevant else 0.0)


def test_bm25_scores_are_positive() -> None:
    index = BM25Index(dim=1024).fit([f"doc{i}" for i in range(len(DOCS))], DOCS)
    for hit in index.search("索引 失效"):
        assert hit.score > 0
        assert hit.matched_idf > 0


def test_bm25_state_roundtrip(state_dir) -> None:
    path = state_dir / "bm25.json"
    index = BM25Index(dim=512, state_path=path).fit(["a", "b"], DOCS[:2])
    index.save()
    reloaded = BM25Index(dim=512, state_path=path).load()
    assert reloaded.n_docs == index.n_docs
    assert reloaded.avgdl == pytest.approx(index.avgdl)


def test_estimate_tokens_is_positive() -> None:
    assert estimate_tokens("你好世界") > 0
    assert estimate_tokens("") == 0
