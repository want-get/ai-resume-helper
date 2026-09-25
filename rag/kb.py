"""RAG 知识库门面：建库、增量入库、混合检索。

一次完整的 RAG 链路：

    文档 → 递归切分(+重叠) → 离线向量化 → Chroma 持久化
                                        ↘ BM25 关键词索引
    查询 → 向量召回 + BM25 召回 → RRF 融合 → Top-K 片段 → 交给大模型并强制引用

对外只暴露同步方法 + ``a*`` 异步包装，FastAPI 里一律用异步版本。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import get_settings

from .bm25 import BM25Index
from .embeddings import HashingTfidfEmbedding, build_embedder, estimate_tokens
from .loader import Document, load_seed_documents
from .splitter import Chunk, RecursiveSplitter, default_splitter
from .store import VectorStore


@dataclass(slots=True)
class RetrievedChunk:
    """一条检索结果。"""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float                       # 融合后的排序分（RRF）
    similarity: float | None = None    # 向量余弦相似度（如有）
    bm25_score: float | None = None    # BM25 原始分（如有）
    collection: str = ""
    source: str = "vector"             # vector / bm25 / hybrid

    @property
    def citation_label(self) -> str:
        """给「引用来源」用的可读标签。"""
        meta = self.metadata or {}
        parts: list[str] = []
        base = meta.get("source") or self.collection
        parts.append(str(base))
        if meta.get("role"):
            parts.append(str(meta["role"]))
        elif meta.get("title"):
            parts.append(str(meta["title"]))
        if meta.get("category"):
            parts.append(str(meta["category"]))
        if meta.get("company"):
            parts.append(str(meta["company"]))
        if meta.get("city"):
            parts.append(str(meta["city"]))
        if meta.get("chunk_index") is not None:
            parts.append(f"片段{int(meta['chunk_index']) + 1}")
        return " · ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "collection": self.collection,
            "text": self.text,
            "label": self.citation_label,
            "metadata": self.metadata,
            "score": round(float(self.score), 6),
            "similarity": None if self.similarity is None else round(float(self.similarity), 4),
            "bm25_score": None if self.bm25_score is None else round(float(self.bm25_score), 4),
            "source": self.source,
            "tokens": estimate_tokens(self.text),
        }


@dataclass(slots=True)
class BuildResult:
    collection: str
    documents: int
    chunks: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "documents": self.documents,
            "chunks": self.chunks,
            "seconds": round(self.seconds, 3),
        }


@dataclass(slots=True)
class SearchResult:
    """一次检索的完整结果（含各阶段耗时，便于压测与调优）。"""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    dense_hits: int = 0
    bm25_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "collections": self.collections,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "dense_hits": self.dense_hits,
            "bm25_hits": self.bm25_hits,
            "chunks": [c.to_dict() for c in self.chunks],
        }


class KnowledgeBase:
    """Chroma + BM25 混合检索知识库。"""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        splitter: RecursiveSplitter | None = None,
    ) -> None:
        cfg = get_settings()
        self.cfg = cfg
        self.persist_dir = Path(persist_dir or cfg.chroma_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.persist_dir / "kb_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.splitter = splitter or default_splitter()
        # Chroma 的 EF 实例：仅作为占位，实际向量都由我们自己算好传入
        self._placeholder_embedder = HashingTfidfEmbedding(
            dim=cfg.rag_embed_dim,
            ngram_min=cfg.rag_ngram_min,
            ngram_max=cfg.rag_ngram_max,
            state_path=self.state_dir / "_placeholder.json",
        )
        self.store = VectorStore(
            self.persist_dir,
            self._placeholder_embedder,
            memory_index=cfg.rag_memory_index,
            memory_index_max_chunks=cfg.rag_memory_index_max_chunks,
        )

        # 检索用的 IDF 统计量与 BM25 索引都是「读多写少」的：
        # 每次检索都从磁盘反序列化会直接拖垮并发性能，因此这里做 mtime 缓存，
        # 只有重建/增量入库导致文件变化时才会重新加载。
        self._embedder_cache: dict[str, tuple[float, HashingTfidfEmbedding]] = {}
        self._bm25_cache: dict[str, tuple[float, BM25Index]] = {}
        self._cache_checked_at: dict[str, float] = {}
        self._cache_lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # 带 mtime 失效的内存缓存
    # ------------------------------------------------------------------ #
    def _state_mtime(self, collection: str) -> float:
        """状态文件时间戳；最多每 ``rag_state_ttl`` 秒才真正 stat 一次。

        这样既避免了「每次检索都发 8 次 stat 系统调用」的开销，
        又能保证脚本重建知识库后，运行中的服务最多 TTL 秒后自动生效。
        """
        now = time.monotonic()
        last = self._cache_checked_at.get(collection, 0.0)
        cache_key = f"__mtime__{collection}"
        if now - last < self.cfg.rag_state_ttl:
            return self._cache_checked_at.get(cache_key, 0.0)

        paths = (self._idf_path(collection), self._bm25_path(collection))
        mtime = max((p.stat().st_mtime if p.exists() else 0.0) for p in paths)
        with self._cache_lock:
            self._cache_checked_at[collection] = now
            self._cache_checked_at[cache_key] = mtime
        return mtime

    def _cached_embedder(self, collection: str) -> HashingTfidfEmbedding:
        mtime = self._state_mtime(collection)
        with self._cache_lock:
            cached = self._embedder_cache.get(collection)
            if cached is not None and cached[0] == mtime:
                return cached[1]
            embedder = build_embedder(self._idf_path(collection)).load()
            self._embedder_cache[collection] = (mtime, embedder)
            return embedder

    def _cached_bm25(self, collection: str) -> BM25Index:
        mtime = self._state_mtime(collection)
        with self._cache_lock:
            cached = self._bm25_cache.get(collection)
            if cached is not None and cached[0] == mtime:
                return cached[1]
            index = BM25Index(
                dim=self.cfg.rag_embed_dim,
                ngram_min=self.cfg.rag_ngram_min,
                ngram_max=self.cfg.rag_ngram_max,
                state_path=self._bm25_path(collection),
            ).load()
            self._bm25_cache[collection] = (mtime, index)
            return index

    def _invalidate_cache(self, collection: str) -> None:
        with self._cache_lock:
            self._embedder_cache.pop(collection, None)
            self._bm25_cache.pop(collection, None)
            self._cache_checked_at.pop(collection, None)
            self._cache_checked_at.pop(f"__mtime__{collection}", None)

    # ------------------------------------------------------------------ #
    # 状态文件
    # ------------------------------------------------------------------ #
    def _idf_path(self, collection: str) -> Path:
        return self.state_dir / f"{collection}.idf.json"

    def _bm25_path(self, collection: str) -> Path:
        return self.state_dir / f"{collection}.bm25.json"

    def _manifest_path(self) -> Path:
        return self.state_dir / "manifest.json"

    def load_embedder(self, collection: str) -> HashingTfidfEmbedding:
        """加载（带缓存）该集合的向量化器。"""
        return self._cached_embedder(collection)

    def load_bm25(self, collection: str) -> BM25Index:
        """加载（带缓存）该集合的 BM25 索引。"""
        return self._cached_bm25(collection)

    def _write_manifest(self, collection: str, **info: Any) -> None:
        manifest: dict[str, Any] = {}
        path = self._manifest_path()
        if path.exists():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        entry = manifest.get(collection, {})
        entry.update(info)
        entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest[collection] = entry
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 建库
    # ------------------------------------------------------------------ #
    def build(
        self,
        collection: str,
        documents: Sequence[Document],
        rebuild: bool = True,
    ) -> BuildResult:
        """切分 + 向量化 + 写入 Chroma，并重建 BM25 索引。"""
        started = time.perf_counter()

        chunks: list[Chunk] = self.splitter.split_documents(documents)
        if rebuild:
            self.store.drop(collection)

        if not chunks:
            self._write_manifest(collection, documents=len(documents), chunks=0)
            return BuildResult(collection, len(documents), 0, time.perf_counter() - started)

        texts = [chunk.text for chunk in chunks]
        ids = [f"{collection}-{i:06d}" for i in range(len(chunks))]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        # 1) 在该集合的全量语料上拟合 IDF（TF-IDF 的「IDF」需要语料统计）
        embedder = build_embedder(self._idf_path(collection)).fit(texts)
        embedder.save()
        vectors = embedder.transform(texts)

        # 2) 写入向量库（显式传入我们自己算的向量）
        self.store.upsert(collection, ids, texts, metadatas, vectors)

        # 3) 构建 BM25 索引
        bm25 = BM25Index(
            dim=self.cfg.rag_embed_dim,
            ngram_min=self.cfg.rag_ngram_min,
            ngram_max=self.cfg.rag_ngram_max,
            state_path=self._bm25_path(collection),
        ).fit(ids, texts)
        bm25.save()
        self._invalidate_cache(collection)

        self._write_manifest(
            collection,
            documents=len(documents),
            chunks=len(chunks),
            chunk_size=self.splitter.chunk_size,
            chunk_overlap=self.splitter.chunk_overlap,
            embed_dim=self.cfg.rag_embed_dim,
            avg_chunk_chars=round(sum(len(t) for t in texts) / len(texts), 1),
        )
        return BuildResult(collection, len(documents), len(chunks), time.perf_counter() - started)

    def add_documents(self, collection: str, documents: Sequence[Document]) -> BuildResult:
        """增量入库。

        沿用已有的 IDF 统计量（不重新拟合），因此新文档与旧文档仍在同一
        向量空间；BM25 索引会基于全量文本重建，保证 df / avgdl 正确。
        """
        started = time.perf_counter()
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            return BuildResult(collection, len(documents), 0, time.perf_counter() - started)

        embedder = self.load_embedder(collection)
        if embedder.n_docs == 0:
            # 该集合还是空的，退化为全量建库
            return self.build(collection, documents, rebuild=False)

        base = self.store.count(collection)
        texts = [chunk.text for chunk in chunks]
        ids = [f"{collection}-{base + i:06d}" for i in range(len(chunks))]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        vectors = embedder.transform(texts)

        self.store.upsert(collection, ids, texts, metadatas, vectors)
        self._rebuild_bm25(collection)
        self._write_manifest(
            collection,
            documents=len(documents),
            chunks=self.store.count(collection),
            last_added=len(chunks),
        )
        return BuildResult(collection, len(documents), len(chunks), time.perf_counter() - started)

    def _rebuild_bm25(self, collection: str) -> None:
        collection_obj = self.store.get_collection(collection)
        raw = collection_obj.get(include=["documents"])
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        if not ids:
            return
        bm25 = BM25Index(
            dim=self.cfg.rag_embed_dim,
            ngram_min=self.cfg.rag_ngram_min,
            ngram_max=self.cfg.rag_ngram_max,
            state_path=self._bm25_path(collection),
        ).fit(ids, docs)
        bm25.save()
        self._invalidate_cache(collection)

    def build_from_seed(self, seed_dir: str | Path | None = None) -> list[BuildResult]:
        """用 ``data/seed`` 下的种子数据重建全部集合。"""
        cfg = self.cfg
        seed_dir = Path(seed_dir) if seed_dir else cfg.seed_dir if hasattr(cfg, "seed_dir") else None
        if seed_dir is None:
            from config import SEED_DIR

            seed_dir = SEED_DIR
        documents = load_seed_documents(seed_dir)
        return [
            self.build(collection, docs, rebuild=True)
            for collection, docs in documents.items()
            if docs
        ]

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        collections: Sequence[str] | None = None,
        top_k: int | None = None,
        role_type: str | None = None,
        hybrid: bool | None = None,
        candidate_k: int | None = None,
    ) -> SearchResult:
        """混合检索：向量召回 + BM25 召回 + RRF 融合。"""
        started = time.perf_counter()
        cfg = self.cfg
        top_k = top_k or cfg.rag_top_k
        candidate_k = candidate_k or cfg.rag_candidate_k
        hybrid = cfg.rag_hybrid if hybrid is None else hybrid
        targets = list(collections or cfg.kb_collections)

        result = SearchResult(query=query, collections=targets)
        if not query.strip():
            return result

        fused: dict[str, dict[str, Any]] = {}

        for collection in targets:
            if self.store.count(collection) == 0:
                continue

            # ---- 向量召回 ----
            dense_hits: list[dict[str, Any]] = []
            embedder = self.load_embedder(collection)
            if embedder.n_docs > 0:
                query_vector = embedder.transform_one(query)
                where = {"role_type": role_type} if role_type else None
                dense_hits = self.store.query(
                    collection, query_vector, n_results=candidate_k, where=where
                )
                if not dense_hits and where is not None:
                    # 岗位类型过滤后为空，退化为不过滤，避免"什么都搜不到"
                    dense_hits = self.store.query(
                        collection, query_vector, n_results=candidate_k, where=None
                    )

            # ---- BM25 召回 ----
            bm25_hits: list[Any] = []
            if hybrid:
                bm25 = self.load_bm25(collection)
                bm25_hits = bm25.search(query, top_k=candidate_k)

            result.dense_hits += len(dense_hits)
            result.bm25_hits += len(bm25_hits)

            # ---- RRF 融合 ----
            for rank, hit in enumerate(dense_hits, start=1):
                entry = fused.setdefault(
                    hit["id"],
                    {
                        "id": hit["id"],
                        "text": hit["text"],
                        "metadata": hit["metadata"],
                        "collection": collection,
                        "rrf": 0.0,
                        "similarity": None,
                        "bm25_score": None,
                        "matched_idf": None,
                        "paths": set(),
                    },
                )
                entry["rrf"] += 1.0 / (cfg.rag_rrf_k + rank)
                entry["similarity"] = hit["similarity"]
                entry["paths"].add("vector")

            # BM25 命中的文档需要回表拿文本
            missing = [hit.doc_id for hit in bm25_hits if hit.doc_id not in fused]
            fetched = self.store.memory_get(collection, missing) if missing else {}
            for rank, hit in enumerate(bm25_hits, start=1):
                entry = fused.get(hit.doc_id)
                if entry is None:
                    payload = fetched.get(hit.doc_id)
                    if payload is None:
                        continue
                    entry = fused.setdefault(
                        hit.doc_id,
                        {
                            "id": hit.doc_id,
                            "text": payload["text"],
                            "metadata": payload["metadata"],
                            "collection": collection,
                            "rrf": 0.0,
                            "similarity": None,
                            "bm25_score": None,
                            "matched_idf": None,
                            "paths": set(),
                        },
                    )
                entry["rrf"] += 1.0 / (cfg.rag_rrf_k + rank)
                entry["bm25_score"] = hit.score
                entry["matched_idf"] = hit.matched_idf
                entry["paths"].add("bm25")

        ranked = sorted(fused.values(), key=lambda item: item["rrf"], reverse=True)
        candidates = ranked[: top_k * 2]

        # ---- 相关性判定：向量相似度 OR 关键词信息量，任一过线即认为相关 ----
        # 两条判据互补：短问句靠向量，长问句靠关键词信息量；
        # 都不达线说明知识库里确实没有相关内容，返回空让上层走「拒答」分支。
        if candidates:
            top_similarity = max((item["similarity"] or 0.0) for item in candidates)
            top_matched_idf = max((item["matched_idf"] or 0.0) for item in candidates)
            if (
                top_similarity < cfg.rag_min_score
                and top_matched_idf < cfg.rag_min_matched_idf
            ):
                result.chunks = []
                result.elapsed_ms = (time.perf_counter() - started) * 1000
                return result

        chunks: list[RetrievedChunk] = []
        for entry in candidates:
            similarity = entry["similarity"]
            matched_idf = entry["matched_idf"] or 0.0
            relevant = (similarity is not None and similarity >= cfg.rag_min_score) or (
                matched_idf >= cfg.rag_min_matched_idf
            )
            if not relevant:
                continue
            paths = entry["paths"]
            source = "hybrid" if len(paths) > 1 else ("vector" if "vector" in paths else "bm25")
            chunk = RetrievedChunk(
                id=entry["id"],
                text=entry["text"],
                metadata=entry["metadata"] or {},
                score=entry["rrf"],
                similarity=similarity,
                bm25_score=entry["bm25_score"],
                collection=entry["collection"],
                source=source,
            )
            chunks.append(chunk)
            if len(chunks) >= top_k:
                break

        result.chunks = chunks
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    # ------------------------------------------------------------------ #
    # 运维
    # ------------------------------------------------------------------ #
    def stats(self) -> list[dict[str, Any]]:
        manifest: dict[str, Any] = {}
        path = self._manifest_path()
        if path.exists():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}

        out: list[dict[str, Any]] = []
        for collection in self.cfg.all_kb_collections:
            bm25 = self.load_bm25(collection)
            out.append(
                {
                    "collection": collection,
                    "chunks": self.store.count(collection),
                    "bm25": bm25.stats() if bm25.n_docs else None,
                    "chunk_size": self.splitter.chunk_size,
                    "chunk_overlap": self.splitter.chunk_overlap,
                    "embed_dim": self.cfg.rag_embed_dim,
                    **{k: v for k, v in manifest.get(collection, {}).items() if k != "updated_at"},
                    "updated_at": manifest.get(collection, {}).get("updated_at"),
                }
            )
        return out

    def reset(self, collection: str) -> None:
        self.store.drop(collection)
        for path in (self._idf_path(collection), self._bm25_path(collection)):
            if path.exists():
                path.unlink()
        self._invalidate_cache(collection)
        self._write_manifest(collection, documents=0, chunks=0, reset=True)

    # ------------------------------------------------------------------ #
    # 异步包装
    # ------------------------------------------------------------------ #
    async def asearch(self, *args: Any, **kwargs: Any) -> SearchResult:
        return await asyncio.to_thread(self.search, *args, **kwargs)

    async def abuild(self, *args: Any, **kwargs: Any) -> BuildResult:
        return await asyncio.to_thread(self.build, *args, **kwargs)

    async def aadd_documents(self, *args: Any, **kwargs: Any) -> BuildResult:
        return await asyncio.to_thread(self.add_documents, *args, **kwargs)

    async def astats(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.stats)

    async def acount(self, collection: str) -> int:
        """某个集合的文本块数量（门禁判定与前端展示都要用）。"""
        return await asyncio.to_thread(self.store.count, collection)

    async def areset(self, collection: str) -> None:
        await asyncio.to_thread(self.reset, collection)


_kb_singleton: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """进程内单例（Chroma 客户端较重，复用连接与索引）。"""
    global _kb_singleton
    if _kb_singleton is None:
        _kb_singleton = KnowledgeBase()
    return _kb_singleton
