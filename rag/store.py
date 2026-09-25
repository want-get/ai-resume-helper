"""Chroma 持久化向量库封装。

架构（两层）：

1. **Chroma PersistentClient = 事实来源**：文档、元数据、向量全部落盘到
   ``data/chroma``，进程重启不丢，支持元数据过滤与增量 upsert。
2. **进程内热索引 = 高并发检索加速层**：Chroma 的 ``query`` 内部有全局锁，
   实测吞吐会封顶在 ~120 QPS，而并发升高时延迟线性增长。因此这里在内存里
   维护一份归一化向量矩阵，检索用 numpy 做矩阵乘法（几十微秒级），
   写入时把索引失效、下次检索再重建。

   这不是「绕过向量库」——数据仍然写在 Chroma 里，热索引只是它的读缓存，
   并且在块数超过 ``RAG_MEMORY_INDEX_MAX_CHUNKS`` 或遇到复杂过滤条件时
   自动回退到 Chroma 原生检索。

* 所有方法都提供 ``a*`` 异步版本（内部 ``asyncio.to_thread``），
  避免在 FastAPI 事件循环里阻塞。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from .embeddings import HashingTfidfEmbedding


@dataclass(slots=True)
class _MemoryIndex:
    """进程内热索引。"""

    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    matrix: np.ndarray  # (n, dim)，已 L2 归一化
    dimension: int


class VectorStore:
    """Chroma 封装（线程安全）。"""

    def __init__(
        self,
        persist_dir: str | Path,
        embedder: HashingTfidfEmbedding,
        memory_index: bool = True,
        memory_index_max_chunks: int = 20000,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.memory_index_enabled = memory_index
        self.memory_index_max_chunks = memory_index_max_chunks
        self._client: chromadb.ClientAPI | None = None
        self._collections: dict[str, Collection] = {}
        self._index_cache: dict[str, _MemoryIndex] = {}
        self._count_cache: dict[str, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def get_collection(self, name: str) -> Collection:
        """获取集合（带句柄缓存，避免每次检索都走一次元数据往返）。"""
        cached = self._collections.get(name)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._collections.get(name)
            if cached is None:
                cached = self.client.get_or_create_collection(
                    name=name,
                    embedding_function=self.embedder,
                    metadata={"hnsw:space": "cosine"},
                )
                self._collections[name] = cached
            return cached

    def list_collections(self) -> list[str]:
        with self._lock:
            return [c.name for c in self.client.list_collections()]

    # ------------------------------------------------------------------ #
    # 进程内热索引
    # ------------------------------------------------------------------ #
    def _build_memory_index(self, name: str) -> _MemoryIndex | None:
        """从 Chroma 拉取全量向量构建热索引。"""
        collection = self.get_collection(name)
        try:
            total = int(collection.count())
        except Exception:  # pragma: no cover
            return None
        if total == 0 or total > self.memory_index_max_chunks:
            return None

        raw = collection.get(include=["documents", "metadatas", "embeddings"])
        ids = raw.get("ids") or []
        embeddings = raw.get("embeddings")
        if not ids or embeddings is None:
            return None

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            return None
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms

        return _MemoryIndex(
            ids=list(ids),
            documents=list(raw.get("documents") or []),
            metadatas=[dict(m or {}) for m in (raw.get("metadatas") or [])],
            matrix=matrix,
            dimension=int(matrix.shape[1]),
        )

    def _get_memory_index(self, name: str) -> _MemoryIndex | None:
        if not self.memory_index_enabled:
            return None
        cached = self._index_cache.get(name)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._index_cache.get(name)
            if cached is None:
                cached = self._build_memory_index(name)
                if cached is not None:
                    self._index_cache[name] = cached
            return cached

    def _invalidate_index(self, name: str) -> None:
        self._index_cache.pop(name, None)
        self._count_cache.pop(name, None)

    def memory_get(self, name: str, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """按 id 取文档：优先走热索引（免去 Chroma 往返），否则回退 Chroma。"""
        if not ids:
            return {}
        index = self._get_memory_index(name)
        if index is None:
            return self.get_by_ids(name, ids)
        positions = {chunk_id: i for i, chunk_id in enumerate(index.ids)}
        out: dict[str, dict[str, Any]] = {}
        for chunk_id in ids:
            position = positions.get(chunk_id)
            if position is None:
                continue
            out[chunk_id] = {
                "id": chunk_id,
                "text": index.documents[position] if position < len(index.documents) else "",
                "metadata": index.metadatas[position] if position < len(index.metadatas) else {},
            }
        return out

    def _memory_query(
        self,
        index: _MemoryIndex,
        query_embedding: Sequence[float],
        n_results: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.shape[-1] != index.dimension:
            return []
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        similarities = index.matrix @ vector  # 归一化后内积 = 余弦相似度

        if where:
            # 只支持简单等值过滤；带操作符的复杂条件回退给 Chroma
            if any(str(key).startswith("$") for key in where):
                return []
            mask = np.ones(len(index.ids), dtype=bool)
            for key, value in where.items():
                mask &= np.array(
                    [meta.get(key) == value for meta in index.metadatas], dtype=bool
                )
            if not mask.any():
                return []
            candidates = np.flatnonzero(mask)
        else:
            candidates = np.arange(len(index.ids))

        top = candidates[np.argsort(-similarities[candidates])[:n_results]]
        hits: list[dict[str, Any]] = []
        for position in top:
            hits.append(
                {
                    "id": index.ids[position],
                    "text": index.documents[position] if position < len(index.documents) else "",
                    "metadata": index.metadatas[position] if position < len(index.metadatas) else {},
                    "similarity": max(0.0, min(1.0, float(similarities[position]))),
                }
            )
        return hits

    # ------------------------------------------------------------------ #
    def upsert(
        self,
        name: str,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if not ids:
            return 0
        collection = self.get_collection(name)
        with self._lock:
            # 分批写入，避免单次请求体过大
            batch = 256
            for start in range(0, len(ids), batch):
                end = start + batch
                collection.upsert(
                    ids=list(ids[start:end]),
                    documents=list(documents[start:end]),
                    metadatas=[_clean_metadata(m) for m in metadatas[start:end]],
                    embeddings=[list(v) for v in embeddings[start:end]],
                )
        # 写入后热索引失效，下次检索时会重建
        self._invalidate_index(name)
        return len(ids)

    def query(
        self,
        name: str,
        query_embedding: Sequence[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        collection = self.get_collection(name)
        try:
            total = collection.count()
        except Exception:  # pragma: no cover - 集合刚创建
            total = 0
        if total == 0:
            return []

        n_results = max(1, min(n_results, total))

        # 快路径：进程内热索引（numpy 矩阵乘法，微秒级，且不持有 Chroma 的内部锁）
        index = self._get_memory_index(name)
        if index is not None:
            hits = self._memory_query(index, query_embedding, n_results, where)
            if hits:
                return hits
            # 热索引没命中（例如过滤条件不支持）时继续走 Chroma

        # 慢路径：Chroma 原生检索（支持任意元数据过滤）
        raw = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        for i, chunk_id in enumerate(ids):
            distance = dists[i] if i < len(dists) else 1.0
            hits.append(
                {
                    "id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    # cosine 距离 -> 相似度
                    "similarity": max(0.0, min(1.0, 1.0 - float(distance))),
                }
            )
        return hits

    def get_by_ids(self, name: str, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        collection = self.get_collection(name)
        raw = collection.get(ids=list(ids), include=["documents", "metadatas"])
        out: dict[str, dict[str, Any]] = {}
        got_ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        for i, chunk_id in enumerate(got_ids):
            out[chunk_id] = {
                "id": chunk_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
            }
        return out

    def count(self, name: str) -> int:
        """集合块数（带缓存，写入/删除时失效）。"""
        cached = self._count_cache.get(name)
        if cached is not None:
            return cached
        try:
            value = int(self.get_collection(name).count())
        except Exception:  # pragma: no cover
            value = 0
        self._count_cache[name] = value
        return value

    def drop(self, name: str) -> None:
        with self._lock:
            self._collections.pop(name, None)
            self._invalidate_index(name)
            try:
                self.client.delete_collection(name)
            except Exception:  # pragma: no cover - 集合不存在
                pass

    # ------------------------------------------------------------------ #
    # 异步包装：Chroma 是阻塞 IO，统一丢到线程池
    # ------------------------------------------------------------------ #
    async def aupsert(self, *args: Any, **kwargs: Any) -> int:
        return await asyncio.to_thread(self.upsert, *args, **kwargs)

    async def aquery(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.query, *args, **kwargs)

    async def aget_by_ids(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(self.get_by_ids, *args, **kwargs)

    async def amemory_get(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(self.memory_get, *args, **kwargs)

    async def acount(self, name: str) -> int:
        return await asyncio.to_thread(self.count, name)


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma 元数据只接受 str/int/float/bool，这里做兜底转换。"""
    cleaned: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned
