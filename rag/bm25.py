"""轻量 BM25 关键词检索（纯标准库，无第三方依赖）。

为什么需要它？
    哈希 TF-IDF 向量擅长「语义相近」的召回，但对专有名词、术语缩写
    （如 ``GIL``、``Kubernetes``、``STAR``）不够敏感。BM25 作为经典
    概率检索模型，对精确词命中非常强。两者用 RRF 融合，能明显提升
    中文面试题库 / JD 的召回质量。

特征与向量化器保持一致（字符 n-gram + 英文整词），保证两条召回路径
看的是同一套「词」。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .embeddings import _hash_feature, analyze


@dataclass(slots=True)
class BM25Hit:
    """一条 BM25 召回结果。"""

    doc_id: str
    score: float
    # 命中的查询词 IDF 之和。它是一个「绝对信息量」指标：
    # 无关提问通常只能匹配到「你是、什么、怎么」这类高频填充词，
    # IDF 之和很小；相关提问能匹配到「缓存、穿透、GIL」这类低频实词，IDF 之和明显更大。
    matched_idf: float


class BM25Index:
    """BM25（Okapi）实现，使用哈希桶作为词项，避免保存完整词表。"""

    def __init__(
        self,
        dim: int = 4096,
        ngram_min: int = 2,
        ngram_max: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
        state_path: str | Path | None = None,
    ) -> None:
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.k1 = k1
        self.b = b
        self.state_path = Path(state_path) if state_path else None

        self.n_docs: int = 0
        self.avgdl: float = 0.0
        self.doc_ids: list[str] = []
        # doc_id -> {bucket: tf}
        self.term_freqs: dict[str, dict[int, int]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.df: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    def fit(self, doc_ids: Sequence[str], documents: Sequence[str]) -> "BM25Index":
        self.doc_ids = list(doc_ids)
        self.term_freqs = {}
        self.doc_lengths = {}
        self.df = {}

        total_len = 0
        for doc_id, text in zip(doc_ids, documents):
            tf: dict[int, int] = {}
            for feature in analyze(text, self.ngram_min, self.ngram_max):
                bucket = _hash_feature(feature, self.dim)
                tf[bucket] = tf.get(bucket, 0) + 1
            self.term_freqs[doc_id] = tf
            self.doc_lengths[doc_id] = sum(tf.values())
            total_len += self.doc_lengths[doc_id]
            for bucket in tf:
                self.df[bucket] = self.df.get(bucket, 0) + 1

        self.n_docs = len(self.doc_ids)
        self.avgdl = (total_len / self.n_docs) if self.n_docs else 0.0
        return self

    def save(self, path: str | Path | None = None) -> Path | None:
        target = Path(path) if path else self.state_path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "k1": self.k1,
            "b": self.b,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "df": {str(k): v for k, v in self.df.items()},
            "term_freqs": {
                doc_id: {str(k): v for k, v in tf.items()}
                for doc_id, tf in self.term_freqs.items()
            },
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def load(self, path: str | Path | None = None) -> "BM25Index":
        target = Path(path) if path else self.state_path
        if target is None or not target.exists():
            return self
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.dim = int(payload.get("dim", self.dim))
        self.ngram_min = int(payload.get("ngram_min", self.ngram_min))
        self.ngram_max = int(payload.get("ngram_max", self.ngram_max))
        self.k1 = float(payload.get("k1", self.k1))
        self.b = float(payload.get("b", self.b))
        self.n_docs = int(payload.get("n_docs", 0))
        self.avgdl = float(payload.get("avgdl", 0.0))
        self.doc_ids = list(payload.get("doc_ids", []))
        self.doc_lengths = {k: int(v) for k, v in payload.get("doc_lengths", {}).items()}
        self.df = {int(k): int(v) for k, v in payload.get("df", {}).items()}
        raw_tf: dict[str, dict[str, int]] = payload.get("term_freqs", {})
        self.term_freqs = {
            doc_id: {int(k): int(v) for k, v in tf.items()}
            for doc_id, tf in raw_tf.items()
        }
        return self

    # ------------------------------------------------------------------ #
    def _idf(self, bucket: int) -> float:
        df = self.df.get(bucket, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[BM25Hit]:
        """返回按分数降序的召回结果。"""
        if not self.n_docs or not self.avgdl:
            return []

        query_tf: dict[int, int] = {}
        for feature in analyze(query, self.ngram_min, self.ngram_max):
            bucket = _hash_feature(feature, self.dim)
            query_tf[bucket] = query_tf.get(bucket, 0) + 1

        if not query_tf:
            return []

        query_idf = {bucket: self._idf(bucket) for bucket in query_tf}

        scores: dict[str, BM25Hit] = {}
        for doc_id in self.doc_ids:
            tf_map = self.term_freqs.get(doc_id, {})
            if not tf_map:
                continue
            doc_len = self.doc_lengths.get(doc_id, 0)
            score = 0.0
            matched_idf = 0.0
            for bucket, idf in query_idf.items():
                tf = tf_map.get(bucket)
                if not tf:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (tf * (self.k1 + 1.0)) / denom
                matched_idf += idf
            if score > 0:
                scores[doc_id] = BM25Hit(doc_id, score, matched_idf)

        return sorted(scores.values(), key=lambda hit: hit.score, reverse=True)[:top_k]

    def stats(self) -> dict[str, Any]:
        return {
            "n_docs": self.n_docs,
            "avgdl": round(self.avgdl, 2),
            "vocab": len(self.df),
            "k1": self.k1,
            "b": self.b,
        }
