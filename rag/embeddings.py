"""离线向量化：字符 n-gram 哈希 TF-IDF。

为什么不用在线 Embedding？
    DeepSeek 官方只提供对话模型，不提供 embedding 接口；Chroma 自带的 ONNX
    MiniLM 需要联网下载模型且以英文为主。为了让项目「克隆下来就能跑」，
    这里实现一个完全离线的中文友好向量化器：

    * **特征**：字符 n-gram（默认 2~3 元）——中文没有分词空格，
      字符 n-gram 对中文关键词/短语匹配非常有效；英文单词额外整词入袋。
    * **权重**：TF-IDF。TF 用次线性缩放 ``1 + log(tf)``，IDF 用
      ``log((1 + N) / (1 + df)) + 1``（平滑，避免除零）。
    * **哈希技巧**：特征名经 blake2b 稳定哈希映射到固定维度，因此
      **无需保存词表**，增量入库时向量空间保持一致。
    * **归一化**：L2 归一化后，内积等价于余弦相似度，可直接交给 Chroma。

IDF 统计量（每个哈希桶的文档频率）会持久化到知识库目录，
查询时读取同一份统计量，保证入库与检索在同一个向量空间。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from chromadb.utils.embedding_functions import register_embedding_function

# 中文标点/空白：切特征时统一丢弃
_PUNCT_RE = re.compile(
    r"[\s\u3000!-/:-@\[-`{-~·—…“”‘’《》〈〉【】（）〔〕「」『』、。！？；：，]+"
)
_ASCII_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#._-]{1,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（DeepSeek 中文约 1 token ≈ 1.5 字）。"""
    if not text:
        return 0
    from config import get_settings

    return max(1, int(len(text) * get_settings().tokens_per_char))


def normalize_text(text: str) -> str:
    """归一化：全角转半角、小写、压缩空白。"""
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            ch = " "
        elif 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII
            ch = chr(code - 0xFEE0)
        out.append(ch)
    return " ".join("".join(out).lower().split())


def analyze(text: str, ngram_min: int = 2, ngram_max: int = 3) -> list[str]:
    """把文本切成用于匹配的特征集合。

    包含三类特征：
    * 英文/数字整词（小写）
    * 中文/混合字符 n-gram（跨标点会被截断，保留语义连续性）
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    features: list[str] = []

    # 1) 英文单词整词
    for match in _ASCII_WORD_RE.finditer(normalized):
        features.append(f"w:{match.group(0)}")

    # 2) 字符 n-gram：以标点/空白为界切成 run，再在 run 内部滑窗
    for run in _PUNCT_RE.split(normalized):
        if not run:
            continue
        length = len(run)
        if length == 1:
            features.append(f"c:{run}")
            continue
        for n in range(ngram_min, ngram_max + 1):
            if n > length:
                break
            for i in range(length - n + 1):
                features.append(f"c:{run[i:i + n]}")
    return features


def _hash_feature(feature: str, dim: int) -> int:
    """稳定哈希到 [0, dim)。用 blake2b 而非内置 hash()，保证跨进程一致。"""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


@register_embedding_function
class HashingTfidfEmbedding:
    """哈希 TF-IDF 向量化器，可持久化 IDF 统计量。

    实现了 Chroma 的 ``EmbeddingFunction`` 协议（``__call__`` / ``name`` /
    ``get_config`` / ``build_from_config``），因此可以直接注册进 Chroma，
    重建集合时能自动从 ``state_path`` 恢复 IDF 统计量。
    """

    def __init__(
        self,
        dim: int = 4096,
        ngram_min: int = 2,
        ngram_max: int = 3,
        state_path: str | Path | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.state_path = Path(state_path) if state_path else None

        self.n_docs: int = 0
        # 稀疏存储：桶 -> 包含该桶的文档数
        self.df: dict[int, int] = {}
        self._idf_cache: np.ndarray | None = None
        self._unknown_idf: float = math.log(2.0) + 1.0  # 未见过特征的 IDF 上限

    # ------------------------------------------------------------------ #
    # 拟合 / 持久化
    # ------------------------------------------------------------------ #
    def fit(self, documents: Sequence[str]) -> "HashingTfidfEmbedding":
        """统计文档频率。语料越大，IDF 越可靠。"""
        df: dict[int, int] = {}
        n_docs = 0
        for doc in documents:
            if not doc or not doc.strip():
                continue
            n_docs += 1
            buckets = {_hash_feature(f, self.dim) for f in analyze(doc, self.ngram_min, self.ngram_max)}
            for bucket in buckets:
                df[bucket] = df.get(bucket, 0) + 1

        self.n_docs = n_docs
        self.df = df
        self._idf_cache = None
        self._unknown_idf = math.log((1 + max(n_docs, 1)) / 1.0) + 1.0
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
            "n_docs": self.n_docs,
            "df": {str(k): v for k, v in self.df.items()},
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return target

    def load(self, path: str | Path | None = None) -> "HashingTfidfEmbedding":
        target = Path(path) if path else self.state_path
        if target is None or not target.exists():
            return self
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.dim = int(payload.get("dim", self.dim))
        self.ngram_min = int(payload.get("ngram_min", self.ngram_min))
        self.ngram_max = int(payload.get("ngram_max", self.ngram_max))
        self.n_docs = int(payload.get("n_docs", 0))
        self.df = {int(k): int(v) for k, v in payload.get("df", {}).items()}
        self._idf_cache = None
        self._unknown_idf = math.log((1 + max(self.n_docs, 1)) / 1.0) + 1.0
        return self

    # ------------------------------------------------------------------ #
    # 变换
    # ------------------------------------------------------------------ #
    def _idf_vector(self) -> np.ndarray:
        if self._idf_cache is None:
            vec = np.full(self.dim, self._unknown_idf, dtype=np.float32)
            if self.df:
                buckets = np.fromiter(self.df.keys(), dtype=np.int64, count=len(self.df))
                counts = np.fromiter(self.df.values(), dtype=np.float32, count=len(self.df))
                vec[buckets] = np.log((1.0 + self.n_docs) / (1.0 + counts)) + 1.0
            self._idf_cache = vec
        return self._idf_cache

    def transform(self, texts: Sequence[str]) -> list[list[float]]:
        """把文本列表转成稠密向量（已 L2 归一化）。"""
        idf = self._idf_vector()
        vectors: list[list[float]] = []
        for text in texts:
            tf: dict[int, float] = {}
            for feature in analyze(text, self.ngram_min, self.ngram_max):
                bucket = _hash_feature(feature, self.dim)
                tf[bucket] = tf.get(bucket, 0.0) + 1.0

            vec = np.zeros(self.dim, dtype=np.float32)
            if tf:
                buckets = np.fromiter(tf.keys(), dtype=np.int64, count=len(tf))
                counts = np.fromiter(tf.values(), dtype=np.float32, count=len(tf))
                # 次线性 TF：1 + log(tf)
                weights = (1.0 + np.log(counts)) * idf[buckets]
                vec[buckets] = weights
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec /= norm
            vectors.append(vec.tolist())
        return vectors

    def transform_one(self, text: str) -> list[float]:
        return self.transform([text])[0]

    # ------------------------------------------------------------------ #
    # Chroma EmbeddingFunction 协议
    # ------------------------------------------------------------------ #
    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        return self.transform(input)

    @staticmethod
    def name() -> str:
        return "hashing-tfidf"

    def get_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "state_path": str(self.state_path) if self.state_path else "",
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashingTfidfEmbedding":
        state_path = config.get("state_path") or None
        embedder = HashingTfidfEmbedding(
            dim=int(config.get("dim", 4096)),
            ngram_min=int(config.get("ngram_min", 2)),
            ngram_max=int(config.get("ngram_max", 3)),
            state_path=state_path,
        )
        embedder.load()
        return embedder

    def validate_config(self, config: dict[str, Any]) -> None:  # pragma: no cover
        if not isinstance(config, dict):
            raise ValueError("embedding function config 必须是 dict")


def build_embedder(state_path: str | Path | None = None) -> HashingTfidfEmbedding:
    """按全局配置构造向量化器。"""
    from config import get_settings

    cfg = get_settings()
    return HashingTfidfEmbedding(
        dim=cfg.rag_embed_dim,
        ngram_min=cfg.rag_ngram_min,
        ngram_max=cfg.rag_ngram_max,
        state_path=state_path,
    )
