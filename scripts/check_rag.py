"""RAG 知识库自检脚本（不需要大模型 Key）。

    python scripts/check_rag.py

验证内容：
1. 递归切分是否真的递归 + 带重叠
2. 离线向量化是否可用（维度、归一化）
3. Chroma 读写是否正常
4. 种子数据能否建库并检索到期望结果
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SEED_DIR, get_settings  # noqa: E402
from rag import (  # noqa: E402
    KnowledgeBase,
    RecursiveSplitter,
    build_embedder,
    load_seed_documents,
)


def check_splitter() -> None:
    print("=" * 70)
    print("[1] 递归切分 + 重叠")
    text = (
        "第一段：Python 的 GIL 是解释器级别的互斥锁。它限制了多线程的并行能力。\n\n"
        "第二段：IO 密集型任务在等待时会释放 GIL，所以多线程依然有效。"
        "CPU 密集型任务应该使用多进程。\n\n"
        "第三段：asyncio 是单线程事件循环，通过 await 主动让出控制权。"
    )
    splitter = RecursiveSplitter(chunk_size=60, chunk_overlap=20)
    chunks = splitter.split(text)
    print(f"原文 {len(text)} 字 → 切成 {len(chunks)} 块")
    for chunk in chunks:
        print(f"  #{chunk.index} [{chunk.start}:{chunk.end}] len={len(chunk.text)} :: {chunk.text[:40]}...")

    assert len(chunks) > 1, "应该被切成多块"
    # 验证重叠：相邻块在原文中的区间必须相交
    overlapped = all(
        chunks[i].start < chunks[i - 1].end for i in range(1, len(chunks))
    )
    assert overlapped, "相邻块之间必须有重叠"
    print(f"  ✓ 相邻块区间均有重叠，overlap={splitter.chunk_overlap}")


def check_embedding() -> None:
    print("=" * 70)
    print("[2] 离线向量化")
    cfg = get_settings()
    docs = ["Python 的 GIL 是什么", "MySQL 索引失效的场景", "用户运营如何提升留存"]
    embedder = build_embedder().fit(docs)
    vectors = embedder.transform(docs + ["GIL 对多线程有什么影响"])
    assert len(vectors[0]) == cfg.rag_embed_dim, "向量维度不对"
    norm = sum(v * v for v in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-4, f"向量未归一化：{norm}"
    print(f"  ✓ 维度 = {len(vectors[0])}，已 L2 归一化")

    # 相似度应该是：GIL 问题 更接近 第 1 条，而不是第 2/3 条
    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    sim_python = dot(vectors[3], vectors[0])
    sim_mysql = dot(vectors[3], vectors[1])
    sim_ops = dot(vectors[3], vectors[2])
    print(f"  GIL 问句 vs Python 文档 = {sim_python:.4f}")
    print(f"  GIL 问句 vs MySQL 文档  = {sim_mysql:.4f}")
    print(f"  GIL 问句 vs 运营文档    = {sim_ops:.4f}")
    assert sim_python > sim_mysql and sim_python > sim_ops, "相似度排序不符合预期"
    print("  ✓ 相关文档相似度最高")


def check_kb() -> None:
    print("=" * 70)
    print("[3] 种子数据建库 + 混合检索")
    documents = load_seed_documents(SEED_DIR)
    total_docs = sum(len(v) for v in documents.values())
    for name, docs in documents.items():
        print(f"  载入 {name}: {len(docs)} 篇")
    assert total_docs > 0, "种子数据为空"

    kb = KnowledgeBase()
    for collection, docs in documents.items():
        if not docs:
            continue
        result = kb.build(collection, docs, rebuild=True)
        print(
            f"  ✓ {collection}: {result.documents} 篇 → {result.chunks} 块"
            f"（{result.seconds:.2f}s）"
        )

    cases = [
        ("Python GIL 多线程 多进程", "interview_questions"),
        ("缓存穿透 缓存雪崩 怎么解决", "interview_questions"),
        ("如何提升用户留存", "interview_questions"),
        ("杭州 Python 后端 薪资", "job_descriptions"),
        ("大模型应用开发工程师 岗位要求", "job_descriptions"),
    ]
    for query, collection in cases:
        result = kb.search(query, collections=[collection], top_k=3)
        print(f"\n  查询：{query}  →  {len(result.chunks)} 条（{result.elapsed_ms:.1f}ms，"
              f"vector={result.dense_hits} bm25={result.bm25_hits}）")
        assert result.chunks, f"检索不到结果：{query}"
        for chunk in result.chunks:
            sim = "-" if chunk.similarity is None else f"{chunk.similarity:.3f}"
            print(f"    [{chunk.source:6}] sim={sim:>5} · {chunk.citation_label}")
            print(f"        {chunk.text[:70].replace(chr(10), ' ')}...")

    print("\n" + "=" * 70)
    print("[4] 知识库统计")
    for item in kb.stats():
        print(f"  {item}")


if __name__ == "__main__":
    check_splitter()
    check_embedding()
    check_kb()
    print("\n✅ RAG 自检全部通过")
