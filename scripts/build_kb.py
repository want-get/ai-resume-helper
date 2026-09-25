"""构建 / 重建 RAG 知识库。

    python scripts/build_kb.py                    # 用 data/seed 重建全部集合
    python scripts/build_kb.py --stats            # 只看当前知识库状态
    python scripts/build_kb.py --add docs/xx.md --collection job_descriptions
    python scripts/build_kb.py --reset            # 删除全部集合

知识库落地在 ``data/chroma``（Chroma 持久化目录），
切分参数与向量维度来自 ``.env``（RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP / RAG_EMBED_DIM）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SEED_DIR, get_settings  # noqa: E402
from rag import Document, get_knowledge_base, load_any, load_seed_documents  # noqa: E402


async def show_stats(kb) -> None:
    print("=" * 78)
    for item in await kb.astats():
        print(f"集合 {item['collection']}")
        print(f"  块数　　　: {item['chunks']}")
        print(f"  切分参数　: chunk_size={item['chunk_size']} overlap={item['chunk_overlap']}")
        print(f"  向量维度　: {item['embed_dim']}")
        if item.get("bm25"):
            print(f"  BM25 索引 : {item['bm25']}")
        if item.get("avg_chunk_chars"):
            print(f"  平均块长　: {item['avg_chunk_chars']} 字")
        print(f"  最近更新　: {item.get('updated_at')}")
    print("=" * 78)


async def main(args: argparse.Namespace) -> None:
    settings = get_settings()
    kb = get_knowledge_base()

    print(f"知识库目录：{settings.chroma_dir}")
    print(f"切分配置　：chunk_size={settings.rag_chunk_size} "
          f"overlap={settings.rag_chunk_overlap} dim={settings.rag_embed_dim}")

    if args.reset:
        for collection in settings.kb_collections:
            await kb.areset(collection)
            print(f"已删除集合 {collection}")
        await show_stats(kb)
        return

    if args.stats:
        await show_stats(kb)
        return

    if args.add:
        documents: list[Document] = []
        for raw in args.add:
            path = Path(raw)
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                print(f"❌ 文件不存在：{path}")
                continue
            loaded = load_any(path, {"source": path.name})
            documents.extend(loaded)
            print(f"载入 {path.name}：{len(loaded)} 篇")
        if not documents:
            print("没有可入库的文档")
            return
        collection = args.collection or settings.rag_collection_jobs
        result = await kb.aadd_documents(collection, documents)
        print(f"✅ 增量入库 {collection}：{result.documents} 篇 → {result.chunks} 块"
              f"（{result.seconds:.2f}s）")
        await show_stats(kb)
        return

    documents = load_seed_documents(SEED_DIR)
    total = sum(len(v) for v in documents.values())
    if total == 0:
        print("❌ data/seed 下没有种子数据")
        return
    for collection, docs in documents.items():
        if not docs:
            continue
        result = await kb.abuild(collection, docs, rebuild=True)
        print(f"✅ {collection}：{result.documents} 篇 → {result.chunks} 块"
              f"（{result.seconds:.2f}s）")
    await show_stats(kb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 RAG 知识库")
    parser.add_argument("--stats", action="store_true", help="只查看状态")
    parser.add_argument("--reset", action="store_true", help="删除全部集合")
    parser.add_argument("--add", nargs="+", help="增量入库的文件路径")
    parser.add_argument(
        "--collection",
        default=None,
        choices=["interview_questions", "job_descriptions"],
        help="增量入库的目标集合",
    )
    asyncio.run(main(parser.parse_args()))
