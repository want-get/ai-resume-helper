"""检索质量评测：recall@k / precision@k / P@1 / MRR。

## 这个脚本解决什么问题

「检索好不好」不能靠感觉。改一个阈值、换个切分大小、调 top_k，
到底是变好还是变坏，必须有**可量化的指标**，否则调参就是拍脑袋。

## 标签是怎么来的（关键，别跳过）

``LABELED_CASES`` 的期望命中是**人工阅读 ``data/seed/`` 语料正文**后确定的：

    先跑 ``--list`` 导出全部 chunk → 人工读内容 → 再写标签。

**不是**「先跑一次检索、把返回结果当正确答案」。为什么强调这点：

    拿检索结果反推标签，等于用同一个系统给自己判卷，
    recall 必然接近 100%，**指标看起来很美但零信息量**。
    真实召回率低恰恰是因为「该找到的没找到」，而这只有人工读语料才能发现。

## 指标口径（为什么同时算四个）

单个查询通常只有 1 篇真正相关的文档，此时 **precision@k 的数学上限就是 1/k**
（top_k=5 时最高只有 20%），单独看它会严重低估检索质量。所以本脚本同时给出：

    recall@k    该找到的，找到了多少比例          —— 召回
    precision@k 找回来的，有多少是相关的          —— 准确（受 1/k 上限约束，仅作参考）
    P@1         第一条就是正确答案的比例          —— 最贴近用户体感的「准确」
    MRR         第一条正确答案排名的倒数的均值    —— 排序质量

## 语料范围

只评**公共知识库**（``interview_questions`` + ``job_descriptions``）。
``personal_kb`` 是用户专属数据、会随使用变化，纳入评测会导致结果不可复现。

## 用法

    python scripts/check_retrieval_quality.py --list      # 导出语料，供人工标注
    python scripts/check_retrieval_quality.py             # 跑评测
    python scripts/check_retrieval_quality.py --k 3
    python scripts/check_retrieval_quality.py --show-miss  # 只看失败用例
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ====================================================================== #
# 人工标注的评测集（24 条）
#
# 标注依据：逐条读过 data/seed/*.jsonl 渲染后的正文，确认该记录**确实回答**了这个问题。
# kind 说明：
#   direct    提问用词与文档接近（考「能不能找到」）
#   paraphrase 换了说法（考语义泛化——这正是离线 TF-IDF 的弱项）
#   job       岗位库定向查询（考职位/城市/薪资的结构化匹配）
# ====================================================================== #
LABELED_CASES: list[dict[str, object]] = [
    # ---------------- 面试题库：用词接近 ----------------
    {"kind": "direct", "query": "Python 的 GIL 是什么，对多线程有什么影响",
     "expected": ["iq-tech-python-001"]},
    {"kind": "direct", "query": "asyncio 事件循环是怎么工作的",
     "expected": ["iq-tech-python-002"]},
    {"kind": "direct", "query": "为什么不能用可变对象做函数默认参数",
     "expected": ["iq-tech-python-003"]},
    {"kind": "direct", "query": "装饰器怎么实现的，闭包有什么用",
     "expected": ["iq-tech-python-004"]},
    {"kind": "direct", "query": "Python 内存管理和垃圾回收机制",
     "expected": ["iq-tech-python-005"]},
    {"kind": "direct", "query": "生成器和迭代器有什么区别",
     "expected": ["iq-tech-python-006"]},
    {"kind": "direct", "query": "FastAPI 里 def 和 async def 路由有什么区别",
     "expected": ["iq-tech-framework-007"]},
    {"kind": "direct", "query": "什么情况下 MySQL 索引会失效",
     "expected": ["iq-tech-db-008"]},
    {"kind": "direct", "query": "事务隔离级别有哪些，脏读和幻读是什么",
     "expected": ["iq-tech-db-009"]},
    {"kind": "direct", "query": "缓存穿透、缓存击穿、缓存雪崩分别怎么解决",
     "expected": ["iq-tech-db-010"]},
    {"kind": "direct", "query": "RAG 完整链路是什么，为什么能降低幻觉",
     "expected": ["iq-tech-llm-013"]},
    {"kind": "direct", "query": "Function Calling 的原理，模型真的会执行函数吗",
     "expected": ["iq-tech-llm-014"]},
    {"kind": "direct", "query": "多轮对话超出上下文窗口怎么处理",
     "expected": ["iq-tech-llm-015"]},
    {"kind": "direct", "query": "线上故障排查的思路",
     "expected": ["iq-tech-project-017"]},

    # ---------------- 面试题库：换说法（考语义泛化） ----------------
    {"kind": "paraphrase", "query": "Python 多线程为什么不能真正并行执行",
     "expected": ["iq-tech-python-001"]},
    {"kind": "paraphrase", "query": "接口响应突然变慢，从数据库层面怎么查原因",
     "expected": ["iq-tech-db-008"]},
    {"kind": "paraphrase", "query": "热点数据集中失效，请求瞬间把数据库打爆了怎么办",
     "expected": ["iq-tech-db-010"]},
    {"kind": "paraphrase", "query": "怎么让大模型去调外部接口拿实时数据",
     "expected": ["iq-tech-llm-014"]},
    {"kind": "paraphrase", "query": "对话历史太长塞不进模型了怎么办",
     "expected": ["iq-tech-llm-015"]},

    # ---------------- 岗位 JD 库：职位 + 城市 ----------------
    {"kind": "job", "query": "杭州 Python 后端开发工程师的薪资和要求",
     "expected": ["jd-tech-001"]},
    {"kind": "job", "query": "北京大模型应用开发工程师需要什么技能",
     "expected": ["jd-tech-002"]},
    {"kind": "job", "query": "深圳前端开发工程师的薪资范围",
     "expected": ["jd-tech-003"]},
    {"kind": "job", "query": "成都数据分析师的岗位职责和薪资",
     "expected": ["jd-tech-005"]},
    {"kind": "job", "query": "上海 Java 后端开发工程师的薪资",
     # 注意：Java 后端是 jd-tech-008，jd-tech-007 是「运维开发工程师（SRE）」。
     # 这里最初标成了 jd-tech-007（记错了），跑出来显示"未命中"，
     # 逐一核对语料后确认是**标注错了、检索是对的**。
     # 这件事说明：标签必须对着语料核，凭印象写必然出错。
     "expected": ["jd-tech-008"]},
]


# ====================================================================== #
async def list_corpus() -> int:
    """导出全部 chunk（含 record_id 与正文），供人工标注。"""
    from rag import get_knowledge_base

    kb = get_knowledge_base()
    stats = await kb.astats()
    out: list[str] = []
    total = 0

    for item in stats:
        collection = item["collection"]
        out.append("=" * 88)
        out.append(f"集合：{collection}（{item.get('chunks', 0)} 块）")
        out.append("=" * 88)

        raw = kb.store.get_collection(collection).get(include=["documents", "metadatas"])
        for chunk_id, text, meta in zip(
            raw.get("ids") or [], raw.get("documents") or [], raw.get("metadatas") or []
        ):
            total += 1
            meta = meta or {}
            out.append(
                f"\n[chunk] {chunk_id}"
                f"\n  record_id : {meta.get('record_id', '(无)')}"
                f"\n  title     : {meta.get('title', '')}"
                f"\n  来源      : {meta.get('source', '')} | 类别: {meta.get('category', '')}"
                f" | 城市: {meta.get('city', '')} | 薪资: {meta.get('salary', '')}"
            )
            out.append("  正文      :\n    " + str(text).replace("\n", "\n    "))
        out.append("")

    dest = ROOT / "scripts" / "_corpus_dump.txt"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"共导出 {total} 个 chunk -> {dest.name}")
    print("请人工阅读该文件后，更新本脚本顶部的 LABELED_CASES。")
    return 0


async def run_eval(top_k: int, only_miss: bool) -> int:
    from config import get_settings
    from rag import get_knowledge_base

    if not LABELED_CASES:
        print("LABELED_CASES 为空，先运行 --list 完成标注。")
        return 1

    cfg = get_settings()
    # 只评公共知识库：personal_kb 随用户数据变化，纳入会导致结果不可复现
    public_collections = [
        name for name in cfg.kb_collections if name != cfg.rag_collection_personal
    ]

    kb = get_knowledge_base()
    recalls: list[float] = []
    precisions: list[float] = []
    top1_hits = 0
    reciprocal_ranks: list[float] = []
    kinds: dict[str, list[float]] = {}
    rows: list[dict[str, object]] = []

    for case in LABELED_CASES:
        query = str(case["query"])
        expected = set(case["expected"])  # type: ignore[arg-type]
        kind = str(case.get("kind", "direct"))

        result = await kb.asearch(query, collections=public_collections, top_k=top_k)

        got: list[str] = []
        for chunk in result.chunks:
            rid = str((chunk.metadata or {}).get("record_id", ""))
            if rid and rid not in got:
                got.append(rid)  # 去重但保序（同一文档可能切出多块）

        hit = len(set(got) & expected)
        recall = hit / len(expected) if expected else 0.0
        precision = hit / len(got) if got else 0.0

        # 首个正确命中的排名
        rank = 0
        for index, rid in enumerate(got, 1):
            if rid in expected:
                rank = index
                break
        if rank == 1:
            top1_hits += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        recalls.append(recall)
        precisions.append(precision)
        kinds.setdefault(kind, []).append(recall)
        rows.append(
            {"query": query, "kind": kind, "recall": recall, "precision": precision,
             "expected": sorted(expected), "got": got, "rank": rank}
        )

    n = len(rows)
    mean_recall = sum(recalls) / n
    mean_precision = sum(precisions) / n
    p_at_1 = top1_hits / n
    mrr = sum(reciprocal_ranks) / n

    print("=" * 92)
    print(f"检索质量评测  |  top_k={top_k}  用例={n}  语料={public_collections}")
    print("=" * 92)

    for row in rows:
        if only_miss and row["recall"] >= 1.0:  # type: ignore[operator]
            continue
        flag = "✅" if row["recall"] >= 1.0 else "❌"  # type: ignore[operator]
        print(f"\n{flag} [{row['kind']}] {row['query']}")
        print(f"     期望 : {row['expected']}")
        print(f"     实际 : {row['got']}")
        print(
            f"     recall={row['recall']:.0%}  precision={row['precision']:.0%}"  # type: ignore[str-format]
            f"  首个命中排名={row['rank'] or '未命中'}"
        )

    print("\n" + "-" * 92)
    print(f"  recall@{top_k}     = {mean_recall:6.1%}   该找到的，找到了多少")
    print(f"  precision@{top_k}  = {mean_precision:6.1%}   找回来的有多少相关（单答案查询上限仅 1/k）")
    print(f"  P@1           = {p_at_1:6.1%}   第一条就正确的比例")
    print(f"  MRR           = {mrr:6.3f}   排序质量（1=永远第一条命中）")

    print("\n  按用例类型拆解 recall：")
    for kind, values in sorted(kinds.items()):
        print(f"    {kind:<12} n={len(values):<3} recall={sum(values)/len(values):.1%}")

    print("-" * 92)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--list", action="store_true", help="导出语料供人工标注")
    parser.add_argument("--k", type=int, default=5, help="top_k，默认 5")
    parser.add_argument("--show-miss", action="store_true", help="只打印未完全命中的用例")
    args = parser.parse_args()

    if args.list:
        return asyncio.run(list_corpus())
    return asyncio.run(run_eval(args.k, args.show_miss))


if __name__ == "__main__":
    raise SystemExit(main())
