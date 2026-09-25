"""用户专属知识库。

需求：**模拟面试的题目要结合「这个用户 + 这个岗位」的知识库出题**，
而不是拿一份通用题库糊弄。

知识库的素材（全部来自用户自己的数据）：

    1. 个人简历            —— 切成块，让模型记住他做过什么
    2. 目标岗位 JD          —— 用户从抓取结果里选中的那个岗位
    3. 同类岗位 JD（N 条）   —— 同城市/同方向抓到的其他岗位要求，用于横向对比
    4. 通用题库里的相关面试题 —— 从内置题库按「简历 + 目标岗位」检索出来的同类题

最终落到 Chroma 的一个独立集合 ``personal_kb``，与公共题库物理隔离：
重建个人库不会动公共题，删掉个人库也不影响通用问答。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config import get_settings
from rag import Document, get_knowledge_base
from rag.embeddings import estimate_tokens

from .db import repository, session_scope
from .profile import UserProfileData

logger = logging.getLogger("ai_resume_helper.personal_kb")


@dataclass
class PersonalKbReport:
    """建档结果（如实上报每类素材的数量）。"""

    ok: bool = False
    chunks: int = 0
    documents: int = 0
    resume_chunks: int = 0
    target_job_chunks: int = 0
    similar_job_chunks: int = 0
    question_chunks: int = 0
    seconds: float = 0.0
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "chunks": self.chunks,
            "documents": self.documents,
            "resume_chunks": self.resume_chunks,
            "target_job_chunks": self.target_job_chunks,
            "similar_job_chunks": self.similar_job_chunks,
            "question_chunks": self.question_chunks,
            "seconds": round(self.seconds, 2),
            "error": self.error,
            "notes": self.notes,
        }


def _render_job(job: dict[str, Any], label: str) -> str:
    lines = [f"【{label}】{job.get('title', '')}"]
    if job.get("company"):
        lines.append(f"公司：{job['company']}")
    if job.get("city"):
        lines.append(f"城市：{job['city']}")
    if job.get("salary_text"):
        lines.append(f"薪资：{job['salary_text']}")
    if job.get("tags"):
        lines.append("技能标签：" + "、".join(str(t) for t in job["tags"][:15]))
    if job.get("jd_text"):
        lines.append("岗位描述：")
        lines.append(str(job["jd_text"])[:3000])
    return "\n".join(lines)


async def build_personal_kb(
    profile: UserProfileData,
    *,
    similar_job_limit: int = 12,
    question_limit: int = 12,
    rebuild: bool = True,
) -> PersonalKbReport:
    """按用户档案重建专属知识库。"""
    started = time.perf_counter()
    settings = get_settings()
    collection = settings.rag_collection_personal
    report = PersonalKbReport()

    documents: list[Document] = []

    # ---- 1) 简历 ----
    resume_text = ""
    if profile.resume_id:
        async with session_scope() as db:
            resume = await repository.get_resume(db, profile.resume_id)
        if resume is not None:
            resume_text = resume.content

    if resume_text:
        documents.append(
            Document(
                text=resume_text,
                metadata={
                    "doc_type": "resume",
                    "source": "个人简历",
                    "title": "我的简历",
                    "role_type": "personal",
                },
            )
        )
        report.resume_chunks = 1
    else:
        report.notes.append("没有找到简历，个人知识库不含简历内容")

    # ---- 2) 目标岗位 + 3) 同类岗位 ----
    target_job = profile.target_job or {}
    if target_job.get("title"):
        documents.append(
            Document(
                text=_render_job(target_job, "目标岗位"),
                metadata={
                    "doc_type": "target_job",
                    "source": "目标岗位 JD",
                    "title": str(target_job.get("title"))[:80],
                    "company": str(target_job.get("company") or ""),
                    "city": str(target_job.get("city") or ""),
                    "role_type": "personal",
                },
            )
        )
        report.target_job_chunks = 1

        similar = await _load_similar_jobs(profile, target_job, similar_job_limit)
        for index, job in enumerate(similar):
            documents.append(
                Document(
                    text=_render_job(job, "同类岗位"),
                    metadata={
                        "doc_type": "similar_job",
                        "source": "同类岗位 JD",
                        "title": str(job.get("title") or "")[:80],
                        "company": str(job.get("company") or ""),
                        "city": str(job.get("city") or ""),
                        "salary": str(job.get("salary_text") or ""),
                        "chunk_index": index,
                        "role_type": "personal",
                    },
                )
            )
            report.similar_job_chunks += 1
        if not similar:
            report.notes.append("没有抓到同类岗位，建议先执行一次岗位抓取")
    else:
        report.notes.append("还没有选定目标岗位，个人知识库不含岗位 JD")

    # ---- 4) 通用题库里与该简历/岗位相关的面试题 ----
    questions = await _retrieve_questions(resume_text, target_job, question_limit)
    for index, chunk in enumerate(questions):
        documents.append(
            Document(
                text=chunk["text"],
                metadata={
                    "doc_type": "interview_question",
                    "source": "面试题库",
                    "title": str(chunk.get("title") or "")[:80],
                    "chunk_index": index,
                    "role_type": "personal",
                },
            )
        )
        report.question_chunks += 1
    if not questions:
        report.notes.append("没有检索到相关面试题，出题将主要依据简历与岗位 JD")

    if not documents:
        report.error = "没有任何可入库的素材（缺少简历、目标岗位与同类岗位）"
        return report

    # ---- 建库 ----
    try:
        kb = get_knowledge_base()
        result = await kb.abuild(collection, documents, rebuild=rebuild)
        report.ok = True
        report.documents = result.documents
        report.chunks = result.chunks
        report.seconds = time.perf_counter() - started
        logger.info(
            "个人知识库建成：%d 篇 -> %d 块（简历%d 目标岗位%d 同类岗位%d 面试题%d）",
            result.documents, result.chunks, report.resume_chunks,
            report.target_job_chunks, report.similar_job_chunks, report.question_chunks,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("建立个人知识库失败")
        report.error = f"{type(exc).__name__}: {exc}"

    report.seconds = time.perf_counter() - started
    return report


async def _load_similar_jobs(
    profile: UserProfileData, target_job: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    """取同方向/同城市抓到的其他岗位，优先取有薪资、有 JD 正文的。"""
    async with session_scope() as db:
        rows = await repository.list_crawled_jobs(
            db, query_role=profile.target_role, query_city=profile.expect_city, limit=200
        )
        jobs = [repository.crawled_job_to_dict(row) for row in rows]

    target_key = str(target_job.get("job_key") or "")
    target_title = str(target_job.get("title") or "").strip().lower()
    jobs = [
        job
        for job in jobs
        if job.get("job_key") != target_key
        and str(job.get("title") or "").strip().lower() != target_title
    ]

    def score(job: dict[str, Any]) -> tuple[int, float]:
        has_jd = 1 if len(str(job.get("jd_text") or "")) >= 80 else 0
        salary = float(job.get("salary_mid_k") or 0)
        return (has_jd, salary)

    jobs.sort(key=score, reverse=True)
    return jobs[:limit]


async def _retrieve_questions(
    resume_text: str, target_job: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    """从通用面试题库里检索与「简历 + 目标岗位」最相关的题目。"""
    settings = get_settings()
    query_parts = [
        str(target_job.get("title") or ""),
        " ".join(str(t) for t in (target_job.get("tags") or [])[:10]),
        str(target_job.get("jd_text") or "")[:400],
        resume_text[:400],
    ]
    query = " ".join(part for part in query_parts if part).strip()
    if not query:
        return []

    kb = get_knowledge_base()
    result = await kb.asearch(
        query, collections=[settings.rag_collection_questions], top_k=limit
    )
    return [
        {
            "text": chunk.text,
            "title": (chunk.metadata or {}).get("title", ""),
            "label": chunk.citation_label,
        }
        for chunk in result.chunks
    ]


async def personal_kb_stats() -> dict[str, Any]:
    """个人知识库状态，给前端展示与门禁判定用。"""
    settings = get_settings()
    kb = get_knowledge_base()
    collection = settings.rag_collection_personal
    total = await kb.acount(collection)
    return {
        "collection": collection,
        "chunks": total,
        "ready": total > 0,
        "estimated_tokens": 0,
    }


__all__ = ["PersonalKbReport", "build_personal_kb", "personal_kb_stats"]
