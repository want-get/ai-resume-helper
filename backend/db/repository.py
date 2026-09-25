"""数据访问层：把 SQL 细节收拢在这里，业务层只调用语义化函数。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Sequence

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    ChatMessage,
    ChatSession,
    CrawledJob,
    JobPosting,
    KbDocument,
    RequestLog,
    Resume,
    SalaryRecord,
    UserProfile,
)


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


# ====================================================================== #
# 简历
# ====================================================================== #
async def create_resume(
    db: AsyncSession, resume_id: str, user_id: str, filename: str, content: str
) -> Resume:
    resume = Resume(
        id=resume_id,
        user_id=user_id,
        filename=filename,
        content=content,
        content_length=len(content),
    )
    db.add(resume)
    await db.flush()
    return resume


async def get_resume(db: AsyncSession, resume_id: str) -> Resume | None:
    return await db.get(Resume, resume_id)


# ====================================================================== #
# 会话
# ====================================================================== #
async def create_chat_session(
    db: AsyncSession,
    session_id: str,
    user_id: str,
    mode: str,
    role_type: str,
    title: str = "",
    resume_id: str | None = None,
    job_title: str = "",
    job_company: str = "",
    job_context: str = "",
) -> ChatSession:
    session = ChatSession(
        id=session_id,
        user_id=user_id,
        mode=mode,
        role_type=role_type,
        title=title,
        resume_id=resume_id,
        job_title=job_title,
        job_company=job_company,
        job_context=job_context,
    )
    db.add(session)
    await db.flush()
    return session


async def list_target_jobs(db: AsyncSession, user_id: str | None = None, limit: int = 20) -> list[ChatSession]:
    """列出带目标岗位的会话（用于前端「最近选定的岗位」）。"""
    stmt = (
        select(ChatSession)
        .where(ChatSession.job_title != "")
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
    )
    if user_id:
        stmt = stmt.where(ChatSession.user_id == user_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_chat_session(db: AsyncSession, session_id: str) -> ChatSession | None:
    return await db.get(ChatSession, session_id)


async def list_chat_sessions(
    db: AsyncSession, user_id: str | None = None, limit: int = 50
) -> list[ChatSession]:
    stmt = select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
    if user_id:
        stmt = stmt.where(ChatSession.user_id == user_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_session_summary(
    db: AsyncSession, session_id: str, summary: str, upto_seq: int
) -> None:
    await db.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(summary=summary, summary_upto_seq=upto_seq, updated_at=datetime.now())
    )


async def touch_session(
    db: AsyncSession, session_id: str, message_count: int, title: str | None = None
) -> None:
    values: dict[str, Any] = {
        "message_count": message_count,
        "updated_at": datetime.now(),
    }
    if title is not None:
        values["title"] = title
    await db.execute(update(ChatSession).where(ChatSession.id == session_id).values(**values))


async def delete_chat_session(db: AsyncSession, session_id: str) -> None:
    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    await db.execute(delete(ChatSession).where(ChatSession.id == session_id))


# ====================================================================== #
# 消息
# ====================================================================== #
async def add_message(
    db: AsyncSession,
    session_id: str,
    seq: int,
    role: str,
    content: str,
    sources: Sequence[dict[str, Any]] | None = None,
    tool_calls: Sequence[dict[str, Any]] | None = None,
    citations: Sequence[int] | None = None,
    token_estimate: int = 0,
    latency_ms: float = 0.0,
) -> ChatMessage:
    message = ChatMessage(
        session_id=session_id,
        seq=seq,
        role=role,
        content=content,
        sources_json=json.dumps(list(sources or []), ensure_ascii=False),
        tool_calls_json=json.dumps(list(tool_calls or []), ensure_ascii=False),
        citations_json=json.dumps(list(citations or [])),
        token_estimate=token_estimate,
        latency_ms=latency_ms,
    )
    db.add(message)
    await db.flush()
    return message


async def list_messages(
    db: AsyncSession,
    session_id: str,
    after_seq: int | None = None,
    limit: int | None = None,
    ascending: bool = True,
) -> list[ChatMessage]:
    stmt = select(ChatMessage).where(ChatMessage.session_id == session_id)
    if after_seq is not None:
        stmt = stmt.where(ChatMessage.seq > after_seq)
    stmt = stmt.order_by(ChatMessage.seq.asc() if ascending else ChatMessage.seq.desc())
    if limit:
        stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    return rows if ascending else list(reversed(rows))


def session_to_dict(session: ChatSession, message_count: int | None = None) -> dict[str, Any]:
    return {
        "id": session.id,
        "mode": session.mode,
        "role_type": session.role_type,
        "title": session.title,
        "resume_id": session.resume_id,
        "job_title": session.job_title or "",
        "job_company": session.job_company or "",
        "has_job_context": bool(session.job_context),
        "message_count": message_count if message_count is not None else session.message_count,
        "summary": session.summary,
        "summary_chars": len(session.summary or ""),
        "summary_upto_seq": session.summary_upto_seq,
        "created_at": session.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": session.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


async def count_messages(db: AsyncSession, session_id: str) -> int:
    result = await db.execute(
        select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id)
    )
    return int(result.scalar() or 0)


def message_to_dict(message: ChatMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "seq": message.seq,
        "role": message.role,
        "content": message.content,
        "sources": _loads(message.sources_json, []),
        "tool_calls": _loads(message.tool_calls_json, []),
        "citations": _loads(message.citations_json, []),
        "token_estimate": message.token_estimate,
        "latency_ms": round(message.latency_ms, 2),
        "created_at": message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ====================================================================== #
# 岗位 / 薪资（Function Calling 数据源）
# ====================================================================== #
async def upsert_job_postings(db: AsyncSession, records: Sequence[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        payload = {
            "id": str(record.get("id")),
            "role_type": record.get("role_type", "tech"),
            "title": record.get("title", ""),
            "company": record.get("company", ""),
            "city": record.get("city", ""),
            "salary": record.get("salary", ""),
            "salary_min": float(record.get("salary_min") or 0),
            "salary_max": float(record.get("salary_max") or 0),
            "salary_unit": record.get("salary_unit", "K/月"),
            "experience": record.get("experience", ""),
            "years_min": float(record.get("years_min") or 0),
            "years_max": float(record.get("years_max") or 99),
            "education": record.get("education", ""),
            "employment_type": record.get("employment_type", "全职"),
            "skills_json": json.dumps(record.get("skills") or [], ensure_ascii=False),
            "responsibilities_json": json.dumps(record.get("responsibilities") or [], ensure_ascii=False),
            "requirements_json": json.dumps(record.get("requirements") or [], ensure_ascii=False),
            "description": record.get("description", ""),
            "source": record.get("source", ""),
            "url": record.get("url", ""),
            "posted_at": record.get("posted_at", ""),
        }
        existing = await db.get(JobPosting, payload["id"])
        if existing is None:
            db.add(JobPosting(**payload))
        else:
            for key, value in payload.items():
                setattr(existing, key, value)
        count += 1
    await db.flush()
    return count


async def search_job_postings(
    db: AsyncSession,
    keyword: str | None = None,
    city: str | None = None,
    salary_min: float | None = None,
    role_type: str | None = None,
    limit: int = 5,
) -> list[JobPosting]:
    stmt = select(JobPosting)
    if keyword:
        pattern = f"%{keyword}%"
        stmt = stmt.where(
            or_(
                JobPosting.title.like(pattern),
                JobPosting.company.like(pattern),
                JobPosting.skills_json.like(pattern),
                JobPosting.requirements_json.like(pattern),
                JobPosting.description.like(pattern),
            )
        )
    if city:
        stmt = stmt.where(JobPosting.city.like(f"%{city}%"))
    if salary_min is not None:
        stmt = stmt.where(JobPosting.salary_max >= salary_min)
    if role_type:
        stmt = stmt.where(JobPosting.role_type == role_type)
    stmt = stmt.order_by(JobPosting.salary_max.desc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def count_job_postings(db: AsyncSession) -> int:
    result = await db.execute(select(func.count()).select_from(JobPosting))
    return int(result.scalar() or 0)


def job_to_dict(job: JobPosting) -> dict[str, Any]:
    return {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "city": job.city,
        "salary": job.salary,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "salary_unit": job.salary_unit,
        "experience": job.experience,
        "education": job.education,
        "employment_type": job.employment_type,
        "skills": _loads(job.skills_json, []),
        "responsibilities": _loads(job.responsibilities_json, []),
        "requirements": _loads(job.requirements_json, []),
        "description": job.description,
        "source": job.source,
        "url": job.url,
        "posted_at": job.posted_at,
        "role_type": job.role_type,
    }


async def upsert_salary_records(db: AsyncSession, records: Sequence[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        payload = {
            "id": str(record.get("id")),
            "role": record.get("role", ""),
            "role_type": record.get("role_type", "tech"),
            "city": record.get("city", ""),
            "level": record.get("level", ""),
            "years_min": float(record.get("years_min") or 0),
            "years_max": float(record.get("years_max") or 99),
            "p25": float(record.get("p25") or 0),
            "p50": float(record.get("p50") or 0),
            "p75": float(record.get("p75") or 0),
            "p90": float(record.get("p90") or 0),
            "unit": record.get("unit", "K/月"),
            "currency": record.get("currency", "CNY"),
            "sample_size": int(record.get("sample_size") or 0),
            "source": record.get("source", ""),
        }
        existing = await db.get(SalaryRecord, payload["id"])
        if existing is None:
            db.add(SalaryRecord(**payload))
        else:
            for key, value in payload.items():
                setattr(existing, key, value)
        count += 1
    await db.flush()
    return count


async def list_salary_records(
    db: AsyncSession,
    city: str | None = None,
    role_type: str | None = None,
    level: str | None = None,
) -> list[SalaryRecord]:
    stmt = select(SalaryRecord)
    if city:
        stmt = stmt.where(SalaryRecord.city.like(f"%{city}%"))
    if role_type:
        stmt = stmt.where(SalaryRecord.role_type == role_type)
    if level:
        stmt = stmt.where(SalaryRecord.level == level)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def count_salary_records(db: AsyncSession) -> int:
    result = await db.execute(select(func.count()).select_from(SalaryRecord))
    return int(result.scalar() or 0)


def salary_to_dict(record: SalaryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "role": record.role,
        "role_type": record.role_type,
        "city": record.city,
        "level": record.level,
        "years_min": record.years_min,
        "years_max": record.years_max,
        "p25": record.p25,
        "p50": record.p50,
        "p75": record.p75,
        "p90": record.p90,
        "unit": record.unit,
        "currency": record.currency,
        "sample_size": record.sample_size,
        "source": record.source,
        "updated_at": record.updated_at.strftime("%Y-%m-%d"),
    }


# ====================================================================== #
# 知识库登记 / 请求日志
# ====================================================================== #
async def record_kb_documents(db: AsyncSession, rows: Sequence[dict[str, Any]]) -> int:
    for row in rows:
        db.add(KbDocument(**row))
    await db.flush()
    return len(rows)


async def clear_kb_documents(db: AsyncSession, collection: str) -> None:
    await db.execute(delete(KbDocument).where(KbDocument.collection == collection))


async def count_kb_documents(db: AsyncSession) -> dict[str, int]:
    result = await db.execute(
        select(KbDocument.collection, func.count()).group_by(KbDocument.collection)
    )
    return {row[0]: int(row[1]) for row in result.all()}


async def add_request_log(db: AsyncSession, **kwargs: Any) -> None:
    db.add(RequestLog(**kwargs))
    await db.flush()


async def request_stats(db: AsyncSession, minutes: int = 60) -> dict[str, Any]:
    since = datetime.now() - timedelta(minutes=minutes)
    total = await db.execute(
        select(func.count()).select_from(RequestLog).where(RequestLog.created_at >= since)
    )
    avg_latency = await db.execute(
        select(func.avg(RequestLog.latency_ms)).where(RequestLog.created_at >= since)
    )
    max_latency = await db.execute(
        select(func.max(RequestLog.latency_ms)).where(RequestLog.created_at >= since)
    )
    max_in_flight = await db.execute(
        select(func.max(RequestLog.in_flight)).where(RequestLog.created_at >= since)
    )
    return {
        "window_minutes": minutes,
        "requests": int(total.scalar() or 0),
        "avg_latency_ms": round(float(avg_latency.scalar() or 0.0), 2),
        "max_latency_ms": round(float(max_latency.scalar() or 0.0), 2),
        "peak_in_flight": int(max_in_flight.scalar() or 0),
    }


async def prune_request_logs(db: AsyncSession, keep_minutes: int = 120) -> int:
    cutoff = datetime.now() - timedelta(minutes=keep_minutes)
    result = await db.execute(delete(RequestLog).where(RequestLog.created_at < cutoff))
    return int(result.rowcount or 0)


# ====================================================================== #
# 用户档案（单机版只有一行）
# ====================================================================== #
async def get_user_profile(db: AsyncSession, profile_id: int = 1) -> UserProfile | None:
    return await db.get(UserProfile, profile_id)


async def upsert_user_profile(
    db: AsyncSession, profile_id: int, values: dict[str, Any]
) -> UserProfile:
    row = await db.get(UserProfile, profile_id)
    if row is None:
        row = UserProfile(id=profile_id)
        db.add(row)
    for key, value in values.items():
        if hasattr(row, key):
            setattr(row, key, value)
    await db.flush()
    await db.refresh(row)
    return row


# ====================================================================== #
# 爬取到的真实岗位
# ====================================================================== #
async def upsert_crawled_jobs(db: AsyncSession, records: Sequence[dict[str, Any]]) -> int:
    """按 ``job_key`` 去重写入；同一岗位重复抓到只更新，不新增。"""
    written = 0
    for record in records:
        job_key = str(record.get("job_key") or "").strip()
        if not job_key:
            continue
        existing = (
            await db.execute(select(CrawledJob).where(CrawledJob.job_key == job_key))
        ).scalar_one_or_none()
        payload = {
            "job_key": job_key,
            "source": record.get("source", ""),
            "source_url": record.get("source_url", ""),
            "title": record.get("title", ""),
            "company": record.get("company", ""),
            "city": record.get("city", ""),
            "job_type": record.get("job_type", ""),
            "salary_text": record.get("salary_text", ""),
            "salary_low_k": record.get("salary_low_k"),
            "salary_high_k": record.get("salary_high_k"),
            "salary_mid_k": record.get("salary_mid_k"),
            "salary_months": int(record.get("salary_months") or 12),
            "tags_json": json.dumps(record.get("tags") or [], ensure_ascii=False),
            "jd_text": record.get("jd_text", ""),
            "published_at": record.get("published_at", ""),
            "query_role": record.get("query_role", ""),
            "query_city": record.get("query_city", ""),
            "matched_keywords_json": json.dumps(
                record.get("matched_keywords") or [], ensure_ascii=False
            ),
            "crawled_at": datetime.now(),
        }
        if existing is None:
            db.add(CrawledJob(**payload))
        else:
            for key, value in payload.items():
                setattr(existing, key, value)
        written += 1
    await db.flush()
    return written


async def list_crawled_jobs(
    db: AsyncSession,
    query_role: str | None = None,
    query_city: str | None = None,
    with_salary_only: bool = False,
    limit: int = 200,
) -> list[CrawledJob]:
    stmt = select(CrawledJob)
    if query_role:
        stmt = stmt.where(CrawledJob.query_role == query_role)
    if query_city:
        stmt = stmt.where(CrawledJob.query_city == query_city)
    if with_salary_only:
        stmt = stmt.where(CrawledJob.salary_mid_k.is_not(None))
    stmt = stmt.order_by(CrawledJob.crawled_at.desc()).limit(max(1, min(limit, 1000)))
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def count_crawled_jobs(
    db: AsyncSession, query_role: str | None = None, query_city: str | None = None
) -> int:
    stmt = select(func.count()).select_from(CrawledJob)
    if query_role:
        stmt = stmt.where(CrawledJob.query_role == query_role)
    if query_city:
        stmt = stmt.where(CrawledJob.query_city == query_city)
    result = await db.execute(stmt)
    return int(result.scalar() or 0)


async def clear_crawled_jobs(
    db: AsyncSession, query_role: str | None = None, query_city: str | None = None
) -> int:
    """清掉某个查询条件下的旧结果，避免反复抓取后新旧混在一起。"""
    stmt = delete(CrawledJob)
    if query_role:
        stmt = stmt.where(CrawledJob.query_role == query_role)
    if query_city:
        stmt = stmt.where(CrawledJob.query_city == query_city)
    result = await db.execute(stmt)
    return int(result.rowcount or 0)


async def clear_all_crawled_jobs(db: AsyncSession) -> int:
    result = await db.execute(delete(CrawledJob))
    return int(result.rowcount or 0)


def crawled_job_to_dict(row: CrawledJob) -> dict[str, Any]:
    return {
        "id": row.id,
        "job_key": row.job_key,
        "source": row.source,
        "source_url": row.source_url,
        "title": row.title,
        "company": row.company,
        "city": row.city,
        "job_type": row.job_type,
        "salary_text": row.salary_text,
        "salary_low_k": row.salary_low_k,
        "salary_high_k": row.salary_high_k,
        "salary_mid_k": row.salary_mid_k,
        "salary_months": row.salary_months,
        "tags": _loads(row.tags_json, []),
        "jd_text": row.jd_text,
        "published_at": row.published_at,
        "matched_keywords": _loads(row.matched_keywords_json, []),
        "query_role": row.query_role,
        "query_city": row.query_city,
        "crawled_at": row.crawled_at.strftime("%Y-%m-%d %H:%M:%S"),
    }
