"""SQLAlchemy 异步 ORM 模型。

设计要点：
* 全部使用可移植类型（String / Text / Integer / Float / DateTime），
  ``Text`` 在 MySQL 上会加长成 ``LONGTEXT``，避免简历、摘要被 64KB 截断。
* JSON 数据（引用来源、工具调用记录）统一序列化成 Text 存储，
  这样 MySQL 与 SQLite 都能跑，不需要为方言写两套代码。
* 时间字段用 Python 端默认值，避免依赖数据库的 ``NOW()``。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# MySQL 下把 Text 提升为 LONGTEXT（TEXT 只有 64KB）
LongText = Text().with_variant(mysql.LONGTEXT(), "mysql")
MediumText = Text().with_variant(mysql.MEDIUMTEXT(), "mysql")


def _now() -> datetime:
    return datetime.now()


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class UserProfile(Base):
    """用户求职档案（单机版只有一行，id 恒为 1）。

    这是整个应用的「起点」：求职方向 / 期望城市 / 期望薪资是**必填**的，
    没填全就不允许进入抓岗、建知识库、模拟面试等后续步骤。
    """

    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)

    # ---- 必填 ----
    target_role: Mapped[str] = mapped_column(String(255), default="")
    expect_city: Mapped[str] = mapped_column(String(64), default="")
    expect_salary_min: Mapped[float] = mapped_column(Float, default=0.0)   # K/月
    expect_salary_max: Mapped[float] = mapped_column(Float, default=0.0)   # K/月，0 表示不限
    resume_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # ---- 选填（用于提升抓取与出题质量）----
    experience_years: Mapped[float] = mapped_column(Float, default=0.0)
    education: Mapped[str] = mapped_column(String(32), default="")
    skills_json: Mapped[str] = mapped_column(MediumText, default="[]")
    target_keywords_json: Mapped[str] = mapped_column(MediumText, default="[]")
    extra_notes: Mapped[str] = mapped_column(MediumText, default="")

    # ---- 抓取与市场行情快照 ----
    market_summary_json: Mapped[str] = mapped_column(MediumText, default="{}")
    market_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    target_job_json: Mapped[str] = mapped_column(MediumText, default="{}")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class CrawledJob(Base):
    """真实爬取到的岗位。

    与 ``job_postings``（内置示例种子数据）刻意分开：
    示例数据用来离线兜底和演示，爬取数据才代表真实市场行情。
    ``job_key`` 是规范化后的岗位身份哈希，用于跨次抓取去重。
    """

    __tablename__ = "crawled_jobs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    job_key: Mapped[str] = mapped_column(String(64), index=True, unique=True)

    source: Mapped[str] = mapped_column(String(64), default="")
    source_url: Mapped[str] = mapped_column(String(1024), default="")
    title: Mapped[str] = mapped_column(String(255), index=True)
    company: Mapped[str] = mapped_column(String(255), default="")
    city: Mapped[str] = mapped_column(String(64), index=True, default="")
    job_type: Mapped[str] = mapped_column(String(32), default="")

    # 原始薪资文本 + 归一化后的月薪区间（K）
    salary_text: Mapped[str] = mapped_column(String(128), default="")
    salary_low_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_high_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_mid_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_months: Mapped[int] = mapped_column(Integer, default=12)

    tags_json: Mapped[str] = mapped_column(MediumText, default="[]")
    jd_text: Mapped[str] = mapped_column(LongText, default="")
    published_at: Mapped[str] = mapped_column(String(32), default="")

    # 这次抓取用的查询条件，便于「按方向/城市」回捞
    query_role: Mapped[str] = mapped_column(String(255), index=True, default="")
    query_city: Mapped[str] = mapped_column(String(64), index=True, default="")
    matched_keywords_json: Mapped[str] = mapped_column(MediumText, default="[]")

    crawled_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


class Resume(Base):
    """用户上传并解析后的简历。"""

    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="anonymous")
    filename: Mapped[str] = mapped_column(String(255), default="")
    content: Mapped[str] = mapped_column(LongText)
    content_length: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class ChatSession(Base):
    """多轮对话会话。

    ``summary`` 与 ``summary_upto_seq`` 一起实现「长上下文」：
    序号 <= summary_upto_seq 的消息已经被压缩进摘要，不再以原文发送给模型。
    """

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="anonymous")
    mode: Mapped[str] = mapped_column(String(32), default="mock_interview")
    role_type: Mapped[str] = mapped_column(String(16), default="tech")
    title: Mapped[str] = mapped_column(String(255), default="")
    resume_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # 目标岗位：用户在「薪资/岗位」检索后选定，面试/出题/评分都围绕它展开
    job_title: Mapped[str] = mapped_column(String(255), default="")
    job_company: Mapped[str] = mapped_column(String(255), default="")
    job_context: Mapped[str] = mapped_column(MediumText, default="")

    summary: Mapped[str] = mapped_column(MediumText, default="")
    summary_upto_seq: Mapped[int] = mapped_column(Integer, default=0)
    message_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class ChatMessage(Base):
    """会话中的一条消息。"""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(LongText)

    sources_json: Mapped[str] = mapped_column(MediumText, default="[]")
    tool_calls_json: Mapped[str] = mapped_column(MediumText, default="[]")
    citations_json: Mapped[str] = mapped_column(String(512), default="[]")

    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    __table_args__ = (Index("ix_chat_messages_session_seq", "session_id", "seq"),)


class JobPosting(Base):
    """在招岗位（Function Calling 岗位搜索工具的数据源）。"""

    __tablename__ = "job_postings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role_type: Mapped[str] = mapped_column(String(16), index=True, default="tech")
    title: Mapped[str] = mapped_column(String(255), index=True)
    company: Mapped[str] = mapped_column(String(255), default="")
    city: Mapped[str] = mapped_column(String(64), index=True, default="")
    salary: Mapped[str] = mapped_column(String(64), default="")
    salary_min: Mapped[float] = mapped_column(Float, default=0.0)
    salary_max: Mapped[float] = mapped_column(Float, default=0.0)
    salary_unit: Mapped[str] = mapped_column(String(32), default="K/月")
    experience: Mapped[str] = mapped_column(String(64), default="")
    years_min: Mapped[float] = mapped_column(Float, default=0.0)
    years_max: Mapped[float] = mapped_column(Float, default=99.0)
    education: Mapped[str] = mapped_column(String(32), default="")
    employment_type: Mapped[str] = mapped_column(String(32), default="全职")
    skills_json: Mapped[str] = mapped_column(MediumText, default="[]")
    responsibilities_json: Mapped[str] = mapped_column(MediumText, default="[]")
    requirements_json: Mapped[str] = mapped_column(MediumText, default="[]")
    description: Mapped[str] = mapped_column(MediumText, default="")
    source: Mapped[str] = mapped_column(String(128), default="")
    url: Mapped[str] = mapped_column(String(512), default="")
    posted_at: Mapped[str] = mapped_column(String(32), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class SalaryRecord(Base):
    """薪资分位数据（Function Calling 薪资查询工具的数据源）。"""

    __tablename__ = "salary_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(128), index=True)
    role_type: Mapped[str] = mapped_column(String(16), index=True, default="tech")
    city: Mapped[str] = mapped_column(String(64), index=True, default="")
    level: Mapped[str] = mapped_column(String(32), default="")
    years_min: Mapped[float] = mapped_column(Float, default=0.0)
    years_max: Mapped[float] = mapped_column(Float, default=99.0)
    p25: Mapped[float] = mapped_column(Float, default=0.0)
    p50: Mapped[float] = mapped_column(Float, default=0.0)
    p75: Mapped[float] = mapped_column(Float, default=0.0)
    p90: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(32), default="K/月")
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(128), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class KbDocument(Base):
    """知识库文档登记表（记录入库的文档与块数，便于运维与增量更新）。"""

    __tablename__ = "kb_documents"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    collection: Mapped[str] = mapped_column(String(64), index=True)
    doc_id: Mapped[str] = mapped_column(String(128), default="")
    title: Mapped[str] = mapped_column(String(255), default="")
    source: Mapped[str] = mapped_column(String(255), default="")
    role_type: Mapped[str] = mapped_column(String(16), default="")
    category: Mapped[str] = mapped_column(String(64), default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class RequestLog(Base):
    """请求日志：并发压测与可观测性的事实依据。"""

    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(128), index=True)
    method: Mapped[str] = mapped_column(String(8), default="GET")
    status_code: Mapped[int] = mapped_column(Integer, default=200)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    in_flight: Mapped[int] = mapped_column(Integer, default=0)
    client: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


__all__ = [
    "Base",
    "ChatMessage",
    "ChatSession",
    "CrawledJob",
    "JobPosting",
    "KbDocument",
    "RequestLog",
    "Resume",
    "SalaryRecord",
    "UserProfile",
]
