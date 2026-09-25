"""HTTP 请求 / 响应模型（Pydantic v2）。

请求模型做严格校验（长度、枚举、范围），把脏数据挡在业务层之前；
响应统一用 ``ServiceResult.to_dict()`` 的结构，这里只定义必要的包装。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

RoleType = Literal["tech", "non_tech"]
CollectionName = Literal["interview_questions", "job_descriptions"]
SessionMode = Literal["mock_interview", "rag_chat"]


# ====================================================================== #
# 通用
# ====================================================================== #
class HealthResponse(BaseModel):
    status: str
    version: str
    database: dict[str, Any]
    knowledge_base: list[dict[str, Any]]
    llm: dict[str, Any]
    concurrency: dict[str, Any]


# ====================================================================== #
# 知识库
# ====================================================================== #
class KBSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    collections: list[CollectionName] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    role_type: RoleType | None = None
    hybrid: bool = True


class KBDocumentIn(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBAddRequest(BaseModel):
    collection: CollectionName
    documents: list[KBDocumentIn] = Field(min_length=1, max_length=500)
    rebuild: bool = False


class KBRebuildRequest(BaseModel):
    collection: CollectionName | None = None
    from_seed: bool = True


# ====================================================================== #
# RAG 问答 / Function Calling
# ====================================================================== #
class AskRequest(BaseModel):
    """知识库问答请求。"""

    question: str = Field(min_length=1, max_length=4000)
    role_type: RoleType = "tech"
    collections: list[CollectionName] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


class ToolAskRequest(BaseModel):
    """Function Calling 请求。"""

    question: str = Field(min_length=1, max_length=2000)
    role_type: RoleType = "tech"
    use_tools: bool = True
    strict: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


class ToolCallRequestModel(BaseModel):
    """直接调用某个工具（不经过模型决策，便于调试）。"""

    name: Literal["query_salary", "search_jobs", "search_knowledge_base"]
    arguments: dict[str, Any] = Field(default_factory=dict)


# ====================================================================== #
# 简历
# ====================================================================== #
class ResumeCreateRequest(BaseModel):
    filename: str = Field(default="resume.pdf", max_length=255)
    content: str = Field(min_length=30, max_length=200_000)


class ResumeFeatureRequest(BaseModel):
    """简历优化 / 面试题 / 评分 共用请求。"""

    role_type: RoleType = "tech"
    resume_text: str | None = Field(default=None, max_length=200_000)
    resume_id: str | None = Field(default=None, max_length=36)
    # 目标岗位 JD 文本（用户在「薪资/岗位」页选定后带入）。
    # 提供了它就用它作为岗位要求上下文，替代/补充知识库检索结果。
    job_context: str | None = Field(default=None, max_length=20_000)

    @field_validator("resume_id")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        return value.strip() if value else value


# ====================================================================== #
# 多轮对话
# ====================================================================== #
class SessionCreateRequest(BaseModel):
    mode: SessionMode = "mock_interview"
    role_type: RoleType = "tech"
    title: str = Field(default="", max_length=255)
    resume_id: str | None = None
    resume_text: str | None = Field(default=None, max_length=200_000)
    # 目标岗位定位：来自岗位检索结果，用于让面试/出题围绕该岗位展开
    job_title: str = Field(default="", max_length=255)
    job_company: str = Field(default="", max_length=255)
    job_context: str | None = Field(default=None, max_length=20_000)


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    use_tools: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)


class SessionOut(BaseModel):
    id: str
    mode: str
    role_type: str
    title: str
    resume_id: str | None = None
    message_count: int = 0
    summary_chars: int = 0
    summary_upto_seq: int = 0
    created_at: str = ""
    updated_at: str = ""


class MessageOut(BaseModel):
    id: int
    seq: int
    role: str
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)
    token_estimate: int = 0
    latency_ms: float = 0.0
    created_at: str = ""


class ChatReplyOut(BaseModel):
    session_id: str
    reply: MessageOut
    result: dict[str, Any]
    memory: dict[str, Any] | None = None
    context_snapshot: dict[str, Any] | None = None


# ====================================================================== #
# 运行时设置（在线填 API Key，免改 .env / 免重启）
# ====================================================================== #
class LLMSettingsUpdate(BaseModel):
    """在线修改大模型配置。只传需要改的字段。"""

    api_key: str | None = Field(default=None, max_length=200)
    base_url: str | None = Field(default=None, max_length=300)
    model: str | None = Field(default=None, max_length=100)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # 是否写回 .env（重启后仍保留）
    persist: bool = True
    # 是否立刻用一次最小调用验证 Key 可用
    verify: bool = True


# ====================================================================== #
# 直接工具调用
# ====================================================================== #
class SalaryQueryRequest(BaseModel):
    role: str = Field(min_length=1, max_length=128)
    city: str | None = Field(default=None, max_length=64)
    level: Literal["初级", "中级", "高级"] | None = None
    years: float | None = Field(default=None, ge=0, le=50)


class JobSearchRequest(BaseModel):
    """岗位检索：关键字只匹配职位名称，多个关键字用空格/逗号分隔。"""

    keywords: str = Field(min_length=1, max_length=200)
    city: str | None = Field(default=None, max_length=64)
    sources: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=8, ge=1, le=30)
    with_description: bool = True
