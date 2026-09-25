"""业务编排层：把 RAG、防幻觉、长上下文、Function Calling 组装成对外能力。

对外提供的服务：
    * ``KBService``           知识库问答（RAG + 引用校验）
    * ``ToolService``         Function Calling 智能体（薪资 / 岗位 / 知识库）
    * ``ResumeService``       简历优化、面试题生成、简历评分（RAG 增强）
    * ``InterviewService``    模拟面试多轮对话（长上下文 + 滚动摘要）

所有服务都是无状态的，会话状态存在数据库里，因此可以安全地被
多个 asyncio 任务并发调用。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from config import Settings, get_settings
from prompts import NON_TECH, ROLE_LABELS, TECH, get_prompts
from prompts import common as common_prompts
from prompts import system as system_prompts
from rag import RetrievedChunk, get_knowledge_base

from .ai_client import AsyncAIClient, LLMResult, ToolCallRequest, get_ai_client
from .anti_hallucination import (
    GuardedAnswer,
    SourceRef,
    build_context_block,
    build_tool_context,
    guard_answer,
)
from .memory import BuiltContext, ConversationMemory, MemoryStats
from .tools import execute_tool_calls

logger = logging.getLogger("ai_resume_helper.services")

VALID_ROLE_TYPES = (TECH, NON_TECH)


def normalize_role_type(role_type: str | None) -> str:
    return role_type if role_type in VALID_ROLE_TYPES else TECH


def role_label(role_type: str | None) -> str:
    return ROLE_LABELS[normalize_role_type(role_type)]


def make_job_context_chunk(
    job_context: str, job_title: str = "", job_company: str = ""
) -> RetrievedChunk:
    """把用户选定的「目标岗位 JD」包装成一个检索片段。

    这样它就能和其它知识库片段一起走同一套编号 / 引用 / 展示逻辑，
    模型看到的是统一的「参考资料」，引用编号也不会错位。
    """
    title = job_title or "目标岗位"
    company = f" @ {job_company}" if job_company else ""
    return RetrievedChunk(
        id="target-job-context",
        text=f"【目标岗位】{title}{company}\n{job_context.strip()}",
        metadata={
            "source": "目标岗位 JD",
            "title": title,
            "company": job_company,
            "role": job_title,
        },
        score=1.0,
        similarity=1.0,
        collection="target_job",
        source="user_selected",
    )


# ====================================================================== #
# 统一结果结构
# ====================================================================== #
@dataclass(slots=True)
class ServiceResult:
    """服务层统一返回结构。"""

    content: str
    sources: list[SourceRef] = field(default_factory=list)
    report: Any = None
    confidence: float = 0.0
    refused: bool = False
    warnings: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_stats: MemoryStats | None = None
    retrieval: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    model: str = ""
    mock: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "citation_report": self.report.to_dict() if self.report is not None else None,
            "confidence": round(self.confidence, 3),
            "refused": self.refused,
            "warnings": self.warnings,
            "tool_calls": self.tool_calls,
            "memory": self.memory_stats.to_dict() if self.memory_stats else None,
            "retrieval": self.retrieval,
            "usage": self.usage,
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model,
            "mock": self.mock,
            "error": self.error,
        }


def _merge_usage(results: Sequence[LLMResult]) -> dict[str, Any]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": len(results)}
    for result in results:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int((result.usage or {}).get(key) or 0)
    return usage


def _retrieval_info(query: str, chunks: Sequence[RetrievedChunk], elapsed_ms: float = 0.0) -> dict[str, Any]:
    return {
        "query": query,
        "hits": len(chunks),
        "elapsed_ms": round(elapsed_ms, 2),
        "items": [
            {
                "index": i,
                "label": chunk.citation_label,
                "collection": chunk.collection,
                "similarity": None if chunk.similarity is None else round(chunk.similarity, 4),
                "retrieval": chunk.source,
            }
            for i, chunk in enumerate(chunks, start=1)
        ],
    }


# ====================================================================== #
# 知识库问答（RAG）
# ====================================================================== #
class KBService:
    """检索增强生成 + 防幻觉。"""

    def __init__(
        self, ai_client: AsyncAIClient | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.ai_client = ai_client or get_ai_client()

    async def personal_collections(self) -> list[str] | None:
        """用户专属知识库有内容时优先用它。

        这是「模拟面试要结合用户自己的知识库出题」的落点：
        有个人库就只搜个人库（简历 + 目标岗位 + 同类岗位 + 相关面试题），
        没有才退回公共题库/JD 库。
        """
        personal = self.settings.rag_collection_personal
        try:
            count = await get_knowledge_base().acount(personal)
        except Exception:  # noqa: BLE001 - 个人库不存在时按没有处理
            return None
        return [personal] if count > 0 else None

    async def retrieve(
        self,
        query: str,
        collections: Sequence[str] | None = None,
        top_k: int | None = None,
        role_type: str | None = None,
    ) -> tuple[list[RetrievedChunk], float]:
        kb = get_knowledge_base()
        result = await kb.asearch(query, collections=collections, top_k=top_k, role_type=role_type)
        return result.chunks, result.elapsed_ms

    async def answer(
        self,
        question: str,
        role_type: str | None = None,
        collections: Sequence[str] | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_context_chars: int = 6000,
    ) -> ServiceResult:
        """知识库问答：先检索，再生成，最后做引用校验。"""
        started = time.perf_counter()
        settings = self.settings
        role_type = normalize_role_type(role_type)

        chunks, retrieve_ms = await self.retrieve(
            question, collections=collections, top_k=top_k, role_type=role_type
        )
        context, sources = build_context_block(chunks, max_chars=max_context_chars)

        if not context and settings.refuse_without_context:
            message = (
                f"知识库中未收录与该问题相关的内容。\n\n"
                f"你问的是：{question}\n\n"
                "建议补充更具体的关键词（例如岗位名称、技术名词、面试维度），或换一种问法。"
            )
            guarded = guard_answer(message, [], refused=True)
            return ServiceResult(
                content=guarded.content,
                sources=[],
                report=guarded.report,
                confidence=0.0,
                refused=True,
                warnings=["未检索到资料，已拒绝自由发挥"],
                retrieval=_retrieval_info(question, chunks, retrieve_ms),
                latency_ms=(time.perf_counter() - started) * 1000,
                model=settings.deepseek_model,
                mock=self.ai_client.mock_mode,
            )

        system_prompt = system_prompts.RAG_SYSTEM_PROMPT.format(
            role_label=role_label(role_type),
            anti_hallucination_rules=system_prompts.ANTI_HALLUCINATION_RULES,
            context=context or "（无）",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        result = await self.ai_client.chat(
            messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
        )
        if not result.ok:
            return ServiceResult(
                content="",
                error=result.error,
                sources=sources,
                latency_ms=(time.perf_counter() - started) * 1000,
                mock=result.mock,
            )

        guarded = guard_answer(result.content, sources)
        return ServiceResult(
            content=guarded.content,
            sources=guarded.sources,
            report=guarded.report,
            confidence=guarded.confidence,
            refused=guarded.refused,
            warnings=guarded.warnings,
            retrieval=_retrieval_info(question, chunks, retrieve_ms),
            usage=_merge_usage([result]),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=result.model,
            mock=result.mock,
        )


# ====================================================================== #
# Function Calling 智能体
# ====================================================================== #
async def run_tool_agent(
    messages: list[dict[str, Any]],
    ai_client: AsyncAIClient,
    settings: Settings,
    use_tools: bool = True,
    strict: bool = False,
    max_rounds: int | None = None,
    temperature: float | None = None,
) -> tuple[LLMResult, list[dict[str, Any]], list[LLMResult]]:
    """工具调用循环：模型决策 → 并行执行 → 结果回传 → 生成最终回答。"""
    from .tools import tool_schemas

    max_rounds = max_rounds or settings.max_tool_rounds
    tools = tool_schemas(strict=strict) if (use_tools and settings.tools_enabled) else []
    calls: list[LLMResult] = []
    executions: list[dict[str, Any]] = []
    working = list(messages)
    result = LLMResult(content="", error="未执行")

    for round_index in range(max_rounds + 1):
        result = await ai_client.chat(
            working,
            tools=tools or None,
            tool_choice="auto" if tools else None,
            strict=strict,
            temperature=settings.llm_temperature if temperature is None else temperature,
        )
        calls.append(result)

        if not result.ok or not result.tool_calls:
            break
        if round_index >= max_rounds:
            logger.warning("工具调用轮次达到上限 %s，停止继续调用", max_rounds)
            break

        # 把模型的调用意图加入历史
        working.append(
            {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [call.to_message_dict() for call in result.tool_calls],
            }
        )

        # 并行执行全部工具
        pairs = await execute_tool_calls(result.tool_calls, timeout=settings.tool_timeout)
        for call, execution in pairs:
            executions.append(execution.to_dict())
            working.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": execution.name,
                    "content": execution.to_content(),
                }
            )

    return result, executions, calls


class ToolService:
    """Function Calling 智能体服务。"""

    def __init__(
        self, ai_client: AsyncAIClient | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.ai_client = ai_client or get_ai_client()

    async def ask(
        self,
        question: str,
        role_type: str | None = None,
        use_tools: bool = True,
        strict: bool = False,
        temperature: float | None = None,
    ) -> ServiceResult:
        """让模型自主决定是否调用薪资 / 岗位 / 知识库工具。"""
        started = time.perf_counter()
        settings = self.settings

        system_prompt = system_prompts.TOOL_AGENT_SYSTEM_PROMPT.format(
            anti_hallucination_rules=system_prompts.ANTI_HALLUCINATION_RULES
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        result, executions, calls = await run_tool_agent(
            messages,
            self.ai_client,
            settings,
            use_tools=use_tools,
            strict=strict,
            temperature=temperature,
        )

        if not result.ok:
            return ServiceResult(
                content="",
                error=result.error,
                tool_calls=executions,
                latency_ms=(time.perf_counter() - started) * 1000,
                mock=result.mock,
            )

        warnings: list[str] = []
        if any(not item["ok"] for item in executions):
            warnings.append("部分工具调用失败，回答中已如实说明")
        if not executions and use_tools:
            warnings.append("本次未触发任何工具调用（模型判断无需外部数据）")

        guarded = guard_answer(result.content, [], extra_warnings=warnings)
        return ServiceResult(
            content=guarded.content,
            sources=[],
            report=guarded.report,
            confidence=guarded.confidence,
            warnings=guarded.warnings,
            tool_calls=executions,
            usage=_merge_usage(calls),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=result.model,
            mock=result.mock,
        )


# ====================================================================== #
# 简历相关功能（RAG 增强）
# ====================================================================== #
class ResumeService:
    """简历优化 / 面试题生成 / 简历评分。"""

    def __init__(
        self, ai_client: AsyncAIClient | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.ai_client = ai_client or get_ai_client()
        self.kb = KBService(self.ai_client, self.settings)

    async def _jd_context(
        self,
        resume_text: str,
        role_type: str,
        top_k: int = 4,
        job_context: str | None = None,
        job_title: str = "",
        job_company: str = "",
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        """检索岗位 JD 知识库。

        若用户选定了目标岗位（job_context），把它排在第一位作为主要依据，
        知识库检索结果作为补充——这样出题/评分会优先对齐用户真正想投的岗位。
        """
        chunks: list[RetrievedChunk] = []
        if job_context:
            chunks.append(make_job_context_chunk(job_context, job_title, job_company))

        query = f"{role_label(role_type)} 岗位要求 技能 职责 {resume_text[:300]}"
        # 有个人知识库就只搜个人库：简历 + 目标岗位 + 同类岗位都在里面，
        # 这样出的题才真正贴着这个人、这个岗位。
        personal = await self.kb.personal_collections()
        retrieved, elapsed = await self.kb.retrieve(
            query,
            collections=personal or [self.settings.rag_collection_jobs],
            top_k=top_k,
        )
        chunks.extend(retrieved)
        return chunks, _retrieval_info(query, retrieved, elapsed)

    async def _run(
        self,
        system_prompt: str,
        user_prompt: str,
        sources: Sequence[SourceRef],
        retrieval: dict[str, Any] | None,
        started: float,
        require_citation: bool = True,
    ) -> ServiceResult:
        result = await self.ai_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.settings.llm_temperature,
        )
        if not result.ok:
            return ServiceResult(
                content="",
                error=result.error,
                sources=list(sources),
                retrieval=retrieval,
                latency_ms=(time.perf_counter() - started) * 1000,
                mock=result.mock,
            )

        guarded = guard_answer(result.content, sources)
        if not require_citation:
            guarded.warnings = [
                w for w in guarded.warnings if "未标注任何引用编号" not in w
            ]
        return ServiceResult(
            content=guarded.content,
            sources=guarded.sources,
            report=guarded.report,
            confidence=guarded.confidence,
            warnings=guarded.warnings,
            retrieval=retrieval,
            usage=_merge_usage([result]),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=result.model,
            mock=result.mock,
        )

    async def generate_interview_questions(
        self,
        resume_text: str,
        role_type: str | None = None,
        job_context: str | None = None,
        job_title: str = "",
        job_company: str = "",
    ) -> ServiceResult:
        started = time.perf_counter()
        role_type = normalize_role_type(role_type)

        # 两路检索：岗位 JD（对齐岗位要求）+ 题库（对齐同类面试题）
        jd_chunks, retrieval = await self._jd_context(
            resume_text, role_type, job_context=job_context, job_title=job_title,
            job_company=job_company,
        )
        q_query = resume_text[:400]
        personal = await self.kb.personal_collections()
        q_chunks, q_elapsed = await self.kb.retrieve(
            q_query,
            collections=personal or [self.settings.rag_collection_questions],
            top_k=4 if personal else 4,
        )
        retrieval["question_bank_hits"] = _retrieval_info(q_query, q_chunks, q_elapsed)
        retrieval["target_job"] = job_title or None
        retrieval["using_personal_kb"] = bool(personal)

        # 合并后再统一编号，保证 Prompt 里的 [n] 与返回的 sources 完全一致
        context, sources = build_context_block(list(jd_chunks) + list(q_chunks))

        system_prompt = system_prompts.INTERVIEW_QUESTIONS_RAG_SYSTEM_PROMPT.format(
            role_label=role_label(role_type),
            anti_hallucination_rules=system_prompts.ANTI_HALLUCINATION_RULES,
            context=context or "（无）",
        )
        user_prompt = get_prompts(role_type).INTERVIEW_QUESTIONS_PROMPT.format(
            resume_text=resume_text
        )
        if job_title:
            user_prompt = (
                f"目标岗位：{job_title}"
                + (f"（{job_company}）" if job_company else "")
                + "\n请针对该岗位出题。\n\n"
                + user_prompt
            )
        return await self._run(system_prompt, user_prompt, sources, retrieval, started)

    async def optimize_resume(
        self,
        resume_text: str,
        role_type: str | None = None,
        job_context: str | None = None,
        job_title: str = "",
        job_company: str = "",
    ) -> ServiceResult:
        started = time.perf_counter()
        role_type = normalize_role_type(role_type)
        chunks, retrieval = await self._jd_context(
            resume_text, role_type, job_context=job_context, job_title=job_title,
            job_company=job_company,
        )
        retrieval["target_job"] = job_title or None
        context, sources = build_context_block(chunks)
        system_prompt = system_prompts.RESUME_OPTIMIZE_RAG_SYSTEM_PROMPT.format(
            role_label=role_label(role_type),
            anti_hallucination_rules=system_prompts.ANTI_HALLUCINATION_RULES,
            context=context or "（无）",
        )
        user_prompt = get_prompts(role_type).RESUME_OPTIMIZE_PROMPT.format(resume_text=resume_text)
        return await self._run(system_prompt, user_prompt, sources, retrieval, started)

    async def score_resume(
        self,
        resume_text: str,
        role_type: str | None = None,
        job_context: str | None = None,
        job_title: str = "",
        job_company: str = "",
    ) -> ServiceResult:
        started = time.perf_counter()
        role_type = normalize_role_type(role_type)
        chunks, retrieval = await self._jd_context(
            resume_text, role_type, job_context=job_context, job_title=job_title,
            job_company=job_company,
        )
        retrieval["target_job"] = job_title or None
        context, sources = build_context_block(chunks)
        system_prompt = system_prompts.RESUME_SCORE_RAG_SYSTEM_PROMPT.format(
            role_label=role_label(role_type),
            anti_hallucination_rules=system_prompts.ANTI_HALLUCINATION_RULES,
            context=context or "（无）",
        )
        user_prompt = get_prompts(role_type).RESUME_SCORE_PROMPT.format(resume_text=resume_text)
        return await self._run(system_prompt, user_prompt, sources, retrieval, started)


# ====================================================================== #
# 模拟面试（多轮对话 + 长上下文）
# ====================================================================== #
class InterviewService:
    """模拟面试：多轮对话、滑动窗口 + 滚动摘要、结构化面试报告。"""

    def __init__(
        self, ai_client: AsyncAIClient | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.ai_client = ai_client or get_ai_client()
        self.memory = ConversationMemory(self.ai_client, self.settings)
        self.kb = KBService(self.ai_client, self.settings)

    async def build_system_prompt(
        self,
        resume_text: str,
        role_type: str,
        reference_top_k: int = 3,
        job_context: str | None = None,
        job_title: str = "",
        job_company: str = "",
    ) -> tuple[str, list[SourceRef]]:
        """构建面试官人格 Prompt，并把题库里的同类题目作为出题参考。

        有目标岗位时，目标岗位 JD 会作为 [1] 号参考资料优先给出，
        面试将围绕该岗位的职责与技能要求展开。
        """
        chunks: list[RetrievedChunk] = []
        if job_context:
            chunks.append(make_job_context_chunk(job_context, job_title, job_company))

        # 优先从用户专属知识库取参考（简历 + 目标岗位 + 同类岗位 + 相关面试题）
        personal = await self.kb.personal_collections()
        reference, _ = await self.kb.retrieve(
            resume_text[:400],
            collections=personal or [self.settings.rag_collection_questions],
            top_k=reference_top_k if personal else reference_top_k,
            role_type=None if personal else normalize_role_type(role_type),
        )
        chunks.extend(reference)
        context, sources = build_context_block(chunks, max_chars=4000)

        job_line = ""
        if job_title:
            company_part = f"（{job_company}）" if job_company else ""
            job_line = system_prompts.TARGET_JOB_LINE.format(
                job_title=job_title, company_part=company_part
            )

        prompt = system_prompts.MOCK_INTERVIEW_SYSTEM_PROMPT.format(
            role_label=role_label(role_type),
            job_line=job_line,
            resume_text=resume_text,
            context=context or "（暂无参考题目，可自行出题）",
            interview_rules=common_prompts.INTERVIEW_RULES,
        )
        return prompt, sources

    async def reply(
        self,
        system_prompt: str,
        history: Sequence[dict[str, Any]],
        summary: str = "",
        summary_upto_seq: int = 0,
        sources: Sequence[SourceRef] | None = None,
    ) -> tuple[ServiceResult, BuiltContext]:
        """基于完整历史生成面试官的下一句。"""
        started = time.perf_counter()
        built = await self.memory.build(
            history, system_prompt, summary=summary, summary_upto_seq=summary_upto_seq
        )
        result = await self.ai_client.chat(
            built.messages, temperature=self.settings.llm_temperature
        )
        if not result.ok:
            return (
                ServiceResult(
                    content="",
                    error=result.error,
                    memory_stats=built.stats,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    mock=result.mock,
                ),
                built,
            )

        # 面试场景不强制引用编号，因此不注入 sources 以避免误报「未引用」
        guarded = guard_answer(result.content, [], extra_warnings=[])
        return (
            ServiceResult(
                content=guarded.content,
                sources=list(sources or []),
                report=guarded.report,
                confidence=guarded.confidence,
                memory_stats=built.stats,
                usage=_merge_usage([result]),
                latency_ms=(time.perf_counter() - started) * 1000,
                model=result.model,
                mock=result.mock,
            ),
            built,
        )

    async def report(
        self,
        history: Sequence[dict[str, Any]],
        resume_text: str,
        role_type: str,
    ) -> ServiceResult:
        """面试结束后生成结构化评估报告。"""
        started = time.perf_counter()
        transcript = "\n".join(
            f"{'候选人' if m.get('role') == 'user' else '面试官'}：{m.get('content')}"
            for m in history
            if m.get("role") in ("user", "assistant")
        )
        system_prompt = system_prompts.INTERVIEW_EVALUATION_SYSTEM_PROMPT.format(
            anti_hallucination_rules=system_prompts.ANTI_HALLUCINATION_RULES
        )
        user_prompt = (
            f"目标岗位方向：{role_label(role_type)}\n\n"
            f"候选人简历：\n{resume_text[:2000]}\n\n"
            f"面试完整对话：\n{transcript}\n\n"
            "请严格依据以上对话内容输出面试报告，不要编造对话中没有出现的表现。"
        )
        result = await self.ai_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.settings.llm_temperature,
        )
        if not result.ok:
            return ServiceResult(
                content="",
                error=result.error,
                latency_ms=(time.perf_counter() - started) * 1000,
                mock=result.mock,
            )
        guarded = guard_answer(result.content, [])
        return ServiceResult(
            content=guarded.content,
            confidence=guarded.confidence,
            usage=_merge_usage([result]),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=result.model,
            mock=result.mock,
        )
