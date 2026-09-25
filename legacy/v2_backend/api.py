"""FastAPI 应用装配与全部 HTTP 路由。

高并发设计：
1. **全异步链路**：路由 -> 服务 -> 异步 ORM / AsyncOpenAI / asyncio.to_thread(Chroma)，
   整条链路没有任何阻塞事件循环的调用。
2. **两道并发闸门**：
   - ``ConcurrencyGate`` 限制同时在处理的请求数，超出排队，超时快速返回 503（防雪崩）。
   - ``AsyncAIClient`` 内部的 Semaphore 限制同时在飞的大模型请求，保护上游配额。
3. **连接池**：SQLAlchemy 异步连接池（pool_size + max_overflow）+ ``pool_pre_ping``。
4. **可观测**：``/metrics`` 实时给出在飞请求数、峰值并发、P50/P90/P95/P99 延迟。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Sequence

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import SEED_DIR, get_settings
from rag import get_knowledge_base, load_jsonl, load_seed_documents

from .ai_client import AIClientError, get_ai_client, reset_ai_client
from .db import dispose_database, get_backend_info, init_database, session_scope
from .db import repository
from .metrics import ConcurrencyGate, get_metrics
from .schemas import (
    AskRequest,
    ChatReplyOut,
    HealthResponse,
    JobSearchRequest,
    KBAddRequest,
    KBRebuildRequest,
    KBSearchRequest,
    LLMSettingsUpdate,
    MessageCreateRequest,
    ResumeCreateRequest,
    ResumeFeatureRequest,
    SalaryQueryRequest,
    SessionCreateRequest,
    ToolAskRequest,
    ToolCallRequestModel,
)
from .services import KBService, ResumeService, ToolService, InterviewService, ServiceResult
from .tools import get_tool_registry, execute_tool_calls
from .ai_client import ToolCallRequest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("ai_resume_helper.api")

settings = get_settings()
metrics = get_metrics()
gate = ConcurrencyGate(settings.max_concurrent_requests, settings.concurrency_acquire_timeout)


# ====================================================================== #
# 请求日志批量写入（避免每个请求都同步写库）
# ====================================================================== #
class RequestLogWriter:
    """把请求日志塞进队列，由后台任务批量落库。"""

    def __init__(self, maxsize: int = 20000) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task[None] | None = None
        self.dropped = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="request-log-writer")

    def submit(self, payload: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            self.dropped += 1

    async def _run(self) -> None:
        while True:
            try:
                batch: list[dict[str, Any]] = [await self.queue.get()]
                await asyncio.sleep(1.0)
                while not self.queue.empty() and len(batch) < 500:
                    batch.append(self.queue.get_nowait())
                if batch:
                    async with session_scope() as db:
                        for item in batch:
                            await repository.add_request_log(db, **item)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 - 日志写入失败绝不能影响业务
                logger.warning("请求日志落库失败：%s", exc)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None


log_writer = RequestLogWriter()


# ====================================================================== #
# 启动时引导：建表 -> 导种子数据 -> 建知识库
# ====================================================================== #
async def _bootstrap_seed_data() -> dict[str, Any]:
    """把岗位 / 薪资种子数据导入数据库（表为空时才做）。"""
    info: dict[str, Any] = {"jobs": 0, "salaries": 0, "skipped": False}
    if not settings.auto_seed_db:
        info["skipped"] = True
        return info

    async with session_scope() as db:
        if await repository.count_job_postings(db) > 0:
            info["skipped"] = True
            return info
        jobs = load_jsonl(SEED_DIR / "job_postings.jsonl")
        salaries = load_jsonl(SEED_DIR / "salary_records.jsonl")
        info["jobs"] = await repository.upsert_job_postings(db, jobs)
        info["salaries"] = await repository.upsert_salary_records(db, salaries)
    logger.info("种子数据导入完成：%s", info)
    return info


async def _bootstrap_knowledge_base() -> list[dict[str, Any]]:
    """知识库为空时用种子数据自动建库。"""
    if not settings.auto_build_kb:
        return []
    kb = get_knowledge_base()
    stats = await kb.astats()
    if all(item["chunks"] > 0 for item in stats):
        return stats

    documents = load_seed_documents(SEED_DIR)
    for collection, docs in documents.items():
        if not docs:
            continue
        result = await kb.abuild(collection, docs, rebuild=True)
        logger.info(
            "知识库 %s 构建完成：%s 篇 -> %s 块（%.2fs）",
            collection,
            result.documents,
            result.chunks,
            result.seconds,
        )
    return await kb.astats()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时优雅释放。"""
    # Chroma 的检索是同步阻塞的，统一丢到线程池执行；
    # 默认线程池只有 min(32, CPU+4) 个线程，并发高时会成为瓶颈，这里显式放大。
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=settings.thread_pool_size, thread_name_prefix="kb-worker"
    )
    loop.set_default_executor(executor)

    db_info = await init_database()
    logger.info("数据库后端：%s", db_info)

    seed_info = await _bootstrap_seed_data()
    kb_stats = await _bootstrap_knowledge_base()
    logger.info("知识库状态：%s", [(s["collection"], s["chunks"]) for s in kb_stats])
    logger.info("大模型：%s", get_ai_client().snapshot())

    log_writer.start()
    app.state.seed_info = seed_info
    try:
        yield
    finally:
        await log_writer.stop()
        await reset_ai_client()
        await dispose_database()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info("资源已释放")


# ====================================================================== #
# 应用
# ====================================================================== #
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI 面试与简历助手后端：RAG 知识库（递归切分+重叠、混合检索）、"
            "Function Calling（薪资/岗位实时数据）、异步高并发、防幻觉（Prompt 约束 + "
            "低 temperature + 引用校验）、多轮对话与长上下文（滑动窗口 + 滚动摘要）。"
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------- 中间件：并发闸门 + 指标 ---------------- #
    @app.middleware("http")
    async def concurrency_and_metrics(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        if path in ("/metrics", "/health"):
            return await call_next(request)

        acquired = await gate.acquire()
        if not acquired:
            metrics.mark_rejected()
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "服务繁忙，请稍后重试",
                    "concurrency_limit": gate.limit,
                    "in_flight": metrics.in_flight,
                },
                headers={"Retry-After": "1"},
            )

        started = time.perf_counter()
        in_flight = metrics.enter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            logger.exception("请求处理异常：%s %s", request.method, path)
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            metrics.leave()
            # 必须归还并发令牌，否则令牌耗尽后所有请求都会被 503 拒绝
            gate.release()
            metrics.record(path, status_code, latency_ms, in_flight)
            if settings.persist_request_logs:
                log_writer.submit(
                    {
                        "path": path,
                        "method": request.method,
                        "status_code": status_code,
                        "latency_ms": latency_ms,
                        "in_flight": in_flight,
                        "client": request.client.host if request.client else "",
                    }
                )

    # ---------------- 鉴权 ---------------- #
    async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.service_api_key and x_api_key != settings.service_api_key:
            raise HTTPException(status_code=401, detail="X-API-Key 无效")

    auth = Depends(verify_api_key)

    # ================= 基础 ================= #
    @app.get("/", tags=["基础"])
    async def root() -> dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "endpoints": [
                "/health",
                "/metrics",
                "/api/v1/knowledge/search",
                "/api/v1/rag/ask",
                "/api/v1/tools/ask",
                "/api/v1/sessions",
            ],
        }

    @app.get("/health", response_model=HealthResponse, tags=["基础"])
    async def health() -> dict[str, Any]:
        kb = get_knowledge_base()
        return {
            "status": "ok",
            "version": settings.app_version,
            "database": get_backend_info(),
            "knowledge_base": await kb.astats(),
            "llm": get_ai_client().snapshot(),
            "concurrency": {
                **gate.snapshot(),
                "max_concurrent_llm": settings.max_concurrent_llm,
                "in_flight": metrics.in_flight,
            },
        }

    @app.get("/metrics", tags=["基础"])
    async def read_metrics() -> dict[str, Any]:
        return {
            "app": metrics.snapshot(),
            "requests_gate": gate.snapshot(),
            "llm": get_ai_client().snapshot(),
            "request_log_writer": {"pending": log_writer.queue.qsize(), "dropped": log_writer.dropped},
        }

    # ================= 知识库 ================= #
    @app.get("/api/v1/knowledge/stats", tags=["知识库"], dependencies=[auth])
    async def knowledge_stats() -> dict[str, Any]:
        kb = get_knowledge_base()
        return {"collections": await kb.astats()}

    @app.post("/api/v1/knowledge/search", tags=["知识库"], dependencies=[auth])
    async def knowledge_search(payload: KBSearchRequest) -> dict[str, Any]:
        kb = get_knowledge_base()
        result = await kb.asearch(
            payload.query,
            collections=payload.collections,
            top_k=payload.top_k,
            role_type=payload.role_type,
            hybrid=payload.hybrid,
        )
        return result.to_dict()

    @app.post("/api/v1/knowledge/documents", tags=["知识库"], dependencies=[auth])
    async def knowledge_add(payload: KBAddRequest) -> dict[str, Any]:
        from rag import Document

        kb = get_knowledge_base()
        documents = [Document(text=doc.text, metadata=doc.metadata) for doc in payload.documents]
        if payload.rebuild:
            result = await kb.abuild(payload.collection, documents, rebuild=True)
        else:
            result = await kb.aadd_documents(payload.collection, documents)
        return {"ok": True, **result.to_dict()}

    @app.post("/api/v1/knowledge/rebuild", tags=["知识库"], dependencies=[auth])
    async def knowledge_rebuild(payload: KBRebuildRequest) -> dict[str, Any]:
        kb = get_knowledge_base()
        if not payload.from_seed:
            raise HTTPException(status_code=400, detail="当前仅支持从 data/seed 重建")
        documents = load_seed_documents(SEED_DIR)
        results = []
        for collection, docs in documents.items():
            if payload.collection and collection != payload.collection:
                continue
            if not docs:
                continue
            results.append((await kb.abuild(collection, docs, rebuild=True)).to_dict())
        return {"ok": True, "results": results, "stats": await kb.astats()}

    @app.delete("/api/v1/knowledge/{collection}", tags=["知识库"], dependencies=[auth])
    async def knowledge_reset(collection: str) -> dict[str, Any]:
        if collection not in settings.kb_collections:
            raise HTTPException(status_code=404, detail=f"未知集合：{collection}")
        kb = get_knowledge_base()
        await kb.areset(collection)
        return {"ok": True, "collection": collection}

    # ================= RAG 问答 ================= #
    @app.post("/api/v1/rag/ask", tags=["RAG 问答"], dependencies=[auth])
    async def rag_ask(payload: AskRequest) -> dict[str, Any]:
        service = KBService()
        result = await service.answer(
            payload.question,
            role_type=payload.role_type,
            collections=payload.collections,
            top_k=payload.top_k,
            temperature=payload.temperature,
        )
        _raise_on_error(result)
        return result.to_dict()

    # ================= Function Calling ================= #
    @app.get("/api/v1/tools", tags=["Function Calling"], dependencies=[auth])
    async def list_tools() -> dict[str, Any]:
        return {"tools": get_tool_registry().describe()}

    @app.post("/api/v1/tools/ask", tags=["Function Calling"], dependencies=[auth])
    async def tools_ask(payload: ToolAskRequest) -> dict[str, Any]:
        service = ToolService()
        result = await service.ask(
            payload.question,
            role_type=payload.role_type,
            use_tools=payload.use_tools,
            strict=payload.strict,
            temperature=payload.temperature,
        )
        _raise_on_error(result)
        return result.to_dict()

    @app.post("/api/v1/tools/call", tags=["Function Calling"], dependencies=[auth])
    async def tools_call(payload: ToolCallRequestModel) -> dict[str, Any]:
        """绕过模型，直接执行某个工具（调试 / 前端直连用）。"""
        call = ToolCallRequest(
            id=f"direct-{uuid.uuid4().hex[:8]}",
            name=payload.name,
            arguments=__import__("json").dumps(payload.arguments, ensure_ascii=False),
        )
        pairs = await execute_tool_calls([call])
        return {"result": pairs[0][1].to_dict()}

    @app.post("/api/v1/tools/salary", tags=["Function Calling"], dependencies=[auth])
    async def tools_salary(payload: SalaryQueryRequest) -> dict[str, Any]:
        from .tools.salary import query_salary

        return await query_salary(
            payload.role, city=payload.city, level=payload.level, years=payload.years
        )

    @app.post("/api/v1/tools/jobs", tags=["Function Calling"], dependencies=[auth])
    async def tools_jobs(payload: JobSearchRequest) -> dict[str, Any]:
        """按关键字检索真实在招岗位（只匹配职位名称）。"""
        from .tools.job_search import search_jobs

        return await search_jobs(
            keywords=payload.keywords,
            city=payload.city,
            sources=payload.sources,
            limit=payload.limit,
            with_description=payload.with_description,
        )

    @app.get("/api/v1/tools/job-sources", tags=["Function Calling"], dependencies=[auth])
    async def job_sources() -> dict[str, Any]:
        """列出岗位数据源清单（让用户知道数据到底从哪来）。"""
        from .tools.job_search import describe_sources

        return await describe_sources()

    # ================= 运行时设置（在线填 API Key） ================= #
    @app.get("/api/v1/settings/llm", tags=["设置"], dependencies=[auth])
    async def get_llm_settings() -> dict[str, Any]:
        """当前大模型配置状态（不返回明文 Key）。"""
        from .runtime_settings import llm_status

        return llm_status()

    @app.put("/api/v1/settings/llm", tags=["设置"], dependencies=[auth])
    async def put_llm_settings(payload: LLMSettingsUpdate) -> dict[str, Any]:
        """在线修改大模型配置，立即生效，可选择写回 .env 并当场验证。"""
        from .runtime_settings import update_llm_settings

        try:
            return await update_llm_settings(
                api_key=payload.api_key,
                base_url=payload.base_url,
                model=payload.model,
                temperature=payload.temperature,
                persist=payload.persist,
                verify=payload.verify,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/settings/llm/verify", tags=["设置"], dependencies=[auth])
    async def verify_llm() -> dict[str, Any]:
        """用一次最小代价的真实调用校验当前 Key 是否可用。"""
        from .runtime_settings import verify_llm_connection

        return await verify_llm_connection()

    @app.delete("/api/v1/settings/llm/key", tags=["设置"], dependencies=[auth])
    async def delete_llm_key(persist: bool = False) -> dict[str, Any]:
        """清空 Key，回到离线 Mock 模式。"""
        from .runtime_settings import clear_llm_key

        try:
            return await clear_llm_key(persist=persist)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    # ================= 简历 ================= #
    @app.post("/api/v1/resumes", tags=["简历"], dependencies=[auth])
    async def create_resume(payload: ResumeCreateRequest, x_user_id: str = Header(default="anonymous")) -> dict[str, Any]:
        resume_id = str(uuid.uuid4())
        async with session_scope() as db:
            await repository.create_resume(
                db, resume_id, x_user_id, payload.filename, payload.content
            )
        return {"resume_id": resume_id, "chars": len(payload.content)}

    @app.post("/api/v1/resumes/upload", tags=["简历"], dependencies=[auth])
    async def upload_resume(
        file: UploadFile = File(...), x_user_id: str = Header(default="anonymous")
    ) -> dict[str, Any]:
        """上传 PDF 简历，后端用 pdfplumber 解析后入库。"""
        import io

        from pdf_reader import extract_text_from_pdf, is_valid_resume_text

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="文件内容为空")
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="文件过大（上限 10MB）")

        text = extract_text_from_pdf(io.BytesIO(raw))
        if not is_valid_resume_text(text):
            raise HTTPException(
                status_code=422,
                detail=f"PDF 解析失败或内容过短：{text[:200] if text else '空内容'}",
            )

        resume_id = str(uuid.uuid4())
        async with session_scope() as db:
            await repository.create_resume(db, resume_id, x_user_id, file.filename or "resume.pdf", text)
        return {"resume_id": resume_id, "filename": file.filename, "chars": len(text)}

    @app.get("/api/v1/resumes/{resume_id}", tags=["简历"], dependencies=[auth])
    async def read_resume(resume_id: str) -> dict[str, Any]:
        async with session_scope() as db:
            resume = await repository.get_resume(db, resume_id)
        if resume is None:
            raise HTTPException(status_code=404, detail="简历不存在")
        return {
            "resume_id": resume.id,
            "filename": resume.filename,
            "chars": resume.content_length,
            "created_at": resume.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "content": resume.content,
        }

    async def _resolve_resume_text(payload: ResumeFeatureRequest) -> str:
        if payload.resume_text and len(payload.resume_text.strip()) >= 30:
            return payload.resume_text
        if payload.resume_id:
            async with session_scope() as db:
                resume = await repository.get_resume(db, payload.resume_id)
            if resume is None:
                raise HTTPException(status_code=404, detail="简历不存在")
            return resume.content
        raise HTTPException(status_code=422, detail="必须提供 resume_text 或 resume_id")

    @app.post("/api/v1/resume/optimize", tags=["简历"], dependencies=[auth])
    async def resume_optimize(payload: ResumeFeatureRequest) -> dict[str, Any]:
        text = await _resolve_resume_text(payload)
        result = await ResumeService().optimize_resume(
            text,
            payload.role_type,
            job_context=payload.job_context,
            job_title=_job_title_from_context(payload.job_context),
        )
        _raise_on_error(result)
        return result.to_dict()

    @app.post("/api/v1/resume/questions", tags=["简历"], dependencies=[auth])
    async def resume_questions(payload: ResumeFeatureRequest) -> dict[str, Any]:
        text = await _resolve_resume_text(payload)
        result = await ResumeService().generate_interview_questions(
            text,
            payload.role_type,
            job_context=payload.job_context,
            job_title=_job_title_from_context(payload.job_context),
        )
        _raise_on_error(result)
        return result.to_dict()

    @app.post("/api/v1/resume/score", tags=["简历"], dependencies=[auth])
    async def resume_score(payload: ResumeFeatureRequest) -> dict[str, Any]:
        text = await _resolve_resume_text(payload)
        result = await ResumeService().score_resume(
            text,
            payload.role_type,
            job_context=payload.job_context,
            job_title=_job_title_from_context(payload.job_context),
        )
        _raise_on_error(result)
        return result.to_dict()

    # ================= 多轮对话 / 模拟面试 ================= #
    @app.post("/api/v1/sessions", tags=["多轮对话"], dependencies=[auth])
    async def create_session(
        payload: SessionCreateRequest, x_user_id: str = Header(default="anonymous")
    ) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        resume_id = payload.resume_id
        async with session_scope() as db:
            if payload.resume_text and len(payload.resume_text.strip()) >= 30:
                resume_id = str(uuid.uuid4())
                await repository.create_resume(
                    db, resume_id, x_user_id, "session-resume.txt", payload.resume_text
                )
            await repository.create_chat_session(
                db,
                session_id,
                x_user_id,
                payload.mode,
                payload.role_type,
                title=payload.title or f"{payload.role_type} {payload.mode}",
                resume_id=resume_id,
                job_title=payload.job_title,
                job_company=payload.job_company,
                job_context=payload.job_context or "",
            )
        return {
            "session_id": session_id,
            "resume_id": resume_id,
            "mode": payload.mode,
            "job_title": payload.job_title,
            "job_company": payload.job_company,
            "has_job_context": bool(payload.job_context),
        }

    @app.get("/api/v1/sessions", tags=["多轮对话"], dependencies=[auth])
    async def list_sessions(x_user_id: str | None = Header(default=None), limit: int = 50) -> dict[str, Any]:
        async with session_scope() as db:
            sessions = await repository.list_chat_sessions(db, x_user_id, limit=limit)
        return {"sessions": [repository.session_to_dict(s) for s in sessions]}

    @app.get("/api/v1/sessions/{session_id}", tags=["多轮对话"], dependencies=[auth])
    async def get_session(session_id: str) -> dict[str, Any]:
        async with session_scope() as db:
            session = await repository.get_chat_session(db, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            count = await repository.count_messages(db, session_id)
        return repository.session_to_dict(session, message_count=count)

    @app.get("/api/v1/sessions/{session_id}/messages", tags=["多轮对话"], dependencies=[auth])
    async def get_messages(session_id: str, limit: int = 200) -> dict[str, Any]:
        async with session_scope() as db:
            rows = await repository.list_messages(db, session_id, limit=limit)
        return {"messages": [repository.message_to_dict(row) for row in rows]}

    @app.delete("/api/v1/sessions/{session_id}", tags=["多轮对话"], dependencies=[auth])
    async def remove_session(session_id: str) -> dict[str, Any]:
        async with session_scope() as db:
            await repository.delete_chat_session(db, session_id)
        return {"ok": True}

    @app.get("/api/v1/sessions/{session_id}/context", tags=["多轮对话"], dependencies=[auth])
    async def session_context(session_id: str) -> dict[str, Any]:
        """查看长上下文压缩结果：摘要 + 窗口原文 + 压缩比。"""
        from .memory import context_preview

        async with session_scope() as db:
            session = await repository.get_chat_session(db, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            rows = await repository.list_messages(db, session_id)
            resume = (
                await repository.get_resume(db, session.resume_id) if session.resume_id else None
            )

        history = [repository.message_to_dict(row) for row in rows]
        service = InterviewService()
        system_prompt, sources = await service.build_system_prompt(
            resume.content if resume else "（未提供简历）",
            session.role_type,
            job_context=session.job_context or None,
            job_title=session.job_title or "",
            job_company=session.job_company or "",
        )
        built = await service.memory.build(
            history, system_prompt, session.summary, session.summary_upto_seq
        )
        return {
            "session_id": session_id,
            "reference_sources": [s.to_dict() for s in sources],
            **context_preview(built),
        }

    @app.post("/api/v1/sessions/{session_id}/start", tags=["多轮对话"], dependencies=[auth])
    async def start_session(session_id: str) -> dict[str, Any]:
        """让面试官开场（不写入用户消息，保持对话记录干净）。"""
        async with session_scope() as db:
            session = await repository.get_chat_session(db, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            if session.mode != "mock_interview":
                raise HTTPException(status_code=400, detail="仅模拟面试需要开场")
            existing = await repository.count_messages(db, session_id)
            if existing > 0:
                raise HTTPException(status_code=409, detail="会话已经开始")
            resume = (
                await repository.get_resume(db, session.resume_id) if session.resume_id else None
            )

        resume_text = resume.content if resume else "（候选人未提供简历，请先询问其背景）"
        service = InterviewService()
        system_prompt, sources = await service.build_system_prompt(
            resume_text,
            session.role_type,
            job_context=session.job_context or None,
            job_title=session.job_title or "",
            job_company=session.job_company or "",
        )
        result, built = await service.reply(
            system_prompt, [], summary=session.summary, summary_upto_seq=session.summary_upto_seq
        )
        _raise_on_error(result)

        seq = 1
        async with session_scope() as db:
            message = await repository.add_message(
                db,
                session_id,
                seq,
                "assistant",
                result.content,
                sources=[s.to_dict() for s in sources],
                token_estimate=_tokens(result.content),
                latency_ms=result.latency_ms,
            )
            await repository.touch_session(db, session_id, seq)
            payload = repository.message_to_dict(message)

        return {"session_id": session_id, "reply": payload, "result": result.to_dict()}

    @app.post("/api/v1/sessions/{session_id}/messages", response_model=ChatReplyOut, tags=["多轮对话"], dependencies=[auth])
    async def post_message(session_id: str, payload: MessageCreateRequest) -> dict[str, Any]:
        """发送一条消息，返回 AI 的回复（含引用、工具调用、长上下文指标）。"""
        async with session_scope() as db:
            session = await repository.get_chat_session(db, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            rows = await repository.list_messages(db, session_id)
            resume = (
                await repository.get_resume(db, session.resume_id) if session.resume_id else None
            )

        history = [repository.message_to_dict(row) for row in rows]
        next_seq = (history[-1]["seq"] + 1) if history else 1
        user_seq = next_seq
        history.append({"seq": user_seq, "role": "user", "content": payload.content})

        service = InterviewService()
        resume_text = resume.content if resume else "（候选人未提供简历）"

        if session.mode == "mock_interview":
            system_prompt, sources = await service.build_system_prompt(
                resume_text,
                session.role_type,
                job_context=session.job_context or None,
                job_title=session.job_title or "",
                job_company=session.job_company or "",
            )
            result, built = await service.reply(
                system_prompt,
                history,
                summary=session.summary,
                summary_upto_seq=session.summary_upto_seq,
                sources=sources,
            )
            context_snapshot = None
        else:
            # 通用知识库对话：先检索再回答，可选工具
            kb_service = KBService()
            chunks, retrieve_ms = await kb_service.retrieve(
                payload.content,
                collections=None,
                top_k=payload.top_k,
                role_type=session.role_type,
            )
            from .anti_hallucination import build_context_block
            from prompts import system as system_prompts

            context, sources = build_context_block(chunks)
            from .services import role_label, normalize_role_type

            system_prompt = system_prompts.RAG_SYSTEM_PROMPT.format(
                role_label=role_label(session.role_type),
                anti_hallucination_rules=system_prompts.ANTI_HALLUCINATION_RULES,
                context=context or "（无）",
            )
            if payload.use_tools:
                system_prompt += "\n\n" + system_prompts.TOOL_AGENT_SYSTEM_PROMPT.format(
                    anti_hallucination_rules=""
                )
            built = await service.memory.build(
                history, system_prompt, session.summary, session.summary_upto_seq
            )
            from .tools import tool_schemas
            from .ai_client import get_ai_client

            ai_client = get_ai_client()
            tools = tool_schemas() if (payload.use_tools and settings.tools_enabled) else None
            llm = await ai_client.chat(built.messages, tools=tools)
            tool_calls: list[dict[str, Any]] = []
            if llm.ok and llm.tool_calls:
                pairs = await execute_tool_calls(llm.tool_calls)
                tool_calls = [execution.to_dict() for _, execution in pairs]
                working = list(built.messages) + [
                    {
                        "role": "assistant",
                        "content": llm.content or "",
                        "tool_calls": [c.to_message_dict() for c in llm.tool_calls],
                    }
                ]
                for call, execution in pairs:
                    working.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": execution.name,
                            "content": execution.to_content(),
                        }
                    )
                llm = await ai_client.chat(working)

            from .anti_hallucination import guard_answer
            from .memory import MemoryStats

            if not llm.ok:
                result = ServiceResult(content="", error=llm.error, memory_stats=built.stats)
            else:
                guarded = guard_answer(llm.content, sources)
                result = ServiceResult(
                    content=guarded.content,
                    sources=guarded.sources,
                    report=guarded.report,
                    confidence=guarded.confidence,
                    warnings=guarded.warnings,
                    tool_calls=tool_calls,
                    memory_stats=built.stats,
                    retrieval={
                        "query": payload.content,
                        "hits": len(chunks),
                        "elapsed_ms": round(retrieve_ms, 2),
                    },
                    usage={
                        "total_tokens": (llm.usage or {}).get("total_tokens", 0),
                        "llm_calls": 1,
                    },
                    latency_ms=llm.latency_ms,
                    model=llm.model,
                    mock=llm.mock,
                )
            context_snapshot = None

        _raise_on_error(result)

        assistant_seq = user_seq + 1
        async with session_scope() as db:
            await repository.add_message(
                db, session_id, user_seq, "user", payload.content, token_estimate=_tokens(payload.content)
            )
            message = await repository.add_message(
                db,
                session_id,
                assistant_seq,
                "assistant",
                result.content,
                sources=[s.to_dict() for s in result.sources],
                tool_calls=result.tool_calls,
                citations=list(result.report.cited) if result.report else [],
                token_estimate=_tokens(result.content),
                latency_ms=result.latency_ms,
            )
            await repository.touch_session(db, session_id, assistant_seq)
            if built.stats.summary_updated:
                await repository.update_session_summary(
                    db, session_id, built.summary, built.summary_upto_seq
                )
            reply_payload = repository.message_to_dict(message)

        return {
            "session_id": session_id,
            "reply": reply_payload,
            "result": result.to_dict(),
            "memory": result.memory_stats.to_dict() if result.memory_stats else None,
            "context_snapshot": context_snapshot,
        }

    @app.post("/api/v1/sessions/{session_id}/report", tags=["多轮对话"], dependencies=[auth])
    async def session_report(session_id: str) -> dict[str, Any]:
        """生成结构化面试评估报告。"""
        async with session_scope() as db:
            session = await repository.get_chat_session(db, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            rows = await repository.list_messages(db, session_id)
            resume = (
                await repository.get_resume(db, session.resume_id) if session.resume_id else None
            )
        history = [repository.message_to_dict(row) for row in rows]
        if not history:
            raise HTTPException(status_code=409, detail="会话还没有任何对话内容")
        result = await InterviewService().report(
            history, resume.content if resume else "（未提供简历）", session.role_type
        )
        _raise_on_error(result)
        return result.to_dict()

    # ================= 运维统计 ================= #
    @app.get("/api/v1/stats", tags=["基础"], dependencies=[auth])
    async def system_stats() -> dict[str, Any]:
        async with session_scope() as db:
            jobs = await repository.count_job_postings(db)
            salaries = await repository.count_salary_records(db)
            kb_docs = await repository.count_kb_documents(db)
            recent = await repository.request_stats(db, minutes=60)
        return {
            "job_postings": jobs,
            "salary_records": salaries,
            "kb_documents": kb_docs,
            "recent_requests": recent,
            "metrics": metrics.snapshot(),
        }

    @app.get("/api/v1/stats/baseline", tags=["基础"])
    async def service_baseline() -> dict[str, Any]:
        """并发能力基线：把当前生效的并发配置一次性暴露出来，便于对比压测结果。"""
        return {
            "max_concurrent_requests": settings.max_concurrent_requests,
            "max_concurrent_llm": settings.max_concurrent_llm,
            "db_pool_size": settings.db_pool_size,
            "db_max_overflow": settings.db_max_overflow,
            "uvicorn": "建议 --workers 1 + 异步，或按 CPU 核数横向扩展",
            "current": metrics.snapshot(),
        }

    return app


# ====================================================================== #
# 工具函数
# ====================================================================== #
def _raise_on_error(result: ServiceResult) -> None:
    if result.error:
        raise HTTPException(status_code=502, detail=f"大模型调用失败：{result.error}")


def _tokens(text: str) -> int:
    from rag import estimate_tokens

    return estimate_tokens(text)


def _job_title_from_context(job_context: str | None) -> str:
    """从目标岗位 JD 文本里取第一行做标题（前端会把标题写在第一行）。"""
    if not job_context:
        return ""
    first_line = job_context.strip().splitlines()[0] if job_context.strip() else ""
    return first_line.lstrip("# ").strip()[:120]


app = create_app()
