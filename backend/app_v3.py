"""单机版应用装配：一个 FastAPI 进程同时提供 API 与前端页面。

和 v2 的区别：

* **不再需要 Streamlit**：前端是 ``web/`` 下的单页应用，由 FastAPI 直接托管，
  同源、免 CORS、打包成 exe 后只有一个进程。
* **不再有并发闸门**：单机单用户，64 并发的闸门只会碍事；只保留轻量指标。
* **不再连 MySQL**：默认 SQLite，数据落在 exe 同级的 ``data/`` 目录。
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import SEED_DIR, WEB_DIR, get_settings
from paths import LOG_DIR, describe as describe_paths
from rag import get_knowledge_base, load_jsonl, load_seed_documents

from .api_v3 import register_page_routes, router as v3_router
from .db import dispose_database, get_backend_info, init_database, session_scope
from .db import repository
from .llm_config import get_llm_config

logger = logging.getLogger("ai_resume_helper.app")


def _setup_logging() -> None:
    """日志同时写控制台与数据目录，方便用户反馈问题时提供日志。"""
    import sys

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = []
    # Windows 控制台默认 GBK，中文日志会乱码，这里强制 UTF-8
    try:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        handlers.append(stream)
    except Exception:  # noqa: BLE001 - 无控制台（pythonw）时忽略
        pass
    try:
        handlers.append(
            logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        )
    except OSError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers or [logging.NullHandler()],
        force=True,
    )


async def _bootstrap_public_kb() -> None:
    """公共知识库（面试题库 + 岗位 JD 示例）为空时用种子数据建库。"""
    settings = get_settings()
    if not settings.auto_build_kb:
        return
    kb = get_knowledge_base()
    stats = {item["collection"]: item["chunks"] for item in await kb.astats()}
    if all(stats.get(name, 0) > 0 for name in settings.kb_collections):
        return
    documents = load_seed_documents(SEED_DIR)
    for collection, docs in documents.items():
        if not docs:
            continue
        result = await kb.abuild(collection, docs, rebuild=True)
        logger.info("公共知识库 %s：%s 篇 -> %s 块", collection, result.documents, result.chunks)


async def _bootstrap_seed_jobs() -> None:
    """岗位/薪资示例数据为空时导入（仅用于离线演示，界面会标注为示例）。"""
    settings = get_settings()
    if not settings.auto_seed_db:
        return
    async with session_scope() as db:
        if await repository.count_job_postings(db) > 0:
            return
        jobs_file = SEED_DIR / "job_postings.jsonl"
        salaries_file = SEED_DIR / "salary_records.jsonl"
        if jobs_file.exists():
            await repository.upsert_job_postings(db, load_jsonl(jobs_file))
        if salaries_file.exists():
            await repository.upsert_salary_records(db, load_jsonl(salaries_file))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    logger.info("数据目录：%s", describe_paths()["data_home"])

    db_info = await init_database()
    logger.info("数据库：%s", db_info.get("url"))

    await _bootstrap_seed_jobs()
    await _bootstrap_public_kb()

    config = get_llm_config()
    logger.info(
        "大模型：%s / %s（已配置=%s）",
        config.provider, config.resolved_model(), config.configured,
    )
    if not config.configured:
        logger.warning("尚未配置 API Key，请在前端「模型设置」里填写")

    try:
        yield
    finally:
        from .ai_client import reset_ai_client

        await reset_ai_client()
        await dispose_database()
        logger.info("已释放资源")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="单机版 AI 求职助手：定向抓岗 + 薪资行情 + 专属知识库 + 模拟面试",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------- 轻量指标（单机版只做展示，不做限流）---------------- #
    @app.middleware("http")
    async def timing(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("请求异常：%s %s", request.method, request.url.path)
            return JSONResponse(status_code=500, content={"error": "服务内部错误，详见日志"})
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Elapsed-Ms"] = f"{elapsed:.1f}"
        if request.url.path.startswith("/api/") and elapsed > 3000:
            logger.info("慢请求 %.0fms %s %s", elapsed, request.method, request.url.path)
        return response

    app.include_router(v3_router)
    register_page_routes(app)

    # 静态资源（前端 JS/CSS）
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    else:  # pragma: no cover - 打包异常时给出明确提示
        logger.error("前端目录不存在：%s", WEB_DIR)

    @app.get("/health", tags=["系统"])
    async def health() -> dict[str, Any]:
        kb = get_knowledge_base()
        return {
            "status": "ok",
            "version": settings.app_version,
            "database": get_backend_info(),
            "llm": get_llm_config().to_public_dict(),
            "knowledge_base": await kb.astats(),
            "paths": describe_paths(),
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]
