"""单机版 HTTP 接口。

设计原则：

* **单机单用户**：没有登录、没有多租户、没有鉴权。只监听 127.0.0.1。
* **强制流程门禁**：档案没填全 → 抓岗、建库、面试全部返回 428 并说明缺什么。
  前端据此做向导，但**后端每个接口都独立校验**，不能只靠前端。
* **静态页面同源托管**：前端是 FastAPI 自己托管的单页应用，
  不需要另起 Streamlit，打包成 exe 后只有一个进程。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from config import get_settings
from paths import WEB_DIR, describe as describe_paths

from .crawlers import (
    crawl_for_profile,
    load_sources,
    save_sources,
    test_source,
)
from .db import repository, session_scope
from .llm_config import clear_api_key, get_llm_config, save_llm_config
from .llm_providers import provider_list
from .personal_kb import build_personal_kb, personal_kb_stats
from .profile import (
    ProfileIncompleteError,
    load_profile,
    readiness,
    require_interview_ready,
    require_profile_ready,
    save_market_result,
    save_profile,
)

logger = logging.getLogger("ai_resume_helper.api.v3")

router = APIRouter(prefix="/api/v1")


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message, **extra})


# ====================================================================== #
# 页面
# ====================================================================== #
def register_page_routes(app: Any) -> None:
    """把单页前端挂到根路径（同源，免 CORS）。"""

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        index_file = WEB_DIR / "index.html"
        if not index_file.exists():
            return JSONResponse(
                status_code=500,
                content={"error": f"前端文件缺失：{index_file}"},
            )
        return FileResponse(index_file)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Any:
        icon = WEB_DIR / "favicon.svg"
        if icon.exists():
            return FileResponse(icon, media_type="image/svg+xml")
        return JSONResponse(status_code=204, content=None)


# ====================================================================== #
# 系统状态
# ====================================================================== #
@router.get("/system", tags=["系统"])
async def system_info() -> dict[str, Any]:
    from .ai_client import get_ai_client

    settings = get_settings()
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "single_user_mode": settings.single_user_mode,
        "paths": describe_paths(),
        "llm": get_llm_config().to_public_dict(),
        "llm_runtime": get_ai_client().snapshot(),
    }


# ====================================================================== #
# 大模型配置（前端在线填写）
# ====================================================================== #
@router.get("/settings/llm", tags=["设置"])
async def get_llm_settings() -> dict[str, Any]:
    return {
        "config": get_llm_config().to_public_dict(),
        "providers": provider_list(),
    }


@router.put("/settings/llm", tags=["设置"])
async def put_llm_settings(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        config = save_llm_config(
            provider=payload.get("provider"),
            base_url=payload.get("base_url"),
            model=payload.get("model"),
            api_key=payload.get("api_key"),
            temperature=payload.get("temperature"),
            max_tokens=payload.get("max_tokens"),
            persist=bool(payload.get("persist", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    from .ai_client import reset_ai_client

    await reset_ai_client()

    result: dict[str, Any] = {"config": config.to_public_dict()}
    if payload.get("verify"):
        result["verify"] = await _verify_llm()
    return result


@router.post("/settings/llm/verify", tags=["设置"])
async def verify_llm() -> dict[str, Any]:
    return await _verify_llm()


@router.delete("/settings/llm/key", tags=["设置"])
async def delete_llm_key(persist: bool = True) -> dict[str, Any]:
    from .ai_client import reset_ai_client

    config = clear_api_key(persist=persist)
    await reset_ai_client()
    return {"config": config.to_public_dict()}


async def _verify_llm() -> dict[str, Any]:
    """用一次最小代价的真实调用验证 Key 是否可用。"""
    config = get_llm_config()
    if not config.configured:
        return {"ok": False, "error": "尚未配置 API Key 或 Base URL"}
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.resolved_base_url(),
            timeout=min(20.0, config.timeout),
            max_retries=0,
        )
        response = await client.chat.completions.create(
            model=config.resolved_model(),
            messages=[{"role": "user", "content": "回复两个字：可用"}],
            temperature=0.0,
            max_tokens=8,
        )
        await client.close()
        return {
            "ok": True,
            "model": config.resolved_model(),
            "reply_preview": (response.choices[0].message.content or "").strip()[:40],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


# ====================================================================== #
# 用户档案
# ====================================================================== #
@router.get("/profile", tags=["档案"])
async def get_profile() -> dict[str, Any]:
    return await readiness()


@router.put("/profile", tags=["档案"])
async def put_profile(request: Request) -> dict[str, Any]:
    payload = await request.json()
    await save_profile(payload)
    return await readiness()


@router.delete("/profile", tags=["档案"])
async def reset_profile(also_kb: bool = True) -> dict[str, Any]:
    """重置：清空档案（可选连带清掉专属知识库与抓取结果）。

    提供给用户「重新开始」用，也让自动化自检可以重复运行。
    """
    from paths import DATA_HOME

    async with session_scope() as db:
        row = await repository.get_user_profile(db, 1)
        if row is not None:
            row.target_role = ""
            row.expect_city = ""
            row.expect_salary_min = 0.0
            row.expect_salary_max = 0.0
            row.resume_id = None
            row.experience_years = 0.0
            row.education = ""
            row.skills_json = "[]"
            row.target_keywords_json = "[]"
            row.extra_notes = ""
            row.market_summary_json = "{}"
            row.market_updated_at = None
            row.target_job_json = "{}"
        await repository.clear_all_crawled_jobs(db)

    if also_kb:
        try:
            from .personal_kb import personal_kb_stats  # noqa: F401
            from config import get_settings
            from rag import get_knowledge_base

            await get_knowledge_base().areset(get_settings().rag_collection_personal)
        except Exception as exc:  # noqa: BLE001
            logger.warning("清理个人知识库失败：%s", exc)

    return {"ok": True, "state": await readiness(), "data_home": str(DATA_HOME)}


# ====================================================================== #
# 岗位抓取与市场行情
# ====================================================================== #
@router.get("/sources", tags=["岗位来源"])
async def list_sources_api() -> dict[str, Any]:
    return {"sources": load_sources()}


@router.put("/sources", tags=["岗位来源"])
async def put_sources_api(request: Request) -> dict[str, Any]:
    payload = await request.json()
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise HTTPException(status_code=422, detail="sources 必须是数组")
    save_sources(sources)
    return {"sources": load_sources()}


@router.post("/sources/{source_key}/test", tags=["岗位来源"])
async def test_source_api(source_key: str) -> dict[str, Any]:
    try:
        return await test_source(source_key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/market/crawl", tags=["岗位市场"])
async def market_crawl(request: Request) -> Any:
    """按档案抓取真实岗位并统计薪资（这是「必填门禁」的第一道）。"""
    try:
        profile = await require_profile_ready()
    except ProfileIncompleteError as exc:
        return _error(428, str(exc), missing_fields=exc.missing)

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - 允许空请求体
        payload = {}

    source_keys = payload.get("source_keys") or None
    extra_keywords = payload.get("extra_keywords") or None

    report = await crawl_for_profile(
        profile, source_keys=source_keys, extra_keywords=extra_keywords, save=True
    )
    # 把行情快照存进档案，供后续展示与对比
    await save_market_result(report.salary, target_job=profile.target_job or None)
    return report.to_dict()


@router.get("/market/jobs", tags=["岗位市场"])
async def market_jobs(limit: int = 100) -> dict[str, Any]:
    profile = await load_profile()
    async with session_scope() as db:
        rows = await repository.list_crawled_jobs(
            db,
            query_role=profile.target_role or None,
            query_city=profile.expect_city or None,
            limit=limit,
        )
        jobs = [repository.crawled_job_to_dict(row) for row in rows]
    return {"jobs": jobs, "total": len(jobs), "market": profile.market_summary}


@router.post("/market/target", tags=["岗位市场"])
async def choose_target_job(request: Request) -> dict[str, Any]:
    """选定目标岗位（后续建库、面试都围绕它）。"""
    payload = await request.json()
    job_key = str(payload.get("job_key") or "")
    if not job_key:
        raise HTTPException(status_code=422, detail="缺少 job_key")

    async with session_scope() as db:
        rows = await repository.list_crawled_jobs(db, limit=1000)
        job = next(
            (repository.crawled_job_to_dict(row) for row in rows if row.job_key == job_key),
            None,
        )
    if job is None:
        raise HTTPException(status_code=404, detail="没有找到这个岗位，请重新抓取")

    profile = await load_profile()
    await save_market_result(profile.market_summary, target_job=job)
    # 目标岗位变了，旧知识库就失效了，提示前端需要重建
    return {"target_job": job, "need_rebuild_kb": True, "state": await readiness()}


# ====================================================================== #
# 个人知识库
# ====================================================================== #
@router.get("/knowledge/personal", tags=["个人知识库"])
async def get_personal_kb() -> dict[str, Any]:
    return await personal_kb_stats()


@router.post("/knowledge/personal/build", tags=["个人知识库"])
async def build_personal_knowledge_base() -> Any:
    """建立用户专属知识库（第二道门禁：需要档案 + 目标岗位）。"""
    try:
        profile = await require_profile_ready()
    except ProfileIncompleteError as exc:
        return _error(428, str(exc), missing_fields=exc.missing)

    if not (profile.target_job or {}).get("title"):
        return _error(
            428,
            "还没有选定目标岗位，请先抓取岗位并从结果中选一个",
            next_step="market",
        )

    report = await build_personal_kb(profile)
    payload = report.to_dict()
    payload["state"] = await readiness()
    return payload


# ====================================================================== #
# 面试与报告
# ====================================================================== #
@router.post("/interview/start", tags=["模拟面试"])
async def interview_start(request: Request) -> Any:
    """开始模拟面试（第三道门禁：档案 + 市场 + 个人知识库都就绪）。"""
    try:
        profile = await require_profile_ready()
        await require_interview_ready(profile)
    except ProfileIncompleteError as exc:
        return _error(428, str(exc), missing_fields=exc.missing)

    from .services import InterviewService

    payload = await request.json() if await request.body() else {}
    rounds = int(payload.get("rounds") or 5)

    async with session_scope() as db:
        resume = (
            await repository.get_resume(db, profile.resume_id) if profile.resume_id else None
        )

    service = InterviewService()
    system_prompt, sources = await service.build_system_prompt(
        resume.content if resume else "（未提供简历）",
        "tech",
        job_context=None,
        job_title=profile.target_job.get("title", ""),
        job_company=profile.target_job.get("company", ""),
    )
    prompt = (
        f"{system_prompt}\n\n"
        f"【本次要求】请针对以上简历与目标岗位，连续提出 {rounds} 道面试题，"
        "每题给出考察点；先用一句话说明你的整体考察思路，再开始出题。"
    )

    result = await service.ai_client.chat(
        [{"role": "system", "content": prompt}, {"role": "user", "content": "请开始。"}],
        temperature=service.settings.llm_temperature,
    )
    if not result.ok:
        return _error(502, f"大模型调用失败：{result.error}")

    from .anti_hallucination import guard_answer

    guarded = guard_answer(result.content, sources)
    return {
        "questions": guarded.content,
        "sources": [s.to_dict() for s in sources],
        "model": result.model,
        "usage": result.usage,
        "latency_ms": round(result.latency_ms, 1),
        "profile": profile.to_dict(),
    }


@router.post("/interview/answer", tags=["模拟面试"])
async def interview_answer(request: Request) -> Any:
    """提交一轮回答：面试官给出评价 + 下一题。

    长上下文由 ``ConversationMemory`` 管理：窗口保留最近若干轮原文，
    更早的对话压缩成滚动摘要，因此可以聊很多轮而不丢上下文。
    """
    try:
        profile = await require_profile_ready()
        await require_interview_ready(profile)
    except ProfileIncompleteError as exc:
        return _error(428, str(exc), missing_fields=exc.missing)

    payload = await request.json()
    history = payload.get("history") or []
    rounds = int(payload.get("rounds") or 5)
    if not history:
        return _error(422, "缺少对话历史")

    async with session_scope() as db:
        resume = (
            await repository.get_resume(db, profile.resume_id) if profile.resume_id else None
        )

    from .services import InterviewService

    service = InterviewService()
    system_prompt, sources = await service.build_system_prompt(
        resume.content if resume else "（未提供简历）",
        "tech",
        job_context=None,
        job_title=profile.target_job.get("title", ""),
        job_company=profile.target_job.get("company", ""),
    )
    answered = sum(1 for item in history if item.get("role") == "user")
    remaining = max(0, rounds - answered)
    system_prompt += (
        f"\n\n【进度】候选人已答 {answered} 题，计划共 {rounds} 题。"
        + ("请继续提出下一题。" if remaining > 0 else "题数已够，请给出总体评价与改进建议，然后结束面试。")
    )

    history_dicts = [
        {"seq": index + 1, "role": item.get("role"), "content": str(item.get("content") or "")}
        for index, item in enumerate(history)
        if item.get("role") in ("user", "assistant")
    ]

    result, built = await service.reply(system_prompt, history_dicts, sources=sources)
    if result.error:
        return _error(502, f"大模型调用失败：{result.error}")

    return {
        "reply": result.content,
        "memory": result.memory_stats.to_dict() if result.memory_stats else None,
        "summary": built.summary,
        "remaining": remaining,
        "sources": [s.to_dict() for s in sources],
    }


@router.post("/interview/report", tags=["模拟面试"])
async def interview_report(request: Request) -> Any:
    """基于完整对话生成结构化面试报告。"""
    try:
        profile = await require_profile_ready()
    except ProfileIncompleteError as exc:
        return _error(428, str(exc), missing_fields=exc.missing)

    payload = await request.json()
    history = payload.get("history") or []
    if not history:
        return _error(422, "还没有对话内容")

    async with session_scope() as db:
        resume = (
            await repository.get_resume(db, profile.resume_id) if profile.resume_id else None
        )

    from .services import InterviewService

    history_dicts = [
        {"seq": index + 1, "role": item.get("role"), "content": str(item.get("content") or "")}
        for index, item in enumerate(history)
        if item.get("role") in ("user", "assistant")
    ]
    result = await InterviewService().report(
        history_dicts, resume.content if resume else "（未提供简历）", "tech"
    )
    if result.error:
        return _error(502, f"大模型调用失败：{result.error}")
    return result.to_dict()


@router.post("/resume/optimize", tags=["简历"])
async def resume_optimize() -> Any:
    """简历优化（同样需要门禁：优化方向来自目标岗位）。"""
    try:
        profile = await require_profile_ready()
    except ProfileIncompleteError as exc:
        return _error(428, str(exc), missing_fields=exc.missing)

    from .services import ResumeService

    async with session_scope() as db:
        resume = (
            await repository.get_resume(db, profile.resume_id) if profile.resume_id else None
        )
    if resume is None:
        return _error(428, "没有找到简历，请重新上传")

    target = profile.target_job or {}
    job_context = None
    if target.get("title"):
        job_context = render_target_job(target)

    result = await ResumeService().optimize_resume(resume.content, "tech", job_context=job_context)
    if result.error:
        return _error(502, f"大模型调用失败：{result.error}")
    return result.to_dict()


def render_target_job(job: dict[str, Any]) -> str:
    """把选定的目标岗位渲染成可放进 Prompt 的 JD 文本。"""
    lines = [f"{job.get('title', '')}" + (f" @ {job['company']}" if job.get("company") else "")]
    if job.get("city"):
        lines.append(f"工作地点：{job['city']}")
    if job.get("salary_text"):
        lines.append(f"薪资：{job['salary_text']}")
    if job.get("tags"):
        lines.append("技能标签：" + "、".join(str(t) for t in job["tags"]))
    if job.get("jd_text"):
        lines.append("岗位描述：" + str(job["jd_text"])[:3000])
    if job.get("source_url"):
        lines.append(f"原文链接：{job['source_url']}")
    lines.append(f"数据来源：{job.get('source', '未知')}")
    return "\n".join(lines)


# ====================================================================== #
# 简历
# ====================================================================== #
@router.post("/resume/upload", tags=["简历"])
async def upload_resume(request: Request) -> Any:
    """上传简历：支持 multipart 文件（PDF）或 JSON 文本。"""
    import io
    import uuid

    from pdf_reader import extract_text_from_pdf, is_valid_resume_text

    content_type = request.headers.get("content-type", "")
    filename = "resume.pdf"
    text = ""

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return _error(422, "没有收到文件字段 file")
        raw = await upload.read()
        filename = getattr(upload, "filename", None) or "resume.pdf"
        if not raw:
            return _error(422, "文件内容为空")
        if len(raw) > 10 * 1024 * 1024:
            return _error(413, "文件过大（上限 10MB）")
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(io.BytesIO(raw))
        else:
            text = raw.decode("utf-8", errors="ignore")
    else:
        payload = await request.json()
        text = str(payload.get("content") or "")
        filename = str(payload.get("filename") or "resume.txt")

    if not is_valid_resume_text(text):
        return _error(422, f"简历解析失败或内容过短：{(text or '')[:200]}")

    resume_id = str(uuid.uuid4())
    async with session_scope() as db:
        await repository.create_resume(db, resume_id, "local", filename, text)

    await save_profile({"resume_id": resume_id})
    return {
        "resume_id": resume_id,
        "filename": filename,
        "chars": len(text),
        "preview": text[:600],
        "state": await readiness(),
    }


@router.get("/resume", tags=["简历"])
async def get_resume() -> Any:
    profile = await load_profile()
    if not profile.resume_id:
        return _error(404, "尚未上传简历")
    async with session_scope() as db:
        resume = await repository.get_resume(db, profile.resume_id)
    if resume is None:
        return _error(404, "简历记录不存在，请重新上传")
    return {
        "resume_id": resume.id,
        "filename": resume.filename,
        "chars": resume.content_length,
        "created_at": resume.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "content": resume.content,
    }


__all__ = ["register_page_routes", "router"]
