"""网页版（Streamlit 前端）。

    python server.py                  # 1. 启动 FastAPI 后端
    streamlit run app.py              # 2. 启动网页前端

前端只负责交互与展示，所有 AI 能力都在后端：
RAG 知识库检索、Function Calling、多轮对话、长上下文（滑动窗口 + 滚动摘要）。

使用流程（重要）：
    1. 侧边栏把 API Key 填好（不用改 .env、不用重启）
    2. 顶部「共用简历」上传一次简历
    3. 顶部「目标岗位」用关键字搜真实岗位，选一个作为本次求职目标
    4. 再去各个页签做简历优化 / 出题 / 模拟面试 / 评分
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import streamlit as st

from client import (
    BackendClient,
    BackendError,
    format_memory,
    format_sources,
    format_target_job,
    format_tool_calls,
)
from config import get_settings
from prompts import NON_TECH, ROLE_LABELS, TECH

try:
    if st.secrets.get("BACKEND_BASE_URL"):
        os.environ["BACKEND_BASE_URL"] = st.secrets["BACKEND_BASE_URL"]
    if st.secrets.get("SERVICE_API_KEY"):
        os.environ["SERVICE_API_KEY"] = st.secrets["SERVICE_API_KEY"]
except Exception:  # noqa: BLE001 - 本地没有 secrets.toml 时正常跳过
    pass

st.set_page_config(page_title="AI 面试与简历助手", page_icon="🎯", layout="wide")

settings = get_settings()


# ====================================================================== #
# 客户端与状态
# ====================================================================== #
@st.cache_resource(show_spinner=False)
def get_client(base_url: str, api_key: str) -> BackendClient:
    return BackendClient(base_url=base_url, api_key=api_key)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "history": [],
        "resume_id": None,
        "resume_name": "",
        "resume_preview": "",
        "target_job": None,
        "job_results": None,
        "interview_session": None,
        "interview_messages": [],
        "interview_started": False,
        # 输入框轮次计数：每提交一次就换一个组件 key，保证输入框被清空
        "answer_round": 0,
        "rag_round": 0,
        "agent_round": 0,
        "rag_result": None,
        "agent_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


def add_history(mode: str, summary: str) -> None:
    st.session_state.history.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "mode": mode,
            "summary": str(summary)[:120],
        }
    )


def show_error(exc: Exception) -> None:
    """把后端错误翻译成人能看懂的话，而不是甩一段 JSON。"""
    text = str(exc)
    if isinstance(exc, BackendError) and "无法连接后端" in text:
        st.error("❌ 连不上后端服务")
        st.info("请在另一个终端启动后端：")
        st.code("python server.py", language="bash")
        return
    if "401" in text:
        st.error("❌ 鉴权失败：前端的 X-API-Key 与后端的 SERVICE_API_KEY 不一致")
        return
    if "502" in text:
        st.error("❌ 大模型调用失败")
        st.info(
            "常见原因：API Key 无效或过期、账户余额不足、网络不通、模型名写错。\n\n"
            "可以在侧边栏「🔑 模型设置」里点「验证当前 Key」排查。"
        )
        st.caption(text[:500])
        return
    if "503" in text:
        st.warning("⏳ 服务繁忙（并发已满），请稍后重试")
        return
    st.error(f"❌ {text}")


# ====================================================================== #
# 展示组件
# ====================================================================== #
def render_confidence(result: dict[str, Any]) -> None:
    """防幻觉指标：可信度、引用覆盖率、检索命中、耗时。"""
    columns = st.columns(4)
    columns[0].metric("可信度", f"{result.get('confidence', 0) * 100:.0f}%")
    report = result.get("citation_report") or {}
    columns[1].metric("引用覆盖", f"{report.get('citation_rate', 0) * 100:.0f}%")
    retrieval = result.get("retrieval") or {}
    columns[2].metric("检索命中", f"{retrieval.get('hits', 0)} 条")
    columns[3].metric("总耗时", f"{result.get('latency_ms', 0):.0f} ms")

    for warning in result.get("warnings") or []:
        st.warning(f"⚠️ {warning}")
    if result.get("mock"):
        st.info("当前为离线 Mock 模式（未配置 API Key），内容为桩响应。请在侧边栏「🔑 模型设置」填入 Key。")


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"📚 引用来源（{len(sources)} 条，可展开核对原文）", expanded=False):
        for source in sources:
            similarity = source.get("similarity")
            title = f"[{source['index']}] {source['label']}"
            if isinstance(similarity, float):
                title += f"　相似度 {similarity:.3f}"
            title += f"　（{source.get('retrieval')}）"
            st.markdown(f"**{title}**")
            st.caption(source["text"][:600] + ("…" if len(source["text"]) > 600 else ""))
            st.divider()


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        return
    with st.expander(f"🔧 工具调用链（{len(tool_calls)} 次）", expanded=True):
        for call in tool_calls:
            icon = "✅" if call.get("ok") else "❌"
            st.markdown(
                f"{icon} **{call['name']}**　参数 `{json.dumps(call['arguments'], ensure_ascii=False)}`"
                f"　耗时 {call['elapsed_ms']} ms"
            )
            if call.get("ok"):
                with st.expander("查看工具返回的原始数据"):
                    st.json(call.get("result"))
            else:
                st.error(call.get("error"))


def render_memory(memory: dict[str, Any] | None) -> None:
    if memory:
        st.caption(f"🧠 长上下文：{format_memory(memory)}")


def render_job_card(job: dict[str, Any], key_prefix: str) -> None:
    """一张岗位卡片：信息 + 原文链接 + 设为目标岗位。"""
    with st.container(border=True):
        head = st.columns([5, 2])
        head[0].markdown(f"#### {job.get('title', '')}")
        head[1].caption(f"来源：{job.get('source', '未知')}")
        st.caption(
            f"🏢 {job.get('company') or '未提供'}　|　📍 {job.get('location') or '未提供'}"
            f"　|　💰 {job.get('salary') or '未提供'}"
        )
        if job.get("tags"):
            st.caption("🏷️ " + "、".join(str(t) for t in job["tags"][:12]))
        if job.get("matched_keywords"):
            st.caption("🎯 命中关键字：" + "、".join(str(k) for k in job["matched_keywords"]))
        if job.get("posted_at"):
            st.caption(f"🕒 发布：{job['posted_at']}")

        info_col, pick_col = st.columns([3, 1])
        with info_col:
            if job.get("description"):
                with st.expander("岗位描述"):
                    st.write(job["description"])
            if job.get("url"):
                st.markdown(f"[🔗 查看原文]({job['url']})")
        with pick_col:
            if st.button("🎯 设为目标岗位", key=f"{key_prefix}-{job.get('id')}"):
                st.session_state.target_job = job
                st.success("已设为目标岗位")
                st.rerun()


# ====================================================================== #
# 顶部上下文区：共用简历 + 目标岗位
# ====================================================================== #
def render_resume_panel() -> None:
    """共用简历区：只调用一次，避免同一次渲染里重复注册组件 key。"""
    tab_upload, tab_paste, tab_current = st.tabs(["📤 上传 PDF", "⌨️ 粘贴文本", "📄 当前简历"])

    with tab_upload:
        uploaded = st.file_uploader("选择 PDF 简历", type=["pdf"], key="resume_pdf")
        if uploaded is not None and st.button(
            "解析并保存简历", type="primary", key="resume_upload_btn"
        ):
            with st.spinner("正在解析 PDF..."):
                try:
                    result = client.upload_resume_file(uploaded.name, uploaded.getvalue())
                    st.session_state.resume_id = result["resume_id"]
                    st.session_state.resume_name = uploaded.name
                    st.session_state.resume_preview = ""
                    st.success(f"✅ 已保存（{result['chars']} 字），下面所有功能都可直接使用")
                except BackendError as exc:
                    show_error(exc)

    with tab_paste:
        text = st.text_area("粘贴简历文本", height=200, key="resume_paste_text")
        if st.button("保存文本简历", key="resume_paste_btn"):
            if len(text.strip()) < 30:
                st.warning("文本太短（至少 30 字）")
            else:
                try:
                    result = client.create_resume("pasted-resume.txt", text)
                    st.session_state.resume_id = result["resume_id"]
                    st.session_state.resume_name = "pasted-resume.txt"
                    st.session_state.resume_preview = text
                    st.success("✅ 已保存，下面所有功能都可直接使用")
                except BackendError as exc:
                    show_error(exc)

    with tab_current:
        if not st.session_state.resume_id:
            st.info("尚未上传简历。上传后，简历优化 / 面试题 / 评分 / 模拟面试 都会使用它。")
            return

        st.success(f"当前简历：**{st.session_state.resume_name}**")
        if st.button("🔍 查看/核对解析结果", key="resume_preview_btn"):
            try:
                detail = client.get_resume(st.session_state.resume_id)
                st.session_state.resume_preview = detail.get("content", "")
            except BackendError as exc:
                show_error(exc)

        if st.session_state.resume_preview:
            st.caption("下面是后端实际保存并会送给模型的文本，请核对是否有乱码或缺失：")
            st.text_area(
                "简历原文",
                st.session_state.resume_preview,
                height=260,
                disabled=True,
                key="resume_preview_area",
            )

        if st.button("🗑️ 清除当前简历", key="resume_clear_btn"):
            st.session_state.resume_id = None
            st.session_state.resume_name = ""
            st.session_state.resume_preview = ""
            st.rerun()


def render_target_job_panel() -> None:
    """目标岗位区：用关键字检索真实岗位，选一个作为本次求职目标。"""
    current = st.session_state.target_job

    if current:
        st.success(
            f"当前目标岗位：**{current.get('title')}**"
            + (f" @ {current['company']}" if current.get("company") else "")
            + f"　（来源：{current.get('source', '未知')}）"
        )
        if st.button("🗑️ 取消目标岗位", key="job_clear_btn"):
            st.session_state.target_job = None
            st.rerun()
    else:
        st.info(
            "还没有选定目标岗位。建议先用关键字搜一下真实在招岗位——"
            "**模拟面试、面试题、简历评分都会围绕选中的岗位展开**。不选也能用，"
            "只是岗位要求只能来自内置示例库。"
        )

    search_cols = st.columns([3, 1, 1, 1])
    keywords = search_cols[0].text_input(
        "搜索关键字（只匹配职位名称，多个用空格/逗号分隔）",
        value="",
        placeholder="例如：python后端 AI Agent",
        key="job_keywords_input",
    )
    limit = search_cols[1].number_input("条数", 1, 30, 6, key="job_limit_input")
    city = search_cols[2].text_input("城市(可选)", value="", key="job_city_input")
    search_cols[3].write("")
    search_cols[3].write("")
    do_search = search_cols[3].button("🔍 搜索岗位", type="primary", key="job_search_btn")

    with st.expander("⚙️ 数据源设置（这些岗位到底从哪来的）"):
        try:
            sources_info = client.job_sources()
            st.caption(sources_info.get("note", ""))
            picked = st.multiselect(
                "使用哪些数据源",
                options=[item["key"] for item in sources_info["available"]],
                default=None,
                format_func=lambda key: next(
                    (item["label"] for item in sources_info["available"] if item["key"] == key),
                    key,
                ),
                key="job_sources_pick",
            )
            st.caption(
                f"缓存 {sources_info.get('cache_ttl_seconds')} 秒；"
                f"自建 API {'已配置' if sources_info.get('custom_api_configured') else '未配置（JOB_API_BASE）'}"
            )
        except BackendError as exc:
            show_error(exc)
            picked = []

    if do_search:
        if not keywords.strip():
            st.warning("请输入至少一个关键字，例如：python后端、AI、Agent")
        else:
            with st.spinner("正在从公开招聘接口抓取岗位..."):
                try:
                    result = client.jobs(
                        keywords=keywords.strip(),
                        city=(city or "").strip() or None,
                        sources=",".join(picked) if picked else None,
                        limit=int(limit),
                    )
                    st.session_state.job_results = result
                except BackendError as exc:
                    show_error(exc)

    result = st.session_state.job_results
    if not result:
        return

    st.divider()
    jobs = result.get("jobs") or []
    if not jobs:
        st.warning("没有职位名匹配到这些关键字的岗位。")
        st.caption("可以换更通用的词，例如把「python后端」换成 `python`、`backend`、`AI`。")
    else:
        st.markdown(f"**找到 {len(jobs)} 个岗位**（按命中关键字数量排序）")
        for index, job in enumerate(jobs):
            render_job_card(job, f"top-pick-{index}")

    with st.expander("🔎 检索详情（数据源、关键字扩展、失败原因）"):
        st.write("**实际用于匹配的词**（中文关键字会自动扩展英文同义词）：")
        st.json(result.get("keyword_expansion"))
        st.write("**各数据源结果**：")
        for item in result.get("sources") or []:
            icon = "✅" if item.get("ok") else "❌"
            st.markdown(
                f"{icon} **{item['source']}**：抓取 {item.get('fetched')} 条，"
                f"命中 {item.get('matched')} 条，耗时 {item.get('elapsed_ms')} ms"
                + (f"　—　{item.get('error')}" if item.get("error") else "")
            )
            if item.get("note"):
                st.caption(f"　　{item['note']}")
        for note in result.get("notes") or []:
            st.warning(note)


def current_resume_id() -> str | None:
    resume_id = st.session_state.resume_id
    if not resume_id:
        st.info("请先在上方「📄 共用简历」区域上传 PDF 或粘贴简历文本")
    return resume_id


def target_job_context() -> tuple[str | None, str, str]:
    """返回 (job_context, job_title, job_company)。"""
    job = st.session_state.target_job
    if not job:
        return None, "", ""
    return format_target_job(job), job.get("title", ""), job.get("company", "")


# ====================================================================== #
# 侧边栏
# ====================================================================== #
st.sidebar.title("🎯 AI 面试与简历助手")

backend_url = st.sidebar.text_input("后端地址", value=settings.backend_base_url)
api_key = st.sidebar.text_input(
    "X-API-Key（后端未开鉴权时留空）", value=settings.service_api_key, type="password"
)
client = get_client(backend_url, api_key)

with st.sidebar:
    st.divider()
    st.subheader("🩺 服务状态")
    health: dict[str, Any] | None = None
    try:
        health = client.health()
        db = health["database"]
        st.success(
            f"后端在线 · v{health['version']}\n\n"
            f"数据库：`{db['backend']}`" + ("（SQLite 回退）" if db["fallback"] else "")
        )
        total_chunks = sum(item["chunks"] for item in health["knowledge_base"])
        st.caption(f"知识库：{total_chunks} 个文本块")
    except BackendError as exc:
        st.error(f"后端不可用\n\n{exc}")
        st.code("python server.py", language="bash")

    # ---------------- 模型设置：在线填 Key，不用改 .env / 不用重启 ---------------- #
    st.divider()
    llm_ready = bool(health and health["llm"]["configured"] and not health["llm"]["mock_mode"])
    with st.expander("🔑 模型设置（API Key）", expanded=not llm_ready):
        if health:
            llm = health["llm"]
            if llm_ready:
                st.success(f"已接入真实模型：`{llm.get('model', '')}`")
            elif llm.get("configured"):
                st.warning("已配置 Key，但当前仍处于 Mock 模式（LLM_MOCK_MODE=true）")
            else:
                st.warning("未配置 Key，当前是离线 Mock 模式（回答为桩数据）")

        st.caption("填入后**立即生效**，不需要改 .env、也不需要重启后端。")
        new_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            key="llm_key_input",
            help="在 https://platform.deepseek.com/ 的 API Keys 页面创建",
        )
        persist = st.checkbox("同时写入 .env（重启后仍保留）", value=True, key="llm_persist")
        verify = st.checkbox("保存后立即验证一次真实调用", value=True, key="llm_verify_opt")

        save_col, verify_col, clear_col = st.columns(3)
        if save_col.button("💾 保存", type="primary", key="llm_save_btn"):
            if not new_key.strip():
                st.warning("请先粘贴 API Key")
            else:
                with st.spinner("正在保存并验证..."):
                    try:
                        outcome = client.update_llm_settings(
                            api_key=new_key.strip(), persist=persist, verify=verify
                        )
                        status = outcome.get("status", {})
                        st.success(
                            f"✅ 已生效：`{status.get('model')}`"
                            f"（key {status.get('key_masked')}）"
                        )
                        if outcome.get("persisted_env_keys"):
                            st.caption("已写回 .env：" + "、".join(outcome["persisted_env_keys"]))
                        check = outcome.get("verify")
                        if check:
                            if check.get("ok"):
                                st.success(
                                    f"验证通过，模型回复：{check.get('reply_preview')}"
                                    f"（{check.get('latency_ms')} ms）"
                                )
                            else:
                                st.error(f"验证失败：{check.get('error')}")
                        st.rerun()
                    except BackendError as exc:
                        show_error(exc)

        if verify_col.button("🧪 验证当前", key="llm_verify_btn"):
            with st.spinner("正在调用真实模型..."):
                try:
                    check = client.verify_llm()
                    if check.get("ok"):
                        st.success(f"✅ 可用：{check.get('reply_preview')}")
                    else:
                        st.error(f"❌ {check.get('error')}")
                except BackendError as exc:
                    show_error(exc)

        if clear_col.button("🗑️ 清除", key="llm_clear_btn"):
            try:
                client.clear_llm_key(persist=persist)
                st.info("已清除，回到 Mock 模式")
                st.rerun()
            except BackendError as exc:
                show_error(exc)

    st.divider()
    st.subheader("💼 岗位类型")
    role_type = st.selectbox(
        "选择岗位", options=list(ROLE_LABELS.keys()), format_func=lambda k: ROLE_LABELS[k]
    )

    st.divider()
    st.subheader("🕑 操作记录")
    if st.session_state.history:
        for record in reversed(st.session_state.history[-8:]):
            st.write(f"[{record['time']}] {record['mode']}")
    else:
        st.write("暂无记录")


# ====================================================================== #
# 页面
# ====================================================================== #
st.title("🎯 AI 面试与简历助手")
st.caption(
    "RAG 知识库（递归切分 + 重叠、向量 + BM25 混合检索） · "
    "Function Calling（真实公开岗位数据） · 异步高并发后端 · "
    "防幻觉（Prompt 约束 + 低 temperature + 引用校验） · "
    "多轮对话与长上下文（滑动窗口 + 滚动摘要）"
)

with st.expander(
    "📄 共用简历（简历优化 / 面试题 / 评分 / 模拟面试 都使用这一份）",
    expanded=not st.session_state.resume_id,
):
    render_resume_panel()

with st.expander(
    "🎯 目标岗位（先搜岗位、再选目标，下面的功能都会围绕它展开）",
    expanded=bool(st.session_state.target_job) or not st.session_state.target_job,
):
    render_target_job_panel()

tabs = st.tabs(
    [
        "✨ 简历优化",
        "🎯 面试题生成",
        "📊 简历评分",
        "🎤 模拟面试",
        "📚 知识库问答",
        "💼 薪资 / 岗位",
        "⚙️ 知识与系统",
    ]
)


# ---------------- 1. 简历优化 ----------------
with tabs[0]:
    st.subheader("✨ 简历优化（RAG 增强）")
    st.write("结合目标岗位 / 岗位 JD 知识库优化措辞、按 STAR 重组项目经历，并标注岗位要求来源。")
    if st.session_state.target_job:
        st.caption(f"将针对目标岗位：**{st.session_state.target_job.get('title')}**")
    resume_id = current_resume_id()
    if st.button("开始优化", type="primary", disabled=not resume_id, key="optimize_btn"):
        job_context, job_title, job_company = target_job_context()
        with st.spinner("AI 正在检索岗位 JD 并优化..."):
            try:
                result = client.optimize_resume(
                    role_type=role_type,
                    resume_id=resume_id,
                    job_context=job_context,
                )
                st.markdown(result["content"])
                render_confidence(result)
                render_sources(result.get("sources") or [])
                st.download_button(
                    "📥 下载优化结果",
                    result["content"],
                    file_name="optimized_resume.txt",
                    key="dl_optimize",
                )
                add_history("简历优化", result["content"])
            except BackendError as exc:
                show_error(exc)


# ---------------- 2. 面试题生成 ----------------
with tabs[1]:
    st.subheader("🎯 面试题生成（RAG 增强）")
    st.write("参考知识库同类题与目标岗位要求，生成针对简历的面试题、考察点与参考答案要点。")
    if st.session_state.target_job:
        st.caption(f"将针对目标岗位：**{st.session_state.target_job.get('title')}**")
    resume_id = current_resume_id()
    if st.button("生成面试题", type="primary", disabled=not resume_id, key="gen_q"):
        job_context, _t, _c = target_job_context()
        with st.spinner("AI 正在检索题库并生成..."):
            try:
                result = client.interview_questions(
                    role_type=role_type, resume_id=resume_id, job_context=job_context
                )
                st.markdown(result["content"])
                render_confidence(result)
                render_sources(result.get("sources") or [])
                st.download_button(
                    "📥 下载面试题",
                    result["content"],
                    file_name="interview_questions.txt",
                    key="dl_questions",
                )
                add_history("面试题生成", result["content"])
            except BackendError as exc:
                show_error(exc)


# ---------------- 3. 简历评分 ----------------
with tabs[2]:
    st.subheader("📊 简历评分（RAG 增强）")
    st.write("参照目标岗位 / 岗位 JD 知识库的真实要求打分（满分 100），理由可追溯到原文或资料。")
    if st.session_state.target_job:
        st.caption(f"将针对目标岗位：**{st.session_state.target_job.get('title')}**")
    resume_id = current_resume_id()
    if st.button("开始评分", type="primary", disabled=not resume_id, key="score_btn"):
        job_context, _t, _c = target_job_context()
        with st.spinner("AI 正在评分..."):
            try:
                result = client.score_resume(
                    role_type=role_type, resume_id=resume_id, job_context=job_context
                )
                st.markdown(result["content"])
                render_confidence(result)
                render_sources(result.get("sources") or [])
                st.download_button(
                    "📥 下载评分结果",
                    result["content"],
                    file_name="resume_score.txt",
                    key="dl_score",
                )
                add_history("简历评分", result["content"])
            except BackendError as exc:
                show_error(exc)


# ---------------- 4. 模拟面试 ----------------
with tabs[3]:
    st.subheader("🎤 模拟面试（多轮对话 + 长上下文）")
    st.write(
        "AI 扮演面试官进行多轮面试。对话历史用「滑动窗口保留原文 + 早期对话滚动摘要」管理，"
        "因此可以聊很多轮而不丢上下文。"
    )
    target = st.session_state.target_job
    if target:
        st.caption(
            f"面试将围绕目标岗位：**{target.get('title')}**"
            + (f" @ {target['company']}" if target.get("company") else "")
        )
    else:
        st.caption("未选目标岗位——面试只能依据简历与内置示例库出题，建议先在上方选定目标岗位。")

    col_start, col_reset = st.columns([1, 1])
    if not st.session_state.interview_started:
        current_resume_id()

    if col_start.button(
        "开始面试",
        type="primary",
        disabled=st.session_state.interview_started,
        key="interview_start_btn",
    ):
        if not st.session_state.resume_id:
            st.warning("请先上传或粘贴简历")
        else:
            job_context, job_title, job_company = target_job_context()
            with st.spinner("面试官正在准备问题..."):
                try:
                    session = client.create_session(
                        mode="mock_interview",
                        role_type=role_type,
                        title=f"{ROLE_LABELS[role_type]}模拟面试",
                        resume_id=st.session_state.resume_id,
                        job_title=job_title,
                        job_company=job_company,
                        job_context=job_context,
                    )
                    opening = client.start_session(session["session_id"])
                    st.session_state.interview_session = session["session_id"]
                    st.session_state.interview_started = True
                    st.session_state.interview_messages = [opening["reply"]]
                    st.rerun()
                except BackendError as exc:
                    show_error(exc)

    if col_reset.button(
        "🗑️ 清空对话",
        disabled=not st.session_state.interview_started,
        key="interview_reset_btn",
    ):
        session_id = st.session_state.interview_session
        if session_id:
            try:
                client.delete_session(session_id)
            except BackendError:
                pass
        st.session_state.interview_session = None
        st.session_state.interview_started = False
        st.session_state.interview_messages = []
        st.rerun()

    if st.session_state.interview_started:
        question_no = 0
        for message in st.session_state.interview_messages:
            if message["role"] == "assistant":
                question_no += 1
                st.info(f"**【面试官】（第 {question_no} 题）**\n\n{message['content']}")
            else:
                st.success(f"**【你】**\n\n{message['content']}")

        st.caption(f"已进行 {question_no} 题。建议问满 5 题以上再生成报告，评估会更准确。")

        # 输入框必须清空，否则上一轮的答案会残留（用户反馈的问题）。
        # 只靠 st.form(clear_on_submit=True) 不够：提交分支里调用了 st.rerun()，
        # 清空动作会被跳过。所以这里再叠加「轮次 key」——每提交一次就换一个
        # 全新的组件 key，新旧组件互不影响，输入框必然是空的。
        round_no = st.session_state.get("answer_round", 0)
        with st.form(key=f"interview_answer_form_{round_no}", clear_on_submit=True):
            answer = st.text_area("你的回答", height=120, key=f"interview_answer_{round_no}")
            submitted = st.form_submit_button("提交回答", type="primary")

        if submitted:
            if not answer.strip():
                st.warning("请输入你的回答！")
            else:
                with st.spinner("面试官正在思考..."):
                    try:
                        payload = client.send_message(
                            st.session_state.interview_session, answer
                        )
                        st.session_state.interview_messages.append(
                            {"role": "user", "content": answer}
                        )
                        st.session_state.interview_messages.append(payload["reply"])
                        st.session_state.last_memory = payload.get("memory")
                        st.session_state.answer_round = round_no + 1
                        st.rerun()
                    except BackendError as exc:
                        show_error(exc)

        action_cols = st.columns(3)
        if action_cols[0].button("📝 结束并生成报告", key="interview_report_btn"):
            with st.spinner("正在生成结构化面试报告..."):
                try:
                    report = client.session_report(st.session_state.interview_session)
                    st.markdown(report["content"])
                    st.download_button(
                        "📥 下载面试报告",
                        report["content"],
                        file_name="interview_report.txt",
                        key="dl_report",
                    )
                    add_history("面试报告", report["content"])
                except BackendError as exc:
                    show_error(exc)

        if action_cols[1].button("🧠 查看上下文压缩", key="interview_context_btn"):
            try:
                context = client.session_context(st.session_state.interview_session)
                st.caption(f"🧠 {format_memory(context.get('stats'))}")
                st.text_area(
                    "滚动摘要（早期对话被压缩成这样）",
                    context.get("summary") or "（尚未生成摘要）",
                    height=160,
                    disabled=True,
                    key="interview_summary_view",
                )
                with st.expander("送给模型的完整上下文结构"):
                    st.json(context.get("stats"))
            except BackendError as exc:
                show_error(exc)

        if action_cols[2].button(
            "🙈 隐藏会话" if st.session_state.get("hide_transcript") else "👁️ 只看最新",
            key="interview_toggle_btn",
        ):
            st.session_state.hide_transcript = not st.session_state.get("hide_transcript")
            st.rerun()

        render_memory(st.session_state.get("last_memory"))


# ---------------- 5. 知识库问答 ----------------
with tabs[4]:
    st.subheader("📚 知识库问答（RAG + 防幻觉）")
    st.write(
        "先检索面试题库与岗位 JD（向量召回 + BM25 召回 + RRF 融合），再让模型**只依据检索结果**作答，"
        "并强制标注引用来源。检索不到资料时会直接拒答，而不是编造。"
    )

    # 结果从 session_state 渲染：提交后要 st.rerun() 才能让输入框清空，
# 而 rerun 会丢弃本次渲染的输出，所以结果必须存起来下次再画。
    previous = st.session_state.get("rag_result")
    if previous:
        if previous.get("refused"):
            st.warning(previous["content"])
        else:
            st.markdown(previous["content"])
        render_confidence(previous)
        render_sources(previous.get("sources") or [])
        with st.expander("🔍 检索详情"):
            st.json(previous.get("retrieval"))

    rag_round = st.session_state.get("rag_round", 0)
    with st.form(key=f"rag_ask_form_{rag_round}", clear_on_submit=True):
        question = st.text_area(
            "你的问题",
            height=100,
            placeholder="例如：缓存穿透和缓存雪崩分别怎么解决？",
            key=f"rag_question_{rag_round}",
        )
        controls = st.columns([1, 1, 2])
        top_k = controls[0].slider("Top-K", 1, 10, settings.rag_top_k, key="rag_top_k")
        scope = controls[1].selectbox("检索范围", ["全部", "面试题库", "岗位 JD 库"], key="rag_scope")
        controls[2].write("")
        ask = controls[2].form_submit_button("提问", type="primary")

    if ask:
        if not question.strip():
            st.warning("请输入问题")
        else:
            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    collections = {
                        "全部": None,
                        "面试题库": [settings.rag_collection_questions],
                        "岗位 JD 库": [settings.rag_collection_jobs],
                    }[scope]
                    result = client.rag_ask(
                        question, role_type=role_type, top_k=top_k, collections=collections
                    )
                    st.session_state.rag_result = result
                    st.session_state.rag_round = rag_round + 1
                    if not result.get("refused"):
                        add_history("知识库问答", result["content"])
                    st.rerun()
                except BackendError as exc:
                    show_error(exc)


# ---------------- 6. 薪资 / 岗位 ----------------
with tabs[5]:
    st.subheader("💼 薪资 / 岗位查询（Function Calling）")
    st.write(
        "模型会自主决定调用哪些工具（薪资查询 / 岗位搜索 / 知识库检索），"
        "并行执行后把真实数据回传，因此不会凭记忆编造薪资和岗位。"
    )
    st.caption(
        "岗位数据来自免 Key 的公开招聘接口（实时，但以英文远程岗位为主）；"
        "本地示例库只用于离线兜底。详见上方「目标岗位 → 数据源设置」。"
    )

    mode = st.radio("模式", ["让 AI 自主决策", "直接调用工具"], horizontal=True, key="tool_mode")

    if mode == "让 AI 自主决策":
        previous = st.session_state.get("agent_result")
        if previous:
            render_tool_calls(previous.get("tool_calls") or [])
            st.markdown(previous.get("content", ""))
            for warning in previous.get("warnings") or []:
                st.warning(warning)

        agent_round = st.session_state.get("agent_round", 0)
        with st.form(key=f"agent_ask_form_{agent_round}", clear_on_submit=True):
            question = st.text_area(
                "你的问题",
                height=100,
                placeholder="例如：帮我找找 Agent 相关的岗位，另外杭州 Python 后端 3 年薪资多少？",
                key=f"agent_question_{agent_round}",
            )
            strict = st.checkbox(
                "strict 模式（严格 JSON Schema，走 /beta 端点）", value=False, key="tool_strict"
            )
            ask = st.form_submit_button("提问", type="primary")

        if ask:
            if not question.strip():
                st.warning("请输入问题")
            else:
                with st.spinner("AI 正在决策并调用工具..."):
                    try:
                        result = client.tools_ask(question, role_type=role_type, strict=strict)
                        st.session_state.agent_result = result
                        st.session_state.agent_round = agent_round + 1
                        add_history("工具问答", result.get("content", ""))
                        st.rerun()
                    except BackendError as exc:
                        show_error(exc)
    else:
        tool = st.selectbox(
            "选择工具", ["search_jobs（按关键字搜岗位）", "query_salary（查薪资）"], key="direct_tool"
        )
        if tool.startswith("search_jobs"):
            with st.form(key="direct_jobs_form", clear_on_submit=False):
                cols = st.columns([3, 1, 1])
                keywords = cols[0].text_input(
                    "关键字（只匹配职位名称）",
                    value="python backend AI Agent",
                    key="direct_job_keywords",
                )
                limit = cols[1].number_input("条数", 1, 30, 6, key="direct_job_limit")
                city = cols[2].text_input("城市(可选)", value="", key="direct_job_city")
                run = st.form_submit_button("搜索", type="primary")
            if run:
                try:
                    result = client.jobs(
                        keywords=keywords, city=city or None, limit=int(limit)
                    )
                    st.session_state.job_results = result
                    for index, job in enumerate(result.get("jobs") or []):
                        render_job_card(job, f"direct-pick-{index}")
                    if not result.get("jobs"):
                        st.warning("没有职位名匹配到这些关键字，试试更通用的词。")
                except BackendError as exc:
                    show_error(exc)
        else:
            with st.form(key="direct_salary_form", clear_on_submit=False):
                cols = st.columns(3)
                role = cols[0].text_input("岗位", value="Python后端开发工程师", key="salary_role")
                city = cols[1].text_input("城市", value="杭州", key="salary_city")
                level = cols[2].selectbox(
                    "职级", ["（不限）", "初级", "中级", "高级"], key="salary_level"
                )
                run = st.form_submit_button("查询薪资", type="primary")
            if run:
                try:
                    data = client.salary(role, city or None, None if level == "（不限）" else level)
                    if data.get("found"):
                        columns = st.columns(4)
                        percentiles = data.get("percentiles", {})
                        columns[0].metric("P25", f"{percentiles.get('p25')} {data.get('unit')}")
                        columns[1].metric("P50（中位）", f"{percentiles.get('p50')} {data.get('unit')}")
                        columns[2].metric("P75", f"{percentiles.get('p75')} {data.get('unit')}")
                        columns[3].metric("P90", f"{percentiles.get('p90')} {data.get('unit')}")
                        st.caption(
                            f"{data.get('matched_role')} · {data.get('city')} · {data.get('level')}"
                            f"（{data.get('years')}）｜样本量 {data.get('sample_size')}"
                            f"｜{data.get('source')}"
                        )
                        for note in data.get("notes") or []:
                            st.info(note)
                    else:
                        st.warning(data.get("error", "未找到匹配数据"))
                        if data.get("available_roles"):
                            st.caption("可查询的岗位：" + "、".join(data["available_roles"]))
                except BackendError as exc:
                    show_error(exc)

    with st.expander("查看已注册的工具定义（JSON Schema）"):
        try:
            st.json(client.list_tools())
        except BackendError as exc:
            show_error(exc)


# ---------------- 7. 知识与系统 ----------------
with tabs[6]:
    st.subheader("⚙️ 知识库与系统状态")

    st.markdown("### 📖 知识库")
    try:
        stats = client.kb_stats()["collections"]
        columns = st.columns(len(stats) or 1)
        for column, item in zip(columns, stats):
            with column:
                st.metric(f"{item['collection']} 文本块", item["chunks"])
                st.caption(
                    f"chunk_size={item['chunk_size']}　overlap={item['chunk_overlap']}　"
                    f"dim={item['embed_dim']}"
                )
                st.caption(f"平均块长 {item.get('avg_chunk_chars', '-')} 字")
                st.caption(f"更新于 {item.get('updated_at')}")
                if item.get("bm25"):
                    st.caption(f"BM25：{item['bm25']}")
    except BackendError as exc:
        show_error(exc)

    if st.button("🔁 用 data/seed 重建知识库", key="kb_rebuild_btn"):
        with st.spinner("正在递归切分 + 向量化 + 建 BM25 索引..."):
            try:
                result = client.kb_rebuild()
                for item in result["results"]:
                    st.success(
                        f"{item['collection']}：{item['documents']} 篇 → {item['chunks']} 块"
                        f"（{item['seconds']}s）"
                    )
            except BackendError as exc:
                show_error(exc)

    st.divider()
    st.markdown("### 🗂️ 岗位数据源")
    try:
        info = client.job_sources()
        st.caption(info.get("note", ""))
        for source in info["available"]:
            st.write(f"- `{source['key']}`：{source['label']}")
    except BackendError as exc:
        show_error(exc)

    st.divider()
    st.markdown("### 📈 运行时指标（并发能力）")
    if st.button("刷新指标", key="refresh_metrics_btn"):
        st.rerun()
    try:
        metrics = client.metrics()
        app_metrics = metrics["app"]
        columns = st.columns(5)
        columns[0].metric("累计请求", app_metrics["total_requests"])
        columns[1].metric("当前在飞", app_metrics["in_flight"])
        columns[2].metric("峰值并发", app_metrics["peak_in_flight"])
        columns[3].metric("P95 延迟", f"{app_metrics['latency_ms']['p95']} ms")
        columns[4].metric("错误率", f"{app_metrics['error_rate'] * 100:.1f}%")
        st.caption(
            f"并发闸门：上限 {metrics['requests_gate']['limit']}，等待 {metrics['requests_gate']['waiting']}；"
            f"大模型在飞 {metrics['llm']['in_flight']}，峰值 {metrics['llm']['peak_in_flight']}，"
            f"并发上限 {metrics['llm']['concurrency_limit']}"
        )
        with st.expander("完整指标"):
            st.json(metrics)
    except BackendError as exc:
        show_error(exc)

    st.divider()
    st.markdown("### 🗂️ 会话记录")
    try:
        sessions = client.list_sessions()
        if not sessions:
            st.caption("暂无会话")
        for session in sessions:
            columns = st.columns([4, 1, 1, 1])
            label = session["title"]
            if session.get("job_title"):
                label += f"　🎯 {session['job_title']}"
            columns[0].write(f"[{session['mode']}] {label}")
            columns[1].write(f"{session['message_count']} 条")
            columns[2].write(f"摘要 {session['summary_chars']} 字")
            if columns[3].button("删除", key=f"del-{session['id']}"):
                try:
                    client.delete_session(session["id"])
                    st.rerun()
                except BackendError as exc:
                    show_error(exc)
    except BackendError as exc:
        show_error(exc)
