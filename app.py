import os
from datetime import datetime

import streamlit as st

from pdf_reader import extract_text_from_pdf, is_valid_resume_text
from ai_client import AIClient
from prompts import get_prompts, ROLE_LABELS, TECH, NON_TECH

# 将 Streamlit Secrets 中的 API Key 注入环境变量，供 AIClient 读取
# 本地开发：在 .streamlit/secrets.toml 配置；云端：在 Secrets 面板配置
if st.secrets.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]


# === 历史记录（保存在会话内，避免云端多用户共写文件导致冲突/丢失） ===
def save_history(mode, resume, result):
    """把使用记录追加到当前会话的历史列表"""
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "resume": resume[:200],   # 只保留前 200 字
        "result": result[:500]    # 只保留前 500 字
    }
    st.session_state.history.append(record)


def load_history():
    """读取当前会话的历史记录"""
    return st.session_state.history


# === 页面配置 ===
st.set_page_config(
    page_title="AI简历优化助手",
    page_icon="📝",
    layout="centered"
)

# 会话级历史记录初始化（每个用户会话独立，互不影响）
if "history" not in st.session_state:
    st.session_state.history = []

# === 标题区 ===
st.title("📝 AI简历优化助手")
st.write("基于 DeepSeek 大模型，上传你的 PDF 简历，AI 帮你优化措辞、生成面试题、模拟面试")
st.divider()

# === 侧边栏：功能导航 + 使用说明 + 历史记录 ===
st.sidebar.title("🧭 功能导航")
mode = st.sidebar.radio(
    "选择功能",
    ["简历优化", "生成面试题", "模拟面试", "简历评分"]
)

st.sidebar.divider()
st.sidebar.header("💼 岗位类型")
role_type = st.sidebar.selectbox(
    "选择面试岗位",
    options=list(ROLE_LABELS.keys()),
    format_func=lambda k: ROLE_LABELS[k],
)

st.sidebar.divider()
st.sidebar.header("📖 使用说明")
st.sidebar.markdown(
    "1. 上传文本型 PDF 简历（非扫描图片）\n"
    "2. 在左侧选择功能后点击对应按钮\n"
    "3. 模拟面试为多轮对话，可点击「清空对话」重置\n"
    "4. 需配置 `DEEPSEEK_API_KEY`（环境变量或 secrets）"
)

st.sidebar.divider()
st.sidebar.header("🕑 历史记录")
history = load_history()
if history:
    # 显示最近 5 条
    for record in reversed(history[-5:]):
        st.sidebar.write(f"[{record['time']}] {record['mode']}")
else:
    st.sidebar.write("暂无历史记录")

# === 上传区（所有功能共用） ===
st.header("📤 上传简历 PDF")
uploaded_file = st.file_uploader(
    label="选择你的 PDF 简历文件",
    type=["pdf"],
    help="仅支持 .pdf 格式文件"
)

if uploaded_file is not None:
    st.caption(f"📁 文件名：{uploaded_file.name}")
    st.caption(f"📦 文件大小：{uploaded_file.size} 字节")

st.divider()


def read_resume_or_warn():
    """读取上传的 PDF 简历，成功返回文本，失败返回 None 并显示错误"""
    if uploaded_file is None:
        st.warning("请先上传 PDF 简历文件！")
        return None

    with st.spinner("正在读取 PDF 内容..."):
        resume_text = extract_text_from_pdf(uploaded_file)

    if not is_valid_resume_text(resume_text):
        st.error(f"❌ {resume_text if resume_text else 'PDF 内容为空'}")
        st.info("请确认上传的是文本型 PDF（非扫描图片），且包含简历内容")
        return None

    st.success(f"✅ 成功读取简历（{len(resume_text)} 字）")
    return resume_text


def render_resume_optimize():
    """简历优化功能"""
    st.header("✨ 简历优化")
    st.write("AI 帮你改进措辞、量化成果、按 STAR 法则重组项目经历，并给出 3 条改进建议")

    if st.button("开始优化", type="primary"):
        resume_text = read_resume_or_warn()
        if resume_text is None:
            return

        with st.spinner("AI 正在分析和优化你的简历，请稍候..."):
            try:
                client = AIClient()
                prompt = get_prompts(role_type).RESUME_OPTIMIZE_PROMPT.format(resume_text=resume_text)
                result = client.chat(prompt)

                if result.startswith("AI调用失败"):
                    st.error(f"❌ {result}")
                    return

                st.success("✅ 优化完成！")
                st.markdown(result)
                st.download_button(
                    label="📥 下载优化结果",
                    data=result,
                    file_name="optimized_resume.txt",
                    mime="text/plain"
                )
                save_history("简历优化", resume_text, result)
            except ValueError as e:
                st.error(f"❌ {e}")
                st.info("请设置 DEEPSEEK_API_KEY（环境变量或 Streamlit Secrets）")


def render_generate_questions():
    """生成面试题功能"""
    st.header("🎯 面试题生成")
    st.write("根据简历内容生成 5 个针对性面试问题（技术能力 + 项目经验 + 综合素质）")

    if st.button("生成面试题", type="primary"):
        resume_text = read_resume_or_warn()
        if resume_text is None:
            return

        with st.spinner("AI 正在生成面试题..."):
            try:
                client = AIClient()
                prompt = get_prompts(role_type).INTERVIEW_QUESTIONS_PROMPT.format(resume_text=resume_text)
                result = client.chat(prompt)

                if result.startswith("AI调用失败"):
                    st.error(f"❌ {result}")
                    return

                st.success("✅ 生成完成！")
                st.markdown(result)
                st.download_button(
                    label="📥 下载面试题",
                    data=result,
                    file_name="interview_questions.txt",
                    mime="text/plain"
                )
                save_history("生成面试题", resume_text, result)
            except ValueError as e:
                st.error(f"❌ {e}")
                st.info("请设置 DEEPSEEK_API_KEY（环境变量或 Streamlit Secrets）")


def render_mock_interview():
    """模拟面试功能（多轮对话）"""
    st.header("🎤 模拟面试")
    st.write("AI 扮演面试官，根据你的简历进行多轮对话式面试")

    # session_state 初始化：用 OpenAI messages 格式保存完整对话
    if "interview_messages" not in st.session_state:
        st.session_state.interview_messages = []
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False

    # 清空对话按钮（始终显示，方便随时重置）
    if st.session_state.interview_started:
        if st.button("🗑️ 清空对话", type="secondary"):
            st.session_state.interview_messages = []
            st.session_state.interview_started = False
            st.rerun()

    # 开始面试按钮
    if not st.session_state.interview_started:
        if st.button("开始面试", type="primary"):
            resume_text = read_resume_or_warn()
            if resume_text is None:
                return

            with st.spinner("面试官正在准备问题，请稍候..."):
                try:
                    client = AIClient()
                    system_prompt = get_prompts(role_type).MOCK_INTERVIEW_SYSTEM_PROMPT.format(resume_text=resume_text)
                    messages = [{"role": "system", "content": system_prompt}]
                    first_question = client.chat_with_history(messages)

                    if first_question.startswith("AI调用失败"):
                        st.error(f"❌ {first_question}")
                        return

                    messages.append({"role": "assistant", "content": first_question})
                    st.session_state.interview_messages = messages
                    st.session_state.interview_started = True
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.info("请设置 DEEPSEEK_API_KEY（环境变量或 Streamlit Secrets）")
        return

    # 显示对话历史（跳过 system 消息），用计数器生成题号
    q_num = 0
    for msg in st.session_state.interview_messages:
        if msg["role"] == "assistant":
            q_num += 1
            st.info(f"【面试官】（第{q_num}题）\n{msg['content']}")
        elif msg["role"] == "user":
            st.success(f"【你】\n{msg['content']}")

    # 用户回答输入区
    user_answer = st.text_area("你的回答：", height=100)
    if st.button("提交回答", type="primary"):
        if not user_answer.strip():
            st.warning("请输入你的回答！")
        else:
            with st.spinner("面试官正在思考..."):
                try:
                    client = AIClient()
                    messages = st.session_state.interview_messages + [
                        {"role": "user", "content": user_answer}
                    ]
                    response = client.chat_with_history(messages)

                    if response.startswith("AI调用失败"):
                        st.error(f"❌ {response}")
                        return

                    messages.append({"role": "assistant", "content": response})
                    st.session_state.interview_messages = messages
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.info("请设置 DEEPSEEK_API_KEY（环境变量或 Streamlit Secrets）")


def render_resume_score():
    """简历评分功能"""
    st.header("📊 简历评分")
    st.write("AI 从 5 个维度给简历打分（满分 100 分），并给出改进建议")

    if st.button("开始评分", type="primary"):
        resume_text = read_resume_or_warn()
        if resume_text is None:
            return

        with st.spinner("AI 正在给简历打分，请稍候..."):
            try:
                client = AIClient()
                prompt = get_prompts(role_type).RESUME_SCORE_PROMPT.format(resume_text=resume_text)
                result = client.chat(prompt)

                if result.startswith("AI调用失败"):
                    st.error(f"❌ {result}")
                    return

                st.success("✅ 评分完成！")
                st.markdown(result)
                st.download_button(
                    label="📥 下载评分结果",
                    data=result,
                    file_name="resume_score.txt",
                    mime="text/plain"
                )
                save_history("简历评分", resume_text, result)
            except ValueError as e:
                st.error(f"❌ {e}")
                st.info("请设置 DEEPSEEK_API_KEY（环境变量或 Streamlit Secrets）")


# === 根据侧边栏选择渲染对应功能 ===
if mode == "简历优化":
    render_resume_optimize()
elif mode == "生成面试题":
    render_generate_questions()
elif mode == "模拟面试":
    render_mock_interview()
elif mode == "简历评分":
    render_resume_score()
