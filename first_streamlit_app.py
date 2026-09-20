import streamlit as st

st.set_page_config(
    page_title="我的第一个 Streamlit 应用",
    page_icon="🌟",
    layout="centered"
)

st.title("🌟 我的第一个 Streamlit 应用")
st.write("这是用纯 Python 写的网页，不需要 HTML/CSS/JS！")

st.divider()

st.subheader("🎯 基本组件演示")

name = st.text_input("你的名字")
age = st.number_input("你的年龄", min_value=0, max_value=150)
hobby = st.text_area("你的爱好（可以写多行）", height=100)

if st.button("提交", type="primary"):
    if not name:
        st.warning("请输入你的名字！")
    else:
        with st.spinner("正在生成结果..."):
            import time
            time.sleep(1)
            st.success("提交成功！")
            st.write(f"👋 你好，{name}！")
            st.write(f"🎂 你今年 {age} 岁")
            st.write(f"🎨 你的爱好：{hobby}")

st.divider()

st.subheader("🎲 选择器演示")

option = st.selectbox(
    "你最喜欢的颜色",
    ["红色", "蓝色", "绿色", "黄色", "紫色"]
)

mode = st.radio(
    "选择模式",
    ["快速模式", "详细模式", "极简模式"]
)

if st.button("查看选择"):
    st.info(f"你选择了：{option}，模式：{mode}")

st.divider()

st.subheader("📊 数据展示")

data = {
    "功能": ["简历优化", "面试题生成", "模拟面试", "简历评分"],
    "状态": ["✅ 已完成", "✅ 已完成", "✅ 已完成", "✅ 已完成"]
}

st.dataframe(data)

st.divider()

st.subheader("💡 提示消息")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success("成功")
with col2:
    st.warning("警告")
with col3:
    st.error("错误")
with col4:
    st.info("信息")
