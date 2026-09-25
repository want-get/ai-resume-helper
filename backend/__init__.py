"""backend 包：单机版 FastAPI 后端。

分层：
    app_v3.py             FastAPI 应用装配（同时托管前端单页）
    api_v3.py             全部 HTTP 接口
    profile.py            用户档案 + 必填门禁
    salary.py             薪资文本解析 + 行情统计
    personal_kb.py        用户专属知识库
    crawlers/             岗位抓取（公开 JSON 接口 + 通用渲染）
    services.py           业务编排（RAG 问答 / 简历优化 / 模拟面试）
    memory.py             长上下文（滑动窗口 + 滚动摘要）
    anti_hallucination.py 防幻觉：上下文编排、引用校验
    llm_config.py         大模型配置（存 data/settings.json，前端热更新）
    db/                   异步 ORM（SQLite）

v2 时代的 ``api.py`` / ``schemas.py`` / ``runtime_settings.py``（Streamlit + MySQL
那一套）已归档到 ``legacy/v2_backend/``，不参与 v3 构建。
"""

__all__ = ["create_app"]


def create_app(*args, **kwargs):  # pragma: no cover - 延迟导入，避免循环依赖
    from .app_v3 import create_app as _create_app

    return _create_app(*args, **kwargs)
