"""Streamlit 前端的渲染回归测试。

用 ``streamlit.testing.v1.AppTest`` 真正把 ``app.py`` 跑一遍（不需要启动后端，
用假客户端替换 ``client.BackendClient``）。这类测试的价值在于：
**同一次渲染中重复注册组件 key** 这类错误只有真正渲染才会暴露，
之前正因为缺少它，四个页签各调用一次上传组件导致
``StreamlitDuplicateElementKey`` 直接白屏。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

import client as client_module  # noqa: E402

APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


# ---------------------------------------------------------------------- #
# 假的 BackendClient：只覆盖默认渲染路径会调用的方法
# ---------------------------------------------------------------------- #
class FakeBackendClient:
    """返回固定数据的假客户端，让前端可以完全离线渲染。"""

    #: 记录最后一次带 job_context 的调用，供断言检查
    last_call: dict[str, Any] = {}

    def __init__(self, base_url: str | None = None, api_key: str | None = None, **_: Any) -> None:
        self.base_url = base_url or "http://fake-backend"
        self.api_key = api_key

    # --- 基础 ---
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "2.0.0-test",
            "database": {"backend": "mysql", "fallback": False, "url": "mysql://fake"},
            "knowledge_base": [
                {"collection": "interview_questions", "chunks": 36},
                {"collection": "job_descriptions", "chunks": 16},
            ],
            "llm": {"mock_mode": True, "model": "deepseek-chat", "configured": False},
            "concurrency": {"limit": 256, "max_concurrent_llm": 64},
        }

    def llm_settings(self) -> dict[str, Any]:
        return {
            "configured": False,
            "mock_mode": True,
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "key_masked": "",
            "runtime_editable": True,
        }

    def update_llm_settings(self, **_: Any) -> dict[str, Any]:
        return {
            "changed": ["api_key"],
            "persisted_env_keys": ["DEEPSEEK_API_KEY"],
            "status": {**self.llm_settings(), "configured": True, "mock_mode": False},
            "verify": {"ok": True, "reply_preview": "可用", "latency_ms": 1.0},
        }

    def verify_llm(self) -> dict[str, Any]:
        return {"ok": True, "reply_preview": "可用"}

    def clear_llm_key(self, persist: bool = True) -> dict[str, Any]:
        return {"changed": ["api_key=cleared"], "status": self.llm_settings()}

    def job_sources(self) -> dict[str, Any]:
        return {
            "available": [
                {"key": "jobicy", "label": "Jobicy（公开 API）"},
                {"key": "local", "label": "本地示例库"},
            ],
            "cache_ttl_seconds": 300,
            "custom_api_configured": False,
            "note": "内置数据源为免 Key 的公开招聘接口。",
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "app": {
                "total_requests": 10,
                "in_flight": 1,
                "peak_in_flight": 8,
                "error_rate": 0.0,
                "latency_ms": {"p95": 12.3},
            },
            "requests_gate": {"limit": 256, "waiting": 0},
            "llm": {"in_flight": 0, "peak_in_flight": 3, "concurrency_limit": 64},
        }

    def kb_stats(self) -> dict[str, Any]:
        return {
            "collections": [
                {
                    "collection": "interview_questions",
                    "chunks": 36,
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "embed_dim": 4096,
                    "avg_chunk_chars": 272.0,
                    "updated_at": "2026-01-01 00:00:00",
                    "bm25": {"n_docs": 36},
                },
                {
                    "collection": "job_descriptions",
                    "chunks": 16,
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "embed_dim": 4096,
                    "avg_chunk_chars": 342.0,
                    "updated_at": "2026-01-01 00:00:00",
                    "bm25": {"n_docs": 16},
                },
            ]
        }

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return []

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "query_salary", "description": "查询薪资", "tags": ["salary"]},
            {"name": "search_jobs", "description": "搜索岗位", "tags": ["job"]},
        ]

    def create_resume(self, filename: str, content: str) -> dict[str, Any]:
        return {"resume_id": "fake-resume-id", "chars": len(content)}

    def get_resume(self, resume_id: str) -> dict[str, Any]:
        return {
            "resume_id": resume_id,
            "filename": "fake.pdf",
            "chars": 100,
            "content": "（假）简历解析出来的文本内容",
        }

    def optimize_resume(self, **kwargs: Any) -> dict[str, Any]:
        FakeBackendClient.last_call = dict(kwargs)
        return _fake_result("（假）优化后的简历内容 [1]")

    def interview_questions(self, **kwargs: Any) -> dict[str, Any]:
        FakeBackendClient.last_call = dict(kwargs)
        return _fake_result("（假）面试题内容 [1]")

    def score_resume(self, **kwargs: Any) -> dict[str, Any]:
        FakeBackendClient.last_call = dict(kwargs)
        return _fake_result("（假）评分结果 [1]")

    def rag_ask(self, *_: Any, **__: Any) -> dict[str, Any]:
        return _fake_result("（假）知识库回答 [1]")

    def tools_ask(self, *_: Any, **__: Any) -> dict[str, Any]:
        result = _fake_result("（假）工具回答")
        result["tool_calls"] = [
            {"name": "query_salary", "ok": True, "arguments": {"role": "x"}, "result": {}, "elapsed_ms": 1.0}
        ]
        return result

    def create_session(self, **_: Any) -> dict[str, Any]:
        return {"session_id": "fake-session", "resume_id": "fake-resume-id", "mode": "mock_interview"}

    def start_session(self, session_id: str) -> dict[str, Any]:
        return {"session_id": session_id, "reply": {"role": "assistant", "content": "（假）第一个问题"}}

    def send_message(self, session_id: str, content: str, use_tools: bool = False) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "reply": {"role": "assistant", "content": "（假）下一个问题"},
            "memory": {
                "total_messages": 4,
                "window_messages": 2,
                "summarized_messages": 2,
                "summary_chars": 50,
                "window_chars": 80,
                "history_chars": 200,
                "compressed_ratio": 0.35,
                "summary_updated": False,
                "summary_upto_seq": 2,
            },
            "result": {},
        }

    def session_context(self, session_id: str) -> dict[str, Any]:
        return {
            "stats": {
                "total_messages": 4,
                "window_messages": 2,
                "summarized_messages": 2,
                "summary_chars": 50,
                "window_chars": 80,
                "history_chars": 200,
                "compressed_ratio": 0.35,
                "summary_updated": False,
                "summary_upto_seq": 2,
            },
            "summary": "（假）滚动摘要",
            "messages": [],
        }

    def session_report(self, session_id: str) -> dict[str, Any]:
        return {"content": "（假）面试报告", "confidence": 0.5}

    def delete_session(self, session_id: str) -> dict[str, Any]:
        return {"ok": True}

    def jobs(self, *args: Any, **__: Any) -> dict[str, Any]:
        return {"found": False, "jobs": [], "sources": [], "notes": [], "error": "无结果"}

    def salary(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {"found": True}

    def kb_rebuild(self) -> dict[str, Any]:
        return {"results": []}


def _fake_result(content: str) -> dict[str, Any]:
    return {
        "content": content,
        "sources": [
            {
                "index": 1,
                "label": "面试题库 · Python后端开发",
                "text": "（假）来源原文",
                "similarity": 0.42,
                "retrieval": "hybrid",
            }
        ],
        "citation_report": {"cited": [1], "invalid": [], "available": 1, "has_rate": 1.0, "citation_rate": 1.0},
        "confidence": 0.7,
        "warnings": [],
        "refused": False,
        "latency_ms": 12.0,
        "retrieval": {"hits": 1, "elapsed_ms": 3.0},
        "mock": True,
    }


# ---------------------------------------------------------------------- #
# 测试
# ---------------------------------------------------------------------- #
@pytest.fixture()
def app_test(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """把 app.py 里的 BackendClient 换成假实现，然后渲染整个页面。"""
    monkeypatch.setattr(client_module, "BackendClient", FakeBackendClient)
    return AppTest.from_file(str(APP_FILE), default_timeout=60)


def _button(at: AppTest, label: str):
    """按标签找按钮：表单提交按钮不支持自定义 key，只能按文字定位。"""
    for button in at.button:
        if button.label == label:
            return button
    raise AssertionError(f"没有找到按钮：{label}（现有：{[b.label for b in at.button]}）")


def test_app_renders_without_exception(app_test: AppTest) -> None:
    """整页渲染不能有任何异常。

    这里就是 ``StreamlitDuplicateElementKey`` 的回归点：
    简历相关的四个页签曾各自调用一次上传组件，导致同一次渲染里重复注册 key。
    """
    app_test.run()
    assert not app_test.exception, [str(e) for e in app_test.exception]


def test_no_duplicate_widget_keys(app_test: AppTest) -> None:
    """显式检查页面上的交互组件 key 没有重复。"""
    app_test.run()
    keys: list[str] = []
    for collection in (
        app_test.button,
        app_test.text_input,
        app_test.text_area,
        app_test.selectbox,
        app_test.slider,
        app_test.checkbox,
        app_test.radio,
        app_test.file_uploader,
        app_test.number_input,
        app_test.multiselect,
    ):
        for widget in collection:
            keys.append(widget.key)
    duplicates = {key for key in keys if key and keys.count(key) > 1}
    assert not duplicates, f"存在重复的组件 key：{sorted(duplicates)}"


def test_resume_uploader_registered_exactly_once(app_test: AppTest) -> None:
    """共用简历区只应存在一个上传组件（这正是之前的 bug）。"""
    app_test.run()
    assert len(app_test.file_uploader) == 1


def test_target_job_panel_present(app_test: AppTest) -> None:
    """顶部要有「目标岗位」区，让用户先设定求职目标。"""
    app_test.run()
    assert not app_test.exception
    assert app_test.text_input(key="job_keywords_input") is not None


def test_sidebar_and_tabs_present(app_test: AppTest) -> None:
    app_test.run()
    assert not app_test.exception
    assert len(app_test.tabs) >= 7


def test_resume_feature_button_disabled_without_resume(app_test: AppTest) -> None:
    """没有简历时，简历相关按钮应当是禁用的（而不是点了报错）。"""
    app_test.run()
    assert not app_test.exception
    assert app_test.button(key="optimize_btn").disabled is True
    assert app_test.button(key="gen_q").disabled is True
    assert app_test.button(key="score_btn").disabled is True


def test_optimize_flow_sends_target_job(app_test: AppTest) -> None:
    """有简历 + 有目标岗位时，优化请求要把目标岗位带过去。"""
    app_test.session_state["resume_id"] = "fake-resume-id"
    app_test.session_state["resume_name"] = "fake.pdf"
    app_test.session_state["target_job"] = {
        "id": "job-1",
        "title": "Python 后端开发工程师",
        "company": "星野科技",
        "location": "杭州",
        "salary": "25-40K",
        "source": "Fake",
    }
    app_test.run()
    assert not app_test.exception

    app_test.button(key="optimize_btn").click().run()
    assert not app_test.exception, [str(e) for e in app_test.exception]
    assert any("优化后的简历内容" in block.value for block in app_test.markdown)
    # 假客户端把收到的 job_context 记在 last_call 里
    assert "Python 后端开发工程师" in (FakeBackendClient.last_call.get("job_context") or "")


def test_rag_question_flow_and_form_clears(app_test: AppTest) -> None:
    """知识库问答：提问后要出结果，且输入框不能残留上一次的问题。"""
    app_test.run()
    app_test.text_area(key="rag_question_0").set_value("缓存穿透怎么解决").run()
    _button(app_test, "提问").click().run()
    assert not app_test.exception, [str(e) for e in app_test.exception]
    assert any("知识库回答" in block.value for block in app_test.markdown)

    # 关键断言：页面上任何输入框都不应再残留刚才的问题
    assert "缓存穿透怎么解决" not in [area.value for area in app_test.text_area]
    # 轮次已推进 → 下一次渲染用的是全新的空输入框
    assert app_test.session_state["rag_round"] == 1
    assert app_test.text_area(key="rag_question_1").value in ("", None)


def test_interview_answer_box_clears_after_submit(app_test: AppTest) -> None:
    """模拟面试：提交回答后输入框不能残留上一轮内容（用户反馈的问题）。"""
    app_test.session_state["resume_id"] = "fake-resume-id"
    app_test.session_state["resume_name"] = "fake.pdf"
    app_test.run()

    _button(app_test, "开始面试").click().run()
    assert not app_test.exception, [str(e) for e in app_test.exception]
    assert app_test.session_state["interview_started"] is True

    app_test.text_area(key="interview_answer_0").set_value("我的第一轮回答").run()
    _button(app_test, "提交回答").click().run()
    assert not app_test.exception, [str(e) for e in app_test.exception]

    # 关键的三个断言
    assert "我的第一轮回答" not in [area.value for area in app_test.text_area], "输入框残留了上一轮答案"
    assert app_test.session_state["answer_round"] == 1
    assert app_test.text_area(key="interview_answer_1").value in ("", None)
    # 对话记录里应当同时有用户回答和面试官回复
    roles = [m["role"] for m in app_test.session_state["interview_messages"]]
    assert roles.count("user") == 1
    assert roles.count("assistant") == 2


def test_interview_shows_question_number(app_test: AppTest) -> None:
    """面试对话要显示题号，让用户知道进行到第几题。"""
    app_test.session_state["resume_id"] = "fake-resume-id"
    app_test.session_state["interview_started"] = True
    app_test.session_state["interview_session"] = "fake-session"
    app_test.session_state["interview_messages"] = [
        {"role": "assistant", "content": "第一题"},
    ]
    app_test.run()
    assert not app_test.exception
    assert any("第 1 题" in block.value for block in app_test.info)
