"""HTTP 接口测试（进程内 TestClient + 离线 Mock 模式）。

覆盖：健康检查、RAG 检索与问答、防幻觉拒答、Function Calling、
多轮对话与长上下文、简历相关功能、并发闸门。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api import app

RESUME = (
    "张三，Python 后端开发工程师，5 年经验。负责订单中台，使用 FastAPI + MySQL + Redis，"
    "把订单创建接口 P95 从 480ms 降到 120ms，QPS 从 300 提升到 1500。"
    "熟悉 asyncio 并发控制、缓存一致性方案、MySQL 索引优化。"
) * 2


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"]["backend"] in ("mysql", "sqlite")
    assert all(item["chunks"] > 0 for item in body["knowledge_base"])


def test_metrics(client: TestClient) -> None:
    body = client.get("/metrics").json()
    assert "app" in body and "llm" in body
    assert body["app"]["total_requests"] >= 0


def test_knowledge_search_returns_hybrid_hits(client: TestClient) -> None:
    response = client.post(
        "/api/v1/knowledge/search", json={"query": "缓存穿透怎么解决", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"], "应当检索到内容"
    assert all(chunk["source"] in ("vector", "bm25", "hybrid") for chunk in body["chunks"])
    assert all(chunk["label"] for chunk in body["chunks"])


def test_rag_ask_returns_citations(client: TestClient) -> None:
    response = client.post(
        "/api/v1/rag/ask", json={"question": "缓存穿透和缓存雪崩的区别？", "role_type": "tech"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content"]
    assert body["sources"], "必须返回引用来源"
    assert body["citation_report"] is not None
    assert body["sources"][0]["index"] == 1


def test_rag_ask_refuses_without_context(client: TestClient) -> None:
    """防幻觉：知识库没有资料时必须拒答，而不是自由发挥。"""
    body = client.post("/api/v1/rag/ask", json={"question": "今天天气怎么样？"}).json()
    assert body["refused"] is True
    assert body["confidence"] == 0.0
    assert "知识库中未收录" in body["content"]


def test_salary_tool_direct(client: TestClient) -> None:
    body = client.post(
        "/api/v1/tools/salary", json={"role": "Python后端开发", "city": "杭州"}
    ).json()
    assert body["found"] is True
    assert set(body["percentiles"]) == {"p25", "p50", "p75", "p90"}
    assert body["sample_size"] > 0
    assert body["source"]


def test_job_search_tool_direct(client: TestClient) -> None:
    """按关键字搜岗位（只匹配职位名称）。

    这里固定只用 ``local`` 数据源，保证测试不依赖外网；
    真实公开数据源的联调由 test_job_search_public_sources 覆盖。
    """
    body = client.post(
        "/api/v1/tools/jobs",
        json={"keywords": "Python 后端", "sources": "local", "limit": 5},
    ).json()
    assert body["found"] is True
    assert 0 < len(body["jobs"]) <= 5
    titles = " ".join(job["title"] for job in body["jobs"])
    assert "Python" in titles or "后端" in titles
    # 每条结果都要说明自己是哪个数据源来的
    assert all(job.get("source") for job in body["jobs"])
    assert all(job.get("matched_keywords") for job in body["jobs"])


def test_job_search_requires_keywords(client: TestClient) -> None:
    assert client.post("/api/v1/tools/jobs", json={"keywords": ""}).status_code == 422


def test_job_search_matches_title_only(client: TestClient) -> None:
    """关键字必须匹配「职位名称」，不能因为描述里出现就命中。"""
    body = client.post(
        "/api/v1/tools/jobs",
        json={"keywords": "星野科技", "sources": "local", "limit": 10},
    ).json()
    # 「星野科技」是公司名，不是职位名 → 不应命中
    assert body["found"] is False
    assert body["jobs"] == []


def test_job_sources_endpoint(client: TestClient) -> None:
    body = client.get("/api/v1/tools/job-sources").json()
    keys = {item["key"] for item in body["available"]}
    assert {"jobicy", "remotive", "remoteok", "arbeitnow", "local"} <= keys
    assert body["note"]


def test_job_search_public_sources(client: TestClient) -> None:
    """真实公开招聘接口联调（网络不可用时跳过，不算失败）。"""
    body = client.post(
        "/api/v1/tools/jobs",
        json={"keywords": "python backend", "sources": "jobicy,remotive", "limit": 5},
    ).json()
    public = [item for item in body["sources"] if item["source"] != "local"]
    if not any(item["ok"] for item in public):
        pytest.skip(f"公开数据源当前不可用：{public}")
    if not body["jobs"]:
        pytest.skip("公开数据源可用，但本次没有职位名命中")
    titles = " ".join(job["title"].lower() for job in body["jobs"])
    assert "python" in titles or "backend" in titles


def test_llm_settings_status_and_runtime_update(client: TestClient) -> None:
    """在线修改 API Key：立即生效、不回显明文、可清除。"""
    import asyncio

    from backend.ai_client import reset_ai_client
    from config import get_settings

    settings = get_settings()
    original_key = settings.deepseek_api_key
    original_mock = settings.llm_mock_mode

    try:
        status = client.get("/api/v1/settings/llm").json()
        assert status["runtime_editable"] is True
        assert "key_masked" in status
        assert "deepseek_api_key" not in status

        fake_key = "sk-test-not-a-real-key-0123456789"
        response = client.put(
            "/api/v1/settings/llm",
            json={"api_key": fake_key, "persist": False, "verify": False},
        )
        assert response.status_code == 200
        body = response.json()
        assert "api_key" in body["changed"]
        assert body["status"]["configured"] is True
        assert body["status"]["mock_mode"] is False
        # 任何地方都不能回显完整 Key
        assert fake_key not in json.dumps(body, ensure_ascii=False)
        assert body["persisted_env_keys"] == []  # 明确没写 .env

        cleared = client.delete("/api/v1/settings/llm/key?persist=false").json()
        assert cleared["status"]["configured"] is False
        assert cleared["status"]["mock_mode"] is True
    finally:
        settings.deepseek_api_key = original_key
        settings.llm_mock_mode = original_mock
        asyncio.run(reset_ai_client())


def test_target_job_flows_into_interview(client: TestClient) -> None:
    """选定目标岗位后，面试会话要真的把它记下来并作为出题依据。"""
    resume_id = client.post(
        "/api/v1/resumes", json={"filename": "r.pdf", "content": RESUME}
    ).json()["resume_id"]

    job_context = "Python 后端开发工程师 @ 星野科技\n技能要求：FastAPI、asyncio、MySQL、Redis"
    session = client.post(
        "/api/v1/sessions",
        json={
            "mode": "mock_interview",
            "role_type": "tech",
            "resume_id": resume_id,
            "job_title": "Python 后端开发工程师",
            "job_company": "星野科技",
            "job_context": job_context,
        },
    ).json()
    assert session["job_title"] == "Python 后端开发工程师"
    assert session["has_job_context"] is True

    detail = client.get(f"/api/v1/sessions/{session['session_id']}").json()
    assert detail["job_title"] == "Python 后端开发工程师"
    assert detail["job_company"] == "星野科技"
    assert detail["has_job_context"] is True

    opened = client.post(f"/api/v1/sessions/{session['session_id']}/start")
    assert opened.status_code == 200
    # 目标岗位应当以引用来源的形式进入出题上下文
    labels = [s["label"] for s in opened.json()["reply"]["sources"]]
    assert any("目标岗位" in label for label in labels), labels

    context = client.get(f"/api/v1/sessions/{session['session_id']}/context").json()
    assert context["stats"]["total_messages"] >= 1


def test_resume_feature_accepts_job_context(client: TestClient) -> None:
    """简历功能接受目标岗位 JD，并把它作为 [1] 号引用来源。"""
    resume_id = client.post(
        "/api/v1/resumes", json={"filename": "r.pdf", "content": RESUME}
    ).json()["resume_id"]

    body = client.post(
        "/api/v1/resume/score",
        json={
            "resume_id": resume_id,
            "role_type": "tech",
            "job_context": "大模型应用开发工程师\n要求：RAG、Function Calling、FastAPI",
        },
    ).json()
    assert body["sources"], "应当返回引用来源"
    assert "目标岗位" in body["sources"][0]["label"]
    assert body["retrieval"]["target_job"] == "大模型应用开发工程师"


def test_tools_call_endpoint(client: TestClient) -> None:
    body = client.post(
        "/api/v1/tools/call",
        json={"name": "query_salary", "arguments": {"role": "算法工程师", "city": "北京"}},
    ).json()
    assert body["result"]["ok"] is True


def test_tools_ask_triggers_autonomous_tool_call(client: TestClient) -> None:
    body = client.post(
        "/api/v1/tools/ask",
        json={"question": "杭州 Python 后端 3 年经验薪资多少？", "use_tools": True},
    ).json()
    assert body["tool_calls"], "模型应当自主发起工具调用"
    assert all(call["ok"] for call in body["tool_calls"])


def test_tool_list(client: TestClient) -> None:
    tools = client.get("/api/v1/tools").json()["tools"]
    names = {tool["name"] for tool in tools}
    assert {"query_salary", "search_jobs", "search_knowledge_base"} <= names


def test_resume_crud_and_features(client: TestClient) -> None:
    created = client.post(
        "/api/v1/resumes", json={"filename": "resume.pdf", "content": RESUME}
    )
    assert created.status_code == 200
    resume_id = created.json()["resume_id"]

    fetched = client.get(f"/api/v1/resumes/{resume_id}").json()
    assert fetched["chars"] == len(RESUME)

    for path in ("/api/v1/resume/questions", "/api/v1/resume/optimize", "/api/v1/resume/score"):
        body = client.post(path, json={"resume_id": resume_id, "role_type": "tech"}).json()
        assert body["content"], f"{path} 应当有输出"
        assert body["sources"], f"{path} 应当带知识库引用"


def test_resume_requires_text_or_id(client: TestClient) -> None:
    assert client.post("/api/v1/resume/score", json={}).status_code == 422


def test_multi_turn_interview_and_long_context(client: TestClient) -> None:
    session_id = client.post(
        "/api/v1/sessions",
        json={"mode": "mock_interview", "role_type": "tech", "resume_text": RESUME},
    ).json()["session_id"]

    assert client.post(f"/api/v1/sessions/{session_id}/start").status_code == 200

    for i in range(10):
        response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": f"第{i + 1}轮回答：{RESUME}"},
        )
        assert response.status_code == 200

    session = client.get(f"/api/v1/sessions/{session_id}").json()
    assert session["message_count"] >= 20

    stats = client.get(f"/api/v1/sessions/{session_id}/context").json()["stats"]
    assert stats["window_messages"] < stats["total_messages"], "滑动窗口应当生效"
    assert stats["summary_chars"] > 0, "早期对话应当被压缩为摘要"
    assert stats["summary_upto_seq"] > 0
    assert stats["compressed_ratio"] > 0

    report = client.post(f"/api/v1/sessions/{session_id}/report").json()
    assert report["content"]


def test_rag_chat_session_returns_sources(client: TestClient) -> None:
    session_id = client.post(
        "/api/v1/sessions", json={"mode": "rag_chat", "role_type": "tech"}
    ).json()["session_id"]
    body = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"content": "面试官常问的缓存问题有哪些？", "use_tools": True},
    ).json()
    assert body["result"]["sources"]
    assert body["memory"] is not None


def test_session_not_found(client: TestClient) -> None:
    assert client.get("/api/v1/sessions/does-not-exist").status_code == 404
    assert client.post("/api/v1/sessions/does-not-exist/messages", json={"content": "hi"}).status_code == 404


def test_knowledge_rebuild_and_stats(client: TestClient) -> None:
    stats = client.get("/api/v1/knowledge/stats").json()["collections"]
    names = {item["collection"] for item in stats}
    # v3 起多了用户专属知识库 personal_kb，所以至少要有这三个集合
    assert {"interview_questions", "job_descriptions", "personal_kb"} <= names
    # 公共集合必须有内容；个人库在没建过档案时可以是空的
    assert all(
        item["chunks"] > 0 for item in stats if item["collection"] != "personal_kb"
    )


def test_system_stats(client: TestClient) -> None:
    body = client.get("/api/v1/stats").json()
    assert body["job_postings"] > 0
    assert body["salary_records"] > 0
    assert "metrics" in body
