"""v3 单机版接口测试（TestClient 进程内运行，不联网、不调用真实大模型）。

覆盖：
* 必填门禁（档案不全 → 后续接口全部 428）
* 档案读写与关键字推导
* 简历上传（文本）
* 岗位来源管理
* 薪资行情接口的降级行为
* 前端静态页与接口可达
* 大模型配置的在线读写（不落盘、不回显明文）

真实抓取与真实模型由 ``scripts/check_app.py`` 覆盖（需要联网）。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.app_v3 import app

RESUME = (
    "李四，3 年 AI 应用开发经验。主导企业知识库 RAG 系统，"
    "用 FastAPI + Chroma 实现检索增强生成，回答准确率从 60% 提升到 88%。"
) * 3


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        # 每个模块从干净状态开始（测试用的是 .pytest_data/，不影响用户数据）
        test_client.delete("/api/v1/profile")
        yield test_client


# ====================================================================== #
# 前端与系统
# ====================================================================== #
def test_frontend_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>" in response.text
    assert "/static/app.js" in response.text


@pytest.mark.parametrize("asset", ["/static/app.js", "/static/style.css", "/static/favicon.svg"])
def test_static_assets(client: TestClient, asset: str) -> None:
    response = client.get(asset)
    assert response.status_code == 200
    assert len(response.content) > 100


def test_system_info_single_machine(client: TestClient) -> None:
    info = client.get("/api/v1/system").json()
    assert info["version"].startswith("3.")
    assert info["single_user_mode"] is True
    # 单机版默认 SQLite，且路径落在数据目录里
    assert "app.db" in info["paths"]["database"]


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"


# ====================================================================== #
# 门禁
# ====================================================================== #
def test_empty_profile_blocks_everything(client: TestClient) -> None:
    client.delete("/api/v1/profile")
    state = client.get("/api/v1/profile").json()
    assert state["profile_ready"] is False
    assert "求职方向" in state["missing_fields"]
    assert "简历" in state["missing_fields"]

    assert client.post("/api/v1/market/crawl", json={}).status_code == 428
    assert client.post("/api/v1/interview/start", json={"rounds": 3}).status_code == 428
    assert client.post("/api/v1/resume/optimize", json={}).status_code == 428


def test_profile_save_and_keywords(client: TestClient) -> None:
    state = client.put(
        "/api/v1/profile",
        json={
            "target_role": "AI Agent 开发工程师",
            "expect_city": "北京",
            "expect_salary_min": 25,
            "expect_salary_max": 40,
        },
    ).json()
    profile = state["profile"]
    assert profile["target_role"] == "AI Agent 开发工程师"
    assert profile["salary_text"] == "25-40K"
    # 关键字里必须包含「agent」，否则抓岗匹配不到岗位
    keywords = profile["target_keywords"]
    assert any("agent" in word for word in keywords), keywords
    assert state["missing_fields"] == ["简历"]


def test_resume_upload_completes_profile(client: TestClient) -> None:
    response = client.post(
        "/api/v1/resume/upload", json={"content": RESUME, "filename": "resume.txt"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chars"] == len(RESUME)
    assert body["state"]["profile_ready"] is True


def test_resume_upload_rejects_too_short(client: TestClient) -> None:
    response = client.post("/api/v1/resume/upload", json={"content": "太短"})
    assert response.status_code == 422


def test_knowledge_base_needs_target_job(client: TestClient) -> None:
    """档案齐全但没选目标岗位时，建库要被挡住并提示下一步。"""
    response = client.post("/api/v1/knowledge/personal/build", json={})
    assert response.status_code == 428
    assert response.json().get("next_step") == "market"


# ====================================================================== #
# 岗位来源
# ====================================================================== #
def test_sources_crud(client: TestClient) -> None:
    sources = client.get("/api/v1/sources").json()["sources"]
    assert isinstance(sources, list) and sources
    keys = {item["key"] for item in sources}
    assert "nowcoder_campus" in keys  # 唯一带结构化薪资的中文来源

    sources[0]["enabled"] = False
    updated = client.put("/api/v1/sources", json={"sources": sources}).json()["sources"]
    assert updated[0]["enabled"] is False
    # 恢复，避免影响同模块后续用例
    sources[0]["enabled"] = True
    client.put("/api/v1/sources", json={"sources": sources})


def test_sources_rejects_bad_payload(client: TestClient) -> None:
    assert client.put("/api/v1/sources", json={"sources": "oops"}).status_code == 422


def test_market_endpoints_without_data(client: TestClient) -> None:
    """没抓过岗时，岗位列表为空但接口不应报错。"""
    body = client.get("/api/v1/market/jobs").json()
    assert "jobs" in body and isinstance(body["jobs"], list)

    personal = client.get("/api/v1/knowledge/personal").json()
    assert "ready" in personal


def test_choose_target_job_requires_existing_key(client: TestClient) -> None:
    assert client.post("/api/v1/market/target", json={}).status_code == 422
    assert client.post("/api/v1/market/target", json={"job_key": "nope"}).status_code == 404


# ====================================================================== #
# 大模型配置
# ====================================================================== #
def test_llm_settings_never_leaks_key(client: TestClient) -> None:
    """在线改配置：立即生效、任何响应都不回显明文 Key。"""
    import asyncio

    from backend.llm_config import get_llm_config

    original = get_llm_config().api_key
    fake = "sk-unit-test-not-a-real-key-000"
    try:
        response = client.put(
            "/api/v1/settings/llm",
            json={"provider": "deepseek", "api_key": fake, "verify": False},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["config"]["configured"] is True
        assert fake not in json.dumps(body, ensure_ascii=False)
        assert body["config"]["key_masked"].endswith("000")

        listed = client.get("/api/v1/settings/llm").json()
        assert fake not in json.dumps(listed, ensure_ascii=False)
        providers = {item["key"] for item in listed["providers"]}
        assert {"deepseek", "kimi", "dashscope", "openai_compatible"} <= providers
    finally:
        asyncio.run(_restore_llm_key(original))


async def _restore_llm_key(original: str) -> None:
    from backend.ai_client import reset_ai_client
    from backend.llm_config import save_llm_config

    save_llm_config(api_key=original or "")
    await reset_ai_client()


def test_llm_invalid_temperature_rejected(client: TestClient) -> None:
    response = client.put("/api/v1/settings/llm", json={"temperature": 9.9, "verify": False})
    assert response.status_code == 422


# ====================================================================== #
# 重置
# ====================================================================== #
def test_profile_reset(client: TestClient) -> None:
    body = client.delete("/api/v1/profile").json()
    assert body["ok"] is True
    assert body["state"]["profile_ready"] is False
    assert client.get("/api/v1/profile").json()["profile_ready"] is False
