"""大模型配置的来源一致性测试。

**这个文件是为了防止一个真实发生过的严重 bug 复现**：

    界面上保存的 Key 落在 ``settings.json``（由 ``llm_config`` 管理），
    而 ``AsyncAIClient`` 原先读的是 ``.env``/环境变量（``Settings``）。
    两处各自为政，后果是：

    * 源码运行（有 .env）→ 一切正常
    * **打包成 exe 后**（没有 .env）→ 用户在界面填了 Key，
      「测试连接」成功，但出题/简历优化仍然走 **Mock 桩**，
      返回「未配置 DEEPSEEK_API_KEY」——用户拿到的是假结果。

所以这里断言：**界面保存的配置必须立刻对 AI 客户端生效**。
"""

from __future__ import annotations

import pytest

from backend.ai_client import AsyncAIClient, get_ai_client, reset_ai_client
from backend.llm_config import get_llm_config, reset_for_tests, save_llm_config


@pytest.fixture(autouse=True)
def _clean_llm_config():
    """每个用例前后都把配置恢复到干净状态，避免相互污染。"""
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture()
def online_settings(monkeypatch):
    """把强制 Mock 关掉，用于验证「真的会去调模型」的那条分支。

    conftest 为了让单元测试不消耗额度、结果稳定，全局设了 ``LLM_MOCK_MODE=true``。
    要验证「配置生效后不再 Mock」就必须显式关掉它。
    """
    from config import get_settings
    import backend.ai_client as ai_client_module

    real = get_settings().model_copy(update={"llm_mock_mode": False})
    monkeypatch.setattr(ai_client_module, "get_settings", lambda: real)
    return real


def test_saved_key_makes_client_configured_and_disables_mock(online_settings):
    """核心回归：界面保存 Key 后，客户端必须**立刻**变成已配置且不再 Mock。"""
    save_llm_config(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key="sk-test-1234567890abcdef",
        persist=False,
    )

    client = AsyncAIClient()
    assert client.configured, "保存了 Key，客户端就应该认为已配置"
    assert not client.mock_mode, "已配置后不应再走 Mock 桩"


def test_client_without_key_uses_mock_mode(online_settings):
    save_llm_config(api_key="", persist=False)
    client = AsyncAIClient()
    assert not client.configured
    assert client.mock_mode, "没配 Key 时应明确进入 Mock 模式"


def test_client_picks_up_runtime_config_change(online_settings):
    """先没配 Key，再在运行期配上——模拟「启动后才在界面填 Key」。"""
    save_llm_config(api_key="", persist=False)
    client = AsyncAIClient()
    assert client.mock_mode

    save_llm_config(
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key="sk-runtime-key-abcdef123456",
        persist=False,
    )
    # 同一个客户端实例必须能读到新配置（不能把配置在构造时固化）
    assert client.configured
    assert not client.mock_mode


def test_client_uses_saved_base_url_and_model():
    save_llm_config(
        provider="kimi",
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k2-0905-preview",
        api_key="sk-kimi-abcdef123456",
        persist=False,
    )
    client = AsyncAIClient()
    assert client.llm_config.resolved_base_url() == "https://api.moonshot.cn/v1"
    assert client.llm_config.resolved_model() == "kimi-k2-0905-preview"


def test_client_snapshot_reports_saved_model(online_settings):
    save_llm_config(
        base_url="https://api.deepseek.com",
        model="deepseek-reasoner",
        api_key="sk-snapshot-abcdef123456",
        persist=False,
    )
    snapshot = AsyncAIClient().snapshot()
    assert snapshot["configured"] is True
    assert snapshot["mock_mode"] is False
    assert snapshot["model"] == "deepseek-reasoner"


@pytest.mark.asyncio
async def test_mock_mode_still_works_when_unconfigured():
    """Mock 模式本身要保留：没配 Key 时能跑通全链路，但不冒充真实模型。"""
    save_llm_config(api_key="", persist=False)
    client = AsyncAIClient()
    result = await client.chat([{"role": "user", "content": "你好"}])
    assert result.ok
    assert result.mock is True
    assert result.model.startswith("mock-")


@pytest.mark.asyncio
async def test_get_ai_client_reset_reflects_new_config(online_settings):
    """``reset_ai_client()`` 之后新建的客户端必须用最新配置。"""
    save_llm_config(api_key="", persist=False)
    await reset_ai_client()
    assert get_ai_client().mock_mode

    save_llm_config(
        base_url="https://api.deepseek.com",
        api_key="sk-after-reset-abcdef123456",
        persist=False,
    )
    await reset_ai_client()
    assert not get_ai_client().mock_mode


def test_config_file_roundtrip_keeps_key(tmp_path, monkeypatch):
    """落到 settings.json 的 Key 必须能读回来（exe 重启后仍然有效）。"""
    import backend.llm_config as llm_config

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(llm_config, "SETTINGS_FILE", settings_file)
    reset_for_tests()

    save_llm_config(
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key="sk-persisted-abcdef123456",
        persist=True,
    )
    assert settings_file.exists()

    # 模拟进程重启：清掉内存单例，从文件重新加载
    reset_for_tests()
    config = get_llm_config()
    assert config.api_key == "sk-persisted-abcdef123456"
    assert config.configured
