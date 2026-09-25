"""运行时设置：让用户在前端界面直接填 API Key，改完立即生效（无需改 .env、无需重启）。

为什么需要它？
    配置只在进程启动时读一次，用户为了换个 Key 就得去编辑 `.env` 再重启后端，
    这对使用者完全不可接受。这里把「大模型配置」做成可热更新的运行时状态：

    * 修改后立即写入内存中的 ``Settings`` 单例 → 下一个请求就生效；
    * 重置 ``AsyncAIClient``，丢弃用旧 Key 建立的连接与缓存；
    * 可选 ``persist=True`` 把值写回 `.env`，重启后依然保留；
    * 可选 ``verify=True`` 用一次最小代价的真实调用校验 Key 是否可用。

安全：
    * 任何接口都**不会回显完整 Key**，只返回掩码；
    * 由 ``ALLOW_RUNTIME_SETTINGS`` 开关控制，对外暴露时可关掉；
    * 若设置了 ``SERVICE_API_KEY``，这些接口同样受 ``X-API-Key`` 鉴权保护。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, get_settings

logger = logging.getLogger("ai_resume_helper.settings")

ENV_PATH = PROJECT_ROOT / ".env"

# 允许在线修改的字段 -> .env 中的键名
EDITABLE_ENV_KEYS = {
    "api_key": "DEEPSEEK_API_KEY",
    "base_url": "DEEPSEEK_BASE_URL",
    "model": "DEEPSEEK_MODEL",
    "temperature": "LLM_TEMPERATURE",
}


def mask_key(key: str) -> str:
    """把 Key 变成掩码，避免任何接口回显明文。"""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 10:
        return key[:2] + "*" * (len(key) - 2)
    return f"{key[:6]}{'*' * 8}{key[-4:]}"


def llm_status() -> dict[str, Any]:
    """当前大模型配置状态（不含明文 Key）。"""
    settings = get_settings()
    key = settings.deepseek_api_key.strip()
    return {
        "configured": bool(key),
        "mock_mode": not bool(key) or settings.llm_mock_mode,
        "model": settings.deepseek_model,
        "base_url": settings.deepseek_base_url,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "key_masked": mask_key(key),
        "key_length": len(key),
        "key_looks_valid": key.startswith("sk-") and len(key) > 20,
        "runtime_editable": settings.allow_runtime_settings,
        "env_file": str(ENV_PATH),
        "env_file_exists": ENV_PATH.exists(),
    }


def _persist_to_env(updates: dict[str, str]) -> list[str]:
    """把配置写回 .env（只改指定键，其余内容原样保留）。"""
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    updated_keys: list[str] = []
    for env_key, value in updates.items():
        pattern = re.compile(rf"^\s*{re.escape(env_key)}\s*=")
        replaced = False
        for index, line in enumerate(lines):
            if pattern.match(line):
                lines[index] = f"{env_key}={value}"
                replaced = True
                updated_keys.append(env_key)
                break
        if not replaced:
            lines.append(f"{env_key}={value}")
            updated_keys.append(env_key)

    ENV_PATH.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    logger.info("配置已写回 %s：%s", ENV_PATH.name, updated_keys)
    return updated_keys


async def update_llm_settings(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    persist: bool = False,
    verify: bool = False,
) -> dict[str, Any]:
    """热更新大模型配置，返回更新后的状态与（可选的）校验结果。"""
    settings = get_settings()
    if not settings.allow_runtime_settings:
        raise PermissionError("服务端已关闭在线修改配置（ALLOW_RUNTIME_SETTINGS=false）")

    changed: list[str] = []
    env_updates: dict[str, str] = {}

    if api_key is not None:
        cleaned = api_key.strip().strip('"').strip("'")
        if cleaned and not cleaned.startswith("sk-"):
            # 不阻断（有些兼容网关不是 sk- 开头），但记录下来
            logger.info("API Key 不是常见的 sk- 前缀，仍按原样保存")
        settings.deepseek_api_key = cleaned
        changed.append("api_key")
        env_updates[EDITABLE_ENV_KEYS["api_key"]] = cleaned

    if base_url:
        settings.deepseek_base_url = base_url.strip().rstrip("/")
        settings.deepseek_beta_base_url = settings.deepseek_base_url + "/beta"
        changed.append("base_url")
        env_updates[EDITABLE_ENV_KEYS["base_url"]] = settings.deepseek_base_url

    if model:
        settings.deepseek_model = model.strip()
        changed.append("model")
        env_updates[EDITABLE_ENV_KEYS["model"]] = settings.deepseek_model

    if temperature is not None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature 必须在 0.0 ~ 2.0 之间")
        settings.llm_temperature = float(temperature)
        changed.append("temperature")
        env_updates[EDITABLE_ENV_KEYS["temperature"]] = str(settings.llm_temperature)

    # 关掉显式的 mock 开关（用户填了 Key 就说明想走真实模型）
    if "api_key" in changed and settings.llm_mock_mode:
        settings.llm_mock_mode = False
        changed.append("mock_mode=off")

    persisted: list[str] = []
    if persist and env_updates:
        try:
            persisted = _persist_to_env(env_updates)
        except OSError as exc:
            logger.warning("写回 .env 失败：%s", exc)

    # 必须重置客户端：旧客户端是用旧 Key/旧 base_url 建的
    from .ai_client import reset_ai_client

    await reset_ai_client()

    result: dict[str, Any] = {
        "changed": changed,
        "persisted_env_keys": persisted,
        "status": llm_status(),
    }

    if verify:
        result["verify"] = await verify_llm_connection()
    return result


async def clear_llm_key(persist: bool = False) -> dict[str, Any]:
    """清空 Key（回到离线 Mock 模式）。"""
    settings = get_settings()
    if not settings.allow_runtime_settings:
        raise PermissionError("服务端已关闭在线修改配置（ALLOW_RUNTIME_SETTINGS=false）")

    settings.deepseek_api_key = ""
    persisted: list[str] = []
    if persist:
        try:
            persisted = _persist_to_env({EDITABLE_ENV_KEYS["api_key"]: ""})
        except OSError as exc:
            logger.warning("写回 .env 失败：%s", exc)

    from .ai_client import reset_ai_client

    await reset_ai_client()
    return {"changed": ["api_key=cleared"], "persisted_env_keys": persisted, "status": llm_status()}


async def verify_llm_connection() -> dict[str, Any]:
    """用一次最小代价的真实调用校验 Key 是否可用。"""
    settings = get_settings()
    if not settings.deepseek_api_key:
        return {"ok": False, "error": "尚未配置 API Key"}

    from .ai_client import AsyncAIClient

    client = AsyncAIClient(settings)
    try:
        result = await client.chat(
            [{"role": "user", "content": "回复两个字：可用"}],
            temperature=0.0,
            max_tokens=8,
        )
        if result.ok:
            return {
                "ok": True,
                "model": result.model,
                "latency_ms": round(result.latency_ms, 1),
                "reply_preview": result.content[:40],
            }
        return {"ok": False, "error": result.error or "调用失败"}
    finally:
        await client.close()
