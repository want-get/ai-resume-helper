"""大模型配置：前端在线填写 → 落盘到 ``settings.json`` → 立即生效。

优先级：``settings.json``（用户在界面上保存的） > 环境变量 / .env > 内置默认。

为什么不用 .env？
    用户在界面上改配置是常规操作，而 .env 是「部署配置」，不适合程序自己回写；
    单机版把用户配置写进数据目录的 JSON 文件更干净，也便于备份和排查。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from paths import SETTINGS_FILE

from .llm_providers import DEFAULT_PROVIDER, get_provider, guess_provider

logger = logging.getLogger("ai_resume_helper.llm_config")


@dataclass
class LLMConfig:
    """一份完整的大模型调用配置。"""

    provider: str = DEFAULT_PROVIDER
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 2

    def resolved_base_url(self) -> str:
        if self.base_url.strip():
            return self.base_url.strip().rstrip("/")
        return get_provider(self.provider).base_url.rstrip("/")

    def resolved_model(self) -> str:
        if self.model.strip():
            return self.model.strip()
        return get_provider(self.provider).default_model

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip()) and bool(self.resolved_base_url())

    def masked_key(self) -> str:
        key = self.api_key.strip()
        if not key:
            return ""
        if len(key) <= 10:
            return key[:2] + "*" * max(0, len(key) - 2)
        return f"{key[:6]}{'*' * 8}{key[-4:]}"

    def to_public_dict(self) -> dict[str, Any]:
        """给前端的状态（**绝不包含明文 Key**）。"""
        base_url = self.resolved_base_url()
        return {
            "provider": self.provider,
            "provider_label": get_provider(self.provider).label,
            "base_url": base_url,
            "model": self.resolved_model(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "configured": self.configured,
            "key_masked": self.masked_key(),
            "key_length": len(self.api_key.strip()),
            "single_user_mode": True,
        }


_lock = threading.RLock()
_config: LLMConfig | None = None


def _defaults_from_env() -> LLMConfig:
    """环境变量 / .env 作为初始默认值（开发时方便）。

    注意：``.env`` 是被 pydantic-settings 读进 ``Settings`` 的，
    **不会**出现在 ``os.environ`` 里，所以必须经由 ``get_settings()`` 取，
    否则开发时填在 .env 里的 Key 会被忽略。
    """
    from config import get_settings

    settings = get_settings()
    base_url = (settings.deepseek_base_url or "").strip()
    api_key = (settings.deepseek_api_key or "").strip()
    model = (settings.deepseek_model or "").strip()
    provider = guess_provider(base_url) if base_url else DEFAULT_PROVIDER
    return LLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=float(settings.llm_temperature),
        max_tokens=int(settings.llm_max_tokens),
        timeout=float(settings.llm_timeout),
        max_retries=int(settings.llm_max_retries),
    )


def _load_from_file() -> LLMConfig | None:
    if not SETTINGS_FILE.exists():
        return None
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 %s 失败，使用默认配置：%s", SETTINGS_FILE.name, exc)
        return None

    section = raw.get("llm") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        return None

    base = _defaults_from_env()
    for key, value in section.items():
        if hasattr(base, key) and value is not None:
            setattr(base, key, value)
    if not base.provider:
        base.provider = guess_provider(base.base_url)
    return base


def _read_all() -> dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_all(data: dict[str, Any]) -> None:
    """原子写入，避免程序被强杀时把配置文件写坏。"""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(SETTINGS_FILE.parent), prefix=".settings-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_name, SETTINGS_FILE)
    except OSError:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def get_llm_config() -> LLMConfig:
    """进程内单例。"""
    global _config
    with _lock:
        if _config is None:
            _config = _load_from_file() or _defaults_from_env()
        return _config


def save_llm_config(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    persist: bool = True,
) -> LLMConfig:
    """更新配置并（可选）落盘。返回更新后的配置对象。"""
    global _config
    with _lock:
        config = get_llm_config()

        if provider is not None:
            config.provider = provider.strip() or DEFAULT_PROVIDER
            # 换厂商时，如果用户没手动改地址/模型，就跟着换成新厂商的默认值
            if base_url is None or not base_url.strip():
                config.base_url = ""
            if model is None or not model.strip():
                config.model = ""
        if base_url is not None:
            clean = base_url.strip()
            config.base_url = clean
            if clean and not provider:
                config.provider = guess_provider(clean)
        if model is not None:
            config.model = model.strip()
        if api_key is not None:
            config.api_key = api_key.strip().strip('"').strip("'")
        if temperature is not None:
            if not 0.0 <= float(temperature) <= 2.0:
                raise ValueError("temperature 必须在 0.0 ~ 2.0 之间")
            config.temperature = float(temperature)
        if max_tokens is not None:
            if int(max_tokens) < 64:
                raise ValueError("max_tokens 至少为 64")
            config.max_tokens = int(max_tokens)

        if persist:
            data = _read_all()
            data["llm"] = asdict(config)
            data["version"] = 3
            _write_all(data)
            logger.info("大模型配置已保存到 %s", SETTINGS_FILE)

        return config


def clear_api_key(persist: bool = True) -> LLMConfig:
    return save_llm_config(api_key="", persist=persist)


def reset_for_tests() -> None:
    """测试用：丢弃内存缓存，下次重新读文件。"""
    global _config
    with _lock:
        _config = None


__all__ = [
    "LLMConfig",
    "clear_api_key",
    "get_llm_config",
    "reset_for_tests",
    "save_llm_config",
]
