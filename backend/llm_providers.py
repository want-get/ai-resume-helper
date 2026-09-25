"""大模型厂商预设。

只收录「OpenAI 兼容接口」的厂商 —— 这样一套 SDK 就能全部支持，
新增厂商也只是加一条配置，不需要改代码。

用户在前端选择厂商后会自动填好 Base URL 与常用模型名，
仍然允许手动改成任意 OpenAI 兼容地址（自建网关、中转、本地 Ollama 等）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LLMProvider:
    key: str
    label: str
    base_url: str
    models: tuple[str, ...]
    default_model: str
    api_key_hint: str = "sk-..."
    docs_url: str = ""
    note: str = ""
    # 是否必须联网（本地部署的模型可以是 False，目前都用 True）
    requires_api_key: bool = True


PROVIDERS: dict[str, LLMProvider] = {
    "deepseek": LLMProvider(
        key="deepseek",
        label="DeepSeek（推荐）",
        base_url="https://api.deepseek.com",
        models=("deepseek-chat", "deepseek-reasoner"),
        default_model="deepseek-chat",
        api_key_hint="sk-...",
        docs_url="https://platform.deepseek.com/api_keys",
        note="性价比高；deepseek-reasoner 是推理模型，速度较慢但更严谨。",
    ),
    "kimi": LLMProvider(
        key="kimi",
        label="Kimi / 月之暗面",
        base_url="https://api.moonshot.cn/v1",
        models=(
            "kimi-k2-0905-preview",
            "kimi-k2-turbo-preview",
            "moonshot-v1-128k",
            "moonshot-v1-32k",
            "moonshot-v1-8k",
        ),
        default_model="moonshot-v1-32k",
        docs_url="https://platform.moonshot.cn/console/api-keys",
        note="长上下文见长，适合长简历与长对话。",
    ),
    "dashscope": LLMProvider(
        key="dashscope",
        label="通义千问（阿里云 DashScope 兼容模式）",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        models=("qwen-plus", "qwen-max", "qwen-turbo", "qwen3-max", "qwen-long"),
        default_model="qwen-plus",
        docs_url="https://bailian.console.aliyun.com/",
        note="必须用「兼容模式」的 Base URL，否则 OpenAI SDK 无法调用。",
    ),
    "openai_compatible": LLMProvider(
        key="openai_compatible",
        label="自定义 OpenAI 兼容接口",
        base_url="",
        models=(),
        default_model="",
        docs_url="",
        note="填入任意 OpenAI 兼容服务的 Base URL 与模型名（自建网关、中转、Ollama、vLLM 等）。",
    ),
}

DEFAULT_PROVIDER = "deepseek"

# 让前端下拉框直接可用
def provider_list() -> list[dict[str, object]]:
    return [
        {
            "key": item.key,
            "label": item.label,
            "base_url": item.base_url,
            "models": list(item.models),
            "default_model": item.default_model,
            "api_key_hint": item.api_key_hint,
            "docs_url": item.docs_url,
            "note": item.note,
        }
        for item in PROVIDERS.values()
    ]


def get_provider(key: str | None) -> LLMProvider:
    return PROVIDERS.get(key or "", PROVIDERS[DEFAULT_PROVIDER])


def guess_provider(base_url: str) -> str:
    """按 Base URL 反推厂商，方便用户手填地址后自动识别。"""
    url = (base_url or "").lower()
    for provider in PROVIDERS.values():
        if provider.base_url and provider.base_url.lower() in url:
            return provider.key
    return "openai_compatible"


__all__ = [
    "DEFAULT_PROVIDER",
    "LLMProvider",
    "PROVIDERS",
    "get_provider",
    "guess_provider",
    "provider_list",
]
