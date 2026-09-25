"""全局配置模块（单机版）。

分两类配置，不要混：

1. **部署/调优参数**（本文件）——端口、并发、切分参数、防幻觉阈值等。
   通过环境变量或 `.env` 覆盖，打包成 exe 后基本用默认值即可。
2. **用户配置**（大模型 Key / 厂商 / 模型名）——由用户在界面上填写，
   存在数据目录的 ``settings.json``，见 ``backend/llm_config.py``。

数据与代码分离：可写数据统一由 ``paths.py`` 解析（源码运行在 ``<项目>/data``，
打包后落在 exe 同级目录或 ``%LOCALAPPDATA%``），代码/静态资源只读。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from paths import BUNDLE_ROOT, DATA_HOME, SEED_DIR, WEB_DIR  # noqa: F401

# 兼容旧引用：项目根 / 只读资源目录
PROJECT_ROOT = BUNDLE_ROOT
DATA_DIR = DATA_HOME


class Settings(BaseSettings):
    """应用配置。"""

    model_config = SettingsConfigDict(
        env_file=str(BUNDLE_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==================== 应用 ====================
    app_name: str = "AI 求职助手"
    app_version: str = "3.0.0"
    debug: bool = False

    # 单机模式：没有多用户、没有登录、没有鉴权
    single_user_mode: bool = True

    # 后端监听地址（main.py / server.py 使用）
    # 单机版只监听本机，避免把用户的 API Key 暴露到局域网
    host: str = "127.0.0.1"
    port: int = 8000
    # 端口被占用时向后顺延尝试的个数
    port_scan_limit: int = 20
    # 启动后自动打开浏览器
    open_browser: bool = True

    # 简单鉴权：为空表示不校验；设置后请求需带 X-API-Key 头
    service_api_key: str = ""

    # 允许跨域的来源，逗号分隔；"*" 表示全部放行
    cors_origins: str = "*"

    # 前端调用后端使用的地址（GUI 与 CLI 客户端使用）
    backend_base_url: str = "http://127.0.0.1:8000"

    # ==================== 大模型 ====================
    # 注意：这些只是「开发期的环境变量默认值」。
    # 用户在界面上保存的配置存在 settings.json，优先级高于这里，
    # 具体见 backend/llm_config.py。单机版不要求用户碰 .env。
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    # strict 模式（Function Calling 严格 JSON Schema）需要 /beta 端点
    deepseek_beta_base_url: str = "https://api.deepseek.com/beta"
    deepseek_model: str = "deepseek-chat"
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    # 离线 Mock 模式：不调用真实大模型，用确定性桩响应跑通全链路
    # （自测、压测用；未配置 API Key 时会自动进入该模式）
    llm_mock_mode: bool = False

    # 防幻觉核心参数之一：低 temperature
    llm_temperature: float = 0.2
    llm_temperature_creative: float = 0.5
    llm_max_tokens: int = 2048

    # ==================== 并发控制 ====================
    # 单机版不需要为上百并发做预留，默认值调小，避免无谓占用资源
    # 同时进行的「大模型调用」上限，保护上游配额
    max_concurrent_llm: int = 8
    # 同时处理的 HTTP 请求上限
    max_concurrent_requests: int = 64
    # 单个请求获取并发令牌的最长等待时间（秒），超时返回 503
    concurrency_acquire_timeout: float = 30.0
    # Chroma 同步检索使用的线程池大小（asyncio.to_thread 的默认执行器）
    thread_pool_size: int = 16

    # ==================== 数据库 ====================
    # 单机版默认 SQLite（本地文件），不需要用户安装任何数据库服务。
    # 想用 MySQL 的话把 DATABASE_URL 改成 mysql+aiomysql://... 即可；
    # 连接失败会自动回退 SQLite，见 backend/db/session.py。
    database_url: str = f"sqlite+aiosqlite:///{(DATA_DIR / 'app.db').as_posix()}"
    sqlite_fallback_url: str = f"sqlite+aiosqlite:///{(DATA_DIR / 'app.db').as_posix()}"
    mysql_auto_create_database: bool = True
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_recycle: int = 1800
    # 启动时若岗位/薪资表为空，自动导入 data/seed 种子数据
    auto_seed_db: bool = True
    # 把每个请求的延迟写入 request_logs 表（批量异步写，不阻塞请求）
    # 单机版没有并发压力，默认关掉，减少无谓写盘
    persist_request_logs: bool = False

    # ==================== RAG 知识库 ====================
    chroma_dir: str = str(DATA_DIR / "chroma")
    rag_collection_questions: str = "interview_questions"
    rag_collection_jobs: str = "job_descriptions"
    # 用户专属知识库（简历 + 目标岗位 + 抓到的同类岗位）
    rag_collection_personal: str = "personal_kb"

    # 递归切分 + 重叠
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 100
    # 中文按字计算长度，英文按字符
    rag_min_chunk_chars: int = 40

    # 离线向量化（字符 n-gram 哈希 TF-IDF）
    rag_embed_dim: int = 4096
    rag_ngram_min: int = 2
    rag_ngram_max: int = 3

    # 检索
    rag_top_k: int = 5
    rag_candidate_k: int = 20          # 混合检索各路召回的候选数
    rag_hybrid: bool = True            # 向量 + BM25 混合
    # Chroma 的 query 内部有锁，吞吐会封顶；开启后进程内维护一份热向量索引
    # 作为读缓存（数据仍以 Chroma 为准），显著提升高并发检索能力。
    rag_memory_index: bool = True
    rag_memory_index_max_chunks: int = 20000
    # 内存索引 / IDF 状态文件的有效期（秒）。外部脚本重建知识库后，
    # 运行中的服务最多这么久之后自动热加载新索引。
    rag_state_ttl: float = 3.0
    rag_rrf_k: int = 60                # RRF 融合常数
    # 相关性下限（向量余弦相似度）。低于该值且关键词信息量也不足时，
    # 认为「知识库里没有相关内容」，上层会据此拒答而不是硬凑片段。
    # 经验值（本知识库语料）：相关提问通常 >= 0.13，无关提问通常 <= 0.09。
    rag_min_score: float = 0.10
    # 关键词信息量下限：命中的查询词 IDF 之和（BM25 的绝对信息量）。
    # 长问句的向量相似度会被稀释，用这条判据兜底：只要提问里的低频实词
    # 真的被文档覆盖（例如「缓存/穿透/GIL」），就认定相关。
    # 经验值：相关提问 >= 13，无关闲聊 <= 9。换语料后建议重新校准。
    rag_min_matched_idf: float = 11.0
    # 启动时若知识库为空，自动用种子数据建库
    auto_build_kb: bool = True

    # ==================== 防幻觉 ====================
    enable_citation_check: bool = True     # 校验模型引用的编号是否真实存在
    require_citation: bool = True          # 要求知识库问答必须给出引用
    refuse_without_context: bool = True    # 没有检索到资料时直接拒答而非自由发挥

    # ==================== 长上下文（滑动窗口 + 摘要） ====================
    # 保留原文的最近消息条数
    memory_window_messages: int = 8
    # 窗口原文的最大字符数（超出则继续向前淘汰）
    memory_window_max_chars: int = 4000
    # 未摘要的历史累积超过该字符数时触发一次摘要
    memory_summary_trigger_chars: int = 1500
    # 摘要文本的最大字数
    memory_summary_max_chars: int = 600
    memory_enable_summary: bool = True
    # 估算 token 时使用的系数（中文约 1 token ≈ 1.5 字）
    tokens_per_char: float = 0.67

    # ==================== Function Calling ====================
    tools_enabled: bool = True
    max_tool_rounds: int = 4               # 单次请求最多几轮工具调用
    tool_timeout: float = 10.0

    # 外部实时数据 API（可选）。配置后优先走真实外部接口，
    # 未配置或调用失败时回退到本地数据集，保证离线可用。
    job_api_base: str = ""
    job_api_key: str = ""
    salary_api_base: str = ""
    salary_api_key: str = ""
    # 岗位数据源的 TTL 缓存（秒）。公开接口单个响应可达 2MB，
    # 每次调用都重新抓取既慢又容易触发限流。
    job_cache_ttl: float = 300.0

    # ==================== 运行时设置 ====================
    # 是否允许通过 API / 前端界面在线修改大模型配置（API Key 等）。
    # 单机版必须开启：用户就是在界面上填 Key 的。
    allow_runtime_settings: bool = True

    # ==================== 定向抓取（中文岗位） ====================
    # 用 Playwright 驱动**系统自带的 Edge/Chrome**，不打包 Chromium，
    # 所以 exe 里不含浏览器；用户机器没有 Edge 时会自动尝试 Chrome。
    crawler_browser_channel: str = "msedge"     # msedge / chrome / chromium
    crawler_headless: bool = True
    crawler_navigate_timeout_ms: int = 25000
    crawler_render_wait_ms: int = 1200          # 等 JS 渲染出列表的额外等待
    crawler_max_pages: int = 5                  # 应用内分页最多翻几页
    crawler_max_jobs: int = 200                 # 单次抓取最多入库多少条
    crawler_concurrency: int = 3                # 同时抓几个来源
    # 详情页补全 JD：只对前 N 条做，避免抓取时间过长
    crawler_detail_limit: int = 15
    # 常见公司官网职位入口（前端可编辑；这里只给默认值）
    crawler_sources_file: str = "sources.json"

    # ==================== 校验 ====================
    @field_validator("llm_temperature", "llm_temperature_creative")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature 必须在 0.0 ~ 2.0 之间")
        return v

    @field_validator("rag_chunk_overlap")
    @classmethod
    def _check_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("chunk_overlap 不能为负数")
        return v

    # ==================== 派生属性 ====================
    @property
    def cors_origin_list(self) -> list[str]:
        """单机版只服务本机页面，默认放开同源即可。"""
        raw = (self.cors_origins or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def kb_collections(self) -> list[str]:
        return [self.rag_collection_questions, self.rag_collection_jobs]

    @property
    def all_kb_collections(self) -> list[str]:
        return [*self.kb_collections, self.rag_collection_personal]

    def ensure_dirs(self) -> None:
        """确保数据与资源目录存在。"""
        from paths import ensure_dirs as _ensure

        _ensure()
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        SEED_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""
    settings = Settings()
    settings.ensure_dirs()
    return settings


# 便捷引用
settings = get_settings()
