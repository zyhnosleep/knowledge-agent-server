"""应用配置：集中管理所有环境变量（.env 文件）并做校验。

`Settings` 继承自 pydantic-settings 的 BaseSettings，每个字段都从同名
环境变量读取（通过 alias 指定），未配置时使用默认值。

主要分组：
- 应用基本信息与数据库；
- 语义分块（semantic chunking）参数；
- Ollama 生成/嵌入模型与上下文长度；
- 上下文化（contextualization）参数；
- PDF 解析（MinerU / Document Intelligence）；
- 外部 API（synthesis 备选 provider）；
- MinIO 对象存储；
- Agent 执行约束与合成 provider；
- 认证（Feishu OAuth / session cookie）。

通过 `get_settings()`（带 lru_cache）获取单例配置实例。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_mineru_bin() -> str:
    """根据操作系统返回 MinerU 可执行文件的默认路径。"""
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "mineru.exe" if os.name == "nt" else "mineru"
    return str(Path(".venv") / scripts_dir / executable)


class Settings(BaseSettings):
    """所有应用配置。每个字段通过 alias 映射到同名环境变量。"""

    # 忽略 .env 中未声明的额外变量。
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- 应用基本信息 ----
    app_name: str = Field(default="Knowledge Agent", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # ---- 数据库与任务队列 ----
    database_url: str = Field(default="sqlite:///./data/app.db", alias="DATABASE_URL")
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    queue_job_timeout: int = Field(default=9000, alias="QUEUE_JOB_TIMEOUT")

    # ---- 数据目录 ----
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    raw_dir: Path = Field(default=Path("./data/raw"), alias="RAW_DIR")
    cache_dir: Path = Field(default=Path("./data/cache"), alias="CACHE_DIR")
    canonical_artifacts_dir: Path = Field(
        default=Path("./data/parsed"), alias="CANONICAL_ARTIFACTS_DIR"
    )
    canonical_pipeline_version: str = Field(
        default="canonical-v4", alias="CANONICAL_PIPELINE_VERSION"
    )
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")

    semantic_splitting_enabled: bool = Field(default=True, alias="SEMANTIC_SPLITTING_ENABLED")
    semantic_splitting_model: str = Field(
        default="qwen3-embedding:4b", alias="SEMANTIC_SPLITTING_MODEL"
    )
    semantic_break_percentile: int = Field(
        default=20, ge=1, le=99, alias="SEMANTIC_BREAK_PERCENTILE"
    )
    semantic_parent_min_tokens: int = Field(
        default=500, gt=0, alias="SEMANTIC_PARENT_MIN_TOKENS"
    )
    semantic_parent_target_tokens: int = Field(
        default=1200, gt=0, alias="SEMANTIC_PARENT_TARGET_TOKENS"
    )
    semantic_parent_max_tokens: int = Field(
        default=1800, gt=0, alias="SEMANTIC_PARENT_MAX_TOKENS"
    )
    semantic_child_min_tokens: int = Field(
        default=180, gt=0, alias="SEMANTIC_CHILD_MIN_TOKENS"
    )
    semantic_child_target_tokens: int = Field(
        default=400, gt=0, alias="SEMANTIC_CHILD_TARGET_TOKENS"
    )
    semantic_child_max_tokens: int = Field(
        default=600, gt=0, alias="SEMANTIC_CHILD_MAX_TOKENS"
    )
    semantic_child_overlap_tokens: int = Field(
        default=50, ge=0, alias="SEMANTIC_CHILD_OVERLAP_TOKENS"
    )
    semantic_tokenizer_name: str = Field(
        default="Qwen/Qwen3-Embedding-4B", alias="SEMANTIC_TOKENIZER_NAME"
    )
    semantic_tokenizer_revision: str = Field(
        default="5cf2132abc99cad020ac570b19d031efec650f2b",
        alias="SEMANTIC_TOKENIZER_REVISION",
    )
    semantic_tokenizer_local_path: Path | None = Field(
        default=None, alias="SEMANTIC_TOKENIZER_LOCAL_PATH"
    )

    # ---- 默认项目与检索开关 ----
    default_project_slug: str = Field(default="internal-research", alias="DEFAULT_PROJECT_SLUG")
    default_project_name: str = Field(default="Internal Research", alias="DEFAULT_PROJECT_NAME")
    query_mode: str = Field(default="rag", alias="QUERY_MODE")
    sac_kg_enabled: bool = Field(default=True, alias="SAC_KG_ENABLED")
    vector_store_enabled: bool = Field(default=True, alias="VECTOR_STORE_ENABLED")
    vector_store_backend: str = Field(default="sqlite-vec", alias="VECTOR_STORE_BACKEND")
    vector_store_strict: bool = Field(default=False, alias="VECTOR_STORE_STRICT")

    # ---- Ollama 生成/嵌入模型 ----
    ollama_generation_base_url: str = Field(
        default="http://localhost:11435", alias="OLLAMA_GENERATION_BASE_URL"
    )
    ollama_generation_model: str = Field(default="qwen3.5:9b", alias="OLLAMA_GENERATION_MODEL")
    ollama_embedding_base_url: str = Field(
        default="http://localhost:11435", alias="OLLAMA_EMBEDDING_BASE_URL"
    )
    ollama_generation_context_length: int = Field(
        default=32768, gt=0, alias="OLLAMA_GENERATION_CONTEXT_LENGTH"
    )
    ollama_generation_parallelism: int = Field(
        default=1, gt=0, alias="OLLAMA_GENERATION_PARALLELISM"
    )
    ollama_batch_model: str = Field(default="qwen3.5:9b", alias="OLLAMA_BATCH_MODEL")
    ollama_embedding_model: str = Field(default="qwen3-embedding:4b", alias="OLLAMA_EMBEDDING_MODEL")
    ollama_embedding_dimensions: int = Field(default=2560, alias="OLLAMA_EMBEDDING_DIMENSIONS")
    ollama_vision_model: str | None = Field(default=None, alias="OLLAMA_VISION_MODEL")
    ollama_request_timeout: int = Field(default=180, alias="OLLAMA_REQUEST_TIMEOUT")
    ollama_keep_alive: str | None = Field(default=None, alias="OLLAMA_KEEP_ALIVE")
    ollama_synthesis_model: str | None = Field(default=None, alias="OLLAMA_SYNTHESIS_MODEL")

    # ---- Embedding provider ----
    # ``ollama`` keeps the existing local path. ``openai-compatible`` calls a
    # remote /v1/embeddings-style API, for example a hosted
    # Qwen/Qwen3-Embedding-4B endpoint.  The API key is server-side only.
    embedding_provider: str = Field(default="ollama", alias="EMBEDDING_PROVIDER")
    embedding_api_base_url: str | None = Field(default=None, alias="EMBEDDING_API_BASE_URL")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_api_model: str = Field(
        default="Qwen/Qwen3-Embedding-4B", alias="EMBEDDING_API_MODEL"
    )
    embedding_api_timeout: int = Field(default=180, gt=0, alias="EMBEDDING_API_TIMEOUT")
    # MaaS-compatible embedding endpoints commonly cap one request at 20
    # inputs.  Keep the limit configurable while using that safe default.
    embedding_api_batch_size: int = Field(
        default=20, gt=0, alias="EMBEDDING_API_BATCH_SIZE"
    )
    # Optional provider-neutral dimension override.  When omitted, the legacy
    # ``OLLAMA_EMBEDDING_DIMENSIONS`` value remains the compatibility default,
    # so existing local deployments and tests keep the same index contract.
    embedding_dimensions: int | None = Field(
        default=None, gt=0, alias="EMBEDDING_DIMENSIONS"
    )
    embedding_revision: str | None = Field(default=None, alias="EMBEDDING_REVISION")
    embedding_processor_hash: str | None = Field(default=None, alias="EMBEDDING_PROCESSOR_HASH")

    @property
    def active_embedding_provider(self) -> str:
        """Return the normalized embedding provider identifier."""
        return self.embedding_provider.strip().lower()

    @property
    def active_embedding_model(self) -> str:
        """Return the model identity used to create vectors for this run."""
        if self.active_embedding_provider == "openai-compatible":
            return self.embedding_api_model
        return self.ollama_embedding_model

    @property
    def active_embedding_dimensions(self) -> int:
        """Return the vector dimension enforced by the active index contract."""
        return self.embedding_dimensions or self.ollama_embedding_dimensions

    contextualization_enabled: bool = Field(default=True, alias="CONTEXTUALIZATION_ENABLED")
    contextualization_base_url: str = Field(
        default="http://localhost:11435", alias="CONTEXTUALIZATION_BASE_URL"
    )
    contextualization_model: str = Field(default="qwen3.5:9b", alias="CONTEXTUALIZATION_MODEL")
    contextualization_batch_size: int = Field(
        default=12, gt=0, alias="CONTEXTUALIZATION_BATCH_SIZE"
    )
    contextualization_max_retries: int = Field(
        default=2, ge=0, alias="CONTEXTUALIZATION_MAX_RETRIES"
    )
    contextualization_timeout: int = Field(
        default=180, gt=0, alias="CONTEXTUALIZATION_TIMEOUT"
    )
    contextualization_max_sentences: int = Field(
        default=2, ge=1, le=2, alias="CONTEXTUALIZATION_MAX_SENTENCES"
    )
    contextualization_prompt_version: str = Field(
        default="context-v1", alias="CONTEXTUALIZATION_PROMPT_VERSION"
    )

    # ---- PDF 解析（MinerU / Document Intelligence / OCR） ----
    document_intelligence_enabled: bool = Field(default=True, alias="DOCUMENT_INTELLIGENCE_ENABLED")
    pdf_render_dpi: int = Field(default=160, alias="PDF_RENDER_DPI")
    ocr_fallback_enabled: bool = Field(default=False, alias="OCR_FALLBACK_ENABLED")
    mineru_enabled: bool = Field(default=True, alias="MINERU_ENABLED")
    mineru_bin: str = Field(default_factory=_default_mineru_bin, alias="MINERU_BIN")
    mineru_backend: str = Field(default="pipeline", alias="MINERU_BACKEND")
    mineru_model_source: str | None = Field(default=None, alias="MINERU_MODEL_SOURCE")
    mineru_output_dir: Path | None = Field(default=None, alias="MINERU_OUTPUT_DIR")
    mineru_timeout: int = Field(default=3600, alias="MINERU_TIMEOUT")
    mineru_extra_args: str = Field(default="", alias="MINERU_EXTRA_ARGS")
    figure_analysis_model: str = Field(default="qwen3.5:9b", alias="FIGURE_ANALYSIS_MODEL")
    formula_analysis_model: str = Field(default="qwen3.5:9b", alias="FORMULA_ANALYSIS_MODEL")
    maintenance_mode_enabled: bool = Field(default=False, alias="MAINTENANCE_MODE_ENABLED")

    # ---- 外部 API（Agent 综合的备选 provider） ----
    external_api_enabled: bool = Field(default=False, alias="EXTERNAL_API_ENABLED")
    external_api_base_url: str = Field(default="https://api.openai.com/v1", alias="EXTERNAL_API_BASE_URL")
    external_api_key: str | None = Field(default=None, alias="EXTERNAL_API_KEY")
    external_api_model: str = Field(default="gpt-4o-mini", alias="EXTERNAL_API_MODEL")
    external_api_timeout: int = Field(default=90, alias="EXTERNAL_API_TIMEOUT")

    # ---- Generation provider (DeepSeek API is opt-in) ----
    # Keep Ollama as the compatibility default.  Deployments that choose
    # DeepSeek set GENERATION_PROVIDER=deepseek and provide the key only in
    # the server environment; no generation credential is exposed to clients.
    generation_provider: str = Field(default="ollama", alias="GENERATION_PROVIDER")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    generation_timeout_seconds: int = Field(
        default=90, gt=0, alias="GENERATION_TIMEOUT_SECONDS"
    )
    generation_max_retries: int = Field(
        default=1, ge=0, alias="GENERATION_MAX_RETRIES"
    )
    generation_max_output_tokens: int = Field(
        default=2048, gt=0, alias="GENERATION_MAX_OUTPUT_TOKENS"
    )
    generation_retry_backoff_seconds: float = Field(
        default=0.5, ge=0, alias="GENERATION_RETRY_BACKOFF_SECONDS"
    )
    generation_concurrency: int = Field(default=1, gt=0, alias="GENERATION_CONCURRENCY")

    # ---- MinIO 对象存储（可选） ----
    minio_enabled: bool = Field(default=False, alias="MINIO_ENABLED")
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="knowledge-agent", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # ---- Agent 执行约束与合成 ----
    agent_enabled: bool = Field(default=True, alias="AGENT_ENABLED")
    agent_max_steps: int = Field(default=8, alias="AGENT_MAX_STEPS")
    agent_max_tool_calls: int = Field(default=5, alias="AGENT_MAX_TOOL_CALLS")
    agent_budget_tokens: int = Field(default=20000, alias="AGENT_BUDGET_TOKENS")
    agent_timeout_seconds: int = Field(default=90, alias="AGENT_TIMEOUT_SECONDS")
    agent_allow_external_network: bool = Field(default=False, alias="AGENT_ALLOW_EXTERNAL_NETWORK")
    # 40 轮测试会话（每轮约 3-5 条 turn）不超过 200 条，不触发压缩删除；
    # 超长会话仍由 compact 删除最旧轮次兜底（2026-08-11 grill 收敛）。
    agent_max_conversation_turns: int = Field(default=200, alias="AGENT_MAX_CONVERSATION_TURNS")
    agent_synthesis_provider: str = Field(default="auto", alias="AGENT_SYNTHESIS_PROVIDER")
    agent_conversation_ttl_days: int = Field(default=30, alias="AGENT_CONVERSATION_TTL_DAYS")
    agent_trace_retention_days: int = Field(default=30, alias="AGENT_TRACE_RETENTION_DAYS")
    agent_stream_heartbeat_seconds: int = Field(default=15, alias="AGENT_STREAM_HEARTBEAT_SECONDS")

    quality_reports_dir: Path = Field(default=Path("./tmp"), alias="QUALITY_REPORTS_DIR")

    # ---- 认证（Feishu OAuth + session cookie） ----
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    auth_session_secret: str | None = Field(default=None, alias="AUTH_SESSION_SECRET")
    auth_session_cookie_name: str = Field(default="nri_session", alias="AUTH_SESSION_COOKIE_NAME")
    auth_csrf_cookie_name: str = Field(default="nri_csrf", alias="AUTH_CSRF_COOKIE_NAME")
    auth_state_cookie_name: str = Field(default="nri_oauth_state", alias="AUTH_STATE_COOKIE_NAME")
    auth_session_max_age_seconds: int = Field(default=604_800, alias="AUTH_SESSION_MAX_AGE_SECONDS")
    auth_state_max_age_seconds: int = Field(default=600, alias="AUTH_STATE_MAX_AGE_SECONDS")
    auth_cookie_secure: bool = Field(default=True, alias="AUTH_COOKIE_SECURE")
    auth_cookie_samesite: str = Field(default="lax", alias="AUTH_COOKIE_SAMESITE")

    feishu_app_id: str | None = Field(default=None, alias="FEISHU_APP_ID")
    feishu_app_secret: str | None = Field(default=None, alias="FEISHU_APP_SECRET")
    feishu_redirect_uri: str | None = Field(default=None, alias="FEISHU_REDIRECT_URI")
    feishu_auth_url: str = Field(
        default="https://accounts.feishu.cn/open-apis/authen/v1/authorize",
        alias="FEISHU_AUTH_URL",
    )
    feishu_token_url: str = Field(
        default="https://open.feishu.cn/open-apis/authen/v2/oauth/token",
        alias="FEISHU_TOKEN_URL",
    )
    feishu_user_info_url: str = Field(
        default="https://open.feishu.cn/open-apis/authen/v1/user_info",
        alias="FEISHU_USER_INFO_URL",
    )
    feishu_allowed_tenant: str | None = Field(default=None, alias="FEISHU_ALLOWED_TENANT")

    # ---- 飞书机器人（长连接群聊，独立进程） ----
    feishu_bot_enabled: bool = Field(default=False, alias="FEISHU_BOT_ENABLED")
    feishu_bot_allowed_chat_ids: str = Field(default="", alias="FEISHU_BOT_ALLOWED_CHAT_IDS")
    feishu_bot_inbox_project: str = Field(
        default="feishu-inbox", alias="FEISHU_BOT_INBOX_PROJECT"
    )
    feishu_bot_api_base_url: str = Field(
        default="http://127.0.0.1:8002", alias="FEISHU_BOT_API_BASE_URL"
    )

    @model_validator(mode="after")
    def validate_semantic_token_limits(self) -> Settings:
        """校验语义分块的 token 上下限：min <= target <= max，且 overlap < min。"""
        if not (
            self.semantic_parent_min_tokens
            <= self.semantic_parent_target_tokens
            <= self.semantic_parent_max_tokens
        ):
            raise ValueError("semantic parent token limits must satisfy min <= target <= max")

        if not (
            self.semantic_child_min_tokens
            <= self.semantic_child_target_tokens
            <= self.semantic_child_max_tokens
        ):
            raise ValueError("semantic child token limits must satisfy min <= target <= max")

        if not (0 <= self.semantic_child_overlap_tokens < self.semantic_child_min_tokens):
            raise ValueError("semantic child overlap tokens must satisfy 0 <= overlap < min")

        return self

    @property
    def parent_token_limits(self) -> tuple[int, int, int]:
        """父块（parent chunk）的 (min, target, max) token 限制。"""
        return (
            self.semantic_parent_min_tokens,
            self.semantic_parent_target_tokens,
            self.semantic_parent_max_tokens,
        )

    @property
    def child_token_limits(self) -> tuple[int, int, int]:
        """子块（child chunk）的 (min, target, max) token 限制。"""
        return (
            self.semantic_child_min_tokens,
            self.semantic_child_target_tokens,
            self.semantic_child_max_tokens,
        )

    @property
    def child_overlap_tokens(self) -> int:
        return self.semantic_child_overlap_tokens


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局单例配置，并确保基础数据目录存在。"""
    settings = Settings()
    for path in (settings.data_dir, settings.raw_dir, settings.cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings
