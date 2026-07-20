from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_mineru_bin() -> str:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "mineru.exe" if os.name == "nt" else "mineru"
    return str(Path(".venv") / scripts_dir / executable)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Knowledge Agent", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    database_url: str = Field(default="sqlite:///./data/app.db", alias="DATABASE_URL")
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    queue_job_timeout: int = Field(default=9000, alias="QUEUE_JOB_TIMEOUT")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    raw_dir: Path = Field(default=Path("./data/raw"), alias="RAW_DIR")
    cache_dir: Path = Field(default=Path("./data/cache"), alias="CACHE_DIR")
    canonical_artifacts_dir: Path = Field(
        default=Path("./data/parsed"), alias="CANONICAL_ARTIFACTS_DIR"
    )
    canonical_pipeline_version: str = Field(
        default="canonical-v1", alias="CANONICAL_PIPELINE_VERSION"
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

    default_project_slug: str = Field(default="internal-research", alias="DEFAULT_PROJECT_SLUG")
    default_project_name: str = Field(default="Internal Research", alias="DEFAULT_PROJECT_NAME")
    query_mode: str = Field(default="rag", alias="QUERY_MODE")
    sac_kg_enabled: bool = Field(default=True, alias="SAC_KG_ENABLED")
    vector_store_enabled: bool = Field(default=True, alias="VECTOR_STORE_ENABLED")
    vector_store_backend: str = Field(default="sqlite-vec", alias="VECTOR_STORE_BACKEND")

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

    external_api_enabled: bool = Field(default=False, alias="EXTERNAL_API_ENABLED")
    external_api_base_url: str = Field(default="https://api.openai.com/v1", alias="EXTERNAL_API_BASE_URL")
    external_api_key: str | None = Field(default=None, alias="EXTERNAL_API_KEY")
    external_api_model: str = Field(default="gpt-4o-mini", alias="EXTERNAL_API_MODEL")
    external_api_timeout: int = Field(default=90, alias="EXTERNAL_API_TIMEOUT")

    minio_enabled: bool = Field(default=False, alias="MINIO_ENABLED")
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="knowledge-agent", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    agent_enabled: bool = Field(default=True, alias="AGENT_ENABLED")
    agent_max_steps: int = Field(default=8, alias="AGENT_MAX_STEPS")
    agent_max_tool_calls: int = Field(default=5, alias="AGENT_MAX_TOOL_CALLS")
    agent_budget_tokens: int = Field(default=20000, alias="AGENT_BUDGET_TOKENS")
    agent_timeout_seconds: int = Field(default=90, alias="AGENT_TIMEOUT_SECONDS")
    agent_allow_external_network: bool = Field(default=False, alias="AGENT_ALLOW_EXTERNAL_NETWORK")
    agent_max_conversation_turns: int = Field(default=20, alias="AGENT_MAX_CONVERSATION_TURNS")
    agent_synthesis_provider: str = Field(default="auto", alias="AGENT_SYNTHESIS_PROVIDER")
    agent_conversation_ttl_days: int = Field(default=30, alias="AGENT_CONVERSATION_TTL_DAYS")
    agent_trace_retention_days: int = Field(default=30, alias="AGENT_TRACE_RETENTION_DAYS")
    agent_stream_heartbeat_seconds: int = Field(default=15, alias="AGENT_STREAM_HEARTBEAT_SECONDS")

    quality_reports_dir: Path = Field(default=Path("./tmp"), alias="QUALITY_REPORTS_DIR")

    # Authentication -------------------------------------------------------
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

    @model_validator(mode="after")
    def validate_semantic_token_limits(self) -> Settings:
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
        return (
            self.semantic_parent_min_tokens,
            self.semantic_parent_target_tokens,
            self.semantic_parent_max_tokens,
        )

    @property
    def child_token_limits(self) -> tuple[int, int, int]:
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
    settings = Settings()
    for path in (settings.data_dir, settings.raw_dir, settings.cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings
