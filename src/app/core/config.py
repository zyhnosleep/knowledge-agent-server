from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_mineru_bin() -> str:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "mineru.exe" if os.name == "nt" else "mineru"
    return str(Path(".venv") / scripts_dir / executable)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="LLM Wiki Server", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    database_url: str = Field(default="sqlite:///./data/app.db", alias="DATABASE_URL")
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    queue_job_timeout: int = Field(default=9000, alias="QUEUE_JOB_TIMEOUT")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    raw_dir: Path = Field(default=Path("./data/raw"), alias="RAW_DIR")
    cache_dir: Path = Field(default=Path("./data/cache"), alias="CACHE_DIR")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")

    default_project_slug: str = Field(default="internal-research", alias="DEFAULT_PROJECT_SLUG")
    default_project_name: str = Field(default="Internal Research", alias="DEFAULT_PROJECT_NAME")
    query_mode: str = Field(default="rag", alias="QUERY_MODE")
    sac_kg_enabled: bool = Field(default=True, alias="SAC_KG_ENABLED")
    vector_store_enabled: bool = Field(default=True, alias="VECTOR_STORE_ENABLED")
    vector_store_backend: str = Field(default="sqlite-vec", alias="VECTOR_STORE_BACKEND")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_generation_model: str = Field(default="qwen3:14b", alias="OLLAMA_GENERATION_MODEL")
    ollama_fast_base_url: str = Field(default="http://localhost:11435", alias="OLLAMA_FAST_BASE_URL")
    ollama_deep_base_url: str = Field(default="http://localhost:11436", alias="OLLAMA_DEEP_BASE_URL")
    ollama_embedding_base_url: str = Field(
        default="http://localhost:11435", alias="OLLAMA_EMBEDDING_BASE_URL"
    )
    ollama_fast_model: str = Field(default="qwen3:14b", alias="OLLAMA_FAST_MODEL")
    ollama_deep_model: str = Field(default="qwen3.6:27b", alias="OLLAMA_DEEP_MODEL")
    ollama_fast_context_length: int = Field(default=16384, gt=0, alias="OLLAMA_FAST_CONTEXT_LENGTH")
    ollama_deep_context_length: int = Field(default=32768, gt=0, alias="OLLAMA_DEEP_CONTEXT_LENGTH")
    ollama_fast_parallelism: int = Field(default=1, gt=0, alias="OLLAMA_FAST_PARALLELISM")
    ollama_deep_parallelism: int = Field(default=1, gt=0, alias="OLLAMA_DEEP_PARALLELISM")
    ollama_batch_model: str = Field(default="qwen3:30b", alias="OLLAMA_BATCH_MODEL")
    ollama_embedding_model: str = Field(default="qwen3-embedding:8b", alias="OLLAMA_EMBEDDING_MODEL")
    ollama_embedding_dimensions: int = Field(default=4096, alias="OLLAMA_EMBEDDING_DIMENSIONS")
    ollama_vision_model: str | None = Field(default=None, alias="OLLAMA_VISION_MODEL")
    ollama_request_timeout: int = Field(default=180, alias="OLLAMA_REQUEST_TIMEOUT")
    ollama_keep_alive: str | None = Field(default=None, alias="OLLAMA_KEEP_ALIVE")

    document_intelligence_enabled: bool = Field(default=True, alias="DOCUMENT_INTELLIGENCE_ENABLED")
    pdf_render_dpi: int = Field(default=160, alias="PDF_RENDER_DPI")
    ocr_fallback_enabled: bool = Field(default=False, alias="OCR_FALLBACK_ENABLED")
    mineru_enabled: bool = Field(default=False, alias="MINERU_ENABLED")
    mineru_bin: str = Field(default_factory=_default_mineru_bin, alias="MINERU_BIN")
    mineru_backend: str = Field(default="pipeline", alias="MINERU_BACKEND")
    mineru_model_source: str | None = Field(default=None, alias="MINERU_MODEL_SOURCE")
    mineru_output_dir: Path | None = Field(default=None, alias="MINERU_OUTPUT_DIR")
    mineru_timeout: int = Field(default=3600, alias="MINERU_TIMEOUT")
    mineru_extra_args: str = Field(default="", alias="MINERU_EXTRA_ARGS")

    external_api_enabled: bool = Field(default=False, alias="EXTERNAL_API_ENABLED")
    external_api_base_url: str = Field(default="https://api.openai.com/v1", alias="EXTERNAL_API_BASE_URL")
    external_api_key: str | None = Field(default=None, alias="EXTERNAL_API_KEY")
    external_api_model: str = Field(default="gpt-4o-mini", alias="EXTERNAL_API_MODEL")
    external_api_timeout: int = Field(default=90, alias="EXTERNAL_API_TIMEOUT")

    minio_enabled: bool = Field(default=False, alias="MINIO_ENABLED")
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="llm-wiki", alias="MINIO_BUCKET")
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    for path in (settings.data_dir, settings.raw_dir, settings.cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings
