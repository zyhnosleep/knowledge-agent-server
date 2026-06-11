from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="LLM Wiki Server", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    database_url: str = Field(default="sqlite:///./data/app.db", alias="DATABASE_URL")
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    queue_job_timeout: int = Field(default=3600, alias="QUEUE_JOB_TIMEOUT")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    raw_dir: Path = Field(default=Path("./data/raw"), alias="RAW_DIR")
    wiki_dir: Path = Field(default=Path("./data/wiki"), alias="WIKI_DIR")
    cache_dir: Path = Field(default=Path("./data/cache"), alias="CACHE_DIR")

    default_project_slug: str = Field(default="internal-research", alias="DEFAULT_PROJECT_SLUG")
    default_project_name: str = Field(default="Internal Research", alias="DEFAULT_PROJECT_NAME")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_generation_model: str = Field(default="qwen3:14b", alias="OLLAMA_GENERATION_MODEL")
    ollama_batch_model: str = Field(default="qwen3:30b", alias="OLLAMA_BATCH_MODEL")
    ollama_embedding_model: str = Field(default="qwen3-embedding:8b", alias="OLLAMA_EMBEDDING_MODEL")
    ollama_vision_model: str | None = Field(default=None, alias="OLLAMA_VISION_MODEL")
    ollama_request_timeout: int = Field(default=180, alias="OLLAMA_REQUEST_TIMEOUT")

    document_intelligence_enabled: bool = Field(default=True, alias="DOCUMENT_INTELLIGENCE_ENABLED")
    pdf_render_dpi: int = Field(default=160, alias="PDF_RENDER_DPI")
    ocr_fallback_enabled: bool = Field(default=False, alias="OCR_FALLBACK_ENABLED")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    for path in (settings.data_dir, settings.raw_dir, settings.wiki_dir, settings.cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings
