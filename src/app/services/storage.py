from __future__ import annotations

from pathlib import Path

from minio import Minio

from app.core.config import get_settings

settings = get_settings()


class ObjectStorage:
    def __init__(self) -> None:
        self.enabled = settings.minio_enabled
        self.client = None
        if self.enabled:
            self.client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            )
            self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        assert self.client is not None
        if not self.client.bucket_exists(settings.minio_bucket):
            self.client.make_bucket(settings.minio_bucket)

    def upload(self, local_path: Path, object_name: str) -> str | None:
        if not self.enabled or self.client is None:
            return None
        self.client.fput_object(settings.minio_bucket, object_name, str(local_path))
        return object_name
