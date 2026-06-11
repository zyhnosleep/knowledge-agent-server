from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings

settings = get_settings()
UPLOAD_PREFIX_PATTERN = re.compile(r"^[0-9a-f]{32}-(.+)$", re.IGNORECASE)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.strip().lower())
    return normalized.strip("-") or uuid4().hex


def strip_upload_prefix(value: str) -> str:
    match = UPLOAD_PREFIX_PATTERN.match(value.strip())
    if not match:
        return value
    return match.group(1)


def display_title_from_path(path: Path) -> str:
    return strip_upload_prefix(path.stem)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_paths(project_slug: str) -> dict[str, Path]:
    raw_root = settings.raw_dir / project_slug
    wiki_root = settings.wiki_dir / project_slug
    raw_root.mkdir(parents=True, exist_ok=True)
    wiki_root.mkdir(parents=True, exist_ok=True)
    return {"raw_root": raw_root, "wiki_root": wiki_root}


async def save_upload(project_slug: str, upload: UploadFile) -> Path:
    paths = project_paths(project_slug)
    target_name = f"{uuid4().hex}-{upload.filename}"
    target_path = paths["raw_root"] / target_name
    with target_path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()
    return target_path
