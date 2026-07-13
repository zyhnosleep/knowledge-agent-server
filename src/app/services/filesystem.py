from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings

settings = get_settings()
UPLOAD_PREFIX_PATTERN = re.compile(r"^[0-9a-f]{32}-(.+)$", re.IGNORECASE)
PROJECT_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff._-]{0,119}$")
# Heuristic for filenames that are just UUIDs, hashes, or other internal sample
# identifiers rather than human-readable document titles.
INTERNAL_SAMPLE_PATTERN = re.compile(r"^[0-9a-f]{24,}$", re.IGNORECASE)
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class StoragePathError(ValueError):
    """Base class for storage errors that API routes can map to client errors."""


class InvalidStoragePathError(StoragePathError):
    pass


class UploadTooLargeError(StoragePathError):
    pass


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


def looks_like_internal_sample(value: str) -> bool:
    """Return True when *value* looks like a UUID/hash sample name, not a title."""
    text = strip_upload_prefix(str(value)).strip()
    if not text:
        return True
    if UUID_PATTERN.fullmatch(text):
        return True
    if INTERNAL_SAMPLE_PATTERN.fullmatch(re.sub(r"[^0-9a-fA-F]", "", text)):
        return True
    return False


def readable_title_from_path(path: Path) -> str | None:
    """Return a human-readable title from *path* or None if it is just an internal id."""
    title = display_title_from_path(path)
    if looks_like_internal_sample(title):
        return None
    return title


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_within(child: Path, parent: Path) -> Path:
    resolved_child = child.expanduser().resolve()
    resolved_parent = parent.expanduser().resolve()
    try:
        resolved_child.relative_to(resolved_parent)
    except ValueError as exc:
        raise InvalidStoragePathError(f"Path escapes storage root: {child}") from exc
    return resolved_child


def safe_project_slug(project_slug: str) -> str:
    value = (project_slug or "").strip()
    if not value:
        raise InvalidStoragePathError("Project slug is required.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "/" in value or "\\" in value:
        raise InvalidStoragePathError("Project slug must not contain path components.")
    if not PROJECT_SLUG_PATTERN.fullmatch(value):
        raise InvalidStoragePathError("Project slug contains unsupported characters.")
    return value


def _safe_upload_filename(filename: str | None) -> str:
    value = (filename or "").strip()
    if not value or value in {".", ".."}:
        raise InvalidStoragePathError("File name is required.")
    path = Path(value)
    if path.is_absolute() or path.name != value or ".." in path.parts or "/" in value or "\\" in value:
        raise InvalidStoragePathError("File name must not contain path components.")
    sanitized = re.sub(r"[\x00-\x1f\x7f]+", "", value).strip()
    if ":" in sanitized:
        raise InvalidStoragePathError("File name contains unsupported characters.")
    if not sanitized or sanitized in {".", ".."}:
        raise InvalidStoragePathError("File name is invalid.")
    return sanitized


def project_paths(project_slug: str) -> dict[str, Path]:
    safe_slug = safe_project_slug(project_slug)
    raw_base = settings.raw_dir.expanduser().resolve()
    raw_base.mkdir(parents=True, exist_ok=True)
    raw_root = _ensure_within(raw_base / safe_slug, raw_base)
    raw_root.mkdir(parents=True, exist_ok=True)
    return {"raw_root": raw_root}


async def save_upload(project_slug: str, upload: UploadFile) -> Path:
    partial_path: Path | None = None
    try:
        paths = project_paths(project_slug)
        safe_filename = _safe_upload_filename(upload.filename)
        target_name = f"{uuid4().hex}-{safe_filename}"
        target_path = _ensure_within(paths["raw_root"] / target_name, paths["raw_root"])
        partial_path = _ensure_within(paths["raw_root"] / f".{target_name}.part", paths["raw_root"])
        total_bytes = 0
        with partial_path.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise UploadTooLargeError(f"Upload exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes}).")
                handle.write(chunk)
        partial_path.replace(target_path)
        return target_path
    except Exception:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
