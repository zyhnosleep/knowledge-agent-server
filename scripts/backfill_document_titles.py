#!/usr/bin/env python3
"""Idempotent backfill of human-readable document titles.

Walks every Document row and rewrites ``document.title`` using a generic
resolution from parser metadata, existing title, and filename, skipping
internal-looking UUID/hash sample names. Existing ``metadata.source_slug``
and ``metadata.source_title`` are preserved so source slugs remain stable.

Usage:
    python scripts/backfill_document_titles.py

The script uses the application's DATABASE_URL from the environment or
``.env`` file and is safe to run repeatedly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Make src/ importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select

from app.db.session import SessionLocal, init_db
from app.models.records import Document
from app.services.filesystem import (
    looks_like_internal_sample,
    readable_title_from_path,
    strip_upload_prefix,
)


def _generic_resolve_title(document: Document) -> str:
    """Return the best human-readable title without re-parsing the file.

    Preference order:
      1. Non-internal title already in parser metadata.
      2. Non-internal existing document.title.
      3. Human-readable filename.
      4. Stripped fallback from any available identifier.
      5. "Untitled document".
    """
    candidates: list[str] = []

    metadata = document.metadata_json or {}
    metadata_title = metadata.get("title") if isinstance(metadata, dict) else None
    if metadata_title and not looks_like_internal_sample(metadata_title):
        candidates.append(str(metadata_title).strip())

    existing_title = document.title
    if existing_title and not looks_like_internal_sample(existing_title):
        candidates.append(existing_title.strip())

    file_name = document.file_name
    if file_name:
        readable = readable_title_from_path(Path(file_name))
        if readable and not looks_like_internal_sample(readable):
            candidates.append(readable.strip())

    # Final fallback: strip upload prefixes from whatever we have.
    for fallback in (existing_title, file_name, document.raw_path):
        if fallback:
            cleaned = strip_upload_prefix(str(fallback)).strip()
            if cleaned and not looks_like_internal_sample(cleaned):
                candidates.append(cleaned)
                break

    return candidates[0] if candidates else "Untitled document"


def _is_hex_hash(value: str) -> bool:
    """Return True when *value* looks like a hexadecimal hash/UUID string."""
    cleaned = re.sub(r"[^a-fA-F0-9]", "", value)
    return len(cleaned) >= 24 and len(cleaned) / len(value) >= 0.75


def backfill_titles(dry_run: bool = False) -> dict[str, int]:
    """Backfill document titles. Returns counts for changed/skipped/total."""
    init_db()
    db = SessionLocal()
    try:
        documents = db.scalars(select(Document)).all()
        changed = 0
        skipped = 0
        for document in documents:
            current = document.title or ""
            resolved = _generic_resolve_title(document)
            # Never promote a hash-like string over a human-readable current title.
            if _is_hex_hash(resolved) and current and not _is_hex_hash(current):
                skipped += 1
                continue
            if resolved != current:
                if not dry_run:
                    document.title = resolved
                changed += 1
            else:
                skipped += 1
        if not dry_run:
            db.commit()
        return {"total": len(documents), "changed": changed, "skipped": skipped}
    finally:
        db.close()


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    counts = backfill_titles(dry_run=dry_run)
    mode = "Dry-run" if dry_run else "Backfill"
    print(
        f"{mode} complete: total={counts['total']}, "
        f"changed={counts['changed']}, skipped={counts['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
