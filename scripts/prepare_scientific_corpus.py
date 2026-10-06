"""Copy a small, reproducible scientific-paper corpus into the app raw store.

This script deliberately stops before database registration/ingestion.  It
validates the input PDFs, records their SHA-256 identities and creates a
manifest that can be consumed by a later ingestion run once PostgreSQL and
Redis are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_FILES = (
    "crag_2401.15884.pdf",
    "rag_2005.11401.pdf",
    "react_2210.03629.pdf",
    "selfrag_2310.11511.pdf",
    "visrag_2410.10594.pdf",
    "murag_2210.02928.pdf",
    "colbert_2004.12832.pdf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(source_dir: Path, destination_dir: Path, names: tuple[str, ...]) -> dict:
    destination_dir.mkdir(parents=True, exist_ok=True)
    papers: list[dict[str, object]] = []
    for name in names:
        source = (source_dir / name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF inputs are allowed: {source}")
        target = destination_dir / name
        if source != target:
            shutil.copy2(source, target)
        papers.append(
            {
                "file_name": name,
                "path": str(target.resolve()),
                "source_path": str(source),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    return {
        "schema_version": "scientific-corpus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "destination_dir": str(destination_dir.resolve()),
        "papers": papers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, default=Path("runtime/data/raw/internal-research"))
    parser.add_argument("--manifest", type=Path, default=Path("runtime/data/scientific-corpus-manifest.json"))
    parser.add_argument("--file", dest="files", action="append", default=None)
    args = parser.parse_args()
    names = tuple(args.files or DEFAULT_FILES)
    manifest = build_manifest(args.source_dir.expanduser(), args.destination_dir, names)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
