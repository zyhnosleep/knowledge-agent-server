"""Dependency-free asset fingerprinting shared by API tooling and model serving."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _digest(paths: list[Path], root: Path, descriptor: Mapping[str, Any]) -> str:
    result = hashlib.sha256(json.dumps(dict(descriptor), sort_keys=True, separators=(",", ":")).encode())
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        result.update(path.relative_to(root).as_posix().encode() + b"\0")
        asset_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                asset_hash.update(block)
        result.update(asset_hash.digest())
    return "sha256:" + result.hexdigest()


def fingerprint_embedding_assets(root: Path, *, loader_identity: Mapping[str, Any]) -> dict[str, str]:
    """Hash actual bytes, including custom code, tokenizer and preprocessing config.

    A deployment binds this fingerprint before loading, checks it again after
    loading, and retains it for the process lifetime. No inference occurs here.
    """
    root = root.resolve(strict=True)
    paths = [item for item in root.rglob("*") if item.is_file() and ".git" not in item.relative_to(root).parts]
    weights = [item for item in paths if item.suffix in {".safetensors", ".bin"}]
    processor = [item for item in paths if item.suffix in {".json", ".txt", ".py", ".model", ".tiktoken"}]
    if not weights or not (root / "config.json").is_file() or not processor:
        raise ValueError("embedding_assets_incomplete")
    return {"revision": _digest([*weights, root / "config.json"], root, {}),
        "processor_hash": _digest(processor, root, loader_identity)}
