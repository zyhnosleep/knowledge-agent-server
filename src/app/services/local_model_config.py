"""Dependency-free, fail-closed asset and pixel roots for the model environment."""
from __future__ import annotations

from collections.abc import Mapping
import ipaddress
from pathlib import Path


def read_local_model_paths(env: Mapping[str, str], root: Path) -> dict:
    root = root.resolve(strict=True)
    try:
        if not ipaddress.ip_address(env.get("LOCAL_MODEL_HOST", "127.0.0.1")).is_loopback:
            raise ValueError("non_loopback_listener")
    except ValueError:
        raise ValueError("non_loopback_listener") from None
    backend = env.get("LOCAL_MODEL_EMBED_BACKEND", "vl").lower().strip()
    if backend not in ("vl", "text"):
        raise ValueError("embedding_backend_invalid")

    def asset(key, required=True):
        raw = env.get(key, "").strip()
        if not raw and not required:
            return None
        path = Path(raw)
        if not raw or not path.is_absolute() or not path.is_dir():
            raise ValueError("model_assets_missing")
        return path.resolve(strict=True)

    allowed = []
    for key, default in (("CANONICAL_ARTIFACTS_DIR", "runtime/data/parsed"), ("CACHE_DIR", "runtime/data/cache")):
        try:
            path = (root / env.get(key, default)).resolve(strict=True)
        except OSError:
            raise ValueError("image_root_invalid") from None
        # Do not trust a symlinked runtime/data parent as a new allowed root.
        if not path.is_dir() or not path.is_relative_to(root / "runtime/data") or path == root / "runtime/data":
            raise ValueError("image_root_invalid")
        allowed.append(path)
    raw_roots = env.get("LOCAL_MODEL_IMAGE_ROOTS", "").strip()
    try:
        roots = tuple(Path(p.strip()).resolve(strict=True) for p in raw_roots.split(",") if p.strip()) if raw_roots else tuple(allowed)
    except OSError:
        raise ValueError("image_root_invalid") from None
    if not roots or not set(roots) <= set(allowed):
        raise ValueError("image_root_invalid")
    return {"text_embed": asset("LOCAL_MODEL_TEXT_EMBED", backend == "text"),
        "image_embed": asset("LOCAL_MODEL_IMAGE_EMBED"), "chat": asset("LOCAL_MODEL_CHAT"),
        "adapter": asset("LOCAL_MODEL_CHAT_ADAPTER", False), "image_roots": roots}
