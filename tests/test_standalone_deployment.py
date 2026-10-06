"""Safe model paths at the real configuration boundary, without a GPU import."""
from pathlib import Path

import pytest

from app.services.local_model_config import read_local_model_paths


@pytest.fixture
def model_env(tmp_path):
    for directory in ("models/encoder", "models/chat", "runtime/data/parsed", "runtime/data/cache"):
        (tmp_path / directory).mkdir(parents=True)
    return tmp_path, {
        "LOCAL_MODEL_IMAGE_EMBED": str(tmp_path / "models/encoder"),
        "LOCAL_MODEL_CHAT": str(tmp_path / "models/chat"),
    }


def test_no_default_model_path_is_guessed(model_env):
    root, env = model_env
    env.pop("LOCAL_MODEL_CHAT")
    with pytest.raises(ValueError, match="model_assets_missing"):
        read_local_model_paths(env, root)


def test_default_image_access_is_only_parsed_and_cache(model_env):
    root, env = model_env
    paths = read_local_model_paths(env, root)
    assert paths["image_roots"] == (root / "runtime/data/parsed", root / "runtime/data/cache")
    assert paths["text_embed"] is None
    assert paths["chat"] == root / "models/chat"


def test_explicit_broad_root_is_rejected(model_env):
    root, env = model_env
    env["LOCAL_MODEL_IMAGE_ROOTS"] = str(root)
    with pytest.raises(ValueError, match="image_root_invalid"):
        read_local_model_paths(env, root)


def test_text_fallback_needs_explicit_assets(model_env):
    root, env = model_env
    env["LOCAL_MODEL_EMBED_BACKEND"] = "text"
    with pytest.raises(ValueError, match="model_assets_missing"):
        read_local_model_paths(env, root)


def test_non_loopback_model_listener_is_rejected(model_env):
    root, env = model_env
    env["LOCAL_MODEL_HOST"] = "0.0.0.0"
    with pytest.raises(ValueError, match="non_loopback_listener"):
        read_local_model_paths(env, root)


def test_allowed_directory_symlink_cannot_escape(model_env):
    root, env = model_env
    outside = root.parent / (root.name + "-outside")
    outside.mkdir()
    try:
        (root / "runtime/data/link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Linux acceptance covers symlink creation")
    env["CACHE_DIR"] = str(root / "runtime/data/link")
    with pytest.raises(ValueError, match="image_root_invalid"):
        read_local_model_paths(env, root)
