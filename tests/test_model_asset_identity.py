"""Catch false identity based on a directory name rather than loaded assets."""
from pathlib import Path

import pytest


def assets(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"weights-a")
    (tmp_path / "config.json").write_text('{"hidden_size":2048}', encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text('{"vocab":[]}', encoding="utf-8")
    return tmp_path


def test_revision_tracks_actual_weight_bytes(tmp_path):
    from app.services.model_asset_identity import fingerprint_embedding_assets
    root = assets(tmp_path)
    before = fingerprint_embedding_assets(root, loader_identity={"library": "1"})
    (root / "model.safetensors").write_bytes(b"weights-b")
    after = fingerprint_embedding_assets(root, loader_identity={"library": "1"})
    assert before["revision"] != after["revision"]
    assert before["processor_hash"] == after["processor_hash"]


def test_processor_and_loader_changes_are_part_of_embedding_space(tmp_path):
    from app.services.model_asset_identity import fingerprint_embedding_assets
    root = assets(tmp_path)
    before = fingerprint_embedding_assets(root, loader_identity={"library": "1"})
    upgraded = fingerprint_embedding_assets(root, loader_identity={"library": "2"})
    (root / "tokenizer.json").write_text('{"vocab":["changed"]}', encoding="utf-8")
    after = fingerprint_embedding_assets(root, loader_identity={"library": "1"})
    assert upgraded["processor_hash"] != before["processor_hash"]
    assert after["processor_hash"] != before["processor_hash"]


def test_missing_weights_cannot_receive_verified_identity(tmp_path):
    from app.services.model_asset_identity import fingerprint_embedding_assets
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="embedding_assets_incomplete"):
        fingerprint_embedding_assets(tmp_path, loader_identity={"library": "1"})
