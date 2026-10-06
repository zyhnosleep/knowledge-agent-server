from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


ROOT_DIR = Path(__file__).resolve().parents[1]


def _clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in Settings.model_fields.values():
        if isinstance(field.alias, str):
            monkeypatch.delenv(field.alias, raising=False)


def test_contextual_ingestion_defaults_are_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_environment(monkeypatch)
    settings = Settings(_env_file=None)

    assert settings.canonical_artifacts_dir.name == "parsed"
    assert settings.semantic_splitting_model == "qwen3-embedding:4b"
    assert settings.semantic_tokenizer_name == "Qwen/Qwen3-Embedding-4B"
    assert (
        getattr(settings, "semantic_tokenizer_revision", None)
        == "5cf2132abc99cad020ac570b19d031efec650f2b"
    )
    assert getattr(settings, "semantic_tokenizer_local_path", None) is None
    assert settings.contextualization_model == "qwen3.5:9b"
    assert settings.contextualization_batch_size == 12
    assert settings.contextualization_max_retries == 2
    assert settings.contextualization_max_sentences == 2
    assert settings.parent_token_limits == (500, 1200, 1800)
    assert settings.child_token_limits == (180, 400, 600)
    assert settings.child_overlap_tokens == 50
    assert settings.mineru_enabled is True
    assert settings.maintenance_mode_enabled is False


def test_ingestion_snapshot_versions_all_fidelity_affecting_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.ingestion_identity import build_ingestion_config_snapshot

    _clear_settings_environment(monkeypatch)
    settings = Settings(_env_file=None)

    snapshot = build_ingestion_config_snapshot(
        settings,
        tokenizer_identity={
            "name": "Qwen/Qwen3-Embedding-4B",
            "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
            "content_sha256": "a" * 64,
        },
    )

    assert snapshot["algorithm_revisions"] == {
        "parser": "canonical-parser-v6",
        "pdf_recovery": "pdf-recovery-v2",
        "structured_splitting": "structured-splitting-v3",
        "source_fidelity_algorithm": "source-fidelity-v1",
        "source_fidelity_schema": "source-fidelity-schema-v1",
    }


def test_ingestion_snapshot_uses_provider_neutral_embedding_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.ingestion_identity import build_ingestion_config_snapshot

    _clear_settings_environment(monkeypatch)
    tokenizer_identity = {
        "name": "Qwen/Qwen3-Embedding-4B",
        "revision": "r1",
        "content_sha256": "a" * 64,
    }
    local = build_ingestion_config_snapshot(
        Settings(_env_file=None), tokenizer_identity=tokenizer_identity
    )
    remote = build_ingestion_config_snapshot(
        Settings(
            _env_file=None,
            EMBEDDING_PROVIDER="openai-compatible",
            EMBEDDING_API_MODEL="Qwen/Qwen3-Embedding-4B",
            EMBEDDING_DIMENSIONS=2560,
        ),
        tokenizer_identity=tokenizer_identity,
    )

    assert local["embedding"] == {
        "provider": "ollama",
        "model": "qwen3-embedding:4b",
        "dimensions": 2560,
    }
    assert remote["embedding"] == {
        "provider": "openai-compatible",
        "model": "Qwen/Qwen3-Embedding-4B",
        "dimensions": 2560,
    }


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        (
            {"SEMANTIC_PARENT_MIN_TOKENS": 1201},
            "semantic parent token limits must satisfy min <= target <= max",
        ),
        (
            {"SEMANTIC_PARENT_TARGET_TOKENS": 1801},
            "semantic parent token limits must satisfy min <= target <= max",
        ),
        (
            {"SEMANTIC_CHILD_MIN_TOKENS": 401},
            "semantic child token limits must satisfy min <= target <= max",
        ),
        (
            {"SEMANTIC_CHILD_TARGET_TOKENS": 601},
            "semantic child token limits must satisfy min <= target <= max",
        ),
        (
            {"SEMANTIC_CHILD_OVERLAP_TOKENS": 180},
            "semantic child overlap tokens must satisfy 0 <= overlap < min",
        ),
    ],
)
def test_contextual_ingestion_rejects_invalid_token_limit_combinations(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, int],
    expected_message: str,
) -> None:
    _clear_settings_environment(monkeypatch)

    with pytest.raises(ValidationError, match=expected_message):
        Settings(_env_file=None, **overrides)


def test_contextual_ingestion_environment_examples_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_settings_environment(monkeypatch)

    development = Settings(_env_file=ROOT_DIR / ".env.development.example")
    test = Settings(_env_file=ROOT_DIR / ".env.test.example")

    assert development.contextualization_base_url == "http://127.0.0.1:11435"
    assert test.contextualization_base_url == "http://127.0.0.1:11436"
    assert development.contextualization_base_url != test.contextualization_base_url

    for settings in (development, test):
        assert settings.canonical_artifacts_dir == Path("runtime/data/parsed")
        assert settings.canonical_pipeline_version == "canonical-v4"
        assert (
            settings.semantic_tokenizer_revision
            == "5cf2132abc99cad020ac570b19d031efec650f2b"
        )
        assert settings.semantic_tokenizer_local_path == Path(
            "runtime/models/Qwen3-Embedding-4B-tokenizer"
        )
        assert settings.mineru_enabled is True
        assert settings.maintenance_mode_enabled is False
        assert settings.parent_token_limits == (500, 1200, 1800)
        assert settings.child_token_limits == (180, 400, 600)
        assert settings.child_overlap_tokens == 50
