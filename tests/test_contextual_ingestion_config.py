from app.core.config import Settings


def test_contextual_ingestion_defaults_are_strict() -> None:
    settings = Settings(_env_file=None)

    assert settings.canonical_artifacts_dir.name == "parsed"
    assert settings.semantic_splitting_model == "qwen3-embedding:4b"
    assert settings.contextualization_model == "qwen3.5:9b"
    assert settings.contextualization_batch_size == 12
    assert settings.contextualization_max_retries == 2
    assert settings.contextualization_max_sentences == 2
    assert settings.parent_token_limits == (500, 1200, 1800)
    assert settings.child_token_limits == (180, 400, 600)
    assert settings.child_overlap_tokens == 50
    assert settings.mineru_enabled is True
    assert settings.maintenance_mode_enabled is False
