"""Current deployment behavior, not historical Systemd/source string checks."""
import json
from pathlib import Path
import subprocess
import sys

from dotenv import dotenv_values
import pytest

from app.core.config import Settings
from app.services.runtime_contract import EmbeddingIdentity, RuntimeContractError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile(monkeypatch):
    for field in Settings.model_fields.values():
        if isinstance(field.alias, str):
            monkeypatch.delenv(field.alias, raising=False)
    return Settings(_env_file=ROOT / "deploy/standalone.env.example")


def test_production_profile_uses_unified_multimodal_space(profile):
    assert profile.vector_store_strict and profile.vector_store_enabled
    assert profile.vector_store_backend == "pgvector"
    assert profile.database_url.startswith("postgresql+psycopg:")
    assert profile.active_embedding_dimensions == 2048
    assert profile.active_embedding_model == "qwen3-vl-embedding:2b"
    assert profile.ollama_generation_model == profile.ollama_vision_model == "qwen3-vl:4b"
    assert profile.ollama_generation_base_url == profile.ollama_embedding_base_url == "http://127.0.0.1:18080"


def test_template_cannot_claim_historical_index_identity(profile):
    with pytest.raises(RuntimeContractError, match="embedding_identity_unverified"):
        EmbeddingIdentity.from_settings(profile)
    assert profile.agent_adaptive_enabled is False


def test_operator_auth_choice_and_secret_are_not_shared(profile):
    values = dict(dotenv_values(ROOT / "deploy/standalone.env.example"))
    authenticated = Settings(_env_file=None, **{**values, "AUTH_ENABLED": "true", "AUTH_SESSION_SECRET": "isolated-test-secret"})
    assert authenticated.auth_enabled is True
    assert authenticated.auth_session_secret == "isolated-test-secret"
    assert profile.auth_enabled is False and profile.auth_session_secret is None
    assert profile.app_host == authenticated.app_host == "127.0.0.1"


def test_without_redis_ingest_dispatch_really_runs_synchronously(profile, monkeypatch):
    import app.services.queue as queue
    monkeypatch.setattr(queue, "settings", profile)
    completed = []
    queue.JobDispatcher().enqueue_or_run(lambda value: completed.append(value), "parsed")
    assert completed == ["parsed"]


def test_cli_invalid_environment_is_nonzero_and_redacted(tmp_path):
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql+psycopg://user:private-password@localhost/db\n")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/project_ctl.py"), "preflight", "--root", str(tmp_path)],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "blocked"
    assert "private-password" not in result.stdout + result.stderr
