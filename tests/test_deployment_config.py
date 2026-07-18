from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_development_environment_is_private_and_uses_gpu_zero() -> None:
    env = _read(".env.development.example")
    api = _read("deploy/systemd/knowledge-agent-dev-api.service")
    ollama = _read("deploy/systemd/knowledge-agent-dev-ollama.service")

    assert 'APP_NAME="Knowledge Agent"' in env
    assert "DATABASE_URL=postgresql+psycopg://knowledge_agent_dev" in env
    assert "AUTH_ENABLED=false" in env
    assert "OLLAMA_GENERATION_BASE_URL=http://127.0.0.1:11435" in env
    assert "OLLAMA_GENERATION_MODEL=qwen3.5:9b" in env
    assert "OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b" in env
    assert "OLLAMA_EMBEDDING_DIMENSIONS=2560" in env
    assert "WorkingDirectory=%h/knowledge-agent-dev" in api
    assert "--host 127.0.0.1 --port 8002" in api
    assert "CUDA_VISIBLE_DEVICES=0" in ollama
    assert "OLLAMA_HOST=127.0.0.1:11435" in ollama


def test_test_environment_preserves_auth_and_uses_gpu_one() -> None:
    env = _read(".env.test.example")
    api = _read("deploy/systemd/knowledge-agent-test-api.service")
    ollama = _read("deploy/systemd/knowledge-agent-test-ollama.service")
    tunnel = _read("deploy/systemd/knowledge-agent-test-tunnel.service")

    assert 'APP_NAME="Knowledge Agent"' in env
    assert "DATABASE_URL=postgresql+psycopg://knowledge_agent_test" in env
    assert "AUTH_ENABLED=true" in env
    assert "OLLAMA_GENERATION_BASE_URL=http://127.0.0.1:11436" in env
    assert "WorkingDirectory=%h/knowledge-agent-test" in api
    assert "--host 127.0.0.1 --port 8001" in api
    assert "CUDA_VISIBLE_DEVICES=1" in ollama
    assert "OLLAMA_HOST=127.0.0.1:11436" in ollama
    assert "After=network-online.target knowledge-agent-test-api.service" in tunnel
    assert "%h/knowledge-agent-test/runtime/cloudflared/config.yml" in tunnel


def test_both_ollama_services_share_performance_and_idle_settings() -> None:
    for path in (
        "deploy/systemd/knowledge-agent-dev-ollama.service",
        "deploy/systemd/knowledge-agent-test-ollama.service",
    ):
        service = _read(path)
        assert "OLLAMA_MODELS=%h/knowledge-agent-models" in service
        assert "OLLAMA_FLASH_ATTENTION=1" in service
        assert "OLLAMA_KV_CACHE_TYPE=q8_0" in service
        assert "OLLAMA_KEEP_ALIVE=5m" in service
        assert "OLLAMA_NUM_PARALLEL=1" in service
        assert "OLLAMA_CONTEXT_LENGTH=32768" in service


def test_redis_is_local_and_each_environment_has_an_isolated_worker() -> None:
    redis = _read("deploy/systemd/knowledge-agent-redis.service")
    dev_env = _read(".env.development.example")
    test_env = _read(".env.test.example")
    dev_worker = _read("deploy/systemd/knowledge-agent-dev-worker.service")
    test_worker = _read("deploy/systemd/knowledge-agent-test-worker.service")

    assert "127.0.0.1:6379:6379" in redis
    assert "redis:7-alpine" in redis
    assert "--appendonly yes" in redis
    assert "REDIS_URL=redis://127.0.0.1:6379/1" in dev_env
    assert "REDIS_URL=redis://127.0.0.1:6379/2" in test_env
    assert "WorkingDirectory=%h/knowledge-agent-dev" in dev_worker
    assert "WorkingDirectory=%h/knowledge-agent-test" in test_worker
    assert "app.workers.runner" in dev_worker
    assert "app.workers.runner" in test_worker
    assert "EnvironmentFile=%h/knowledge-agent-dev/runtime/app.env" in dev_worker
    assert "EnvironmentFile=%h/knowledge-agent-test/runtime/app.env" in test_worker


def test_active_product_files_use_knowledge_agent_brand() -> None:
    files = [
        Path("README.md"),
        Path("src/app/__init__.py"),
        Path("src/app/core/config.py"),
        *Path("deploy").rglob("*"),
    ]
    texts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in files
        if path.is_file()
    ).lower()
    for separator in (" ", "-", "_"):
        assert "llm" + separator + "wi" + "ki" not in texts
    assert "knowledge agent" in texts
