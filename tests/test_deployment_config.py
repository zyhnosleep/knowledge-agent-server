from pathlib import Path


def test_server_environment_template_matches_internal_pilot_architecture() -> None:
    template = Path(".env.server.example").read_text(encoding="utf-8")

    assert "DATABASE_URL=postgresql+psycopg://" in template
    assert "VECTOR_STORE_BACKEND=pgvector" in template
    assert "OLLAMA_EMBEDDING_DIMENSIONS=4096" in template
    assert "AUTH_ENABLED=true" in template
    assert "AUTH_SESSION_SECRET=" in template
    assert "AUTH_COOKIE_SECURE=true" in template
    assert "FEISHU_APP_ID=" in template
    assert "FEISHU_APP_SECRET=" in template
    assert "FEISHU_REDIRECT_URI=https://" in template
    assert "EXTERNAL_API_ENABLED=false" in template


def test_user_service_and_named_tunnel_templates_are_restartable() -> None:
    api_service = Path("deploy/systemd/llm-wiki-pilot.service").read_text(
        encoding="utf-8"
    )
    tunnel_service = Path("deploy/systemd/llm-wiki-tunnel.service").read_text(
        encoding="utf-8"
    )
    tunnel_config = Path("deploy/cloudflared/config.yml.example").read_text(
        encoding="utf-8"
    )

    assert "EnvironmentFile=%h/llm_wiki_internal_pilot/runtime/auth.env" in api_service
    assert "--host 127.0.0.1 --port 8001" in api_service
    assert "Restart=on-failure" in api_service
    assert "cloudflared tunnel --config" in tunnel_service
    assert "Restart=on-failure" in tunnel_service
    assert "hostname: research.example.com" in tunnel_config
    assert "service: http://127.0.0.1:8001" in tunnel_config
    assert "service: http_status:404" in tunnel_config


def test_dual_ollama_services_pin_gpus_and_performance_flags() -> None:
    fast = Path("deploy/systemd/llm-wiki-ollama-fast.service").read_text(encoding="utf-8")
    deep = Path("deploy/systemd/llm-wiki-ollama-deep.service").read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=0" in fast
    assert "OLLAMA_HOST=127.0.0.1:11435" in fast
    assert "OLLAMA_FLASH_ATTENTION=1" in fast
    assert "OLLAMA_KV_CACHE_TYPE=q8_0" in fast
    assert "OLLAMA_KEEP_ALIVE=-1" in fast
    assert "OLLAMA_CONTEXT_LENGTH=16384" in fast
    assert "CUDA_VISIBLE_DEVICES=1" in deep
    assert "OLLAMA_HOST=127.0.0.1:11436" in deep
    assert "OLLAMA_FLASH_ATTENTION=1" in deep
    assert "OLLAMA_KV_CACHE_TYPE=q8_0" in deep
    assert "OLLAMA_KEEP_ALIVE=-1" in deep
    assert "OLLAMA_CONTEXT_LENGTH=32768" in deep
