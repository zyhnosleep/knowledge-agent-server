from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import _maintenance_blocks, app, settings


def test_maintenance_policy_blocks_business_mutations_but_keeps_reads_and_auth() -> None:
    assert _maintenance_blocks("POST", "/api/ingest/upload") is True
    assert _maintenance_blocks("POST", "/api/query") is True
    assert _maintenance_blocks("POST", "/api/agent/query") is True
    assert _maintenance_blocks("POST", "/api/agent/query/stream") is True
    assert _maintenance_blocks("DELETE", "/api/projects/demo") is True

    assert _maintenance_blocks("GET", "/api/health") is False
    assert _maintenance_blocks("GET", "/api/documents/d1/parse") is False
    assert _maintenance_blocks("GET", "/api/pipeline/runs/run-1") is False
    assert _maintenance_blocks("POST", "/api/auth/login") is False
    assert _maintenance_blocks("GET", "/") is False


def test_maintenance_middleware_returns_503_before_auth_or_route_execution(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "maintenance_mode_enabled", True)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/query",
        json={"project_slug": "demo", "question": "blocked"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Service is in maintenance mode."}
    assert response.headers["retry-after"] == "60"


def test_disabled_maintenance_mode_does_not_intercept_business_routes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "maintenance_mode_enabled", False)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/query",
        json={"project_slug": "demo", "question": "not intercepted"},
    )

    assert response.status_code != 503
