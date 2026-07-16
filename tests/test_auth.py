from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.session import Base, get_db
from app.main import app
from app.models.records import AgentTraceRun, AuthSession, ConversationSession, User


def _db_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


@pytest.fixture
def db():
    db = _db_session()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def override_get_db(db):
    def _get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = _get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def enabled_settings(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_SESSION_SECRET", "test-secret-32-bytes-long-for-hmac")
    monkeypatch.setenv("FEISHU_APP_ID", "test_app_id")
    monkeypatch.setenv("FEISHU_APP_SECRET", "test_app_secret")
    monkeypatch.setenv("FEISHU_REDIRECT_URI", "http://localhost:8000/api/auth/callback")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)


def _extract_state_cookie(response) -> str | None:
    for header, value in response.headers.raw:
        if header.lower() == b"set-cookie":
            cookie = value.decode()
            if "nri_oauth_state=" in cookie:
                start = cookie.index("nri_oauth_state=") + len("nri_oauth_state=")
                end = cookie.find(";", start)
                return cookie[start:end] if end != -1 else cookie[start:]
    return None


def _extract_cookie_value(response, name: str) -> str | None:
    for header, value in response.headers.raw:
        if header.lower() == b"set-cookie":
            cookie = value.decode()
            prefix = f"{name}="
            if prefix in cookie:
                start = cookie.index(prefix) + len(prefix)
                end = cookie.find(";", start)
                raw = cookie[start:end] if end != -1 else cookie[start:]
                return raw.strip('"')
    return None


def test_login_returns_503_when_auth_disabled(client, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()
    response = client.get("/api/auth/login", follow_redirects=False)
    assert response.status_code == 503


def test_auth_status_reports_disabled_and_configured_states(client, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()
    disabled = client.get("/api/auth/status")

    assert disabled.status_code == 200
    assert disabled.json() == {"enabled": False, "configured": False}

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_SESSION_SECRET", "test-secret-32-bytes-long-for-hmac")
    monkeypatch.setenv("FEISHU_APP_ID", "test_app_id")
    monkeypatch.setenv("FEISHU_APP_SECRET", "test_app_secret")
    monkeypatch.setenv("FEISHU_REDIRECT_URI", "https://example.test/api/auth/callback")
    get_settings.cache_clear()
    configured = client.get("/api/auth/status")

    assert configured.status_code == 200
    assert configured.json() == {"enabled": True, "configured": True}


def test_business_api_requires_session_when_auth_enabled(enabled_settings, client):
    response = client.get("/api/projects")

    assert response.status_code == 401


def test_business_api_remains_available_when_auth_disabled(client, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()

    response = client.get("/api/projects")

    assert response.status_code == 200


def test_agent_sessions_and_traces_are_scoped_to_authenticated_user(
    enabled_settings, client, db
):
    now = datetime.utcnow()
    first_user = User(id="u1", feishu_open_id="ou_u1", display_name="User One")
    second_user = User(id="u2", feishu_open_id="ou_u2", display_name="User Two")
    db.add_all([first_user, second_user])
    db.add_all(
        [
            AuthSession(
                user_id="u1",
                token_hash=hashlib.sha256(b"token-u1").hexdigest(),
                csrf_hash=hashlib.sha256(b"csrf-u1").hexdigest(),
                expires_at=now + timedelta(hours=1),
            ),
            AuthSession(
                user_id="u2",
                token_hash=hashlib.sha256(b"token-u2").hexdigest(),
                csrf_hash=hashlib.sha256(b"csrf-u2").hexdigest(),
                expires_at=now + timedelta(hours=1),
            ),
        ]
    )
    db.add_all(
        [
            ConversationSession(
                id="session-u1",
                owner_user_id="u1",
                project_slug="demo",
                expires_at=now + timedelta(days=1),
            ),
            ConversationSession(
                id="session-u2",
                owner_user_id="u2",
                project_slug="demo",
                expires_at=now + timedelta(days=1),
            ),
            AgentTraceRun(
                id="trace-u1",
                owner_user_id="u1",
                request_id="request-u1",
                session_id="session-u1",
                project_slug="demo",
                query="one",
                final_answer="one",
                status="completed",
            ),
            AgentTraceRun(
                id="trace-u2",
                owner_user_id="u2",
                request_id="request-u2",
                session_id="session-u2",
                project_slug="demo",
                query="two",
                final_answer="two",
                status="completed",
            ),
        ]
    )
    db.commit()
    client.cookies.set("nri_session", "token-u1")

    sessions = client.get("/api/agent/sessions?project_slug=demo")
    traces = client.get("/api/agent/traces?project_slug=demo")
    foreign_trace = client.get("/api/agent/traces/trace-u2")

    assert sessions.status_code == 200
    assert [item["id"] for item in sessions.json()] == ["session-u1"]
    assert traces.status_code == 200
    assert [item["trace_id"] for item in traces.json()["traces"]] == ["trace-u1"]
    assert foreign_trace.status_code == 404


def test_login_returns_503_when_credentials_missing(client, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("FEISHU_APP_ID", "")
    monkeypatch.setenv("FEISHU_APP_SECRET", "")
    get_settings.cache_clear()
    response = client.get("/api/auth/login", follow_redirects=False)
    assert response.status_code == 503


def test_login_returns_503_when_session_secret_missing(client, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("FEISHU_APP_ID", "test_app_id")
    monkeypatch.setenv("FEISHU_APP_SECRET", "test_app_secret")
    monkeypatch.setenv("FEISHU_REDIRECT_URI", "http://localhost:8000/api/auth/callback")
    monkeypatch.delenv("AUTH_SESSION_SECRET", raising=False)
    get_settings.cache_clear()
    response = client.get("/api/auth/login", follow_redirects=False)
    assert response.status_code == 503


def test_login_redirects_to_feishu_with_state_cookie(enabled_settings, client):
    response = client.get("/api/auth/login", follow_redirects=False)
    assert response.status_code == 307
    location = response.headers["location"]
    assert location.startswith("https://accounts.feishu.cn/open-apis/authen/v1/authorize")
    assert "response_type=code" in location
    assert "app_id=test_app_id" in location
    assert "code_challenge_method=S256" in location
    assert "code_challenge=" in location
    assert "state=" in location

    state_cookie = _extract_state_cookie(response)
    assert state_cookie is not None
    assert state_cookie != ""


def test_callback_rejects_missing_state_cookie(enabled_settings, client):
    response = client.get("/api/auth/callback?state=abc&code=def", follow_redirects=False)
    assert response.status_code == 400


def test_callback_rejects_mismatched_state(enabled_settings, client):
    login_resp = client.get("/api/auth/login", follow_redirects=False)
    state_cookie = _extract_state_cookie(login_resp)
    client.cookies.set("nri_oauth_state", state_cookie)
    response = client.get("/api/auth/callback?state=wrong-state&code=def", follow_redirects=False)
    assert response.status_code == 400


def _mock_feishu_exchange(access_token: str = "fake_access_token"):
    return patch(
        "app.services.auth.httpx.post",
        return_value=type(
            "Resp",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "code": 0,
                    "msg": "ok",
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": 7200,
                },
                "status_code": 200,
            },
        )(),
    )


def _mock_feishu_user_info(user_info=None):
    info = user_info or {
        "code": 0,
        "data": {
            "open_id": "ou_123",
            "union_id": "on_456",
            "tenant_key": "tenant_abc",
            "name": "Test User",
            "avatar_url": "https://example.com/avatar.png",
        },
    }
    return patch(
        "app.services.auth.httpx.get",
        return_value=type(
            "Resp",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: info,
                "status_code": 200,
            },
        )(),
    )


def test_callback_creates_user_and_session(enabled_settings, client):
    login_resp = client.get("/api/auth/login", follow_redirects=False)
    state_cookie = _extract_state_cookie(login_resp)
    client.cookies.set("nri_oauth_state", state_cookie)

    query_state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    with _mock_feishu_exchange(), _mock_feishu_user_info():
        response = client.get(
            f"/api/auth/callback?state={query_state}&code=authcode",
            follow_redirects=False,
        )

    assert response.status_code == 307
    assert response.headers["location"] == "/"

    session_cookie = _extract_cookie_value(response, "nri_session")
    csrf_cookie = _extract_cookie_value(response, "nri_csrf")
    assert session_cookie is not None
    assert csrf_cookie is not None

    state_cleared = _extract_cookie_value(response, "nri_oauth_state")
    assert state_cleared == ""


def test_callback_uses_v2_token_endpoint_and_request_fields(enabled_settings, client):
    login_resp = client.get("/api/auth/login", follow_redirects=False)
    state_cookie = _extract_state_cookie(login_resp)
    client.cookies.set("nri_oauth_state", state_cookie)
    query_state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    captured = {}

    def fake_post(url, *, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return type(
            "Resp",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "code": 0,
                    "access_token": "v2_token",
                    "token_type": "Bearer",
                    "expires_in": 7200,
                },
            },
        )()

    with patch("app.services.auth.httpx.post", fake_post), _mock_feishu_user_info():
        response = client.get(
            f"/api/auth/callback?state={query_state}&code=authcode",
            follow_redirects=False,
        )

    assert response.status_code == 307
    assert captured["url"] == "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
    assert captured["json"]["grant_type"] == "authorization_code"
    assert captured["json"]["code"] == "authcode"
    assert captured["json"]["client_id"] == "test_app_id"
    assert captured["json"]["client_secret"] == "test_app_secret"
    assert captured["json"]["redirect_uri"] == "http://localhost:8000/api/auth/callback"
    assert captured["json"]["code_verifier"]


def test_callback_supports_legacy_nested_access_token(enabled_settings, client):
    login_resp = client.get("/api/auth/login", follow_redirects=False)
    state_cookie = _extract_state_cookie(login_resp)
    client.cookies.set("nri_oauth_state", state_cookie)
    query_state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    with patch(
        "app.services.auth.httpx.post",
        return_value=type(
            "Resp",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "code": 0,
                    "data": {"access_token": "legacy_token"},
                },
            },
        )(),
    ), _mock_feishu_user_info():
        response = client.get(
            f"/api/auth/callback?state={query_state}&code=authcode",
            follow_redirects=False,
        )

    assert response.status_code == 307


def test_callback_rejects_missing_access_token(enabled_settings, client):
    login_resp = client.get("/api/auth/login", follow_redirects=False)
    state_cookie = _extract_state_cookie(login_resp)
    client.cookies.set("nri_oauth_state", state_cookie)
    query_state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    with patch(
        "app.services.auth.httpx.post",
        return_value=type(
            "Resp",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"code": 0, "msg": "ok"},
            },
        )(),
    ):
        response = client.get(
            f"/api/auth/callback?state={query_state}&code=authcode",
            follow_redirects=False,
        )

    assert response.status_code == 400


def test_me_returns_401_without_session(enabled_settings, client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_me_returns_user_for_valid_session(enabled_settings, client, db):
    user = User(
        id="u-1",
        feishu_open_id="ou_123",
        feishu_union_id="on_456",
        tenant_key="tenant_abc",
        display_name="Test User",
        avatar_url="https://example.com/avatar.png",
        is_active=True,
    )
    db.add(user)
    session = AuthSession(
        id="s-1",
        user_id="u-1",
        token_hash=hashlib.sha256("session-token".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("csrf-token".encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(session)
    db.commit()

    client.cookies.set("nri_session", "session-token")
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "u-1"
    assert data["feishu_open_id"] == "ou_123"
    assert data["display_name"] == "Test User"


def test_me_does_not_mutate_session_last_seen(enabled_settings, client, db):
    user = User(
        id="u-1-read",
        feishu_open_id="ou_123_read",
        display_name="Read Only",
        is_active=True,
    )
    db.add(user)
    session = AuthSession(
        id="s-1-read",
        user_id="u-1-read",
        token_hash=hashlib.sha256("session-token-read".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("csrf-token-read".encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
        last_seen_at=datetime(2026, 1, 1, 0, 0, 0),
    )
    db.add(session)
    db.commit()

    original_last_seen = session.last_seen_at
    client.cookies.set("nri_session", "session-token-read")
    response = client.get("/api/auth/me")
    assert response.status_code == 200

    db.refresh(session)
    assert session.last_seen_at == original_last_seen


def test_me_returns_401_for_expired_session(enabled_settings, client, db):
    user = User(
        id="u-2",
        feishu_open_id="ou_789",
        display_name="Expired User",
        is_active=True,
    )
    db.add(user)
    session = AuthSession(
        id="s-2",
        user_id="u-2",
        token_hash=hashlib.sha256("session-token".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("csrf-token".encode()).hexdigest(),
        expires_at=datetime.utcnow() - timedelta(minutes=1),
    )
    db.add(session)
    db.commit()

    client.cookies.set("nri_session", "session-token")
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_me_returns_401_for_inactive_user(enabled_settings, client, db):
    user = User(
        id="u-3",
        feishu_open_id="ou_inactive",
        display_name="Inactive User",
        is_active=False,
    )
    db.add(user)
    session = AuthSession(
        id="s-3",
        user_id="u-3",
        token_hash=hashlib.sha256("session-token".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("csrf-token".encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(session)
    db.commit()

    client.cookies.set("nri_session", "session-token")
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_callback_rejects_disallowed_tenant(enabled_settings, client, monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_TENANT", "tenant_allowed")
    get_settings.cache_clear()

    login_resp = client.get("/api/auth/login", follow_redirects=False)
    state_cookie = _extract_state_cookie(login_resp)
    client.cookies.set("nri_oauth_state", state_cookie)
    query_state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    with _mock_feishu_exchange(), _mock_feishu_user_info():
        response = client.get(
            f"/api/auth/callback?state={query_state}&code=authcode",
            follow_redirects=False,
        )
    assert response.status_code == 403


def test_logout_requires_csrf(enabled_settings, client, db):
    user = User(
        id="u-4",
        feishu_open_id="ou_logout",
        display_name="Logout User",
        is_active=True,
    )
    db.add(user)
    session = AuthSession(
        id="s-4",
        user_id="u-4",
        token_hash=hashlib.sha256("session-token".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("csrf-token".encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(session)
    db.commit()

    client.cookies.set("nri_session", "session-token")
    response = client.post("/api/auth/logout")
    assert response.status_code == 403


def test_logout_revokes_session(enabled_settings, client, db):
    user = User(
        id="u-5",
        feishu_open_id="ou_logout_ok",
        display_name="Logout OK User",
        is_active=True,
    )
    db.add(user)
    session = AuthSession(
        id="s-5",
        user_id="u-5",
        token_hash=hashlib.sha256("session-token".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("csrf-token".encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(session)
    db.commit()

    client.cookies.set("nri_session", "session-token")
    client.cookies.set("nri_csrf", "csrf-token")
    response = client.post("/api/auth/logout", headers={"X-CSRF-Token": "csrf-token"})
    assert response.status_code == 200
    assert response.json()["success"] is True

    db.refresh(session)
    assert session.revoked_at is not None

    cleared_session = _extract_cookie_value(response, "nri_session")
    assert cleared_session == ""


def test_logout_rejects_valid_header_without_csrf_cookie(enabled_settings, client, db):
    user = User(
        id="u-6",
        feishu_open_id="ou_no_csrf_cookie",
        display_name="No CSRF Cookie",
        is_active=True,
    )
    db.add(user)
    session = AuthSession(
        id="s-6",
        user_id="u-6",
        token_hash=hashlib.sha256("session-token".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("csrf-token".encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(session)
    db.commit()

    client.cookies.set("nri_session", "session-token")
    response = client.post("/api/auth/logout", headers={"X-CSRF-Token": "csrf-token"})
    assert response.status_code == 403


def test_logout_rejects_mismatched_csrf_cookie_and_header(enabled_settings, client, db):
    user = User(
        id="u-7",
        feishu_open_id="ou_mismatch",
        display_name="CSRF Mismatch",
        is_active=True,
    )
    db.add(user)
    session = AuthSession(
        id="s-7",
        user_id="u-7",
        token_hash=hashlib.sha256("session-token".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("stored-token".encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(session)
    db.commit()

    client.cookies.set("nri_session", "session-token")
    client.cookies.set("nri_csrf", "cookie-token")
    response = client.post("/api/auth/logout", headers={"X-CSRF-Token": "header-token"})
    assert response.status_code == 403


def test_logout_rejects_csrf_cookie_header_match_but_wrong_database_hash(enabled_settings, client, db):
    user = User(
        id="u-8",
        feishu_open_id="ou_db_mismatch",
        display_name="DB Hash Mismatch",
        is_active=True,
    )
    db.add(user)
    session = AuthSession(
        id="s-8",
        user_id="u-8",
        token_hash=hashlib.sha256("session-token".encode()).hexdigest(),
        csrf_hash=hashlib.sha256("stored-token".encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(session)
    db.commit()

    client.cookies.set("nri_session", "session-token")
    client.cookies.set("nri_csrf", "same-token")
    response = client.post("/api/auth/logout", headers={"X-CSRF-Token": "same-token"})
    assert response.status_code == 403


def test_urlsafe_b64decode_padding_is_aligned_only(enabled_settings):
    from app.services.auth import _urlsafe_b64decode, _urlsafe_b64encode

    # Lengths 0, 1, 2, 3, 4 should all round-trip without adding spurious padding.
    for length in (0, 1, 2, 3, 4, 5, 8, 16, 32):
        data = bytes(range(length))
        encoded = _urlsafe_b64encode(data)
        decoded = _urlsafe_b64decode(encoded)
        assert decoded == data


def test_health_route_is_protected_with_other_business_apis(
    enabled_settings, client, db
):
    response = client.get("/api/health")
    assert response.status_code == 401
