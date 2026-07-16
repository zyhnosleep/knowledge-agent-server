from __future__ import annotations

import hashlib
import hmac
import secrets
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timedelta
from typing import Any

import httpx
from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.records import AuthSession, User


def _urlsafe_b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).decode().rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = (-len(value)) % 4
    if padding:
        value += "=" * padding
    return urlsafe_b64decode(value)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_token() -> str:
    return _urlsafe_b64encode(secrets.token_bytes(32))


def _sign(data: str, secret: str, max_age_seconds: int) -> str:
    expires = int(datetime.utcnow().timestamp()) + max_age_seconds
    payload = f"{expires}:{data}"
    payload_b64 = _urlsafe_b64encode(payload.encode())
    signature = hmac.new(secret.encode("utf-8"), payload_b64.encode(), hashlib.sha256).digest()
    signature_b64 = _urlsafe_b64encode(signature)
    return f"{payload_b64}.{signature_b64}"


def _unsign(signed_value: str, secret: str) -> str:
    try:
        payload_b64, signature_b64 = signed_value.split(".")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid signed value.") from exc

    expected_signature = _urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), payload_b64.encode(), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(signature_b64, expected_signature):
        raise HTTPException(status_code=400, detail="Invalid signed value signature.")

    payload = _urlsafe_b64decode(payload_b64).decode("utf-8")
    try:
        expires_str, data = payload.split(":", 1)
        expires = int(expires_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Malformed signed payload.") from exc

    if datetime.utcnow().timestamp() > expires:
        raise HTTPException(status_code=400, detail="Signed value expired.")

    return data


def build_pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _urlsafe_b64encode(digest)


def create_oauth_state(secret: str, max_age_seconds: int) -> tuple[str, str, str]:
    state = generate_token()
    verifier = generate_token()
    signed = _sign(f"{state}:{verifier}", secret, max_age_seconds)
    return state, signed, verifier


def verify_oauth_state(signed_state: str, query_state: str, secret: str) -> str:
    data = _unsign(signed_state, secret)
    try:
        stored_state, verifier = data.split(":", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid OAuth state payload.") from exc
    if not hmac.compare_digest(stored_state, query_state):
        raise HTTPException(status_code=400, detail="OAuth state mismatch.")
    return verifier


def exchange_code_for_token(code: str, verifier: str, settings: Settings) -> str:
    if not settings.feishu_app_id or not settings.feishu_app_secret or not settings.feishu_redirect_uri:
        raise HTTPException(status_code=503, detail="Feishu OAuth is not configured.")

    response = httpx.post(
        settings.feishu_token_url,
        json={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": settings.feishu_app_id,
            "client_secret": settings.feishu_app_secret,
            "redirect_uri": settings.feishu_redirect_uri,
            "code_verifier": verifier,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Feishu token exchange failed: {payload.get('msg')}"
        )

    access_token = payload.get("access_token")
    if not access_token and isinstance(payload.get("data"), dict):
        access_token = payload["data"].get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=400,
            detail="Feishu token response missing access_token.",
        )
    return access_token


def fetch_feishu_user_info(access_token: str, settings: Settings) -> dict[str, Any]:
    response = httpx.get(
        settings.feishu_user_info_url,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Feishu user info failed: {payload.get('msg')}"
        )
    return payload["data"]


def upsert_user_from_feishu(db: Session, user_info: dict[str, Any]) -> User:
    open_id = user_info.get("open_id")
    if not open_id:
        raise HTTPException(status_code=400, detail="Feishu user info missing open_id.")

    user = db.scalar(select(User).where(User.feishu_open_id == open_id))
    now = datetime.utcnow()
    if user is None:
        user = User(
            feishu_open_id=open_id,
            feishu_union_id=user_info.get("union_id"),
            tenant_key=user_info.get("tenant_key"),
            display_name=user_info.get("name") or "",
            avatar_url=user_info.get("avatar_url"),
            is_active=True,
            last_login_at=now,
        )
        db.add(user)
    else:
        user.feishu_union_id = user_info.get("union_id") or user.feishu_union_id
        user.tenant_key = user_info.get("tenant_key") or user.tenant_key
        user.display_name = user_info.get("name") or user.display_name
        user.avatar_url = user_info.get("avatar_url") or user.avatar_url
        user.is_active = True
        user.last_login_at = now
    db.commit()
    db.refresh(user)
    return user


def create_session(db: Session, user: User, max_age_seconds: int) -> tuple[str, str, AuthSession]:
    token = generate_token()
    csrf_token = generate_token()
    now = datetime.utcnow()
    session = AuthSession(
        user_id=user.id,
        token_hash=hash_token(token),
        csrf_hash=hash_token(csrf_token),
        expires_at=now + timedelta(seconds=max_age_seconds),
        last_seen_at=now,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return token, csrf_token, session


def get_session_by_token(db: Session, token: str | None) -> AuthSession | None:
    if not token:
        return None
    return db.scalar(
        select(AuthSession).where(
            AuthSession.token_hash == hash_token(token),
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > datetime.utcnow(),
        )
    )


def get_current_user(request: Request, db: Session) -> User | None:
    settings = get_settings()
    token = request.cookies.get(settings.auth_session_cookie_name)
    session = get_session_by_token(db, token)
    if session is None or session.user is None or not session.user.is_active:
        return None
    return session.user


def get_current_user_or_401(request: Request, db: Session) -> User:
    user = get_current_user(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def verify_csrf(request: Request, db: Session) -> bool:
    settings = get_settings()
    session_token = request.cookies.get(settings.auth_session_cookie_name)
    session = get_session_by_token(db, session_token)
    if session is None:
        return False

    csrf_cookie = request.cookies.get(settings.auth_csrf_cookie_name)
    header_token = request.headers.get("X-CSRF-Token")
    if not csrf_cookie or not header_token:
        return False

    if not hmac.compare_digest(csrf_cookie, header_token):
        return False

    return hmac.compare_digest(session.csrf_hash, hash_token(header_token))


def revoke_session(request: Request, db: Session) -> bool:
    settings = get_settings()
    token = request.cookies.get(settings.auth_session_cookie_name)
    session = get_session_by_token(db, token)
    if session is None:
        return False
    session.revoked_at = datetime.utcnow()
    db.commit()
    return True


def auth_is_configured(settings: Settings) -> bool:
    return bool(
        settings.auth_enabled
        and settings.auth_session_secret
        and settings.feishu_app_id
        and settings.feishu_app_secret
        and settings.feishu_redirect_uri
    )
