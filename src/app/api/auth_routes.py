from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.auth import AuthLogoutResponse, AuthUserRead
from app.services.auth import (
    Settings,
    auth_is_configured,
    build_pkce_challenge,
    create_oauth_state,
    create_session,
    exchange_code_for_token,
    fetch_feishu_user_info,
    get_current_user_or_401,
    revoke_session,
    upsert_user_from_feishu,
    verify_csrf,
    verify_oauth_state,
)

auth_router = APIRouter()


def _settings() -> Settings:
    return get_settings()


@auth_router.get("/status")
def auth_status() -> dict[str, bool]:
    settings = _settings()
    return {
        "enabled": settings.auth_enabled,
        "configured": auth_is_configured(settings),
    }


def _set_cookie(
    response: RedirectResponse,
    name: str,
    value: str,
    max_age: int,
    httponly: bool = True,
) -> None:
    settings = _settings()
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        httponly=httponly,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite.lower(),
        path="/",
    )


def _clear_cookie(response: RedirectResponse, name: str) -> None:
    settings = _settings()
    response.delete_cookie(
        key=name,
        path="/",
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite.lower(),
        httponly=True,
    )


@auth_router.get("/login")
def login(request: Request) -> RedirectResponse:
    settings = _settings()
    if not auth_is_configured(settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not enabled or Feishu credentials are missing.",
        )
    if not settings.auth_session_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not enabled or Feishu credentials are missing.",
        )

    state, signed_state, verifier = create_oauth_state(
        settings.auth_session_secret,
        settings.auth_state_max_age_seconds,
    )
    challenge = build_pkce_challenge(verifier)

    params = {
        "app_id": settings.feishu_app_id,
        "redirect_uri": settings.feishu_redirect_uri,
        "response_type": "code",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{settings.feishu_auth_url}?{urlencode(params)}"

    response = RedirectResponse(auth_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    _set_cookie(
        response,
        settings.auth_state_cookie_name,
        signed_state,
        settings.auth_state_max_age_seconds,
        httponly=True,
    )
    return response


@auth_router.get("/callback")
def callback(request: Request, db: Session = Depends(get_db)) -> RedirectResponse:
    settings = _settings()
    if not auth_is_configured(settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not enabled.",
        )
    if not settings.auth_session_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not enabled.",
        )

    signed_state = request.cookies.get(settings.auth_state_cookie_name)
    if not signed_state:
        raise HTTPException(status_code=400, detail="Missing OAuth state cookie.")

    query_state = request.query_params.get("state")
    if not query_state:
        raise HTTPException(status_code=400, detail="Missing OAuth state parameter.")

    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing OAuth authorization code.")

    verifier = verify_oauth_state(signed_state, query_state, settings.auth_session_secret)
    access_token = exchange_code_for_token(code, verifier, settings)
    user_info = fetch_feishu_user_info(access_token, settings)

    tenant_key = user_info.get("tenant_key")
    if settings.feishu_allowed_tenant and tenant_key != settings.feishu_allowed_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User tenant is not allowed.",
        )

    user = upsert_user_from_feishu(db, user_info)
    session_token, csrf_token, _ = create_session(
        db, user, settings.auth_session_max_age_seconds
    )

    response = RedirectResponse("/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    _clear_cookie(response, settings.auth_state_cookie_name)
    _set_cookie(
        response,
        settings.auth_session_cookie_name,
        session_token,
        settings.auth_session_max_age_seconds,
        httponly=True,
    )
    _set_cookie(
        response,
        settings.auth_csrf_cookie_name,
        csrf_token,
        settings.auth_session_max_age_seconds,
        httponly=False,
    )
    return response


@auth_router.get("/me", response_model=AuthUserRead)
def me(request: Request, db: Session = Depends(get_db)) -> AuthUserRead:
    user = get_current_user_or_401(request, db)
    return AuthUserRead.model_validate(user)


@auth_router.post("/logout", response_model=AuthLogoutResponse)
def logout(request: Request, db: Session = Depends(get_db)) -> AuthLogoutResponse:
    settings = _settings()
    if not verify_csrf(request, db):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token.")

    revoke_session(request, db)

    response_payload = AuthLogoutResponse(success=True, message="Logged out successfully.")
    json_response = JSONResponse(content=response_payload.model_dump())
    json_response.delete_cookie(
        key=settings.auth_session_cookie_name,
        path="/",
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite.lower(),
        httponly=True,
    )
    json_response.delete_cookie(
        key=settings.auth_csrf_cookie_name,
        path="/",
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite.lower(),
        httponly=False,
    )
    return json_response  # type: ignore[return-value]
