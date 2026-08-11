"""认证 API 路由模块。

本模块实现基于飞书（Feishu）OAuth 2.0 + PKCE 的登录认证流程，以及
会话（session）管理相关的 HTTP 端点。登录成功后通过 Cookie 维持会话，
后续业务路由通过 dependencies.require_business_api_user 识别当前用户。

API 端点：
    GET  /auth/status   —— 查询认证功能是否启用/配置完成（供前端决定是否
                           展示登录入口）。
    GET  /auth/login    —— 发起 OAuth 登录：生成 state 与 PKCE challenge，
                           302/307 重定向到飞书授权页。
    GET  /auth/callback —— 飞书授权页回调：校验 state/PKCE、换取 token、
                           获取用户信息、落库并写入会话 Cookie。
    GET  /auth/me       —— 返回当前登录用户信息（未登录时 401）。
    POST /auth/logout   —— 校验 CSRF 后注销会话并清除相关 Cookie。

涉及的核心安全机制：
    - OAuth state：防止 CSRF / 重放攻击，state 经签名后写入 HttpOnly
      Cookie，回调时与 query 中的 state 比对。
    - PKCE S256：授权码换 token 前必须携带并校验 code_verifier，
      防止授权码被第三方截获后冒用。
    - 会话 Cookie（HttpOnly）：服务端保存的登录态凭据。
    - CSRF Cookie（非 HttpOnly）：前端在 POST 请求时需要回传以通过校验。
"""

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

# 认证路由专用路由器实例，由应用启动代码挂载到 /api 前缀之下。
auth_router = APIRouter()


def _settings() -> Settings:
    """返回应用配置对象（供本模块内部复用）。

    返回：
        Settings: 应用配置单例，包含认证相关的全部配置项
            （AUTH_ENABLED、飞书凭证、Cookie 策略、会话有效期等）。
    """
    return get_settings()


@auth_router.get("/status")
def auth_status() -> dict[str, bool]:
    """查询认证功能当前的状态。

    返回：
        dict[str, bool]: 包含两个字段——
            - ``enabled``: 认证开关是否打开（AUTH_ENABLED）；
            - ``configured``: 认证所需的飞书凭证等配置是否齐全。
        前端可据此决定是否展示"登录"入口或直接进入匿名模式。
    """
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
    """在重定向响应上写入一个带统一安全策略的 Cookie。

    统一读取配置中的 ``auth_cookie_secure``（是否仅 HTTPS 下发）与
    ``auth_cookie_samesite``（SameSite 策略），保证所有认证相关
    Cookie 的安全属性一致。

    参数：
        response (RedirectResponse): 目标响应对象（重定向响应）。
        name (str): Cookie 名称。
        value (str): Cookie 值。
        max_age (int): Cookie 有效期（秒）。
        httponly (bool): 是否设置 HttpOnly 标志（默认 True；CSRF Cookie
            需要前端读取，必须设为 False）。
    """
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
    """从重定向响应中删除（失效）指定认证 Cookie。

    使用与写入时相同的安全参数删除 Cookie，确保浏览器能正确匹配并清除
    该 Cookie（Secure/SameSite/Path 不一致会导致删除失败）。

    参数：
        response (RedirectResponse): 目标响应对象。
        name (str): 要删除的 Cookie 名称。
    """
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
    """发起飞书 OAuth 登录，重定向到飞书授权页。

    流程：
        1. 校验认证是否启用且配置完整（否则返回 503）。
        2. 生成随机的 state 与其签名版本，以及 PKCE 的 code_verifier。
        3. 用 code_verifier 计算 S256 的 code_challenge。
        4. 构造飞书授权 URL，携带 app_id / redirect_uri / state /
           code_challenge 等参数。
        5. 将签名的 state 写入 HttpOnly Cookie（供回调时校验），
           并以 307 临时重定向跳转到飞书。

    参数：
        request (Request): 当前 HTTP 请求（用于生成并写入 state Cookie）。

    返回：
        RedirectResponse: 指向飞书授权页的 307 重定向响应。

    异常：
        HTTPException(503): 认证未启用或飞书凭证缺失。
    """
    settings = _settings()
    # 认证未启用或凭证不完整时直接拒绝发起登录。
    if not auth_is_configured(settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not enabled or Feishu credentials are missing.",
        )
    # 缺少用于签名 state 的会话密钥时同样拒绝（防止生成无法校验的 state）。
    if not settings.auth_session_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not enabled or Feishu credentials are missing.",
        )

    # 生成一次性 OAuth state（明文 + 签名版）以及 PKCE 的随机 verifier。
    state, signed_state, verifier = create_oauth_state(
        settings.auth_session_secret,
        settings.auth_state_max_age_seconds,
    )
    # 基于 verifier 计算 PKCE 的 S256 code_challenge，用于授权码换 token 校验。
    challenge = build_pkce_challenge(verifier)

    # 组装飞书授权页的查询参数。
    params = {
        "app_id": settings.feishu_app_id,
        "redirect_uri": settings.feishu_redirect_uri,
        "response_type": "code",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{settings.feishu_auth_url}?{urlencode(params)}"

    # 307 临时重定向到飞书授权页；同时把签名 state 写入 HttpOnly Cookie，
    # 供 /callback 回调时与飞书回传的 state 比对以防御 CSRF/重放。
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
    """处理飞书 OAuth 回调，完成登录并建立本地会话。

    流程：
        1. 校验认证启用与配置完整。
        2. 从 Cookie 取签名 state、从 query 取明文 state，比对校验通过后
           还原出 PKCE 的 code_verifier。
        3. 用 code + verifier 向飞书换取 access_token，再获取用户信息。
        4. 若配置了 ``feishu_allowed_tenant``，校验用户租户是否被允许。
        5. 将飞书用户信息写入/更新到本地数据库，创建登录会话与 CSRF token。
        6. 清除 OAuth state Cookie，写入会话 Cookie 与 CSRF Cookie，
           并重定向回首页。

    参数：
        request (Request): 当前 HTTP 请求，包含飞书回传的 query 参数
            （state / code）以及之前写入的 state Cookie。
        db (Session): 数据库会话，用于查询/写入用户与会话记录。

    返回：
        RedirectResponse: 指向首页的 307 重定向，携带登录态 Cookie。

    异常：
        HTTPException(503): 认证未启用或配置缺失。
        HTTPException(400): 缺少 state Cookie / state 参数 / 授权码，
            或 OAuth state 校验失败（含重放）。
        HTTPException(403): 用户所在租户不在白名单内。
    """
    settings = _settings()
    # 认证未启用时拒绝处理回调。
    if not auth_is_configured(settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not enabled.",
        )
    # 缺少会话密钥则无法完成 state 校验，直接拒绝。
    if not settings.auth_session_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not enabled.",
        )

    # 取出此前写在 Cookie 中的签名 state（丢失说明登录流程不完整）。
    signed_state = request.cookies.get(settings.auth_state_cookie_name)
    if not signed_state:
        raise HTTPException(status_code=400, detail="Missing OAuth state cookie.")

    # 取出飞书回传的明文 state（丢失说明回调链路异常）。
    query_state = request.query_params.get("state")
    if not query_state:
        raise HTTPException(status_code=400, detail="Missing OAuth state parameter.")

    # 授权码 code 是换取 token 的关键凭据，缺失则无法继续。
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing OAuth authorization code.")

    # 校验签名 state：防 CSRF/防重放，并还原出本次登录的 PKCE verifier。
    verifier = verify_oauth_state(signed_state, query_state, settings.auth_session_secret)
    # 携带 code + verifier 向飞书换取访问令牌（PKCE 在此生效）。
    access_token = exchange_code_for_token(code, verifier, settings)
    # 用访问令牌获取飞书用户信息。
    user_info = fetch_feishu_user_info(access_token, settings)

    # 若配置了租户白名单，则校验用户所属租户是否被允许访问。
    tenant_key = user_info.get("tenant_key")
    if settings.feishu_allowed_tenant and tenant_key != settings.feishu_allowed_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User tenant is not allowed.",
        )

    # 将飞书用户信息写入/更新本地用户表，并创建登录会话与 CSRF token。
    user = upsert_user_from_feishu(db, user_info)
    session_token, csrf_token, _ = create_session(
        db, user, settings.auth_session_max_age_seconds
    )

    # 登录成功：清除一次性 OAuth state Cookie，写入会话 Cookie（HttpOnly）
    # 与 CSRF Cookie（非 HttpOnly，供前端读取并在 POST 时回传），跳回首页。
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
    """返回当前登录用户的信息。

    参数：
        request (Request): 当前 HTTP 请求，用于提取登录会话 Cookie。
        db (Session): 数据库会话。

    返回：
        AuthUserRead: 当前登录用户的公开信息。

    异常：
        HTTPException(401): 请求未携带有效登录会话（由
            ``get_current_user_or_401`` 抛出）。
    """
    # 解析当前用户；未登录时抛出 401。
    user = get_current_user_or_401(request, db)
    return AuthUserRead.model_validate(user)


@auth_router.post("/logout", response_model=AuthLogoutResponse)
def logout(request: Request, db: Session = Depends(get_db)) -> AuthLogoutResponse:
    """注销当前登录会话并清除认证相关 Cookie。

    流程：
        1. 先校验请求携带的 CSRF token（防御跨站请求伪造）。
        2. 吊销数据库中的会话记录。
        3. 清除浏览器端的会话 Cookie 与 CSRF Cookie，返回 JSON 结果。

    参数：
        request (Request): 当前 HTTP 请求，包含会话 Cookie 与 CSRF token。
        db (Session): 数据库会话，用于吊销会话。

    返回：
        AuthLogoutResponse: 注销成功的结果（success + 提示消息）。

    异常：
        HTTPException(403): CSRF token 校验失败。
    """
    settings = _settings()
    # 登出属于写操作，必须先校验 CSRF token，防止恶意站点诱导登出。
    if not verify_csrf(request, db):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token.")

    # 吊销数据库中的登录会话记录。
    revoke_session(request, db)

    # 构造注销成功响应，并清除浏览器端的两个认证 Cookie。
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
