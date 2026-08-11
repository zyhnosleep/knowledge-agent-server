"""
auth.py —— 认证与授权（OAuth / 会话 / CSRF）模块
================================================

职责：
- 提供一套完整的认证基础设施：
  - Token 生成与哈希（会话令牌、CSRF 令牌）。
  - 带过期时间与 HMAC 签名的"签名值"编解码（用于 OAuth state）。
  - PKCE 挑战值计算。
  - 飞书（Feishu/Lark）OAuth 授权码换 token、拉取用户信息。
  - 用户 upsert、会话创建/查询/吊销。
  - 当前用户解析、CSRF 校验。
- 认证通过浏览器 Cookie（会话令牌 + CSRF 令牌）与 API 结合的方式工作。

安全设计：
- 会话令牌只以 SHA-256 哈希形式入库（``hash_token``），数据库泄露也
  不会暴露可用令牌。
- HMAC 签名使用 ``hmac.compare_digest`` 做常量时间比较，防时序攻击。
- 签名值中内嵌过期时间戳，服务端不保存 state 也能校验有效期。
- 所有对外错误统一抛 ``HTTPException``，由 FastAPI 转成标准 HTTP 响应。
"""

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
    """URL 安全的 Base64 编码，并去掉填充字符 ``=``。

    用于生成紧凑的 token（如 ``secrets.token_bytes(32)`` 的编码结果），
    使 token 可以直接放进 Cookie / URL 而不含 ``+``、``/``、``=``。
    """
    return urlsafe_b64encode(data).decode().rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    """URL 安全的 Base64 解码，自动补齐被 ``_urlsafe_b64encode`` 去掉的填充。

    解码前先计算补齐到 4 的倍数所需的 ``=`` 数量并追加，
    以兼容未补 pad 的 URL-safe Base64 串。
    """
    padding = (-len(value)) % 4
    if padding:
        value += "=" * padding
    return urlsafe_b64decode(value)


def hash_token(token: str) -> str:
    """计算令牌的 SHA-256 十六进制哈希。

    用于安全存储：数据库中只保存令牌哈希而非明文令牌；
    校验时对用户提交的令牌重新哈希后比对。
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_token() -> str:
    """生成 256 位（32 字节）密码学安全的随机令牌，URL-safe Base64 编码。

    用于会话令牌、CSRF 令牌与 OAuth verifier 等一次性/短期凭据。
    """
    return _urlsafe_b64encode(secrets.token_bytes(32))


def _sign(data: str, secret: str, max_age_seconds: int) -> str:
    """创建带过期时间与 HMAC-SHA256 签名的签名值。

    格式：``<payload_b64>.<signature_b64>``，其中 payload 为
    ``"<expires>:<data>"`` 的 URL-safe Base64 编码。

    参数：
    - ``data``：要签名的原始数据（如 OAuth state 与 verifier 的组合）。
    - ``secret``：HMAC 签名密钥（服务端机密，不随数据下发）。
    - ``max_age_seconds``：有效期（秒），从当前时刻起算。

    用途：为 OAuth state 生成自校验、可过期、无需服务端存储的令牌。
    """
    # 过期时间 = 当前 UTC 时间戳 + 有效期；与 data 一起作为签名对象，
    # 防止篡改过期时间
    expires = int(datetime.utcnow().timestamp()) + max_age_seconds
    payload = f"{expires}:{data}"
    payload_b64 = _urlsafe_b64encode(payload.encode())
    # 对 payload 做 HMAC-SHA256，防止内容被篡改
    signature = hmac.new(secret.encode("utf-8"), payload_b64.encode(), hashlib.sha256).digest()
    signature_b64 = _urlsafe_b64encode(signature)
    return f"{payload_b64}.{signature_b64}"


def _unsign(signed_value: str, secret: str) -> str:
    """校验签名值并返回其携带的原始数据；签名/格式/过期任一不合法则报错。

    步骤：
    1. 按 ``.`` 拆分出 payload 与 signature；格式不对抛 400。
    2. 用同一密钥重算 payload 的签名，并 ``compare_digest`` 常量时间比对；
       不匹配说明被篡改，抛 400。
    3. 解码 payload 并拆出过期时间与数据；格式损坏抛 400。
    4. 当前时间超过过期时间则抛 400。

    返回：原始数据字符串。

    说明：本函数把校验失败统一映射为 ``HTTPException(400)``，
    供 OAuth state 等场景直接向上抛给客户端。
    """
    try:
        payload_b64, signature_b64 = signed_value.split(".")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid signed value.") from exc

    # 重算期望签名并与客户端提交的签名做常量时间比较
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

    # 过期校验：当前时间戳大于签名中的过期时间则拒绝
    if datetime.utcnow().timestamp() > expires:
        raise HTTPException(status_code=400, detail="Signed value expired.")

    return data


def build_pkce_challenge(verifier: str) -> str:
    """计算 PKCE challenge（SHA-256(verifier) 的 URL-safe Base64）。

    用于 OAuth Authorization Code + PKCE 流程：客户端先生成随机的
    code_verifier，把它的哈希 challenge 放入授权请求，授权回调时再
    提交原始 verifier，供授权服务器验证代码绑定。
    """
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _urlsafe_b64encode(digest)


def create_oauth_state(secret: str, max_age_seconds: int) -> tuple[str, str, str]:
    """创建 OAuth 授权流程所需的 state 三元组。

    返回 ``(state, signed, verifier)``：
    - ``state``：随机 CSRF 令牌，随授权请求发送给飞书并随回调带回，
      用于校验回调来源。
    - ``signed``：把 ``"{state}:{verifier}"`` 签名后得到的自校验令牌，
      由服务端放在 Cookie 中。
    - ``verifier``：PKCE code_verifier，回调时换取 token 用。

    说明：state 与 verifier 都经 ``_sign`` 绑定，回调时通过
    ``verify_oauth_state`` 校验状态一致性并取回 verifier。
    """
    state = generate_token()
    verifier = generate_token()
    signed = _sign(f"{state}:{verifier}", secret, max_age_seconds)
    return state, signed, verifier


def verify_oauth_state(signed_state: str, query_state: str, secret: str) -> str:
    """校验 OAuth 回调携带的 state 与 Cookie 中签名值内嵌的 state 一致。

    流程：
    1. ``_unsign`` 校验签名并解码，取出 ``"{state}:{verifier}"``。
    2. 拆分出存储的 state 与 verifier。
    3. 常量时间比较存储的 state 与回调参数 ``query_state``，防止 CSRF
       式 state 伪造；不一致抛 400。

    返回：verifier（PKCE code_verifier），供后续换 token 使用。
    """
    data = _unsign(signed_state, secret)
    try:
        stored_state, verifier = data.split(":", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid OAuth state payload.") from exc
    if not hmac.compare_digest(stored_state, query_state):
        raise HTTPException(status_code=400, detail="OAuth state mismatch.")
    return verifier


def exchange_code_for_token(code: str, verifier: str, settings: Settings) -> str:
    """用飞书授权码 + PKCE verifier 换取 access_token。

    参数：
    - ``code``：飞书回调携带的授权码。
    - ``verifier``：PKCE code_verifier。
    - ``settings``：包含飞书 OAuth 相关配置的应用配置。

    流程：
    1. 若飞书相关配置缺失，返回 503（OAuth 未配置）。
    2. POST 到 ``settings.feishu_token_url``，携带授权码、client 凭据、
       回调地址与 code_verifier。
    3. 请求失败（非 2xx）直接抛 httpx 异常；业务错误码非 0 抛 400。
    4. access_token 可能在顶层 ``payload["access_token"]`` 或嵌套在
       ``payload["data"]["access_token"]``，两种位置都尝试提取。

    返回：飞书 access_token 字符串。
    """
    if not settings.feishu_app_id or not settings.feishu_app_secret or not settings.feishu_redirect_uri:
        raise HTTPException(status_code=503, detail="Feishu OAuth is not configured.")

    response = httpx.post(
        settings.feishu_token_url,
        json={
            "grant_type": "authorization_code",  # 授权码模式
            "code": code,
            "client_id": settings.feishu_app_id,
            "client_secret": settings.feishu_app_secret,
            "redirect_uri": settings.feishu_redirect_uri,
            "code_verifier": verifier,  # PKCE 验证值
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    # 飞书 API 顶层 code != 0 表示业务失败
    if payload.get("code") != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Feishu token exchange failed: {payload.get('msg')}"
        )

    # access_token 的位置在飞书各版本接口中可能不同，两处都尝试
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
    """用 access_token 调用飞书接口获取用户信息。

    使用 Bearer Token 认证；业务错误码非 0 抛 400；
    成功后返回 ``payload["data"]``（含 open_id、union_id、name 等）。
    """
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
    """根据飞书用户信息新建或更新本地 ``User`` 记录（upsert）。

    流程：
    1. 必须有 ``open_id``，否则抛 400。
    2. 按 ``feishu_open_id`` 查询现有用户。
    3. 不存在则创建新用户（设置初始资料与激活状态、记录登录时间）。
    4. 已存在则更新资料字段（union_id、tenant_key、display_name、
       avatar_url），置为激活并刷新登录时间；缺失的新值不回填旧值
       （用 ``or`` 保留已有数据）。
    5. 提交事务并刷新对象。

    返回：数据库中的 ``User`` 实例。
    """
    open_id = user_info.get("open_id")
    if not open_id:
        raise HTTPException(status_code=400, detail="Feishu user info missing open_id.")

    user = db.scalar(select(User).where(User.feishu_open_id == open_id))
    now = datetime.utcnow()
    if user is None:
        # 首次登录：创建新用户
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
        # 老用户：更新资料，缺失字段保留旧值
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
    """创建用户会话，返回 ``(token, csrf_token, session)``。

    流程：
    1. 生成随机会话令牌与 CSRF 令牌（各自仅以哈希入库）。
    2. 创建 ``AuthSession`` 记录：token_hash、csrf_hash、过期时间
       （now + max_age_seconds）、首次活动时间。
    3. 提交并刷新。

    说明：令牌明文仅在本次返回中交给调用方写入 Cookie，
    数据库侧只保留 SHA-256 哈希，降低令牌泄露风险。
    """
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
    """按明文令牌查找有效会话。

    条件（全部满足才算有效）：
    - 令牌存在（None 直接返回 None）。
    - ``token_hash == hash_token(token)``（数据库中保存的是哈希）。
    - 未被吊销（``revoked_at is None``）。
    - 未过期（``expires_at > now``）。

    返回：有效的 ``AuthSession``，否则 None。
    """
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
    """从请求 Cookie 中解析当前登录用户；未登录返回 None。

    流程：读取 ``auth_session_cookie_name`` 指定的会话 Cookie，
    用 ``get_session_by_token`` 找到有效会话，再取其关联用户并检查
    用户激活状态。

    返回：激活的 ``User`` 实例，或 None。
    """
    settings = get_settings()
    token = request.cookies.get(settings.auth_session_cookie_name)
    session = get_session_by_token(db, token)
    if session is None or session.user is None or not session.user.is_active:
        return None
    return session.user


def get_current_user_or_401(request: Request, db: Session) -> User:
    """同 ``get_current_user``，但未登录时抛 401 异常。

    用于要求强制登录的路由/依赖。
    """
    user = get_current_user(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def verify_csrf(request: Request, db: Session) -> bool:
    """校验请求的 CSRF 令牌是否与服务端会话绑定一致。

    校验链路（全部满足才算通过）：
    1. 会话有效：根据会话 Cookie 找到未过期/未吊销的会话。
    2. 双通道令牌：CSRF 同时出现在 Cookie（``auth_csrf_cookie_name``）
       与请求头（``X-CSRF-Token``）中。
    3. Cookie 值与请求头值常量时间相等。
    4. 请求头值的哈希与会话记录的 ``csrf_hash`` 常量时间相等
       （会话级绑定）。

    返回：True 表示校验通过，否则 False。
    """
    settings = get_settings()
    session_token = request.cookies.get(settings.auth_session_cookie_name)
    session = get_session_by_token(db, session_token)
    if session is None:
        return False

    # CSRF 令牌要求 Cookie 与请求头双通道同时携带
    csrf_cookie = request.cookies.get(settings.auth_csrf_cookie_name)
    header_token = request.headers.get("X-CSRF-Token")
    if not csrf_cookie or not header_token:
        return False

    # Cookie 与请求头令牌一致
    if not hmac.compare_digest(csrf_cookie, header_token):
        return False

    # 请求头令牌哈希与会话记录的 CSRF 哈希一致（会话级绑定）
    return hmac.compare_digest(session.csrf_hash, hash_token(header_token))


def revoke_session(request: Request, db: Session) -> bool:
    """吊销当前请求对应的会话（登出）。

    流程：读取会话 Cookie 找到会话，若有效则把 ``revoked_at`` 置为当前
    时间并提交；之后 ``get_session_by_token`` 会因 ``revoked_at`` 非空而
    拒绝该令牌。

    返回：成功吊销返回 True；无有效会话返回 False。
    """
    settings = get_settings()
    token = request.cookies.get(settings.auth_session_cookie_name)
    session = get_session_by_token(db, token)
    if session is None:
        return False
    session.revoked_at = datetime.utcnow()
    db.commit()
    return True


def auth_is_configured(settings: Settings) -> bool:
    """判断认证功能是否已完整配置（可启用）。

    需要同时满足：
    - ``auth_enabled``：总开关。
    - ``auth_session_secret``：会话签名/哈希密钥。
    - 飞书 OAuth 三件套：``feishu_app_id``、``feishu_app_secret``、
      ``feishu_redirect_uri``。

    任一缺失即认为认证未配置完成。
    """
    return bool(
        settings.auth_enabled
        and settings.auth_session_secret
        and settings.feishu_app_id
        and settings.feishu_app_secret
        and settings.feishu_redirect_uri
    )
