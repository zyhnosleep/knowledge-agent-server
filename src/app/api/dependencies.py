"""API 公共依赖（FastAPI 依赖注入）模块。

本模块集中定义跨路由共享的认证依赖项，供其它 API 路由模块复用，
避免每个路由重复实现"解析当前登录用户"的逻辑。

当前提供的依赖：
    require_business_api_user —— 业务 API 的"当前用户"依赖。
        当系统启用了认证（AUTH_ENABLED=true）时，该依赖会要求请求携带
        有效的登录会话 Cookie，并在无法解析出有效用户时抛出 HTTP 401；
        当认证未启用时，它返回 ``None``，使所有路由退化为匿名访问模式。

被依赖方：
    - ``get_settings``    读取系统配置（判断认证是否启用）。
    - ``get_db``          提供数据库会话（用于查会话/用户）。
    - ``get_current_user_or_401``  核心的用户解析逻辑（在 services.auth 中）。
"""

from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.models.records import User
from app.services.auth import get_current_user_or_401


def require_business_api_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User | None:
    """解析并返回当前登录用户（认证未启用时返回 None）。

    该函数被设计为 FastAPI 依赖注入项：在路由参数中通过
    ``current_user: User | None = Depends(require_business_api_user)``
    引入，即可自动完成身份解析。返回类型允许为 ``None``，因此
    各路由需要自行决定：当用户为 ``None`` 时是否允许匿名访问。

    参数：
        request (Request): 当前 HTTP 请求对象，用于从中提取登录会话
            Cookie 以识别用户身份。
        db (Session): 数据库会话，由 ``Depends(get_db)`` 注入，
            用于在数据库中查询会话与用户记录。

    返回：
        User | None: 若系统启用了认证且请求携带有效会话，返回对应的
            User 对象；否则（认证未启用）返回 ``None``。

    异常：
        HTTPException: 当系统启用了认证，但请求未携带有效登录会话时，
            由 ``get_current_user_or_401`` 抛出 HTTP 401。
    """
    if not get_settings().auth_enabled:
        # 认证未启用：直接放行并返回 None，表示匿名访问，
        # 是否允许匿名调用由具体路由自行判断。
        return None
    # 认证已启用：必须能从请求会话中解析出有效用户，否则抛出 401。
    return get_current_user_or_401(request, db)
