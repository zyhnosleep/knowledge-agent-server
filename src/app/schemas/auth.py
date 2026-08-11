"""认证与用户相关的 Pydantic Schema 定义。

本模块定义基于飞书（Feishu/Lark）SSO 的用户认证接口所使用的数据模型：

- ``AuthUserRead``：当前登录用户的只读信息（用于认证接口响应；配置了
  ``from_attributes=True``，可直接从 ORM 对象转换）。
- ``AuthLogoutResponse``：登出接口的响应体。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AuthUserRead(BaseModel):
    """用户只读视图（认证接口响应用）。

    展示当前登录用户的基本信息。``model_config`` 开启 ``from_attributes=True``，
    允许直接从 SQLAlchemy 的 ``User`` ORM 对象构造本模型。
    """

    id: str  # 用户 ID
    feishu_open_id: str  # 飞书 open_id（登录主标识）
    feishu_union_id: str | None = None  # 飞书 union_id（跨应用统一 ID，可空）
    tenant_key: str | None = None  # 飞书租户键（tenant key，可空）
    display_name: str  # 显示名称
    avatar_url: str | None = None  # 头像 URL（可空）
    is_active: bool  # 账号是否启用

    model_config = ConfigDict(from_attributes=True)  # 允许从 ORM 对象（from_attributes）直接构造模型


class AuthLogoutResponse(BaseModel):
    """登出接口响应体。

    返回登出操作是否成功及提示消息。
    """

    success: bool  # 登出是否成功
    message: str  # 提示消息
