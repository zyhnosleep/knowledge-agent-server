from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AuthUserRead(BaseModel):
    id: str
    feishu_open_id: str
    feishu_union_id: str | None = None
    tenant_key: str | None = None
    display_name: str
    avatar_url: str | None = None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class AuthLogoutResponse(BaseModel):
    success: bool
    message: str
