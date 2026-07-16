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
    if not get_settings().auth_enabled:
        return None
    return get_current_user_or_401(request, db)
