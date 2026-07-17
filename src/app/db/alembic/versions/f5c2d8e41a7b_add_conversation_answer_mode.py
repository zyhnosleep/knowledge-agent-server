"""add conversation answer mode

Revision ID: f5c2d8e41a7b
Revises: 9bd042a0f5ba
Create Date: 2026-07-17
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f5c2d8e41a7b"
down_revision: Union[str, Sequence[str], None] = "9bd042a0f5ba"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversation_sessions",
        sa.Column(
            "answer_mode",
            sa.String(length=16),
            server_default="auto",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("conversation_sessions", "answer_mode")
