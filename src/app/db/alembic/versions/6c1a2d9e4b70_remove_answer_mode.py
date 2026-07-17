"""remove conversation answer mode

Revision ID: 6c1a2d9e4b70
Revises: f5c2d8e41a7b
Create Date: 2026-07-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "6c1a2d9e4b70"
down_revision: Union[str, Sequence[str], None] = "f5c2d8e41a7b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("conversation_sessions") as batch_op:
        batch_op.drop_column("answer_mode")


def downgrade() -> None:
    with op.batch_alter_table("conversation_sessions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "answer_mode",
                sa.String(length=16),
                server_default="auto",
                nullable=False,
            )
        )
