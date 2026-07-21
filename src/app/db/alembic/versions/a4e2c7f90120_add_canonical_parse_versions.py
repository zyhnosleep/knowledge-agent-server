"""add canonical parse versions and parent child chunks

Revision ID: a4e2c7f90120
Revises: 81f6b74cc203
Create Date: 2026-07-21
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a4e2c7f90120"
down_revision: Union[str, Sequence[str], None] = "81f6b74cc203"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


CHUNK_COLUMNS = (
    "parse_version",
    "parent_chunk_id",
    "chunk_role",
    "block_type",
    "section_path",
    "source_block_ids",
    "source_spans",
    "contextual_prefix",
    "embedding_text",
    "contextualization_model",
    "contextualization_version",
    "contextualization_prompt_version",
    "contextualized_at",
    "parser_name",
    "parser_version",
    "splitter_name",
    "splitter_version",
    "splitting_model",
    "semantic_boundary_score",
    "token_count",
    "previous_chunk_id",
    "next_chunk_id",
)


def _create_parse_version_table() -> None:
    op.create_table(
        "document_parse_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("version_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("artifact_dir", sa.Text(), nullable=False),
        sa.Column("parser_name", sa.String(length=120), nullable=True),
        sa.Column("parser_version", sa.String(length=120), nullable=True),
        sa.Column("manifest_json", sa.JSON(), nullable=False),
        sa.Column("quality_json", sa.JSON(), nullable=False),
        sa.Column("stage_state", sa.JSON(), nullable=False),
        sa.Column("activated_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "version_key",
            name="uq_document_parse_versions_document_version",
        ),
    )
    op.create_index(
        "ix_document_parse_versions_document_id",
        "document_parse_versions",
        ["document_id"],
        unique=False,
    )


def _add_chunk_columns() -> None:
    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.add_column(
            sa.Column("parse_version", sa.String(length=128), nullable=True)
        )
        batch_op.add_column(
            sa.Column("parent_chunk_id", sa.String(length=36), nullable=True)
        )
        batch_op.add_column(sa.Column("chunk_role", sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column("block_type", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("section_path", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("source_block_ids", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("source_spans", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("contextual_prefix", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("embedding_text", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("contextualization_model", sa.String(length=120), nullable=True)
        )
        batch_op.add_column(
            sa.Column("contextualization_version", sa.String(length=120), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "contextualization_prompt_version",
                sa.String(length=120),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("contextualized_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("parser_name", sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column("parser_version", sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column("splitter_name", sa.String(length=120), nullable=True))
        batch_op.add_column(
            sa.Column("splitter_version", sa.String(length=120), nullable=True)
        )
        batch_op.add_column(
            sa.Column("splitting_model", sa.String(length=120), nullable=True)
        )
        batch_op.add_column(
            sa.Column("semantic_boundary_score", sa.Float(), nullable=True)
        )
        batch_op.add_column(sa.Column("token_count", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("previous_chunk_id", sa.String(length=36), nullable=True)
        )
        batch_op.add_column(
            sa.Column("next_chunk_id", sa.String(length=36), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_document_chunks_parent_chunk_id",
            "document_chunks",
            ["parent_chunk_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_foreign_key(
            "fk_document_chunks_previous_chunk_id",
            "document_chunks",
            ["previous_chunk_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_document_chunks_next_chunk_id",
            "document_chunks",
            ["next_chunk_id"],
            ["id"],
            ondelete="SET NULL",
        )


def _backfill_legacy_chunks() -> None:
    chunks = sa.table(
        "document_chunks",
        sa.column("document_id", sa.String(length=36)),
        sa.column("text", sa.Text()),
        sa.column("token_estimate", sa.Integer()),
        sa.column("parse_version", sa.String(length=128)),
        sa.column("chunk_role", sa.String(length=20)),
        sa.column("block_type", sa.String(length=40)),
        sa.column("section_path", sa.JSON()),
        sa.column("source_block_ids", sa.JSON()),
        sa.column("source_spans", sa.JSON()),
        sa.column("embedding_text", sa.Text()),
        sa.column("token_count", sa.Integer()),
    )
    op.execute(
        chunks.update().values(
            parse_version="legacy",
            chunk_role="child",
            block_type="narrative",
            section_path=[],
            source_block_ids=[],
            source_spans=[],
            embedding_text=chunks.c.text,
            token_count=chunks.c.token_estimate,
        )
    )

    parse_versions = sa.table(
        "document_parse_versions",
        sa.column("id", sa.String(length=36)),
        sa.column("document_id", sa.String(length=36)),
        sa.column("version_key", sa.String(length=128)),
        sa.column("status", sa.String(length=40)),
        sa.column("artifact_dir", sa.Text()),
        sa.column("parser_name", sa.String(length=120)),
        sa.column("parser_version", sa.String(length=120)),
        sa.column("manifest_json", sa.JSON()),
        sa.column("quality_json", sa.JSON()),
        sa.column("stage_state", sa.JSON()),
        sa.column("activated_at", sa.DateTime()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )
    bind = op.get_bind()
    document_ids = bind.execute(sa.select(chunks.c.document_id).distinct()).scalars()
    now = datetime.utcnow()
    for document_id in document_ids:
        bind.execute(
            parse_versions.insert().values(
                id=str(uuid.uuid4()),
                document_id=document_id,
                version_key="legacy",
                status="quarantined",
                artifact_dir="legacy",
                parser_name=None,
                parser_version=None,
                manifest_json={},
                quality_json={},
                stage_state={"migration": "legacy"},
                activated_at=None,
                created_at=now,
                updated_at=now,
            )
        )


def _finalize_chunk_columns() -> None:
    with op.batch_alter_table("document_chunks") as batch_op:
        for column_name, existing_type in (
            ("parse_version", sa.String(length=128)),
            ("chunk_role", sa.String(length=20)),
            ("block_type", sa.String(length=40)),
            ("section_path", sa.JSON()),
            ("source_block_ids", sa.JSON()),
            ("source_spans", sa.JSON()),
            ("embedding_text", sa.Text()),
            ("token_count", sa.Integer()),
        ):
            batch_op.alter_column(
                column_name, existing_type=existing_type, nullable=False
            )
        batch_op.create_index(
            "ix_document_chunks_document_parse_version_role",
            ["document_id", "parse_version", "chunk_role"],
            unique=False,
        )


def _upgrade_pgvector() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE document_chunk_pgvector_index ADD COLUMN parse_version VARCHAR(128)"
        )
        op.execute(
            "UPDATE document_chunk_pgvector_index SET parse_version = 'legacy' "
            "WHERE parse_version IS NULL"
        )
        op.execute(
            "ALTER TABLE document_chunk_pgvector_index "
            "ALTER COLUMN parse_version SET NOT NULL"
        )
        op.execute(
            "ALTER TABLE document_chunk_pgvector_index "
            "ALTER COLUMN parse_version SET DEFAULT 'legacy'"
        )
        op.execute(
            "CREATE INDEX ix_document_chunk_pgvector_index_document_parse_version "
            "ON document_chunk_pgvector_index (document_id, parse_version)"
        )


def upgrade() -> None:
    _create_parse_version_table()
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column("active_parse_version", sa.String(length=128), nullable=True)
        )
    _add_chunk_columns()
    _backfill_legacy_chunks()
    _finalize_chunk_columns()
    _upgrade_pgvector()


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "DROP INDEX IF EXISTS ix_document_chunk_pgvector_index_document_parse_version"
        )
        op.execute(
            "ALTER TABLE document_chunk_pgvector_index DROP COLUMN IF EXISTS parse_version"
        )

    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.drop_index("ix_document_chunks_document_parse_version_role")
        batch_op.drop_constraint(
            "fk_document_chunks_next_chunk_id", type_="foreignkey"
        )
        batch_op.drop_constraint(
            "fk_document_chunks_previous_chunk_id", type_="foreignkey"
        )
        batch_op.drop_constraint(
            "fk_document_chunks_parent_chunk_id", type_="foreignkey"
        )
        for column_name in reversed(CHUNK_COLUMNS):
            batch_op.drop_column(column_name)

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_column("active_parse_version")
    op.drop_index(
        "ix_document_parse_versions_document_id",
        table_name="document_parse_versions",
    )
    op.drop_table("document_parse_versions")
