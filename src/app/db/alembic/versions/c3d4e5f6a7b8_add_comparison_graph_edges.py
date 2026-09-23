"""Add Graph-lite edges used by the comparison workbench."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b7e8f9a0c1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_edges",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=40), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("relation_type", sa.String(length=60), nullable=False),
        sa.Column("target_type", sa.String(length=40), nullable=False),
        sa.Column("target_id", sa.String(length=255), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=True),
        sa.Column("parse_version", sa.String(length=128), nullable=True),
        sa.Column("evidence_chunk_id", sa.String(length=36), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("extraction_version", sa.String(length=80), nullable=False, server_default="comparison-v1"),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evidence_chunk_id"], ["document_chunks.id"], ondelete="SET NULL"),
        sa.UniqueConstraint(
            "project_id",
            "source_type",
            "source_id",
            "relation_type",
            "target_type",
            "target_id",
            "parse_version",
            name="uq_knowledge_edges_identity",
        ),
    )
    op.create_index(
        "ix_knowledge_edges_project_id", "knowledge_edges", ["project_id"]
    )
    op.create_index(
        "ix_knowledge_edges_document_id", "knowledge_edges", ["document_id"]
    )
    op.create_index(
        "ix_knowledge_edges_project_relation",
        "knowledge_edges",
        ["project_id", "relation_type"],
    )
    op.create_index(
        "ix_knowledge_edges_document_version",
        "knowledge_edges",
        ["document_id", "parse_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_edges_document_version", table_name="knowledge_edges")
    op.drop_index("ix_knowledge_edges_project_relation", table_name="knowledge_edges")
    op.drop_index("ix_knowledge_edges_document_id", table_name="knowledge_edges")
    op.drop_index("ix_knowledge_edges_project_id", table_name="knowledge_edges")
    op.drop_table("knowledge_edges")
