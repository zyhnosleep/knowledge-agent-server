"""resize document embeddings for the hosted Qwen endpoint to 1024 dimensions

Revision ID: b7e8f9a0c1d2
Revises: a4e2c7f90120
Create Date: 2026-09-22

The development deployment uses the OpenAI-compatible Qwen3.7 text embedding
endpoint, which returns 1024-dimensional vectors. The historical schema was
created for the local 2560-dimensional Ollama model, so replace the empty
shadow index table with the active dimension before ingestion.
"""

from typing import Sequence, Union

from alembic import op


revision: str = "b7e8f9a0c1d2"
down_revision: Union[str, Sequence[str], None] = "a4e2c7f90120"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _replace_vector_table(vector_type: str) -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute("DROP TABLE IF EXISTS document_chunk_pgvector_index")
    op.execute(
        f"""
        CREATE TABLE document_chunk_pgvector_index (
            chunk_id VARCHAR(36) PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
            document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            embedding {vector_type} NOT NULL,
            parse_version VARCHAR(128) NOT NULL DEFAULT 'legacy'
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_document_chunk_pgvector_index_document_id "
        "ON document_chunk_pgvector_index (document_id)"
    )
    op.execute(
        "CREATE INDEX ix_document_chunk_pgvector_index_document_parse_version "
        "ON document_chunk_pgvector_index (document_id, parse_version)"
    )


def upgrade() -> None:
    _replace_vector_table("vector(1024)")


def downgrade() -> None:
    _replace_vector_table("vector(2560)")
