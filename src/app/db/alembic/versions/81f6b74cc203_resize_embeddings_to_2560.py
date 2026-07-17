"""resize document embeddings to 2560 dimensions

Revision ID: 81f6b74cc203
Revises: 6c1a2d9e4b70
Create Date: 2026-07-17
"""

from typing import Sequence, Union

from alembic import op


revision: str = "81f6b74cc203"
down_revision: Union[str, Sequence[str], None] = "6c1a2d9e4b70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VECTOR_2560 = "vector(2560)"
VECTOR_4096 = "vector(4096)"


def _replace_vector_table(vector_type: str) -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS document_chunk_pgvector_index")
    op.execute(
        f"""
        CREATE TABLE document_chunk_pgvector_index (
            chunk_id VARCHAR(36) PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
            document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            embedding {vector_type} NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_document_chunk_pgvector_index_document_id "
        "ON document_chunk_pgvector_index (document_id)"
    )


def upgrade() -> None:
    _replace_vector_table(VECTOR_2560)


def downgrade() -> None:
    _replace_vector_table(VECTOR_4096)
