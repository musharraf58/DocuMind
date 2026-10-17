"""chunk embeddings

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-14 23:28:31.300658

Why the dimension is fixed here: a pgvector column stores vectors of exactly
one size, and the size is part of the column type (vector(1024)). Postgres
rejects any vector of another size. 1024 is what voyage-4-lite produces by
default and what EMBEDDING_DIMENSIONS must be set to.

Changing the embedding model (or its output size) therefore needs:
  1. a new migration that changes the column type to vector(N) and rebuilds
     the HNSW index (or adds a second column, to switch over without downtime);
  2. re-embedding every chunk with the new model, because vectors from
     different models can't be compared (see chunks.embedding_model);
  3. updating EMBEDDING_MODEL / EMBEDDING_DIMENSIONS and
     EMBEDDING_COLUMN_DIMENSIONS in app/models/chunk.py.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector


revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embedding", Vector(1024), nullable=True))
    op.add_column("chunks", sa.Column("embedding_model", sa.Text(), nullable=True))
    # HNSW: approximate nearest-neighbour search. vector_cosine_ops means it
    # can only serve queries that order by cosine distance ("<=>").
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "embedding")
