"""usage events

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-22 16:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

usage_kind = postgresql.ENUM(
    "embed_document", "embed_query", "llm_answer", "llm_rewrite", name="usage_kind"
)


def upgrade() -> None:
    # One row per successful provider request: counts and ids only, never any
    # question, answer or document text. document_id / conversation_id have no
    # foreign keys on purpose: the history outlives the document.
    usage_kind.create(op.get_bind())
    op.create_table(
        "usage_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "kind",
            postgresql.ENUM(name="usage_kind", create_type=False),
            nullable=False,
        ),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_usage_events_user_id_created_at", "usage_events", ["user_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_usage_events_user_id_created_at", table_name="usage_events")
    op.drop_table("usage_events")
    usage_kind.drop(op.get_bind())
