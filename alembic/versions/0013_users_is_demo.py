"""users.is_demo

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-27 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The shared, read-only demo account (python -m scripts.seed_demo). It
    # may ask, search and read but not change anything, and retention never
    # deletes it. Existing rows are ordinary accounts.
    op.add_column(
        "users",
        sa.Column("is_demo", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("users", "is_demo")
