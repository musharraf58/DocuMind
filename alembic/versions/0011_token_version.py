"""users.token_version

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-22 16:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Copied into every access token ("ver"). Bumping it (on a password change)
    # makes every token issued before that moment invalid at once, without a
    # list of revoked tokens. Existing rows start at 0.
    op.add_column(
        "users",
        sa.Column("token_version", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("users", "token_version")
