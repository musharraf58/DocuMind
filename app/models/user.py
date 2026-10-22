import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.collection import Collection


class User(Base):
    __tablename__ = "users"
    # Emails are normalized to lowercase in the service layer; this constraint
    # makes the database refuse anything else, even from buggy code paths.
    __table_args__ = (CheckConstraint("email = lower(email)", name="ck_users_email_lowercase"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    # Goes into every access token; bumped when the password changes, which
    # invalidates all tokens issued before (see get_current_user).
    token_version: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    collections: Mapped[list["Collection"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )
