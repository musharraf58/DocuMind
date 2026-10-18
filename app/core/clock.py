from datetime import UTC, datetime


def utcnow() -> datetime:
    """The current time. Always call it as clock.utcnow(), so tests can replace
    it in one place and run long jobs on a fake clock."""
    return datetime.now(UTC)
