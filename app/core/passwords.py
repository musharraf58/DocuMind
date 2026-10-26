"""The password policy, applied when a password is SET (registration, change).

Existing passwords keep working at login: refusing to let someone in because
the rules changed after they registered would lock them out for nothing.

Length is what matters most: each extra character multiplies the guesses
needed. The maximum stops a different attack: hashing a megabyte "password"
with argon2 is an easy way to burn our CPU. The blocklist stops the guesses
attackers try first; a full breach-corpus check (e.g. haveibeenpwned) would
be stronger but means a network call on every registration.
"""

from functools import lru_cache
from pathlib import Path

MIN_LENGTH = 10
MAX_LENGTH = 128

_BLOCKLIST_FILE = Path(__file__).with_name("common_passwords.txt")


@lru_cache
def common_passwords() -> frozenset[str]:
    lines = _BLOCKLIST_FILE.read_text(encoding="utf-8").splitlines()
    return frozenset(
        line.strip().lower() for line in lines if line.strip() and not line.startswith("#")
    )


def password_problem(password: str, email: str | None = None) -> str | None:
    """What's wrong with this new password, or None if it's acceptable.
    The message is shown to the user, so it explains the rule."""
    if len(password) < MIN_LENGTH:
        return f"Password must be at least {MIN_LENGTH} characters"
    if len(password) > MAX_LENGTH:
        return f"Password must be at most {MAX_LENGTH} characters"
    lowered = password.strip().lower()
    if lowered in common_passwords():
        return "This password is too common; choose a less predictable one"
    if email:
        email = email.strip().lower()
        if lowered in {email, email.split("@", 1)[0]}:
            return "Password must not be your email address"
    return None
