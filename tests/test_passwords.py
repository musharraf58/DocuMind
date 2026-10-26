import pytest

from app.core.passwords import MAX_LENGTH, MIN_LENGTH, common_passwords, password_problem
from tests.helpers import PASSWORD, register


def test_good_password_passes():
    assert password_problem("correct-horse-battery", "alice@example.com") is None


@pytest.mark.parametrize(
    ("password", "expected"),
    [
        ("x" * (MIN_LENGTH - 1), "at least 10"),
        ("x" * (MAX_LENGTH + 1), "at most 128"),
        ("password123", "too common"),
        ("Password123", "too common"),  # case doesn't help
        ("  qwertyuiop  ", "too common"),  # nor surrounding spaces
        ("1234567890", "too common"),
        ("alice.smith@example.com", "email"),
        ("ALICE.SMITH", "email"),  # the address's local part
    ],
)
def test_bad_passwords(password, expected):
    problem = password_problem(password, "alice.smith@example.com")
    assert problem is not None
    assert expected in problem


def test_exact_limits_pass():
    assert password_problem("k" * MIN_LENGTH + "zq") is None
    assert password_problem("k" * (MAX_LENGTH - 2) + "zq") is None


def test_blocklist_loads_lowercased_without_comments():
    passwords = common_passwords()
    assert "password123" in passwords
    assert len(passwords) > 100
    assert all(p == p.lower() and not p.startswith("#") for p in passwords)


async def test_register_enforces_the_policy(client):
    for password in ["short-pw", "password1234", "bob@example.com", "x" * 129]:
        response = await register(client, "bob@example.com", password)
        assert response.status_code == 422, password
        errors = response.json()["errors"]
        assert errors[0]["loc"] == ["body", "password"]
        assert password not in response.text  # never echoed back
    assert (await register(client, "bob@example.com", PASSWORD)).status_code == 201


async def test_policy_message_reaches_the_user(client):
    response = await register(client, "carol@example.com", "qwerty1234")
    assert "too common" in response.json()["errors"][0]["msg"]


async def test_existing_short_password_still_logs_in(client, db):
    """The policy applies when a password is set, not at login."""
    from app.core.security import hash_password
    from app.models import User
    from tests.helpers import login

    db.add(User(email="old@example.com", hashed_password=hash_password("eightchr")))
    await db.commit()
    assert (await login(client, "old@example.com", "eightchr")).status_code == 200
