import logging

from pydantic import SecretStr

from app.core.config import get_settings
from app.services.users import invite_code_matches
from tests.helpers import PASSWORD

CODE = "larkspur-invite-7Qm2xk"


def set_mode(monkeypatch, mode: str, code: str = "") -> None:
    monkeypatch.setattr(get_settings(), "REGISTRATION_MODE", mode)
    monkeypatch.setattr(get_settings(), "INVITE_CODE", SecretStr(code))


async def register(client, email: str, invite_code: str | None = None):
    body = {"email": email, "password": PASSWORD}
    if invite_code is not None:
        body["invite_code"] = invite_code
    return await client.post("/auth/register", json=body)


def test_default_mode_is_invite():
    from app.core.config import Settings

    assert Settings.model_fields["REGISTRATION_MODE"].default == "invite"


async def test_open_mode(client, monkeypatch):
    set_mode(monkeypatch, "open")
    assert (await register(client, "open@example.com")).status_code == 201


async def test_closed_mode(client, monkeypatch):
    set_mode(monkeypatch, "closed", CODE)
    response = await register(client, "closed@example.com", CODE)  # even with the code
    assert response.status_code == 403
    assert response.json()["code"] == "registration_closed"


async def test_invite_mode_with_the_right_code(client, monkeypatch):
    set_mode(monkeypatch, "invite", CODE)
    assert (await register(client, "invited@example.com", CODE)).status_code == 201


async def test_invite_mode_with_a_wrong_or_missing_code(client, monkeypatch):
    set_mode(monkeypatch, "invite", CODE)
    for code in [None, "", "wrong-code", CODE.upper(), CODE + "x", CODE[:-1]]:
        response = await register(client, "uninvited@example.com", code)
        assert response.status_code == 403, code
        assert response.json()["code"] == "invalid_invite_code"
    # Nothing was created.
    assert (await register(client, "uninvited@example.com", CODE)).status_code == 201


async def test_invite_mode_without_a_configured_code_is_closed(client, monkeypatch):
    """An empty INVITE_CODE must not mean "an empty code gets in"."""
    set_mode(monkeypatch, "invite", "")
    for code in [None, ""]:
        response = await register(client, "nocode@example.com", code)
        assert response.status_code == 403
        assert response.json()["code"] == "registration_closed"


async def test_invite_code_is_never_logged_or_echoed(client, monkeypatch, caplog):
    set_mode(monkeypatch, "invite", CODE)
    with caplog.at_level(logging.DEBUG):
        wrong = await register(client, "leak@example.com", "guess-" + CODE[6:])
        right = await register(client, "leak2@example.com", CODE)
    assert CODE not in caplog.text
    assert CODE[6:] not in wrong.text
    assert CODE not in right.text
    assert CODE not in repr(get_settings())


def test_constant_time_comparison():
    assert invite_code_matches(CODE, CODE)
    assert not invite_code_matches("", CODE)
    assert not invite_code_matches(CODE, "")
    assert not invite_code_matches(CODE + " ", CODE)


async def test_mode_is_checked_before_creating_anything(client, monkeypatch, db):
    from sqlalchemy import func, select

    from app.models import User

    set_mode(monkeypatch, "closed")
    await register(client, "nobody@example.com")
    assert await db.scalar(select(func.count()).select_from(User)) == 0
