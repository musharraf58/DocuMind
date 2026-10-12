from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.services.storage import LocalStorageBackend


async def _stream(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


async def _read_all(stream: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream])


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


async def test_save_open_delete_roundtrip(tmp_path: Path) -> None:
    storage = LocalStorageBackend(tmp_path)

    written = await storage.save("owner/doc", _stream(b"hello ", b"world"))

    assert written == 11
    assert await _read_all(await storage.open("owner/doc")) == b"hello world"
    await storage.delete("owner/doc")
    assert _files(tmp_path) == []


async def test_delete_missing_key_is_not_an_error(tmp_path: Path) -> None:
    await LocalStorageBackend(tmp_path).delete("owner/never-existed")


async def test_open_missing_key_raises_immediately(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await LocalStorageBackend(tmp_path).open("owner/missing")


async def test_failed_stream_leaves_nothing_behind(tmp_path: Path) -> None:
    storage = LocalStorageBackend(tmp_path)

    async def failing() -> AsyncIterator[bytes]:
        yield b"partial data"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await storage.save("owner/doc", failing())

    assert _files(tmp_path) == []


@pytest.mark.parametrize("key", ["../escape", "owner/../../escape", "/etc/passwd", ""])
async def test_keys_cannot_escape_the_storage_root(tmp_path: Path, key: str) -> None:
    root = tmp_path / "root"
    storage = LocalStorageBackend(root)

    with pytest.raises(ValueError):
        await storage.save(key, _stream(b"x"))

    assert not (tmp_path / "escape").exists()
