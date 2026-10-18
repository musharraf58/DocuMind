import fakeredis
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError

from app.core.database import get_db
from app.core.redis import get_redis
from app.main import app


async def test_health_checks_database_and_redis(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "redis": "ok"}


async def test_health_is_503_when_redis_is_down(client: AsyncClient) -> None:
    server = fakeredis.FakeServer()
    server.connected = False
    app.dependency_overrides[get_redis] = lambda: fakeredis.aioredis.FakeRedis(server=server)

    response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "ok", "redis": "unavailable"}


async def test_health_is_503_when_the_database_is_down(client: AsyncClient) -> None:
    class DownSession:
        async def execute(self, statement):
            raise OperationalError("SELECT 1", {}, ConnectionRefusedError())

    async def down():
        yield DownSession()

    app.dependency_overrides[get_db] = down
    response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["database"] == "unavailable"
