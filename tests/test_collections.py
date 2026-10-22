import uuid

from httpx import AsyncClient

from tests.helpers import auth_headers, error_body


async def _create(client: AsyncClient, headers: dict, name: str = "Docs", **extra) -> dict:
    response = await client.post("/collections", json={"name": name, **extra}, headers=headers)
    assert response.status_code == 201
    return response.json()


async def test_collection_endpoints_require_auth(client: AsyncClient) -> None:
    some_id = uuid.uuid4()

    assert (await client.post("/collections", json={"name": "x"})).status_code == 401
    assert (await client.get("/collections")).status_code == 401
    assert (await client.get(f"/collections/{some_id}")).status_code == 401
    assert (await client.patch(f"/collections/{some_id}", json={"name": "x"})).status_code == 401
    assert (await client.delete(f"/collections/{some_id}")).status_code == 401


async def test_full_crud_lifecycle(client: AsyncClient) -> None:
    headers = await auth_headers(client, "alice@example.com")

    # Create
    created = await _create(client, headers, "Handbook", description="HR docs")
    assert created["name"] == "Handbook"
    assert created["description"] == "HR docs"
    assert set(created) == {"id", "name", "description", "created_at"}
    url = f"/collections/{created['id']}"

    # Read one
    fetched = await client.get(url, headers=headers)
    assert fetched.status_code == 200
    assert fetched.json() == created

    # Update only the name; description must be untouched
    patched = await client.patch(url, json={"name": "Handbook v2"}, headers=headers)
    assert patched.status_code == 200
    assert patched.json()["name"] == "Handbook v2"
    assert patched.json()["description"] == "HR docs"

    # Explicitly clear the description
    cleared = await client.patch(url, json={"description": None}, headers=headers)
    assert cleared.json()["description"] is None
    assert cleared.json()["name"] == "Handbook v2"

    # Delete, then it is gone
    assert (await client.delete(url, headers=headers)).status_code == 204
    assert (await client.get(url, headers=headers)).status_code == 404
    assert (await client.get("/collections", headers=headers)).json() == []


async def test_validation_errors(client: AsyncClient) -> None:
    headers = await auth_headers(client, "alice@example.com")
    created = await _create(client, headers)
    url = f"/collections/{created['id']}"

    assert (
        await client.post("/collections", json={"name": ""}, headers=headers)
    ).status_code == 422
    assert (await client.post("/collections", json={}, headers=headers)).status_code == 422
    assert (await client.patch(url, json={"name": None}, headers=headers)).status_code == 422
    assert (await client.get("/collections/not-a-uuid", headers=headers)).status_code == 422


async def test_list_pagination(client: AsyncClient) -> None:
    headers = await auth_headers(client, "alice@example.com")
    for i in range(5):
        await _create(client, headers, f"c{i}")

    everything = (await client.get("/collections", headers=headers)).json()
    page1 = (await client.get("/collections?limit=2&offset=0", headers=headers)).json()
    page2 = (await client.get("/collections?limit=2&offset=2", headers=headers)).json()
    page3 = (await client.get("/collections?limit=2&offset=4", headers=headers)).json()

    assert len(everything) == 5
    assert [len(page1), len(page2), len(page3)] == [2, 2, 1]
    # Pages are consecutive, non-overlapping slices of the full ordered list.
    assert page1 + page2 + page3 == everything
    # Newest first
    assert everything[0]["name"] == "c4"


async def test_pagination_bounds_are_validated(client: AsyncClient) -> None:
    headers = await auth_headers(client, "alice@example.com")

    for query in ("limit=0", "limit=101", "offset=-1"):
        assert (await client.get(f"/collections?{query}", headers=headers)).status_code == 422


async def test_users_cannot_see_or_touch_each_others_collections(client: AsyncClient) -> None:
    alice = await auth_headers(client, "alice@example.com")
    bob = await auth_headers(client, "bob@example.com")
    alices = await _create(client, alice, "Alice private", description="secret")
    url = f"/collections/{alices['id']}"

    # Bob gets 404 (not 403) on get, patch and delete
    assert (await client.get(url, headers=bob)).status_code == 404
    assert (await client.patch(url, json={"name": "pwned"}, headers=bob)).status_code == 404
    assert (await client.delete(url, headers=bob)).status_code == 404

    # ...and the response is indistinguishable from a collection that never existed
    missing = await client.get(f"/collections/{uuid.uuid4()}", headers=bob)
    forbidden_looking = await client.get(url, headers=bob)
    assert forbidden_looking.status_code == missing.status_code == 404
    assert error_body(forbidden_looking) == error_body(missing)

    # Bob's list doesn't include it
    assert (await client.get("/collections", headers=bob)).json() == []

    # Alice's collection is completely unchanged by Bob's attempts
    still_there = await client.get(url, headers=alice)
    assert still_there.status_code == 200
    assert still_there.json() == alices


async def test_each_user_lists_only_their_own(client: AsyncClient) -> None:
    alice = await auth_headers(client, "alice@example.com")
    bob = await auth_headers(client, "bob@example.com")
    await _create(client, alice, "A1")
    await _create(client, alice, "A2")
    await _create(client, bob, "B1")

    alice_names = {c["name"] for c in (await client.get("/collections", headers=alice)).json()}
    bob_names = {c["name"] for c in (await client.get("/collections", headers=bob)).json()}

    assert alice_names == {"A1", "A2"}
    assert bob_names == {"B1"}
