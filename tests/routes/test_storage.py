# mypy: disable-error-code="no-untyped-def"
"""Tests for owner-scoped saved-route persistence."""

import asyncio

import aiosqlite
import pytest

from blacki.container import reset_container_for_tests, set_container_from_connection
from blacki.routes.storage import (
    DuplicateRouteNameError,
    SavedRoute,
    SavedRouteLimitError,
    SqliteSavedRouteStorage,
    normalize_route_name,
)


@pytest.fixture
async def storage():
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    route_storage = SqliteSavedRouteStorage(conn, asyncio.Lock())
    await route_storage.initialize()
    yield route_storage
    await route_storage.close()
    await conn.close()


def _route(name: str = "Home Office", user_id: str = "user-1") -> SavedRoute:
    return SavedRoute(
        user_id=user_id,
        name=name,
        normalized_name=normalize_route_name(name),
        origin_place_id="origin-place",
        destination_place_id="destination-place",
        origin_label="Home",
        destination_label="Office",
        travel_mode="DRIVE",
        avoid_tolls=False,
        avoid_highways=True,
        avoid_ferries=False,
        created_at="2026-07-24T00:00:00+00:00",
        updated_at="2026-07-24T00:00:00+00:00",
    )


def test_normalize_route_name_handles_unicode_and_whitespace() -> None:
    assert normalize_route_name("  Ｈｏｍｅ   OFFICE ") == "home office"


@pytest.mark.asyncio
async def test_create_list_and_resolve_by_name_or_id(storage) -> None:
    saved = await storage.create_route(_route(), limit=2)

    assert saved.id == 1
    assert await storage.list_routes("other-user") == []
    assert (await storage.get_route("user-1", " HOME   office ")).id == 1
    assert (await storage.get_route("user-1", "id:1")).name == "Home Office"
    assert await storage.get_route("other-user", "id:1") is None
    assert await storage.get_route("user-1", "id:not-a-number") is None


@pytest.mark.asyncio
async def test_create_rejects_duplicate_and_limit_atomically(storage) -> None:
    await storage.create_route(_route(), limit=1)

    with pytest.raises(SavedRouteLimitError, match="at most 1"):
        await storage.create_route(_route("Gym"), limit=1)

    with pytest.raises(DuplicateRouteNameError, match="already exists"):
        await storage.create_route(_route(" HOME OFFICE "), limit=2)

    assert [route.name for route in await storage.list_routes("user-1")] == [
        "Home Office"
    ]


@pytest.mark.asyncio
async def test_update_is_owner_scoped_and_enforces_unique_names(storage) -> None:
    first = await storage.create_route(_route(), limit=3)
    second = await storage.create_route(_route("Gym"), limit=3)

    unchanged = await storage.update_route("user-1", first.id, {})
    assert unchanged.name == "Home Office"

    updated = await storage.update_route(
        "user-1",
        first.id,
        {
            "name": "Commute",
            "normalized_name": "commute",
            "avoid_tolls": 1,
        },
    )
    assert updated.name == "Commute"
    assert updated.avoid_tolls is True
    assert await storage.update_route("other-user", first.id, {"name": "Nope"}) is None

    with pytest.raises(DuplicateRouteNameError):
        await storage.update_route(
            "user-1",
            second.id,
            {"name": "COMMUTE", "normalized_name": "commute"},
        )

    with pytest.raises(ValueError, match="Unsupported"):
        await storage.update_route(
            "user-1",
            first.id,
            {"name = 'unsafe'": "value"},
        )


@pytest.mark.asyncio
async def test_delete_is_owner_scoped(storage) -> None:
    saved = await storage.create_route(_route(), limit=2)

    assert await storage.delete_route("other-user", saved.id) is False
    assert await storage.delete_route("user-1", saved.id) is True
    assert await storage.get_route("user-1", "Home Office") is None


@pytest.mark.asyncio
async def test_global_storage_accessor_requires_initialization() -> None:
    from blacki.routes.storage import get_saved_route_storage

    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    try:
        set_container_from_connection(conn)
        with pytest.raises(RuntimeError, match="not initialized"):
            get_saved_route_storage()
    finally:
        reset_container_for_tests()
        await conn.close()
