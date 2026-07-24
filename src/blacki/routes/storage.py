"""User-scoped persistence for common Google Maps routes."""

from __future__ import annotations

import sqlite3
import unicodedata
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from blacki.storage.base import SqlStorage

if TYPE_CHECKING:
    import asyncio

    import aiosqlite

SAVED_ROUTE_UPDATE_COLUMNS = frozenset(
    {
        "name",
        "normalized_name",
        "origin_place_id",
        "destination_place_id",
        "origin_label",
        "destination_label",
        "travel_mode",
        "avoid_tolls",
        "avoid_highways",
        "avoid_ferries",
        "updated_at",
    }
)


class DuplicateRouteNameError(ValueError):
    """A user already has a route with the normalized name."""


class SavedRouteLimitError(ValueError):
    """A user has reached the configured saved-route limit."""


class SavedRoute(BaseModel):
    """One reusable route containing place IDs rather than raw addresses."""

    id: int | None = None
    user_id: str
    name: str
    normalized_name: str
    origin_place_id: str
    destination_place_id: str
    origin_label: str
    destination_label: str
    travel_mode: str
    avoid_tolls: bool
    avoid_highways: bool
    avoid_ferries: bool
    created_at: str
    updated_at: str


def normalize_route_name(value: str) -> str:
    """Normalize a user-visible route name for owner-scoped uniqueness."""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


class SqliteSavedRouteStorage(SqlStorage):
    """SQLite storage for user-owned common routes."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        super().__init__(conn, lock)

    async def _create_tables(self) -> None:
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                origin_place_id TEXT NOT NULL,
                destination_place_id TEXT NOT NULL,
                origin_label TEXT NOT NULL,
                destination_label TEXT NOT NULL,
                travel_mode TEXT NOT NULL,
                avoid_tolls INTEGER NOT NULL DEFAULT 0,
                avoid_highways INTEGER NOT NULL DEFAULT 0,
                avoid_ferries INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (user_id, normalized_name)
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_saved_routes_user
            ON saved_routes (user_id, normalized_name)
        """)

    async def create_route(self, route: SavedRoute, limit: int) -> SavedRoute:
        """Atomically enforce the per-user limit and insert a route."""
        async with self._lock:
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
                count = await self._count_for_user(route.user_id)
                if count >= limit:
                    raise SavedRouteLimitError(
                        f"You can save at most {limit} common routes."
                    )
                route_id = await self._execute(
                    """
                    INSERT INTO saved_routes (
                        user_id, name, normalized_name, origin_place_id,
                        destination_place_id, origin_label, destination_label,
                        travel_mode, avoid_tolls, avoid_highways, avoid_ferries,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        route.user_id,
                        route.name,
                        route.normalized_name,
                        route.origin_place_id,
                        route.destination_place_id,
                        route.origin_label,
                        route.destination_label,
                        route.travel_mode,
                        int(route.avoid_tolls),
                        int(route.avoid_highways),
                        int(route.avoid_ferries),
                        route.created_at,
                        route.updated_at,
                    ),
                    use_lock=False,
                )
                await self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise DuplicateRouteNameError(
                    f"A common route named '{route.name}' already exists."
                ) from exc
            finally:
                if self._conn.in_transaction:
                    await self._conn.rollback()
        return route.model_copy(update={"id": int(route_id)})

    async def list_routes(self, user_id: str) -> list[SavedRoute]:
        """List a user's routes without exposing another owner."""
        rows = await self._fetch_all(
            """
            SELECT * FROM saved_routes
            WHERE user_id = ?
            ORDER BY normalized_name ASC
            """,
            (user_id,),
        )
        return [self._row_to_route(row) for row in rows]

    async def get_route(self, user_id: str, reference: str) -> SavedRoute | None:
        """Resolve an owner-qualified route by ``id:N`` or normalized name."""
        normalized_reference = reference.strip()
        if normalized_reference.casefold().startswith("id:"):
            route_id_text = normalized_reference.partition(":")[2].strip()
            if route_id_text.isdecimal():
                row = await self._fetch_one(
                    "SELECT * FROM saved_routes WHERE user_id = ? AND id = ?",
                    (user_id, int(route_id_text)),
                )
                return self._row_to_route(row) if row else None
        row = await self._fetch_one(
            """
            SELECT * FROM saved_routes
            WHERE user_id = ? AND normalized_name = ?
            """,
            (user_id, normalize_route_name(normalized_reference)),
        )
        return self._row_to_route(row) if row else None

    async def update_route(
        self,
        user_id: str,
        route_id: int,
        values: dict[str, Any],
    ) -> SavedRoute | None:
        """Update only an owner-qualified route."""
        if not values:
            return await self.get_route(user_id, f"id:{route_id}")
        if not values.keys() <= SAVED_ROUTE_UPDATE_COLUMNS:
            raise ValueError("Unsupported saved-route update field.")
        assignments = ", ".join(f"{column} = ?" for column in values)
        params = (*values.values(), user_id, route_id)
        async with self._lock:
            try:
                cursor = await self._conn.execute(
                    f"""
                    UPDATE saved_routes SET {assignments}
                    WHERE user_id = ? AND id = ?
                    """,  # noqa: S608 - columns are validated against a fixed allowlist
                    params,
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateRouteNameError(
                    "A common route with that name already exists."
                ) from exc
        if cursor.rowcount == 0:
            return None
        return await self.get_route(user_id, f"id:{route_id}")

    async def delete_route(self, user_id: str, route_id: int) -> bool:
        """Delete only an owner-qualified route."""
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM saved_routes WHERE user_id = ? AND id = ?",
                (user_id, route_id),
            )
        return cursor.rowcount > 0

    async def _count_for_user(self, user_id: str) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM saved_routes WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _row_to_route(row: dict[str, Any]) -> SavedRoute:
        return SavedRoute(
            id=int(row["id"]),
            user_id=row["user_id"],
            name=row["name"],
            normalized_name=row["normalized_name"],
            origin_place_id=row["origin_place_id"],
            destination_place_id=row["destination_place_id"],
            origin_label=row["origin_label"],
            destination_label=row["destination_label"],
            travel_mode=row["travel_mode"],
            avoid_tolls=bool(row["avoid_tolls"]),
            avoid_highways=bool(row["avoid_highways"]),
            avoid_ferries=bool(row["avoid_ferries"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def get_saved_route_storage() -> SqliteSavedRouteStorage:
    """Return initialized saved-route storage from the app container."""
    from blacki.container import get_container

    storage = get_container().saved_route_storage
    if not storage.is_initialized:
        raise RuntimeError(
            "Saved route storage not initialized. Call storage.initialize() first."
        )
    return storage
