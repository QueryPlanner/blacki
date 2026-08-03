"""Persistent storage for user preferences backed by SQLite."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from blacki.storage.base import SqlStorage
from blacki.utils.timezone import now_utc

if TYPE_CHECKING:
    import asyncio

    import aiosqlite

logger = logging.getLogger(__name__)


class SqlitePreferencesStorage(SqlStorage):
    """Storage for user preferences using SQLite via aiosqlite."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        super().__init__(conn, lock)

    async def _create_tables(self) -> None:
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
        """)

    async def get(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        row = await self._fetch_one(
            "SELECT value FROM user_preferences WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        if row is None:
            return default
        return json.loads(row["value"])

    async def set(self, user_id: str, key: str, value: Any) -> None:
        """Set a preference value."""
        now = now_utc().isoformat(timespec="seconds")
        value_json = json.dumps(value)
        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO user_preferences (user_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (user_id, key) DO UPDATE
                SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (user_id, key, value_json, now),
            )
        logger.info("Updated preference %s for user %s", key, user_id)

    async def update_dict(
        self,
        user_id: str,
        key: str,
        updates: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically merge JSON object fields into one preference.

        The read, merge, and upsert occur under the storage write lock so two
        concurrent Telegram callbacks cannot overwrite each other's fields.
        A non-object existing value is treated as an empty object, allowing a
        preference key to migrate safely from an older scalar representation.
        """
        now = now_utc().isoformat(timespec="seconds")
        async with self._lock:
            row = await self._fetch_one(
                "SELECT value FROM user_preferences WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
            current: dict[str, Any] = {}
            if row is not None:
                try:
                    decoded = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    decoded = None
                if isinstance(decoded, dict):
                    current.update(decoded)

            current.update(dict(updates))
            await self._conn.execute(
                """
                INSERT INTO user_preferences (user_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (user_id, key) DO UPDATE
                SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (user_id, key, json.dumps(current), now),
            )

        logger.info("Updated preference fields %s for user %s", key, user_id)
        return current

    async def delete(self, user_id: str, key: str) -> bool:
        """Delete a preference."""
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM user_preferences WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
            return cursor.rowcount > 0


_storage: SqlitePreferencesStorage | None = None


def get_preferences_storage() -> SqlitePreferencesStorage:
    """Return the process-wide singleton SqlitePreferencesStorage instance.

    Uses the AppContainer for dependency injection.
    """
    from blacki.container import get_container

    container = get_container()
    storage = container.preferences_storage
    if not storage.is_initialized:
        raise RuntimeError(
            "Preferences storage not initialized. Call storage.initialize() first."
        )
    return storage
