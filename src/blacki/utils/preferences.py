import asyncio
import json
import logging
from typing import Any

import asyncpg  # type: ignore[import-untyped]

from blacki.utils.timezone import now_utc

logger = logging.getLogger(__name__)


class PostgresPreferencesStorage:
    """Storage for user preferences using Postgres via asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._lock = asyncio.Lock()
        self._schema_ready = False

    async def initialize(self) -> None:
        """Ensure schema is created."""
        async with self._lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as conn:
                await self._create_tables(conn)
            self._schema_ready = True
            logger.info("Preferences storage schema ready (Postgres)")

    async def close(self) -> None:
        """Mark uninitialized."""
        async with self._lock:
            self._schema_ready = False

    async def _create_tables(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id    TEXT  NOT NULL,
                key        TEXT  NOT NULL,
                value      JSONB NOT NULL,
                updated_at TEXT  NOT NULL,
                PRIMARY KEY (user_id, key)
            )
        """)

    async def get(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        row = await self._pool.fetchrow(
            "SELECT value FROM user_preferences WHERE user_id = $1 AND key = $2",
            user_id,
            key,
        )
        if row is None:
            return default
        # asyncpg returns JSONB as a JSON string when fetched normally
        # but we can parse it
        return json.loads(row["value"])

    async def set(self, user_id: str, key: str, value: Any) -> None:
        """Set a preference value."""
        now = now_utc().isoformat(timespec="seconds")
        value_json = json.dumps(value)
        await self._pool.execute(
            """
            INSERT INTO user_preferences (user_id, key, value, updated_at)
            VALUES ($1, $2, $3::jsonb, $4)
            ON CONFLICT (user_id, key) DO UPDATE
            SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
            """,
            user_id,
            key,
            value_json,
            now,
        )
        logger.info("Updated preference %s for user %s", key, user_id)

    async def delete(self, user_id: str, key: str) -> bool:
        """Delete a preference."""
        result = await self._pool.execute(
            "DELETE FROM user_preferences WHERE user_id = $1 AND key = $2",
            user_id,
            key,
        )
        return bool(result == "DELETE 1")


_storage: PostgresPreferencesStorage | None = None


def get_preferences_storage() -> PostgresPreferencesStorage:
    """Return the process-wide singleton PostgresPreferencesStorage instance."""
    global _storage
    if _storage is None:
        raise RuntimeError(
            "Preferences storage not initialized. "
            "Call init_preferences_storage() first."
        )
    return _storage


async def init_preferences_storage(pool: asyncpg.Pool) -> PostgresPreferencesStorage:
    """Initialize the preferences storage with a Postgres pool."""
    global _storage
    _storage = PostgresPreferencesStorage(pool)
    await _storage.initialize()
    return _storage


async def close_preferences_storage() -> None:
    """Close the singleton preferences storage."""
    global _storage
    if _storage is not None:
        await _storage.close()
        _storage = None
