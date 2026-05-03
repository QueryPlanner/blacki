import json
import logging
from typing import TYPE_CHECKING, Any

import asyncpg  # type: ignore[import-untyped]

from blacki.storage.base import PostgresStorage
from blacki.utils.timezone import now_utc

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PostgresPreferencesStorage(PostgresStorage):
    """Storage for user preferences using Postgres via asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        super().__init__(pool)

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
        value = row["value"]
        return json.loads(value) if isinstance(value, str) else value

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
    """Return the process-wide singleton PostgresPreferencesStorage instance.

    Uses the AppContainer for dependency injection.
    """
    from blacki.container import get_container

    container = get_container()
    storage = container.preferences_storage
    if not storage.is_initialized:
        raise RuntimeError(
            "Preferences storage not initialized. "
            "Call init_preferences_storage() first."
        )
    return storage


async def init_preferences_storage(pool: asyncpg.Pool) -> PostgresPreferencesStorage:
    """Initialize the preferences storage with a Postgres pool.

    Note: This function is provided for backward compatibility.
    Prefer using AppContainer directly for new code.
    """
    global _storage
    import blacki.container as container_module

    if container_module._container is None:  # pragma: no cover
        container_module.set_container_from_pool(pool)

    if _storage is not None:
        await _storage.close()
        _storage = None

    container = container_module._container
    if container is None:  # pragma: no cover
        raise RuntimeError("Container not initialized")
    if container._preferences_storage is not None:  # pragma: no cover
        await container._preferences_storage.close()

    storage = container.preferences_storage
    await storage.initialize()
    _storage = storage
    return storage


async def close_preferences_storage() -> None:
    """Close the singleton preferences storage.

    Note: This function is provided for backward compatibility.
    Prefer using AppContainer.close() for new code.
    """
    global _storage
    import blacki.container as container_module

    if container_module._container is not None:  # pragma: no cover
        container = container_module._container
        if container._preferences_storage is not None:
            await container._preferences_storage.close()
            container._preferences_storage = None
    _storage = None
