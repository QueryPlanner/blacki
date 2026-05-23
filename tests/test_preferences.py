# mypy: disable-error-code="no-untyped-def"
"""Unit tests for preferences storage."""

import asyncio

import aiosqlite
import pytest

from blacki.utils.preferences import SqlitePreferencesStorage


@pytest.fixture
async def conn():
    """Create an in-memory SQLite connection for testing."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


@pytest.fixture
def lock():
    """Create a lock for write operations."""
    return asyncio.Lock()


@pytest.fixture
async def storage(conn, lock):
    """Create a storage instance with the test connection."""
    storage = SqlitePreferencesStorage(conn, lock)
    await storage.initialize()
    yield storage
    await storage.close()


class TestSqlitePreferencesStorage:
    """Tests for SqlitePreferencesStorage."""

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, conn, lock) -> None:
        """Should create tables on initialization."""
        storage = SqlitePreferencesStorage(conn, lock)
        await storage.initialize()

        assert storage.is_initialized is True

        cursor = await conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='user_preferences'
            """
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_get_existing(self, storage) -> None:
        """Should get an existing preference."""
        await storage.set("user1", "workout_split", {"monday": "push"})

        result = await storage.get("user1", "workout_split")

        assert result == {"monday": "push"}

    @pytest.mark.asyncio
    async def test_get_not_found_returns_default(self, storage) -> None:
        """Should return default when preference not found."""
        result = await storage.get("user1", "calorie_goal", 2000)

        assert result == 2000

    @pytest.mark.asyncio
    async def test_set(self, storage) -> None:
        """Should set a preference."""
        await storage.set("user1", "calorie_goal", 2500)

        result = await storage.get("user1", "calorie_goal")
        assert result == 2500

    @pytest.mark.asyncio
    async def test_set_updates_existing(self, storage) -> None:
        """Should update an existing preference."""
        await storage.set("user1", "calorie_goal", 2000)
        await storage.set("user1", "calorie_goal", 2500)

        result = await storage.get("user1", "calorie_goal")
        assert result == 2500

    @pytest.mark.asyncio
    async def test_delete_success(self, storage) -> None:
        """Should delete a preference."""
        await storage.set("user1", "calorie_goal", 2500)

        result = await storage.delete("user1", "calorie_goal")

        assert result is True
        value = await storage.get("user1", "calorie_goal")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, storage) -> None:
        """Should return False when deleting non-existent preference."""
        result = await storage.delete("user1", "calorie_goal")

        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_keys_per_user(self, storage) -> None:
        """Should handle multiple keys per user."""
        await storage.set("user1", "calorie_goal", 2500)
        await storage.set("user1", "workout_split", {"monday": "push"})
        await storage.set("user1", "timezone", "America/New_York")

        assert await storage.get("user1", "calorie_goal") == 2500
        assert await storage.get("user1", "workout_split") == {"monday": "push"}
        assert await storage.get("user1", "timezone") == "America/New_York"

    @pytest.mark.asyncio
    async def test_multiple_users_isolated(self, storage) -> None:
        """Should isolate preferences between users."""
        await storage.set("user1", "calorie_goal", 2000)
        await storage.set("user2", "calorie_goal", 2500)

        assert await storage.get("user1", "calorie_goal") == 2000
        assert await storage.get("user2", "calorie_goal") == 2500

    @pytest.mark.asyncio
    async def test_complex_value(self, storage) -> None:
        """Should store and retrieve complex values."""
        complex_value = {
            "split_name": "push",
            "exercises": ["bench press", "shoulder press"],
            "rest_days": ["wednesday", "sunday"],
        }
        await storage.set("user1", "workout_split", complex_value)

        result = await storage.get("user1", "workout_split")
        assert result == complex_value


class TestGetPreferencesStorage:
    """Tests for get_preferences_storage function."""

    @pytest.mark.asyncio
    async def test_get_preferences_storage_raises_when_not_initialized(
        self, conn, lock
    ) -> None:
        """Should raise RuntimeError when storage is not initialized."""
        from blacki.container import (
            reset_container_for_tests,
            set_container_from_connection,
        )
        from blacki.utils.preferences import get_preferences_storage

        set_container_from_connection(conn, lock)

        with pytest.raises(RuntimeError, match="Preferences storage not initialized"):
            get_preferences_storage()

        reset_container_for_tests()

    @pytest.mark.asyncio
    async def test_get_preferences_storage_returns_storage_when_initialized(
        self, conn, lock
    ) -> None:
        """Should return storage when initialized."""
        from blacki.container import (
            reset_container_for_tests,
            set_container_from_connection,
        )
        from blacki.utils.preferences import get_preferences_storage

        container = set_container_from_connection(conn, lock)
        await container.preferences_storage.initialize()

        result = get_preferences_storage()

        assert result is container.preferences_storage

        reset_container_for_tests()
