# mypy: disable-error-code="no-untyped-def,method-assign"
"""Tests for the dependency injection container."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from blacki.container import (
    AppContainer,
    close_container,
    get_container,
    init_container,
    reset_container_for_tests,
    set_container,
    set_container_from_connection,
)


class TestContainerGlobals:
    """Tests for container global state management."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container_for_tests()

    def test_get_container_raises_if_not_initialized(self) -> None:
        """Should raise RuntimeError if container not set."""
        reset_container_for_tests()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_container()

    def test_set_and_get_container(self) -> None:
        """Should set and return the container."""
        mock_container = MagicMock(spec=AppContainer)
        set_container(mock_container)

        result = get_container()
        assert result is mock_container

    def test_set_container_none(self) -> None:
        """Should allow setting container to None."""
        mock_container = MagicMock(spec=AppContainer)
        set_container(mock_container)
        set_container(None)

        with pytest.raises(RuntimeError, match="not initialized"):
            get_container()

    def test_reset_container_for_tests(self) -> None:
        """Should clear container reference without closing."""
        mock_container = MagicMock(spec=AppContainer)
        set_container(mock_container)
        reset_container_for_tests()

        with pytest.raises(RuntimeError, match="not initialized"):
            get_container()


class TestSetContainerFromConnection:
    """Tests for set_container_from_connection function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container_for_tests()

    @pytest.mark.asyncio
    async def test_creates_container_from_connection(self) -> None:
        """Should create and set container from existing connection."""
        conn = await aiosqlite.connect(":memory:", isolation_level=None)
        try:
            container = set_container_from_connection(conn)

            assert container.conn is conn
            assert get_container() is container
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_creates_container_with_custom_lock(self) -> None:
        """Should use provided lock."""
        conn = await aiosqlite.connect(":memory:", isolation_level=None)
        try:
            custom_lock = asyncio.Lock()
            container = set_container_from_connection(conn, lock=custom_lock)

            assert container.lock is custom_lock
        finally:
            await conn.close()


class TestCloseContainer:
    """Tests for close_container function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container_for_tests()

    @pytest.mark.asyncio
    async def test_close_container_closes_and_clears(self) -> None:
        """Should close container and clear reference."""
        mock_container = MagicMock(spec=AppContainer)
        mock_container.close = AsyncMock()
        set_container(mock_container)

        await close_container()

        mock_container.close.assert_called_once()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_container()

    @pytest.mark.asyncio
    async def test_close_container_when_none(self) -> None:
        """Should do nothing if container is None."""
        reset_container_for_tests()
        await close_container()


class TestInitContainer:
    """Tests for init_container function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container_for_tests()

    @pytest.mark.asyncio
    async def test_init_container_creates_and_sets(self) -> None:
        """Should create container and set global reference."""
        with patch.object(
            AppContainer, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_container = MagicMock(spec=AppContainer)
            mock_create.return_value = mock_container

            result = await init_container("/tmp/test.db")

            mock_create.assert_called_once_with("/tmp/test.db")
            assert result is mock_container
            assert get_container() is mock_container


class TestAppContainer:
    """Tests for AppContainer class."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container_for_tests()

    @pytest.fixture
    async def conn(self):
        """Create an in-memory SQLite connection for testing."""
        conn = await aiosqlite.connect(":memory:", isolation_level=None)
        yield conn
        await conn.close()

    @pytest.fixture
    def lock(self) -> asyncio.Lock:
        """Create a lock for write operations."""
        return asyncio.Lock()

    @pytest.mark.asyncio
    async def test_container_properties_lazy_instantiate(self, conn, lock) -> None:
        """Should lazily instantiate storage on first access."""
        container = AppContainer(conn=conn, _lock=lock)

        assert container._reminder_storage is None
        storage = container.reminder_storage
        assert container._reminder_storage is storage

    @pytest.mark.asyncio
    async def test_calorie_storage_property(self, conn, lock) -> None:
        """Should lazily instantiate calorie storage."""
        container = AppContainer(conn=conn, _lock=lock)

        storage = container.calorie_storage
        assert storage is not None
        assert container._calorie_storage is storage

    @pytest.mark.asyncio
    async def test_workout_storage_property(self, conn, lock) -> None:
        """Should lazily instantiate workout storage."""
        container = AppContainer(conn=conn, _lock=lock)

        storage = container.workout_storage
        assert storage is not None
        assert container._workout_storage is storage

    @pytest.mark.asyncio
    async def test_preferences_storage_property(self, conn, lock) -> None:
        """Should lazily instantiate preferences storage."""
        container = AppContainer(conn=conn, _lock=lock)

        storage = container.preferences_storage
        assert storage is not None
        assert container._preferences_storage is storage

    @pytest.mark.asyncio
    async def test_declarative_db_storage_property(self, conn, lock) -> None:
        """Should lazily instantiate declarative DB storage."""
        container = AppContainer(conn=conn, _lock=lock)

        storage = container.declarative_db_storage
        assert storage is not None
        assert container._declarative_db_storage is storage

    @pytest.mark.asyncio
    async def test_close_closes_connection_and_storages(self, conn, lock) -> None:
        """Should close connection and all storage instances."""
        container = AppContainer(conn=conn, _lock=lock)

        reminder = container.reminder_storage
        reminder.close = AsyncMock()

        declarative_db = container.declarative_db_storage
        declarative_db.close = AsyncMock()

        await container.close()

        reminder.close.assert_called_once()
        declarative_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_storages_resets_references(self, conn, lock) -> None:
        """Should reset storage references after close."""
        container = AppContainer(conn=conn, _lock=lock)

        _ = container.reminder_storage
        _ = container.calorie_storage
        _ = container.workout_storage
        _ = container.preferences_storage
        _ = container.declarative_db_storage
        user_files = container.user_file_storage
        user_files.close = AsyncMock()

        await container._close_storages()

        assert container._reminder_storage is None
        assert container._calorie_storage is None
        assert container._workout_storage is None
        assert container._preferences_storage is None
        assert container._declarative_db_storage is None
        assert container._user_file_storage is None
        user_files.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_storages_partial(self, conn, lock) -> None:
        """Should handle partial storage initialization."""
        container = AppContainer(conn=conn, _lock=lock)

        _ = container.calorie_storage
        _ = container.workout_storage
        _ = container.preferences_storage
        _ = container.declarative_db_storage

        await container._close_storages()

        assert container._reminder_storage is None
        assert container._calorie_storage is None
        assert container._workout_storage is None
        assert container._preferences_storage is None
        assert container._declarative_db_storage is None
        assert container._workout_storage is None
        assert container._preferences_storage is None

    @pytest.mark.asyncio
    async def test_initialize_all_storages(self, conn, lock) -> None:
        """Should initialize all storage instances."""
        container = AppContainer(conn=conn, _lock=lock)

        reminder = container.reminder_storage
        calorie = container.calorie_storage
        workout = container.workout_storage
        preferences = container.preferences_storage
        user_files = container.user_file_storage

        reminder.initialize = AsyncMock()
        calorie.initialize = AsyncMock()
        workout.initialize = AsyncMock()
        preferences.initialize = AsyncMock()
        user_files.initialize = AsyncMock()

        await container.initialize_all_storages()

        reminder.initialize.assert_called_once()
        calorie.initialize.assert_called_once()
        workout.initialize.assert_called_once()
        preferences.initialize.assert_called_once()
        user_files.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_creates_container_with_connection(self) -> None:
        """Should create container with SQLite connection."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            container = await AppContainer.create(db_path)

            assert container.conn is not None
            assert container.lock is not None

            await container.close()
