"""Tests for the dependency injection container."""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg  # type: ignore[import-untyped]
import pytest

from blacki.container import (
    AppContainer,
    close_container,
    get_container,
    init_container,
    reset_container_for_tests,
    set_container,
    set_container_from_pool,
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


class TestSetContainerFromPool:
    """Tests for set_container_from_pool function."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container_for_tests()

    def test_creates_container_from_pool(self) -> None:
        """Should create and set container from existing pool."""
        mock_pool = MagicMock(spec=asyncpg.Pool)
        container = set_container_from_pool(mock_pool)

        assert container.pool is mock_pool
        assert get_container() is container


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

            result = await init_container("postgres://localhost/test")

            mock_create.assert_called_once_with("postgres://localhost/test", 5)
            assert result is mock_container
            assert get_container() is mock_container

    @pytest.mark.asyncio
    async def test_init_container_with_custom_pool_size(self) -> None:
        """Should pass custom pool size to create."""
        with patch.object(
            AppContainer, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_container = MagicMock(spec=AppContainer)
            mock_create.return_value = mock_container

            await init_container("postgres://localhost/test", pool_size=10)

            mock_create.assert_called_once_with("postgres://localhost/test", 10)


class TestAppContainer:
    """Tests for AppContainer class."""

    def teardown_method(self) -> None:
        """Reset container after each test."""
        reset_container_for_tests()

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock asyncpg Pool."""
        pool = MagicMock(spec=asyncpg.Pool)
        pool.close = AsyncMock()
        return pool

    def test_container_properties_lazy_instantiate(self, mock_pool: MagicMock) -> None:
        """Should lazily instantiate storage on first access."""
        container = AppContainer(pool=mock_pool)

        assert container._reminder_storage is None
        storage = container.reminder_storage
        assert container._reminder_storage is storage

    def test_calorie_storage_property(self, mock_pool: MagicMock) -> None:
        """Should lazily instantiate calorie storage."""
        container = AppContainer(pool=mock_pool)

        storage = container.calorie_storage
        assert storage is not None
        assert container._calorie_storage is storage

    def test_workout_storage_property(self, mock_pool: MagicMock) -> None:
        """Should lazily instantiate workout storage."""
        container = AppContainer(pool=mock_pool)

        storage = container.workout_storage
        assert storage is not None
        assert container._workout_storage is storage

    def test_preferences_storage_property(self, mock_pool: MagicMock) -> None:
        """Should lazily instantiate preferences storage."""
        container = AppContainer(pool=mock_pool)

        storage = container.preferences_storage
        assert storage is not None
        assert container._preferences_storage is storage

    @pytest.mark.asyncio
    async def test_close_closes_pool_and_storages(self, mock_pool: MagicMock) -> None:
        """Should close pool and all storage instances."""
        container = AppContainer(pool=mock_pool)

        reminder = container.reminder_storage
        reminder.close = AsyncMock()  # type: ignore[method-assign]

        await container.close()

        reminder.close.assert_called_once()
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_storages_resets_references(self, mock_pool: MagicMock) -> None:
        """Should reset storage references after close."""
        container = AppContainer(pool=mock_pool)

        _ = container.reminder_storage
        _ = container.calorie_storage
        _ = container.workout_storage
        _ = container.preferences_storage

        await container._close_storages()

        assert container._reminder_storage is None
        assert container._calorie_storage is None
        assert container._workout_storage is None
        assert container._preferences_storage is None

    @pytest.mark.asyncio
    async def test_close_storages_partial(self, mock_pool: MagicMock) -> None:
        """Should handle partial storage initialization."""
        container = AppContainer(pool=mock_pool)

        _ = container.calorie_storage
        _ = container.workout_storage
        _ = container.preferences_storage

        await container._close_storages()

        assert container._reminder_storage is None
        assert container._calorie_storage is None
        assert container._workout_storage is None
        assert container._preferences_storage is None

    @pytest.mark.asyncio
    async def test_initialize_all_storages(self, mock_pool: MagicMock) -> None:
        """Should initialize all storage instances."""
        container = AppContainer(pool=mock_pool)

        reminder = container.reminder_storage
        calorie = container.calorie_storage
        workout = container.workout_storage
        preferences = container.preferences_storage

        reminder.initialize = AsyncMock()  # type: ignore[method-assign]
        calorie.initialize = AsyncMock()  # type: ignore[method-assign]
        workout.initialize = AsyncMock()  # type: ignore[method-assign]
        preferences.initialize = AsyncMock()  # type: ignore[method-assign]

        await container.initialize_all_storages()

        reminder.initialize.assert_called_once()
        calorie.initialize.assert_called_once()
        workout.initialize.assert_called_once()
        preferences.initialize.assert_called_once()
