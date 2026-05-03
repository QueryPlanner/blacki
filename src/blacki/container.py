"""Dependency injection container for managing application resources.

This module provides a lightweight DI container to manage singleton lifecycles,
replacing the global singleton pattern with explicit dependency injection.

Usage:
    container = await AppContainer.create(database_url)
    set_container(container)
    try:
        reminder_storage = container.reminder_storage
        # ... use storage ...
    finally:
        await container.close()
        set_container(None)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    import asyncpg  # type: ignore[import-untyped]

    from blacki.calories.storage import PostgresCalorieStorage
    from blacki.reminders.storage import PostgresReminderStorage
    from blacki.utils.preferences import PostgresPreferencesStorage
    from blacki.workouts.storage import PostgresWorkoutStorage

logger = logging.getLogger(__name__)

_container: AppContainer | None = None


def get_container() -> AppContainer:
    """Return the global container instance.

    Raises:
        RuntimeError: If container is not initialized.
    """
    if _container is None:
        raise RuntimeError("AppContainer not initialized. Call set_container() first.")
    return _container


def set_container(container: AppContainer | None) -> None:
    """Set the global container instance."""
    global _container
    _container = container


async def init_container(database_url: str, pool_size: int = 5) -> AppContainer:
    """Create and set the global container.

    Args:
        database_url: Postgres connection string.
        pool_size: Maximum number of connections (default: 5).

    Returns:
        Initialized container with database pool.
    """
    container = await AppContainer.create(database_url, pool_size)
    set_container(container)
    return container


async def close_container() -> None:
    """Close and clear the global container."""
    global _container
    if _container is not None:
        await _container.close()
        _container = None


def reset_container_for_tests() -> None:
    """Reset the global container reference without closing it.

    Use this in tests when you want to clear the container reference
    without calling async close. For proper cleanup, use close_container().
    """
    global _container
    _container = None


def set_container_from_pool(pool: asyncpg.Pool) -> AppContainer:
    """Create and set a container from an existing pool.

    Useful for tests that create their own mock pool.

    Args:
        pool: An existing asyncpg pool.

    Returns:
        Container instance using the provided pool.
    """
    global _container
    _container = AppContainer(pool=pool)
    return _container


@dataclass
class AppContainer:
    """Container for managing application-wide resources.

    Manages the lifecycle of the database pool and storage singletons.
    All storages are lazily initialized on first access.

    Attributes:
        pool: The asyncpg connection pool.
    """

    pool: asyncpg.Pool
    _reminder_storage: PostgresReminderStorage | None = field(
        default=None, init=False, repr=False
    )
    _calorie_storage: PostgresCalorieStorage | None = field(
        default=None, init=False, repr=False
    )
    _workout_storage: PostgresWorkoutStorage | None = field(
        default=None, init=False, repr=False
    )
    _preferences_storage: PostgresPreferencesStorage | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    async def create(cls, database_url: str, pool_size: int = 5) -> Self:
        """Create and initialize the container with a database pool.

        Args:
            database_url: Postgres connection string.
            pool_size: Maximum number of connections (default: 5).

        Returns:
            Initialized container with database pool.
        """
        import asyncpg

        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=pool_size,
        )
        return cls(pool=pool)

    async def close(self) -> None:
        """Close all storage instances and the pool."""
        await self._close_storages()
        await self.pool.close()
        logger.info("AppContainer closed")

    async def _close_storages(self) -> None:
        """Close all storage instances."""
        if self._reminder_storage is not None:
            await self._reminder_storage.close()
            self._reminder_storage = None

        if self._calorie_storage is not None:
            await self._calorie_storage.close()
            self._calorie_storage = None

        if self._workout_storage is not None:
            await self._workout_storage.close()
            self._workout_storage = None

        if self._preferences_storage is not None:
            await self._preferences_storage.close()
            self._preferences_storage = None

    async def initialize_all_storages(self) -> None:
        """Initialize all storage instances.

        This is optional - storages are also initialized lazily on first access.
        Call this during startup to catch initialization errors early.
        """
        await self.reminder_storage.initialize()
        await self.calorie_storage.initialize()
        await self.workout_storage.initialize()
        await self.preferences_storage.initialize()

    @property
    def reminder_storage(self) -> PostgresReminderStorage:
        """Get or create the reminder storage instance."""
        if self._reminder_storage is None:
            from blacki.reminders.storage import PostgresReminderStorage

            self._reminder_storage = PostgresReminderStorage(self.pool)
        return self._reminder_storage

    @property
    def calorie_storage(self) -> PostgresCalorieStorage:
        """Get or create the calorie storage instance."""
        if self._calorie_storage is None:
            from blacki.calories.storage import PostgresCalorieStorage

            self._calorie_storage = PostgresCalorieStorage(self.pool)
        return self._calorie_storage

    @property
    def workout_storage(self) -> PostgresWorkoutStorage:
        """Get or create the workout storage instance."""
        if self._workout_storage is None:
            from blacki.workouts.storage import PostgresWorkoutStorage

            self._workout_storage = PostgresWorkoutStorage(self.pool)
        return self._workout_storage

    @property
    def preferences_storage(self) -> PostgresPreferencesStorage:
        """Get or create the preferences storage instance."""
        if self._preferences_storage is None:
            from blacki.utils.preferences import PostgresPreferencesStorage

            self._preferences_storage = PostgresPreferencesStorage(self.pool)
        return self._preferences_storage
