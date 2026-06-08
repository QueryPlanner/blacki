"""Dependency injection container for managing application resources.

This module provides a lightweight DI container to manage singleton lifecycles,
replacing the global singleton pattern with explicit dependency injection.

Usage:
    container = await AppContainer.create(sqlite_path)
    await container.initialize_all_storages()
    set_container(container)
    try:
        reminder_storage = container.reminder_storage
        # ... use storage ...
    finally:
        await container.close()
        set_container(None)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    import aiosqlite

    from blacki.calories.storage import SqliteCalorieStorage
    from blacki.declarative_db.storage import SqliteDeclarativeDbStorage
    from blacki.reminders.storage import SqliteReminderStorage
    from blacki.utils.preferences import SqlitePreferencesStorage
    from blacki.workouts.storage import SqliteWorkoutStorage

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


async def init_container(sqlite_path: str | Path) -> AppContainer:
    """Create and set the global container.

    Args:
        sqlite_path: Path to the SQLite database file.

    Returns:
        Initialized container with database connection.
    """
    container = await AppContainer.create(sqlite_path)
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


def set_container_from_connection(
    conn: aiosqlite.Connection,
    lock: asyncio.Lock | None = None,
) -> AppContainer:
    """Create and set a container from an existing connection.

    Useful for tests that create their own mock connection.

    Args:
        conn: An existing aiosqlite connection.
        lock: Optional lock for write operations. If None, creates a new one.

    Returns:
        Container instance using the provided connection.
    """
    global _container
    _container = AppContainer(
        conn=conn,
        _lock=lock or asyncio.Lock(),
    )
    return _container


@dataclass
class AppContainer:
    """Container for managing application-wide resources.

    Manages the lifecycle of the database connection and storage singletons.
    All storages are lazily instantiated on first access.

    Attributes:
        conn: The aiosqlite connection.
        _lock: Shared lock for write operations.
    """

    conn: aiosqlite.Connection
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _reminder_storage: SqliteReminderStorage | None = field(
        default=None, init=False, repr=False
    )
    _calorie_storage: SqliteCalorieStorage | None = field(
        default=None, init=False, repr=False
    )
    _workout_storage: SqliteWorkoutStorage | None = field(
        default=None, init=False, repr=False
    )
    _preferences_storage: SqlitePreferencesStorage | None = field(
        default=None, init=False, repr=False
    )
    _declarative_db_storage: SqliteDeclarativeDbStorage | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    async def create(cls, sqlite_path: str | Path) -> Self:
        """Create and initialize the container with a SQLite database.

        Args:
            sqlite_path: Path to the SQLite database file.

        Returns:
            Initialized container with database connection.
        """
        from blacki.storage.sqlite import create_connection

        conn = await create_connection(sqlite_path)
        return cls(conn=conn)

    async def close(self) -> None:
        """Close all storage instances and the connection."""
        await self._close_storages()
        await self.conn.close()
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

        if self._declarative_db_storage is not None:
            await self._declarative_db_storage.close()
            self._declarative_db_storage = None

    async def initialize_all_storages(self) -> None:
        """Initialize all storage instances.

        This is optional - storages are also instantiated lazily on first access.
        Call this during startup to catch initialization errors early.
        """
        await self.reminder_storage.initialize()
        await self.calorie_storage.initialize()
        await self.workout_storage.initialize()
        await self.preferences_storage.initialize()
        await self.declarative_db_storage.initialize()

    @property
    def lock(self) -> asyncio.Lock:
        """Get the shared write lock."""
        return self._lock

    @property
    def reminder_storage(self) -> SqliteReminderStorage:
        """Get or create the reminder storage instance."""
        if self._reminder_storage is None:
            from blacki.reminders.storage import SqliteReminderStorage

            self._reminder_storage = SqliteReminderStorage(self.conn, self._lock)
        return self._reminder_storage

    @property
    def calorie_storage(self) -> SqliteCalorieStorage:
        """Get or create the calorie storage instance."""
        if self._calorie_storage is None:
            from blacki.calories.storage import SqliteCalorieStorage

            self._calorie_storage = SqliteCalorieStorage(self.conn, self._lock)
        return self._calorie_storage

    @property
    def workout_storage(self) -> SqliteWorkoutStorage:
        """Get or create the workout storage instance."""
        if self._workout_storage is None:
            from blacki.workouts.storage import SqliteWorkoutStorage

            self._workout_storage = SqliteWorkoutStorage(self.conn, self._lock)
        return self._workout_storage

    @property
    def preferences_storage(self) -> SqlitePreferencesStorage:
        """Get or create the preferences storage instance."""
        if self._preferences_storage is None:
            from blacki.utils.preferences import SqlitePreferencesStorage

            self._preferences_storage = SqlitePreferencesStorage(self.conn, self._lock)
        return self._preferences_storage

    @property
    def declarative_db_storage(self) -> SqliteDeclarativeDbStorage:
        """Get or create the declarative database storage instance."""
        if self._declarative_db_storage is None:
            from blacki.declarative_db.storage import SqliteDeclarativeDbStorage

            self._declarative_db_storage = SqliteDeclarativeDbStorage(
                self.conn, self._lock
            )
        return self._declarative_db_storage
