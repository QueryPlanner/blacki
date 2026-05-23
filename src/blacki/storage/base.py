"""Base class for SQLite-backed storage implementations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SqlStorage(ABC):
    """Abstract base class for SQLite-backed storage.

    Provides common initialization pattern with thread-safe schema creation
    and unified query helpers that abstract away SQLite-specific patterns.

    Attributes:
        _conn: The aiosqlite connection.
        _lock: Async lock for thread-safe operations (shared across storages).
        _schema_ready: Whether schema has been created.
    """

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        """Initialize storage with a SQLite connection.

        Args:
            conn: aiosqlite connection.
            lock: Shared lock for write operations.
        """
        self._conn = conn
        self._lock = lock
        self._schema_ready = False

    async def initialize(self) -> None:
        """Ensure schema is created.

        Thread-safe: uses lock to prevent concurrent initialization.
        Idempotent: returns early if already initialized.
        """
        async with self._lock:
            if self._schema_ready:
                return
            await self._create_tables()
            self._schema_ready = True
            logger.info("%s schema ready (SQLite)", self.__class__.__name__)

    async def close(self) -> None:
        """Mark storage as uninitialized.

        Note: Connection lifecycle is managed externally (by AppContainer).
        """
        async with self._lock:
            self._schema_ready = False

    @property
    def is_initialized(self) -> bool:
        """Return True if storage has been initialized."""
        return self._schema_ready

    @abstractmethod
    async def _create_tables(self) -> None:
        """Create tables and indexes.

        Override in subclasses to define schema.
        """

    async def _execute(
        self,
        query: str,
        params: tuple[Any, ...] = (),
        *,
        use_lock: bool = True,
    ) -> int:
        """Execute a write query and return lastrowid.

        Args:
            query: SQL query with ? placeholders.
            params: Query parameters.
            use_lock: Whether to acquire the write lock.

        Returns:
            The last inserted row ID.

        Raises:
            RuntimeError: If lastrowid is None after insert.
        """
        if use_lock:
            async with self._lock:  # noqa: SIM117
                async with self._conn.execute(query, params) as cursor:
                    if cursor.lastrowid is None:
                        raise RuntimeError("Failed to get lastrowid after insert")
                    return cursor.lastrowid
        else:
            async with self._conn.execute(query, params) as cursor:
                if cursor.lastrowid is None:
                    raise RuntimeError("Failed to get lastrowid after insert")
                return cursor.lastrowid

    async def _execute_many(
        self,
        query: str,
        params_list: list[tuple[Any, ...]],
        *,
        use_lock: bool = True,
    ) -> None:
        """Execute a write query multiple times with different params.

        Args:
            query: SQL query with ? placeholders.
            params_list: List of parameter tuples.
            use_lock: Whether to acquire the write lock.
        """
        if use_lock:
            async with self._lock:
                await self._conn.executemany(query, params_list)
        else:
            await self._conn.executemany(query, params_list)

    async def _fetch_one(
        self,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> dict[str, Any] | None:
        """Execute a query and return a single row as dict.

        Args:
            query: SQL query with ? placeholders.
            params: Query parameters.

        Returns:
            A dict representing the row, or None if no result.
        """
        async with self._conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def _fetch_all(
        self,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        """Execute a query and return all rows as dicts.

        Args:
            query: SQL query with ? placeholders.
            params: Query parameters.

        Returns:
            A list of dicts representing the rows.
        """
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def _fetch_val(
        self,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> Any:
        """Execute a query and return a single value.

        Args:
            query: SQL query with ? placeholders.
            params: Query parameters.

        Returns:
            The first column of the first row, or None.
        """
        async with self._conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    @property
    def conn(self) -> aiosqlite.Connection:
        """Get the underlying connection for advanced operations.

        Use with caution - bypasses the write lock.
        """
        return self._conn
