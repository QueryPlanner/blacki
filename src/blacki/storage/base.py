"""Base class for Postgres-backed storage implementations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class PostgresStorage(ABC):
    """Abstract base class for Postgres-backed storage.

    Provides common initialization pattern with thread-safe schema creation.
    Subclasses must implement _create_tables().

    Attributes:
        _pool: The asyncpg connection pool.
        _lock: Async lock for thread-safe initialization.
        _schema_ready: Whether schema has been created.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize storage with a Postgres pool.

        Args:
            pool: asyncpg connection pool.
        """
        self._pool = pool
        self._lock = asyncio.Lock()
        self._schema_ready = False

    async def initialize(self) -> None:
        """Ensure schema is created.

        Thread-safe: uses lock to prevent concurrent initialization.
        Idempotent: returns early if already initialized.
        """
        async with self._lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as conn:
                await self._create_tables(conn)
            self._schema_ready = True
            logger.info("%s schema ready (Postgres)", self.__class__.__name__)

    async def close(self) -> None:
        """Mark storage as uninitialized.

        Note: Pool lifecycle is managed externally (by AppContainer).
        """
        async with self._lock:
            self._schema_ready = False

    @property
    def is_initialized(self) -> bool:
        """Return True if storage has been initialized."""
        return self._schema_ready

    @abstractmethod
    async def _create_tables(self, conn: asyncpg.Connection) -> None:
        """Create tables and indexes.

        Args:
            conn: Database connection to use for DDL operations.
        """
