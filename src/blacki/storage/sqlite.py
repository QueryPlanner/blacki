"""SQLite connection management for tools.db."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_BUSY_TIMEOUT_MS = 5000


async def create_connection(
    db_path: str | Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> aiosqlite.Connection:
    """Create a SQLite connection with WAL mode and optimal settings.

    Args:
        db_path: Path to the SQLite database file.
        busy_timeout_ms: Milliseconds to wait when database is locked.

    Returns:
        Configured aiosqlite connection.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(path, isolation_level=None)
    conn.row_factory = aiosqlite.Row

    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    await conn.execute("PRAGMA foreign_keys=ON")

    logger.info("SQLite connection opened: %s (WAL mode)", path)
    return conn


async def close_connection(conn: aiosqlite.Connection) -> None:
    """Close a SQLite connection.

    Args:
        conn: The connection to close.
    """
    await conn.close()
    logger.info("SQLite connection closed")
