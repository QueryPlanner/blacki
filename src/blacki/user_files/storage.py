"""SQLite catalog for user-scoped R2 objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from blacki.storage.base import SqlStorage

if TYPE_CHECKING:
    import asyncio

    import aiosqlite


@dataclass(frozen=True, slots=True)
class UserFileRecord:
    """One durable object catalog entry."""

    object_id: str
    owner_id: str
    r2_key: str
    display_name: str
    media_kind: str
    mime_type: str | None
    size_bytes: int
    sha256: str
    telegram_file_unique_id: str | None
    uploaded_at: str
    last_seen_at: str
    expires_at: str
    status: str = "available"


class SqliteUserFileStorage(SqlStorage):
    """Persistent metadata catalog for objects held in R2."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        super().__init__(conn, lock)

    async def _create_tables(self) -> None:
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS user_files (
                object_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                r2_key TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                media_kind TEXT NOT NULL,
                mime_type TEXT,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                sha256 TEXT NOT NULL,
                telegram_file_unique_id TEXT,
                uploaded_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'available',
                UNIQUE (owner_id, sha256)
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_files_owner_recent
            ON user_files (owner_id, uploaded_at DESC)
        """)

    async def get_by_hash(
        self, owner_id: str, sha256: str, now_iso: str
    ) -> UserFileRecord | None:
        """Return an available, unexpired duplicate for one owner."""
        row = await self._fetch_one(
            """
            SELECT * FROM user_files
            WHERE owner_id = ? AND sha256 = ? AND status = 'available'
              AND expires_at > ?
            """,
            (owner_id, sha256, now_iso),
        )
        return self._row(row)

    async def get_available(
        self, owner_id: str, object_id: str, now_iso: str
    ) -> UserFileRecord | None:
        """Resolve an opaque object ID within its authenticated owner."""
        row = await self._fetch_one(
            """
            SELECT * FROM user_files
            WHERE owner_id = ? AND object_id = ? AND status = 'available'
              AND expires_at > ?
            """,
            (owner_id, object_id, now_iso),
        )
        return self._row(row)

    async def add(self, record: UserFileRecord) -> None:
        """Insert one catalog record."""
        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO user_files (
                    object_id, owner_id, r2_key, display_name, media_kind,
                    mime_type, size_bytes, sha256, telegram_file_unique_id,
                    uploaded_at, last_seen_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.object_id,
                    record.owner_id,
                    record.r2_key,
                    record.display_name,
                    record.media_kind,
                    record.mime_type,
                    record.size_bytes,
                    record.sha256,
                    record.telegram_file_unique_id,
                    record.uploaded_at,
                    record.last_seen_at,
                    record.expires_at,
                    record.status,
                ),
            )

    async def touch_duplicate(
        self, owner_id: str, object_id: str, display_name: str, last_seen_at: str
    ) -> None:
        """Update non-retention metadata for a duplicate upload."""
        async with self._lock:
            await self._conn.execute(
                """
                UPDATE user_files SET display_name = ?, last_seen_at = ?
                WHERE owner_id = ? AND object_id = ?
                """,
                (display_name, last_seen_at, owner_id, object_id),
            )

    async def list_available(
        self, owner_id: str, query: str, limit: int, now_iso: str
    ) -> list[UserFileRecord]:
        """List recent owner-scoped objects, optionally matching a filename."""
        normalized_query = query.strip().casefold()
        if normalized_query:
            rows = await self._fetch_all(
                """
                SELECT * FROM user_files
                WHERE owner_id = ? AND status = 'available' AND expires_at > ?
                  AND instr(lower(display_name), ?) > 0
                ORDER BY uploaded_at DESC LIMIT ?
                """,
                (owner_id, now_iso, normalized_query, limit),
            )
        else:
            rows = await self._fetch_all(
                """
                SELECT * FROM user_files
                WHERE owner_id = ? AND status = 'available' AND expires_at > ?
                ORDER BY uploaded_at DESC LIMIT ?
                """,
                (owner_id, now_iso, limit),
            )
        return [record for row in rows if (record := self._row(row)) is not None]

    async def delete(self, owner_id: str, object_id: str) -> bool:
        """Delete one owner-scoped catalog entry."""
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM user_files WHERE owner_id = ? AND object_id = ?",
                (owner_id, object_id),
            )
            return cursor.rowcount > 0

    async def cleanup_expired(self, now_iso: str) -> int:
        """Remove metadata whose application-level retention has elapsed."""
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM user_files WHERE expires_at <= ?", (now_iso,)
            )
            return cursor.rowcount

    @staticmethod
    def _row(row: dict[str, Any] | None) -> UserFileRecord | None:
        return UserFileRecord(**row) if row is not None else None
