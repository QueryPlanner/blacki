"""Local Telegram access control and identity storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from blacki.storage.base import SqlStorage
from blacki.utils.timezone import now_utc

if TYPE_CHECKING:
    import asyncio

    import aiosqlite


AuthorizationSource = Literal["legacy", "passphrase"]


@dataclass(frozen=True, slots=True)
class TelegramIdentity:
    """A locally stored, user-controlled Telegram display identity."""

    user_id: int
    display_name: str
    username: str | None


class TelegramAccessStorage(SqlStorage):
    """Persist Telegram authorization and dashboard-only identity labels."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        super().__init__(conn, lock)

    async def _create_tables(self) -> None:
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS telegram_access (
                telegram_user_id INTEGER PRIMARY KEY,
                source TEXT NOT NULL CHECK(source IN ('legacy', 'passphrase')),
                access_code_fingerprint TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS telegram_identities (
                telegram_user_id INTEGER PRIMARY KEY,
                display_name TEXT NOT NULL,
                username TEXT,
                updated_at TEXT NOT NULL
            );
        """)

    async def is_authorized(
        self, telegram_user_id: int, access_code_fingerprint: str
    ) -> bool:
        row = await self._fetch_one(
            """
            SELECT source, access_code_fingerprint
            FROM telegram_access
            WHERE telegram_user_id = ?
            """,
            (telegram_user_id,),
        )
        if row is None:
            return False
        source = row["source"]
        fingerprint = row["access_code_fingerprint"]
        return source == "legacy" or (
            source == "passphrase"
            and isinstance(fingerprint, str)
            and fingerprint == access_code_fingerprint
        )

    async def has_authorization_record(self, telegram_user_id: int) -> bool:
        """Return whether this user was previously granted any authorization."""
        return (
            await self._fetch_one(
                "SELECT 1 FROM telegram_access WHERE telegram_user_id = ?",
                (telegram_user_id,),
            )
            is not None
        )

    async def grant(
        self,
        telegram_user_id: int,
        *,
        source: AuthorizationSource,
        access_code_fingerprint: str | None = None,
    ) -> None:
        now = now_utc().isoformat(timespec="seconds")
        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO telegram_access (
                    telegram_user_id, source, access_code_fingerprint,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(telegram_user_id) DO UPDATE SET
                    source = excluded.source,
                    access_code_fingerprint = excluded.access_code_fingerprint,
                    updated_at = excluded.updated_at
                """,
                (telegram_user_id, source, access_code_fingerprint, now, now),
            )

    async def record_identity(self, identity: TelegramIdentity) -> None:
        now = now_utc().isoformat(timespec="seconds")
        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO telegram_identities (
                    telegram_user_id, display_name, username, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(telegram_user_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    username = excluded.username,
                    updated_at = excluded.updated_at
                """,
                (identity.user_id, identity.display_name, identity.username, now),
            )


def get_telegram_access_storage() -> TelegramAccessStorage:
    """Return the initialized process-wide Telegram access storage."""
    from blacki.container import get_container

    storage = get_container().telegram_access_storage
    if not storage.is_initialized:
        raise RuntimeError("Telegram access storage is not initialized")
    return storage
