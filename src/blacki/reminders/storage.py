"""Persistent storage for scheduled reminders backed by Postgres via asyncpg.

A single asyncpg pool is used for all database operations. The pool is shared
with the ADK session service when DATABASE_URL is configured.
"""

import abc
import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from pydantic import BaseModel

from blacki.utils.timezone import now_utc

logger = logging.getLogger(__name__)


class Reminder(BaseModel):
    """A scheduled reminder.

    Attributes:
        id: Unique identifier (auto-generated).
        user_id: Telegram chat ID of the user who set the reminder.
        message: The reminder message to send.
        trigger_time: Next time to send the reminder (ISO format string).
        is_sent: Whether the reminder has been sent.
        recurrence_rule: Normalized cron rule for recurring reminders.
        recurrence_text: Human-readable recurrence description.
        timezone_name: IANA timezone used for recurring schedule calculation.
        created_at: When the reminder was created (ISO format string).
    """

    id: int | None = None
    user_id: str
    message: str
    trigger_time: str
    is_sent: bool = False
    recurrence_rule: str | None = None
    recurrence_text: str | None = None
    timezone_name: str | None = None
    created_at: str

    @property
    def is_recurring(self) -> bool:
        """True when this reminder will be rescheduled after firing."""
        return bool(self.recurrence_rule)


class BaseReminderStorage(abc.ABC):
    """Abstract base class for reminder storage."""

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize storage (create tables, open connections)."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close storage connections."""

    @abc.abstractmethod
    async def add_reminder(self, reminder: Reminder) -> int:
        """Insert a reminder and return its new row ID."""

    @abc.abstractmethod
    async def get_due_reminders(self) -> list[Reminder]:
        """Return all unsent reminders whose trigger time has passed."""

    @abc.abstractmethod
    async def mark_sent(self, reminder_id: int) -> None:
        """Mark a reminder as sent."""

    @abc.abstractmethod
    async def reschedule_reminder(
        self, reminder_id: int, next_trigger_time: str
    ) -> None:
        """Move a recurring reminder to its next scheduled fire time."""

    @abc.abstractmethod
    async def get_user_reminders(
        self, user_id: str, include_sent: bool = False
    ) -> list[Reminder]:
        """Return reminders for a user."""

    @abc.abstractmethod
    async def delete_reminder(self, reminder_id: int, user_id: str) -> bool:
        """Delete a reminder if it belongs to the given user."""


class PostgresReminderStorage(BaseReminderStorage):
    """Storage for reminders using Postgres via asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._lock = asyncio.Lock()
        self._schema_ready = False

    async def initialize(self) -> None:
        """Ensure schema is created."""
        async with self._lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as conn:
                await self._create_tables(conn)
            self._schema_ready = True
            logger.info("Reminder storage schema ready (Postgres)")

    async def close(self) -> None:
        """Mark uninitialized (pool lifecycle managed externally)."""
        async with self._lock:
            self._schema_ready = False

    async def _create_tables(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id           BIGSERIAL PRIMARY KEY,
                user_id      TEXT    NOT NULL,
                message      TEXT    NOT NULL,
                trigger_time TEXT    NOT NULL,
                is_sent      BOOLEAN NOT NULL DEFAULT FALSE,
                recurrence_rule TEXT,
                recurrence_text TEXT,
                timezone_name TEXT,
                created_at   TEXT    NOT NULL
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reminders_trigger_time_sent
            ON reminders (trigger_time, is_sent)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reminders_user_id
            ON reminders (user_id)
        """)

    async def add_reminder(self, reminder: Reminder) -> int:
        rid = await self._pool.fetchval(
            """
            INSERT INTO reminders
                (
                    user_id,
                    message,
                    trigger_time,
                    is_sent,
                    recurrence_rule,
                    recurrence_text,
                    timezone_name,
                    created_at
                )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """,
            reminder.user_id,
            reminder.message,
            reminder.trigger_time,
            reminder.is_sent,
            reminder.recurrence_rule,
            reminder.recurrence_text,
            reminder.timezone_name,
            reminder.created_at,
        )
        logger.info(
            "Added reminder %s for user %s: '%s...' at %s (Postgres)",
            rid,
            reminder.user_id,
            reminder.message[:30],
            reminder.trigger_time,
        )
        return int(rid)

    async def get_due_reminders(self) -> list[Reminder]:
        now = now_utc().isoformat(timespec="seconds")
        rows = await self._pool.fetch(
            """
            SELECT
                id,
                user_id,
                message,
                trigger_time,
                is_sent,
                recurrence_rule,
                recurrence_text,
                timezone_name,
                created_at
            FROM reminders
            WHERE trigger_time <= $1 AND is_sent = FALSE
            ORDER BY trigger_time ASC
            """,
            now,
        )
        return [self._row_to_reminder(r) for r in rows]

    async def mark_sent(self, reminder_id: int) -> None:
        await self._pool.execute(
            "UPDATE reminders SET is_sent = TRUE WHERE id = $1", reminder_id
        )
        logger.info("Marked reminder %s as sent (Postgres)", reminder_id)

    async def reschedule_reminder(
        self, reminder_id: int, next_trigger_time: str
    ) -> None:
        await self._pool.execute(
            """
            UPDATE reminders
            SET trigger_time = $1, is_sent = FALSE
            WHERE id = $2
            """,
            next_trigger_time,
            reminder_id,
        )
        logger.info(
            "Rescheduled recurring reminder %s for %s (Postgres)",
            reminder_id,
            next_trigger_time,
        )

    async def get_user_reminders(
        self, user_id: str, include_sent: bool = False
    ) -> list[Reminder]:
        query = """
            SELECT
                id,
                user_id,
                message,
                trigger_time,
                is_sent,
                recurrence_rule,
                recurrence_text,
                timezone_name,
                created_at
            FROM reminders WHERE user_id = $1
        """
        if not include_sent:
            query += " AND is_sent = FALSE"
        query += " ORDER BY trigger_time ASC"

        rows = await self._pool.fetch(query, user_id)
        return [self._row_to_reminder(r) for r in rows]

    async def delete_reminder(self, reminder_id: int, user_id: str) -> bool:
        result = await self._pool.execute(
            "DELETE FROM reminders WHERE id = $1 AND user_id = $2",
            reminder_id,
            user_id,
        )
        deleted = bool(result == "DELETE 1")
        if deleted:
            logger.info("Deleted reminder %s (Postgres)", reminder_id)
        return deleted

    def _row_to_reminder(self, row: Mapping[str, Any]) -> Reminder:
        return Reminder(
            id=int(row["id"]),
            user_id=row["user_id"],
            message=row["message"],
            trigger_time=row["trigger_time"],
            is_sent=bool(row["is_sent"]),
            recurrence_rule=row["recurrence_rule"],
            recurrence_text=row["recurrence_text"],
            timezone_name=row["timezone_name"],
            created_at=row["created_at"],
        )


_storage: PostgresReminderStorage | None = None


def get_storage() -> PostgresReminderStorage:
    """Return the process-wide singleton ReminderStorage instance."""
    global _storage
    if _storage is None:
        raise RuntimeError(
            "Reminder storage not initialized. Call init_reminder_storage() first."
        )
    return _storage


async def init_reminder_storage(pool: asyncpg.Pool) -> PostgresReminderStorage:
    """Initialize the reminder storage with a Postgres pool."""
    global _storage
    _storage = PostgresReminderStorage(pool)
    await _storage.initialize()
    return _storage


async def close_reminder_storage() -> None:
    """Close the singleton reminder storage."""
    global _storage
    if _storage is not None:
        await _storage.close()
        _storage = None
