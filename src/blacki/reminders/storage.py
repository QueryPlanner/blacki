"""Persistent storage for scheduled reminders backed by SQLite.

A single aiosqlite connection is used for all database operations.
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from blacki.storage.base import SqlStorage
from blacki.utils.timezone import now_utc

if TYPE_CHECKING:
    import asyncio

    import aiosqlite

logger = logging.getLogger(__name__)

DUE_REMINDERS_FETCH_LIMIT = 100


class Reminder(BaseModel):
    """A scheduled reminder.

    Attributes:
        id: Unique identifier (auto-generated).
        user_id: Telegram chat key for the user who set the reminder (may use a
            negative numeric segment for groups and supergroups).
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
        """Return a batch of unsent reminders whose trigger time has passed."""

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


class SqliteReminderStorage(SqlStorage):
    """Storage for reminders using SQLite via aiosqlite."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        super().__init__(conn, lock)

    async def _create_tables(self) -> None:
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                trigger_time TEXT NOT NULL,
                is_sent INTEGER NOT NULL DEFAULT 0,
                recurrence_rule TEXT,
                recurrence_text TEXT,
                timezone_name TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reminders_due
            ON reminders (is_sent, trigger_time)
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reminders_user_id
            ON reminders (user_id)
        """)

    async def add_reminder(self, reminder: Reminder) -> int:
        rid = await self._execute(
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reminder.user_id,
                reminder.message,
                reminder.trigger_time,
                int(reminder.is_sent),
                reminder.recurrence_rule,
                reminder.recurrence_text,
                reminder.timezone_name,
                reminder.created_at,
            ),
        )
        logger.info(
            "Added reminder %s for user %s at %s (SQLite)",
            rid,
            reminder.user_id,
            reminder.trigger_time,
        )
        return rid

    async def get_due_reminders(self) -> list[Reminder]:
        now = now_utc().isoformat(timespec="seconds")
        rows = await self._fetch_all(
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
            WHERE trigger_time <= ? AND is_sent = 0
            ORDER BY trigger_time ASC
            LIMIT ?
            """,
            (now, DUE_REMINDERS_FETCH_LIMIT),
        )
        return [self._row_to_reminder(r) for r in rows]

    async def mark_sent(self, reminder_id: int) -> None:
        async with self._lock:
            await self._conn.execute(
                "UPDATE reminders SET is_sent = 1 WHERE id = ?",
                (reminder_id,),
            )
        logger.info("Marked reminder %s as sent (SQLite)", reminder_id)

    async def reschedule_reminder(
        self, reminder_id: int, next_trigger_time: str
    ) -> None:
        async with self._lock:
            await self._conn.execute(
                """
                UPDATE reminders
                SET trigger_time = ?, is_sent = 0
                WHERE id = ?
                """,
                (next_trigger_time, reminder_id),
            )
        logger.info(
            "Rescheduled recurring reminder %s for %s (SQLite)",
            reminder_id,
            next_trigger_time,
        )

    async def get_user_reminders(
        self, user_id: str, include_sent: bool = False
    ) -> list[Reminder]:
        query = """
            SELECT
                id, user_id, message, trigger_time, is_sent,
                recurrence_rule, recurrence_text, timezone_name, created_at
            FROM reminders WHERE user_id = ?
        """
        params: list[Any] = [user_id]
        if not include_sent:
            query += " AND is_sent = 0"
        query += " ORDER BY trigger_time ASC"
        rows = await self._fetch_all(query, tuple(params))
        return [self._row_to_reminder(r) for r in rows]

    async def delete_reminder(self, reminder_id: int, user_id: str) -> bool:
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM reminders WHERE id = ? AND user_id = ?",
                (reminder_id, user_id),
            )
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted reminder %s (SQLite)", reminder_id)
        return deleted

    def _row_to_reminder(self, row: dict[str, Any]) -> Reminder:
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


_storage: SqliteReminderStorage | None = None


def get_storage() -> SqliteReminderStorage:
    """Return the process-wide singleton ReminderStorage instance.

    Uses the AppContainer for dependency injection.
    """
    from blacki.container import get_container

    container = get_container()
    storage = container.reminder_storage
    if not storage.is_initialized:
        raise RuntimeError(
            "Reminder storage not initialized. Call storage.initialize() first."
        )
    return storage
