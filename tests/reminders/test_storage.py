# mypy: disable-error-code="no-untyped-def"
"""Unit tests for reminder storage."""

import asyncio

import aiosqlite
import pytest

from blacki.reminders.storage import (
    DUE_REMINDERS_FETCH_LIMIT,
    Reminder,
    SqliteReminderStorage,
)


@pytest.fixture
async def conn():
    """Create an in-memory SQLite connection for testing."""
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


@pytest.fixture
def lock():
    """Create a lock for write operations."""
    return asyncio.Lock()


@pytest.fixture
async def storage(conn, lock):
    """Create a storage instance with the test connection."""
    storage = SqliteReminderStorage(conn, lock)
    await storage.initialize()
    yield storage
    await storage.close()


class TestReminder:
    """Tests for Reminder model."""

    def test_creates_reminder(self) -> None:
        """Should create a Reminder with all fields."""
        reminder = Reminder(
            id=1,
            user_id="telegram-chat-123",
            message="Test reminder",
            trigger_time="2026-04-18T12:00:00+00:00",
            is_sent=False,
            recurrence_rule=None,
            recurrence_text=None,
            timezone_name=None,
            created_at="2026-04-18T10:00:00+00:00",
        )

        assert reminder.id == 1
        assert reminder.user_id == "telegram-chat-123"
        assert reminder.message == "Test reminder"
        assert reminder.is_sent is False

    def test_is_recurring_false_for_one_time(self) -> None:
        """Should return False for one-time reminders."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        assert reminder.is_recurring is False

    def test_is_recurring_true_for_recurring(self) -> None:
        """Should return True for recurring reminders."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            recurrence_rule="*/15 * * * *",
            created_at="2026-04-18T10:00:00+00:00",
        )

        assert reminder.is_recurring is True


class TestSqliteReminderStorage:
    """Tests for SqliteReminderStorage."""

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, conn, lock) -> None:
        """Should create tables on initialization."""
        storage = SqliteReminderStorage(conn, lock)
        await storage.initialize()

        assert storage.is_initialized is True

        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reminders'"
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_add_reminder(self, storage) -> None:
        """Should add a reminder and return its ID."""
        reminder = Reminder(
            user_id="user1",
            message="Test reminder",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        result = await storage.add_reminder(reminder)

        assert result == 1

    @pytest.mark.asyncio
    async def test_get_due_reminders(self, storage) -> None:
        """Should fetch due reminders."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2020-01-01T00:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        await storage.add_reminder(reminder)

        result = await storage.get_due_reminders()

        assert len(result) == 1
        assert result[0].message == "Test"

    @pytest.mark.asyncio
    async def test_get_due_reminders_respects_limit(self, conn, lock) -> None:
        """Should limit the number of due reminders fetched."""
        storage = SqliteReminderStorage(conn, lock)
        await storage.initialize()

        for i in range(DUE_REMINDERS_FETCH_LIMIT + 10):
            reminder = Reminder(
                user_id="user1",
                message=f"Test {i}",
                trigger_time="2020-01-01T00:00:00+00:00",
                created_at="2026-04-18T10:00:00+00:00",
            )
            await storage.add_reminder(reminder)

        result = await storage.get_due_reminders()

        assert len(result) == DUE_REMINDERS_FETCH_LIMIT

    @pytest.mark.asyncio
    async def test_mark_sent(self, storage) -> None:
        """Should mark a reminder as sent."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        rid = await storage.add_reminder(reminder)

        await storage.mark_sent(rid)

        rows = await storage.get_user_reminders("user1", include_sent=True)
        assert rows[0].is_sent is True

    @pytest.mark.asyncio
    async def test_reschedule_reminder(self, storage) -> None:
        """Should reschedule a recurring reminder."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        rid = await storage.add_reminder(reminder)

        await storage.reschedule_reminder(rid, "2026-04-19T12:00:00+00:00")

        rows = await storage.get_user_reminders("user1")
        assert rows[0].trigger_time == "2026-04-19T12:00:00+00:00"
        assert rows[0].is_sent is False

    @pytest.mark.asyncio
    async def test_get_user_reminders(self, storage) -> None:
        """Should get reminders for a user."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        await storage.add_reminder(reminder)

        result = await storage.get_user_reminders("user1")

        assert len(result) == 1
        assert result[0].message == "Test"

    @pytest.mark.asyncio
    async def test_get_user_reminders_include_sent(self, storage) -> None:
        """Should include sent reminders when requested."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        rid = await storage.add_reminder(reminder)
        await storage.mark_sent(rid)

        result = await storage.get_user_reminders("user1", include_sent=True)

        assert len(result) == 1
        assert result[0].is_sent is True

    @pytest.mark.asyncio
    async def test_get_user_reminders_excludes_sent_by_default(self, storage) -> None:
        """Should exclude sent reminders by default."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        rid = await storage.add_reminder(reminder)
        await storage.mark_sent(rid)

        result = await storage.get_user_reminders("user1")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_delete_reminder_found(self, storage) -> None:
        """Should delete a reminder and return True."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        rid = await storage.add_reminder(reminder)

        result = await storage.delete_reminder(rid, "user1")

        assert result is True
        rows = await storage.get_user_reminders("user1")
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_delete_reminder_not_found(self, storage) -> None:
        """Should return False if reminder not found."""
        result = await storage.delete_reminder(999, "user1")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_reminder_wrong_user(self, storage) -> None:
        """Should return False if reminder belongs to different user."""
        reminder = Reminder(
            user_id="user1",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        rid = await storage.add_reminder(reminder)

        result = await storage.delete_reminder(rid, "user2")

        assert result is False
        rows = await storage.get_user_reminders("user1")
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_initialize_returns_early_if_schema_ready(self, conn, lock) -> None:
        """Should return early if schema already ready."""
        storage = SqliteReminderStorage(conn, lock)
        await storage.initialize()

        await storage.initialize()

        assert storage.is_initialized is True

    @pytest.mark.asyncio
    async def test_close_resets_schema_ready(self, storage) -> None:
        """Should reset schema ready flag on close."""
        assert storage.is_initialized is True

        await storage.close()

        assert storage.is_initialized is False


class TestGetStorage:
    """Tests for get_storage function."""

    @pytest.mark.asyncio
    async def test_get_storage_raises_when_not_initialized(self, conn, lock) -> None:
        """Should raise RuntimeError when storage is not initialized."""
        from blacki.container import (
            reset_container_for_tests,
            set_container_from_connection,
        )
        from blacki.reminders.storage import get_storage

        set_container_from_connection(conn, lock)

        with pytest.raises(RuntimeError, match="Reminder storage not initialized"):
            get_storage()

        reset_container_for_tests()

    @pytest.mark.asyncio
    async def test_get_storage_returns_storage_when_initialized(
        self, conn, lock
    ) -> None:
        """Should return storage when initialized."""
        from blacki.container import (
            reset_container_for_tests,
            set_container_from_connection,
        )
        from blacki.reminders.storage import get_storage

        container = set_container_from_connection(conn, lock)
        await container.reminder_storage.initialize()

        result = get_storage()

        assert result is container.reminder_storage

        reset_container_for_tests()
