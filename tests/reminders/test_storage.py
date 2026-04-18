"""Unit tests for reminder storage."""

from unittest.mock import AsyncMock, MagicMock

import asyncpg  # type: ignore[import-untyped]
import pytest

from blacki.reminders.storage import (
    PostgresReminderStorage,
    Reminder,
    close_reminder_storage,
    get_storage,
    init_reminder_storage,
)


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


class TestPostgresReminderStorage:
    """Tests for PostgresReminderStorage."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock asyncpg Pool."""
        pool = MagicMock(spec=asyncpg.Pool)
        pool.acquire = MagicMock()
        pool.fetch = AsyncMock()
        pool.fetchval = AsyncMock()
        pool.execute = AsyncMock()
        return pool

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create a mock asyncpg Connection."""
        conn = MagicMock(spec=asyncpg.Connection)
        conn.execute = AsyncMock()
        return conn

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(
        self, mock_pool: MagicMock, mock_connection: MagicMock
    ) -> None:
        """Should create tables on initialization."""
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_connection
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        storage = PostgresReminderStorage(mock_pool)
        await storage.initialize()

        assert mock_connection.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_add_reminder(self, mock_pool: MagicMock) -> None:
        """Should add a reminder and return its ID."""
        mock_pool.fetchval.return_value = 42

        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        reminder = Reminder(
            user_id="user1",
            message="Test reminder",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        result = await storage.add_reminder(reminder)

        assert result == 42
        mock_pool.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_due_reminders(self, mock_pool: MagicMock) -> None:
        """Should fetch due reminders."""
        mock_pool.fetch.return_value = [
            {
                "id": 1,
                "user_id": "user1",
                "message": "Test",
                "trigger_time": "2026-04-18T12:00:00+00:00",
                "is_sent": False,
                "recurrence_rule": None,
                "recurrence_text": None,
                "timezone_name": None,
                "created_at": "2026-04-18T10:00:00+00:00",
            }
        ]

        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        result = await storage.get_due_reminders()

        assert len(result) == 1
        assert result[0].id == 1
        assert result[0].message == "Test"

    @pytest.mark.asyncio
    async def test_mark_sent(self, mock_pool: MagicMock) -> None:
        """Should mark a reminder as sent."""
        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        await storage.mark_sent(42)

        mock_pool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_reschedule_reminder(self, mock_pool: MagicMock) -> None:
        """Should reschedule a recurring reminder."""
        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        await storage.reschedule_reminder(42, "2026-04-19T12:00:00+00:00")

        mock_pool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_reminders(self, mock_pool: MagicMock) -> None:
        """Should get reminders for a user."""
        mock_pool.fetch.return_value = []

        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        result = await storage.get_user_reminders("user1")

        assert result == []
        mock_pool.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_reminders_include_sent(self, mock_pool: MagicMock) -> None:
        """Should include sent reminders when requested."""
        mock_pool.fetch.return_value = [
            {
                "id": 1,
                "user_id": "user1",
                "message": "Test",
                "trigger_time": "2026-04-18T12:00:00+00:00",
                "is_sent": True,
                "recurrence_rule": None,
                "recurrence_text": None,
                "timezone_name": None,
                "created_at": "2026-04-18T10:00:00+00:00",
            }
        ]

        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        result = await storage.get_user_reminders("user1", include_sent=True)

        assert len(result) == 1
        assert result[0].is_sent is True

    @pytest.mark.asyncio
    async def test_delete_reminder_found(self, mock_pool: MagicMock) -> None:
        """Should delete a reminder and return True."""
        mock_pool.execute.return_value = "DELETE 1"

        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        result = await storage.delete_reminder(42, "user1")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_reminder_not_found(self, mock_pool: MagicMock) -> None:
        """Should return False if reminder not found."""
        mock_pool.execute.return_value = "DELETE 0"

        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        result = await storage.delete_reminder(42, "user1")

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_returns_early_if_schema_ready(
        self, mock_pool: MagicMock
    ) -> None:
        """Should return early if schema already ready."""
        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        await storage.initialize()

        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_resets_schema_ready(self, mock_pool: MagicMock) -> None:
        """Should reset schema ready flag on close."""
        storage = PostgresReminderStorage(mock_pool)
        storage._schema_ready = True

        await storage.close()

        assert storage._schema_ready is False


class TestStorageSingleton:
    """Tests for storage singleton management."""

    @pytest.mark.asyncio
    async def test_get_storage_raises_if_not_initialized(self) -> None:
        """Should raise RuntimeError if storage not initialized."""
        import blacki.reminders.storage as storage_module

        storage_module._storage = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_storage()

    @pytest.mark.asyncio
    async def test_init_and_get_storage(self) -> None:
        """Should initialize and return storage singleton."""
        import blacki.reminders.storage as storage_module

        storage_module._storage = None

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_pool.acquire = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        storage = await init_reminder_storage(mock_pool)

        assert storage is not None
        assert get_storage() is storage

        storage_module._storage = None

    @pytest.mark.asyncio
    async def test_close_reminder_storage(self) -> None:
        """Should close and reset storage singleton."""
        import blacki.reminders.storage as storage_module

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_pool.acquire = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        storage = await init_reminder_storage(mock_pool)
        assert storage is not None

        await close_reminder_storage()

        assert storage_module._storage is None

    @pytest.mark.asyncio
    async def test_close_reminder_storage_no_op_if_none(self) -> None:
        """Should do nothing if storage is already None."""
        import blacki.reminders.storage as storage_module

        storage_module._storage = None

        await close_reminder_storage()

        assert storage_module._storage is None
