"""Unit tests for reminder scheduler."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blacki.reminders.scheduler import (
    ReminderScheduler,
    _extract_telegram_chat_id,
    _is_stale_reminder,
    _parse_stored_trigger_time,
    get_scheduler,
)
from blacki.reminders.storage import Reminder


class TestReminderScheduler:
    """Tests for ReminderScheduler."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """Create a mock ReminderStorage."""
        storage = MagicMock()
        storage.initialize = AsyncMock()
        storage.get_due_reminders = AsyncMock(return_value=[])
        storage.mark_sent = AsyncMock()
        storage.reschedule_reminder = AsyncMock()
        storage.add_reminder = AsyncMock(return_value=1)
        storage.get_user_reminders = AsyncMock(return_value=[])
        storage.delete_reminder = AsyncMock(return_value=True)
        return storage

    @pytest.fixture
    def mock_api(self) -> MagicMock:
        """Create a mock TelegramApiClient."""
        api = MagicMock()
        api.send_message = AsyncMock()
        return api

    def _create_scheduler(self, mock_storage: MagicMock) -> ReminderScheduler:
        """Create a scheduler with mocked storage."""
        with patch("blacki.reminders.scheduler.get_storage", return_value=mock_storage):
            scheduler = ReminderScheduler()
            scheduler.storage = mock_storage
            return scheduler

    @pytest.mark.asyncio
    async def test_start_initializes_storage(self, mock_storage: MagicMock) -> None:
        """Should initialize storage on start."""
        scheduler = self._create_scheduler(mock_storage)

        await scheduler.start()

        mock_storage.initialize.assert_called_once()
        assert scheduler._running is True

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_shuts_down_scheduler(self, mock_storage: MagicMock) -> None:
        """Should shut down scheduler on stop."""
        scheduler = self._create_scheduler(mock_storage)

        await scheduler.start()
        await scheduler.stop()

        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_schedule_reminder(self, mock_storage: MagicMock) -> None:
        """Should schedule a new reminder."""
        mock_storage.add_reminder = AsyncMock(return_value=42)

        scheduler = self._create_scheduler(mock_storage)

        trigger_time = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)
        result = await scheduler.schedule_reminder(
            user_id="user1",
            message="Test reminder",
            trigger_time=trigger_time,
        )

        assert result == 42
        mock_storage.add_reminder.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_reminders(self, mock_storage: MagicMock) -> None:
        """Should get reminders for a user."""
        mock_storage.get_user_reminders = AsyncMock(
            return_value=[
                Reminder(
                    id=1,
                    user_id="user1",
                    message="Test",
                    trigger_time="2026-04-18T12:00:00+00:00",
                    created_at="2026-04-18T10:00:00+00:00",
                )
            ]
        )

        scheduler = self._create_scheduler(mock_storage)

        result = await scheduler.get_user_reminders("user1")

        assert len(result) == 1
        assert result[0].message == "Test"

    @pytest.mark.asyncio
    async def test_delete_reminder(self, mock_storage: MagicMock) -> None:
        """Should delete a reminder."""
        mock_storage.delete_reminder = AsyncMock(return_value=True)

        scheduler = self._create_scheduler(mock_storage)

        result = await scheduler.delete_reminder(42, "user1")

        assert result is True

    @pytest.mark.asyncio
    async def test_send_reminder(
        self, mock_storage: MagicMock, mock_api: MagicMock
    ) -> None:
        """Should send a reminder via Telegram."""
        reminder = Reminder(
            id=1,
            user_id="telegram-chat-123456",
            message="Test reminder",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        scheduler = self._create_scheduler(mock_storage)
        scheduler.set_api(mock_api)

        await scheduler._send_reminder(reminder)

        mock_api.send_message.assert_called_once()
        call_kwargs = mock_api.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 123456
        assert "Test reminder" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_reminder_marks_sent(
        self, mock_storage: MagicMock, mock_api: MagicMock
    ) -> None:
        """Should mark reminder as sent after delivery."""
        reminder = Reminder(
            id=1,
            user_id="telegram-chat-123456",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        scheduler = self._create_scheduler(mock_storage)
        scheduler.set_api(mock_api)

        await scheduler._send_reminder(reminder)

        mock_storage.mark_sent.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_send_recurring_reminder_reschedules(
        self, mock_storage: MagicMock, mock_api: MagicMock
    ) -> None:
        """Should reschedule recurring reminder after delivery."""
        reminder = Reminder(
            id=1,
            user_id="telegram-chat-123456",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            recurrence_rule="*/15 * * * *",
            timezone_name="UTC",
            created_at="2026-04-18T10:00:00+00:00",
        )

        scheduler = self._create_scheduler(mock_storage)
        scheduler.set_api(mock_api)

        await scheduler._send_reminder(reminder)

        mock_storage.reschedule_reminder.assert_called_once()


class TestParseStoredTriggerTime:
    """Tests for _parse_stored_trigger_time function."""

    def test_parses_iso_string_with_z_suffix(self) -> None:
        """Should parse ISO string with Z suffix."""
        result = _parse_stored_trigger_time("2026-04-18T12:00:00Z")

        assert result.year == 2026
        assert result.month == 4
        assert result.day == 18
        assert result.hour == 12
        assert result.tzinfo == UTC

    def test_parses_iso_string_with_offset(self) -> None:
        """Should parse ISO string with offset."""
        result = _parse_stored_trigger_time("2026-04-18T12:00:00+00:00")

        assert result.hour == 12
        assert result.tzinfo == UTC

    def test_converts_to_utc(self) -> None:
        """Should convert non-UTC times to UTC."""
        result = _parse_stored_trigger_time("2026-04-18T18:00:00+05:30")

        assert result.hour == 12
        assert result.minute == 30

    def test_handles_naive_datetime(self) -> None:
        """Should treat naive datetime as UTC."""
        result = _parse_stored_trigger_time("2026-04-18T12:00:00")

        assert result.tzinfo == UTC


class TestIsStaleReminder:
    """Tests for _is_stale_reminder function."""

    def test_returns_false_for_future_reminder(self) -> None:
        """Should return False for future reminder."""
        current_time = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)
        trigger_time = "2026-04-18T12:00:30+00:00"

        result = _is_stale_reminder(trigger_time, current_time)

        assert result is False

    def test_returns_false_for_recent_reminder(self) -> None:
        """Should return False for reminder within grace period."""
        current_time = datetime(2026, 4, 18, 12, 5, 0, tzinfo=UTC)
        trigger_time = "2026-04-18T12:00:00+00:00"

        result = _is_stale_reminder(trigger_time, current_time)

        assert result is False

    def test_returns_true_for_stale_reminder(self) -> None:
        """Should return True for reminder older than grace period."""
        current_time = datetime(2026, 4, 18, 12, 30, 0, tzinfo=UTC)
        trigger_time = "2026-04-18T12:00:00+00:00"

        result = _is_stale_reminder(trigger_time, current_time)

        assert result is True


class TestExtractTelegramChatId:
    """Tests for _extract_telegram_chat_id function."""

    def test_extracts_chat_id_from_user_id(self) -> None:
        """Should extract numeric chat_id from prefixed user_id."""
        result = _extract_telegram_chat_id("telegram-chat-1399736563")

        assert result == 1399736563

    def test_extracts_chat_id_with_thread(self) -> None:
        """Should extract chat_id from user_id with thread suffix."""
        result = _extract_telegram_chat_id("telegram-chat-1399736563-thread-42")

        assert result == 1399736563

    def test_returns_none_for_invalid_format(self) -> None:
        """Should return None for invalid user_id format."""
        result = _extract_telegram_chat_id("invalid-user-id")

        assert result is None

    def test_returns_none_for_unprefixed_chat_id(self) -> None:
        """Should return None for numeric-only user_id."""
        result = _extract_telegram_chat_id("1399736563")

        assert result is None


class TestGetScheduler:
    """Tests for get_scheduler function."""

    def test_returns_singleton(self) -> None:
        """Should return the same scheduler instance."""
        import blacki.reminders.scheduler as scheduler_module

        scheduler_module._scheduler = None

        mock_storage = MagicMock()

        with patch("blacki.reminders.scheduler.get_storage", return_value=mock_storage):
            scheduler1 = get_scheduler()
            scheduler2 = get_scheduler()

            assert scheduler1 is scheduler2

        scheduler_module._scheduler = None
