"""Unit tests for reminder tools."""

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blacki.reminders.storage import Reminder
from blacki.reminders.tools import (
    _build_reminder_schedule,
    _format_reminder,
    _parse_reminder_datetime,
    cancel_reminder,
    list_reminders,
    schedule_reminder,
)
from conftest import MockState, MockToolContext


class TestScheduleReminder:
    """Tests for schedule_reminder tool."""

    @staticmethod
    def _tool_context(user_id: str = "telegram-chat-123") -> MagicMock:
        state = MockState({"user_id": user_id})
        return cast(MagicMock, MockToolContext(state=state))

    @pytest.fixture
    def mock_scheduler(self) -> MagicMock:
        """Create a mock ReminderScheduler."""
        scheduler = MagicMock()
        scheduler.schedule_reminder = AsyncMock(return_value=42)
        return scheduler

    @pytest.mark.asyncio
    async def test_schedule_one_time_reminder(self, mock_scheduler: MagicMock) -> None:
        """Should schedule a one-time reminder."""
        with patch("blacki.reminders.tools.get_scheduler", return_value=mock_scheduler):
            with patch(
                "blacki.reminders.tools.now_utc",
                return_value=datetime(2026, 4, 18, 10, 0, 0, tzinfo=UTC),
            ):
                result = await schedule_reminder(
                    tool_context=self._tool_context(),
                    message="Test reminder",
                    reminder_datetime="2026-04-18 12:00",
                )

        assert result["status"] == "success"
        assert result["reminder_id"] == 42
        assert "Reminder scheduled" in result["message"]

    @pytest.mark.asyncio
    async def test_schedule_reminder_no_user_id(self) -> None:
        """Should return error if user_id not in context."""
        state = MockState({})
        tool_context = cast(MagicMock, MockToolContext(state=state))

        result = await schedule_reminder(
            tool_context=tool_context,
            message="Test reminder",
            reminder_datetime="2026-04-18 12:00",
        )

        assert result["status"] == "error"
        assert "user not identified" in result["message"]

    @pytest.mark.asyncio
    async def test_schedule_reminder_message_too_long(
        self, mock_scheduler: MagicMock
    ) -> None:
        """Should return error if message is too long."""
        with patch("blacki.reminders.tools.get_scheduler", return_value=mock_scheduler):
            result = await schedule_reminder(
                tool_context=self._tool_context(),
                message="x" * 501,
                reminder_datetime="2026-04-18 12:00",
            )

        assert result["status"] == "error"
        assert "too long" in result["message"]

    @pytest.mark.asyncio
    async def test_schedule_reminder_in_past(self, mock_scheduler: MagicMock) -> None:
        """Should return error if reminder time is in the past."""
        with patch("blacki.reminders.tools.get_scheduler", return_value=mock_scheduler):
            with patch(
                "blacki.reminders.tools.now_utc",
                return_value=datetime(2026, 4, 18, 15, 0, 0, tzinfo=UTC),
            ):
                result = await schedule_reminder(
                    tool_context=self._tool_context(),
                    message="Test reminder",
                    reminder_datetime="2026-04-18 12:00",
                )

        assert result["status"] == "error"
        assert "must be in the future" in result["message"]


class TestListReminders:
    """Tests for list_reminders tool."""

    @staticmethod
    def _tool_context(user_id: str = "telegram-chat-123") -> MagicMock:
        state = MockState({"user_id": user_id})
        return cast(MagicMock, MockToolContext(state=state))

    @pytest.fixture
    def mock_scheduler(self) -> MagicMock:
        """Create a mock ReminderScheduler."""
        scheduler = MagicMock()
        scheduler.get_user_reminders = AsyncMock(return_value=[])
        return scheduler

    @pytest.mark.asyncio
    async def test_list_reminders_empty(self, mock_scheduler: MagicMock) -> None:
        """Should return empty list when no reminders."""
        with patch("blacki.reminders.tools.get_scheduler", return_value=mock_scheduler):
            result = await list_reminders(
                tool_context=self._tool_context(),
            )

        assert result["status"] == "success"
        assert result["reminders"] == []
        assert "no scheduled reminders" in result["message"]

    @pytest.mark.asyncio
    async def test_list_reminders_with_items(self, mock_scheduler: MagicMock) -> None:
        """Should return formatted reminders."""
        reminder = Reminder(
            id=1,
            user_id="telegram-chat-123",
            message="Test reminder",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )
        mock_scheduler.get_user_reminders = AsyncMock(return_value=[reminder])

        with patch("blacki.reminders.tools.get_scheduler", return_value=mock_scheduler):
            result = await list_reminders(
                tool_context=self._tool_context(),
            )

        assert result["status"] == "success"
        assert result["count"] == 1
        assert len(result["reminders"]) == 1


class TestCancelReminder:
    """Tests for cancel_reminder tool."""

    @staticmethod
    def _tool_context(user_id: str = "telegram-chat-123") -> MagicMock:
        state = MockState({"user_id": user_id})
        return cast(MagicMock, MockToolContext(state=state))

    @pytest.fixture
    def mock_scheduler(self) -> MagicMock:
        """Create a mock ReminderScheduler."""
        scheduler = MagicMock()
        scheduler.delete_reminder = AsyncMock(return_value=True)
        return scheduler

    @pytest.mark.asyncio
    async def test_cancel_reminder_success(self, mock_scheduler: MagicMock) -> None:
        """Should cancel a reminder successfully."""
        with patch("blacki.reminders.tools.get_scheduler", return_value=mock_scheduler):
            result = await cancel_reminder(
                tool_context=self._tool_context(),
                reminder_id=42,
            )

        assert result["status"] == "success"
        assert "cancelled" in result["message"]

    @pytest.mark.asyncio
    async def test_cancel_reminder_not_found(self, mock_scheduler: MagicMock) -> None:
        """Should return error if reminder not found."""
        mock_scheduler.delete_reminder = AsyncMock(return_value=False)

        with patch("blacki.reminders.tools.get_scheduler", return_value=mock_scheduler):
            result = await cancel_reminder(
                tool_context=self._tool_context(),
                reminder_id=42,
            )

        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestParseReminderDatetime:
    """Tests for _parse_reminder_datetime function."""

    def test_parse_absolute_datetime(self) -> None:
        """Should parse absolute datetime string."""
        from zoneinfo import ZoneInfo

        with patch(
            "blacki.reminders.tools.get_app_timezone",
            return_value=ZoneInfo("UTC"),
        ):
            with patch(
                "blacki.reminders.tools.naive_local_now",
                return_value=datetime(2026, 4, 18, 10, 0, 0),
            ):
                result = _parse_reminder_datetime("2026-04-18 12:00")

        assert result.year == 2026
        assert result.month == 4
        assert result.day == 18
        assert result.hour == 12
        assert result.minute == 0

    def test_parse_relative_datetime(self) -> None:
        """Should parse relative datetime string."""
        from zoneinfo import ZoneInfo

        with patch(
            "blacki.reminders.tools.get_app_timezone",
            return_value=ZoneInfo("UTC"),
        ):
            with patch(
                "blacki.reminders.tools.naive_local_now",
                return_value=datetime(2026, 4, 18, 10, 0, 0),
            ):
                result = _parse_reminder_datetime("in 1 hour")

        assert result.hour == 11

    def test_raises_on_invalid_datetime(self) -> None:
        """Should raise ValueError for invalid datetime."""
        from zoneinfo import ZoneInfo

        with patch(
            "blacki.reminders.tools.get_app_timezone",
            return_value=ZoneInfo("UTC"),
        ):
            with patch(
                "blacki.reminders.tools.naive_local_now",
                return_value=datetime(2026, 4, 18, 10, 0, 0),
            ):
                with pytest.raises(ValueError, match="Could not parse"):
                    _parse_reminder_datetime("invalid datetime")


class TestBuildReminderSchedule:
    """Tests for _build_reminder_schedule function."""

    def test_build_one_time_schedule(self) -> None:
        """Should build schedule for one-time reminder."""
        with patch(
            "blacki.reminders.tools._parse_reminder_datetime",
            return_value=datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC),
        ):
            result = _build_reminder_schedule(
                reminder_datetime="2026-04-18 12:00",
                recurrence=None,
            )

        assert result["recurrence_rule"] is None
        assert result["recurrence_text"] is None
        assert result["timezone_name"] is None

    def test_build_recurring_schedule(self) -> None:
        """Should build schedule for recurring reminder."""
        mock_next_time = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)

        with patch(
            "blacki.reminders.tools.get_app_timezone",
            return_value=MagicMock(key="UTC"),
        ):
            with patch(
                "blacki.reminders.tools.validate_cron_expression",
                return_value="*/15 * * * *",
            ):
                with patch(
                    "blacki.reminders.tools.get_next_trigger_time",
                    return_value=mock_next_time,
                ):
                    result = _build_reminder_schedule(
                        reminder_datetime=None,
                        recurrence="*/15 * * * *",
                    )

        assert result["recurrence_rule"] == "*/15 * * * *"
        assert "cron" in result["recurrence_text"]

    def test_raises_on_both_datetime_and_recurrence(self) -> None:
        """Should raise if both datetime and recurrence provided."""
        with pytest.raises(ValueError, match="recurrence only"):
            _build_reminder_schedule(
                reminder_datetime="2026-04-18 12:00",
                recurrence="*/15 * * * *",
            )

    def test_raises_on_neither_datetime_nor_recurrence(self) -> None:
        """Should raise if neither datetime nor recurrence provided."""
        with pytest.raises(ValueError, match="need reminder_datetime"):
            _build_reminder_schedule(
                reminder_datetime=None,
                recurrence=None,
            )


class TestFormatReminder:
    """Tests for _format_reminder function."""

    def test_format_one_time_reminder(self) -> None:
        """Should format a one-time reminder."""
        reminder = Reminder(
            id=1,
            user_id="user1",
            message="Test reminder",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        with patch(
            "blacki.reminders.tools.format_stored_instant_for_display",
            return_value="2026-04-18 12:00:00 UTC",
        ):
            result = _format_reminder(reminder)

        assert result["id"] == 1
        assert result["message"] == "Test reminder"
        assert result["is_recurring"] is False
        assert result["schedule_type"] == "one_time"

    def test_format_recurring_reminder(self) -> None:
        """Should format a recurring reminder."""
        reminder = Reminder(
            id=1,
            user_id="user1",
            message="Test reminder",
            trigger_time="2026-04-18T12:00:00+00:00",
            recurrence_rule="*/15 * * * *",
            recurrence_text="every 15 minutes",
            created_at="2026-04-18T10:00:00+00:00",
        )

        with patch(
            "blacki.reminders.tools.format_stored_instant_for_display",
            return_value="2026-04-18 12:00:00 UTC",
        ):
            result = _format_reminder(reminder)

        assert result["is_recurring"] is True
        assert result["schedule_type"] == "recurring"
        assert result["recurrence"] == "every 15 minutes"
