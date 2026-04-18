"""Unit tests for reminder recurrence utilities."""

from datetime import UTC, datetime

import pytest

from blacki.reminders.recurrence import (
    RecurringSchedule,
    get_next_trigger_time,
    validate_cron_expression,
)


class TestValidateCronExpression:
    """Tests for validate_cron_expression function."""

    def test_validates_standard_cron(self) -> None:
        """Should validate and normalize a standard 5-field cron expression."""
        result = validate_cron_expression("* * * * *", "UTC")
        assert result == "* * * * *"

    def test_normalizes_whitespace(self) -> None:
        """Should normalize whitespace in cron expression."""
        result = validate_cron_expression("  *   *  *  *  *  ", "UTC")
        assert result == "* * * * *"

    def test_validates_specific_time(self) -> None:
        """Should validate a specific time cron expression."""
        result = validate_cron_expression("30 8 * * 1", "Asia/Kolkata")
        assert result == "30 8 * * 1"

    def test_raises_on_invalid_cron(self) -> None:
        """Should raise ValueError for invalid cron expression."""
        with pytest.raises(ValueError):
            validate_cron_expression("invalid cron", "UTC")

    def test_raises_on_six_fields(self) -> None:
        """Should raise ValueError for 6-field cron expression."""
        with pytest.raises(ValueError):
            validate_cron_expression("* * * * * *", "UTC")


class TestGetNextTriggerTime:
    """Tests for get_next_trigger_time function."""

    def test_returns_future_time(self) -> None:
        """Should return a time in the future."""
        reference_time = datetime(2026, 4, 18, 12, 0, 1, tzinfo=UTC)
        result = get_next_trigger_time(
            "* * * * *",
            "UTC",
            reference_time=reference_time,
        )

        assert result > reference_time
        assert result.tzinfo == UTC

    def test_respects_timezone(self) -> None:
        """Should calculate next trigger in specified timezone."""
        reference_time = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)
        result = get_next_trigger_time(
            "0 18 * * *",
            "Asia/Kolkata",
            reference_time=reference_time,
        )

        assert result > reference_time
        assert result.tzinfo == UTC

    def test_every_minute_cron(self) -> None:
        """Should return next minute for every-minute cron."""
        reference_time = datetime(2026, 4, 18, 12, 30, 45, tzinfo=UTC)
        result = get_next_trigger_time(
            "* * * * *",
            "UTC",
            reference_time=reference_time,
        )

        expected = datetime(2026, 4, 18, 12, 31, 0, tzinfo=UTC)
        assert result == expected

    def test_hourly_cron(self) -> None:
        """Should return next hour for hourly cron."""
        reference_time = datetime(2026, 4, 18, 12, 30, 0, tzinfo=UTC)
        result = get_next_trigger_time(
            "0 * * * *",
            "UTC",
            reference_time=reference_time,
        )

        expected = datetime(2026, 4, 18, 13, 0, 0, tzinfo=UTC)
        assert result == expected

    def test_raises_on_invalid_cron(self) -> None:
        """Should raise ValueError for invalid cron expression."""
        reference_time = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)
        with pytest.raises(ValueError):
            get_next_trigger_time(
                "invalid",
                "UTC",
                reference_time=reference_time,
            )

    def test_handles_naive_reference_time(self) -> None:
        """Should handle reference time without timezone."""
        naive_reference_time = datetime(2026, 4, 18, 12, 0, 30)
        result = get_next_trigger_time(
            "* * * * *",
            "UTC",
            reference_time=naive_reference_time,
        )

        assert result.tzinfo == UTC
        assert result == datetime(2026, 4, 18, 12, 1, 0, tzinfo=UTC)


class TestGetNextTriggerTimeEdgeCases:
    """Edge case tests for get_next_trigger_time function."""

    def test_handles_naive_next_fire_time(self) -> None:
        """Should handle next_fire_time without timezone."""
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        reference_time = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)
        naive_next_time = datetime(2026, 4, 18, 12, 1, 0)

        with patch("blacki.reminders.recurrence.CronTrigger") as mock_cron_trigger_cls:
            mock_trigger = MagicMock()
            mock_trigger.get_next_fire_time.return_value = naive_next_time
            mock_cron_trigger_cls.from_crontab.return_value = mock_trigger

            result = get_next_trigger_time(
                "* * * * *",
                "UTC",
                reference_time=reference_time,
            )

            assert result.tzinfo == UTC

    def test_raises_when_no_future_fire_time(self) -> None:
        """Should raise ValueError when cron has no future fire time."""
        from unittest.mock import MagicMock, patch

        reference_time = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)

        with patch("blacki.reminders.recurrence.CronTrigger") as mock_cron_trigger_cls:
            mock_trigger = MagicMock()
            mock_trigger.get_next_fire_time.return_value = None
            mock_cron_trigger_cls.from_crontab.return_value = mock_trigger

            with pytest.raises(ValueError, match="no future fire time"):
                get_next_trigger_time(
                    "* * * * *",
                    "UTC",
                    reference_time=reference_time,
                )


class TestRecurringSchedule:
    """Tests for RecurringSchedule dataclass."""

    def test_creates_schedule(self) -> None:
        """Should create a RecurringSchedule with all fields."""
        schedule = RecurringSchedule(
            cron_expression="*/15 * * * *",
            description="every 15 minutes",
            timezone_name="UTC",
        )

        assert schedule.cron_expression == "*/15 * * * *"
        assert schedule.description == "every 15 minutes"
        assert schedule.timezone_name == "UTC"

    def test_is_frozen(self) -> None:
        """Should be immutable (frozen dataclass)."""
        schedule = RecurringSchedule(
            cron_expression="* * * * *",
            description="test",
            timezone_name="UTC",
        )

        with pytest.raises(AttributeError):
            schedule.cron_expression = "changed"  # type: ignore[misc]
