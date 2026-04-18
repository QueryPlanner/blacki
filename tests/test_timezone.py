"""Unit tests for timezone utilities."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from blacki.utils.timezone import (
    format_stored_instant_for_display,
    get_app_timezone,
    naive_local_now,
    now_utc,
    utc_iso_seconds,
)


class TestGetAppTimezone:
    """Tests for get_app_timezone function."""

    def test_returns_utc_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return UTC timezone when AGENT_TIMEZONE is not set."""
        monkeypatch.delenv("AGENT_TIMEZONE", raising=False)
        tz = get_app_timezone()
        assert tz == ZoneInfo("UTC")

    def test_returns_configured_timezone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return configured timezone when AGENT_TIMEZONE is set."""
        monkeypatch.setenv("AGENT_TIMEZONE", "Asia/Kolkata")
        tz = get_app_timezone()
        assert tz == ZoneInfo("Asia/Kolkata")

    def test_falls_back_to_utc_on_invalid_timezone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fall back to UTC for invalid timezone names."""
        monkeypatch.setenv("AGENT_TIMEZONE", "Invalid/Timezone")
        tz = get_app_timezone()
        assert tz == ZoneInfo("UTC")

    def test_handles_empty_timezone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return UTC when AGENT_TIMEZONE is empty string."""
        monkeypatch.setenv("AGENT_TIMEZONE", "")
        tz = get_app_timezone()
        assert tz == ZoneInfo("UTC")


class TestNowUtc:
    """Tests for now_utc function."""

    def test_returns_timezone_aware_datetime(self) -> None:
        """Should return a datetime with UTC timezone."""
        result = now_utc()
        assert result.tzinfo is not None
        assert result.tzinfo == UTC

    def test_returns_recent_time(self) -> None:
        """Should return a time close to current time."""
        before = datetime.now(UTC)
        result = now_utc()
        after = datetime.now(UTC)

        assert before <= result <= after


class TestNaiveLocalNow:
    """Tests for naive_local_now function."""

    def test_returns_naive_datetime(self) -> None:
        """Should return a datetime without timezone info."""
        result = naive_local_now()
        assert result.tzinfo is None

    def test_returns_recent_time(self) -> None:
        """Should return a time close to current time."""
        before = datetime.now(UTC)
        result = naive_local_now()
        after = datetime.now(UTC)

        assert (result - before.replace(tzinfo=None)).total_seconds() < 1
        assert (after.replace(tzinfo=None) - result).total_seconds() < 1


class TestUtcIsoSeconds:
    """Tests for utc_iso_seconds function."""

    def test_formats_utc_datetime(self) -> None:
        """Should format UTC datetime with second precision."""
        dt = datetime(2026, 4, 18, 12, 30, 45, 123456, tzinfo=UTC)
        result = utc_iso_seconds(dt)

        assert result == "2026-04-18T12:30:45+00:00"

    def test_converts_non_utc_datetime(self) -> None:
        """Should convert non-UTC datetime to UTC before formatting."""
        dt = datetime(2026, 4, 18, 18, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        result = utc_iso_seconds(dt)

        assert result == "2026-04-18T12:30:00+00:00"

    def test_handles_naive_datetime(self) -> None:
        """Should treat naive datetime as UTC."""
        dt = datetime(2026, 4, 18, 12, 30, 45)
        result = utc_iso_seconds(dt)

        assert result == "2026-04-18T12:30:45+00:00"


class TestFormatStoredInstantForDisplay:
    """Tests for format_stored_instant_for_display function."""

    def test_formats_utc_iso_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should format UTC ISO string in app timezone."""
        monkeypatch.delenv("AGENT_TIMEZONE", raising=False)

        iso_string = "2026-04-18T12:30:45+00:00"
        result = format_stored_instant_for_display(iso_string)

        assert result == "2026-04-18 12:30:45 UTC"

    def test_formats_with_custom_timezone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should format in configured timezone."""
        monkeypatch.setenv("AGENT_TIMEZONE", "Asia/Kolkata")

        iso_string = "2026-04-18T12:30:45+00:00"
        result = format_stored_instant_for_display(iso_string)

        assert result == "2026-04-18 18:00:45 Asia/Kolkata"

    def test_handles_z_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should handle ISO string with Z suffix."""
        monkeypatch.delenv("AGENT_TIMEZONE", raising=False)

        iso_string = "2026-04-18T12:30:45Z"
        result = format_stored_instant_for_display(iso_string)

        assert result == "2026-04-18 12:30:45 UTC"

    def test_handles_naive_iso_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should treat naive ISO string as UTC."""
        monkeypatch.delenv("AGENT_TIMEZONE", raising=False)

        iso_string = "2026-04-18T12:30:45"
        result = format_stored_instant_for_display(iso_string)

        assert result == "2026-04-18 12:30:45 UTC"
