"""Tests for shared application-timezone date parsing."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from blacki.utils.dates import parse_date


def test_parse_date_defaults_to_current_application_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_TIMEZONE", "America/New_York")
    instant = datetime(2025, 1, 15, 2, 0, tzinfo=UTC)

    with patch("blacki.utils.dates.now_utc", return_value=instant):
        assert parse_date(None) == "2025-01-14"
        assert parse_date("today") == "2025-01-14"


def test_parse_date_resolves_relative_dates_from_local_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_TIMEZONE", "UTC")
    relative_base = datetime(2025, 1, 15, 12, 0)

    with patch("blacki.utils.dates.naive_local_now", return_value=relative_base):
        assert parse_date("yesterday") == "2025-01-14"


def test_parse_date_rejects_unparseable_explicit_date() -> None:
    with pytest.raises(ValueError, match="Could not understand date"):
        parse_date("definitely-not-a-real-date")
