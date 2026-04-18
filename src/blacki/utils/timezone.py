"""Application timezone utilities for reminders and display.

Defaults to UTC. Override with AGENT_TIMEZONE (IANA name, e.g. Asia/Kolkata).
"""

import os
from datetime import UTC, datetime
from typing import Final
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE_NAME: Final = "UTC"


def get_app_timezone() -> ZoneInfo:
    """Resolved ZoneInfo for user-facing times and relative date parsing."""
    name = (
        os.environ.get("AGENT_TIMEZONE", DEFAULT_TIMEZONE_NAME).strip()
        or DEFAULT_TIMEZONE_NAME
    )
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo(DEFAULT_TIMEZONE_NAME)


def now_utc() -> datetime:
    """Current instant in UTC (for storage and due-time comparison)."""
    return datetime.now(UTC)


def naive_local_now() -> datetime:
    """Wall-clock 'now' in the app timezone, naive (for dateparser RELATIVE_BASE)."""
    return datetime.now(get_app_timezone()).replace(tzinfo=None)


def utc_iso_seconds(dt: datetime) -> str:
    """Normalize to UTC and serialize with second precision."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def format_stored_instant_for_display(iso_string: str) -> str:
    """Format a stored ISO instant (UTC or offset) in the app timezone with seconds."""
    tz = get_app_timezone()
    text = iso_string.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local = parsed.astimezone(tz)
    tz_label = tz.key if tz.key else "UTC"
    return local.strftime(f"%Y-%m-%d %H:%M:%S {tz_label}")
