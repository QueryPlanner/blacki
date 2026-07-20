"""Shared date parsing utilities for tool modules."""

import dateparser  # type: ignore[import-untyped]

from blacki.utils.timezone import get_app_timezone, naive_local_now, now_utc


def parse_date(date_str: str | None) -> str:
    """Parse a natural-language date to local ``YYYY-MM-DD``.

    Raises:
        ValueError: If an explicit date cannot be parsed. Invalid input must not
            silently become today's date for a persistent tracking operation.
    """
    tz = get_app_timezone()
    if not date_str or date_str.lower() in ("today", "now"):
        return now_utc().astimezone(tz).strftime("%Y-%m-%d")

    dt = dateparser.parse(
        date_str,
        settings={
            "TIMEZONE": str(tz),
            "RETURN_AS_TIMEZONE_AWARE": True,
            "RELATIVE_BASE": naive_local_now(),
        },
    )
    if not dt:
        raise ValueError(f"Could not understand date: {date_str}")

    return str(dt.strftime("%Y-%m-%d"))
