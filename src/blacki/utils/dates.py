"""Shared date parsing utilities for tool modules."""

import dateparser  # type: ignore[import-untyped]

from blacki.utils.timezone import get_app_timezone, now_utc


def parse_date(date_str: str | None) -> str:
    """Parse a natural language date to YYYY-MM-DD local time."""
    tz = get_app_timezone()
    if not date_str or date_str.lower() in ("today", "now"):
        return now_utc().astimezone(tz).strftime("%Y-%m-%d")

    dt = dateparser.parse(  # pragma: no cover
        date_str,
        settings={"TIMEZONE": str(tz), "RETURN_AS_TIMEZONE_AWARE": True},
    )
    if not dt:  # pragma: no cover
        return now_utc().astimezone(tz).strftime("%Y-%m-%d")

    return str(dt.strftime("%Y-%m-%d"))  # pragma: no cover
