"""Helpers for recurring reminder schedules and cron validation."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from blacki.utils.timezone import now_utc


@dataclass(frozen=True)
class RecurringSchedule:
    """Declarative recurring schedule metadata (cron + timezone)."""

    cron_expression: str
    description: str
    timezone_name: str


def validate_cron_expression(cron_expression: str, timezone_name: str) -> str:
    """Validate a five-field cron expression and return the normalized text."""
    normalized_expression = " ".join(cron_expression.split())
    CronTrigger.from_crontab(
        normalized_expression,
        timezone=ZoneInfo(timezone_name),
    )
    return normalized_expression


def get_next_trigger_time(
    cron_expression: str,
    timezone_name: str,
    reference_time: datetime | None = None,
) -> datetime:
    """Return the next UTC fire time strictly after the reference instant."""
    normalized_expression = validate_cron_expression(cron_expression, timezone_name)
    utc_reference_time = reference_time or now_utc()
    if utc_reference_time.tzinfo is None:
        utc_reference_time = utc_reference_time.replace(tzinfo=UTC)

    timezone = ZoneInfo(timezone_name)
    local_reference_time = utc_reference_time.astimezone(timezone)

    trigger = CronTrigger.from_crontab(
        normalized_expression,
        timezone=timezone,
    )
    next_fire_time = cast(
        datetime | None,
        trigger.get_next_fire_time(
            previous_fire_time=None,
            now=local_reference_time,
        ),
    )

    if next_fire_time is None:
        raise ValueError(
            f"Recurring schedule has no future fire time: {normalized_expression}"
        )

    if next_fire_time.tzinfo is None:
        next_fire_time = next_fire_time.replace(tzinfo=timezone)

    return next_fire_time.astimezone(UTC)
