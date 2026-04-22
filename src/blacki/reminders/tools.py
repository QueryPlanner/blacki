"""ADK tools for reminder scheduling and management."""

import logging
from datetime import UTC, datetime
from typing import Any

import dateparser  # type: ignore[import-untyped]
from google.adk.tools import ToolContext

from blacki.utils.timezone import (
    format_stored_instant_for_display,
    get_app_timezone,
    naive_local_now,
    now_utc,
    utc_iso_seconds,
)

from . import Reminder, get_scheduler
from .recurrence import (
    get_next_trigger_time,
    validate_cron_expression,
)

logger = logging.getLogger(__name__)

SUPPORTED_RECURRENCE_MESSAGE = (
    "Recurring reminders must use a 5-field cron expression in the app "
    "timezone, for example '* * * * *' for every minute, "
    "'*/15 * * * *' for every 15 minutes, or '30 8 * * 1' for Mondays "
    "at 08:30."
)


async def schedule_reminder(
    tool_context: ToolContext,
    message: str,
    reminder_datetime: str | int | float | None = None,
    recurrence: str | None = None,
) -> dict[str, Any]:
    """Schedule a reminder to be sent at a specific time.

    The reminder will be delivered through the agent as a Telegram message.
    The stored message is shown back to the agent when the reminder fires, so
    ``message`` should be a self-contained instruction for what the user should
    receive at delivery time.

    Args:
        tool_context: ADK ToolContext with user_id in state.
        message: Self-contained delivery instruction for the future reminder
            (max 500 characters).
        reminder_datetime: For one-time reminders, when to fire in the app
            local timezone (default UTC; set AGENT_TIMEZONE to change).
            Use wall-clock strings such as '2026-03-15 14:30', or relative
            phrases like 'in 30 minutes' or 'tomorrow at 9am'.
        recurrence: Optional recurring schedule as a 5-field cron expression in
            the app timezone. Examples: '* * * * *', '*/15 * * * *', '30 8 * * 1'.

    Returns:
        A dictionary with status, reminder_id, and confirmation message.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        logger.error("No user_id in tool context")
        return {
            "status": "error",
            "message": "Cannot schedule reminder: user not identified.",
        }

    try:
        reminder_schedule = _build_reminder_schedule(
            reminder_datetime=reminder_datetime,
            recurrence=recurrence,
        )
    except ValueError as e:
        logger.warning(
            "Failed to parse reminder schedule (datetime=%r recurrence=%r): %s",
            reminder_datetime,
            recurrence,
            e,
        )
        error_message = str(e)
        if not recurrence:
            error_message = (
                "Could not understand the time. Use wall-clock time, e.g. "
                "'2026-03-15 14:30', or relative phrases like "
                "'in 30 minutes' / 'tomorrow at 9am'."
            )
        return {
            "status": "error",
            "message": error_message,
        }

    if len(message) > 500:
        return {
            "status": "error",
            "message": "Reminder message too long (max 500 characters).",
        }

    if reminder_schedule["trigger_time"] <= now_utc():
        return {
            "status": "error",
            "message": "The reminder time must be in the future.",
        }

    try:
        scheduler = get_scheduler()
        reminder_id = await scheduler.schedule_reminder(
            user_id=user_id,
            message=message,
            trigger_time=reminder_schedule["trigger_time"],
            recurrence_rule=reminder_schedule["recurrence_rule"],
            recurrence_text=reminder_schedule["recurrence_text"],
            timezone_name=reminder_schedule["timezone_name"],
        )

        display_time = format_stored_instant_for_display(
            utc_iso_seconds(reminder_schedule["trigger_time"])
        )
        logger.info(
            "Scheduled reminder %s for user %s at %s (stored as UTC)",
            reminder_id,
            user_id,
            utc_iso_seconds(reminder_schedule["trigger_time"]),
        )

        confirmation_prefix = "Recurring reminder scheduled"
        if reminder_schedule["recurrence_rule"] is None:
            confirmation_prefix = "Reminder scheduled"

        recurrence_suffix = ""
        if reminder_schedule["recurrence_text"]:
            recurrence_suffix = f" ({reminder_schedule['recurrence_text']})"

        return {
            "status": "success",
            "reminder_id": reminder_id,
            "message": f"{confirmation_prefix}{recurrence_suffix}. Next send: "
            f"{display_time}. "
            f"I'll send you: '{message[:50]}{'...' if len(message) > 50 else ''}'",
        }
    except Exception as e:
        logger.exception("Failed to schedule reminder")
        return {
            "status": "error",
            "message": f"Failed to schedule reminder: {e}",
        }


async def list_reminders(
    tool_context: ToolContext,
    include_sent: bool = False,
) -> dict[str, Any]:
    """List all scheduled reminders for the user.

    Args:
        tool_context: ADK ToolContext with user_id in state.
        include_sent: Whether to include already-sent reminders.

    Returns:
        A dictionary with status and list of reminders.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "Cannot list reminders: user not identified.",
        }

    try:
        scheduler = get_scheduler()
        reminders = await scheduler.get_user_reminders(user_id, include_sent)

        if not reminders:
            return {
                "status": "success",
                "reminders": [],
                "message": "You have no scheduled reminders.",
            }

        formatted = [_format_reminder(r) for r in reminders]
        return {
            "status": "success",
            "reminders": formatted,
            "count": len(reminders),
            "message": f"You have {len(reminders)} reminder(s) scheduled.",
        }
    except Exception as e:
        logger.exception("Failed to list reminders")
        return {
            "status": "error",
            "message": f"Failed to list reminders: {e}",
        }


async def cancel_reminder(
    tool_context: ToolContext,
    reminder_id: int,
) -> dict[str, Any]:
    """Cancel a scheduled reminder.

    Args:
        tool_context: ADK ToolContext with user_id in state.
        reminder_id: The ID of the reminder to cancel.

    Returns:
        A dictionary with status and confirmation message.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "Cannot cancel reminder: user not identified.",
        }

    try:
        scheduler = get_scheduler()
        deleted = await scheduler.delete_reminder(reminder_id, user_id)

        if deleted:
            return {
                "status": "success",
                "message": f"Reminder {reminder_id} cancelled.",
            }
        else:
            return {
                "status": "error",
                "message": f"Reminder {reminder_id} not found "
                "or you don't have permission to cancel it.",
            }
    except Exception as e:
        logger.exception("Failed to cancel reminder")
        return {
            "status": "error",
            "message": f"Failed to cancel reminder: {e}",
        }


def _parse_reminder_datetime(datetime_val: str | int | float) -> datetime:
    """Parse natural-language or absolute datetimes in the app timezone, return UTC."""
    tz = get_app_timezone()
    tz_name = tz.key if tz.key else "UTC"
    datetime_str = str(datetime_val)
    parsed_time = dateparser.parse(
        datetime_str,
        settings={
            "PREFER_DATES_FROM": "future",
            "TIMEZONE": tz_name,
            "TO_TIMEZONE": tz_name,
            "RELATIVE_BASE": naive_local_now(),
        },
    )
    if not parsed_time:
        raise ValueError(f"Could not parse datetime: {datetime_str}")

    if parsed_time.tzinfo is None:
        parsed_time = parsed_time.replace(tzinfo=tz)
    else:
        parsed_time = parsed_time.astimezone(tz)

    return parsed_time.astimezone(UTC)  # type: ignore[no-any-return]


def _format_reminder(reminder: Reminder) -> dict[str, Any]:
    """Format a reminder for display."""
    next_trigger_time = format_stored_instant_for_display(reminder.trigger_time)
    return {
        "id": reminder.id,
        "message": reminder.message,
        "trigger_time": next_trigger_time,
        "next_trigger_time": next_trigger_time,
        "is_sent": reminder.is_sent,
        "is_recurring": reminder.is_recurring,
        "schedule_type": "recurring" if reminder.is_recurring else "one_time",
        "recurrence": reminder.recurrence_text,
    }


def _build_reminder_schedule(
    reminder_datetime: str | int | float | None,
    recurrence: str | None,
) -> dict[str, Any]:
    """Build the normalized schedule for one-shot or recurring reminders."""
    normalized_recurrence = (recurrence or "").strip()
    if not normalized_recurrence:
        if not reminder_datetime:
            raise ValueError(
                "One-time reminders need reminder_datetime. Use a time like "
                "'2026-03-15 14:30', 'in 30 minutes', or 'tomorrow at 9am'."
            )

        return {
            "trigger_time": _parse_reminder_datetime(reminder_datetime),
            "recurrence_rule": None,
            "recurrence_text": None,
            "timezone_name": None,
        }

    if reminder_datetime:
        raise ValueError(
            "Recurring reminders use recurrence only. Omit reminder_datetime "
            "and pass a 5-field cron expression. " + SUPPORTED_RECURRENCE_MESSAGE
        )

    recurring_schedule = _parse_recurring_schedule(normalized_recurrence)
    next_trigger_time = get_next_trigger_time(
        recurring_schedule.cron_expression,
        recurring_schedule.timezone_name,
    )

    return {
        "trigger_time": next_trigger_time,
        "recurrence_rule": recurring_schedule.cron_expression,
        "recurrence_text": recurring_schedule.description,
        "timezone_name": recurring_schedule.timezone_name,
    }


def _parse_recurring_schedule(recurrence: str) -> Any:
    """Validate a cron-style recurring schedule for reminder storage."""

    from .recurrence import RecurringSchedule

    timezone_name = get_app_timezone().key or "UTC"
    normalized_recurrence = " ".join(recurrence.strip().split())
    try:
        cron_expression = validate_cron_expression(normalized_recurrence, timezone_name)
    except ValueError as error:
        raise ValueError(
            "Could not understand the recurring schedule. "
            + SUPPORTED_RECURRENCE_MESSAGE
        ) from error

    return RecurringSchedule(
        cron_expression=cron_expression,
        description=f"cron: {cron_expression}",
        timezone_name=timezone_name,
    )
