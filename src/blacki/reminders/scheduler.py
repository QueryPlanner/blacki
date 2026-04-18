"""Scheduler for sending reminders via Telegram.

This module uses APScheduler to periodically check for due reminders
and sends them to users via Telegram.
"""

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from blacki.utils.timezone import get_app_timezone, now_utc, utc_iso_seconds

from .recurrence import get_next_trigger_time
from .storage import Reminder, get_storage

if TYPE_CHECKING:
    from blacki.telegram.api import TelegramApiClient

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 30
MISFIRE_GRACE_PERIOD = timedelta(minutes=10)

_TELEGRAM_USER_ID_PATTERN = re.compile(r"^telegram-chat-(\d+)(?:-thread-\d+)?$")


class ReminderScheduler:
    """Scheduler that sends due reminders via Telegram.

    This class manages an APScheduler instance that periodically checks
    for reminders that are due and sends them to users.

    Attributes:
        api: The TelegramApiClient instance for sending messages.
        scheduler: The APScheduler instance.
        storage: The ReminderStorage instance.
    """

    def __init__(self, api: "TelegramApiClient | None" = None) -> None:
        self._api: TelegramApiClient | None = api
        self.scheduler = AsyncIOScheduler(timezone=str(get_app_timezone()))
        self.storage = get_storage()
        self._running = False

    def set_api(self, api: "TelegramApiClient") -> None:
        """Set the TelegramApiClient instance."""
        self._api = api
        logger.info("Telegram API client set in scheduler")

    @property
    def api(self) -> "TelegramApiClient":
        """Get the API instance."""
        if self._api is None:
            raise RuntimeError("API not set. Call set_api() first.")
        return self._api

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        await self.storage.initialize()

        self.scheduler.add_job(
            self._check_and_send_reminders,
            trigger=IntervalTrigger(seconds=CHECK_INTERVAL_SECONDS),
            id="reminder_check",
            name="Check for due reminders",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        logger.info(
            f"Reminder scheduler started (checking every {CHECK_INTERVAL_SECONDS}s)"
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown(wait=True)
        self._running = False
        logger.info("Reminder scheduler stopped")

    async def _check_and_send_reminders(self) -> None:
        """Check for due reminders and send them."""
        try:
            current_time = now_utc()
            reminders = await self.storage.get_due_reminders()

            if not reminders:
                return

            fresh_reminders: list[Reminder] = []
            stale_reminders: list[Reminder] = []

            for reminder in reminders:
                if _is_stale_reminder(reminder.trigger_time, current_time):
                    stale_reminders.append(reminder)
                    continue

                fresh_reminders.append(reminder)

            if stale_reminders:
                logger.warning(
                    "Skipping %s stale reminder(s) older than %s",
                    len(stale_reminders),
                    MISFIRE_GRACE_PERIOD,
                )
                for reminder in stale_reminders:
                    await self._handle_stale_reminder(
                        reminder,
                        reference_time=current_time,
                    )

            if not fresh_reminders:
                return

            logger.info(f"Processing {len(fresh_reminders)} due reminder(s)")

            for reminder in fresh_reminders:
                await self._send_reminder(reminder)
        except Exception:
            logger.exception("Error checking reminders")

    async def _send_reminder(self, reminder: Reminder) -> None:
        """Send a reminder notification to the user."""
        if reminder.id is None:
            logger.error("Reminder has no ID, skipping")
            return

        reminder_delivery_succeeded = False
        reminder_reference_time = now_utc()

        try:
            chat_id = _extract_telegram_chat_id(reminder.user_id)
            if chat_id is None:
                logger.error(
                    f"Could not extract chat_id from user_id: {reminder.user_id}"
                )
                return

            text = f"⏰ *Reminder*\n\n{reminder.message}"
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
            )

            reminder_delivery_succeeded = True

        except Exception:
            logger.exception(
                f"Failed to send reminder {reminder.id} to user {reminder.user_id}"
            )
        finally:
            await self._complete_reminder_delivery(
                reminder,
                reference_time=reminder_reference_time,
            )

        if reminder_delivery_succeeded:
            logger.info(f"Sent reminder {reminder.id} to user {reminder.user_id}")

    async def _handle_stale_reminder(
        self,
        reminder: Reminder,
        reference_time: datetime,
    ) -> None:
        """Resolve an overdue reminder without delivering a stale notification."""
        if reminder.id is None:
            logger.error("Stale reminder has no ID, skipping")
            return

        await self._complete_reminder_delivery(
            reminder,
            reference_time=reference_time,
        )

    async def _complete_reminder_delivery(
        self,
        reminder: Reminder,
        reference_time: datetime | None = None,
    ) -> None:
        """Finalize a delivery by marking one-shot reminders sent or rescheduling."""
        if reminder.id is None:
            raise ValueError("Reminder must have an ID before completion")

        if not reminder.is_recurring:
            await self.storage.mark_sent(reminder.id)
            return

        await self._reschedule_recurring_reminder(reminder, reference_time)

    async def _reschedule_recurring_reminder(
        self,
        reminder: Reminder,
        reference_time: datetime | None = None,
    ) -> None:
        """Move a recurring reminder to the next future fire time."""
        if reminder.id is None:
            raise ValueError("Reminder must have an ID before rescheduling")

        timezone_name = reminder.timezone_name or str(get_app_timezone())
        effective_reference_time = reference_time or now_utc()
        next_trigger_time = get_next_trigger_time(
            reminder.recurrence_rule or "",
            timezone_name,
            reference_time=effective_reference_time,
        )
        await self.storage.reschedule_reminder(
            reminder.id,
            utc_iso_seconds(next_trigger_time),
        )

    async def schedule_reminder(
        self,
        user_id: str,
        message: str,
        trigger_time: datetime,
        recurrence_rule: str | None = None,
        recurrence_text: str | None = None,
        timezone_name: str | None = None,
    ) -> int:
        """Schedule a new reminder.

        Args:
            user_id: The Telegram chat ID of the user.
            message: The reminder message.
            trigger_time: When to send the reminder.

        Returns:
            The ID of the created reminder.
        """
        reminder = Reminder(
            user_id=user_id,
            message=message,
            trigger_time=utc_iso_seconds(trigger_time),
            recurrence_rule=recurrence_rule,
            recurrence_text=recurrence_text,
            timezone_name=timezone_name,
            created_at=now_utc().isoformat(timespec="seconds"),
        )

        reminder_id = await self.storage.add_reminder(reminder)
        return reminder_id

    async def get_user_reminders(
        self, user_id: str, include_sent: bool = False
    ) -> list[Reminder]:
        """Get all reminders for a user."""
        return await self.storage.get_user_reminders(user_id, include_sent)

    async def delete_reminder(self, reminder_id: int, user_id: str) -> bool:
        """Delete a reminder.

        Returns:
            True if deleted, False otherwise.
        """
        return await self.storage.delete_reminder(reminder_id, user_id)


_scheduler: ReminderScheduler | None = None


def get_scheduler() -> ReminderScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ReminderScheduler()
    return _scheduler


def _parse_stored_trigger_time(trigger_time: str) -> datetime:
    """Parse an ISO timestamp stored in reminder persistence."""
    normalized_trigger_time = trigger_time.replace("Z", "+00:00")
    parsed_trigger_time = datetime.fromisoformat(normalized_trigger_time)
    if parsed_trigger_time.tzinfo is None:
        return parsed_trigger_time.replace(tzinfo=UTC)
    return parsed_trigger_time.astimezone(UTC)


def _is_stale_reminder(trigger_time: str, current_time: datetime) -> bool:
    """Return True when a reminder is older than the misfire grace period."""
    parsed_trigger_time = _parse_stored_trigger_time(trigger_time)
    stale_cutoff_time = current_time - MISFIRE_GRACE_PERIOD
    return parsed_trigger_time < stale_cutoff_time


def _extract_telegram_chat_id(user_id: str) -> int | None:
    """Extract the numeric Telegram chat_id from a prefixed user_id.

    Args:
        user_id: The user_id string (e.g., "telegram-chat-1399736563" or
            "telegram-chat-1399736563-thread-42").

    Returns:
        The numeric chat_id, or None if the user_id doesn't match the expected pattern.
    """
    match = _TELEGRAM_USER_ID_PATTERN.match(user_id)
    if not match:
        return None
    return int(match.group(1))
