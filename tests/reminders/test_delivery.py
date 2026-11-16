"""Exercise reminder delivery against the production SQLite connection."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from blacki.container import AppContainer
from blacki.reminders.scheduler import ReminderScheduler
from blacki.reminders.storage import Reminder
from blacki.storage.sqlite import create_connection

NOW = datetime(2026, 11, 14, 15, tzinfo=UTC)
OWNER = "telegram-chat-123456"


@pytest.fixture
async def scheduler(tmp_path: Path) -> AsyncIterator[ReminderScheduler]:
    conn = await create_connection(tmp_path / "tools.db")
    storage = AppContainer(conn).reminder_storage
    try:
        await storage.initialize()
        with patch("blacki.reminders.scheduler.get_storage", return_value=storage):
            yield ReminderScheduler()
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rule", "next_trigger"),
    [(None, NOW), ("0 15 * * *", NOW + timedelta(days=1))],
)
async def test_full_database_blocks_delivery_until_occurrence_can_be_saved(
    scheduler: ReminderScheduler,
    rule: str | None,
    next_trigger: datetime,
) -> None:
    reminder_id = await scheduler.schedule_reminder(
        OWNER, "Log dinner", NOW, recurrence_rule=rule, timezone_name="UTC"
    )
    attempted: list[Reminder] = []

    async def deliver(reminder: Reminder) -> None:
        attempted.append(reminder)

    scheduler.set_callback(deliver)
    conn = scheduler.storage.conn
    await conn.execute("CREATE TABLE quota_test (payload BLOB)")
    await conn.execute("""
        CREATE TEMP TRIGGER exhaust_quota BEFORE UPDATE ON reminders
        BEGIN
            INSERT INTO quota_test VALUES (zeroblob(1048576));
        END
    """)
    async with conn.execute("PRAGMA page_count") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    await conn.execute(f"PRAGMA max_page_count = {int(row[0])}")

    with (
        patch("blacki.reminders.scheduler.now_utc", return_value=NOW),
        patch("blacki.reminders.storage.now_utc", return_value=NOW),
    ):
        await scheduler._check_and_send_reminders()
        await scheduler._check_and_send_reminders()
        assert attempted == []
        [pending] = await scheduler.storage.get_due_reminders()
        assert pending.id == reminder_id
        assert pending.trigger_time == NOW.isoformat()

        await conn.execute("DROP TRIGGER exhaust_quota")
        await scheduler._check_and_send_reminders()
        await scheduler._check_and_send_reminders()

    assert [item.id for item in attempted] == [reminder_id]
    [saved] = await scheduler.get_user_reminders(OWNER, include_sent=True)
    assert saved.trigger_time == next_trigger.isoformat()
    assert saved.is_sent is (rule is None)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_stale_snapshot_cannot_deliver_an_occurrence_again(
    scheduler: ReminderScheduler, cancel: bool
) -> None:
    reminder_id = await scheduler.schedule_reminder(
        OWNER, "Log dinner", NOW, recurrence_rule="0 15 * * *", timezone_name="UTC"
    )
    [snapshot] = await scheduler.get_user_reminders(OWNER)
    deliveries: list[int | None] = []

    async def deliver(reminder: Reminder) -> None:
        deliveries.append(reminder.id)

    scheduler.set_callback(deliver)
    if cancel:
        assert await scheduler.delete_reminder(reminder_id, OWNER)

    with patch("blacki.reminders.scheduler.now_utc", return_value=NOW):
        await scheduler._send_reminder(snapshot)
        await scheduler._send_reminder(snapshot)

    assert deliveries == ([] if cancel else [reminder_id])
