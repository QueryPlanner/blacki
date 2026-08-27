"""Tests for NutritionStorage: the durable meal export queue."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

from blacki.health.storage import SqliteGoogleHealthStorage

USER_ID = "telegram-chat-1"
OWNER_ID = "telegram-chat-1"
HEALTH_USER_ID = "health-user-1"
PAYLOAD: dict[str, Any] = {
    "nutritionLog": {
        "interval": {
            "startTime": "2026-01-01T12:00:00Z",
            "endTime": "2026-01-01T12:00:01Z",
            "startUtcOffset": "0s",
            "endUtcOffset": "0s",
        },
        "foodDisplayName": "Oatmeal",
        "energy": {"kcal": 300},
    }
}


@pytest.fixture
async def storage() -> AsyncGenerator[SqliteGoogleHealthStorage, None]:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    store = SqliteGoogleHealthStorage(conn, asyncio.Lock())
    await store.initialize()
    yield store
    await store.close()
    await conn.close()


# --- enqueue() validation guards -------------------------------------------


async def test_enqueue_rejects_unsupported_operation(
    storage: SqliteGoogleHealthStorage,
) -> None:
    with pytest.raises(ValueError, match="unsupported nutrition export operation"):
        await storage.nutrition.enqueue(
            meal_id=1,
            owner_id=OWNER_ID,
            telegram_user_id=USER_ID,
            health_user_id=HEALTH_USER_ID,
            payload=PAYLOAD,
            operation="sync",
        )


async def test_enqueue_rejects_owner_change(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=2,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    with pytest.raises(ValueError, match="owner cannot change"):
        await storage.nutrition.enqueue(
            meal_id=2,
            owner_id="a-different-owner",
            telegram_user_id=USER_ID,
            health_user_id=HEALTH_USER_ID,
            payload=PAYLOAD,
            operation="upsert",
        )


async def test_enqueue_rejects_telegram_user_change(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=3,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    with pytest.raises(ValueError, match="identity cannot change"):
        await storage.nutrition.enqueue(
            meal_id=3,
            owner_id=OWNER_ID,
            telegram_user_id="a-different-chat",
            health_user_id=HEALTH_USER_ID,
            payload=PAYLOAD,
            operation="upsert",
        )


async def test_enqueue_rejects_health_account_change(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=4,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    with pytest.raises(ValueError, match="account cannot change"):
        await storage.nutrition.enqueue(
            meal_id=4,
            owner_id=OWNER_ID,
            telegram_user_id=USER_ID,
            health_user_id="a-different-health-account",
            payload=PAYLOAD,
            operation="upsert",
        )


async def test_enqueue_upsert_requires_payload(
    storage: SqliteGoogleHealthStorage,
) -> None:
    with pytest.raises(ValueError, match="upsert requires a payload"):
        await storage.nutrition.enqueue(
            meal_id=5,
            owner_id=OWNER_ID,
            telegram_user_id=USER_ID,
            health_user_id=HEALTH_USER_ID,
            payload=None,
            operation="upsert",
        )


# --- latest_payload() --------------------------------------------------


async def test_latest_payload_returns_none_without_revisions(
    storage: SqliteGoogleHealthStorage,
) -> None:
    assert await storage.nutrition.latest_payload(999) is None


async def test_latest_payload_skips_null_and_returns_older(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=20,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    # A subsequent delete records a revision with payload_json=None, but the
    # older create revision's payload should still be found and returned.
    await storage.nutrition.enqueue(
        meal_id=20,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name="users/health-user-1/dataTypes/nutrition-log/dataPoints/x",
    )

    assert await storage.nutrition.latest_payload(20) == PAYLOAD


async def test_latest_payload_skips_malformed_json(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=21,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    # Simulate a corrupted newer revision row without going through enqueue.
    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_revisions
            (meal_id, resource_name, operation, payload_json, state)
        VALUES (?, ?, ?, ?, 'queued')
        """,
        (21, "resource-x", "upsert", "{not valid json"),
    )

    assert await storage.nutrition.latest_payload(21) == PAYLOAD


async def test_latest_payload_returns_none_when_nothing_decodes_to_dict(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_revisions
            (meal_id, resource_name, operation, payload_json, state)
        VALUES (?, ?, ?, ?, 'queued')
        """,
        (22, "resource-a", "delete", None),
    )
    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_revisions
            (meal_id, resource_name, operation, payload_json, state)
        VALUES (?, ?, ?, ?, 'queued')
        """,
        (22, "resource-b", "upsert", "[1, 2, 3]"),
    )
    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_revisions
            (meal_id, resource_name, operation, payload_json, state)
        VALUES (?, ?, ?, ?, 'queued')
        """,
        (22, "resource-c", "upsert", "not json at all"),
    )

    assert await storage.nutrition.latest_payload(22) is None


# --- result() ------------------------------------------------------------


async def test_result_without_expected_revision_ignores_guard(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """The pre-feature calling convention: no guard is applied at all."""
    await storage.nutrition.enqueue(
        meal_id=30,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )

    updated = await storage.nutrition.result(30, "synced")

    assert updated is True
    row = await storage.nutrition.meal(30)
    assert row is not None
    assert row["status"] == "synced"


async def test_result_expected_revision_none_matches_delete(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=31,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=None,
    )
    row = await storage.nutrition.meal(31)
    assert row is not None
    assert row["desired_revision"] is None

    updated = await storage.nutrition.result(31, "synced", expected_revision=None)

    assert updated is True
    row = await storage.nutrition.meal(31)
    assert row is not None
    assert row["status"] == "synced"


async def test_result_blocks_stale_write_when_revision_changed(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=32,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    original_row = await storage.nutrition.meal(32)
    assert original_row is not None
    original_revision = original_row["desired_revision"]

    # A newer edit lands before the worker records its result.
    await storage.nutrition.enqueue(
        meal_id=32,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    newer_row = await storage.nutrition.meal(32)
    assert newer_row is not None
    assert newer_row["desired_revision"] != original_revision

    updated = await storage.nutrition.result(
        32, "synced", expected_revision=original_revision
    )

    assert updated is False
    row = await storage.nutrition.meal(32)
    assert row is not None
    assert row["status"] == "pending"


async def test_standalone_state_transaction_rolls_back_on_failure(
    storage: SqliteGoogleHealthStorage,
) -> None:
    with (
        patch.object(
            storage.nutrition,
            "_revision_state_unlocked",
            new=AsyncMock(side_effect=RuntimeError("state write failed")),
        ),
        pytest.raises(RuntimeError, match="state write failed"),
    ):
        await storage.nutrition.revision_state(1, "in_flight")

    assert await storage.nutrition.meal(1) is None


async def test_retry_failed_requeues_matching_revision_and_resets_backoff(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=33,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    row = await storage.nutrition.meal(33)
    assert row is not None
    desired_revision = str(row["desired_revision"])
    revision = (await storage.nutrition.revisions(33))[0]
    await storage.nutrition.revision_state(int(revision["sequence"]), "failed")
    await storage.nutrition.result(
        33,
        "failed",
        error="invalid_argument",
        next_attempt=999,
        expected_revision=desired_revision,
    )

    requeued = await storage.nutrition.retry_failed(USER_ID, HEALTH_USER_ID)

    assert requeued == 1
    row = await storage.nutrition.meal(33)
    assert row is not None
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert row["next_attempt"] == 0
    assert row["error_code"] is None
    assert (await storage.nutrition.revisions(33))[0]["state"] == "queued"


async def test_retry_failed_does_not_touch_non_failed_or_mismatched_work(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=34,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.enqueue(
        meal_id=35,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.enqueue(
        meal_id=36,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    for meal_id, state in ((34, "queued"), (35, "in_flight"), (36, "failed")):
        revision = (await storage.nutrition.revisions(meal_id))[0]
        await storage.nutrition.revision_state(int(revision["sequence"]), state)
        await storage.nutrition.result(
            meal_id, "pending" if meal_id != 36 else "failed"
        )
    await storage.nutrition._conn.execute(
        "UPDATE nutrition_exports SET status = 'authorization_required' "
        "WHERE meal_id = 36"
    )

    assert await storage.nutrition.retry_failed(USER_ID, HEALTH_USER_ID) == 0
    assert (await storage.nutrition.revisions(36))[0]["state"] == "failed"


# --- counts() --------------------------------------------------------------


async def test_counts_returns_empty_dict_for_unknown_user(
    storage: SqliteGoogleHealthStorage,
) -> None:
    assert await storage.nutrition.counts("no-such-user") == {}


async def test_counts_excludes_cancelled_rows(
    storage: SqliteGoogleHealthStorage,
) -> None:
    counts_user = "counts-user"
    await storage.nutrition.enqueue(
        meal_id=40,
        owner_id=counts_user,
        telegram_user_id=counts_user,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.enqueue(
        meal_id=41,
        owner_id=counts_user,
        telegram_user_id=counts_user,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.enqueue(
        meal_id=42,
        owner_id=counts_user,
        telegram_user_id=counts_user,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition._conn.execute(
        "UPDATE nutrition_exports SET status = 'synced' WHERE meal_id = ?", (41,)
    )
    await storage.nutrition._conn.execute(
        "UPDATE nutrition_exports SET status = 'cancelled' WHERE meal_id = ?", (42,)
    )

    counts = await storage.nutrition.counts(counts_user)

    assert counts == {"pending": 1, "synced": 1}
    assert "cancelled" not in counts


async def test_has_other_account_detects_retained_account_state(
    storage: SqliteGoogleHealthStorage,
) -> None:
    assert not await storage.nutrition.has_other_account(USER_ID, HEALTH_USER_ID)
    await storage.nutrition.enqueue(
        meal_id=42,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )

    assert not await storage.nutrition.has_other_account(USER_ID, HEALTH_USER_ID)
    assert await storage.nutrition.has_other_account(USER_ID, "other-health")


async def test_latest_remote_resource_filters_revision_accounts_and_null_names(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=50,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    resource_name = (await storage.nutrition.revisions(50))[0]["resource_name"]
    await storage.nutrition.revision_state(1, "synced")
    await storage.nutrition.enqueue(
        meal_id=50,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_revisions
            (meal_id, resource_name, operation, payload_json, state)
        VALUES (?, NULL, 'upsert', ?, 'synced')
        """,
        (50, "{}"),
    )

    assert await storage.nutrition.latest_remote_resource(50, "other-account") is None
    assert (
        await storage.nutrition.latest_remote_resource(50, HEALTH_USER_ID)
        == resource_name
    )

    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_export_history
            (meal_id, telegram_user_id, health_user_id, resource_name,
             operation, payload_json, state, backfill_version, updated_at)
        VALUES (?, ?, ?, ?, 'upsert', ?, 'synced', 1, 1)
        """,
        (
            51,
            USER_ID,
            HEALTH_USER_ID,
            "users/health-user-1/dataTypes/nutrition-log/dataPoints/point-51",
            "{}",
        ),
    )
    assert (
        await storage.nutrition.latest_remote_resource(51)
        == "users/health-user-1/dataTypes/nutrition-log/dataPoints/point-51"
    )


async def test_record_remote_result_ignores_nonterminal_and_invalid_revisions(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=52,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    queued_sequence = int((await storage.nutrition.revisions(52))[0]["sequence"])
    await storage.nutrition.record_remote_result(queued_sequence)

    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_revisions
            (meal_id, resource_name, operation, payload_json, state)
        VALUES (?, NULL, 'upsert', ?, 'synced')
        """,
        (52, "{}"),
    )
    null_sequence = int((await storage.nutrition.revisions(52))[-1]["sequence"])
    await storage.nutrition.record_remote_result(null_sequence)

    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_revisions
            (meal_id, resource_name, operation, payload_json, state)
        VALUES (?, 'not-a-resource', 'upsert', ?, 'synced')
        """,
        (52, "{}"),
    )
    invalid_sequence = int((await storage.nutrition.revisions(52))[-1]["sequence"])
    await storage.nutrition.record_remote_result(invalid_sequence)

    assert (
        await storage.nutrition._fetch_all(
            "SELECT * FROM nutrition_export_history WHERE meal_id = ?", (52,)
        )
        == []
    )


async def test_ensure_backfill_validates_identity_and_reuses_current_work(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=60,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    with pytest.raises(ValueError, match="owner cannot change"):
        await storage.nutrition.ensure_backfill_export(
            meal_id=60,
            owner_id="other-owner",
            telegram_user_id=USER_ID,
            health_user_id=HEALTH_USER_ID,
            payload=PAYLOAD,
        )
    with pytest.raises(ValueError, match="identity cannot change"):
        await storage.nutrition.ensure_backfill_export(
            meal_id=60,
            owner_id=OWNER_ID,
            telegram_user_id="other-user",
            health_user_id=HEALTH_USER_ID,
            payload=PAYLOAD,
        )
    with pytest.raises(ValueError, match="account cannot change"):
        await storage.nutrition.ensure_backfill_export(
            meal_id=60,
            owner_id=OWNER_ID,
            telegram_user_id=USER_ID,
            health_user_id="other-account",
            payload=PAYLOAD,
        )
    assert (
        await storage.nutrition.ensure_backfill_export(
            meal_id=60,
            owner_id=OWNER_ID,
            telegram_user_id=USER_ID,
            health_user_id=HEALTH_USER_ID,
            payload=PAYLOAD,
        )
        == "existing"
    )


async def test_ensure_backfill_restores_history_without_current_export_row(
    storage: SqliteGoogleHealthStorage,
) -> None:
    resource_name = "users/health-user-1/dataTypes/nutrition-log/dataPoints/point-61"
    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_export_history
            (meal_id, telegram_user_id, health_user_id, resource_name,
             operation, payload_json, state, backfill_version, updated_at)
        VALUES (?, ?, ?, ?, 'upsert', ?, 'synced', 1, 1)
        """,
        (61, USER_ID, HEALTH_USER_ID, resource_name, json.dumps(PAYLOAD)),
    )

    result = await storage.nutrition.ensure_backfill_export(
        meal_id=61,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
    )

    assert result == "history"
    row = await storage.nutrition.meal(61)
    assert row is not None
    assert row["status"] == "synced"
    assert row["desired_revision"] == resource_name


async def test_ensure_backfill_restores_synced_revision_when_history_was_not_recorded(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition.enqueue(
        meal_id=62,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    revision = (await storage.nutrition.revisions(62))[0]
    await storage.nutrition.revision_state(int(revision["sequence"]), "synced")
    await storage.nutrition.cancel(USER_ID)

    result = await storage.nutrition.ensure_backfill_export(
        meal_id=62,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
    )

    assert result == "history"
    history = await storage.nutrition._fetch_one(
        "SELECT state FROM nutrition_export_history WHERE meal_id = ?", (62,)
    )
    assert history is not None
    assert history["state"] == "synced"


async def test_ensure_backfill_handles_malformed_history_payload(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_export_history
            (meal_id, telegram_user_id, health_user_id, resource_name,
             operation, payload_json, state, backfill_version, updated_at)
        VALUES (?, ?, ?, ?, 'upsert', '{bad', 'synced', 1, 1)
        """,
        (
            63,
            USER_ID,
            HEALTH_USER_ID,
            "users/health-user-1/dataTypes/nutrition-log/dataPoints/point-63",
        ),
    )

    result = await storage.nutrition.ensure_backfill_export(
        meal_id=63,
        owner_id=OWNER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
    )

    assert result == "queued"
    assert [
        revision["operation"] for revision in await storage.nutrition.revisions(63)
    ] == ["delete", "upsert"]


async def test_retry_failed_skips_missing_revision_and_rolls_back_failures(
    storage: SqliteGoogleHealthStorage,
) -> None:
    await storage.nutrition._conn.execute(
        """
        INSERT INTO nutrition_exports
            (meal_id, owner_id, telegram_user_id, health_user_id,
             desired_revision, desired_operation, status)
        VALUES (?, ?, ?, ?, 'missing-resource', 'upsert', 'failed')
        """,
        (70, OWNER_ID, USER_ID, HEALTH_USER_ID),
    )
    assert await storage.nutrition.retry_failed(USER_ID, HEALTH_USER_ID) == 0

    with (
        patch.object(
            storage.nutrition,
            "_fetch_all",
            new=AsyncMock(side_effect=RuntimeError("database failure")),
        ),
        pytest.raises(RuntimeError, match="database failure"),
    ):
        await storage.nutrition.retry_failed(USER_ID, HEALTH_USER_ID)


async def test_backfill_ledger_claim_resume_and_expiry(
    storage: SqliteGoogleHealthStorage,
) -> None:
    user_id = "telegram-chat-7"
    health_user_id = "health-account-7"
    claimed = await storage.nutrition.claim_backfill(
        user_id, health_user_id, 10, now=100, lease_seconds=10
    )
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["high_water_meal_id"] == 10
    assert (
        await storage.nutrition.claim_backfill(
            user_id, health_user_id, 99, now=105, lease_seconds=10
        )
        is None
    )
    resumed = await storage.nutrition.claim_backfill(
        user_id, health_user_id, 99, now=110, lease_seconds=10
    )
    assert resumed is not None
    assert resumed["high_water_meal_id"] == 10

    assert await storage.nutrition.advance_backfill(
        user_id,
        health_user_id,
        5,
        queued_count=2,
        skipped_count=1,
        now=200,
        lease_seconds=10,
    )
    assert await storage.nutrition.advance_backfill(
        user_id,
        health_user_id,
        10,
        queued_count=1,
        skipped_count=0,
        now=300,
        lease_seconds=10,
    )
    completed = await storage.nutrition.get_backfill(user_id, health_user_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["queued_count"] == 3
    assert completed["skipped_count"] == 1
    assert (
        await storage.nutrition.claim_backfill(user_id, health_user_id, 10, now=400)
        is None
    )


async def test_cancelled_backfill_recaptures_new_high_water_mark(
    storage: SqliteGoogleHealthStorage,
) -> None:
    user_id = "telegram-chat-11"
    health_user_id = "health-account-11"
    await storage.nutrition.claim_backfill(user_id, health_user_id, 10)
    await storage.nutrition.advance_backfill(
        user_id,
        health_user_id,
        10,
        queued_count=1,
        skipped_count=0,
    )

    await storage.nutrition.cancel(user_id)
    reclaimed = await storage.nutrition.claim_backfill(user_id, health_user_id, 20)

    assert reclaimed is not None
    assert reclaimed["high_water_meal_id"] == 20
    assert reclaimed["cursor_meal_id"] == 0


async def test_backfill_ledger_handles_unknown_rows_and_transaction_errors(
    storage: SqliteGoogleHealthStorage,
) -> None:
    assert not await storage.nutrition.advance_backfill(
        "telegram-chat-8",
        "health-account-8",
        1,
        queued_count=0,
        skipped_count=0,
    )
    with (
        patch.object(
            storage.nutrition,
            "_fetch_one",
            new=AsyncMock(side_effect=RuntimeError("claim failure")),
        ),
        pytest.raises(RuntimeError, match="claim failure"),
    ):
        await storage.nutrition.claim_backfill("telegram-chat-9", "health-account-9", 1)

    user_id = "telegram-chat-10"
    health_user_id = "health-account-10"
    await storage.nutrition.claim_backfill(user_id, health_user_id, 1)
    with (
        patch.object(
            storage.nutrition,
            "_advance_backfill_unlocked",
            new=AsyncMock(side_effect=RuntimeError("advance failure")),
        ),
        pytest.raises(RuntimeError, match="advance failure"),
    ):
        await storage.nutrition.advance_backfill(
            user_id,
            health_user_id,
            1,
            queued_count=0,
            skipped_count=0,
        )
    await storage.nutrition.fail_backfill(
        user_id, health_user_id, "bad\nprovider error", now=500
    )
    failed = await storage.nutrition.get_backfill(user_id, health_user_id)
    assert failed is not None
    assert failed["status"] == "pending"
    assert failed["last_error"] == "backfill_error"
