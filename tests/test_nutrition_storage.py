"""Tests for NutritionStorage: the durable meal export queue."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

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
