"""Tests for one-time Google Health nutrition backfill coordination."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from cryptography.fernet import Fernet

from blacki.calories.service import MealService
from blacki.calories.storage import CalorieEntry
from blacki.container import AppContainer
from blacki.health.config import (
    GOOGLE_HEALTH_NUTRITION_SCOPES,
    GOOGLE_HEALTH_READ_SCOPES,
    GoogleHealthConfig,
)
from blacki.health.nutrition_backfill import NutritionBackfillCoordinator
from blacki.health.nutrition_storage import BACKFILL_VERSION

USER_ID = "telegram-chat-42"
HEALTH_USER_ID = "google-account-42"


@pytest.fixture
async def container() -> AsyncGenerator[AppContainer, None]:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    app_container = AppContainer(conn=conn)
    await app_container.initialize_all_storages()
    yield app_container
    await app_container.close()


def _entry(user_id: str, number: int) -> CalorieEntry:
    return CalorieEntry(
        user_id=user_id,
        description=f"Meal {number}",
        calories=300 + number,
        logged_at="2026-08-27T08:00:00+00:00",
        logged_date="2026-08-27",
    )


def _config() -> GoogleHealthConfig:
    return GoogleHealthConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/callback",
        token_encryption_key=Fernet.generate_key().decode(),
    )


async def _connect(
    container: AppContainer,
    *,
    health_user_id: str = HEALTH_USER_ID,
    scopes: tuple[str, ...] = GOOGLE_HEALTH_NUTRITION_SCOPES,
) -> None:
    config = _config()
    await container.google_health_storage.upsert_connection(
        telegram_user_id=USER_ID,
        encrypted_refresh_token=config.cipher.encrypt("refresh-token"),
        health_user_id=health_user_id,
        legacy_fitbit_user_id=None,
        scopes=scopes,
    )


async def _add_meals(container: AppContainer, user_ids: list[str]) -> list[int]:
    ids: list[int] = []
    for number, user_id in enumerate(user_ids, start=1):
        ids.append(await container.calorie_storage.add_entry(_entry(user_id, number)))
    return ids


@pytest.mark.asyncio
async def test_backfill_queues_valid_direct_and_topic_meals_once(
    container: AppContainer,
) -> None:
    ids = await _add_meals(
        container,
        [
            USER_ID,
            "telegram-chat-42-thread-7",
            "telegram-chat-420",
            "telegram-chat-42-thread-not-a-number",
            "telegram-chat-42-thread-8-extra",
            "telegram-chat--100",
        ],
    )
    await _connect(container)
    wake = MagicMock()
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage,
        container.calorie_storage,
        wake=wake,
    )

    result = await coordinator.run_user(USER_ID)

    assert result.status == "completed"
    assert result.queued_count == 2
    assert wake.call_count == 1
    nutrition = container.google_health_storage.nutrition
    for entry_id in ids[:2]:
        row = await nutrition.meal(entry_id)
        assert row is not None
        assert row["telegram_user_id"] == USER_ID
        assert row["status"] == "pending"
        assert len(await nutrition.revisions(entry_id)) == 1
    for entry_id in ids[2:]:
        assert await nutrition.meal(entry_id) is None

    second = await coordinator.run_user(USER_ID)

    assert second.status == "completed"
    assert second.queued_count == 2
    revision_count = 0
    for entry_id in ids[:2]:
        revision_count += len(await nutrition.revisions(entry_id))
    assert revision_count == 2
    ledger = await nutrition.get_backfill(USER_ID, HEALTH_USER_ID, BACKFILL_VERSION)
    assert ledger is not None
    assert ledger["status"] == "completed"
    assert ledger["cursor_meal_id"] == ledger["high_water_meal_id"]


@pytest.mark.asyncio
async def test_backfill_skips_read_only_connection_until_scopes_are_granted(
    container: AppContainer,
) -> None:
    entry_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container, scopes=GOOGLE_HEALTH_READ_SCOPES)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )

    skipped = await coordinator.run_user(USER_ID)

    assert skipped.status == "not_eligible"
    assert await container.google_health_storage.nutrition.meal(entry_id) is None
    assert (
        await container.google_health_storage.nutrition.get_backfill(
            USER_ID, HEALTH_USER_ID, BACKFILL_VERSION
        )
        is None
    )

    await _connect(container)
    completed = await coordinator.run_user(USER_ID)
    assert completed.status == "completed"
    assert await container.google_health_storage.nutrition.meal(entry_id) is not None


@pytest.mark.asyncio
async def test_run_all_sweeps_only_active_nutrition_authorized_connections(
    container: AppContainer,
) -> None:
    await _connect(container)
    config = _config()
    await container.google_health_storage.upsert_connection(
        telegram_user_id="telegram-chat-43",
        encrypted_refresh_token=config.cipher.encrypt("refresh-token"),
        health_user_id="google-account-43",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_READ_SCOPES,
    )

    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    results = await coordinator.run_all_eligible()

    assert [result.telegram_user_id for result in results] == [USER_ID]
    assert results[0].status == "completed"
    assert (
        await container.google_health_storage.nutrition.get_backfill(
            "telegram-chat-43", "google-account-43", BACKFILL_VERSION
        )
        is None
    )


@pytest.mark.asyncio
async def test_run_all_reports_one_account_failure_without_stopping_sweep(
    container: AppContainer,
) -> None:
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    run_user = AsyncMock(side_effect=RuntimeError("queue failed"))
    coordinator.run_user = run_user  # type: ignore[method-assign]

    results = await coordinator.run_all_eligible()

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].telegram_user_id == USER_ID
    assert results[0].health_user_id == HEALTH_USER_ID
    assert results[0].queued_count == 0
    assert results[0].skipped_count == 0


@pytest.mark.asyncio
async def test_run_user_rejects_groups_missing_connections_and_empty_backfills(
    container: AppContainer,
) -> None:
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )

    assert (await coordinator.run_user("telegram-chat--100")).status == "skipped"
    assert (await coordinator.run_user("not-a-telegram-user")).status == "skipped"
    assert (await coordinator.run_user(USER_ID)).status == "not_connected"

    await _connect(container)
    completed = await coordinator.run_user(USER_ID)

    assert completed.status == "completed"
    ledger = await container.google_health_storage.nutrition.get_backfill(
        USER_ID, HEALTH_USER_ID, BACKFILL_VERSION
    )
    assert ledger is not None
    assert ledger["high_water_meal_id"] == 0


@pytest.mark.asyncio
async def test_run_user_handles_an_existing_cursor_at_high_water(
    container: AppContainer,
) -> None:
    await _add_meals(container, [USER_ID])
    await _connect(container)
    nutrition = container.google_health_storage.nutrition
    await nutrition._conn.execute(
        """
        INSERT INTO google_health_nutrition_backfills
            (telegram_user_id, health_user_id, backfill_version,
             high_water_meal_id, cursor_meal_id, status, updated_at)
        VALUES (?, ?, ?, 1, 1, 'pending', 0)
        """,
        (USER_ID, HEALTH_USER_ID, BACKFILL_VERSION),
    )
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )

    result = await coordinator.run_user(USER_ID)

    assert result.status == "completed"
    assert await nutrition.meal(1) is None


@pytest.mark.asyncio
async def test_run_user_releases_lease_when_cursor_update_is_lost(
    container: AppContainer,
) -> None:
    await _add_meals(container, [USER_ID])
    await _connect(container)
    nutrition = container.google_health_storage.nutrition
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    with patch.object(
        nutrition, "_advance_backfill_unlocked", new=AsyncMock(return_value=False)
    ):
        result = await coordinator.run_user(USER_ID)

    assert result.status == "failed"
    ledger = await nutrition.get_backfill(USER_ID, HEALTH_USER_ID, BACKFILL_VERSION)
    assert ledger is not None
    assert ledger["status"] == "pending"
    assert ledger["last_error"] == "backfill_queue_error"


@pytest.mark.asyncio
async def test_run_user_propagates_cancellation_for_lease_recovery(
    container: AppContainer,
) -> None:
    await _add_meals(container, [USER_ID])
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    with (
        patch.object(
            container.calorie_storage,
            "health_backfill_batch",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await coordinator.run_user(USER_ID)

    ledger = await container.google_health_storage.nutrition.get_backfill(
        USER_ID, HEALTH_USER_ID, BACKFILL_VERSION
    )
    assert ledger is not None
    assert ledger["status"] == "running"


@pytest.mark.asyncio
async def test_backfill_resumes_after_batch_failure_without_duplicates(
    container: AppContainer,
) -> None:
    ids = await _add_meals(container, [USER_ID] * 55)
    await _connect(container)
    nutrition = container.google_health_storage.nutrition
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    original = nutrition.ensure_backfill_export
    calls = 0

    async def fail_on_second_batch(**kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 51:
            raise RuntimeError("simulated queue failure")
        return await original(**kwargs)

    nutrition.ensure_backfill_export = fail_on_second_batch  # type: ignore[method-assign]
    failed = await coordinator.run_user(USER_ID)
    nutrition.ensure_backfill_export = original  # type: ignore[method-assign]

    assert failed.status == "failed"
    ledger = await nutrition.get_backfill(USER_ID, HEALTH_USER_ID, BACKFILL_VERSION)
    assert ledger is not None
    assert ledger["status"] == "pending"
    assert ledger["cursor_meal_id"] == 50
    first_revision_count = 0
    for entry_id in ids[:50]:
        first_revision_count += len(await nutrition.revisions(entry_id))
    assert first_revision_count == 50
    second_revision_count = 0
    for entry_id in ids[50:]:
        second_revision_count += len(await nutrition.revisions(entry_id))
    assert second_revision_count == 0

    resumed = await coordinator.run_user(USER_ID)

    assert resumed.status == "completed"
    revision_count = 0
    for entry_id in ids:
        revision_count += len(await nutrition.revisions(entry_id))
    assert revision_count == 55


@pytest.mark.asyncio
async def test_concurrent_backfill_runs_share_one_ledger(
    container: AppContainer,
) -> None:
    ids = await _add_meals(container, [USER_ID, USER_ID])
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )

    first, second = await asyncio.gather(
        coordinator.run_user(USER_ID), coordinator.run_user(USER_ID)
    )

    assert {first.status, second.status} <= {"completed", "running"}
    revision_count = 0
    for entry_id in ids:
        revision_count += len(
            await container.google_health_storage.nutrition.revisions(entry_id)
        )
    assert revision_count == 2


@pytest.mark.asyncio
async def test_reconnect_requeues_unsent_meals_and_preserves_synced_history(
    container: AppContainer,
) -> None:
    entry_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    nutrition = container.google_health_storage.nutrition
    first_revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(first_revision["sequence"]), "synced")
    await nutrition.record_remote_result(int(first_revision["sequence"]))

    await container.google_health_storage.delete_connection(USER_ID)
    await _connect(container)
    result = await coordinator.run_user(USER_ID)

    assert result.status == "completed"
    assert result.queued_count == 0
    assert result.skipped_count == 1
    revisions = await nutrition.revisions(entry_id)
    assert len(revisions) == 1
    assert revisions[0]["state"] == "synced"
    history = await nutrition._fetch_all(
        "SELECT * FROM nutrition_export_history WHERE meal_id = ?", (entry_id,)
    )
    assert len(history) == 1


@pytest.mark.asyncio
async def test_same_account_reconnect_restores_future_edit_export_path(
    container: AppContainer,
) -> None:
    entry_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    nutrition = container.google_health_storage.nutrition
    first_revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(first_revision["sequence"]), "synced")
    await nutrition.record_remote_result(int(first_revision["sequence"]))

    await container.google_health_storage.delete_connection(USER_ID)
    await _connect(container)
    await coordinator.run_user(USER_ID)

    service = MealService(container)
    _, status = await service.mutate(
        USER_ID,
        private=True,
        entry_id=entry_id,
        updates={"calories": 999},
    )

    assert status == "pending"
    revisions = await nutrition.revisions(entry_id)
    assert revisions[-2]["operation"] == "delete"
    assert revisions[-2]["resource_name"] == first_revision["resource_name"]
    assert revisions[-1]["operation"] == "upsert"


@pytest.mark.asyncio
async def test_same_account_reconnect_allows_delete_before_backfill(
    container: AppContainer,
) -> None:
    entry_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    nutrition = container.google_health_storage.nutrition
    first_revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(first_revision["sequence"]), "synced")
    await nutrition.record_remote_result(int(first_revision["sequence"]))

    await container.google_health_storage.delete_connection(USER_ID)
    await _connect(container)

    service = MealService(container)
    _, status = await service.mutate(USER_ID, private=True, entry_id=entry_id)

    assert status == "pending"
    revisions = await nutrition.revisions(entry_id)
    assert revisions[-1]["operation"] == "delete"
    assert revisions[-1]["resource_name"] == first_revision["resource_name"]


@pytest.mark.asyncio
async def test_same_account_reconnect_backfills_meals_logged_while_disconnected(
    container: AppContainer,
) -> None:
    first_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    await container.google_health_storage.delete_connection(USER_ID)
    late_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)

    result = await coordinator.run_user(USER_ID)

    assert result.queued_count == 1
    assert len(await container.google_health_storage.nutrition.revisions(first_id)) == 1
    assert len(await container.google_health_storage.nutrition.revisions(late_id)) == 1


@pytest.mark.asyncio
async def test_reconnect_requeues_an_unsent_historical_intent(
    container: AppContainer,
) -> None:
    entry_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    await container.google_health_storage.delete_connection(USER_ID)
    await _connect(container)

    original_revision = (
        await container.google_health_storage.nutrition.revisions(entry_id)
    )[0]
    result = await coordinator.run_user(USER_ID)

    assert result.queued_count == 0
    assert result.skipped_count == 1
    restored = await container.google_health_storage.nutrition.revisions(entry_id)
    assert len(restored) == 1
    assert restored[0]["resource_name"] == original_revision["resource_name"]
    row = await container.google_health_storage.nutrition.meal(entry_id)
    assert row is not None
    assert row["status"] == "pending"


@pytest.mark.asyncio
async def test_reconnect_preserves_in_flight_resource_for_worker_reconciliation(
    container: AppContainer,
) -> None:
    entry_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    nutrition = container.google_health_storage.nutrition
    revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(revision["sequence"]), "in_flight")
    resource_name = revision["resource_name"]

    await container.google_health_storage.delete_connection(USER_ID)
    await _connect(container)
    result = await coordinator.run_user(USER_ID)

    assert result.queued_count == 0
    assert result.skipped_count == 1
    restored = (await nutrition.revisions(entry_id))[0]
    assert restored["resource_name"] == resource_name
    assert restored["state"] == "in_flight"
    row = await nutrition.meal(entry_id)
    assert row is not None
    assert row["status"] == "pending"


@pytest.mark.asyncio
async def test_replacement_google_account_gets_a_new_backfill_generation(
    container: AppContainer,
) -> None:
    entry_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)

    await _connect(container, health_user_id="google-account-new")
    result = await coordinator.run_user(USER_ID)

    assert result.status == "completed"
    assert result.queued_count == 1
    row = await container.google_health_storage.nutrition.meal(entry_id)
    assert row is not None
    assert row["health_user_id"] == "google-account-new"
    revisions = await container.google_health_storage.nutrition.revisions(entry_id)
    assert len(revisions) == 2
    assert revisions[0]["state"] == "cancelled"
    assert str(revisions[1]["resource_name"]).startswith("users/google-account-new/")
    await container.google_health_storage.nutrition.revision_state(
        int(revisions[1]["sequence"]), "synced"
    )
    await container.google_health_storage.nutrition.record_remote_result(
        int(revisions[1]["sequence"])
    )

    service = MealService(container)
    _, status = await service.mutate(
        USER_ID,
        private=True,
        entry_id=entry_id,
        updates={"calories": 999},
    )
    assert status == "pending"
    revisions = await container.google_health_storage.nutrition.revisions(entry_id)
    assert revisions[-2]["operation"] == "delete"
    assert revisions[-2]["resource_name"] == revisions[1]["resource_name"]


@pytest.mark.asyncio
async def test_backfill_reconciles_an_edited_meal_after_reconnect(
    container: AppContainer,
) -> None:
    entry_id = (await _add_meals(container, [USER_ID]))[0]
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    nutrition = container.google_health_storage.nutrition
    first_revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(first_revision["sequence"]), "synced")
    await nutrition.record_remote_result(int(first_revision["sequence"]))

    await container.google_health_storage.delete_connection(USER_ID)
    await container.calorie_storage.update_entry(entry_id, USER_ID, calories=999)
    await _connect(container)

    result = await coordinator.run_user(USER_ID)

    assert result.queued_count == 1
    revisions = await nutrition.revisions(entry_id)
    assert [revision["operation"] for revision in revisions] == [
        "upsert",
        "delete",
        "upsert",
    ]
    assert revisions[1]["resource_name"] == first_revision["resource_name"]


@pytest.mark.asyncio
async def test_meals_added_after_completed_backfill_are_not_scanned_again(
    container: AppContainer,
) -> None:
    await _add_meals(container, [USER_ID])
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    late_id = (await _add_meals(container, [USER_ID]))[0]

    await coordinator.run_user(USER_ID)

    assert await container.google_health_storage.nutrition.meal(late_id) is None


@pytest.mark.asyncio
async def test_new_meals_after_completed_backfill_use_normal_mutation_path(
    container: AppContainer,
) -> None:
    await _add_meals(container, [USER_ID])
    await _connect(container)
    coordinator = NutritionBackfillCoordinator(
        container.google_health_storage, container.calorie_storage
    )
    await coordinator.run_user(USER_ID)
    worker = MagicMock()
    container.nutrition_export_worker = worker

    service = MealService(container)
    entry_id, status = await service.mutate(
        USER_ID,
        private=True,
        entry=_entry(USER_ID, 99),
    )

    assert status == "pending"
    assert await container.google_health_storage.nutrition.meal(entry_id) is not None
    worker.wake.assert_called_once()
