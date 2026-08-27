"""Tests for atomic meal mutations and Google Health export enrollment."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import aiosqlite
import pytest
from cryptography.fernet import Fernet

from blacki.calories.service import (
    MealService,
    get_meal_service,
    nutrition_payload,
    validate_meal,
)
from blacki.calories.storage import CalorieEntry
from blacki.container import (
    AppContainer,
    reset_container_for_tests,
    set_container,
)
from blacki.health.config import GOOGLE_HEALTH_NUTRITION_SCOPES, GoogleHealthConfig

USER_ID = "telegram-chat-500"
OTHER_HEALTH_USER_ID = "telegram-chat-999"


def _entry(**overrides: object) -> CalorieEntry:
    fields: dict[str, object] = {
        "user_id": USER_ID,
        "description": "Oatmeal",
        "calories": 300,
        "logged_at": "2026-01-05T08:00:00+00:00",
        "logged_date": "2026-01-05",
    }
    fields.update(overrides)
    return CalorieEntry.model_validate(fields)


@pytest.fixture
async def container() -> AsyncGenerator[AppContainer, None]:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    app_container = AppContainer(conn=conn)
    await app_container.initialize_all_storages()
    yield app_container
    await app_container.close()


async def _connect(
    container: AppContainer,
    *,
    telegram_user_id: str = USER_ID,
    health_user_id: str = USER_ID,
    scopes: tuple[str, ...] = GOOGLE_HEALTH_NUTRITION_SCOPES,
) -> None:
    key = Fernet.generate_key().decode()
    config = GoogleHealthConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/google-health/callback",
        token_encryption_key=key,
        sync_interval_hours=12,
        manual_refresh_cooldown_seconds=3600,
        oauth_state_ttl_seconds=600,
    )
    await container.google_health_storage.upsert_connection(
        telegram_user_id=telegram_user_id,
        encrypted_refresh_token=config.cipher.encrypt("refresh-token"),
        health_user_id=health_user_id,
        legacy_fitbit_user_id=None,
        scopes=scopes,
    )


# --- validate_meal -----------------------------------------------------


def test_validate_meal_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description cannot be empty"):
        validate_meal(_entry(description="   "))


def test_validate_meal_rejects_nonpositive_calories() -> None:
    with pytest.raises(ValueError, match="estimated_calories must be > 0"):
        validate_meal(_entry(calories=0))


def test_validate_meal_rejects_unknown_meal_type() -> None:
    with pytest.raises(ValueError, match="meal_type must be"):
        validate_meal(_entry(meal_type="brunch"))


def test_validate_meal_accepts_none_meal_type() -> None:
    validate_meal(_entry(meal_type=None))


@pytest.mark.parametrize("field", ["protein_g", "carbs_g", "fat_g"])
def test_validate_meal_rejects_negative_macros(field: str) -> None:
    with pytest.raises(ValueError, match="macros must be finite and nonnegative"):
        validate_meal(_entry(**{field: -1.0}))


@pytest.mark.parametrize("field", ["protein_g", "carbs_g", "fat_g"])
def test_validate_meal_rejects_nonfinite_macros(field: str) -> None:
    with pytest.raises(ValueError, match="macros must be finite and nonnegative"):
        validate_meal(_entry(**{field: float("nan")}))


# --- nutrition_payload ---------------------------------------------------


def test_nutrition_payload_maps_core_fields() -> None:
    entry = _entry(
        meal_type="breakfast",
        protein_g=10.0,
        carbs_g=20.0,
        fat_g=5.0,
        logged_at="2026-01-05T08:00:00+00:00",
        logged_date="2026-01-05",
    )
    payload = nutrition_payload(entry)["nutritionLog"]
    assert payload["foodDisplayName"] == "Oatmeal"
    assert payload["energy"] == {"kcal": 300}
    assert payload["mealType"] == "BREAKFAST"
    assert payload["nutrients"] == [
        {"nutrient": "PROTEIN", "quantity": {"grams": 10.0}}
    ]
    assert payload["totalCarbohydrate"] == {"grams": 20.0}
    assert payload["totalFat"] == {"grams": 5.0}
    assert payload["interval"]["startTime"] == "2026-01-05T08:00:00Z"


def test_nutrition_payload_omits_absent_optional_fields() -> None:
    entry = _entry(meal_type=None, protein_g=None, carbs_g=None, fat_g=None)
    payload = nutrition_payload(entry)["nutritionLog"]
    assert "mealType" not in payload
    assert "nutrients" not in payload
    assert "totalCarbohydrate" not in payload
    assert "totalFat" not in payload


def test_nutrition_payload_naive_logged_at_uses_app_timezone() -> None:
    entry = _entry(logged_at="2026-01-05T08:00:00", logged_date="2026-01-05")
    payload = nutrition_payload(entry)
    assert payload["nutritionLog"]["interval"]["startTime"].endswith("Z")


def test_nutrition_payload_falls_back_to_noon_when_dates_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backfilled/date-edited meal anchors to noon on the target date."""
    monkeypatch.setenv("AGENT_TIMEZONE", "UTC")
    entry = _entry(logged_at="2026-01-05T23:50:00+00:00", logged_date="2026-01-08")
    payload = nutrition_payload(entry)["nutritionLog"]
    assert payload["interval"]["startTime"] == "2026-01-08T12:00:00Z"
    assert payload["interval"]["endTime"] == "2026-01-08T12:00:01Z"


# --- MealService.mutate: argument validation ------------------------------


async def test_mutate_requires_entry_or_entry_id(container: AppContainer) -> None:
    service = MealService(container)
    with pytest.raises(ValueError, match="A new meal is required"):
        await service.mutate(USER_ID)


async def test_mutate_rejects_both_entry_and_entry_id(container: AppContainer) -> None:
    service = MealService(container)
    with pytest.raises(ValueError, match="cannot both select"):
        await service.mutate(USER_ID, entry=_entry(), entry_id=1)


async def test_mutate_rejects_meal_not_found(container: AppContainer) -> None:
    service = MealService(container)
    with pytest.raises(ValueError, match="Meal not found"):
        await service.mutate(USER_ID, entry_id=999, updates={"calories": 100})


async def test_mutate_rejects_owner_mismatch(container: AppContainer) -> None:
    service = MealService(container)
    with pytest.raises(ValueError, match="owner does not match"):
        await service.mutate(USER_ID, entry=_entry(user_id="someone-else"))


async def test_mutate_rejects_invalid_new_entry(container: AppContainer) -> None:
    service = MealService(container)
    with pytest.raises(ValueError, match="description cannot be empty"):
        await service.mutate(USER_ID, entry=_entry(description=""))


# --- MealService.mutate: not private / not connected -----------------------


async def test_mutate_create_not_private_is_local_only(container: AppContainer) -> None:
    service = MealService(container)
    entry_id, sync_status = await service.mutate(USER_ID, entry=_entry())
    assert sync_status == "not_enabled"
    row = await container.google_health_storage.nutrition.meal(entry_id)
    assert row is None


async def test_mutate_create_private_without_connection(
    container: AppContainer,
) -> None:
    service = MealService(container)
    entry_id, sync_status = await service.mutate(USER_ID, private=True, entry=_entry())
    assert sync_status == "not_enabled"
    row = await container.google_health_storage.nutrition.meal(entry_id)
    assert row is None


async def test_mutate_create_private_missing_scopes(container: AppContainer) -> None:
    """A newly created meal for a connected-but-unscoped account never enrolls."""
    await _connect(container, scopes=())
    service = MealService(container)
    entry_id, sync_status = await service.mutate(USER_ID, private=True, entry=_entry())
    assert sync_status == "authorization_required"
    row = await container.google_health_storage.nutrition.meal(entry_id)
    assert row is None


# --- MealService.mutate: eligible create/edit/delete ------------------------


async def test_mutate_create_private_eligible_enqueues_upsert(
    container: AppContainer,
) -> None:
    await _connect(container)
    worker = MagicMock()
    container.nutrition_export_worker = worker
    service = MealService(container)

    entry_id, sync_status = await service.mutate(USER_ID, private=True, entry=_entry())

    assert sync_status == "pending"
    worker.wake.assert_called_once()
    row = await container.google_health_storage.nutrition.meal(entry_id)
    assert row is not None
    assert row["status"] == "pending"
    assert row["desired_operation"] == "upsert"
    revisions = await container.google_health_storage.nutrition.revisions(entry_id)
    assert len(revisions) == 1
    assert revisions[0]["operation"] == "upsert"


async def test_mutate_does_not_wake_worker_when_not_pending(
    container: AppContainer,
) -> None:
    worker = MagicMock()
    container.nutrition_export_worker = worker
    service = MealService(container)

    await service.mutate(USER_ID, entry=_entry())

    worker.wake.assert_not_called()


async def test_mutate_edit_replaces_synced_point_with_delete_then_upsert(
    container: AppContainer,
) -> None:
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, private=True, entry=_entry())

    nutrition = container.google_health_storage.nutrition
    first_revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(first_revision["sequence"]), "synced")

    _, sync_status = await service.mutate(
        USER_ID,
        private=True,
        entry_id=entry_id,
        updates={"calories": 450},
    )

    assert sync_status == "pending"
    revisions = await nutrition.revisions(entry_id)
    assert len(revisions) == 3
    assert revisions[1]["operation"] == "delete"
    assert revisions[1]["resource_name"] == first_revision["resource_name"]
    assert revisions[2]["operation"] == "upsert"


async def test_mutate_edit_preserves_original_interval(
    container: AppContainer,
) -> None:
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(
        USER_ID,
        private=True,
        entry=_entry(logged_at="2026-01-05T08:00:00+00:00", logged_date="2026-01-05"),
    )
    nutrition = container.google_health_storage.nutrition
    original_interval = (await nutrition.revisions(entry_id))[0]

    await service.mutate(
        USER_ID, private=True, entry_id=entry_id, updates={"calories": 450}
    )

    revisions = await nutrition.revisions(entry_id)
    original_payload = json.loads(original_interval["payload_json"])
    new_payload = json.loads(revisions[-1]["payload_json"])
    assert (
        new_payload["nutritionLog"]["interval"]
        == original_payload["nutritionLog"]["interval"]
    )


async def test_mutate_edit_with_new_date_does_not_preserve_interval(
    container: AppContainer,
) -> None:
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(
        USER_ID,
        private=True,
        entry=_entry(logged_at="2026-01-05T08:00:00+00:00", logged_date="2026-01-05"),
    )

    await service.mutate(
        USER_ID,
        private=True,
        entry_id=entry_id,
        updates={"logged_date": "2026-01-06"},
    )

    nutrition = container.google_health_storage.nutrition
    revisions = await nutrition.revisions(entry_id)
    new_payload = json.loads(revisions[-1]["payload_json"])
    assert new_payload["nutritionLog"]["interval"]["startTime"].startswith("2026-01-06")


async def test_mutate_edit_of_unenrolled_meal_stays_not_enabled(
    container: AppContainer,
) -> None:
    """The backfill coordinator, not an edit, enrolls older meals."""
    service = MealService(container)
    entry_id, sync_status = await service.mutate(USER_ID, entry=_entry())
    assert sync_status == "not_enabled"

    await _connect(container)
    _, sync_status = await service.mutate(
        USER_ID, private=True, entry_id=entry_id, updates={"calories": 450}
    )
    assert sync_status == "not_enabled"


async def test_mutate_edit_revokes_when_scope_lost_mid_flight(
    container: AppContainer,
) -> None:
    await _connect(container)
    service = MealService(container)
    entry_id, sync_status = await service.mutate(USER_ID, private=True, entry=_entry())
    assert sync_status == "pending"

    await _connect(container, scopes=())

    _, sync_status = await service.mutate(
        USER_ID, private=True, entry_id=entry_id, updates={"calories": 450}
    )
    assert sync_status == "authorization_required"
    row = await container.google_health_storage.nutrition.meal(entry_id)
    assert row is not None
    assert row["status"] == "authorization_required"


async def test_mutate_delete_enqueues_target_from_synced_revision(
    container: AppContainer,
) -> None:
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, private=True, entry=_entry())
    nutrition = container.google_health_storage.nutrition
    revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(revision["sequence"]), "synced")

    _, sync_status = await service.mutate(USER_ID, private=True, entry_id=entry_id)

    assert sync_status == "pending"
    row = await nutrition.meal(entry_id)
    assert row is not None
    assert row["desired_operation"] == "delete"
    revisions = await nutrition.revisions(entry_id)
    assert revisions[-1]["operation"] == "delete"
    assert revisions[-1]["resource_name"] == revision["resource_name"]
    remaining = await container.calorie_storage.get_daily_summary(USER_ID, "2026-01-05")
    assert remaining.entry_count == 0


async def test_mutate_delete_never_dispatched_has_no_target(
    container: AppContainer,
) -> None:
    """Deleting a meal whose create never reached Google enqueues no revision."""
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, private=True, entry=_entry())

    _, sync_status = await service.mutate(USER_ID, private=True, entry_id=entry_id)

    assert sync_status == "pending"
    nutrition = container.google_health_storage.nutrition
    row = await nutrition.meal(entry_id)
    assert row is not None
    assert row["desired_operation"] == "delete"
    assert row["desired_revision"] is None
    revisions = await nutrition.revisions(entry_id)
    assert len(revisions) == 1
    assert revisions[-1]["operation"] == "upsert"
    assert revisions[-1]["state"] == "queued"


async def test_mutate_delete_of_cancelled_export_reuses_active_connection(
    container: AppContainer,
) -> None:
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, private=True, entry=_entry())
    nutrition = container.google_health_storage.nutrition
    revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(revision["sequence"]), "synced")

    await nutrition.cancel(USER_ID)

    _, sync_status = await service.mutate(USER_ID, private=True, entry_id=entry_id)
    assert sync_status == "pending"
    revisions = await nutrition.revisions(entry_id)
    assert revisions[-1]["operation"] == "delete"
    assert revisions[-1]["resource_name"] == revision["resource_name"]


async def test_mutate_edit_after_account_replacement_uses_new_account(
    container: AppContainer,
) -> None:
    await _connect(container, health_user_id="old-health-account")
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, private=True, entry=_entry())

    await _connect(container, health_user_id="new-health-account")
    _, sync_status = await service.mutate(
        USER_ID,
        private=True,
        entry_id=entry_id,
        updates={"calories": 450},
    )

    assert sync_status == "pending"
    revisions = await container.google_health_storage.nutrition.revisions(entry_id)
    assert len(revisions) == 2
    assert revisions[-1]["operation"] == "upsert"
    assert str(revisions[-1]["resource_name"]).startswith("users/new-health-account/")


# --- MealService.mutate: transaction integrity ------------------------------


async def test_mutate_rolls_back_local_write_on_export_failure(
    container: AppContainer,
) -> None:
    await _connect(container)
    service = MealService(container)

    nutrition = container.google_health_storage.nutrition
    original_enqueue = nutrition.enqueue

    async def _boom(**kwargs: object) -> None:
        raise RuntimeError("simulated export failure")

    nutrition.enqueue = _boom  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="simulated export failure"):
            await service.mutate(USER_ID, private=True, entry=_entry())
    finally:
        nutrition.enqueue = original_enqueue  # type: ignore[method-assign]

    summary = await container.calorie_storage.get_daily_summary(USER_ID, "2026-01-05")
    assert summary.entry_count == 0


async def test_mutate_account_mismatch_blocks_enqueue(
    container: AppContainer,
) -> None:
    """A stored export row bound to a different account must not be reused."""
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, private=True, entry=_entry())

    await container.conn.execute(
        "UPDATE nutrition_exports SET health_user_id = ? WHERE meal_id = ?",
        (OTHER_HEALTH_USER_ID, entry_id),
    )

    _, sync_status = await service.mutate(
        USER_ID, private=True, entry_id=entry_id, updates={"calories": 450}
    )
    assert sync_status == "not_enabled"


async def test_mutate_falls_back_to_standalone_nutrition_storage(
    container: AppContainer,
) -> None:
    """A health storage without a ``.nutrition`` attribute still works."""
    from blacki.health.nutrition_storage import NutritionStorage

    class _NoNutritionHealthStorage:
        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

    container._google_health_storage = _NoNutritionHealthStorage()  # type: ignore[assignment]
    service = MealService(container)

    entry_id, sync_status = await service.mutate(USER_ID, entry=_entry())

    assert sync_status == "not_enabled"
    assert isinstance(service._nutrition, NutritionStorage)
    assert entry_id > 0


async def test_mutate_delete_pauses_when_stored_health_user_id_is_blank(
    container: AppContainer,
) -> None:
    """A stale export row missing its account id must pause, never crash."""
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, entry=_entry())

    nutrition = container.google_health_storage.nutrition
    await nutrition.enqueue(
        meal_id=entry_id,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id="",
        payload=None,
        operation="delete",
    )

    _, sync_status = await service.mutate(USER_ID, private=True, entry_id=entry_id)
    assert sync_status == "authorization_required"


async def test_mutate_edit_skips_carry_forward_when_prior_payload_missing(
    container: AppContainer,
) -> None:
    """A corrupted/absent prior payload must not stop the new edit from syncing."""
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(
        USER_ID,
        private=True,
        entry=_entry(logged_at="2026-01-05T08:00:00+00:00", logged_date="2026-01-05"),
    )
    await container.conn.execute(
        "UPDATE nutrition_revisions SET payload_json = NULL WHERE meal_id = ?",
        (entry_id,),
    )

    _, sync_status = await service.mutate(
        USER_ID, private=True, entry_id=entry_id, updates={"calories": 450}
    )

    assert sync_status == "pending"
    nutrition = container.google_health_storage.nutrition
    revisions = await nutrition.revisions(entry_id)
    new_payload = json.loads(revisions[-1]["payload_json"])
    assert new_payload["nutritionLog"]["interval"]["startTime"] == (
        "2026-01-05T08:00:00Z"
    )


async def test_mutate_edit_skips_carry_forward_when_prior_payload_has_no_log(
    container: AppContainer,
) -> None:
    """A prior payload missing its nutritionLog key must not carry an interval."""
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, private=True, entry=_entry())
    await container.conn.execute(
        "UPDATE nutrition_revisions SET payload_json = ? WHERE meal_id = ?",
        (json.dumps({"other": True}), entry_id),
    )

    _, sync_status = await service.mutate(
        USER_ID, private=True, entry_id=entry_id, updates={"calories": 450}
    )

    assert sync_status == "pending"


async def test_latest_remote_resource_skips_past_a_pending_delete(
    container: AppContainer,
) -> None:
    """A trailing pending delete must not hide an earlier synced create."""
    await _connect(container)
    service = MealService(container)
    entry_id, _ = await service.mutate(USER_ID, private=True, entry=_entry())

    nutrition = container.google_health_storage.nutrition
    upsert_revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(upsert_revision["sequence"]), "synced")
    await nutrition.enqueue(
        meal_id=entry_id,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=str(upsert_revision["resource_name"]),
    )

    target = await service._latest_remote_resource(nutrition, entry_id)

    assert target == upsert_revision["resource_name"]


async def test_latest_remote_resource_filters_invalid_and_other_accounts(
    container: AppContainer,
) -> None:
    class _NutritionWithoutHistory:
        async def revisions(self, meal_id: int) -> list[dict[str, object]]:
            return [
                {"operation": "upsert", "resource_name": None, "state": "synced"},
                {
                    "operation": "upsert",
                    "resource_name": "users/other/dataTypes/nutrition-log/dataPoints/x",
                    "state": "synced",
                },
                {"operation": "delete", "resource_name": "users/current/x"},
            ]

    service = MealService(container)

    assert (
        await service._latest_remote_resource(_NutritionWithoutHistory(), 1, "current")
        is None
    )


async def test_get_meal_service_binds_the_process_container(
    container: AppContainer,
) -> None:
    set_container(container)
    try:
        service = get_meal_service()
        assert isinstance(service, MealService)
        assert service.container is container
    finally:
        reset_container_for_tests()
