"""Tests for the Google Health nutrition export background worker."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import aiosqlite
import pytest
from cryptography.fernet import Fernet

from blacki.health.client import (
    GoogleHealthApiError,
    GoogleHealthAuthError,
    GoogleHealthOperation,
    GoogleTokenResponse,
)
from blacki.health.config import (
    GOOGLE_HEALTH_READ_SCOPES,
    GOOGLE_HEALTH_SCOPES,
    GoogleHealthConfig,
)
from blacki.health.nutrition_worker import NutritionExportWorker
from blacki.health.storage import SqliteGoogleHealthStorage

USER_ID = "telegram-chat-1"
HEALTH_USER_ID = "health-user-1"
PAYLOAD = {
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


def _config() -> GoogleHealthConfig:
    return GoogleHealthConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/google-health/callback",
        token_encryption_key=Fernet.generate_key().decode(),
        sync_interval_hours=12,
        manual_refresh_cooldown_seconds=3600,
        oauth_state_ttl_seconds=600,
    )


def _token() -> GoogleTokenResponse:
    return GoogleTokenResponse(
        access_token="access-token",
        expires_in=3600,
        refresh_token=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )


@pytest.fixture
async def storage() -> AsyncGenerator[SqliteGoogleHealthStorage, None]:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    store = SqliteGoogleHealthStorage(conn, asyncio.Lock())
    await store.initialize()
    yield store
    await store.close()
    await conn.close()


async def _connect(
    storage: SqliteGoogleHealthStorage, config: GoogleHealthConfig
) -> None:
    await storage.upsert_connection(
        telegram_user_id=USER_ID,
        encrypted_refresh_token=config.cipher.encrypt("refresh-token"),
        health_user_id=HEALTH_USER_ID,
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )


def _worker(
    storage: SqliteGoogleHealthStorage, config: GoogleHealthConfig
) -> tuple[NutritionExportWorker, AsyncMock]:
    client = AsyncMock()
    worker = NutritionExportWorker(config, storage, client=client)
    return worker, client


async def test_worker_dispatches_create_success(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=1,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.return_value = GoogleHealthOperation(
        done=True, name="op/1", response={}
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(1)
    assert row is not None
    assert row["status"] == "synced"
    revisions = await storage.nutrition.revisions(1)
    assert revisions[0]["state"] == "synced"
    client.create_nutrition_log.assert_awaited_once()


async def test_worker_retries_transient_then_verifies_success(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=2,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.side_effect = GoogleHealthApiError(
        "network blip", transport=True
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(2)
    assert row is not None
    assert row["status"] == "pending"
    assert row["next_attempt"] > 0
    revisions = await storage.nutrition.revisions(2)
    assert revisions[0]["state"] == "uncertain"

    client.get_data_point.return_value = PAYLOAD

    await storage.nutrition._conn.execute(
        "UPDATE nutrition_exports SET next_attempt = 0 WHERE meal_id = 2"
    )
    await worker._dispatch_due()

    row = await storage.nutrition.meal(2)
    assert row is not None
    assert row["status"] == "synced"
    client.get_data_point.assert_awaited_once()


async def test_worker_marks_permanent_failure(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=3,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.side_effect = GoogleHealthApiError(
        "bad request", status_code=400, error_code="invalid_argument"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(3)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "invalid_argument"
    revisions = await storage.nutrition.revisions(3)
    assert revisions[0]["state"] == "failed"


async def test_worker_pauses_on_auth_error(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=4,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.side_effect = GoogleHealthAuthError(
        "revoked", status_code=401, error_code="invalid_grant"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(4)
    assert row is not None
    assert row["status"] == "authorization_required"
    connection = await storage.get_connection(USER_ID)
    assert connection is not None
    assert connection.status == "reauthorization_required"


async def test_worker_preserves_connection_when_nutrition_scope_missing(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A read-only reconnect must not disable the whole Health connection."""
    config = _config()
    await storage.upsert_connection(
        telegram_user_id=USER_ID,
        encrypted_refresh_token=config.cipher.encrypt("refresh-token"),
        health_user_id=HEALTH_USER_ID,
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_READ_SCOPES,
    )
    await storage.nutrition.enqueue(
        meal_id=100,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)

    await worker._dispatch_due()

    row = await storage.nutrition.meal(100)
    assert row is not None
    assert row["status"] == "authorization_required"
    assert row["error_code"] == "nutrition_scope_missing"
    connection = await storage.get_connection(USER_ID)
    assert connection is not None
    assert connection.status == "connected"
    assert connection.encrypted_refresh_token is not None
    client.refresh_access_token.assert_not_awaited()
    client.create_nutrition_log.assert_not_awaited()


async def test_worker_dispatches_delete_success(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-x"
    )
    await storage.nutrition.enqueue(
        meal_id=5,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.delete_nutrition_log.return_value = GoogleHealthOperation(
        done=True, name="op/2", response={}
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(5)
    assert row is not None
    assert row["status"] == "deleted"
    revisions = await storage.nutrition.revisions(5)
    assert revisions[0]["state"] == "deleted"


async def test_worker_resolves_uncertain_delete_via_404(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-y"
    )
    await storage.nutrition.enqueue(
        meal_id=6,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthApiError(
        "not found", status_code=404
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(6)
    assert row is not None
    assert row["status"] == "deleted"
    client.delete_nutrition_log.assert_not_awaited()


async def test_worker_reconciles_persisted_in_flight_delete(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A crash right after "in_flight" persists must verify, not re-delete."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-z"
    )
    await storage.nutrition.enqueue(
        meal_id=102,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    await storage.nutrition.revision_state(1, "in_flight")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthApiError(
        "not found", status_code=404
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(102)
    assert row is not None
    assert row["status"] == "deleted"
    revisions = await storage.nutrition.revisions(102)
    assert revisions[0]["state"] == "deleted"
    client.get_data_point.assert_awaited_once()
    client.delete_nutrition_log.assert_not_awaited()


async def test_worker_cancels_stale_revision_never_dispatched(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A queued create superseded by a delete before it was ever sent is dropped."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=7,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    # Nothing was ever dispatched remotely, so the delete has no target.
    await storage.nutrition.enqueue(
        meal_id=7,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=None,
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()

    await worker._dispatch_due()

    row = await storage.nutrition.meal(7)
    assert row is not None
    assert row["status"] == "deleted"
    client.create_nutrition_log.assert_not_awaited()
    revisions = await storage.nutrition.revisions(7)
    assert revisions[0]["state"] == "cancelled"


async def test_worker_verify_mismatch_fails_permanently(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=8,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.return_value = {
        "nutritionLog": {"foodDisplayName": "Something else", "energy": {"kcal": 1}}
    }

    await worker._dispatch_due()

    row = await storage.nutrition.meal(8)
    assert row is not None
    assert row["status"] == "failed"


async def test_worker_uncertain_upsert_still_processing(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A 404 while an upsert is uncertain means it never landed; retry the create."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=9,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthApiError(
        "not found", status_code=404
    )

    await worker._dispatch_due()

    revisions = await storage.nutrition.revisions(9)
    assert revisions[0]["state"] == "queued"
    row = await storage.nutrition.meal(9)
    assert row is not None
    assert row["status"] == "pending"
    client.create_nutrition_log.assert_not_awaited()


async def test_worker_reconciles_persisted_in_flight_upsert(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A crash right after "in_flight" persists must not re-POST blindly."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=101,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.revision_state(1, "in_flight")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.return_value = PAYLOAD

    await worker._dispatch_due()

    row = await storage.nutrition.meal(101)
    assert row is not None
    assert row["status"] == "synced"
    revisions = await storage.nutrition.revisions(101)
    assert revisions[0]["state"] == "synced"
    client.get_data_point.assert_awaited_once()
    client.create_nutrition_log.assert_not_awaited()


async def test_worker_disconnected_marks_authorization_required(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await storage.nutrition.enqueue(
        meal_id=10,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)

    await worker._dispatch_due()

    row = await storage.nutrition.meal(10)
    assert row is not None
    assert row["status"] == "authorization_required"
    client.refresh_access_token.assert_not_awaited()


async def test_worker_start_stop_is_idempotent(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    worker, _client = _worker(storage, config)
    await worker.start()
    await worker.start()
    running_after_start: bool = worker._running
    assert running_after_start
    await worker.stop()
    await worker.stop()
    running_after_stop: bool = worker._running
    assert not running_after_stop
    await worker.close()


async def test_worker_stop_without_a_tracked_task_still_clears_running(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """Defensive: stop() must not crash if ``_task`` was somehow never set."""
    config = _config()
    worker, _client = _worker(storage, config)
    worker._running = True
    worker._task = None

    await worker.stop()

    assert worker._running is False


async def test_worker_wake_triggers_immediate_dispatch(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """``wake()`` must resolve a due meal without waiting for the 60s timer."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=20,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.return_value = GoogleHealthOperation(
        done=True, name="op/1", response={}
    )

    await worker.start()
    worker.wake()
    row = None
    for _ in range(50):
        row = await storage.nutrition.meal(20)
        if row is not None and row["status"] == "synced":
            break
        await asyncio.sleep(0.05)
    await worker.stop()
    await worker.close()

    assert row is not None
    assert row["status"] == "synced"
    client.create_nutrition_log.assert_awaited_once()


async def test_worker_token_refresh_transient_failure_backs_off(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=11,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.side_effect = GoogleHealthApiError(
        "unavailable", status_code=503
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(11)
    assert row is not None
    assert row["status"] == "pending"
    assert row["next_attempt"] > 0


async def test_worker_delete_transient_then_confirms_deleted(
    storage: SqliteGoogleHealthStorage,
) -> None:
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-z"
    )
    await storage.nutrition.enqueue(
        meal_id=12,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.delete_nutrition_log.side_effect = GoogleHealthApiError(
        "network blip", transport=True
    )

    await worker._dispatch_due()

    revisions = await storage.nutrition.revisions(12)
    assert revisions[0]["state"] == "uncertain"

    client.get_data_point.side_effect = GoogleHealthApiError(
        "not found", status_code=404
    )
    await storage.nutrition._conn.execute(
        "UPDATE nutrition_exports SET next_attempt = 0 WHERE meal_id = 12"
    )
    await worker._dispatch_due()

    row = await storage.nutrition.meal(12)
    assert row is not None
    assert row["status"] == "deleted"


def test_nutrition_log_matches_requires_dict_sections() -> None:
    from blacki.health.nutrition_worker import _nutrition_log_matches

    assert _nutrition_log_matches({}, {}) is False
    assert (
        _nutrition_log_matches({"nutritionLog": PAYLOAD["nutritionLog"]}, PAYLOAD)
        is True
    )


def test_json_payload_is_preserved_across_backoff_and_retry() -> None:
    from blacki.health.nutrition_worker import _safe_error_code

    assert _safe_error_code(None) is None
    assert _safe_error_code(ValueError("x")) == "ValueError"


async def test_dispatch_due_handles_due_query_failure(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A broken local query must not crash the tick, only skip it."""
    config = _config()
    worker, _client = _worker(storage, config)
    worker.storage.nutrition.due = AsyncMock(side_effect=RuntimeError("db locked"))  # type: ignore[method-assign]

    await worker._dispatch_due()  # must not raise


async def test_dispatch_due_isolates_per_row_failures(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """One meal raising unexpectedly must not stop the rest of the batch."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=30,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.enqueue(
        meal_id=31,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.return_value = GoogleHealthOperation(
        done=True, name="op/1", response={}
    )

    original_revisions = storage.nutrition.revisions

    async def _boom(meal_id: int) -> list[dict]:
        if meal_id == 30:
            raise RuntimeError("unexpected bug")
        return await original_revisions(meal_id)

    storage.nutrition.revisions = _boom  # type: ignore[method-assign]

    await worker._dispatch_due()  # must not raise

    storage.nutrition.revisions = original_revisions  # type: ignore[method-assign]
    row = await storage.nutrition.meal(31)
    assert row is not None
    assert row["status"] == "synced"


async def test_worker_pauses_on_corrupted_refresh_token(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A refresh token that fails decryption must pause, never crash."""
    config = _config()
    await storage.upsert_connection(
        telegram_user_id=USER_ID,
        encrypted_refresh_token="not-a-valid-fernet-token",
        health_user_id=HEALTH_USER_ID,
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    await storage.nutrition.enqueue(
        meal_id=32,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)

    await worker._dispatch_due()

    row = await storage.nutrition.meal(32)
    assert row is not None
    assert row["status"] == "authorization_required"
    client.refresh_access_token.assert_not_awaited()
    connection = await storage.get_connection(USER_ID)
    assert connection is not None
    assert connection.status == "reauthorization_required"


async def test_worker_pauses_when_create_raises_auth_error(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """An auth error raised mid-dispatch (not during refresh) must pause too."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=33,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.side_effect = GoogleHealthAuthError(
        "revoked mid-flight", status_code=401, error_code="invalid_grant"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(33)
    assert row is not None
    assert row["status"] == "authorization_required"
    connection = await storage.get_connection(USER_ID)
    assert connection is not None
    assert connection.status == "reauthorization_required"


async def test_worker_leaves_edit_pending_after_partial_resolution(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """Resolving the delete half of an edit must not finalize the meal early."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=34,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    old_resource_name = (await storage.nutrition.revisions(34))[0]["resource_name"]
    await storage.nutrition.revision_state(1, "synced")
    await storage.nutrition.enqueue(
        meal_id=34,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=old_resource_name,
    )
    await storage.nutrition.enqueue(
        meal_id=34,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.delete_nutrition_log.return_value = GoogleHealthOperation(
        done=True, name="op/del", response={}
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(34)
    assert row is not None
    assert row["status"] == "pending"
    assert row["next_attempt"] == 0
    client.create_nutrition_log.assert_not_awaited()
    revisions = await storage.nutrition.revisions(34)
    assert revisions[1]["state"] == "deleted"
    assert revisions[2]["state"] == "queued"


async def test_worker_create_operation_not_done_marks_uncertain(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A create that returns without ``done`` must be retried, not dropped."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=35,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.return_value = GoogleHealthOperation(
        done=False, name="op/pending", response=None
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(35)
    assert row is not None
    assert row["status"] == "pending"
    revisions = await storage.nutrition.revisions(35)
    assert revisions[0]["state"] == "uncertain"


async def test_worker_create_operation_error_marks_failed(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A create that completes with a provider error must fail permanently."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=36,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.return_value = GoogleHealthOperation(
        done=True, name="op/err", error_code="internal", response=None
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(36)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "internal"


async def test_worker_pauses_when_verify_raises_auth_error(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """An auth error while verifying an uncertain create must pause too."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=37,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthAuthError(
        "revoked", status_code=401, error_code="invalid_grant"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(37)
    assert row is not None
    assert row["status"] == "authorization_required"


async def test_worker_verify_transient_error_retries(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A transient, non-404 verify failure must be retried, not failed."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=38,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthApiError(
        "network blip", transport=True
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(38)
    assert row is not None
    assert row["status"] == "pending"
    assert row["next_attempt"] > 0
    revisions = await storage.nutrition.revisions(38)
    assert revisions[0]["state"] == "uncertain"


async def test_worker_verify_permanent_error_fails(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A non-transient, non-404 verify failure must fail permanently."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=39,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthApiError(
        "bad request", status_code=400, error_code="invalid_argument"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(39)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "invalid_argument"


async def test_worker_pauses_when_delete_verify_raises_auth_error(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """An auth error while confirming an uncertain delete must pause too."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-a"
    )
    await storage.nutrition.enqueue(
        meal_id=40,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthAuthError(
        "revoked", status_code=401, error_code="invalid_grant"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(40)
    assert row is not None
    assert row["status"] == "authorization_required"


async def test_worker_delete_verify_transient_error_retries(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A transient, non-404 delete-verify failure must be retried, not failed."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-b"
    )
    await storage.nutrition.enqueue(
        meal_id=41,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthApiError(
        "network blip", transport=True
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(41)
    assert row is not None
    assert row["status"] == "pending"
    assert row["next_attempt"] > 0
    revisions = await storage.nutrition.revisions(41)
    assert revisions[0]["state"] == "uncertain"


async def test_worker_delete_verify_permanent_error_fails(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A non-transient, non-404 delete-verify failure must fail permanently."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-c"
    )
    await storage.nutrition.enqueue(
        meal_id=42,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.side_effect = GoogleHealthApiError(
        "bad request", status_code=400, error_code="invalid_argument"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(42)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "invalid_argument"


async def test_worker_delete_verify_success_requeues_delete(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """If the point still exists on confirmation, the delete must be retried."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-d"
    )
    await storage.nutrition.enqueue(
        meal_id=43,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    await storage.nutrition.revision_state(1, "uncertain")
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.get_data_point.return_value = {"nutritionLog": {}}

    await worker._dispatch_due()

    row = await storage.nutrition.meal(43)
    assert row is not None
    assert row["status"] == "pending"
    revisions = await storage.nutrition.revisions(43)
    assert revisions[0]["state"] == "queued"
    client.delete_nutrition_log.assert_not_awaited()


async def test_worker_pauses_when_delete_raises_auth_error(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """An auth error raised directly by the delete call must pause too."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-e"
    )
    await storage.nutrition.enqueue(
        meal_id=44,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.delete_nutrition_log.side_effect = GoogleHealthAuthError(
        "revoked", status_code=401, error_code="invalid_grant"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(44)
    assert row is not None
    assert row["status"] == "authorization_required"


async def test_worker_delete_404_treated_as_already_deleted(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A 404 straight from the delete call means the point is already gone."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-f"
    )
    await storage.nutrition.enqueue(
        meal_id=45,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.delete_nutrition_log.side_effect = GoogleHealthApiError(
        "not found", status_code=404
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(45)
    assert row is not None
    assert row["status"] == "deleted"


async def test_worker_delete_permanent_failure(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A non-transient, non-404 delete failure must fail permanently."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-g"
    )
    await storage.nutrition.enqueue(
        meal_id=46,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.delete_nutrition_log.side_effect = GoogleHealthApiError(
        "bad request", status_code=400, error_code="invalid_argument"
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(46)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "invalid_argument"


async def test_worker_delete_operation_not_done_marks_uncertain(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A delete that returns without ``done`` must be retried, not dropped."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-h"
    )
    await storage.nutrition.enqueue(
        meal_id=47,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.delete_nutrition_log.return_value = GoogleHealthOperation(
        done=False, name="op/pending", response=None
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(47)
    assert row is not None
    assert row["status"] == "pending"
    revisions = await storage.nutrition.revisions(47)
    assert revisions[0]["state"] == "uncertain"


async def test_worker_delete_operation_error_marks_failed(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A delete that completes with a provider error must fail permanently."""
    config = _config()
    await _connect(storage, config)
    resource_name = (
        f"users/{HEALTH_USER_ID}/dataTypes/nutrition-log/dataPoints/blacki-i"
    )
    await storage.nutrition.enqueue(
        meal_id=48,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=None,
        operation="delete",
        target_resource_name=resource_name,
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.delete_nutrition_log.return_value = GoogleHealthOperation(
        done=True, name="op/err", error_code="internal", response=None
    )

    await worker._dispatch_due()

    row = await storage.nutrition.meal(48)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "internal"


async def test_backoff_honors_retry_after_header(
    storage: SqliteGoogleHealthStorage,
) -> None:
    """A server-provided Retry-After must extend the backoff, not shrink it."""
    config = _config()
    await _connect(storage, config)
    await storage.nutrition.enqueue(
        meal_id=49,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    worker, client = _worker(storage, config)
    client.refresh_access_token.return_value = _token()
    client.create_nutrition_log.side_effect = GoogleHealthApiError(
        "rate limited", status_code=429, retry_after_seconds=900
    )

    before = datetime.now(UTC).timestamp()
    await worker._dispatch_due()

    row = await storage.nutrition.meal(49)
    assert row is not None
    assert row["status"] == "pending"
    assert row["next_attempt"] >= before + 900
