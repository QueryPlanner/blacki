"""Tests for private meal-export status and retry tools."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import aiosqlite
import pytest
from cryptography.fernet import Fernet
from google.adk.tools import ToolContext

from blacki.container import AppContainer, reset_container_for_tests, set_container
from blacki.health.config import GOOGLE_HEALTH_NUTRITION_SCOPES, GoogleHealthConfig
from blacki.tools.calories import get_meal_sync_status, retry_meal_sync

USER_ID = "telegram-chat-42"
HEALTH_USER_ID = "google-account-42"
PAYLOAD = {
    "nutritionLog": {
        "foodDisplayName": "Oatmeal",
        "energy": {"kcal": 300},
    }
}


@pytest.fixture
async def container() -> AsyncGenerator[AppContainer, None]:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    app_container = AppContainer(conn=conn)
    await app_container.initialize_all_storages()
    set_container(app_container)
    yield app_container
    reset_container_for_tests()
    await app_container.close()


def _context(*, private: bool = True, user_id: str = USER_ID) -> ToolContext:
    return cast(
        ToolContext,
        SimpleNamespace(
            user_id=user_id,
            state={"telegram_chat_type": "private" if private else "group"},
        ),
    )


async def _connect(container: AppContainer, scopes: tuple[str, ...]) -> None:
    config = GoogleHealthConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/callback",
        token_encryption_key=Fernet.generate_key().decode(),
    )
    await container.google_health_storage.upsert_connection(
        telegram_user_id=USER_ID,
        encrypted_refresh_token=config.cipher.encrypt("refresh-token"),
        health_user_id=HEALTH_USER_ID,
        legacy_fitbit_user_id=None,
        scopes=scopes,
    )


@pytest.mark.asyncio
async def test_get_meal_sync_status_returns_connection_and_counts(
    container: AppContainer,
) -> None:
    await _connect(container, GOOGLE_HEALTH_NUTRITION_SCOPES)
    await container.google_health_storage.nutrition.enqueue(
        meal_id=1,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )

    result = await get_meal_sync_status(_context())

    assert result == {
        "status": "success",
        "google_health_connection": "connected",
        "nutrition_permissions": True,
        "google_health_sync": {"pending": 1},
    }


@pytest.mark.asyncio
async def test_retry_meal_sync_requeues_failed_rows_and_wakes_worker(
    container: AppContainer,
) -> None:
    await _connect(container, GOOGLE_HEALTH_NUTRITION_SCOPES)
    nutrition = container.google_health_storage.nutrition
    await nutrition.enqueue(
        meal_id=2,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    revision = (await nutrition.revisions(2))[0]
    await nutrition.revision_state(int(revision["sequence"]), "failed")
    await nutrition.result(2, "failed", error="bad_request")
    worker = MagicMock()
    container.nutrition_export_worker = worker

    result = await retry_meal_sync(_context())

    assert result["status"] == "success"
    assert result["requeued"] == 1
    assert result["google_health_sync"] == {"pending": 1}
    worker.wake.assert_called_once()
    assert (await nutrition.revisions(2))[0]["state"] == "queued"


@pytest.mark.asyncio
async def test_retry_meal_sync_requires_private_chat_and_nutrition_scopes(
    container: AppContainer,
) -> None:
    private_error = await retry_meal_sync(_context(private=False))
    assert private_error["status"] == "error"

    await _connect(container, ())
    missing_scopes = await retry_meal_sync(_context())
    assert missing_scopes["status"] == "authorization_required"
    assert missing_scopes["requeued"] == 0


@pytest.mark.asyncio
async def test_status_tools_return_safe_no_connection_and_invalid_identity(
    container: AppContainer,
) -> None:
    no_connection = await get_meal_sync_status(_context())
    assert no_connection["google_health_connection"] == "not_connected"
    assert no_connection["google_health_sync"] == {}

    invalid = await get_meal_sync_status(_context(user_id="not-a-private-user"))
    assert invalid["status"] == "error"
    private_error = await get_meal_sync_status(_context(private=False))
    assert private_error["status"] == "error"
    retry_invalid = await retry_meal_sync(_context(user_id="not-a-private-user"))
    assert retry_invalid["status"] == "error"
    retry_no_connection = await retry_meal_sync(_context())
    assert retry_no_connection["status"] == "not_connected"


@pytest.mark.asyncio
async def test_status_tools_handle_uninitialized_storage() -> None:
    reset_container_for_tests()
    context = _context()
    assert (await get_meal_sync_status(context))["status"] == "error"
    assert (await retry_meal_sync(context))["status"] == "error"


@pytest.mark.asyncio
async def test_status_tools_handle_storage_failure(
    container: AppContainer,
) -> None:
    with patch.object(
        container.google_health_storage,
        "get_connection",
        side_effect=RuntimeError("storage unavailable"),
    ):
        result = await get_meal_sync_status(_context())
    assert result["status"] == "error"
    assert result["message"] == "Health storage is not initialized"


@pytest.mark.asyncio
async def test_status_tools_handle_unexpected_failures(
    container: AppContainer,
) -> None:
    with patch.object(
        container.google_health_storage,
        "get_connection",
        side_effect=ValueError("unexpected"),
    ):
        status = await get_meal_sync_status(_context())
    assert status == {
        "status": "error",
        "message": "Could not read meal export status",
    }

    await _connect(container, GOOGLE_HEALTH_NUTRITION_SCOPES)
    with patch.object(
        container.google_health_storage.nutrition,
        "retry_failed",
        side_effect=ValueError("unexpected"),
    ):
        retry = await retry_meal_sync(_context())
    assert retry == {
        "status": "error",
        "message": "Could not retry meal exports",
    }


@pytest.mark.asyncio
async def test_retry_without_running_worker_returns_counts(
    container: AppContainer,
) -> None:
    await _connect(container, GOOGLE_HEALTH_NUTRITION_SCOPES)
    nutrition = container.google_health_storage.nutrition
    await nutrition.enqueue(
        meal_id=3,
        owner_id=USER_ID,
        telegram_user_id=USER_ID,
        health_user_id=HEALTH_USER_ID,
        payload=PAYLOAD,
        operation="upsert",
    )
    revision = (await nutrition.revisions(3))[0]
    await nutrition.revision_state(int(revision["sequence"]), "failed")
    await nutrition.result(3, "failed", error="bad_request")

    result = await retry_meal_sync(_context())

    assert result["status"] == "success"
    assert result["requeued"] == 1
