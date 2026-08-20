"""Tests for local Telegram authorization and identity storage."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, create_autospec, patch

import pytest

from blacki.storage.sqlite import create_connection
from blacki.telegram import TelegramConfig
from blacki.telegram.access import (
    TelegramAccessStorage,
    TelegramIdentity,
    get_telegram_access_storage,
)
from blacki.telegram.api import TelegramApiClient, TelegramApiError
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import Update


@pytest.fixture
async def storage(tmp_path: Path) -> AsyncIterator[TelegramAccessStorage]:
    connection = await create_connection(tmp_path / "tools.db")
    result = TelegramAccessStorage(connection, asyncio.Lock())
    await result.initialize()
    try:
        yield result
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_passphrase_authorization_is_invalidated_by_rotation(
    storage: TelegramAccessStorage,
) -> None:
    await storage.grant(
        42,
        source="passphrase",
        access_code_fingerprint="before-rotation",
    )

    assert await storage.is_authorized(42, "before-rotation") is True
    assert await storage.is_authorized(42, "after-rotation") is False
    assert await storage.has_authorization_record(42) is True


@pytest.mark.asyncio
async def test_legacy_authorization_survives_access_code_rotation(
    storage: TelegramAccessStorage,
) -> None:
    await storage.grant(42, source="legacy")

    assert await storage.is_authorized(42, "before-rotation") is True
    assert await storage.is_authorized(42, "after-rotation") is True


@pytest.mark.asyncio
async def test_identity_is_updated_without_affecting_authorization(
    storage: TelegramAccessStorage,
) -> None:
    await storage.grant(42, source="legacy")
    await storage.record_identity(TelegramIdentity(42, "First Name", "first"))
    await storage.record_identity(TelegramIdentity(42, "New Name", "new"))

    row = await storage._fetch_one(
        """
        SELECT display_name, username
        FROM telegram_identities
        WHERE telegram_user_id = ?
        """,
        (42,),
    )

    assert row == {"display_name": "New Name", "username": "new"}
    assert await storage.is_authorized(42, "any-code") is True


@pytest.mark.asyncio
async def test_get_telegram_access_storage_uses_initialized_container(
    tmp_path: Path,
) -> None:
    from blacki.container import (
        reset_container_for_tests,
        set_container_from_connection,
    )

    connection = await create_connection(tmp_path / "tools.db")
    container = set_container_from_connection(connection)
    try:
        with pytest.raises(RuntimeError, match="not initialized"):
            get_telegram_access_storage()
        await container.telegram_access_storage.initialize()
        assert get_telegram_access_storage() is container.telegram_access_storage
    finally:
        reset_container_for_tests()
        await connection.close()


class _Runtime:
    def __init__(self, has_history: bool) -> None:
        self.has_history = has_history

    async def has_existing_session(self, **_kwargs: object) -> bool:
        return self.has_history


def _update(text: str, *, chat_type: str = "private", sender_id: int = 42) -> Update:
    return Update.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 2,
                "date": "2026-08-20T00:00:00Z",
                "chat": {"id": 42, "type": chat_type},
                "from": {
                    "id": sender_id,
                    "first_name": "Ada",
                    "username": "ada",
                },
                "text": text,
            },
        }
    )


def _callback_update(*, message: dict[str, object] | None) -> Update:
    return Update.model_validate(
        {
            "update_id": 1,
            "callback_query": {
                "id": "callback-1",
                "from": {"id": 42, "first_name": "Ada"},
                "chat_instance": "instance-1",
                "data": "setting:model",
                "message": message,
            },
        }
    )


def _bot(
    storage: TelegramAccessStorage,
    *,
    has_history: bool,
    access_code: str | None = "test-access-code",
) -> TelegramBot:
    config = TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": True,
            "TELEGRAM_BOT_TOKEN": "test-token-123",
            "TELEGRAM_ACCESS_CODE": access_code,
        }
    )
    bot = TelegramBot(config, _Runtime(has_history), access_storage=storage)  # type: ignore[arg-type]
    api = create_autospec(TelegramApiClient, instance=True)
    api.send_message = AsyncMock()
    api.delete_message = AsyncMock()
    bot._api = api
    return bot


@pytest.mark.asyncio
async def test_start_access_code_authorizes_new_user(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)

    allowed = await bot._authorize_update(_update("/start test-access-code"))

    assert allowed is False
    assert await storage.is_authorized(42, bot._access_code_fingerprint()) is True
    assert cast(AsyncMock, cast(Any, bot.api).delete_message).await_count == 1


@pytest.mark.asyncio
async def test_historical_private_chat_is_grandfathered(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=True)

    allowed = await bot._authorize_update(_update("normal message"))

    assert allowed is True
    assert await storage.is_authorized(42, "rotated-code") is True


@pytest.mark.asyncio
async def test_unconfigured_access_control_still_records_identity(
    storage: TelegramAccessStorage,
) -> None:
    """Legacy open mode must still populate dashboard identity labels."""
    bot = _bot(storage, has_history=False, access_code=None)

    assert await bot._authorize_update(_update("normal message")) is True

    row = await storage._fetch_one(
        """
        SELECT display_name, username
        FROM telegram_identities
        WHERE telegram_user_id = ?
        """,
        (42,),
    )
    assert row == {"display_name": "Ada", "username": "ada"}


@pytest.mark.asyncio
async def test_open_mode_continues_when_identity_storage_is_unavailable(
    storage: TelegramAccessStorage,
) -> None:
    """An unavailable optional label store must not block open-mode traffic."""
    bot = _bot(storage, has_history=False, access_code=None)
    bot.access_storage = None

    with patch(
        "blacki.telegram.bot.get_telegram_access_storage",
        side_effect=RuntimeError("storage is not initialized"),
    ):
        assert await bot._authorize_update(_update("normal message")) is True


@pytest.mark.asyncio
async def test_open_mode_continues_when_identity_write_fails(
    storage: TelegramAccessStorage,
) -> None:
    """A local identity write failure must not block the Telegram update."""
    await storage._conn.close()
    bot = _bot(storage, has_history=False, access_code=None)

    assert await bot._authorize_update(_update("normal message")) is True


@pytest.mark.asyncio
async def test_rotated_passphrase_user_is_not_regrandfathered(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=True)
    await storage.grant(
        42,
        source="passphrase",
        access_code_fingerprint="old-code",
    )

    assert await bot._authorize_update(_update("normal message")) is False
    assert await storage.is_authorized(42, bot._access_code_fingerprint()) is False


@pytest.mark.asyncio
async def test_authorized_user_bypasses_legacy_lookup(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)
    await storage.grant(
        42,
        source="passphrase",
        access_code_fingerprint=bot._access_code_fingerprint(),
    )

    assert await bot._authorize_update(_update("normal message")) is True


@pytest.mark.asyncio
async def test_new_user_without_start_code_is_denied(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)

    assert await bot._authorize_update(_update("hello")) is False
    cast(Any, bot.api).send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_start_code_is_deleted_and_denied(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)
    api = cast(Any, bot.api)
    api.delete_message.side_effect = TelegramApiError("delete failed", 400)

    assert await bot._authorize_update(_update("/start wrong-code")) is False
    assert await storage.is_authorized(42, bot._access_code_fingerprint()) is False
    api.delete_message.assert_awaited_once()
    api.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_group_and_mismatched_sender_are_rejected(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)

    assert await bot._authorize_update(_update("hello", chat_type="group")) is False
    assert await bot._authorize_update(_update("hello", sender_id=7)) is False


@pytest.mark.asyncio
async def test_callback_without_private_message_is_rejected(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)

    assert await bot._authorize_update(_callback_update(message=None)) is False
    cast(Any, bot.api).answer_callback_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_unauthorized_callback_is_rejected_after_identity_check(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)
    message = {
        "message_id": 2,
        "date": "2026-08-20T00:00:00Z",
        "chat": {"id": 42, "type": "private"},
        "from": {"id": 42, "first_name": "Ada"},
        "text": "Settings",
    }

    assert await bot._authorize_update(_callback_update(message=message)) is False
    cast(Any, bot.api).answer_callback_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_access_storage_failure_fails_closed(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)
    cast(Any, storage).is_authorized = AsyncMock(side_effect=RuntimeError("database"))

    assert await bot._authorize_update(_update("hello")) is False


@pytest.mark.asyncio
async def test_safe_handler_does_not_route_denied_update(
    storage: TelegramAccessStorage,
) -> None:
    bot = _bot(storage, has_history=False)
    handle_update = AsyncMock()
    with patch.object(bot, "_handle_update", new=handle_update):
        await bot._safe_handle_update(_update("hello"))

    handle_update.assert_not_awaited()
