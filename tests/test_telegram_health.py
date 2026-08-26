"""Telegram command and confirmation tests for Connect Google Health."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from blacki.adk_runtime import AdkRuntime
from blacki.health.service import (
    GoogleHealthOAuthError,
    GoogleHealthService,
    SyncResult,
)
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import (
    TelegramBot,
    _format_google_health_sync_counts,
    _format_health_sync_result,
)
from blacki.telegram.types import CallbackQuery, ChatType, Message


def _config() -> TelegramConfig:
    return TelegramConfig.model_validate(
        {"TELEGRAM_ENABLED": True, "TELEGRAM_BOT_TOKEN": "123:test-token"}
    )


def _message(chat_id: int = 42, chat_type: ChatType = ChatType.PRIVATE) -> Message:
    return Message.model_validate(
        {
            "message_id": 1,
            "date": "2026-08-16T00:00:00Z",
            "chat": {"id": chat_id, "type": chat_type.value},
            "from": {"id": chat_id, "first_name": "Test", "is_bot": False},
            "text": "/health_summary",
        }
    )


def _bot() -> tuple[TelegramBot, MagicMock, MagicMock]:
    service = create_autospec(GoogleHealthService, instance=True, spec_set=True)
    service.begin_authorization = AsyncMock(
        return_value="https://accounts.google.com/o/oauth2/v2/auth?state=state"
    )
    service.summary = AsyncMock(
        return_value={"status": "success", "days": [], "trends": {}}
    )
    service.refresh_user = AsyncMock(
        return_value=SyncResult(
            status="success", telegram_user_id="telegram-chat-42", days_upserted=1
        )
    )
    service.disconnect = AsyncMock(return_value=True)
    runtime = MagicMock()
    bot = TelegramBot(
        _config(),
        cast(AdkRuntime, runtime),
        google_health_service=service,
    )
    api = create_autospec(TelegramApiClient, instance=True, spec_set=True)
    api.send_message = AsyncMock()
    api.answer_callback_query = AsyncMock()
    bot._api = api
    return bot, service, api


@pytest.mark.asyncio
async def test_connect_health_sends_protected_authorization_link() -> None:
    """Telegram receives only a short-lived link and safe explanatory copy."""
    bot, service, api = _bot()
    await bot._handle_command(_message(), "/connect_health")
    service.begin_authorization.assert_awaited_once_with("telegram-chat-42")
    kwargs = api.send_message.call_args.kwargs
    assert kwargs["protect_content"] is True
    assert (
        kwargs["reply_markup"]
        .inline_keyboard[0][0]
        .url.startswith("https://accounts.google.com/")
    )
    assert "future meal logs, edits, and deletions" in kwargs["text"]
    assert "not backfilled" in kwargs["text"]
    assert "verify records it created" in kwargs["text"]
    assert "Read-only summaries remain available" in kwargs["text"]
    assert "Apple ID" in kwargs["text"]


@pytest.mark.asyncio
async def test_register_commands_includes_health_when_configured() -> None:
    """The optional commands appear only on the configured Telegram root bot."""
    bot, _, api = _bot()
    api.set_my_commands = AsyncMock()
    await bot._register_commands()
    commands = api.set_my_commands.call_args.args[0]
    assert "connect_health" in {command.command for command in commands}


@pytest.mark.asyncio
async def test_health_commands_reject_group_chats() -> None:
    """Health commands cannot bind shared group identities to one person."""
    bot, service, api = _bot()
    message = _message(chat_id=-100, chat_type=ChatType.SUPERGROUP)
    await bot._handle_command(message, "/connect_health")
    await bot._handle_command(message, "/health_summary")
    await bot._handle_command(message, "/health_refresh")
    await bot._handle_command(message, "/disconnect_health")
    service.begin_authorization.assert_not_awaited()
    service.summary.assert_not_awaited()
    service.refresh_user.assert_not_awaited()
    service.disconnect.assert_not_awaited()
    assert api.send_message.await_count == 4
    assert all(
        "private Telegram chat" in call.kwargs["text"]
        for call in api.send_message.await_args_list
    )


@pytest.mark.asyncio
async def test_health_summary_and_refresh_are_readable() -> None:
    """Summary reads stored data and refresh reports a safe result."""
    bot, service, api = _bot()
    message = _message()
    await bot._handle_command(message, "/health_summary")
    await bot._handle_command(message, "/health_refresh")
    service.summary.assert_awaited()
    service.refresh_user.assert_awaited_once_with("telegram-chat-42")
    assert api.send_message.await_count == 2
    assert all(
        call.kwargs["protect_content"] for call in api.send_message.await_args_list
    )


@pytest.mark.asyncio
async def test_health_refresh_success_includes_sync_counts() -> None:
    """A successful refresh still reports durable meal-export counts."""
    bot, service, api = _bot()
    service.refresh_user.return_value = SyncResult(
        status="success",
        telegram_user_id="telegram-chat-42",
        days_upserted=1,
        records_fetched=2,
        google_health_sync={"pending": 1, "synced": 3},
    )
    await bot._handle_command(_message(), "/health_refresh")
    text = api.send_message.call_args.kwargs["text"]
    assert "1 pending" in text
    assert "3 synced" in text


@pytest.mark.asyncio
async def test_health_refresh_status_messages() -> None:
    """Non-success sync states remain provider- and identity-safe."""
    bot, service, api = _bot()
    statuses = ("not_connected", "reauthorization_required", "rate_limited", "failed")
    for status in statuses:
        service.refresh_user.return_value = SyncResult(
            status=status, telegram_user_id="telegram-chat-42"
        )
        await bot._handle_command(_message(), "/health_refresh")
    assert api.send_message.await_count == len(statuses)
    assert all(
        "telegram-chat-42" not in call.kwargs["text"]
        for call in api.send_message.await_args_list
    )


@pytest.mark.asyncio
async def test_disconnect_requires_confirmation_and_checks_callback_user() -> None:
    """Deletion occurs only after the private chat's explicit callback click."""
    bot, service, api = _bot()
    message = _message()
    await bot._handle_command(message, "/disconnect_health")
    markup = api.send_message.call_args.kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].callback_data == "health:disconnect"
    assert service.disconnect.await_count == 0

    unauthorized = CallbackQuery.model_validate(
        {
            "id": "unauthorized",
            "from": {"id": 99, "first_name": "Other", "is_bot": False},
            "message": message.model_dump(by_alias=True),
            "chat_instance": "chat",
            "data": "health:disconnect",
        }
    )
    await bot._handle_callback_query(unauthorized)
    service.disconnect.assert_not_awaited()
    assert api.answer_callback_query.call_args.kwargs["text"] == "Not authorized"

    cancelled = CallbackQuery.model_validate(
        {
            "id": "cancel",
            "from": {"id": 42, "first_name": "Test", "is_bot": False},
            "message": message.model_dump(by_alias=True),
            "chat_instance": "chat",
            "data": "health:cancel",
        }
    )
    await bot._handle_callback_query(cancelled)
    service.disconnect.assert_not_awaited()

    confirmed = cancelled.model_copy(
        update={"id": "confirm", "data": "health:disconnect"}
    )
    await bot._handle_callback_query(confirmed)
    service.disconnect.assert_awaited_once_with("telegram-chat-42")
    text = api.send_message.call_args.kwargs["text"]
    assert "Pending meal sync was cancelled" in text
    assert "local calorie logs remain" in text
    assert "did not delete records already sent" in text
    assert "may still finish" in text


@pytest.mark.asyncio
async def test_health_callback_notification_and_private_state() -> None:
    """OAuth completion notifications are scoped to the originating chat."""
    bot, service, api = _bot()
    await bot.notify_health_connection("telegram-chat-42", connected=True)
    await bot.notify_health_connection("telegram-chat-42", connected=False)
    await bot.notify_health_connection("not-a-telegram-user", connected=True)
    assert api.send_message.await_count == 2
    assert "connected" in api.send_message.await_args_list[0].kwargs["text"]
    assert "cancelled" in api.send_message.await_args_list[1].kwargs["text"]

    state = bot._build_session_state(
        chat_id="42",
        message_thread_id=None,
        conversation_key="chat-42",
        chat_type=ChatType.PRIVATE,
    )
    assert state["telegram_chat_type"] == "private"


@pytest.mark.asyncio
async def test_health_command_failures_and_unconfigured_bot() -> None:
    """Optional connector failures degrade to safe Telegram messages."""
    bot, service, api = _bot()
    service.begin_authorization.side_effect = ValueError("bad config")
    await bot._handle_command(_message(), "/connect_health")
    service.summary.side_effect = RuntimeError("database")
    await bot._handle_command(_message(), "/health_summary")
    service.refresh_user.side_effect = RuntimeError("provider")
    await bot._handle_command(_message(), "/health_refresh")
    service.summary.side_effect = GoogleHealthOAuthError("private chat")
    await bot._handle_command(_message(), "/health_summary")
    service.refresh_user.side_effect = GoogleHealthOAuthError("private chat")
    await bot._handle_command(_message(), "/health_refresh")
    assert all(
        "database" not in call.kwargs["text"]
        for call in api.send_message.await_args_list
    )

    unconfigured = TelegramBot(_config(), cast(AdkRuntime, MagicMock()))
    unconfigured._api = api
    await unconfigured._handle_command(_message(), "/health_summary")
    assert "not configured" in api.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_health_callback_handles_missing_service_and_disconnect_failure() -> None:
    """Callbacks fail closed for missing configuration and deletion errors."""
    bot, service, api = _bot()
    message = _message()
    no_message = CallbackQuery.model_validate(
        {
            "id": "missing-message",
            "from": {"id": 42, "first_name": "Test", "is_bot": False},
            "message": None,
            "chat_instance": "chat",
            "data": "health:disconnect",
        }
    )
    await bot._handle_callback_query(no_message)
    assert api.answer_callback_query.call_args.kwargs["text"] == "Confirmation expired"

    unconfigured = TelegramBot(_config(), cast(AdkRuntime, MagicMock()))
    unconfigured._api = api
    authorized = CallbackQuery.model_validate(
        {
            "id": "not-configured",
            "from": {"id": 42, "first_name": "Test", "is_bot": False},
            "message": message.model_dump(by_alias=True),
            "chat_instance": "chat",
            "data": "health:disconnect",
        }
    )
    await unconfigured._handle_callback_query(authorized)
    assert api.answer_callback_query.call_args.kwargs["text"] == "Not configured"

    service.disconnect.side_effect = RuntimeError("database failure")
    await bot._handle_callback_query(authorized)
    assert "couldn't finish" in api.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_configured_start_and_help_list_health_commands() -> None:
    """Configured users see the optional health commands in Telegram help."""
    bot, _, api = _bot()
    await bot._send_start_message(42)
    await bot._send_help_message(42)
    assert "connect\\_health" in api.send_message.await_args_list[0].kwargs["text"]
    assert "health_refresh" in api.send_message.await_args_list[1].kwargs["text"]


def test_health_sync_formatter_covers_safe_statuses() -> None:
    """Keep direct command result formatting deterministic."""
    assert "connect_health" in _format_health_sync_result(
        SyncResult("not_connected", "telegram-chat-42")
    )
    assert "authorization" in _format_health_sync_result(
        SyncResult("reauthorization_required", "telegram-chat-42")
    )
    assert "recently" in _format_health_sync_result(
        SyncResult("rate_limited", "telegram-chat-42")
    )
    assert "1 day" in _format_health_sync_result(
        SyncResult("success", "telegram-chat-42", days_upserted=1, records_fetched=2)
    )
    assert "could not" in _format_health_sync_result(
        SyncResult("failed", "telegram-chat-42")
    )
    result_with_counts = cast(
        SyncResult,
        SimpleNamespace(
            status="failed",
            google_health_sync={
                "pending": 2,
                "synced": 3,
                "failed": 1,
                "authorization_required": 4,
            },
        ),
    )
    text = _format_health_sync_result(result_with_counts)
    assert "2 pending" in text
    assert "3 synced" in text
    assert "1 failed" in text
    assert "4 awaiting authorization" in text
    assert "pending includes deletions" in text


def test_format_google_health_sync_counts_skips_invalid_entries() -> None:
    """Non-mapping input, missing keys, and negative counts are all safe."""
    assert _format_google_health_sync_counts("not-a-mapping") == ""
    assert _format_google_health_sync_counts({}) == ""
    assert _format_google_health_sync_counts({"pending": -1, "synced": "bad"}) == ""

    partial = _format_google_health_sync_counts({"pending": 2, "synced": "bad"})
    assert partial == "Meal sync status (pending includes deletions): 2 pending"
