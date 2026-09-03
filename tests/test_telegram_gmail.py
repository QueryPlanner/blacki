"""Telegram command and callback tests for the Gmail API connector."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from blacki.adk_runtime import AdkRuntime
from blacki.gmail import (
    GmailAlreadyConnectedError,
    GmailCredentialError,
    GmailOAuthService,
)
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import CallbackQuery, ChatType, Message


def _config() -> TelegramConfig:
    return TelegramConfig.model_validate(
        {"TELEGRAM_ENABLED": True, "TELEGRAM_BOT_TOKEN": "123:test-token"}
    )


def _message(
    chat_id: int = 42,
    chat_type: ChatType = ChatType.PRIVATE,
    command: str = "/connect_gmail",
) -> Message:
    return Message.model_validate(
        {
            "message_id": 1,
            "date": "2026-08-16T00:00:00Z",
            "chat": {"id": chat_id, "type": chat_type.value},
            "from": {"id": chat_id, "first_name": "Test", "is_bot": False},
            "text": command,
        }
    )


def _bot() -> tuple[TelegramBot, MagicMock, MagicMock]:
    service = create_autospec(GmailOAuthService, instance=True, spec_set=True)
    service.begin_authorization = AsyncMock(
        return_value="https://accounts.google.com/o/oauth2/v2/auth?state=state"
    )
    service.disconnect = AsyncMock(return_value=True)
    runtime = MagicMock()
    bot = TelegramBot(
        _config(),
        cast(AdkRuntime, runtime),
        gmail_oauth_service=service,
    )
    api = create_autospec(TelegramApiClient, instance=True, spec_set=True)
    api.send_message = AsyncMock()
    api.answer_callback_query = AsyncMock()
    api.set_my_commands = AsyncMock()
    bot._api = api
    return bot, service, api


@pytest.mark.asyncio
async def test_gmail_commands_reject_group_chats() -> None:
    bot, service, api = _bot()
    message = _message(chat_id=-100, chat_type=ChatType.SUPERGROUP)
    await bot._handle_command(message, "/connect_gmail")
    await bot._handle_command(
        message.model_copy(update={"text": "/disconnect_gmail"}), "/disconnect_gmail"
    )
    service.begin_authorization.assert_not_awaited()
    service.disconnect.assert_not_awaited()
    assert api.send_message.await_count == 2
    assert all(
        "private Telegram chat" in call.kwargs["text"]
        for call in api.send_message.await_args_list
    )


@pytest.mark.asyncio
async def test_gmail_commands_report_unconfigured_service() -> None:
    runtime = MagicMock()
    bot = TelegramBot(_config(), cast(AdkRuntime, runtime))
    api = create_autospec(TelegramApiClient, instance=True, spec_set=True)
    api.send_message = AsyncMock()
    bot._api = api
    await bot._handle_command(_message(), "/connect_gmail")
    assert "not configured" in api.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_connect_gmail_sends_protected_api_authorization_link() -> None:
    bot, service, api = _bot()
    await bot._handle_command(_message(), "/connect_gmail")
    service.begin_authorization.assert_awaited_once_with("telegram-chat-42")
    kwargs = api.send_message.call_args.kwargs
    assert kwargs["protect_content"] is True
    assert "Gmail API" in kwargs["text"]
    assert "configured LLM and conversation storage" in kwargs["text"]
    button = kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.url.startswith("https://accounts.google.com/")


@pytest.mark.asyncio
async def test_connect_gmail_refuses_replacing_connected_account() -> None:
    bot, service, api = _bot()
    service.begin_authorization.side_effect = GmailAlreadyConnectedError("connected")
    await bot._handle_command(_message(), "/connect_gmail")
    assert "disconnect_gmail" in api.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_connect_gmail_reports_credential_failure() -> None:
    bot, service, api = _bot()
    service.begin_authorization.side_effect = GmailCredentialError("unavailable")
    await bot._handle_command(_message(), "/connect_gmail")
    assert "not available" in api.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_disconnect_gmail_requires_owned_inline_confirmation() -> None:
    bot, service, api = _bot()
    message = _message(command="/disconnect_gmail")
    await bot._handle_command(message, "/disconnect_gmail")
    markup = api.send_message.call_args.kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].callback_data == "gmail:disconnect"
    assert service.disconnect.await_count == 0

    unauthorized = CallbackQuery.model_validate(
        {
            "id": "unauthorized",
            "from": {"id": 99, "first_name": "Other", "is_bot": False},
            "message": message.model_dump(by_alias=True),
            "chat_instance": "chat",
            "data": "gmail:disconnect",
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
            "data": "gmail:cancel",
        }
    )
    await bot._handle_callback_query(cancelled)
    service.disconnect.assert_not_awaited()

    confirmed = cancelled.model_copy(
        update={"id": "confirm", "data": "gmail:disconnect"}
    )
    await bot._handle_callback_query(confirmed)
    service.disconnect.assert_awaited_once_with("telegram-chat-42")
    assert "remote revocation" in api.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_gmail_callback_handles_missing_service_and_failure() -> None:
    bot, service, api = _bot()
    message = _message(command="/disconnect_gmail")
    no_message = CallbackQuery.model_validate(
        {
            "id": "missing-message",
            "from": {"id": 42, "first_name": "Test", "is_bot": False},
            "message": None,
            "chat_instance": "chat",
            "data": "gmail:disconnect",
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
            "data": "gmail:disconnect",
        }
    )
    await unconfigured._handle_callback_query(authorized)
    assert (
        api.answer_callback_query.call_args.kwargs["text"] == "Not authorized"
        or api.answer_callback_query.call_args.kwargs["text"] == "Not configured"
    )

    service.disconnect.side_effect = RuntimeError("provider")
    await bot._handle_callback_query(authorized)
    assert "couldn't finish" in api.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_gmail_callback_notification_is_private_and_safe() -> None:
    bot, _, api = _bot()
    await bot.notify_gmail_connected("telegram-chat-42", connected=True)
    await bot.notify_gmail_connected("telegram-chat-42", connected=False)
    await bot.notify_gmail_connected(42)
    await bot.notify_gmail_connected("not-a-telegram-user")
    assert api.send_message.await_count == 3
    assert "Gmail API" in api.send_message.await_args_list[0].kwargs["text"]
    assert "cancelled" in api.send_message.await_args_list[1].kwargs["text"]


@pytest.mark.asyncio
async def test_register_commands_includes_gmail() -> None:
    bot, _, api = _bot()
    await bot._register_commands()
    commands = api.set_my_commands.call_args.args[0]
    assert {command.command for command in commands} >= {
        "connect_gmail",
        "disconnect_gmail",
    }
