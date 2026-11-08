"""Telegram bot glue for the Mini App: hooks, menu button, prompts and commands."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock, create_autospec

import pytest

from blacki.miniapp.service import hooks
from blacki.runtime.adk import AdkRuntime
from blacki.telegram import TelegramConfig
from blacki.telegram.access import TelegramAccessStorage
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import ChatType, Message

LAUNCH_URL = "https://app.example.test/miniapp/"
ACCESS_CODE = "open-sesame"


@pytest.fixture(autouse=True)
def restore_miniapp_hooks() -> Iterator[None]:
    """The Mini App hooks are process-wide; restore them after every test."""
    original_authorize = hooks.authorize
    original_send_prompt = hooks.send_prompt
    yield
    hooks.authorize = original_authorize
    hooks.send_prompt = original_send_prompt


def _config(access_code: str | None = None) -> TelegramConfig:
    values: dict[str, object] = {
        "TELEGRAM_ENABLED": True,
        "TELEGRAM_BOT_TOKEN": "123:test-token",
    }
    if access_code is not None:
        values["TELEGRAM_ACCESS_CODE"] = access_code
    return TelegramConfig.model_validate(values)


def _message(chat_id: int = 42, command: str = "/app") -> Message:
    return Message.model_validate(
        {
            "message_id": 1,
            "date": "2026-08-16T00:00:00Z",
            "chat": {"id": chat_id, "type": ChatType.PRIVATE.value},
            "from": {"id": chat_id, "first_name": "Test", "is_bot": False},
            "text": command,
        }
    )


def _bot(
    *,
    access_code: str | None = None,
    access_storage: TelegramAccessStorage | None = None,
    runtime: MagicMock | None = None,
) -> tuple[TelegramBot, MagicMock]:
    runtime = runtime or create_autospec(AdkRuntime, instance=True, spec_set=True)
    bot = TelegramBot(
        _config(access_code),
        cast(AdkRuntime, runtime),
        access_storage=access_storage,
    )
    api = create_autospec(TelegramApiClient, instance=True, spec_set=True)
    bot._api = api
    return bot, api


async def _drain_background(bot: TelegramBot) -> None:
    """Wait for every background task the bot started, including chained ones."""
    while bot._background_tasks:
        await asyncio.gather(*list(bot._background_tasks), return_exceptions=True)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_attach_without_launch_url_wires_hooks_without_menu_button() -> None:
    """With no public URL the bot wires access and prompts but sets no menu."""
    bot, api = _bot()

    bot.attach_miniapp(None)
    await _drain_background(bot)

    assert hooks.authorize == bot.miniapp_user_authorized
    assert hooks.send_prompt == bot.submit_miniapp_prompt
    api.set_chat_menu_button.assert_not_awaited()


@pytest.mark.asyncio
async def test_attach_sets_web_app_menu_button_in_background() -> None:
    """A configured launch URL becomes Telegram's web_app menu button."""
    bot, api = _bot()

    bot.attach_miniapp(LAUNCH_URL)
    await _drain_background(bot)

    api.set_chat_menu_button.assert_awaited_once_with(
        menu_button={
            "type": "web_app",
            "text": "Open app",
            "web_app": {"url": LAUNCH_URL},
        }
    )
    assert bot._miniapp_url == LAUNCH_URL


@pytest.mark.asyncio
async def test_menu_button_failure_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected menu button must not stop the bot from starting."""
    bot, api = _bot()
    api.set_chat_menu_button.side_effect = RuntimeError("Telegram refused")

    with caplog.at_level(logging.ERROR):
        bot.attach_miniapp(LAUNCH_URL)
        await _drain_background(bot)

    assert "Could not set the Mini App menu button" in caplog.text


@pytest.mark.asyncio
async def test_access_check_allows_everyone_without_access_code() -> None:
    """Without an access code the Mini App follows the open bot policy."""
    storage = create_autospec(TelegramAccessStorage, instance=True, spec_set=True)
    bot, _ = _bot(access_storage=storage)

    assert await bot.miniapp_user_authorized(42) is True
    storage.is_authorized.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [True, False])
async def test_access_check_delegates_with_code_fingerprint(authorized: bool) -> None:
    """Access is decided by stored authorization, keyed by the access-code hash."""
    storage = create_autospec(TelegramAccessStorage, instance=True, spec_set=True)
    storage.is_authorized.return_value = authorized
    bot, _ = _bot(access_code=ACCESS_CODE, access_storage=storage)

    assert await bot.miniapp_user_authorized(42) is authorized
    expected_fingerprint = hashlib.sha256(ACCESS_CODE.encode("utf-8")).hexdigest()
    storage.is_authorized.assert_awaited_once_with(42, expected_fingerprint)


@pytest.mark.asyncio
async def test_access_check_fails_closed_on_storage_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A storage failure must deny the Mini App, never grant it."""
    storage = create_autospec(TelegramAccessStorage, instance=True, spec_set=True)
    storage.is_authorized.side_effect = RuntimeError("database locked")
    bot, _ = _bot(access_code=ACCESS_CODE, access_storage=storage)

    with caplog.at_level(logging.ERROR):
        assert await bot.miniapp_user_authorized(42) is False

    assert "Mini App access check failed closed" in caplog.text


@pytest.mark.asyncio
async def test_submit_prompt_echoes_and_runs_turn_with_miniapp_origin() -> None:
    """A confirmed app question is echoed to chat and run as a marked turn."""
    runtime = create_autospec(AdkRuntime, instance=True, spec_set=True)
    runtime.run_user_turn.return_value = "Here is the answer"
    bot, api = _bot(runtime=runtime)

    await bot.submit_miniapp_prompt(42, "plan my week")
    await _drain_background(bot)

    api.send_message.assert_any_await(chat_id=42, text="From the app: plan my week")
    runtime.run_user_turn.assert_awaited_once()
    kwargs = runtime.run_user_turn.call_args.kwargs
    assert kwargs["message_text"].endswith("plan my week")
    assert "Sent from the Blacki Mini App" in kwargs["message_text"]
    assert kwargs["state"]["temp:origin"] == "miniapp"
    assert any(
        "Here is the answer" in call.kwargs.get("text", "")
        for call in api.send_message.await_args_list
    )


@pytest.mark.asyncio
async def test_submit_prompt_logs_turn_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed app-initiated turn is logged and does not crash the bot."""
    runtime = create_autospec(AdkRuntime, instance=True, spec_set=True)
    runtime.run_user_turn.side_effect = RuntimeError("model unavailable")
    bot, _ = _bot(runtime=runtime)

    with caplog.at_level(logging.ERROR):
        await bot.submit_miniapp_prompt(42, "hello")
        await _drain_background(bot)

    assert "Mini App prompt turn failed" in caplog.text


@pytest.mark.asyncio
async def test_submit_prompt_cancelled_turn_is_logged_as_superseded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A turn superseded by a newer message is an expected outcome, not an error."""
    runtime = create_autospec(AdkRuntime, instance=True, spec_set=True)
    runtime.run_user_turn.side_effect = asyncio.CancelledError()
    bot, _ = _bot(runtime=runtime)

    with caplog.at_level(logging.INFO):
        await bot.submit_miniapp_prompt(42, "hello")
        await _drain_background(bot)

    assert "Mini App prompt superseded by a newer message" in caplog.text
    assert "Mini App prompt turn failed" not in caplog.text


@pytest.mark.asyncio
async def test_app_command_explains_setup_without_public_url() -> None:
    """The /app command must say what is missing when no URL is configured."""
    bot, api = _bot()

    await bot._handle_command(_message(), "/app")

    assert "isn't set up yet" in api.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_app_command_sends_open_app_web_app_button() -> None:
    """With a launch URL, /app replies with a button opening the Mini App."""
    bot, api = _bot()
    bot.attach_miniapp(LAUNCH_URL)
    await _drain_background(bot)

    await bot._handle_command(_message(), "/app")

    kwargs = api.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 42
    button = kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.text == "Open app"
    assert button.web_app.url == LAUNCH_URL


@pytest.mark.asyncio
@pytest.mark.parametrize(("env_value", "expected"), [("true", True), ("false", False)])
async def test_app_command_listed_only_when_miniapp_enabled(
    monkeypatch: pytest.MonkeyPatch, env_value: str, expected: bool
) -> None:
    """The /app command appears in Telegram's menu only when the Mini App is on."""
    monkeypatch.setenv("MINIAPP_ENABLED", env_value)
    bot, api = _bot()

    await bot._register_commands()

    commands = api.set_my_commands.call_args.args[0]
    assert ("app" in {command.command for command in commands}) is expected
