"""Behavior tests for always-on live Telegram tool progress."""

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import MockBaseTool, MockState, MockToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.genai.types import Content, FunctionCall, Part

import blacki.callbacks as callbacks_module
from blacki.callbacks import (
    notify_telegram_after_agent,
    notify_telegram_after_model,
    notify_telegram_before_tool,
    reset_telegram_tool_notify_rate_limiter_for_tests,
    telegram_live_tool_progress_enabled,
)
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiError


@pytest.fixture(autouse=True)
async def reset_live_tool_progress() -> AsyncIterator[None]:
    await reset_telegram_tool_notify_rate_limiter_for_tests()
    yield None
    await reset_telegram_tool_notify_rate_limiter_for_tests()


def configure_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "1s"), (4.9, "4s"), (65, "1m 5s"), (125, "2m 5s")],
)
def test_format_elapsed_duration(seconds: float, expected: str) -> None:
    assert callbacks_module._format_elapsed_duration(seconds) == expected


def test_live_progress_requires_configured_telegram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert telegram_live_tool_progress_enabled() is False

    configure_telegram(monkeypatch)
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "false")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "off")
    assert telegram_live_tool_progress_enabled() is True


def test_telegram_config_ignores_removed_progress_options() -> None:
    config = TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": "true",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_TOOL_NOTIFICATIONS": "false",
            "TELEGRAM_TOOL_PROGRESS_MODE": "off",
        }
    )

    assert config.is_configured() is True
    assert not hasattr(config, "telegram_tool_notifications")
    assert not hasattr(config, "telegram_tool_progress_mode")


@pytest.mark.asyncio
async def test_live_progress_is_inert_without_telegram_configuration() -> None:
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    with patch("blacki.callbacks.TelegramApiClient") as client_class:
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "test"},
            cast(ToolContext, context),
        )

    client_class.assert_not_called()


@pytest.mark.asyncio
async def test_live_progress_skips_non_telegram_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "test"},
            cast(ToolContext, MockToolContext(state=MockState({}))),
        )

    client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_live_progress_uses_one_message_for_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    client.edit_message_text = AsyncMock(return_value=MagicMock(message_id=7))
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    context.invocation_id = "turn-1"

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "first"},
            cast(ToolContext, context),
        )
        callbacks_module._INTERMEDIATE_NOTIFY_LAST["42"] = 0
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_calorie_summary")),
            {},
            cast(ToolContext, context),
        )

    client.send_message.assert_awaited_once()
    assert (
        client.send_message.await_args.kwargs["text"]
        == "Searching the web for *first*…"
    )
    client.edit_message_text.assert_awaited_once()
    assert (
        client.edit_message_text.await_args.kwargs["text"]
        == "Checking calorie summary…"
    )


@pytest.mark.asyncio
async def test_live_progress_uses_model_preamble_then_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    client.edit_message_text = AsyncMock(return_value=MagicMock(message_id=7))
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    context.invocation_id = "turn-2"
    response = LlmResponse(
        content=Content(
            parts=[
                Part.from_text(text="Checking your meals"),
                Part(function_call=FunctionCall(name="get_calorie_summary", args={})),
            ]
        )
    )
    tool_context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    tool_context.invocation_id = "turn-2"

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_after_model(context, response)
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_calorie_summary")),
            {},
            cast(ToolContext, tool_context),
        )
        await notify_telegram_after_agent(context)

    assert client.send_message.await_args.kwargs["text"] == "Checking your meals"
    assert client.edit_message_text.await_args.kwargs["text"].startswith(
        "✓ Worked for "
    )
    assert (42, None, "turn-2") not in callbacks_module._LIVE_STATUS_SESSIONS


@pytest.mark.asyncio
async def test_live_progress_redacts_private_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("send_text_to_speech")),
            {"text": "private spoken content"},
            cast(ToolContext, context),
        )

    assert client.send_message.await_args.kwargs["text"] == "Generating speech…"


@pytest.mark.asyncio
async def test_live_progress_telegram_errors_do_not_block_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(
        side_effect=TelegramApiError("bad request", error_code=400)
    )
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "test"},
            cast(ToolContext, context),
        )

    client.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_progress_ignores_thought_only_model_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    context.invocation_id = "turn-3"
    response = LlmResponse(
        content=Content(
            parts=[
                Part(text="internal", thought=True),
                Part(function_call=FunctionCall(name="brave_search", args={})),
            ]
        )
    )

    await notify_telegram_after_model(context, response)

    assert (42, None, "turn-3") not in callbacks_module._LIVE_STATUS_SESSIONS


def test_live_progress_treats_blank_token_as_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "   ")

    assert telegram_live_tool_progress_enabled() is False
