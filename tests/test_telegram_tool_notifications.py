"""Tests for Telegram tool notification callback (issue #14)."""

import asyncio
import logging
from collections.abc import Iterator
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
    notify_telegram_before_tool,
    reset_telegram_tool_notify_rate_limiter_for_tests,
    telegram_tool_notifications_enabled,
)
from blacki.telegram.api import TelegramApiError


@pytest.fixture(autouse=True)
def _clear_tool_notify_rate_limiter() -> Iterator[None]:
    """Isolate rate limiter state between tests."""
    reset_telegram_tool_notify_rate_limiter_for_tests()
    yield None
    reset_telegram_tool_notify_rate_limiter_for_tests()


def test_telegram_tool_notifications_enabled_requires_all_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature is off unless Telegram is configured and the opt-in flag is set."""
    monkeypatch.delenv("TELEGRAM_TOOL_NOTIFICATIONS", raising=False)
    monkeypatch.delenv("TELEGRAM_ENABLED", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert telegram_tool_notifications_enabled() is False

    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "false")
    assert telegram_tool_notifications_enabled() is False

    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")
    assert telegram_tool_notifications_enabled() is True


@pytest.mark.asyncio
async def test_notify_skips_without_telegram_chat_in_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HTTP call when session state has no telegram_chat_id."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(state=MockState({"user_id": "web-user"}))
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_after_model_skips_when_no_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not send Telegram notification if the LLM response has no tool calls."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        response = LlmResponse(content=Content(parts=[Part.from_text(text="hello")]))
        await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_after_model_sends_intermediate_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send Telegram notification if the LLM response has text and tool calls."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        tool_call_part = Part(function_call=FunctionCall(name="test_tool", args={}))
        text_part = Part.from_text(text="I am doing a test")

        response = LlmResponse(content=Content(parts=[text_part, tool_call_part]))
        await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_awaited_once()
    assert "I am doing a test" in mock_client.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_notify_after_model_skips_thoughts_and_think_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not send thought parts or <think> tag contents as intermediate messages."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        tool_call_part = Part(function_call=FunctionCall(name="test_tool", args={}))

        # This part should be ignored because thought=True
        explicit_thought = Part(text="I am thinking...", thought=True)

        # This part has text with a <think> block that should be stripped
        think_tag_part = Part.from_text(
            text="<think>internal monologue</think>Actual message"
        )

        response = LlmResponse(
            content=Content(parts=[explicit_thought, think_tag_part, tool_call_part])
        )
        await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_awaited_once()
    sent_text = mock_client.send_message.await_args.kwargs["text"]
    assert "internal monologue" not in sent_text
    assert "I am thinking..." not in sent_text
    assert "Actual message" in sent_text


@pytest.mark.asyncio
async def test_notify_after_model_skips_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip Telegram notification if disabled."""
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "false")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        response = LlmResponse(
            content=Content(
                parts=[Part(function_call=FunctionCall(name="test", args={}))]
            )
        )
        await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_after_model_skips_no_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip Telegram notification if there is no text in response."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        tool_call_part = Part(function_call=FunctionCall(name="test_tool", args={}))

        response = LlmResponse(content=Content(parts=[tool_call_part]))
        await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_after_model_handles_api_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TelegramApiError from send_message is logged and swallowed."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        side_effect=TelegramApiError("bad", error_code=400),
    )

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        tool_call_part = Part(function_call=FunctionCall(name="test_tool", args={}))
        text_part = Part.from_text(text="I am doing a test")

        response = LlmResponse(content=Content(parts=[text_part, tool_call_part]))
        await callbacks_module.notify_telegram_after_model(ctx, response)

    assert "Telegram intermediate notification failed" in caplog.text


@pytest.mark.asyncio
async def test_notify_after_model_handles_missing_chat_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip if telegram_chat_id is missing or invalid."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({})

        tool_call_part = Part(function_call=FunctionCall(name="test_tool", args={}))
        text_part = Part.from_text(text="text")

        response = LlmResponse(content=Content(parts=[text_part, tool_call_part]))
        await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_not_called()

    # Invalid chat ID
    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "invalid"})
        await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_after_model_handles_missing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip if token is missing."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    callbacks_module._telegram_tool_notifications_enabled_impl.cache_clear()

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        tool_call_part = Part(function_call=FunctionCall(name="test", args={}))
        text_part = Part.from_text(text="text")
        response = LlmResponse(content=Content(parts=[text_part, tool_call_part]))
        await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_after_model_handles_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected error from send_message is logged."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        side_effect=RuntimeError("boom"),
    )

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        tool_call_part = Part(function_call=FunctionCall(name="test_tool", args={}))
        text_part = Part.from_text(text="I am doing a test")

        response = LlmResponse(content=Content(parts=[text_part, tool_call_part]))
        await callbacks_module.notify_telegram_after_model(ctx, response)

    assert "Unexpected error sending Telegram intermediate notification" in caplog.text


@pytest.mark.asyncio
async def test_notify_skips_invalid_chat_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed chat id does not call Telegram."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(
            state=MockState({"telegram_chat_id": "not-an-int"}),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_sends_to_telegram_with_chat_and_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration-style check: send_message receives parsed chat and thread ids."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(
            state=MockState(
                {
                    "telegram_chat_id": "4242",
                    "telegram_thread_id": "7",
                }
            ),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("search_memory")),
            {},
            cast(ToolContext, ctx),
        )

    mock_client.send_message.assert_awaited_once()
    kwargs = mock_client.send_message.await_args.kwargs
    assert kwargs["chat_id"] == 4242
    assert kwargs["message_thread_id"] == 7
    assert "Using tool" in kwargs["text"]


@pytest.mark.asyncio
async def test_notify_rate_limits_per_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two tool calls within the throttle window only produce one Telegram send."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    ctx = MockToolContext(
        state=MockState({"telegram_chat_id": "100"}),
    )

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("a")),
            {},
            cast(ToolContext, ctx),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("b")),
            {},
            cast(ToolContext, ctx),
        )

    assert len(mock_client.send_message.await_args_list) == 1


@pytest.mark.asyncio
async def test_notify_after_model_rate_limits_per_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two intermediate responses within the throttle window only produce one Telegram send."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    ctx = MagicMock(spec=CallbackContext)
    ctx.state = MockState({"telegram_chat_id": "100"})

    response = LlmResponse(
        content=Content(
            parts=[
                Part(function_call=FunctionCall(name="foo", args={})),
                Part.from_text(text="thinking..."),
            ]
        )
    )

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        await callbacks_module.notify_telegram_after_model(ctx, response)
        await callbacks_module.notify_telegram_after_model(ctx, response)

    assert len(mock_client.send_message.await_args_list) == 1


def test_evict_oldest_rate_limit_entries_noop() -> None:
    """Eviction helper returns early for non-positive count or empty map."""
    callbacks_module._TOOL_NOTIFY_LAST.clear()
    callbacks_module._evict_oldest_rate_limit_entries(0)
    callbacks_module._evict_oldest_rate_limit_entries(3)
    assert callbacks_module._TOOL_NOTIFY_LAST == {}

    callbacks_module._TOOL_NOTIFY_LAST["a"] = 1.0
    callbacks_module._evict_oldest_rate_limit_entries(0)
    assert "a" in callbacks_module._TOOL_NOTIFY_LAST


def test_rate_limit_evicts_oldest_when_map_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New chat keys trigger eviction when the rate-limit map is at capacity."""
    monkeypatch.setattr(callbacks_module, "_MAX_TOOL_NOTIFY_RATE_ENTRIES", 4)
    callbacks_module._TOOL_NOTIFY_LAST.clear()
    base = 1000.0
    for index in range(4):
        callbacks_module._TOOL_NOTIFY_LAST[str(index)] = base + index * 0.01
    assert len(callbacks_module._TOOL_NOTIFY_LAST) == 4

    assert callbacks_module._rate_limit_allows_notification("new", base + 100.0) is True
    assert "new" in callbacks_module._TOOL_NOTIFY_LAST
    assert len(callbacks_module._TOOL_NOTIFY_LAST) == 4


@pytest.mark.asyncio
async def test_notify_returns_early_when_feature_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Telegram traffic when TELEGRAM_TOOL_NOTIFICATIONS is off."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "false")
    callbacks_module._telegram_tool_notifications_enabled_impl.cache_clear()

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(
            state=MockState({"telegram_chat_id": "1"}),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_blank_thread_id_sends_without_thread_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank telegram_thread_id is treated as absent for send_message."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(
            state=MockState(
                {
                    "telegram_chat_id": "99",
                    "telegram_thread_id": "",
                }
            ),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )

    kwargs = mock_client.send_message.await_args.kwargs
    assert kwargs.get("message_thread_id") is None


@pytest.mark.asyncio
async def test_notify_send_message_telegram_api_error_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TelegramApiError from send_message is logged and swallowed."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        side_effect=TelegramApiError("bad", error_code=400),
    )

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(state=MockState({"telegram_chat_id": "1"}))
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )

    assert "Telegram tool notification failed" in caplog.text


@pytest.mark.asyncio
async def test_notify_send_message_unexpected_exception_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-Telegram errors from send_message are logged with stack trace."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(state=MockState({"telegram_chat_id": "1"}))
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )

    assert "Unexpected error sending Telegram tool notification" in caplog.text


@pytest.mark.asyncio
async def test_shared_telegram_client_reused_for_same_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second notify with the same bot token does not construct a new API client."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "one-token")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client) as ctor:
        ctx = MockToolContext(state=MockState({"telegram_chat_id": "5"}))
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("a")),
            {},
            cast(ToolContext, ctx),
        )
        await asyncio.sleep(0.4)
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("b")),
            {},
            cast(ToolContext, ctx),
        )

    assert ctor.call_count == 1


@pytest.mark.asyncio
async def test_shared_telegram_client_swaps_when_bot_token_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing TELEGRAM_BOT_TOKEN closes the previous shared client."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "first")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_first = MagicMock()
    mock_first.send_message = AsyncMock()
    mock_first.close = AsyncMock()
    mock_second = MagicMock()
    mock_second.send_message = AsyncMock()
    mock_second.close = AsyncMock()

    ctx = MockToolContext(state=MockState({"telegram_chat_id": "7"}))

    with patch(
        "blacki.callbacks.TelegramApiClient",
        side_effect=[mock_first, mock_second],
    ):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "second")
        callbacks_module._telegram_tool_notifications_enabled_impl.cache_clear()
        await asyncio.sleep(0.4)
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )

    mock_first.close.assert_awaited()
    mock_second.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_reset_schedules_async_close_when_loop_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test reset drops the shared client and schedules close under a running loop."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.close = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(state=MockState({"telegram_chat_id": "1"}))
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )
        reset_telegram_tool_notify_rate_limiter_for_tests()
        await asyncio.sleep(0)

    mock_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_reset_handles_close_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exception during client close in reset is logged, not raised."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")
    caplog.set_level(logging.DEBUG)

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.close = AsyncMock(side_effect=RuntimeError("close failed"))

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(state=MockState({"telegram_chat_id": "1"}))
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )
        reset_telegram_tool_notify_rate_limiter_for_tests()
        await asyncio.sleep(0)

    assert "Telegram notify client close failed" in caplog.text


def test_reset_handles_loop_create_task_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuntimeError from loop.create_task is caught silently."""

    monkeypatch.setattr(callbacks_module, "_shared_notify_client", MagicMock())
    monkeypatch.setattr(callbacks_module, "_shared_notify_token", "tok")

    mock_loop = MagicMock()
    mock_loop.create_task = MagicMock(side_effect=RuntimeError("loop closed"))

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        reset_telegram_tool_notify_rate_limiter_for_tests()

    assert callbacks_module._shared_notify_client is None
    assert callbacks_module._shared_notify_token is None


@pytest.mark.asyncio
async def test_notify_returns_early_when_bot_token_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Telegram traffic when TELEGRAM_BOT_TOKEN is empty string."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")
    callbacks_module._telegram_tool_notifications_enabled_impl.cache_clear()

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(state=MockState({"telegram_chat_id": "1"}))
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("t")),
            {},
            cast(ToolContext, ctx),
        )

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_after_model_handles_no_content_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip if llm_response has no content or parts."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        response1 = LlmResponse(content=None)
        await callbacks_module.notify_telegram_after_model(ctx, response1)

        response2 = LlmResponse(content=Content(parts=[]))
        await callbacks_module.notify_telegram_after_model(ctx, response2)

    mock_client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notify_after_model_handles_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip if split_long_message returns empty chunks."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MagicMock(spec=CallbackContext)
        ctx.state = MockState({"telegram_chat_id": "1"})

        tool_call_part = Part(function_call=FunctionCall(name="test_tool", args={}))
        text_part = Part.from_text(text="hello")

        response = LlmResponse(content=Content(parts=[tool_call_part, text_part]))

        # Mock split_long_message to return empty list
        with patch("blacki.telegram.streaming.split_long_message", return_value=[]):
            await callbacks_module.notify_telegram_after_model(ctx, response)

    mock_client.send_message.assert_not_called()
