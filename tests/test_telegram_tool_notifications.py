# mypy: disable-error-code="no-untyped-def"
"""Tests for Telegram tool notification callback (issue #14)."""

import asyncio
import logging
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
    _format_tool_args,
    notify_telegram_before_tool,
    reset_telegram_tool_notify_rate_limiter_for_tests,
    telegram_tool_notifications_enabled,
)
from blacki.telegram.api import TelegramApiError


@pytest.fixture(autouse=True)
async def _clear_tool_notify_rate_limiter() -> AsyncIterator[None]:
    """Isolate rate limiter state between tests."""
    await reset_telegram_tool_notify_rate_limiter_for_tests()
    yield None
    await reset_telegram_tool_notify_rate_limiter_for_tests()


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
async def test_notify_after_model_skips_when_no_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip notification when LLM response has tool calls but no text."""
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
async def test_notify_sends_for_each_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each tool call gets its own Telegram notification (no rate limiting)."""
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

    # Both tool calls should send notifications
    assert len(mock_client.send_message.await_args_list) == 2


@pytest.mark.asyncio
async def test_notify_after_model_rate_limits_per_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate limit ensures only one Telegram send per throttle window."""
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
    storage: dict[str, float] = {}
    callbacks_module._evict_oldest_rate_limit_entries(storage, 0)
    callbacks_module._evict_oldest_rate_limit_entries(storage, 3)
    assert storage == {}

    storage["a"] = 1.0
    callbacks_module._evict_oldest_rate_limit_entries(storage, 0)
    assert "a" in storage


@pytest.mark.asyncio
async def test_rate_limit_evicts_oldest_when_map_full() -> None:
    """New chat keys trigger eviction when the rate-limit map is at capacity."""
    storage: dict[str, float] = {}
    base = 1000.0
    for index in range(4):
        storage[str(index)] = base + index * 0.01
    assert len(storage) == 4

    lock = asyncio.Lock()
    assert await callbacks_module._rate_limit_allows_notification(
        "new",
        base + 100.0,
        storage=storage,
        min_interval=0.35,
        max_entries=4,
        lock=lock,
    )
    assert "new" in storage
    assert len(storage) == 4


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
        await reset_telegram_tool_notify_rate_limiter_for_tests()
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
        await reset_telegram_tool_notify_rate_limiter_for_tests()
        await asyncio.sleep(0)

    assert "Telegram notify client close failed" in caplog.text


async def test_reset_handles_loop_create_task_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RuntimeError from loop.create_task is caught silently."""

    monkeypatch.setattr(callbacks_module, "_shared_notify_client", MagicMock())
    monkeypatch.setattr(callbacks_module, "_shared_notify_token", "tok")

    mock_loop = MagicMock()
    mock_loop.create_task = MagicMock(side_effect=RuntimeError("loop closed"))

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        await reset_telegram_tool_notify_rate_limiter_for_tests()

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


@pytest.mark.asyncio
async def test_reset_handles_get_running_loop_runtime_error() -> None:
    """RuntimeError from get_running_loop is caught silently before closing."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(callbacks_module, "_shared_notify_client", MagicMock())
    monkeypatch.setattr(callbacks_module, "_shared_notify_token", "tok")  # noqa: S105

    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
        await reset_telegram_tool_notify_rate_limiter_for_tests()

    assert callbacks_module._shared_notify_client is None
    assert callbacks_module._shared_notify_token is None


@pytest.mark.asyncio
async def test_close_shared_notify_client_happy_path() -> None:
    """close_shared_notify_client closes client and clears state."""
    from blacki.callbacks import close_shared_notify_client

    mock_client = MagicMock()
    mock_client.close = AsyncMock()

    callbacks_module._shared_notify_client = mock_client
    callbacks_module._shared_notify_token = "tok"  # noqa: S105

    await close_shared_notify_client()

    mock_client.close.assert_awaited_once()
    assert callbacks_module._shared_notify_client is None
    assert callbacks_module._shared_notify_token is None  # type: ignore[unreachable]


@pytest.mark.asyncio
async def test_close_shared_notify_client_handles_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """close_shared_notify_client logs and clears on close error."""
    from blacki.callbacks import close_shared_notify_client

    mock_client = MagicMock()
    mock_client.close = AsyncMock(side_effect=RuntimeError("close failed"))

    callbacks_module._shared_notify_client = mock_client
    callbacks_module._shared_notify_token = "tok"  # noqa: S105

    await close_shared_notify_client()

    assert "Error closing shared Telegram notify client" in caplog.text
    assert callbacks_module._shared_notify_client is None
    assert callbacks_module._shared_notify_token is None  # type: ignore[unreachable]


@pytest.mark.asyncio
async def test_close_shared_notify_client_when_none() -> None:
    """close_shared_notify_client does nothing when client is already None."""
    from blacki.callbacks import close_shared_notify_client

    callbacks_module._shared_notify_client = None
    callbacks_module._shared_notify_token = None

    await close_shared_notify_client()

    assert callbacks_module._shared_notify_client is None
    assert callbacks_module._shared_notify_token is None


def test_format_tool_args_empty_args() -> None:
    """Empty args dict returns empty string."""
    assert _format_tool_args({}) == ""


def test_format_tool_args_single_arg() -> None:
    """Single arg formatted as key=value pair."""
    result = _format_tool_args({"query": "hello"})
    assert "query" in result
    assert "hello" in result


def test_format_tool_args_multiple_args() -> None:
    """Multiple args are comma-separated."""
    result = _format_tool_args({"query": "hello", "count": "5"})
    assert "query" in result
    assert "hello" in result
    assert "count" in result
    assert "5" in result
    assert ", " in result


def test_format_tool_args_long_value_truncated() -> None:
    """Long values are truncated with '...'."""
    long_value = "a" * 120
    result = _format_tool_args({"text": long_value})
    assert "..." in result
    assert "aaaaaa" in result


def test_format_tool_args_overall_truncated() -> None:
    """Very many args result in an overall truncated string."""
    many_args = {f"k{i}": f"v{i}" for i in range(50)}
    result = _format_tool_args(many_args)
    assert result.endswith("...")


def test_format_tool_args_special_chars_escaped() -> None:
    """Special Markdown characters are escaped, including = separator."""
    result = _format_tool_args({"file_path": "/home/user_1/file.txt"})
    assert r"\=" in result
    assert "file\\_path" in result or "/home" in result


@pytest.mark.asyncio
async def test_notify_sends_args_in_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telegram notification includes tool arguments in the message text."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(
            state=MockState({"telegram_chat_id": "4242"}),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("search_memory")),
            {"query": "capital of France", "limit": 5},
            cast(ToolContext, ctx),
        )

    kwargs = mock_client.send_message.await_args.kwargs
    assert "Using tool" in kwargs["text"]
    assert "query" in kwargs["text"]
    assert "capital of France" in kwargs["text"]
    assert "limit" in kwargs["text"]
    assert "5" in kwargs["text"]


@pytest.mark.asyncio
async def test_notify_redacts_private_tts_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional tool notices may name TTS but never include speech contents."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "true")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        ctx = MockToolContext(state=MockState({"telegram_chat_id": "4242"}))
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("send_text_to_speech")),
            {"text": "private spoken content"},
            cast(ToolContext, ctx),
        )

    sent_text = mock_client.send_message.await_args.kwargs["text"]
    assert r"send\_text\_to\_speech" in sent_text
    assert "private spoken content" not in sent_text
    assert r"\=" not in sent_text


def test_telegram_config_tool_progress_mode_resolution() -> None:
    """TelegramConfig correctly resolves tool progress modes."""
    from blacki.telegram import TelegramConfig

    # Disabled Telegram -> always off
    cfg = TelegramConfig.model_validate({"TELEGRAM_ENABLED": "false"})
    assert cfg.tool_progress_mode() == "off"
    assert cfg.tool_notifications_active() is False

    # Enabled with token and explicit mode
    cfg = TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_TOOL_PROGRESS_MODE": "live",
        }
    )
    assert cfg.tool_progress_mode() == "live"
    assert cfg.tool_notifications_active() is True

    cfg = TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_TOOL_PROGRESS_MODE": "messages",
        }
    )
    assert cfg.tool_progress_mode() == "messages"
    assert cfg.tool_notifications_active() is True

    cfg = TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_TOOL_PROGRESS_MODE": "off",
            "TELEGRAM_TOOL_NOTIFICATIONS": "true",
        }
    )
    assert cfg.tool_progress_mode() == "off"
    assert cfg.tool_notifications_active() is False

    # Back-compat: TELEGRAM_TOOL_NOTIFICATIONS=true without TELEGRAM_TOOL_PROGRESS_MODE
    cfg = TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_TOOL_NOTIFICATIONS": "true",
        }
    )
    assert cfg.tool_progress_mode() == "messages"
    assert cfg.tool_notifications_active() is True

    # Default: both unset
    cfg = TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": "true",
            "TELEGRAM_BOT_TOKEN": "tok",
        }
    )
    assert cfg.tool_progress_mode() == "off"
    assert cfg.tool_notifications_active() is False


@pytest.mark.asyncio
async def test_live_progress_single_send_and_subsequent_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly ONE sendMessage per turn; subsequent tool calls issue editMessageText."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        return_value=MagicMock(message_id=101),
    )
    mock_client.edit_message_text = AsyncMock(
        return_value=MagicMock(message_id=101),
    )

    ctx = MockToolContext(
        state=MockState({"telegram_chat_id": "42"}),
    )
    ctx.invocation_id = "inv-1"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        # Tool 1 -> sends message
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "first query"},
            cast(ToolContext, ctx),
        )
        assert mock_client.send_message.call_count == 1
        assert mock_client.edit_message_text.call_count == 0
        assert (
            "Searching the web for *first query*"
            in mock_client.send_message.await_args.kwargs["text"]
        )

        # Tool 2 -> edits message
        await asyncio.sleep(0.4)
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("log_meal")),
            {"description": "lunch salad"},
            cast(ToolContext, ctx),
        )
        assert mock_client.send_message.call_count == 1
        assert mock_client.edit_message_text.call_count == 1
        assert (
            "Logging meal: *lunch salad*"
            in mock_client.edit_message_text.await_args.kwargs["text"]
        )

        # Tool 3 -> edits message
        await asyncio.sleep(0.4)
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_current_weather")),
            {"location": "Tokyo"},
            cast(ToolContext, ctx),
        )
        assert mock_client.send_message.call_count == 1
        assert mock_client.edit_message_text.call_count == 2
        assert (
            "Checking current weather for *Tokyo*"
            in mock_client.edit_message_text.await_args.kwargs["text"]
        )


@pytest.mark.asyncio
async def test_live_progress_model_preamble_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model preamble overrides hardcoded label; fallback to label when empty."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        return_value=MagicMock(message_id=202),
    )
    mock_client.edit_message_text = AsyncMock(
        return_value=MagicMock(message_id=202),
    )

    cb_ctx = MagicMock(spec=CallbackContext)
    cb_ctx.state = MockState({"telegram_chat_id": "42"})
    cb_ctx.invocation_id = "inv-turn-1"

    tool_ctx = MockToolContext(
        state=MockState({"telegram_chat_id": "42"}),
    )
    tool_ctx.invocation_id = "inv-turn-1"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        # 1. Model emits preamble before tool call
        response_with_preamble = LlmResponse(
            content=Content(
                parts=[
                    Part.from_text(text="Looking up your fitness stats..."),
                    Part(
                        function_call=FunctionCall(name="get_health_summary", args={})
                    ),
                ]
            )
        )
        await callbacks_module.notify_telegram_after_model(
            cb_ctx, response_with_preamble
        )

        # Tool 1 executes -> should use preamble override instead of hardcoded label
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_health_summary")),
            {},
            cast(ToolContext, tool_ctx),
        )
        assert mock_client.send_message.call_count == 1
        sent_text = mock_client.send_message.await_args.kwargs["text"]
        assert "Looking up your fitness stats" in sent_text
        assert "Fetching health summary" not in sent_text

        # 2. Next step: model emits tool call with NO preamble text
        await asyncio.sleep(0.4)
        response_no_preamble = LlmResponse(
            content=Content(
                parts=[
                    Part(
                        function_call=FunctionCall(
                            name="brave_search", args={"query": "stretches"}
                        )
                    ),
                ]
            )
        )
        await callbacks_module.notify_telegram_after_model(cb_ctx, response_no_preamble)

        # Tool 2 executes -> falls back to hardcoded label
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "stretches"},
            cast(ToolContext, tool_ctx),
        )
        assert mock_client.edit_message_text.call_count == 1
        edit_text = mock_client.edit_message_text.await_args.kwargs["text"]
        assert "Searching the web for *stretches*" in edit_text


@pytest.mark.asyncio
async def test_live_progress_rate_limit_coalescing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rapid tool updates collapse and drop intermediate edits (last-write-wins)."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        return_value=MagicMock(message_id=303),
    )
    mock_client.edit_message_text = AsyncMock(
        return_value=MagicMock(message_id=303),
    )

    ctx = MockToolContext(
        state=MockState({"telegram_chat_id": "42"}),
    )
    ctx.invocation_id = "inv-rapid"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        # Tool 1 -> initial send
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "step 1"},
            cast(ToolContext, ctx),
        )
        assert mock_client.send_message.call_count == 1

        # Tool 2 -> immediate call (no sleep), should be coalesced / dropped
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "step 2"},
            cast(ToolContext, ctx),
        )
        assert mock_client.edit_message_text.call_count == 0

        # Wait for throttle interval to elapse
        await asyncio.sleep(0.4)

        # Tool 3 -> should succeed with latest tool's label
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "step 3"},
            cast(ToolContext, ctx),
        )
        assert mock_client.edit_message_text.call_count == 1
        assert (
            "Searching the web for *step 3*"
            in mock_client.edit_message_text.await_args.kwargs["text"]
        )


@pytest.mark.asyncio
async def test_live_progress_not_modified_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'message is not modified' error from editMessageText is swallowed."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        return_value=MagicMock(message_id=404),
    )
    mock_client.edit_message_text = AsyncMock(
        side_effect=TelegramApiError(
            "Bad Request: message is not modified", error_code=400
        ),
    )

    ctx = MockToolContext(
        state=MockState({"telegram_chat_id": "42"}),
    )
    ctx.invocation_id = "inv-mod"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "q1"},
            cast(ToolContext, ctx),
        )
        await asyncio.sleep(0.4)
        # Should not raise
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "q2"},
            cast(ToolContext, ctx),
        )


@pytest.mark.asyncio
async def test_live_progress_telegram_exception_never_propagates(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Any unexpected Telegram exception is logged and swallowed."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        side_effect=RuntimeError("connection terminated"),
    )

    ctx = MockToolContext(
        state=MockState({"telegram_chat_id": "42"}),
    )
    ctx.invocation_id = "inv-err"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        # Should not raise exception
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "q"},
            cast(ToolContext, ctx),
        )

    assert "Unexpected error sending Telegram status message" in caplog.text


@pytest.mark.asyncio
async def test_live_progress_turn_end_collapses_and_evicts_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn end collapses the message to done state and evicts state dict entry."""
    from blacki.callbacks import notify_telegram_after_agent

    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(
        return_value=MagicMock(message_id=505),
    )
    mock_client.edit_message_text = AsyncMock(
        return_value=MagicMock(message_id=505),
    )

    tool_ctx = MockToolContext(
        state=MockState({"telegram_chat_id": "42"}),
    )
    tool_ctx.invocation_id = "inv-turn-done"

    cb_ctx = MagicMock(spec=CallbackContext)
    cb_ctx.state = MockState({"telegram_chat_id": "42"})
    cb_ctx.invocation_id = "inv-turn-done"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        # Tool call
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "done test"},
            cast(ToolContext, tool_ctx),
        )
        assert mock_client.send_message.call_count == 1
        assert (42, None, "inv-turn-done") in callbacks_module._LIVE_STATUS_SESSIONS

        # Turn end
        await notify_telegram_after_agent(cb_ctx)
        assert mock_client.edit_message_text.call_count == 1
        kwargs = mock_client.edit_message_text.await_args.kwargs
        assert kwargs["message_id"] == 505
        assert kwargs["text"] == "✓ Done"

        # Dict state must be evicted
        assert (42, None, "inv-turn-done") not in callbacks_module._LIVE_STATUS_SESSIONS

        # Subsequent call does nothing
        await notify_telegram_after_agent(cb_ctx)
        assert mock_client.edit_message_text.call_count == 1


@pytest.mark.asyncio
async def test_live_progress_concurrent_chats_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent chats do not share or clobber each other's status message."""
    from blacki.callbacks import notify_telegram_after_agent

    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    # Return message_id based on chat_id
    async def mock_send(chat_id, **kwargs):
        return MagicMock(message_id=chat_id * 10)

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(side_effect=mock_send)
    mock_client.edit_message_text = AsyncMock()

    ctx_a = MockToolContext(state=MockState({"telegram_chat_id": "100"}))
    ctx_a.invocation_id = "inv-a"

    ctx_b = MockToolContext(state=MockState({"telegram_chat_id": "200"}))
    ctx_b.invocation_id = "inv-b"

    cb_ctx_a = MagicMock(spec=CallbackContext)
    cb_ctx_a.state = MockState({"telegram_chat_id": "100"})
    cb_ctx_a.invocation_id = "inv-a"

    cb_ctx_b = MagicMock(spec=CallbackContext)
    cb_ctx_b.state = MockState({"telegram_chat_id": "200"})
    cb_ctx_b.invocation_id = "inv-b"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        # Chat A Tool 1
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "chat a query"},
            cast(ToolContext, ctx_a),
        )
        # Chat B Tool 1
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("log_meal")),
            {"description": "chat b meal"},
            cast(ToolContext, ctx_b),
        )

        assert mock_client.send_message.call_count == 2

        # Chat A Tool 2
        await asyncio.sleep(0.4)
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_current_weather")),
            {"location": "London"},
            cast(ToolContext, ctx_a),
        )
        # Chat B Tool 2
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_current_weather")),
            {"location": "Paris"},
            cast(ToolContext, ctx_b),
        )

        # Edits target respective message IDs
        edit_calls = mock_client.edit_message_text.await_args_list
        assert len(edit_calls) == 2
        assert edit_calls[0].kwargs["chat_id"] == 100
        assert edit_calls[0].kwargs["message_id"] == 1000
        assert edit_calls[1].kwargs["chat_id"] == 200
        assert edit_calls[1].kwargs["message_id"] == 2000

        # Turn end for Chat A
        await notify_telegram_after_agent(cb_ctx_a)
        assert (100, None, "inv-a") not in callbacks_module._LIVE_STATUS_SESSIONS
        assert (200, None, "inv-b") in callbacks_module._LIVE_STATUS_SESSIONS

        # Turn end for Chat B
        await notify_telegram_after_agent(cb_ctx_b)
        assert (200, None, "inv-b") not in callbacks_module._LIVE_STATUS_SESSIONS


@pytest.mark.asyncio
async def test_progress_mode_off_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TELEGRAM_TOOL_PROGRESS_MODE=off sends no Telegram messages."""
    from blacki.callbacks import notify_telegram_after_agent

    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "off")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.edit_message_text = AsyncMock()

    ctx = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    ctx.invocation_id = "inv-off"

    cb_ctx = MagicMock(spec=CallbackContext)
    cb_ctx.state = MockState({"telegram_chat_id": "42"})
    cb_ctx.invocation_id = "inv-off"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "off test"},
            cast(ToolContext, ctx),
        )
        response = LlmResponse(
            content=Content(
                parts=[
                    Part.from_text(text="Thinking"),
                    Part(function_call=FunctionCall(name="foo", args={})),
                ]
            )
        )
        await callbacks_module.notify_telegram_after_model(cb_ctx, response)
        await notify_telegram_after_agent(cb_ctx)

    mock_client.send_message.assert_not_called()
    mock_client.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_progress_mode_messages_preserves_per_tool_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TELEGRAM_TOOL_PROGRESS_MODE=messages preserves per-tool messages."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "messages")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.edit_message_text = AsyncMock()

    ctx = MockToolContext(state=MockState({"telegram_chat_id": "42"}))

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "q1"},
            cast(ToolContext, ctx),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("log_meal")),
            {"description": "d1"},
            cast(ToolContext, ctx),
        )

    # In messages mode, each tool call sends a separate message
    assert mock_client.send_message.call_count == 2
    assert mock_client.edit_message_text.call_count == 0
    assert "Using tool" in mock_client.send_message.await_args_list[0].kwargs["text"]
    assert "Using tool" in mock_client.send_message.await_args_list[1].kwargs["text"]


@pytest.mark.asyncio
async def test_live_progress_slow_network_does_not_block_other_chats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow network I/O in one chat does not hold global lock or block others."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    chat_b_finished = False

    async def mock_send(chat_id, **kwargs):
        nonlocal chat_b_finished
        if chat_id == 100:
            # Chat A is slow (simulating slow network / 429 backoff)
            await asyncio.sleep(0.15)
            return MagicMock(message_id=100)
        elif chat_id == 200:
            # Chat B is fast
            chat_b_finished = True
            return MagicMock(message_id=200)
        return MagicMock(message_id=1)

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(side_effect=mock_send)
    mock_client.edit_message_text = AsyncMock()

    ctx_a = MockToolContext(state=MockState({"telegram_chat_id": "100"}))
    ctx_a.invocation_id = "inv-a"

    ctx_b = MockToolContext(state=MockState({"telegram_chat_id": "200"}))
    ctx_b.invocation_id = "inv-b"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        # Start Chat A in background task
        task_a = asyncio.create_task(
            notify_telegram_before_tool(
                cast(BaseTool, MockBaseTool("brave_search")),
                {"query": "slow query"},
                cast(ToolContext, ctx_a),
            )
        )

        # Brief pause to ensure task_a entered its send_message
        await asyncio.sleep(0.02)

        # Chat B should complete immediately without waiting for Chat A
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "fast query"},
            cast(ToolContext, ctx_b),
        )

        assert chat_b_finished is True
        assert not task_a.done()

        await task_a


@pytest.mark.asyncio
async def test_reset_hooks_clear_live_status_and_close_shared_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset rate limiter helper clears all state and closes client."""
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-reset")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(return_value=MagicMock(message_id=999))
    mock_client.close = AsyncMock()

    ctx = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    ctx.invocation_id = "inv-reset-test"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "search query"},
            cast(ToolContext, ctx),
        )

        assert len(callbacks_module._LIVE_STATUS_SESSIONS) == 1
        assert callbacks_module._shared_notify_client is not None

        await callbacks_module.reset_telegram_tool_notify_rate_limiter_for_tests()
        await asyncio.sleep(0)

        assert len(callbacks_module._LIVE_STATUS_SESSIONS) == 0
        mock_client.close.assert_awaited_once()
        assert callbacks_module._shared_notify_client is None


@pytest.mark.asyncio
async def test_live_progress_turn_end_cleans_up_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """notify_telegram_after_agent cleanly evicts session from _LIVE_STATUS_SESSIONS."""
    from blacki.callbacks import notify_telegram_after_agent

    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-clean")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "live")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(return_value=MagicMock(message_id=888))
    mock_client.edit_message_text = AsyncMock(return_value=MagicMock(message_id=888))

    ctx = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    ctx.invocation_id = "inv-clean-turn"

    cb_ctx = MagicMock(spec=CallbackContext)
    cb_ctx.state = MockState({"telegram_chat_id": "42"})
    cb_ctx.invocation_id = "inv-clean-turn"

    with patch("blacki.callbacks.TelegramApiClient", return_value=mock_client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "q"},
            cast(ToolContext, ctx),
        )
        assert len(callbacks_module._LIVE_STATUS_SESSIONS) == 1

        await notify_telegram_after_agent(cb_ctx)
        assert len(callbacks_module._LIVE_STATUS_SESSIONS) == 0


@pytest.mark.asyncio
async def test_unconfigured_telegram_env_is_inert_and_constructs_no_client() -> None:
    """With no Telegram env configured, notifications construct no client."""
    from types import SimpleNamespace

    assert not callbacks_module.telegram_tool_notifications_enabled()
    assert callbacks_module.telegram_tool_progress_mode() == "off"

    with patch("blacki.callbacks.TelegramApiClient") as mock_client_cls:
        ctx = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
        ctx.invocation_id = "inv-unconfigured"

        # notify_telegram_before_tool
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "test query"},
            cast(ToolContext, ctx),
        )

        # notify_telegram_after_model
        cb_ctx = MagicMock(spec=CallbackContext)
        cb_ctx.state = MockState({"telegram_chat_id": "42"})
        cb_ctx.invocation_id = "inv-unconfigured"
        part = SimpleNamespace(
            text="Hello",
            thought=False,
            function_call=SimpleNamespace(name="brave_search"),
        )
        llm_response = MagicMock(spec=LlmResponse)
        llm_response.content = SimpleNamespace(parts=[part])
        await callbacks_module.notify_telegram_after_model(cb_ctx, llm_response)

        # notify_telegram_after_agent
        await callbacks_module.notify_telegram_after_agent(cb_ctx)

        mock_client_cls.assert_not_called()
        assert len(callbacks_module._LIVE_STATUS_SESSIONS) == 0
        assert callbacks_module._shared_notify_client is None
