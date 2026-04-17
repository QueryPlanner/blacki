"""Tests for Telegram tool notification callback (issue #14)."""

from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import MockBaseTool, MockState, MockToolContext
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool

from blacki.callbacks import (
    notify_telegram_before_tool,
    reset_telegram_tool_notify_rate_limiter_for_tests,
    telegram_tool_notifications_enabled,
)


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
            cast(BaseTool, MockBaseTool("example_tool")),
            {},
            cast(ToolContext, ctx),
        )
    mock_client.send_message.assert_not_called()


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
