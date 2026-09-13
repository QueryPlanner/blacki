"""Tests for the model-blind Telegram takeover tool."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from blacki.browser_takeover import BrowserTakeoverError, BrowserTakeoverLease
from blacki.tools.browser_takeover import _send_takeover_link, start_browser_takeover


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        state={
            "telegram_chat_id": "42",
            "telegram_thread_id": "9",
            "telegram_chat_type": "private",
            "temp:telegram_sender_user_id": "42",
        }
    )


async def test_tool_returns_status_without_leaking_link() -> None:
    lease = BrowserTakeoverLease(
        session_id="session",
        takeover_url="https://private.test/browser#secret-link-token",
        expires_in_seconds=300,
    )
    service = SimpleNamespace(
        create=AsyncMock(return_value=lease),
        wait=AsyncMock(return_value=True),
        close=AsyncMock(),
    )
    with (
        patch(
            "blacki.tools.browser_takeover.get_browser_takeover_service",
            return_value=service,
        ),
        patch(
            "blacki.tools.browser_takeover._send_takeover_link",
            new=AsyncMock(),
        ) as send,
    ):
        result = await start_browser_takeover(
            "https://accounts.example.test/login",
            _context(),
        )

    assert result["status"] == "success"
    assert "secret-link-token" not in repr(result)
    send.assert_awaited_once_with(lease.takeover_url, 300, _context().state)
    service.close.assert_awaited_once_with(lease)


async def test_tool_handles_unavailable_create_expiry_and_delivery_failure() -> None:
    with patch(
        "blacki.tools.browser_takeover.get_browser_takeover_service",
        return_value=None,
    ):
        assert (await start_browser_takeover("https://example.test", _context()))[
            "status"
        ] == "unavailable"

    create_failure = SimpleNamespace(
        create=AsyncMock(side_effect=BrowserTakeoverError("safe failure"))
    )
    with patch(
        "blacki.tools.browser_takeover.get_browser_takeover_service",
        return_value=create_failure,
    ):
        result = await start_browser_takeover("https://example.test", _context())
    assert result == {"status": "error", "message": "safe failure"}

    lease = BrowserTakeoverLease("session", "https://private.test/#token", 300)
    expired = SimpleNamespace(
        create=AsyncMock(return_value=lease),
        wait=AsyncMock(return_value=False),
        close=AsyncMock(),
    )
    with (
        patch(
            "blacki.tools.browser_takeover.get_browser_takeover_service",
            return_value=expired,
        ),
        patch(
            "blacki.tools.browser_takeover._send_takeover_link",
            new=AsyncMock(),
        ),
    ):
        result = await start_browser_takeover("https://example.test", _context())
    assert result["status"] == "expired"
    expired.close.assert_awaited_once_with(lease)

    delivery_failure = SimpleNamespace(
        create=AsyncMock(return_value=lease),
        wait=AsyncMock(),
        close=AsyncMock(),
    )
    with (
        patch(
            "blacki.tools.browser_takeover.get_browser_takeover_service",
            return_value=delivery_failure,
        ),
        patch(
            "blacki.tools.browser_takeover._send_takeover_link",
            new=AsyncMock(side_effect=RuntimeError("secret provider failure")),
        ),
    ):
        result = await start_browser_takeover("https://example.test", _context())
    assert result["status"] == "error"
    assert "secret provider failure" not in repr(result)
    delivery_failure.close.assert_awaited_once_with(lease)


async def test_link_is_sent_directly_as_protected_telegram_content(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    api = AsyncMock()
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=api)
    context_manager.__aexit__ = AsyncMock(return_value=None)
    with patch(
        "blacki.tools.browser_takeover.TelegramApiClient",
        return_value=context_manager,
    ):
        await _send_takeover_link(
            "https://private.test/browser#token",
            300,
            _context().state,
        )

    kwargs = api.send_message.await_args.kwargs
    assert kwargs["chat_id"] == 42
    assert kwargs["message_thread_id"] == 9
    assert kwargs["protect_content"] is True
    assert "#token" not in kwargs["text"]
    assert kwargs["reply_markup"].inline_keyboard[0][0].url.endswith("#token")


async def test_link_delivery_requires_telegram_token(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    try:
        await _send_takeover_link("https://private.test/#token", 60, _context().state)
    except BrowserTakeoverError as exc:
        assert str(exc) == "Telegram is not configured"
    else:
        raise AssertionError("Expected missing Telegram configuration to fail")
