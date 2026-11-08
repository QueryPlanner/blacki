from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
from google.adk.tools import ToolContext

import blacki.miniapp.service as svc_module
from blacki.container import AppContainer
from blacki.miniapp.service import MiniAppService
from blacki.miniapp.sources import MiniAppUser
from blacki.miniapp.store import MiniAppStore
from blacki.tools.miniapp import _send_card_button, _user, miniapp_read, miniapp_write


@pytest.fixture
async def service() -> AsyncGenerator[MiniAppService]:
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    lock = asyncio.Lock()
    container = AppContainer(conn, lock)
    store = MiniAppStore(conn, lock)
    await store.initialize()
    svc = MiniAppService(container=container, store=store)
    svc_module._service = svc
    yield svc
    svc_module._service = None
    await conn.close()


def _make_tool_context(state: dict[str, Any]) -> ToolContext:
    class DummyContext:
        def __init__(self, state_dict: dict[str, Any]) -> None:
            self.state = state_dict

    return DummyContext(state)  # type: ignore[return-value]


def test_user_helper() -> None:
    # 1. Not private chat
    ctx_group = _make_tool_context(
        {"telegram_chat_type": "group", "telegram_chat_id": 12345}
    )
    assert _user(ctx_group) is None

    # 2. Invalid chat id (None or non-numeric)
    ctx_invalid = _make_tool_context(
        {"telegram_chat_type": "private", "telegram_chat_id": "not_an_int"}
    )
    assert _user(ctx_invalid) is None

    # 3. Valid private chat
    ctx_ok = _make_tool_context(
        {"telegram_chat_type": "private", "telegram_chat_id": 12345}
    )
    user = _user(ctx_ok)
    assert user is not None
    assert user.telegram_user_id == 12345


@pytest.mark.asyncio
async def test_miniapp_read(service: MiniAppService) -> None:
    # 1. Non-private user
    ctx_group = _make_tool_context(
        {"telegram_chat_type": "group", "telegram_chat_id": 12345}
    )
    res_err = await miniapp_read("help", ctx_group)
    assert res_err["status"] == "error"
    assert "private chat" in res_err["error"]

    # 2. Code too long (>2000 chars)
    ctx_private = _make_tool_context(
        {"telegram_chat_type": "private", "telegram_chat_id": 12345}
    )
    res_long = await miniapp_read("x" * 2001, ctx_private)
    assert res_long["status"] == "error"
    assert "read command is too long" in res_long["error"]

    # 3. Valid read command
    res_ok = await miniapp_read("help", ctx_private)
    assert res_ok["status"] == "ok"
    assert "commands" in res_ok


@pytest.mark.asyncio
async def test_miniapp_write(service: MiniAppService) -> None:
    # 1. Non-private user
    ctx_group = _make_tool_context(
        {"telegram_chat_type": "group", "telegram_chat_id": 12345}
    )
    res_err = await miniapp_write("page custom", ctx_group)
    assert res_err["status"] == "error"
    assert "private chat" in res_err["error"]

    # 2. Code too long (>20000 chars)
    ctx_private = _make_tool_context(
        {"telegram_chat_type": "private", "telegram_chat_id": 12345}
    )
    res_long = await miniapp_write("x" * 20001, ctx_private)
    assert res_long["status"] == "error"
    assert "code is too long" in res_long["error"]

    # 3. Page write (no card delivery)
    res_page = await miniapp_write(
        'page custom "Custom"\nroot = Page("Custom", [])',
        ctx_private,
    )
    assert res_page["status"] == "ok"

    # 4. Card write without public launch URL
    with patch.dict(
        "os.environ",
        {"MINIAPP_PUBLIC_URL": ""},
        clear=False,
    ):
        res_card_nourl = await miniapp_write(
            'card "Card Title"\nroot = Page("Card Title", [])',
            ctx_private,
        )
        assert res_card_nourl["status"] == "ok"
        item = res_card_nourl["results"][0]
        assert item["delivered"] is False
        assert "no public Mini App URL" in item["note"]


@pytest.mark.asyncio
async def test_send_card_button(service: MiniAppService) -> None:
    user = MiniAppUser(12345)
    ctx = _make_tool_context(
        {
            "telegram_chat_type": "private",
            "telegram_chat_id": 12345,
            "telegram_thread_id": "99",
        }
    )

    # 1. Config raises ValueError (e.g. invalid port)
    with patch.dict("os.environ", {"MINIAPP_PORT": "invalid"}):
        assert not await _send_card_button(user, "c1", "Title", ctx)

    # 2. Valid config and launch URL but no bot token
    with patch.dict(
        "os.environ",
        {"MINIAPP_PUBLIC_URL": "https://example.com", "TELEGRAM_BOT_TOKEN": ""},
    ):
        assert not await _send_card_button(user, "c1", "Title", ctx)

    # 3. Valid config and token, TelegramApiClient succeeds
    with (
        patch.dict(
            "os.environ",
            {
                "MINIAPP_PUBLIC_URL": "https://example.com",
                "TELEGRAM_BOT_TOKEN": "123:ABC",
            },
        ),
        patch("blacki.tools.miniapp.TelegramApiClient") as mock_client_cls,
    ):
        mock_api = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_api
        res_sent = await _send_card_button(user, "c1", "Card Title", ctx)
        assert res_sent is True
        mock_api.send_message.assert_called_once()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 12345
        assert call_kwargs["message_thread_id"] == 99
        assert "Card Title" in call_kwargs["text"]

    # 4. Empty title branch
    with (
        patch.dict(
            "os.environ",
            {
                "MINIAPP_PUBLIC_URL": "https://example.com",
                "TELEGRAM_BOT_TOKEN": "123:ABC",
            },
        ),
        patch("blacki.tools.miniapp.TelegramApiClient") as mock_client_cls,
    ):
        mock_api = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_api
        ctx_no_thread = _make_tool_context(
            {"telegram_chat_type": "private", "telegram_chat_id": 12345}
        )
        res_empty_title = await _send_card_button(user, "c1", "", ctx_no_thread)
        assert res_empty_title is True
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert call_kwargs["message_thread_id"] is None
        assert call_kwargs["text"] == "I made a card for you."

    # 5. TelegramApiClient raises Exception
    with (
        patch.dict(
            "os.environ",
            {
                "MINIAPP_PUBLIC_URL": "https://example.com",
                "TELEGRAM_BOT_TOKEN": "123:ABC",
            },
        ),
        patch("blacki.tools.miniapp.TelegramApiClient") as mock_client_cls,
    ):
        mock_api = AsyncMock()
        mock_api.send_message.side_effect = RuntimeError("Network error")
        mock_client_cls.return_value.__aenter__.return_value = mock_api
        res_fail = await _send_card_button(user, "c1", "Card", ctx)
        assert res_fail is False

    # 6. miniapp_write with successful card delivery
    with (
        patch.dict(
            "os.environ",
            {
                "MINIAPP_PUBLIC_URL": "https://example.com",
                "TELEGRAM_BOT_TOKEN": "123:ABC",
            },
        ),
        patch("blacki.tools.miniapp.TelegramApiClient") as mock_client_cls,
    ):
        mock_api = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_api
        res_write_delivered = await miniapp_write(
            'card "Delivered Card"\nroot = Page("Delivered Card", [])',
            ctx,
        )
        assert res_write_delivered["status"] == "ok"
        item = res_write_delivered["results"][0]
        assert item["delivered"] is True
        assert "Open in app button" in item["note"]
