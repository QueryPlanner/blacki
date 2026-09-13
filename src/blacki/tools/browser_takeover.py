"""Private Telegram tool for handing Agent Browser control to the user."""

from __future__ import annotations

import os
from typing import Any

from google.adk.tools import ToolContext

from blacki.browser_takeover import BrowserTakeoverError, get_browser_takeover_service
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.types import InlineKeyboardButton, InlineKeyboardMarkup


async def start_browser_takeover(
    login_url: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Let the Telegram user privately complete login in the session browser.

    Use this only when a website requires a password, OTP, CAPTCHA, or other
    sensitive human input. The takeover link is sent directly through
    Telegram and is deliberately omitted from this tool's result.
    """
    service = get_browser_takeover_service()
    if service is None:
        return {
            "status": "unavailable",
            "message": "Private browser takeover is not configured.",
        }

    try:
        lease = await service.create(login_url=login_url, state=tool_context.state)
    except BrowserTakeoverError as exc:
        return {"status": "error", "message": str(exc)}

    try:
        await _send_takeover_link(
            lease.takeover_url,
            lease.expires_in_seconds,
            tool_context.state,
        )
        completed = await service.wait(lease)
        if not completed:
            return {
                "status": "expired",
                "message": "The private browser takeover expired before completion.",
            }
        return {
            "status": "success",
            "message": "The user returned control of the authenticated browser.",
        }
    except Exception:
        return {
            "status": "error",
            "message": "The private browser takeover could not be delivered.",
        }
    finally:
        await service.close(lease)


async def _send_takeover_link(
    takeover_url: str,
    expires_in_seconds: int,
    state: Any,
) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise BrowserTakeoverError("Telegram is not configured")
    chat_id = int(state["telegram_chat_id"])
    thread_value = state.get("telegram_thread_id")
    thread_id = int(thread_value) if thread_value else None
    minutes = max(1, expires_in_seconds // 60)
    text = (
        "Private browser control is ready. Open this link, complete the login, "
        f"then tap Done. It expires in {minutes} minutes."
    )
    async with TelegramApiClient(token) as api:
        await api.send_message(
            chat_id=chat_id,
            text=text,
            message_thread_id=thread_id,
            protect_content=True,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="Take over browser", url=takeover_url)]
                ]
            ),
        )
