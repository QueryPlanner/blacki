"""Blacki's two Mini App tools: read and write pages with the Blacki UI language.

Load the ``miniapp`` skill for the language and catalog before writing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from google.adk.tools import ToolContext

from blacki.miniapp.config import MiniAppConfig
from blacki.miniapp.service import get_miniapp_service
from blacki.miniapp.sources import MiniAppUser
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.types import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo

logger = logging.getLogger(__name__)


def _user(tool_context: ToolContext) -> MiniAppUser | None:
    state = tool_context.state
    if state.get("telegram_chat_type") != "private":
        return None
    try:
        return MiniAppUser(int(str(state.get("telegram_chat_id"))))
    except (TypeError, ValueError):
        return None


async def miniapp_read(code: str, tool_context: ToolContext) -> dict[str, Any]:
    """Look at the user's Mini App before changing it.

    code is one read command: "help", "pages", "page <id>", "history <id>",
    "version <id> <n>", "cards", "card <id>", "sources", "catalog", or a data
    peek such as Query("health.sleep", {days: 7}) that returns up to 20 rows.
    Built-in pages (today, health, food, training, files) can be read but
    never edited. Load the miniapp skill for the language before writing.
    """
    user = _user(tool_context)
    if user is None:
        return {
            "status": "error",
            "error": "The Mini App is only available in a private chat",
        }
    if len(code) > 2000:
        return {"status": "error", "error": "read command is too long"}
    return await get_miniapp_service().tool_read(user, code)


async def miniapp_write(
    code: str, tool_context: ToolContext, base_version: int | None = None
) -> dict[str, Any]:
    """Create or change Mini App pages and chat cards with Blacki UI code.

    code starts with a command line, followed by statements in the Blacki UI
    language:
      page <id> "Title"         create a page, or edit it (only changed lines needed)
      page <id> "Title" replace  replace the whole page
      remove <name> [<name>...]  inside a page block: delete lines by name
      note "why"                 inside a page block: a note for the version history
      card "Title"               a one-off card; the user gets an Open in app button
      delete page <id> | revert <id> to <n> | move <id> to <position>
    Editing an existing page needs base_version from miniapp_read("page <id>").
    All blocks in one call save together or not at all: if any block fails,
    nothing is saved. A page can appear in one block per call.
    The server validates everything and returns line-numbered errors to fix.
    Saved pages are versioned; the user can undo. The theme can't be changed.
    """
    user = _user(tool_context)
    if user is None:
        return {
            "status": "error",
            "error": "The Mini App is only available in a private chat",
        }
    if len(code) > 20000:
        return {"status": "error", "error": "code is too long"}
    result = await get_miniapp_service().tool_write(user, code, base_version)
    for item in result.get("results", []):
        if item.get("status") == "ok" and item.get("card_id"):
            sent = await _send_card_button(
                user, item["card_id"], item.get("title", ""), tool_context
            )
            item["delivered"] = sent
            item["note"] = (
                "The user got a message with an Open in app button; it may arrive "
                "before your reply."
                if sent
                else "The card was saved but no button was sent (no public Mini App URL)."  # noqa: E501
            )
    return result


async def _send_card_button(
    user: MiniAppUser, card_id: str, title: str, tool_context: ToolContext
) -> bool:
    try:
        config = MiniAppConfig.from_environment()
    except ValueError:
        return False
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not config.launch_url or not token:
        return False
    thread = tool_context.state.get("telegram_thread_id")
    try:
        async with TelegramApiClient(token) as api:
            await api.send_message(
                chat_id=user.telegram_user_id,
                text=f"I made a card: {title}" if title else "I made a card for you.",
                message_thread_id=int(thread) if thread else None,
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="Open in app",
                                callback_data=None,
                                web_app=WebAppInfo(
                                    url=f"{config.launch_url}?card={card_id}"
                                ),
                            )
                        ]
                    ]
                ),
            )
        return True
    except Exception:
        logger.exception("Could not send the Mini App card button")
        return False
