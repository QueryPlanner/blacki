"""Read-only Google Health tools exposed to the authenticated Telegram agent."""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from blacki.container import get_container

from .service import summarize_stored_health


async def get_health_summary(
    tool_context: ToolContext,
    days: int = 7,
) -> dict[str, Any]:
    """Read a user's normalized Google Health summary.

    This tool is intentionally read-only. It never calls Google directly; the
    scheduled or explicit refresh path owns provider access and persistence.
    """
    if days < 1 or days > 14:
        return {"status": "error", "error": "days must be between 1 and 14"}

    user_id = getattr(tool_context, "user_id", None)
    if not isinstance(user_id, str) or not user_id:
        return {"status": "error", "error": "Missing Telegram user identity"}

    state = getattr(tool_context, "state", None)
    chat_type = state.get("telegram_chat_type") if state is not None else None
    if chat_type != "private":
        return {
            "status": "error",
            "error": (
                "Google Health summaries are available only in private Telegram chats"
            ),
        }

    try:
        storage = get_container().google_health_storage
    except RuntimeError:
        return {"status": "error", "error": "Health storage is not initialized"}
    return await summarize_stored_health(storage, user_id, days=days)
