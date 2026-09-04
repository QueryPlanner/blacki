"""Telegram live progress callbacks for status updates.

This module manages Telegram status delivery, rate limiting, and client
lifecycle for Telegram-backed sessions.
"""

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool

from ..security.tool_privacy import is_private_tool
from .api import TelegramApiClient, TelegramApiError
from .formatting import format_for_telegram
from .progress import describe_tool
from .streaming import is_message_not_modified_error
from .types import ParseMode

logger = logging.getLogger("blacki.callbacks")

# Per-chat monotonic timestamps for rate limiting (bounded; see _touch_rate_limit).

_INTERMEDIATE_NOTIFY_LAST: dict[str, float] = {}
_INTERMEDIATE_NOTIFY_MIN_INTERVAL_SEC = 0.35
_MAX_INTERMEDIATE_NOTIFY_RATE_ENTRIES = 8192
_INTERMEDIATE_NOTIFY_LOCK = asyncio.Lock()


# Live status session state keyed by (chat_id, thread_id, invocation_id).
@dataclass
class _LiveStatusSession:
    message_id: int | None = None
    started_at: float = field(default_factory=time.monotonic)
    last_sent_text: str = ""
    last_sent_time: float = 0.0
    pending_preamble: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_LIVE_STATUS_SESSIONS: dict[tuple[int, int | None, str | None], _LiveStatusSession] = {}
_MAX_LIVE_STATUS_SESSIONS = 8192
_LIVE_STATUS_LOCK = asyncio.Lock()


def telegram_live_tool_progress_enabled() -> bool:
    """Return whether Telegram is configured to receive live tool progress."""
    return os.environ.get("TELEGRAM_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    } and bool(os.environ.get("TELEGRAM_BOT_TOKEN", "").strip())


def _format_elapsed_duration(seconds: float) -> str:
    """Format a non-negative duration for the terminal live-status message."""
    total_seconds = max(1, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


async def _get_or_create_live_status_session(
    session_key: tuple[int, int | None, str | None],
) -> _LiveStatusSession:
    """Retrieve or create a live status session under the global dict lock."""
    async with _LIVE_STATUS_LOCK:
        if session_key not in _LIVE_STATUS_SESSIONS:
            if len(_LIVE_STATUS_SESSIONS) >= _MAX_LIVE_STATUS_SESSIONS:
                _evict_oldest_live_status_sessions(
                    _LIVE_STATUS_SESSIONS, _MAX_LIVE_STATUS_SESSIONS // 8
                )
            _LIVE_STATUS_SESSIONS[session_key] = _LiveStatusSession()
        return _LIVE_STATUS_SESSIONS[session_key]


# Reuse one HTTP client per bot token and serialize use with lifecycle changes.
_NOTIFY_CLIENT_LOCK = asyncio.Lock()
_shared_notify_client: TelegramApiClient | None = None
_shared_notify_token: str | None = None


def _evict_oldest_rate_limit_entries(storage: dict[str, float], count: int) -> None:
    if count <= 0 or not storage:
        return
    sorted_keys = sorted(storage, key=lambda key: storage[key])
    for key in sorted_keys[:count]:
        del storage[key]


def _evict_oldest_live_status_sessions(
    storage: dict[tuple[int, int | None, str | None], _LiveStatusSession],
    count: int,
) -> None:
    if count <= 0 or not storage:
        return
    sorted_keys = sorted(storage, key=lambda key: storage[key].last_sent_time)
    for key in sorted_keys[:count]:
        del storage[key]


async def _record_notification_timestamp(chat_key: str, now: float) -> None:
    """Record a successful send while keeping the rate-limit map bounded."""
    async with _INTERMEDIATE_NOTIFY_LOCK:
        map_is_full = (
            len(_INTERMEDIATE_NOTIFY_LAST) >= _MAX_INTERMEDIATE_NOTIFY_RATE_ENTRIES
            and chat_key not in _INTERMEDIATE_NOTIFY_LAST
        )
        if map_is_full:
            evict_count = max(1, _MAX_INTERMEDIATE_NOTIFY_RATE_ENTRIES // 8)
            _evict_oldest_rate_limit_entries(_INTERMEDIATE_NOTIFY_LAST, evict_count)
        _INTERMEDIATE_NOTIFY_LAST[chat_key] = now


async def _rate_limit_allows_notification(
    chat_key: str,
    now: float,
    *,
    storage: dict[str, float],
    min_interval: float,
    max_entries: int,
    lock: asyncio.Lock,
) -> bool:
    async with lock:
        last_sent = storage.get(chat_key, 0.0)
        if now - last_sent < min_interval:
            return False
        map_is_full = len(storage) >= max_entries and chat_key not in storage
        if map_is_full:
            evict_count = max(1, max_entries // 8)
            _evict_oldest_rate_limit_entries(storage, evict_count)
        storage[chat_key] = now
        return True


async def _safe_send_status_message(
    client: TelegramApiClient,
    chat_id: int | str,
    text: str,
    *,
    message_thread_id: int | None = None,
    parse_mode: ParseMode = ParseMode.MARKDOWN_V2,
) -> int | None:
    """Send initial live status message; return message_id on success, None on error."""
    try:
        sent_message = await client.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            message_thread_id=message_thread_id,
            disable_notification=True,
        )
        return sent_message.message_id
    except TelegramApiError as exc:
        logger.warning(
            "Telegram send_message failed for status update (chat_id=%s): %s",
            chat_id,
            exc,
        )
        return None
    except Exception:
        logger.exception("Unexpected error sending Telegram status message")
        return None


async def _safe_edit_message_text(
    client: TelegramApiClient,
    chat_id: int | str,
    message_id: int,
    text: str,
    *,
    parse_mode: ParseMode = ParseMode.MARKDOWN_V2,
) -> bool:
    """Edit status message text; return True if edited/unchanged, False on error."""
    try:
        await client.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode=parse_mode,
        )
        return True
    except TelegramApiError as exc:
        if is_message_not_modified_error(exc):
            return True
        logger.warning(
            "Telegram edit_message_text failed for status update "
            "(chat_id=%s, msg_id=%s): %s",
            chat_id,
            message_id,
            exc,
        )
        return False
    except Exception:
        logger.exception("Unexpected error editing Telegram status message")
        return False


def _schedule_shared_notify_client_close_for_tests() -> None:
    """Drop shared client refs; best-effort async close when a loop is running."""
    global _shared_notify_client, _shared_notify_token
    client = _shared_notify_client
    _shared_notify_client = None
    _shared_notify_token = None
    if client is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _close_wrapper() -> None:
        try:
            await client.close()
        except Exception:
            logger.debug(
                "Telegram notify client close failed after test reset",
                exc_info=True,
            )

    coro = _close_wrapper()
    try:
        loop.create_task(coro)
    except RuntimeError:
        coro.close()
        return


async def close_shared_notify_client() -> None:
    """Close the shared Telegram notify client (production shutdown path)."""
    global _shared_notify_client, _shared_notify_token
    async with _NOTIFY_CLIENT_LOCK:
        if _shared_notify_client is not None:
            try:
                await _shared_notify_client.close()
            except Exception:
                logger.exception("Error closing shared Telegram notify client")
        _shared_notify_client = None
        _shared_notify_token = None
        async with _INTERMEDIATE_NOTIFY_LOCK:
            _INTERMEDIATE_NOTIFY_LAST.clear()
        async with _LIVE_STATUS_LOCK:
            _LIVE_STATUS_SESSIONS.clear()


async def _ensure_shared_telegram_notify_client(token: str) -> TelegramApiClient:
    """Create or swap the shared client while its lock is held."""
    global _shared_notify_client, _shared_notify_token
    if _shared_notify_client is not None and _shared_notify_token == token:
        return _shared_notify_client
    if _shared_notify_client is not None:
        await _shared_notify_client.close()
    _shared_notify_client = TelegramApiClient(token)
    _shared_notify_token = token
    return _shared_notify_client


async def _shared_telegram_notify_client(token: str) -> TelegramApiClient:
    """Return a shared ``TelegramApiClient`` for this bot token (create or swap)."""
    async with _NOTIFY_CLIENT_LOCK:
        return await _ensure_shared_telegram_notify_client(token)


@asynccontextmanager
async def _shared_telegram_notify_client_lease(
    token: str,
) -> AsyncIterator[TelegramApiClient]:
    """Hold the client lock across one network operation."""
    async with _NOTIFY_CLIENT_LOCK:
        yield await _ensure_shared_telegram_notify_client(token)


async def reset_telegram_tool_notify_rate_limiter_for_tests() -> None:
    """Clear per-chat rate limit state, live status, and env cache (tests only)."""
    async with _INTERMEDIATE_NOTIFY_LOCK:
        _INTERMEDIATE_NOTIFY_LAST.clear()
    async with _LIVE_STATUS_LOCK:
        _LIVE_STATUS_SESSIONS.clear()
    _schedule_shared_notify_client_close_for_tests()


async def clear_telegram_progress_for_conversation(
    chat_id: int,
    message_thread_id: int | None,
) -> None:
    """Remove live progress state after a Telegram turn exits unexpectedly."""
    async with _LIVE_STATUS_LOCK:
        stale_keys = [
            key
            for key in _LIVE_STATUS_SESSIONS
            if key[0] == chat_id and key[1] == message_thread_id
        ]
        for key in stale_keys:
            _LIVE_STATUS_SESSIONS.pop(key, None)


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


async def notify_telegram_before_tool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> None:
    """Send or edit status before tool execution (Telegram sessions only)."""
    if not telegram_live_tool_progress_enabled():
        return None

    chat_id_raw = tool_context.state.get("telegram_chat_id")
    if not chat_id_raw:
        logger.debug("notify_telegram_before_tool: no telegram_chat_id in state")
        return None

    chat_id = _parse_optional_int(chat_id_raw)
    if chat_id is None:
        logger.warning("Invalid telegram_chat_id in state: %r", chat_id_raw)
        return None

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None

    thread_id = _parse_optional_int(tool_context.state.get("telegram_thread_id"))
    raw_invocation_id = getattr(tool_context, "invocation_id", None)
    invocation_id = str(raw_invocation_id) if raw_invocation_id is not None else None

    session_key = (chat_id, thread_id, invocation_id)
    session = await _get_or_create_live_status_session(session_key)

    async with session.lock:
        if session.pending_preamble:
            label = format_for_telegram(session.pending_preamble)
            session.pending_preamble = None
        else:
            label = describe_tool(tool.name, args, private=is_private_tool(tool))

        if session.message_id is None:
            async with _shared_telegram_notify_client_lease(token) as client:
                message_id = await _safe_send_status_message(
                    client,
                    chat_id=chat_id,
                    text=label,
                    message_thread_id=thread_id,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            if message_id is not None:
                session.message_id = message_id
                session.last_sent_text = label
                now = time.monotonic()
                session.last_sent_time = now
                await _record_notification_timestamp(str(chat_id), now)
        else:
            if label == session.last_sent_text:
                return None

            now = time.monotonic()
            chat_key = str(chat_id)
            allowed = await _rate_limit_allows_notification(
                chat_key,
                now,
                storage=_INTERMEDIATE_NOTIFY_LAST,
                min_interval=_INTERMEDIATE_NOTIFY_MIN_INTERVAL_SEC,
                max_entries=_MAX_INTERMEDIATE_NOTIFY_RATE_ENTRIES,
                lock=_INTERMEDIATE_NOTIFY_LOCK,
            )
            if not allowed:
                logger.debug(
                    "Coalescing Telegram tool status edit for chat_id=%s",
                    chat_id,
                )
                return None

            async with _shared_telegram_notify_client_lease(token) as client:
                success = await _safe_edit_message_text(
                    client,
                    chat_id=chat_id,
                    message_id=session.message_id,
                    text=label,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            if success:
                session.last_sent_text = label
                session.last_sent_time = now

    return None


async def notify_telegram_after_model(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> None:
    """Handle intermediate text responses or record model preamble for live status."""
    if not telegram_live_tool_progress_enabled():
        return None

    if not llm_response.content or not llm_response.content.parts:
        return None

    has_function_call = any(
        getattr(part, "function_call", None) for part in llm_response.content.parts
    )
    if not has_function_call:
        return None

    text_parts = [
        re.sub(r"<think>.*?</think>", "", part.text, flags=re.DOTALL).strip()
        for part in llm_response.content.parts
        if not getattr(part, "thought", False) and part.text
    ]
    text = "".join(p for p in text_parts if p).strip()

    if not text:
        return None

    chat_id_raw = callback_context.state.get("telegram_chat_id")
    if not chat_id_raw:
        return None

    chat_id = _parse_optional_int(chat_id_raw)
    if chat_id is None:
        return None

    thread_id = _parse_optional_int(callback_context.state.get("telegram_thread_id"))
    raw_invocation_id = getattr(callback_context, "invocation_id", None)
    invocation_id = str(raw_invocation_id) if raw_invocation_id is not None else None

    session_key = (chat_id, thread_id, invocation_id)
    session = await _get_or_create_live_status_session(session_key)
    async with session.lock:
        session.pending_preamble = text

    return None


async def notify_telegram_after_agent(
    callback_context: CallbackContext,
) -> None:
    """Collapse the live status message to done and clean up state at turn end."""
    if not telegram_live_tool_progress_enabled():
        return None

    chat_id_raw = callback_context.state.get("telegram_chat_id")
    if not chat_id_raw:
        return None

    chat_id = _parse_optional_int(chat_id_raw)
    if chat_id is None:
        return None

    thread_id = _parse_optional_int(callback_context.state.get("telegram_thread_id"))
    raw_invocation_id = getattr(callback_context, "invocation_id", None)
    invocation_id = str(raw_invocation_id) if raw_invocation_id is not None else None
    session_key = (chat_id, thread_id, invocation_id)

    session: _LiveStatusSession | None = None
    async with _LIVE_STATUS_LOCK:
        session = _LIVE_STATUS_SESSIONS.pop(session_key, None)

    if session is None or session.message_id is None:
        return None

    async with session.lock:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return None

        async with _shared_telegram_notify_client_lease(token) as client:
            elapsed = time.monotonic() - session.started_at
            await _safe_edit_message_text(
                client,
                chat_id=chat_id,
                message_id=session.message_id,
                text=f"✓ Worked for {_format_elapsed_duration(elapsed)}",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
    return None
