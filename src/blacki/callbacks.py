"""Agent lifecycle callback functions for monitoring.

This module provides callback functions that execute at various stages of the
agent lifecycle. These callbacks enable comprehensive logging and optional
Telegram tool notifications for Telegram-backed sessions.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool

from .privacy import is_private_tool, private_tool_privacy_enabled
from .telegram.api import TelegramApiClient, TelegramApiError
from .telegram.formatting import format_for_telegram
from .telegram.progress import describe_tool
from .telegram.streaming import is_message_not_modified_error
from .telegram.types import ParseMode

logger = logging.getLogger(__name__)

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


# Reuse one HTTP client per bot token (narrow lock only for swap / teardown).
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


async def _shared_telegram_notify_client(token: str) -> TelegramApiClient:
    """Return a shared ``TelegramApiClient`` for this bot token (create or swap)."""
    global _shared_notify_client, _shared_notify_token
    async with _NOTIFY_CLIENT_LOCK:
        if _shared_notify_client is not None and _shared_notify_token == token:
            return _shared_notify_client
        if _shared_notify_client is not None:
            await _shared_notify_client.close()
        _shared_notify_client = TelegramApiClient(token)
        _shared_notify_token = token
        return _shared_notify_client


async def reset_telegram_tool_notify_rate_limiter_for_tests() -> None:
    """Clear per-chat rate limit state, live status, and env cache (tests only)."""
    async with _INTERMEDIATE_NOTIFY_LOCK:
        _INTERMEDIATE_NOTIFY_LAST.clear()
    async with _LIVE_STATUS_LOCK:
        _LIVE_STATUS_SESSIONS.clear()
    _schedule_shared_notify_client_close_for_tests()


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
            client = await _shared_telegram_notify_client(token)
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
                async with _INTERMEDIATE_NOTIFY_LOCK:
                    _INTERMEDIATE_NOTIFY_LAST[str(chat_id)] = now
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

            client = await _shared_telegram_notify_client(token)
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

        client = await _shared_telegram_notify_client(token)
        elapsed = time.monotonic() - session.started_at
        await _safe_edit_message_text(
            client,
            chat_id=chat_id,
            message_id=session.message_id,
            text=f"✓ Worked for {_format_elapsed_duration(elapsed)}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    return None


class LoggingCallbacks:
    """Provides comprehensive logging callbacks for ADK agent lifecycle events.

    This class groups all agent lifecycle callback methods together and supports
    logger injection following the strategy pattern. All callbacks are
    non-intrusive and return None.

    Attributes:
        logger: Logger instance for recording agent lifecycle events.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize logging callbacks with optional logger.

        Args:
            logger: Optional logger instance. If not provided, creates one
                   using the module name.
        """
        if logger is None:
            logger = logging.getLogger(self.__class__.__module__)
        self.logger = logger

    def before_agent(self, callback_context: CallbackContext) -> None:
        """Callback executed before agent processing begins.

        Args:
            callback_context (CallbackContext): Context containing agent name,
                invocation ID, state, and user content.
        """
        self.logger.info(
            f"*** Starting agent '{callback_context.agent_name}' "
            f"with invocation_id '{callback_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        if not private_tool_privacy_enabled() and (
            user_content := callback_context.user_content
        ):
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        return None

    def after_agent(self, callback_context: CallbackContext) -> None:
        """Callback executed after agent processing completes.

        Args:
            callback_context (CallbackContext): Context containing agent name,
                invocation ID, state, and user content.
        """
        self.logger.info(
            f"*** Leaving agent '{callback_context.agent_name}' "
            f"with invocation_id '{callback_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        if not private_tool_privacy_enabled() and (
            user_content := callback_context.user_content
        ):
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        return None

    def before_model(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> None:
        """Callback executed before LLM model invocation.

        Args:
            callback_context (CallbackContext): Context containing agent name,
                invocation ID, state, and user content.
            llm_request (LlmRequest): The request being sent to the LLM model
                containing message contents.
        """
        self.logger.info(
            f"*** Before LLM call for agent '{callback_context.agent_name}' "
            f"with invocation_id '{callback_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        redact_content = private_tool_privacy_enabled()
        if not redact_content and (user_content := callback_context.user_content):
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        self.logger.debug(f"LLM request contains {len(llm_request.contents)} messages:")
        if redact_content:
            self.logger.debug("LLM request content redacted in private-tool mode")
        else:
            for i, content in enumerate(llm_request.contents, start=1):
                self.logger.debug(
                    f"Content {i}: {content.model_dump(exclude_none=True, mode='json')}"
                )

        return None

    def after_model(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        """Callback executed after LLM model responds.

        Args:
            callback_context (CallbackContext): Context containing agent name,
                invocation ID, state, and user content.
            llm_response (LlmResponse): The response received from the LLM model.
        """
        self.logger.info(
            f"*** After LLM call for agent '{callback_context.agent_name}' "
            f"with invocation_id '{callback_context.invocation_id}' ***"
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        redact_content = private_tool_privacy_enabled()
        if not redact_content and (user_content := callback_context.user_content):
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        if redact_content:
            self.logger.debug("LLM response content redacted in private-tool mode")
        elif llm_content := llm_response.content:
            response_data = llm_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"LLM response: {response_data}")

        return None

    def before_tool(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
        """Callback executed before tool invocation.

        Args:
            tool (BaseTool): The tool being invoked.
            args (dict[str, Any]): Arguments being passed to the tool.
            tool_context (ToolContext): Context containing agent name, invocation ID,
                state, user content, and event actions.
        """
        self.logger.info(
            f"*** Before invoking tool '{tool.name}' in agent "
            f"'{tool_context.agent_name}' with invocation_id "
            f"'{tool_context.invocation_id}' ***"
        )
        if private_tool_privacy_enabled():
            self.logger.debug("Tool payload redacted in private-tool mode")
            return None
        if is_private_tool(tool):
            self.logger.debug("Private tool payload redacted")
            return None
        self.logger.debug(f"State keys: {tool_context.state.to_dict().keys()}")

        if content := tool_context.user_content:
            self.logger.debug(
                f"User Content: {content.model_dump(exclude_none=True, mode='json')}"
            )

        actions_data = tool_context.actions.model_dump(exclude_none=True, mode="json")
        self.logger.debug(f"EventActions: {actions_data}")
        self.logger.debug(f"args: {args}")

        return None

    def after_tool(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict[str, Any],
    ) -> None:
        """Callback executed after tool invocation completes.

        Args:
            tool (BaseTool): The tool that was invoked.
            args (dict[str, Any]): Arguments that were passed to the tool.
            tool_context (ToolContext): Context containing agent name, invocation ID,
                state, user content, and event actions.
            tool_response (dict[str, Any]): The response returned by the tool.
        """
        self.logger.info(
            f"*** After invoking tool '{tool.name}' in agent "
            f"'{tool_context.agent_name}' with invocation_id "
            f"'{tool_context.invocation_id}' ***"
        )
        if private_tool_privacy_enabled():
            self.logger.debug("Tool payload redacted in private-tool mode")
            return None
        if is_private_tool(tool):
            self.logger.debug("Private tool payload redacted")
            return None
        self.logger.debug(f"State keys: {tool_context.state.to_dict().keys()}")

        if content := tool_context.user_content:
            self.logger.debug(
                f"User Content: {content.model_dump(exclude_none=True, mode='json')}"
            )

        actions_data = tool_context.actions.model_dump(exclude_none=True, mode="json")
        self.logger.debug(f"EventActions: {actions_data}")
        self.logger.debug(f"args: {args}")
        self.logger.debug(f"Tool response: {tool_response}")

        return None
