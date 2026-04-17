"""Agent lifecycle callback functions for monitoring.

This module provides callback functions that execute at various stages of the
agent lifecycle. These callbacks enable comprehensive logging and optional
Telegram tool notifications for Telegram-backed sessions.
"""

import asyncio
import functools
import logging
import os
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool

from .telegram import TelegramConfig
from .telegram.api import TelegramApiClient, TelegramApiError
from .telegram.formatting import escape_markdown
from .telegram.types import ParseMode

logger = logging.getLogger(__name__)

_TOOL_NOTIFY_LAST: dict[str, float] = {}
_TOOL_NOTIFY_LOCK = asyncio.Lock()
_TOOL_NOTIFY_MIN_INTERVAL_SEC = 0.35


def reset_telegram_tool_notify_rate_limiter_for_tests() -> None:
    """Clear per-chat rate limit state and env lookup cache (tests only)."""
    _TOOL_NOTIFY_LAST.clear()
    _telegram_tool_notifications_enabled_impl.cache_clear()


@functools.lru_cache(maxsize=32)
def _telegram_tool_notifications_enabled_impl(
    telegram_enabled: str,
    telegram_bot_token: str | None,
    telegram_tool_notifications: str,
) -> bool:
    """Return whether Telegram tool notifications are on for this env snapshot."""
    try:
        cfg = TelegramConfig.model_validate(
            {
                "TELEGRAM_ENABLED": telegram_enabled,
                "TELEGRAM_BOT_TOKEN": telegram_bot_token,
                "TELEGRAM_TOOL_NOTIFICATIONS": telegram_tool_notifications,
            }
        )
    except Exception:
        return False
    return cfg.tool_notifications_active()


def telegram_tool_notifications_enabled() -> bool:
    """Return True when Telegram tool notifications are configured and opted in.

    Uses a small LRU cache keyed by the relevant env vars so repeated tool
    calls do not re-parse the full environment each time.
    """
    return _telegram_tool_notifications_enabled_impl(
        os.environ.get("TELEGRAM_ENABLED", ""),
        os.environ.get("TELEGRAM_BOT_TOKEN"),
        os.environ.get("TELEGRAM_TOOL_NOTIFICATIONS", ""),
    )


def _parse_optional_int(value: Any) -> int | None:
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
    args: dict[str, Any],  # noqa: ARG001
    tool_context: ToolContext,
) -> None:
    """Send a short Telegram notice before a tool runs (Telegram sessions only).

    Skips silently when the feature is disabled, Telegram is not configured,
    or session state has no ``telegram_chat_id``. Failures are logged and do
    not block tool execution.
    """
    if not telegram_tool_notifications_enabled():
        return None

    chat_id_raw = tool_context.state.get("telegram_chat_id")
    if not chat_id_raw:
        return None

    chat_id = _parse_optional_int(chat_id_raw)
    if chat_id is None:
        logger.warning("Invalid telegram_chat_id in state: %r", chat_id_raw)
        return None

    thread_id = _parse_optional_int(tool_context.state.get("telegram_thread_id"))

    chat_key = str(chat_id)
    async with _TOOL_NOTIFY_LOCK:
        now = time.monotonic()
        last_sent = _TOOL_NOTIFY_LAST.get(chat_key, 0.0)
        elapsed = now - last_sent
        if elapsed < _TOOL_NOTIFY_MIN_INTERVAL_SEC:
            logger.debug(
                "Skipping Telegram tool notify (rate limit) chat_id=%s tool=%s",
                chat_id,
                tool.name,
            )
            return None
        _TOOL_NOTIFY_LAST[chat_key] = now

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None

    escaped_name = escape_markdown(tool.name)
    text = f"🔧 Using tool: *{escaped_name}*"

    try:
        async with TelegramApiClient(token) as client:
            await client.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=thread_id,
                disable_notification=True,
            )
    except TelegramApiError as exc:
        logger.warning(
            "Telegram tool notification failed for tool=%s: %s",
            tool.name,
            exc,
        )
    except Exception:
        logger.exception(
            "Unexpected error sending Telegram tool notification tool=%s",
            tool.name,
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

        if user_content := callback_context.user_content:
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

        if user_content := callback_context.user_content:
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

        if user_content := callback_context.user_content:
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        self.logger.debug(f"LLM request contains {len(llm_request.contents)} messages:")
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

        if user_content := callback_context.user_content:
            content_data = user_content.model_dump(exclude_none=True, mode="json")
            self.logger.debug(f"User Content: {content_data}")

        if llm_content := llm_response.content:
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
