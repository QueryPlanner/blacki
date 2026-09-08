"""Privacy controls for user-scoped tools."""

from __future__ import annotations

import logging
import os

from google.adk.tools.base_tool import BaseTool

_ENABLED_VALUES = frozenset({"1", "true", "yes"})
_ZEPTO_TOOL_PREFIX = "zepto_"
_GMAIL_TOOL_PREFIX = "gmail_"
_PRIVATE_TOOL_NAMES = frozenset(
    {
        "log_meal",
        "edit_meal",
        "delete_meal",
        "get_calorie_summary",
        "get_meal_sync_status",
        "retry_meal_sync",
        "set_calorie_goal",
        "get_health_summary",
        "send_text_to_speech",
        "list_user_files",
        "restore_user_file",
        "delete_user_file",
    }
)


def zepto_mcp_enabled() -> bool:
    """Return whether secure Zepto mode is explicitly enabled."""
    return os.getenv("ZEPTO_MCP_ENABLED", "false").strip().lower() in _ENABLED_VALUES


def gmail_configured() -> bool:
    """Return whether shared Gmail OAuth settings are present."""
    from ..gmail.config import GmailConfig
    from ..gmail.errors import GmailConfigurationError

    try:
        return GmailConfig.from_environment() is not None
    except GmailConfigurationError:
        # Keep redaction on while an operator repairs a partial configuration.
        return True


def kokoro_tts_enabled() -> bool:
    """Return whether the private Kokoro TTS integration is configured."""
    return bool(os.getenv("KOKORO_TTS_BASE_URL", "").strip())


def r2_files_enabled() -> bool:
    """Return whether private durable-file tools are configured."""
    return os.getenv("R2_FILES_ENABLED", "false").strip().lower() in _ENABLED_VALUES


def google_health_enabled() -> bool:
    """Return whether a complete Google Health connector is configured."""
    from ..health.config import google_health_configured_from_environment

    return google_health_configured_from_environment()


def private_tool_privacy_enabled() -> bool:
    """Return whether any configured tool needs content-level redaction."""
    return (
        zepto_mcp_enabled()
        or gmail_configured()
        or kokoro_tts_enabled()
        or google_health_enabled()
        or r2_files_enabled()
    )


def configure_zepto_privacy() -> bool:
    """Disable content-rich tracing and MCP body debug in secure Zepto mode."""
    if not zepto_mcp_enabled():
        return False
    os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "false"
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "false"
    logging.getLogger("google_adk.google.adk.tools.mcp_tool").setLevel(logging.INFO)
    logging.getLogger("google.adk.tools.mcp_tool").setLevel(logging.INFO)
    return True


def configure_private_tool_privacy() -> bool:
    """Disable content-rich tracing when any private tool is configured."""
    if not private_tool_privacy_enabled():
        return False
    os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "false"
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "false"
    if zepto_mcp_enabled():
        logging.getLogger("google_adk.google.adk.tools.mcp_tool").setLevel(logging.INFO)
        logging.getLogger("google.adk.tools.mcp_tool").setLevel(logging.INFO)
    return True


def is_private_tool(tool: BaseTool) -> bool:
    """Return whether a tool can expose private account or message data."""
    is_zepto = tool.name.startswith(_ZEPTO_TOOL_PREFIX)
    is_gmail = tool.name.startswith(_GMAIL_TOOL_PREFIX)
    is_named_private = tool.name in _PRIVATE_TOOL_NAMES
    return is_zepto or is_gmail or is_named_private
