"""Privacy controls for user-scoped tools."""

from __future__ import annotations

import logging
import os
from typing import Any

from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

_ENABLED_VALUES = frozenset({"1", "true", "yes"})
_ZEPTO_TOOL_PREFIX = "zepto_"
_PRIVATE_TOOL_NAMES = frozenset(
    {
        "send_text_to_speech",
        "list_user_files",
        "restore_user_file",
        "delete_user_file",
    }
)


def zepto_mcp_enabled() -> bool:
    """Return whether secure Zepto mode is explicitly enabled."""
    return os.getenv("ZEPTO_MCP_ENABLED", "false").strip().lower() in _ENABLED_VALUES


def kokoro_tts_enabled() -> bool:
    """Return whether the private Kokoro TTS integration is configured."""
    return bool(os.getenv("KOKORO_TTS_BASE_URL", "").strip())


def r2_files_enabled() -> bool:
    """Return whether private durable-file tools are configured."""
    return os.getenv("R2_FILES_ENABLED", "false").strip().lower() in _ENABLED_VALUES


def private_tool_privacy_enabled() -> bool:
    """Return whether any configured tool needs content-level redaction."""
    return zepto_mcp_enabled() or kokoro_tts_enabled() or r2_files_enabled()


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
    return tool.name.startswith(_ZEPTO_TOOL_PREFIX) or tool.name in _PRIVATE_TOOL_NAMES


class PrivacyAwareLoggingPlugin(LoggingPlugin):
    """Keep ADK lifecycle logs while redacting private tool payloads."""

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        if not is_private_tool(tool):
            return await super().before_tool_callback(
                tool=tool,
                tool_args=tool_args,
                tool_context=tool_context,
            )
        self._log("🔧 PRIVATE TOOL STARTING")
        self._log(f"   Tool Name: {tool.name}")
        self._log(f"   Function Call ID: {tool_context.function_call_id}")
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not is_private_tool(tool):
            return await super().after_tool_callback(
                tool=tool,
                tool_args=tool_args,
                tool_context=tool_context,
                result=result,
            )
        self._log("🔧 PRIVATE TOOL COMPLETED")
        self._log(f"   Tool Name: {tool.name}")
        self._log(f"   Function Call ID: {tool_context.function_call_id}")
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        if not is_private_tool(tool):
            return await super().on_tool_error_callback(
                tool=tool,
                tool_args=tool_args,
                tool_context=tool_context,
                error=error,
            )
        self._log("🔧 PRIVATE TOOL ERROR")
        self._log(f"   Tool Name: {tool.name}")
        self._log(f"   Function Call ID: {tool_context.function_call_id}")
        return None
