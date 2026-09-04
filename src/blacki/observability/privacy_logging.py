"""ADK logging adapter for private tool payload redaction."""

from __future__ import annotations

from typing import Any

from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from ..security.tool_privacy import is_private_tool


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
