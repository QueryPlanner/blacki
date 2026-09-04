"""Sandbox image result handling for ADK model input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.plugins.multimodal_tool_results_plugin import (
    MultimodalToolResultsPlugin,
)
from google.adk.tools import ToolContext
from google.genai import types

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    from google.adk.tools.base_tool import BaseTool


class SandboxMultimodalToolResultsPlugin(MultimodalToolResultsPlugin):
    """Attach image parts while keeping binary data out of tool history."""

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: Any,
    ) -> dict[str, Any] | None:
        if tool.name != "sandbox_view_image":
            return None

        await super().after_tool_callback(
            tool=tool,
            tool_args=tool_args,
            tool_context=tool_context,
            result=result,
        )
        if isinstance(result, types.Part) or (
            isinstance(result, list) and result and isinstance(result[0], types.Part)
        ):
            return {
                "status": "success",
                "message": "Image attached as a separate visual input.",
            }
        return None

    async def on_event_callback(
        self,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> Event | None:
        """Remove the binary copy from the persisted function response event."""
        _ = invocation_context
        content = event.content
        if content is None or not content.parts:
            return None
        for part in content.parts:
            function_response = part.function_response
            if (
                function_response is None
                or function_response.name != "sandbox_view_image"
            ):
                continue
            response = function_response.response
            if not isinstance(response, dict):
                continue
            result = response.get("result")
            if not (
                isinstance(result, list)
                and result
                and isinstance(result[0], types.Part)
            ):
                continue
            function_response.response = {
                "status": "success",
                "message": "Image attached as a separate visual input.",
            }
        return None
