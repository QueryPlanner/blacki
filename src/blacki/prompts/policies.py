"""ADK response policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_response import LlmResponse


class ResponsePolicyPlugin(BasePlugin):
    """Remove marked thought text from eligible final responses."""

    def __init__(self, name: str = "response_policy") -> None:
        super().__init__(name=name)

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        _ = callback_context
        if llm_response.partial or llm_response.content is None:
            return

        parts = llm_response.content.parts or []
        if any(part.function_call or part.function_response for part in parts):
            return
        answer_parts = [part for part in parts if part.text and not part.thought]
        if len(answer_parts) != 1:
            return

        llm_response.content.parts = [
            part for part in parts if not (part.text and part.thought)
        ]
