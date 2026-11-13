"""Agent lifecycle logging callbacks."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from .costs import (
    _attach_cost_metadata,
    _begin_cost_capture,
)

# Part fields that carry provider reasoning state rather than readable text.
_REDACTED_PART_FIELDS = ("part_metadata", "thought_signature")


def begin_cost_capture(
    *,
    user_id: str | None,
    session_id: str | None,
    invocation_id: str | None,
) -> None:
    """Start the cost capture associated with one ADK model callback."""
    _begin_cost_capture(
        user_id=user_id,
        session_id=session_id,
        invocation_id=invocation_id,
    )


def attach_cost_metadata(response: LlmResponse) -> dict[str, Any] | None:
    """Attach the captured cost to the current ADK response event."""
    return _attach_cost_metadata(response)


def _content_for_log(content: types.Content) -> dict[str, Any]:
    """Dump ``content`` for a debug log without encrypted reasoning.

    ``part_metadata`` holds raw OpenAI output items, which include
    ``encrypted_content``, and ``thought_signature`` is provider state. Neither
    is written to logs (design D11). Readable text is kept.
    """
    data: dict[str, Any] = content.model_dump(exclude_none=True, mode="json")
    for part in data.get("parts", []):
        for field in _REDACTED_PART_FIELDS:
            part.pop(field, None)
    return data


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
                   using Blacki's established lifecycle logger name.
        """
        if logger is None:
            logger = logging.getLogger("blacki.callbacks")
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
            content_data = _content_for_log(user_content)
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
            content_data = _content_for_log(user_content)
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
        session = getattr(callback_context, "session", None)
        begin_cost_capture(
            user_id=getattr(callback_context, "user_id", None),
            session_id=getattr(session, "id", None),
            invocation_id=getattr(callback_context, "invocation_id", None),
        )
        self.logger.debug(f"State keys: {callback_context.state.to_dict().keys()}")

        if user_content := callback_context.user_content:
            content_data = _content_for_log(user_content)
            self.logger.debug(f"User Content: {content_data}")

        self.logger.debug(f"LLM request contains {len(llm_request.contents)} messages:")
        for i, content in enumerate(llm_request.contents, start=1):
            self.logger.debug(f"Content {i}: {_content_for_log(content)}")

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
            content_data = _content_for_log(user_content)
            self.logger.debug(f"User Content: {content_data}")

        if llm_content := llm_response.content:
            response_data = _content_for_log(llm_content)
            self.logger.debug(f"LLM response: {response_data}")

        cost_observation = attach_cost_metadata(llm_response)
        if cost_observation:
            self.logger.debug(
                "LLM cost captured: %s",
                {
                    key: cost_observation[key]
                    for key in (
                        "cost_usd",
                        "upstream_cost_usd",
                        "estimated_cost_usd",
                        "cost_kind",
                    )
                    if key in cost_observation
                },
            )

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
            self.logger.debug(f"User Content: {_content_for_log(content)}")

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
            self.logger.debug(f"User Content: {_content_for_log(content)}")

        actions_data = tool_context.actions.model_dump(exclude_none=True, mode="json")
        self.logger.debug(f"EventActions: {actions_data}")
        self.logger.debug(f"args: {args}")
        self.logger.debug(f"Tool response: {tool_response}")

        return None
