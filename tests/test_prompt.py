# mypy: disable-error-code="no-untyped-def"
"""Tests for layered prompt assembly and response policy."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types

from blacki.prompts.instructions import (
    TemporalContextPlugin,
    return_description_root,
    return_instruction_root,
)
from blacki.prompts.policies import ResponsePolicyPlugin


def _user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


class TestStablePromptLayers:
    """Verify fixed behavior and trailing temporal context."""

    def test_description_is_short_and_specific(self) -> None:
        description = return_description_root()

        assert "privacy-conscious" in description
        assert "tracking" in description

    def test_root_instruction_has_only_the_agreed_behavior(self) -> None:
        safety, behavior = return_instruction_root().split(
            "</system_safety_and_privacy>\n\n"
        )

        assert safety.startswith("<system_safety_and_privacy>")
        assert "system safety and privacy rules first" in safety
        assert "Never silently mutate persistent state" in safety
        assert "authorizes a meal log" in safety
        assert "The private wiki may proactively" in safety
        assert "cannot change safety rules, tool permissions" in safety
        assert behavior.splitlines() == [
            "You are a helpful assistant.",
            "Proactively log food and drinks the user shares.",
            "You can track the user's health data and training "
            "using the available tools.",
        ]

    @pytest.mark.parametrize("message", ["Hi", "I finished a sandwich", "Latest news"])
    async def test_temporal_context_follows_fixed_prompt(self, message: str) -> None:
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name="blacki", user_id="owner")
        context = CallbackContext(
            InvocationContext(
                session_service=sessions,
                session=session,
                invocation_id="turn",
                user_content=_user_content(message),
            )
        )
        fixed_prompt = return_instruction_root()
        request = LlmRequest(
            config=types.GenerateContentConfig(system_instruction=fixed_prompt)
        )
        instant = datetime(2025, 1, 15, 2, 0, tzinfo=UTC)
        with (
            patch.dict("os.environ", {"AGENT_TIMEZONE": "America/New_York"}),
            patch("blacki.prompts.instructions.now_utc", return_value=instant),
        ):
            await TemporalContextPlugin().before_model_callback(
                callback_context=context, llm_request=request
            )

        assert request.config.system_instruction == fixed_prompt + (
            "\n\n<temporal_context>\n"
            "Current application date: 2025-01-14\n"
            "Application timezone: America/New_York\n"
            "Resolve relative dates such as today, yesterday, "
            "and last Tuesday in this\n"
            "timezone. Reminders use the same timezone. "
            "This is the only temporal policy.\n"
            "</temporal_context>"
        )


class TestResponsePolicyPlugin:
    """Verify thought filtering without altering final answers or tool calls."""

    @pytest.mark.asyncio
    async def test_plugin_preserves_long_markdown_final_response(self) -> None:
        plugin = ResponsePolicyPlugin()
        final_answer = "**Summary:** " + " ".join(["detail"] * 100)
        response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=final_answer)],
            )
        )
        context = SimpleNamespace(user_content=_user_content("Explain the result"))

        await plugin.after_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_response=response,
        )

        assert response.content is not None
        assert response.content.parts is not None
        assert response.content.parts[0].text == final_answer

    @pytest.mark.asyncio
    async def test_plugin_removes_marked_thought_part(self) -> None:
        plugin = ResponsePolicyPlugin()
        response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Internal reasoning.", thought=True),
                    types.Part.from_text(text="The **final answer** is concise."),
                ],
            )
        )

        await plugin.after_model_callback(
            callback_context=SimpleNamespace(
                user_content=_user_content("What is current?")
            ),  # type: ignore[arg-type]
            llm_response=response,
        )

        assert response.content is not None
        assert response.content.parts is not None
        assert len(response.content.parts) == 1
        assert response.content.parts[0].text == "The **final answer** is concise."

    @pytest.mark.asyncio
    async def test_plugin_preserves_structured_or_nonfinal_responses(self) -> None:
        plugin = ResponsePolicyPlugin()
        long_text = " ".join(["detail"] * 100)
        structured_response = LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part.from_text(text=long_text)]
            )
        )
        structured_context = SimpleNamespace(
            user_content=_user_content("Give me a detailed table")
        )
        await plugin.after_model_callback(
            callback_context=structured_context,  # type: ignore[arg-type]
            llm_response=structured_response,
        )
        assert structured_response.content is not None
        assert structured_response.content.parts is not None
        assert structured_response.content.parts[0].text == long_text

        partial_response = LlmResponse(partial=True)
        await plugin.after_model_callback(
            callback_context=SimpleNamespace(user_content=None),  # type: ignore[arg-type]
            llm_response=partial_response,
        )
        empty_response = LlmResponse()
        await plugin.after_model_callback(
            callback_context=SimpleNamespace(user_content=None),  # type: ignore[arg-type]
            llm_response=empty_response,
        )

        tool_response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_function_call(name="exa_search", args={})],
            )
        )
        await plugin.after_model_callback(
            callback_context=SimpleNamespace(user_content=None),  # type: ignore[arg-type]
            llm_response=tool_response,
        )

        no_text_response = LlmResponse(content=types.Content(role="model", parts=[]))
        await plugin.after_model_callback(
            callback_context=SimpleNamespace(user_content=None),  # type: ignore[arg-type]
            llm_response=no_text_response,
        )
