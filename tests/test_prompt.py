# mypy: disable-error-code="no-untyped-def"
"""Tests for layered prompt assembly."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions import InMemorySessionService
from google.genai import types

from blacki.prompts.instructions import (
    TemporalContextPlugin,
    return_description_root,
    return_instruction_root,
)


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
