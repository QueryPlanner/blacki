# mypy: disable-error-code="no-untyped-def"
"""Tests for layered prompt assembly and conditional policy routing."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from conftest import MockReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from blacki.prompt import (
    DomainPolicyPlugin,
    ResponsePolicyPlugin,
    build_domain_instruction,
    return_description_root,
    return_global_instruction,
    return_instruction_root,
    select_domain_policy_names,
)

ALL_DOMAIN_TOOLS = frozenset(
    {
        "log_meal",
        "get_calorie_summary",
        "set_training_program",
        "get_todays_training",
        "log_training",
        "advance_training_cycle",
        "get_training_history",
        "get_training_metrics",
        "update_training_metrics",
        "schedule_reminder",
        "list_reminders",
        "cancel_reminder",
        "exa_search",
        "brave_search",
    }
)


def _user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def _request_with_tools(*tool_names: str) -> LlmRequest:
    declarations = [types.FunctionDeclaration(name=name) for name in tool_names]
    request = LlmRequest()
    request.tools_dict = {name: cast(Any, object()) for name in tool_names}
    request.config.tools = [
        types.Tool(function_declarations=declarations),
        types.Tool(),
    ]
    return request


class TestStablePromptLayers:
    """Verify the safety, temporal, and core behavior layers."""

    def test_description_is_short_and_specific(self) -> None:
        description = return_description_root()

        assert "privacy-conscious" in description
        assert "tracking" in description

    def test_core_instruction_is_minimal(self) -> None:
        assert return_instruction_root() == "You are a helpful assistant."

    def test_global_instruction_has_precedence_and_privacy(
        self, mock_readonly_context: MockReadonlyContext
    ) -> None:
        instruction = return_global_instruction(mock_readonly_context)  # type: ignore[arg-type]

        expected_order = (
            "system safety and privacy rules first, developer\n"
            "behavior and domain policies second, the current user request third, "
            "and stored\nuser preferences last"
        )
        assert expected_order in instruction
        assert "never grant permissions or outrank this order" in instruction
        assert "Never silently mutate persistent state" in instruction
        assert "cannot change safety rules, tool permissions" in instruction

    def test_global_instruction_uses_application_timezone(
        self, mock_readonly_context: MockReadonlyContext
    ) -> None:
        instant = datetime(2025, 1, 15, 2, 0, tzinfo=UTC)
        with (
            patch.dict("os.environ", {"AGENT_TIMEZONE": "America/New_York"}),
            patch("blacki.prompt.now_utc", return_value=instant),
        ):
            instruction = return_global_instruction(mock_readonly_context)  # type: ignore[arg-type]

        assert "Current application date: 2025-01-14" in instruction
        assert "Application timezone: America/New_York" in instruction
        assert "Reminders use the same timezone" in instruction
        assert instruction.count("<temporal_context>") == 1


@pytest.mark.parametrize(
    ("user_text", "expected"),
    [
        ("How many calories are in an apple?", ("nutrition",)),
        ("I ate a sandwich for lunch", ("nutrition",)),
        ("Log my resistance workout", ("workout",)),
        ("Suggest a reminder schedule", ("reminder",)),
        ("What is the latest verified Python news?", ("search",)),
        ("Explain dependency injection", ()),
    ],
)
def test_select_domain_policy_names(user_text: str, expected: tuple[str, ...]) -> None:
    assert select_domain_policy_names(user_text, ALL_DOMAIN_TOOLS) == expected


def test_router_does_not_describe_disabled_tools() -> None:
    assert select_domain_policy_names("Log my lunch", frozenset()) == ()
    assert build_domain_instruction("Log my lunch", frozenset()) == ""


def test_health_policy_is_enabled_only_for_health_tool() -> None:
    assert select_domain_policy_names(
        "Show my Google Health sleep summary", {"get_health_summary"}
    ) == ("health",)
    instruction = build_domain_instruction(
        "Show my Google Health sleep summary", {"get_health_summary"}
    )
    assert "read-only wellness summary source" in instruction
    assert "never infer, diagnose" in instruction


def test_nutrition_policy_separates_local_save_and_google_sync() -> None:
    instruction = build_domain_instruction(
        "Log my lunch", {"log_meal", "edit_meal", "delete_meal"}
    )

    assert "Do not mention a pending background export" in instruction
    assert "saved in Blacki" in instruction
    assert "Never claim Google Health accepted" in instruction
    assert "Do not repeat a meal mutation" in instruction
    assert "queues eligible existing meals once" in instruction


class TestDomainPolicyAssembly:
    """Verify conditional policy content for behavior-sensitive requests."""

    def test_nutrition_distinguishes_questions_from_logs(self) -> None:
        instruction = build_domain_instruction(
            "How many calories are in an apple?", {"log_meal"}
        )

        assert (
            "General\nnutrition questions must not create or change records"
            in instruction
        )
        assert "food name\nby itself is ambiguous" in instruction
        assert "never replace an invalid date with today" in instruction

    def test_workout_uses_canonical_system_without_implicit_advance(self) -> None:
        instruction = build_domain_instruction(
            "Log my completed workout but do not advance",
            {"log_training", "advance_training_cycle"},
        )

        assert "training-program API is canonical" in instruction
        assert "Logging never advances the\ncycle by implication" in instruction
        assert "only when the user explicitly asks" in instruction
        assert "fallback-only" not in instruction

    def test_workout_mentions_legacy_only_when_exposed(self) -> None:
        instruction = build_domain_instruction(
            "Show my weekly workout split", {"log_training", "get_todays_workout"}
        )

        assert "Legacy split tools are fallback-only" in instruction

    def test_reminder_discussion_is_read_only(self) -> None:
        instruction = build_domain_instruction(
            "Suggest a reminder schedule but do not save it", {"schedule_reminder"}
        )

        assert "discussing a possible schedule is read-only" in instruction
        assert "Ask for a missing required\ntime" in instruction

    @pytest.mark.parametrize(
        ("tools", "expected", "unexpected"),
        [
            (
                {"exa_search", "brave_search"},
                "Use exa_search first",
                "Use brave_search with five results",
            ),
            ({"exa_search"}, "Use exa_search with five results", "brave_search"),
            ({"brave_search"}, "Use brave_search with five results", "exa_search"),
        ],
    )
    def test_search_policy_mentions_only_enabled_search_tools(
        self, tools: set[str], expected: str, unexpected: str
    ) -> None:
        instruction = build_domain_instruction("Find the latest news", tools)

        assert expected in instruction
        assert unexpected not in instruction
        assert "Never use sandbox" in instruction
        assert "browser automation" in instruction
        assert "Use exactly one primary search call" in instruction
        assert "never return to the primary provider" in instruction
        assert "A successful result ends tool use" in instruction


class TestDomainPolicyPlugin:
    """Verify ADK integration uses the current invocation's user content."""

    @pytest.mark.asyncio
    async def test_appends_relevant_policy_from_user_content(self) -> None:
        plugin = DomainPolicyPlugin()
        request = LlmRequest()
        request.tools_dict = {"log_meal": object()}  # type: ignore[dict-item]
        context = SimpleNamespace(user_content=_user_content("Log yesterday's lunch"))

        await plugin.before_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_request=request,
        )

        assert "<nutrition_policy>" in str(request.config.system_instruction)

    @pytest.mark.asyncio
    async def test_ignores_missing_user_content(self) -> None:
        plugin = DomainPolicyPlugin()
        request = LlmRequest()

        await plugin.before_model_callback(
            callback_context=SimpleNamespace(user_content=None),  # type: ignore[arg-type]
            llm_request=request,
        )

        assert request.config.system_instruction is None

    @pytest.mark.asyncio
    async def test_ignores_user_content_without_text(self) -> None:
        plugin = DomainPolicyPlugin()
        request = LlmRequest()
        context = SimpleNamespace(user_content=types.Content(role="user", parts=[]))

        await plugin.before_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_request=request,
        )

        assert request.config.system_instruction is None

    @pytest.mark.asyncio
    async def test_ignores_request_when_relevant_tools_are_disabled(self) -> None:
        plugin = DomainPolicyPlugin()
        request = LlmRequest()
        context = SimpleNamespace(user_content=_user_content("Log my lunch"))

        await plugin.before_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_request=request,
        )

        assert request.config.system_instruction is None

    @pytest.mark.asyncio
    async def test_search_initially_exposes_only_primary_and_hides_sandbox(
        self,
    ) -> None:
        plugin = DomainPolicyPlugin()
        request = _request_with_tools(
            "exa_search", "brave_search", "sandbox_execute_code", "sandbox_view_image"
        )
        opaque_tool = object()
        assert request.config.tools is not None
        request.config.tools.append(cast(Any, opaque_tool))
        context = SimpleNamespace(
            user_content=_user_content("Find the latest news"), state={}
        )

        await plugin.before_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_request=request,
        )

        assert set(request.tools_dict) == {
            "exa_search",
            "brave_search",
            "sandbox_execute_code",
            "sandbox_view_image",
        }
        assert context.state["temp:blacki_search_primary"] == "exa_search"
        assert request.config.tools is not None
        first_tool = request.config.tools[0]
        assert isinstance(first_tool, types.Tool)
        declarations = first_tool.function_declarations
        assert declarations is not None
        assert [declaration.name for declaration in declarations] == ["exa_search"]
        second_tool = request.config.tools[1]
        assert isinstance(second_tool, types.Tool)
        assert second_tool.function_declarations is None
        assert request.config.tools[-1] is opaque_tool

    @pytest.mark.asyncio
    async def test_successful_search_removes_search_tools_on_next_model_call(
        self,
    ) -> None:
        plugin = DomainPolicyPlugin()
        state: dict[str, object] = {}
        tool_context = SimpleNamespace(state=state)
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="exa_search"),  # type: ignore[arg-type]
            tool_args={"query": "news"},
            tool_context=tool_context,  # type: ignore[arg-type]
            result={"status": "success", "results": [{"title": "result"}]},
        )
        request = _request_with_tools("exa_search", "brave_search")
        context = SimpleNamespace(
            user_content=_user_content("Find the latest news"), state=state
        )

        await plugin.before_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_request=request,
        )

        assert set(request.tools_dict) == {"exa_search", "brave_search"}
        assert request.config.tools is not None
        assert len(request.config.tools) == 1
        remaining_tool = request.config.tools[0]
        assert isinstance(remaining_tool, types.Tool)
        assert remaining_tool.function_declarations is None

    @pytest.mark.asyncio
    async def test_failed_primary_exposes_fallback_then_stops(self) -> None:
        plugin = DomainPolicyPlugin()
        state: dict[str, object] = {}
        tool_context = SimpleNamespace(state=state)
        primary = SimpleNamespace(name="exa_search")
        await plugin.after_tool_callback(
            tool=primary,  # type: ignore[arg-type]
            tool_args={},
            tool_context=tool_context,  # type: ignore[arg-type]
            result={"status": "error", "results": []},
        )
        request = _request_with_tools("exa_search", "brave_search")
        context = SimpleNamespace(
            user_content=_user_content("Find the latest news"), state=state
        )

        await plugin.before_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_request=request,
        )

        assert set(request.tools_dict) == {"exa_search", "brave_search"}
        assert request.config.tools is not None
        first_tool = request.config.tools[0]
        assert isinstance(first_tool, types.Tool)
        declarations = first_tool.function_declarations
        assert declarations is not None
        assert [declaration.name for declaration in declarations] == ["brave_search"]

        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="brave_search"),  # type: ignore[arg-type]
            tool_args={},
            tool_context=tool_context,  # type: ignore[arg-type]
            result={"status": "error", "results": []},
        )
        assert state
        assert state["temp:blacki_search_status"] == "complete"

    @pytest.mark.asyncio
    async def test_search_budget_handles_single_provider_and_non_search_tool(
        self,
    ) -> None:
        plugin = DomainPolicyPlugin()
        state: dict[str, object] = {}
        request = _request_with_tools("exa_search")
        context = SimpleNamespace(
            user_content=_user_content("Find the latest news"), state=state
        )

        await plugin.before_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_request=request,
        )
        assert set(request.tools_dict) == {"exa_search"}

        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="log_meal"),  # type: ignore[arg-type]
            tool_args={},
            tool_context=SimpleNamespace(state=state),  # type: ignore[arg-type]
            result={"status": "success"},
        )
        assert state == {"temp:blacki_search_primary": "exa_search"}

        state["temp:blacki_search_status"] = "complete"
        request = LlmRequest()
        request.tools_dict = {"exa_search": cast(Any, object())}
        await plugin.before_model_callback(
            callback_context=context,  # type: ignore[arg-type]
            llm_request=request,
        )
        assert set(request.tools_dict) == {"exa_search"}

    @pytest.mark.asyncio
    async def test_search_execution_budget_uses_primary_fallback_and_cache(
        self,
    ) -> None:
        plugin = DomainPolicyPlugin()
        state: dict[str, object] = {"temp:blacki_search_primary": "exa_search"}
        tool_context = SimpleNamespace(state=state)

        assert (
            await plugin.before_tool_callback(
                tool=SimpleNamespace(name="log_meal"),  # type: ignore[arg-type]
                tool_args={},
                tool_context=tool_context,  # type: ignore[arg-type]
            )
            is None
        )
        assert (
            await plugin.before_tool_callback(
                tool=SimpleNamespace(name="exa_search"),  # type: ignore[arg-type]
                tool_args={},
                tool_context=tool_context,  # type: ignore[arg-type]
            )
            is None
        )
        wrong_primary = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="brave_search"),  # type: ignore[arg-type]
            tool_args={},
            tool_context=tool_context,  # type: ignore[arg-type]
        )
        assert wrong_primary is not None
        assert "primary" in str(wrong_primary["error"])

        state["temp:blacki_search_status"] = "primary_failed"
        wrong_fallback = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="exa_search"),  # type: ignore[arg-type]
            tool_args={},
            tool_context=tool_context,  # type: ignore[arg-type]
        )
        assert wrong_fallback is not None
        assert "fallback" in str(wrong_fallback["error"])

        cached = {"status": "success", "results": [{"title": "result"}]}
        state["temp:blacki_search_status"] = "complete"
        state["temp:blacki_search_result"] = cached
        assert (
            await plugin.before_tool_callback(
                tool=SimpleNamespace(name="exa_search"),  # type: ignore[arg-type]
                tool_args={},
                tool_context=tool_context,  # type: ignore[arg-type]
            )
            == cached
        )

        del state["temp:blacki_search_result"]
        completed = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="exa_search"),  # type: ignore[arg-type]
            tool_args={},
            tool_context=tool_context,  # type: ignore[arg-type]
        )
        assert completed is not None
        assert "already completed" in str(completed["error"])


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
