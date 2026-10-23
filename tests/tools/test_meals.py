"""Exercise meal discovery and dispatch through real ADK and SQLite boundaries."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.events import Event
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types

from blacki.container import AppContainer, reset_container_for_tests, set_container
from blacki.health.config import GOOGLE_HEALTH_NUTRITION_SCOPES
from blacki.models.openrouter_request import build_chat_request
from blacki.observability.lifecycle import LoggingCallbacks
from blacki.observability.privacy_logging import PrivacyAwareLoggingPlugin
from blacki.prompts.policies import DomainPolicyPlugin
from blacki.security.tool_privacy import is_private_tool
from blacki.telegram.progress import describe_tool
from blacki.telegram.progress_callbacks import (
    block_telegram_tool_retry,
    recover_telegram_tool_error,
)
from blacki.tools.calories import log_meal
from blacki.tools.meals import (
    MEAL_ACTIONS,
    DescribeActionTool,
    MealsTool,
    create_meal_tools,
)
from blacki.tools.registry import ToolConfig, build_tools

USER = "telegram-chat-42"


@pytest.fixture
async def meal_container(tmp_path: Path) -> AsyncGenerator[AppContainer, None]:
    container = await AppContainer.create(tmp_path / "meals.db")
    await container.initialize_all_storages()
    set_container(container)
    try:
        yield container
    finally:
        reset_container_for_tests()
        await container.close()


async def make_context(user_id: str = USER, *, private: bool = True) -> ToolContext:
    sessions = InMemorySessionService()
    session = await sessions.create_session(
        app_name="meals_test",
        user_id=user_id,
        state={
            "telegram_chat_id": "42",
            "telegram_chat_type": "private" if private else "group",
        },
    )
    return ToolContext(
        InvocationContext(
            session_service=sessions,
            invocation_id="meal-turn",
            session=session,
            agent=LlmAgent(name="meals_test", model="gemini-2.5-flash"),
        ),
        function_call_id="meal-call",
    )


@pytest.fixture
async def meal_context() -> ToolContext:
    return await make_context()


@pytest.fixture
def meals() -> MealsTool:
    return MealsTool()


async def dispatch(
    tool: MealsTool, context: ToolContext, action: str, **arguments: Any
) -> dict[str, Any]:
    return await tool.run_async(
        args={"action": action, "arguments": arguments}, tool_context=context
    )


async def connect_health(container: AppContainer) -> None:
    cipher = Fernet(Fernet.generate_key())
    await container.google_health_storage.upsert_connection(
        telegram_user_id=USER,
        encrypted_refresh_token=cipher.encrypt(b"test-refresh").decode(),
        health_user_id=USER,
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_NUTRITION_SCOPES,
    )


async def test_meal_lifecycle_keeps_storage_and_export_semantics(
    meal_container: AppContainer,
    meal_context: ToolContext,
    meals: MealsTool,
) -> None:
    await connect_health(meal_container)
    created = await dispatch(
        meals,
        meal_context,
        "log_meal",
        description="Rice and dal",
        estimated_calories=500,
        date="2026-10-20",
        meal_type="lunch",
        protein_g=20,
    )
    assert created["status"] == "success"
    assert created["google_health_sync"] == "pending"
    entry_id = created["entry_id"]
    entry = (
        await meal_container.calorie_storage.get_daily_summary(USER, "2026-10-20")
    ).entries[0]
    assert entry is not None and entry.calories == 500 and entry.protein_g == 20

    goal = await dispatch(meals, meal_context, "set_calorie_goal", daily_calories=2400)
    assert goal["new_goal"] == 2400
    summary = await dispatch(
        meals, meal_context, "get_calorie_summary", date="2026-10-20"
    )
    assert summary["summary"]["total_calories"] == 500
    assert summary["summary"]["entry_count"] == 1
    assert summary["calorie_goal"] == 2400

    nutrition = meal_container.google_health_storage.nutrition
    revision = (await nutrition.revisions(entry_id))[0]
    await nutrition.revision_state(int(revision["sequence"]), "failed")
    await nutrition.result(entry_id, "failed", error="provider_unavailable")
    status = await dispatch(meals, meal_context, "get_meal_sync_status")
    assert status["google_health_sync"] == {"failed": 1}
    # A failed remote export leaves precisely one local meal.
    assert (
        await meal_container.calorie_storage.get_daily_summary(USER, "2026-10-20")
    ).entry_count == 1
    retried = await dispatch(meals, meal_context, "retry_meal_sync")
    assert retried["requeued"] == 1
    assert retried["google_health_sync"] == {"pending": 1}

    edited = await dispatch(
        meals, meal_context, "edit_meal", entry_id=entry_id, estimated_calories=550
    )
    assert edited["status"] == "success" and edited["google_health_sync"] == "pending"
    entry = (
        await meal_container.calorie_storage.get_daily_summary(USER, "2026-10-20")
    ).entries[0]
    assert entry is not None and entry.calories == 550
    deleted = await dispatch(meals, meal_context, "delete_meal", entry_id=entry_id)
    assert deleted["status"] == "success" and deleted["google_health_sync"] == "pending"
    assert (
        await meal_container.calorie_storage.get_daily_summary(USER, "2026-10-20")
    ).entry_count == 0


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"action": "log_meal"},
        {"action": 1, "arguments": {}},
        {"action": "log_meal", "arguments": []},
        {"action": "log_meal", "arguments": {}, "user_id": "victim"},
        {"action": "unknown", "arguments": {}},
        {"action": "log_meal", "arguments": {"description": "private-input"}},
        {
            "action": "log_meal",
            "arguments": {"description": "private-input", "estimated_calories": "500"},
        },
        {
            "action": "log_meal",
            "arguments": {"description": "private-input", "estimated_calories": 500.0},
        },
        {
            "action": "log_meal",
            "arguments": {"description": "private-input", "estimated_calories": True},
        },
        {
            "action": "log_meal",
            "arguments": {
                "description": "private-input",
                "estimated_calories": 500,
                "tool_context": {"user_id": "victim"},
            },
        },
        {
            "action": "log_meal",
            "arguments": {
                "description": "private-input",
                "estimated_calories": 500,
                "user_id": "victim",
            },
        },
        {
            "action": "log_meal",
            "arguments": {
                "description": "private-input",
                "estimated_calories": 500,
                "date": "not-a-valid-date",
            },
        },
    ],
)
async def test_invalid_payloads_cannot_write_or_echo_private_values(
    args: dict[str, Any],
    meal_container: AppContainer,
    meal_context: ToolContext,
    meals: MealsTool,
) -> None:
    result = await meals.run_async(args=args, tool_context=meal_context)
    assert result["status"] == "error"
    assert "private-input" not in json.dumps(result)
    async with meal_container.conn.execute(
        "SELECT count(*) FROM calorie_logs"
    ) as cursor:
        row = await cursor.fetchone()
        assert row is not None and row[0] == 0


@pytest.mark.parametrize(
    ("action", "arguments"),
    [
        ("get_meal_sync_status", None),
        ("get_meal_sync_status", {}),
        ("retry_meal_sync", None),
        ("retry_meal_sync", {}),
    ],
)
async def test_zero_argument_actions_accept_omitted_arguments(
    action: str,
    arguments: dict[str, Any] | None,
    meal_container: AppContainer,
    meal_context: ToolContext,
    meals: MealsTool,
) -> None:
    payload: dict[str, Any] = {"action": action}
    if arguments is not None:
        payload["arguments"] = arguments
    result = await meals.run_async(args=payload, tool_context=meal_context)
    assert result["status"] != "error", result


async def test_dispatch_keeps_user_and_private_chat_boundaries(
    meal_container: AppContainer,
    meal_context: ToolContext,
    meals: MealsTool,
) -> None:
    created = await dispatch(
        meals,
        meal_context,
        "log_meal",
        description="Owner meal",
        estimated_calories=100,
        date="2026-10-20",
    )
    assert created["google_health_sync"] == "not_enabled"
    other = await make_context("telegram-chat-43")
    for action, arguments in [
        ("edit_meal", {"estimated_calories": 999}),
        ("delete_meal", {}),
    ]:
        result = await dispatch(
            meals, other, action, entry_id=created["entry_id"], **arguments
        )
        assert result["status"] == "error"
    summary = await dispatch(meals, other, "get_calorie_summary")
    assert summary["summary"]["entry_count"] == 0
    owner_entry = (
        await meal_container.calorie_storage.get_daily_summary(USER, "2026-10-20")
    ).entries[0]
    assert owner_entry is not None and owner_entry.calories == 100
    group = await make_context("telegram-chat--42", private=False)
    for action in ("get_meal_sync_status", "retry_meal_sync"):
        assert (await dispatch(meals, group, action))["status"] == "error"


@pytest.mark.parametrize(
    "action",
    [
        "log_meal",
        "get_calorie_summary",
        "edit_meal",
        "delete_meal",
        "set_calorie_goal",
        "get_meal_sync_status",
        "retry_meal_sync",
    ],
)
async def test_discovery_is_read_only_and_describes_callable_parameters(
    action: str,
    meal_container: AppContainer,
    meal_context: ToolContext,
    meals: MealsTool,
) -> None:
    discovery = DescribeActionTool(meals)
    result = await discovery.run_async(
        args={"domain": "meals", "action": action}, tool_context=meal_context
    )
    assert result["status"] == "success"
    assert result["description"]
    assert "tool_context" not in result["parameters"]["properties"]
    assert result["parameters"]["additionalProperties"] is False
    async with meal_container.conn.execute(
        "SELECT count(*) FROM calorie_logs"
    ) as cursor:
        row = await cursor.fetchone()
        assert row is not None and row[0] == 0


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"domain": "meals", "action": []},
        {"domain": "gmail", "action": "gmail_send_draft"},
        {"domain": "meals", "action": "unknown"},
        {"domain": "meals", "action": "log_meal", "arguments": {}},
    ],
)
async def test_discovery_rejects_disabled_or_invalid_actions(
    args: dict[str, Any], meal_context: ToolContext
) -> None:
    discovery = create_meal_tools()[1]
    assert (await discovery.run_async(args=args, tool_context=meal_context))[
        "status"
    ] == "error"


async def test_registry_exposes_discovery_only_with_meals(
    meal_context: ToolContext,
) -> None:
    enabled = build_tools(ToolConfig(sqlite_path="unused", weather_enabled=False))
    disabled = build_tools(ToolConfig(weather_enabled=False))
    request = LlmRequest()
    for tool in enabled:
        if isinstance(tool, (MealsTool, DescribeActionTool)):
            await tool.process_llm_request(
                tool_context=meal_context, llm_request=request
            )
    assert set(request.tools_dict) == {"meals", "describe_action"}
    assert not any(
        isinstance(tool, (MealsTool, DescribeActionTool)) for tool in disabled
    )
    assert not any(getattr(tool, "__name__", "") == "log_meal" for tool in enabled)


async def test_outer_hooks_keep_action_identity_and_redact_payloads(
    meal_context: ToolContext,
    meals: MealsTool,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    args = {
        "action": "log_meal",
        "arguments": {"description": "private-food", "estimated_calories": 100},
    }
    assert is_private_tool(meals)
    assert is_private_tool(DescribeActionTool(meals))
    assert describe_tool(meals.name, args, private=True) == "Logging meal…"
    assert describe_tool("meals", {"action": []}, private=True) == "Working on meals…"
    plugin = PrivacyAwareLoggingPlugin()
    callbacks = LoggingCallbacks()
    callbacks.before_tool(meals, args, meal_context)
    await plugin.before_tool_callback(
        tool=meals, tool_args=args, tool_context=meal_context
    )
    recover_telegram_tool_error(
        meals, args, meal_context, RuntimeError("private-provider-error")
    )
    assert block_telegram_tool_retry(meals, args, meal_context) is not None
    assert (
        block_telegram_tool_retry(
            meals, {"action": "get_calorie_summary", "arguments": {}}, meal_context
        )
        is None
    )
    assert "private-food" not in caplog.text + capsys.readouterr().out
    assert "private-provider-error" not in caplog.text


class SummaryLlm(BaseLlm):
    """External model boundary with deterministic outputs for runner integration."""

    calls: int = 0
    saw_legacy_history: bool = False

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.calls += 1
        assert (
            "meals" in llm_request.tools_dict
            and "log_meal" not in llm_request.tools_dict
        )
        self.saw_legacy_history = any(
            part.function_response and part.function_response.name == "log_meal"
            for content in llm_request.contents
            for part in content.parts or []
        )
        if self.calls == 1:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="summary",
                                name="meals",
                                args={
                                    "action": "get_calorie_summary",
                                    "arguments": {"date": "2026-10-20"},
                                },
                            )
                        )
                    ],
                )
            )
        else:
            responses = [
                part.function_response
                for content in llm_request.contents
                for part in content.parts or []
                if part.function_response and part.function_response.name == "meals"
            ]
            response = responses[-1].response
            assert response is not None and response["summary"]["total_calories"] == 200
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="200 calories logged.")]
                )
            )


async def test_runner_continues_completed_legacy_history_with_compact_tools(
    meal_container: AppContainer,
    meal_context: ToolContext,
) -> None:
    old_result = await log_meal(
        meal_context, description="Old meal", estimated_calories=200, date="2026-10-20"
    )
    model = SummaryLlm(model="deterministic-meals")
    agent = LlmAgent(name="meals_test", model=model)
    agent.tools.extend(create_meal_tools())
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="meals_test", user_id=USER)
    for part in [
        types.Part(
            function_call=types.FunctionCall(
                id="old",
                name="log_meal",
                args={"description": "Old meal", "estimated_calories": 200},
            )
        ),
        types.Part(
            function_response=types.FunctionResponse(
                id="old", name="log_meal", response=old_result
            )
        ),
    ]:
        await sessions.append_event(
            session,
            Event(
                author=agent.name,
                invocation_id="old-turn",
                content=types.Content(
                    role="model" if part.function_call else "user", parts=[part]
                ),
            ),
        )
    runner = Runner(
        app=App(name="meals_test", root_agent=agent, plugins=[DomainPolicyPlugin()]),
        session_service=sessions,
    )
    events = [
        event
        async for event in runner.run_async(
            user_id=USER,
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text="Show my calories")]
            ),
        )
    ]
    assert model.calls == 2 and model.saw_legacy_history
    assert any(
        event.is_final_response()
        and event.content is not None
        and event.content.parts
        and event.content.parts[0].text == "200 calories logged."
        for event in events
    )
    assert (
        await meal_container.calorie_storage.get_daily_summary(USER, "2026-10-20")
    ).entry_count == 1


def test_compact_meal_schemas_reduce_actual_provider_payload() -> None:
    """Unused action schemas stay out of the serialized provider tool list."""
    legacy = LlmRequest()
    legacy.append_tools([FunctionTool(function) for function in MEAL_ACTIONS.values()])
    compact = LlmRequest()
    compact.append_tools(create_meal_tools())
    legacy_tools = build_chat_request(legacy, model="google/gemini-2.5-flash").tools
    compact_tools = build_chat_request(compact, model="google/gemini-2.5-flash").tools
    assert len(json.dumps(compact_tools)) < len(json.dumps(legacy_tools)) * 0.3
    assert [tool["function"]["name"] for tool in compact_tools or []] == [
        "meals",
        "describe_action",
    ]
