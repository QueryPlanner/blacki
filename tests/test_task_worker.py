"""Tests for the opt-in ADK task worker."""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

import pytest
from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from blacki.agent import (
    TASK_WORKER_NAME,
    _task_worker_enabled,
    create_agent,
)
from blacki.registry import ToolConfig
from blacki.sandbox.config import SANDBOX_STATE_KEY


def _tool_capability(tool: object) -> str:
    """Return a stable capability name for a function, tool, or toolset."""
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    if name:
        return str(name)
    tool_type = type(tool)
    return f"{tool_type.__module__}.{tool_type.__qualname__}"


def _task_worker_test_config() -> ToolConfig:
    """Build a deterministic config with sandbox and skill toolsets enabled."""
    skills_dir = Path(__file__).parents[1] / "src" / "blacki" / "skills"
    return ToolConfig(
        google_maps_routes_api_key="routes-key",
        sandbox_enabled=True,
        skills_dir=skills_dir,
        weather_enabled=False,
    )


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes "])
def test_task_worker_feature_flag_accepts_explicit_true_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """The worker should require a documented affirmative flag value."""
    monkeypatch.setenv("TASK_WORKER_ENABLED", value)

    assert _task_worker_enabled() is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "enabled"])
def test_task_worker_feature_flag_defaults_to_disabled(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """Missing and unrecognized values should preserve the single-agent setup."""
    if value is None:
        monkeypatch.delenv("TASK_WORKER_ENABLED", raising=False)
    else:
        monkeypatch.setenv("TASK_WORKER_ENABLED", value)

    assert _task_worker_enabled() is False


def test_disabled_task_worker_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default agent should not expose a delegation tool."""
    monkeypatch.setenv("TASK_WORKER_ENABLED", "false")

    with patch(
        "blacki.agent.build_tool_config_from_env",
        return_value=_task_worker_test_config(),
    ):
        agent = create_agent()

    assert agent.sub_agents == []
    assert TASK_WORKER_NAME not in {_tool_capability(tool) for tool in agent.tools}


def test_enabled_task_worker_has_equivalent_isolated_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker should match capabilities without sharing mutable toolsets."""
    monkeypatch.setenv("TASK_WORKER_ENABLED", "true")

    with patch(
        "blacki.agent.build_tool_config_from_env",
        return_value=_task_worker_test_config(),
    ):
        agent = create_agent()

    assert len(agent.sub_agents) == 1
    worker = agent.sub_agents[0]
    assert isinstance(worker, LlmAgent)
    assert worker.name == TASK_WORKER_NAME
    assert worker.mode == "task"
    assert worker.parent_agent is agent
    assert worker.sub_agents == []
    assert "Do not create or delegate to another worker" in str(worker.instruction)

    root_capabilities = Counter(_tool_capability(tool) for tool in agent.tools)
    worker_capabilities = Counter(_tool_capability(tool) for tool in worker.tools)
    root_capabilities.pop(TASK_WORKER_NAME)
    worker_capabilities.pop("finish_task")

    assert root_capabilities == worker_capabilities
    assert root_capabilities["get_route_estimate"] == 1
    assert root_capabilities["compare_route_scenarios"] == 1

    root_toolsets = [tool for tool in agent.tools if isinstance(tool, BaseToolset)]
    worker_toolsets = [tool for tool in worker.tools if isinstance(tool, BaseToolset)]
    assert len(root_toolsets) == len(worker_toolsets) == 1
    assert root_toolsets[0] is not worker_toolsets[0]

    root_preload = next(
        tool for tool in agent.tools if _tool_capability(tool) == "preload_memory"
    )
    worker_preload = next(
        tool for tool in worker.tools if _tool_capability(tool) == "preload_memory"
    )
    assert root_preload is not worker_preload


class DelegationLlm(BaseLlm):
    """Deterministic model that drives root-to-worker task delegation."""

    root_calls: int = 0
    worker_calls: int = 0
    saw_task_result: bool = False

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Emit a fixed sequence of function calls for each agent."""
        _ = stream
        tool_names = set(llm_request.tools_dict)

        if "finish_task" in tool_names:
            self.worker_calls += 1
            if self.worker_calls == 1:
                yield _function_call_response(
                    call_id="worker-probe",
                    name="inspect_sandbox_state",
                    args={"origin": "worker"},
                )
                return

            yield _function_call_response(
                call_id="worker-finish",
                name="finish_task",
                args={"result": "worker completed"},
            )
            return

        self.root_calls += 1
        if self.root_calls == 1:
            yield _function_call_response(
                call_id="delegate-1",
                name=TASK_WORKER_NAME,
                args={"request": "Inspect the shared sandbox state"},
            )
            return

        if self.root_calls == 2:
            self.saw_task_result = any(
                part.function_response
                and part.function_response.name == TASK_WORKER_NAME
                for content in llm_request.contents
                for part in content.parts or []
            )
            yield _function_call_response(
                call_id="root-probe",
                name="inspect_sandbox_state",
                args={"origin": "root"},
            )
            return

        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Delegated task complete.")],
            )
        )


def _function_call_response(
    *, call_id: str, name: str, args: dict[str, str]
) -> LlmResponse:
    """Build one deterministic model function-call response."""
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=call_id,
                        name=name,
                        args=args,
                    )
                )
            ],
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_state", "expected_sandbox_id"),
    [
        ({SANDBOX_STATE_KEY: "existing-sandbox"}, "existing-sandbox"),
        ({}, "worker-created-sandbox"),
    ],
)
async def test_task_worker_shares_session_sandbox_state_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    initial_state: dict[str, str],
    expected_sandbox_id: str,
) -> None:
    """A real Runner delegation should preserve sandbox state in both directions."""
    observations: list[tuple[str, str, int]] = []

    def inspect_sandbox_state(origin: str, tool_context: ToolContext) -> dict[str, str]:
        sandbox_id = tool_context.state.get(SANDBOX_STATE_KEY)
        if sandbox_id is None:
            sandbox_id = "worker-created-sandbox"
            tool_context.state[SANDBOX_STATE_KEY] = sandbox_id
        observations.append((origin, str(sandbox_id), id(tool_context.session.state)))
        return {"sandbox_id": str(sandbox_id)}

    model = DelegationLlm(model="delegation-test")
    monkeypatch.setenv("TASK_WORKER_ENABLED", "true")

    with (
        patch(
            "blacki.agent.build_tool_config_from_env",
            return_value=ToolConfig(weather_enabled=False),
        ),
        patch(
            "blacki.agent.build_tools",
            side_effect=[
                [inspect_sandbox_state],
                [inspect_sandbox_state],
            ],
        ),
        patch("blacki.agent._build_model", return_value=model),
        patch("blacki.agent.telegram_tool_notifications_enabled", return_value=False),
    ):
        agent = create_agent()

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="task-worker-test",
        user_id="user",
        session_id="session",
        state=initial_state,
    )
    runner = Runner(
        agent=agent,
        app_name="task-worker-test",
        session_service=session_service,
    )

    events = [
        event
        async for event in runner.run_async(
            user_id="user",
            session_id="session",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Delegate this sandbox inspection.")],
            ),
        )
    ]

    assert model.root_calls == 3
    assert model.worker_calls == 2
    assert model.saw_task_result is True
    assert [(origin, sandbox_id) for origin, sandbox_id, _ in observations] == [
        ("worker", expected_sandbox_id),
        ("root", expected_sandbox_id),
    ]
    assert observations[0][2] == observations[1][2]
    assert any(
        event.author == TASK_WORKER_NAME and event.isolation_scope == "delegate-1"
        for event in events
    )
    assert any(
        event.content
        and any(
            part.text == "Delegated task complete."
            for part in event.content.parts or []
        )
        for event in events
    )

    persisted_session = await session_service.get_session(
        app_name="task-worker-test",
        user_id="user",
        session_id="session",
    )
    assert persisted_session is not None
    assert persisted_session.state[SANDBOX_STATE_KEY] == expected_sandbox_id
