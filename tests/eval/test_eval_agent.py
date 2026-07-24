"""Tests for the ADK CLI evaluation adapter."""

import os
from unittest.mock import MagicMock

import pytest
from google.adk.agents import LlmAgent

from eval.blacki_eval.agent import (
    _callback_list,
    _configure_route_eval_boundary,
    _ensure_eval_container,
    _route_eval_compute_routes,
    create_eval_agent,
)


def test_callback_list_normalizes_adk_callback_fields() -> None:
    callback = MagicMock()

    assert _callback_list(None) == []
    assert _callback_list(callback) == [callback]
    assert _callback_list([callback]) == [callback]


def test_eval_agent_attaches_policy_callbacks() -> None:
    agent = LlmAgent(name="eval_test", model="gemini-2.5-flash")

    result = create_eval_agent(agent)

    assert result is agent
    assert len(result.canonical_before_agent_callbacks) == 1
    assert len(result.canonical_after_agent_callbacks) == 1
    assert len(result.canonical_before_model_callbacks) == 4
    assert len(result.canonical_after_model_callbacks) == 1
    assert len(result.canonical_before_tool_callbacks) == 1
    assert len(result.canonical_after_tool_callbacks) == 1


@pytest.mark.asyncio
async def test_eval_container_requires_explicit_sqlite_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SQLITE_PATH", raising=False)
    monkeypatch.setattr(
        "eval.blacki_eval.agent.get_container",
        MagicMock(side_effect=RuntimeError("not initialized")),
    )

    try:
        await _ensure_eval_container(callback_context=MagicMock())
    except RuntimeError as error:
        assert str(error) == "SQLITE_PATH is required for prompt evaluations"
    else:
        raise AssertionError("missing SQLITE_PATH should fail")


@pytest.mark.asyncio
async def test_route_eval_boundary_is_deterministic() -> None:
    result = await _route_eval_compute_routes(
        {"origin": {"address": "private"}},
        "eval-only",
    )

    assert result == {
        "routes": [
            {
                "distanceMeters": 12500,
                "duration": "1800s",
                "staticDuration": "1200s",
            }
        ]
    }


def test_route_eval_boundary_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blacki.routes import tools as route_tools

    original = route_tools.compute_routes
    monkeypatch.delenv("BLACKI_EVAL_ROUTES", raising=False)
    monkeypatch.delenv("GOOGLE_MAPS_ROUTES_API_KEY", raising=False)

    _configure_route_eval_boundary()

    assert route_tools.compute_routes is original
    assert "GOOGLE_MAPS_ROUTES_API_KEY" not in os.environ


def test_route_eval_boundary_replaces_only_maps_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blacki.routes import tools as route_tools

    original = route_tools.compute_routes
    monkeypatch.setenv("BLACKI_EVAL_ROUTES", "true")
    monkeypatch.delenv("GOOGLE_MAPS_ROUTES_API_KEY", raising=False)
    monkeypatch.setattr(route_tools, "compute_routes", original)

    _configure_route_eval_boundary()

    assert route_tools.compute_routes is _route_eval_compute_routes
    assert os.environ["GOOGLE_MAPS_ROUTES_API_KEY"] == "eval-only"
