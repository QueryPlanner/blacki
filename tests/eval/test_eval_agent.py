"""Tests for the ADK CLI evaluation adapter."""

from unittest.mock import MagicMock

import pytest
from google.adk.agents import LlmAgent

from eval.blacki_eval.agent import (
    _callback_list,
    _ensure_eval_container,
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
