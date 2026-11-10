"""Adapt Blacki's App plugins for ADK CLI root-agent evaluation."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from google.adk.agents import LlmAgent

from blacki.agent import create_agent
from blacki.container import close_container, get_container, init_container
from blacki.declarative_db.plugin import (
    DeclarativeDbPlugin,
    StoredPreferencesPlugin,
)
from blacki.prompts.instructions import TemporalContextPlugin

_container_lock = asyncio.Lock()
_active_invocations = 0


async def _ensure_eval_container(*, callback_context: Any) -> None:
    """Initialize the real storage container once for stateful eval cases."""
    global _active_invocations
    eval_sender = callback_context.state.get("telegram_sender_user_id_for_eval")
    if eval_sender is not None:
        callback_context.state["temp:telegram_sender_user_id"] = str(eval_sender)
    async with _container_lock:
        try:
            get_container()
        except RuntimeError:
            sqlite_path = os.environ.get("SQLITE_PATH")
            if not sqlite_path:
                raise RuntimeError(
                    "SQLITE_PATH is required for prompt evaluations"
                ) from None
            container = await init_container(sqlite_path)
            await container.initialize_all_storages()
        _active_invocations += 1


async def _release_eval_container(*, callback_context: Any) -> None:
    """Close the evaluation container after the final concurrent case."""
    global _active_invocations
    del callback_context
    async with _container_lock:
        _active_invocations = max(0, _active_invocations - 1)
        if _active_invocations == 0:
            await close_container()


def _callback_list(callback: Any) -> list[Any]:
    """Normalize an ADK callback field without changing callback order."""
    if callback is None:
        return []
    return callback if isinstance(callback, list) else [callback]


def create_eval_agent(agent: LlmAgent | None = None) -> LlmAgent:
    """Attach production App policy hooks to an ADK CLI-evaluated agent."""
    eval_agent = agent or create_agent()
    temporal_context = TemporalContextPlugin()
    declarative_db = DeclarativeDbPlugin()
    stored_preferences = StoredPreferencesPlugin()

    before_model_callbacks: list[Any] = [
        *_callback_list(eval_agent.before_model_callback),
        temporal_context.before_model_callback,
        declarative_db.before_model_callback,
        stored_preferences.before_model_callback,
    ]
    before_agent_callbacks: list[Any] = [
        _ensure_eval_container,
        *_callback_list(eval_agent.before_agent_callback),
    ]
    after_agent_callbacks: list[Any] = [
        *_callback_list(eval_agent.after_agent_callback),
        _release_eval_container,
    ]
    eval_agent.before_agent_callback = before_agent_callbacks
    eval_agent.after_agent_callback = after_agent_callbacks
    eval_agent.before_model_callback = before_model_callbacks
    return eval_agent


root_agent = create_eval_agent(create_agent(include_user_scoped_tools=True))
