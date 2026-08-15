"""Adapt Blacki's App plugins for ADK CLI root-agent evaluation."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin

from blacki.agent import create_agent
from blacki.container import close_container, get_container, init_container
from blacki.declarative_db.plugin import (
    DeclarativeDbPlugin,
    StoredPreferencesPlugin,
)
from blacki.prompt import (
    DomainPolicyPlugin,
    ResponsePolicyPlugin,
    return_global_instruction,
)

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
    global_instruction = GlobalInstructionPlugin(return_global_instruction)
    domain_policy = DomainPolicyPlugin()
    declarative_db = DeclarativeDbPlugin()
    stored_preferences = StoredPreferencesPlugin()
    response_policy = ResponsePolicyPlugin()

    async def before_tool_policy(
        *, tool: Any, args: dict[str, Any], tool_context: Any
    ) -> dict[str, object] | None:
        return await domain_policy.before_tool_callback(
            tool=tool,
            tool_args=args,
            tool_context=tool_context,
        )

    async def after_tool_policy(
        *,
        tool: Any,
        args: dict[str, Any],
        tool_context: Any,
        tool_response: dict[str, object],
    ) -> None:
        await domain_policy.after_tool_callback(
            tool=tool,
            tool_args=args,
            tool_context=tool_context,
            result=tool_response,
        )

    before_model_callbacks: list[Any] = [
        *_callback_list(eval_agent.before_model_callback),
        global_instruction.before_model_callback,
        domain_policy.before_model_callback,
        declarative_db.before_model_callback,
        stored_preferences.before_model_callback,
    ]
    after_model_callbacks: list[Any] = [
        response_policy.after_model_callback,
        *_callback_list(eval_agent.after_model_callback),
    ]
    before_tool_callbacks: list[Any] = [
        before_tool_policy,
        *_callback_list(eval_agent.before_tool_callback),
    ]
    after_tool_callbacks: list[Any] = [
        after_tool_policy,
        *_callback_list(eval_agent.after_tool_callback),
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
    eval_agent.after_model_callback = after_model_callbacks
    eval_agent.before_tool_callback = before_tool_callbacks
    eval_agent.after_tool_callback = after_tool_callbacks
    return eval_agent


root_agent = create_eval_agent(create_agent(include_user_scoped_tools=True))
