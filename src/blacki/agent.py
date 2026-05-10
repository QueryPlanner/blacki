"""ADK LlmAgent configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.plugins.logging_plugin import LoggingPlugin

from .adk_runtime import DeepSeekReasoningPlugin
from .callbacks import (
    LoggingCallbacks,
    notify_telegram_after_model,
    notify_telegram_before_tool,
    telegram_tool_notifications_enabled,
)
from .prompt import (
    return_description_root,
    return_global_instruction,
    return_instruction_root,
)
from .registry import build_tool_config_from_env, build_tools

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.models.llm_request import LlmRequest

logger = logging.getLogger(__name__)

logging_callbacks = LoggingCallbacks()


class TelegramModelOverridePlugin(BasePlugin):
    """Override the model dynamically based on session state.

    This is used by the Telegram bot to steer the model on a per-chat basis.
    """

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        if not callback_context.session:
            return

        model_override = callback_context.session.state.get("telegram_model_override")
        if not model_override:
            return

        logger.info("Overriding model for Telegram chat to: %s", model_override)

        # Normalize to litellm string format if OpenRouter API key is set
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            model_override = _normalize_model_for_openrouter(model_override)

        llm_request.model = model_override


def _find_and_load_dotenv() -> None:
    """Load a nearby ``.env`` so ``ROOT_AGENT_MODEL`` is set before we read it.

    The ADK agent loader calls ``load_dotenv_for_agent`` before importing this
    module, but other import paths (tests, tooling) may import ``agent`` first.
    Loading here avoids defaulting to native Gemini without ``GOOGLE_API_KEY``.
    """
    here = Path(__file__).resolve().parent
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break


def _normalize_model_for_openrouter(model_name: str) -> str:
    """Map common IDs to OpenRouter/LiteLLM form when routing via OpenRouter only.

    Examples:
        ``gemini-2.5-flash`` → ``openrouter/google/gemini-2.5-flash``
        ``google/gemini-2.0-flash-001`` → ``openrouter/google/gemini-2.0-flash-001``
        ``openrouter/openai/gpt-oss-120b`` → unchanged
    """
    normalized = model_name.strip()
    lower = normalized.lower()
    if lower.startswith("openrouter/"):
        return normalized
    if "/" in normalized:
        return f"openrouter/{normalized}"
    if normalized.startswith("gemini-"):
        return f"openrouter/google/{normalized}"
    return normalized


def _build_model() -> str | LiteLlm:
    """Build the model configuration from environment variables.

    Returns:
        Either a string model name or a LiteLlm instance.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")
    model: str | LiteLlm = model_name

    use_litellm = openrouter_api_key is not None or "/" in model_name.lower()
    if openrouter_api_key:
        model_name = _normalize_model_for_openrouter(model_name)

    if use_litellm:
        try:
            from google.adk.models import LiteLlm

            litellm_kwargs: dict[str, Any] = {}
            if model_name.lower().startswith("openrouter/") and openrouter_api_key:
                litellm_kwargs["api_key"] = openrouter_api_key

            logger.info("Using LiteLlm for model: %s", model_name)
            return LiteLlm(model=model_name, **litellm_kwargs)
        except ImportError:
            logger.warning(
                "LiteLlm not available, falling back to string model name. "
                "OpenRouter models may not work."
            )

    return model


def create_agent() -> LlmAgent:
    """Create and configure the root agent.

    Uses factory-based tool registration and explicit configuration.
    This function should be called after environment is configured.

    Returns:
        Configured LlmAgent instance.
    """
    tool_config = build_tool_config_from_env()
    agent_tools = build_tools(tool_config)

    before_tool_callbacks: list[Any] = [logging_callbacks.before_tool]
    after_model_callbacks: list[Any] = [logging_callbacks.after_model]

    if telegram_tool_notifications_enabled():
        logger.info(
            "Telegram tool notifications enabled; "
            "registering before_tool and after_model callbacks"
        )
        before_tool_callbacks.append(notify_telegram_before_tool)
        after_model_callbacks.append(notify_telegram_after_model)

    return LlmAgent(
        name="blacki",
        description=return_description_root(),
        before_agent_callback=logging_callbacks.before_agent,
        after_agent_callback=logging_callbacks.after_agent,
        model=_build_model(),
        instruction=return_instruction_root(),
        tools=agent_tools,
        before_model_callback=logging_callbacks.before_model,
        after_model_callback=after_model_callbacks,
        before_tool_callback=before_tool_callbacks,
        after_tool_callback=logging_callbacks.after_tool,
    )


def create_app(agent: LlmAgent | None = None) -> App:
    """Create the ADK App with configured plugins.

    Args:
        agent: Optional pre-configured agent. If not provided, creates one.

    Returns:
        Configured App instance.
    """
    if agent is None:
        agent = create_agent()

    return App(
        name="blacki",
        root_agent=agent,
        plugins=[
            TelegramModelOverridePlugin(name="telegram_model_override"),
            GlobalInstructionPlugin(return_global_instruction),
            LoggingPlugin(),
            DeepSeekReasoningPlugin(name="deepseek_reasoning"),
        ],
        events_compaction_config=None,
        context_cache_config=None,
        resumability_config=None,
    )


_find_and_load_dotenv()

root_agent = create_agent()

app = create_app(root_agent)

# ADK requires module-level `root_agent` and `app` globals for agent discovery.
# These are created at import time to support the ADK runtime's module scanning.
# For explicit initialization control, use `create_agent()` and `create_app()`
# directly from an entry point.
