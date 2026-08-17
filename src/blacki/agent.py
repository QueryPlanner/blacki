"""ADK LlmAgent configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin

from .callbacks import (
    LoggingCallbacks,
    notify_telegram_after_agent,
    notify_telegram_after_model,
    notify_telegram_before_tool,
    telegram_tool_notifications_enabled,
)
from .inference import (
    InferenceProfile,
    apply_inference_profile,
    get_active_inference_profile,
    inference_profile_from_environment,
    load_inference_profile,
)
from .privacy import (
    PrivacyAwareLoggingPlugin,
    configure_private_tool_privacy,
    private_tool_privacy_enabled,
)
from .prompt import (
    DomainPolicyPlugin,
    ResponsePolicyPlugin,
    return_description_root,
    return_global_instruction,
    return_instruction_root,
    return_instruction_task_worker,
)
from .registry import build_tool_config_from_env, build_tools

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.models.llm_request import LlmRequest

logger = logging.getLogger(__name__)

logging_callbacks = LoggingCallbacks()
TASK_WORKER_NAME = "task_worker"
TASK_WORKER_ENABLED_VALUES = frozenset({"1", "true", "yes"})


class TelegramModelOverridePlugin(BasePlugin):
    """Override the model dynamically based on session state.

    This is used by the Telegram bot to steer the model on a per-chat basis.
    """

    def __init__(self, name: str = "telegram_model_override") -> None:
        super().__init__(name=name)
        self.normalize_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        profile = get_active_inference_profile()
        if profile is None:
            profile = await self._load_fallback_profile(callback_context)

        if profile.model:
            model_name = profile.model
            if self.normalize_openrouter:
                model_name = _normalize_model_for_openrouter(model_name)
            profile = profile.model_copy(update={"model": model_name})
            logger.info("Overriding model for Telegram chat to: %s", model_name)

        effective_model = profile.model or llm_request.model or ""
        is_openrouter_request = (
            self.normalize_openrouter
            or effective_model.lower().startswith("openrouter/")
        )
        if profile.reasoning is not None and not is_openrouter_request:
            logger.info(
                "Ignoring OpenRouter reasoning controls for native model: %s",
                effective_model,
            )
            profile = profile.model_copy(update={"reasoning": None})

        apply_inference_profile(llm_request, profile)

    async def _load_fallback_profile(
        self,
        callback_context: CallbackContext,
    ) -> InferenceProfile:
        """Resolve preferences when a transport did not provide a snapshot."""
        chat_id: object | None = None
        if callback_context.session:
            chat_id = callback_context.session.state.get("telegram_chat_id")

        if chat_id:
            from .utils.preferences import get_preferences_storage

            try:
                storage = get_preferences_storage()
                return await load_inference_profile(storage, str(chat_id))
            except Exception:
                logger.exception("Failed to fetch inference preferences")

        return inference_profile_from_environment()


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


def _task_worker_enabled() -> bool:
    """Return whether the default-on delegated task worker is enabled."""
    return (
        os.getenv("TASK_WORKER_ENABLED", "true").strip().lower()
        in TASK_WORKER_ENABLED_VALUES
    )


def create_agent(*, include_user_scoped_tools: bool = False) -> LlmAgent:
    """Create and configure the root agent.

    Uses factory-based tool registration and explicit configuration.
    User-scoped tools are excluded by default so the ADK HTTP surface cannot
    expose transport-bound credentials. The Telegram runtime opts in
    explicitly after it has authenticated updates through the bot API.

    Returns:
        Configured LlmAgent instance.
    """
    from google.adk.tools.preload_memory_tool import PreloadMemoryTool

    tool_config = build_tool_config_from_env()
    agent_tools = build_tools(
        tool_config,
        include_user_scoped_tools=include_user_scoped_tools,
    )
    agent_tools.append(PreloadMemoryTool())

    before_tool_callbacks: list[Any] = [logging_callbacks.before_tool]
    after_model_callbacks: list[Any] = [logging_callbacks.after_model]
    after_agent_callbacks: list[Any] = [logging_callbacks.after_agent]

    if telegram_tool_notifications_enabled():
        logger.info(
            "Telegram tool notifications enabled; "
            "registering before_tool, after_model, and after_agent callbacks"
        )
        before_tool_callbacks.append(notify_telegram_before_tool)
        after_model_callbacks.append(notify_telegram_after_model)
        after_agent_callbacks.append(notify_telegram_after_agent)

    sub_agents: list[BaseAgent] = []
    if _task_worker_enabled():
        worker_tools = build_tools(tool_config, include_user_scoped_tools=False)
        worker_tools.append(PreloadMemoryTool())
        sub_agents.append(
            LlmAgent(
                name=TASK_WORKER_NAME,
                description=(
                    "Complete one complex delegated task with Blacki's "
                    "non-private tools and shared session sandbox"
                ),
                mode="task",
                before_agent_callback=logging_callbacks.before_agent,
                after_agent_callback=after_agent_callbacks.copy(),
                model=_build_model(),
                instruction=return_instruction_task_worker(),
                tools=worker_tools,
                before_model_callback=logging_callbacks.before_model,
                after_model_callback=after_model_callbacks.copy(),
                before_tool_callback=before_tool_callbacks.copy(),
                after_tool_callback=logging_callbacks.after_tool,
            )
        )

    return LlmAgent(
        name="blacki",
        description=return_description_root(),
        before_agent_callback=logging_callbacks.before_agent,
        after_agent_callback=after_agent_callbacks,
        model=_build_model(),
        instruction=return_instruction_root(),
        tools=agent_tools,
        before_model_callback=logging_callbacks.before_model,
        after_model_callback=after_model_callbacks,
        before_tool_callback=before_tool_callbacks,
        after_tool_callback=logging_callbacks.after_tool,
        sub_agents=sub_agents,
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

    from blacki.declarative_db.plugin import (
        DeclarativeDbPlugin,
        StoredPreferencesPlugin,
    )

    plugins: list[BasePlugin] = [
        TelegramModelOverridePlugin(name="telegram_model_override"),
        GlobalInstructionPlugin(return_global_instruction),
        DomainPolicyPlugin(name="domain_policy"),
        DeclarativeDbPlugin(name="declarative_db"),
        StoredPreferencesPlugin(name="stored_preferences"),
        ResponsePolicyPlugin(name="response_policy"),
    ]
    if not private_tool_privacy_enabled():
        plugins.append(PrivacyAwareLoggingPlugin())
    else:
        logger.info("ADK content logging plugin disabled in private-tool mode")

    return App(
        name="blacki",
        root_agent=agent,
        plugins=plugins,
        events_compaction_config=None,
        context_cache_config=None,
        resumability_config=None,
    )


_find_and_load_dotenv()
configure_private_tool_privacy()

root_agent = create_agent(include_user_scoped_tools=False)

app = create_app(root_agent)

# ADK requires module-level `root_agent` and `app` globals for agent discovery.
# These are created at import time to support the ADK runtime's module scanning.
# For explicit initialization control, use `create_agent()` and `create_app()`
# directly from an entry point.
