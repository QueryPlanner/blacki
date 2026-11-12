"""ADK LlmAgent configuration."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .models.catalog import get_model
from .models.factory import build_model
from .models.inference import (
    InferenceProfile,
    apply_inference_profile,
    get_active_inference_profile,
    inference_profile_from_environment,
    load_inference_profile,
)
from .models.openai_items import without_openai_metadata
from .models.openai_responses import SESSION_LABEL_KEY, OpenAIResponsesLlm
from .models.openai_summarizer import OpenAIEventsSummarizer
from .observability.lifecycle import LoggingCallbacks
from .prompts.instructions import (
    TemporalContextPlugin,
    return_description_root,
    return_instruction_root,
)
from .telegram.progress_callbacks import (
    block_telegram_tool_retry,
    clear_telegram_tool_failure,
    notify_telegram_after_agent,
    notify_telegram_after_model,
    notify_telegram_before_tool,
    recover_telegram_tool_error,
    telegram_live_tool_progress_enabled,
)
from .tools.registry import build_tool_config_from_env, build_tools

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.sessions.session import Session

    from .models.catalog import ModelEntry

logger = logging.getLogger(__name__)

logging_callbacks = LoggingCallbacks()
AUTO_COMPACTION_TOKEN_THRESHOLD = 200_000
AUTO_COMPACTION_EVENT_RETENTION_SIZE = 8
# ADK 2.5.0 requires the sliding-window fields even for token-only
# configuration. Keep the interval above any realistic session length so the
# token threshold remains the only practical compaction trigger.
_AUTO_COMPACTION_INTERVAL_SENTINEL = 1_000_000_000
# Per-process secret for session labels (design D7). It is never stored, so a
# label cannot be linked across restarts and reveals nothing about session ids.
_SESSION_LABEL_SECRET: Final = secrets.token_bytes(32)
# Saved selections that mean "use the default model".
_DEFAULT_SELECTIONS: Final = frozenset({"", "default"})


class TelegramModelOverridePlugin(BasePlugin):
    """Apply each Telegram chat's model and reasoning choice to every model call.

    ``root_is_openai`` says whether the root agent uses the OpenAI adapter. Only
    then can a saved catalog model be used. Any other saved selection falls back
    to the root model, so a stale selection cannot fail a turn.
    """

    def __init__(
        self,
        name: str = "telegram_model_override",
        *,
        root_is_openai: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._root_is_openai = root_is_openai

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        profile = get_active_inference_profile()
        if profile is None:
            profile = await self._load_fallback_profile(callback_context)

        profile = self._select_model(profile)
        effective_model = profile.model or llm_request.model or ""
        entry = get_model(effective_model) if self._root_is_openai else None
        if entry is None:
            profile = _native_profile(profile, effective_model)
        else:
            profile = _catalog_profile(profile, entry)

        apply_inference_profile(llm_request, profile)
        if entry is None:
            _remove_openai_data(llm_request)
        else:
            _label_openai_session(callback_context, llm_request)

    def _select_model(self, profile: InferenceProfile) -> InferenceProfile:
        """Use a saved model only when it is a catalog model and root is OpenAI."""
        requested = profile.model
        if requested is None:
            return profile
        if requested.strip().lower() in _DEFAULT_SELECTIONS:
            return profile.model_copy(update={"model": None})
        entry = get_model(requested) if self._root_is_openai else None
        if entry is None:
            logger.warning(
                "Ignoring saved model selection that the root model cannot serve; "
                "using the default model."
            )
            return profile.model_copy(update={"model": None})
        logger.info("Overriding model for Telegram chat to: %s", entry.id)
        return profile.model_copy(update={"model": entry.id})

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


def _native_profile(profile: InferenceProfile, model: str) -> InferenceProfile:
    """Drop reasoning controls, which native Gemini requests do not carry."""
    if profile.reasoning is None:
        return profile
    logger.info("Ignoring reasoning controls for native model: %s", model)
    return profile.model_copy(update={"reasoning": None})


def _catalog_profile(profile: InferenceProfile, entry: ModelEntry) -> InferenceProfile:
    """Drop an effort the catalog model does not accept, so its default applies."""
    reasoning = profile.reasoning
    if reasoning is None or reasoning.effort is None:
        return profile
    if reasoning.effort.value in entry.efforts:
        return profile
    logger.warning(
        "Ignoring reasoning effort %s: %s does not accept it; using its default.",
        reasoning.effort.value,
        entry.id,
    )
    return profile.model_copy(update={"reasoning": None})


def _label_openai_session(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Label an OpenAI request with a non-reversible key for its session (D7)."""
    labels = dict(llm_request.config.labels or {})
    labels[SESSION_LABEL_KEY] = _session_label(callback_context.session)
    llm_request.config = llm_request.config.model_copy(update={"labels": labels})


def _session_label(session: Session) -> str:
    """Return the HMAC of the session identity, keyed by the process secret."""
    identity = json.dumps([session.app_name, session.user_id, session.id])
    digest = hmac.new(_SESSION_LABEL_SECRET, identity.encode(), hashlib.sha256)
    return digest.hexdigest()


def _remove_openai_data(llm_request: LlmRequest) -> None:
    """Keep OpenAI items and session labels out of a native Gemini request (D11)."""
    labels = llm_request.config.labels or {}
    if SESSION_LABEL_KEY in labels:
        remaining = {k: v for k, v in labels.items() if k != SESSION_LABEL_KEY}
        llm_request.config = llm_request.config.model_copy(update={"labels": remaining})
    llm_request.contents = _without_openai_parts(llm_request.contents)


def _without_openai_parts(contents: list[types.Content]) -> list[types.Content]:
    """Remove stored OpenAI items and empty thought parts from every turn.

    A turn left with no parts is dropped. Thought signatures are kept because
    Gemini uses them to continue its own reasoning.
    """
    cleaned: list[types.Content] = []
    for content in contents:
        if not content.parts:
            cleaned.append(content)
            continue
        parts = [
            part
            for part in (without_openai_metadata(part) for part in content.parts)
            if not _is_empty_thought(part)
        ]
        if parts:
            cleaned.append(content.model_copy(update={"parts": parts}))
    return cleaned


def _is_empty_thought(part: types.Part) -> bool:
    """Return whether ``part`` is a thought with no text and no other content."""
    if part.thought is not True or part.text or part.thought_signature:
        return False
    remaining = part.model_dump(
        exclude_none=True,
        exclude={"thought", "text", "thought_signature", "part_metadata"},
    )
    return not remaining


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


def create_agent(*, include_user_scoped_tools: bool = False) -> LlmAgent:
    """Create and configure the root agent.

    Uses factory-based tool registration and explicit configuration.
    User-scoped tools are excluded by default so the ADK HTTP surface cannot
    expose transport-bound credentials. The Telegram runtime opts in
    explicitly after it has authenticated updates through the bot API.

    Returns:
        Configured LlmAgent instance.
    """
    tool_config = build_tool_config_from_env()
    agent_tools = build_tools(
        tool_config,
        include_user_scoped_tools=include_user_scoped_tools,
    )
    if include_user_scoped_tools:
        from blacki.wiki import wiki_configured
        from blacki.wiki.tools import WikiToolset

        if wiki_configured():
            agent_tools.append(WikiToolset())

    before_tool_callbacks: list[Any] = [logging_callbacks.before_tool]
    after_model_callbacks: list[Any] = [logging_callbacks.after_model]
    after_agent_callbacks: list[Any] = [logging_callbacks.after_agent]

    if telegram_live_tool_progress_enabled():
        before_tool_callbacks.append(notify_telegram_before_tool)
        after_model_callbacks.append(notify_telegram_after_model)
        after_agent_callbacks.append(notify_telegram_after_agent)

    if include_user_scoped_tools:
        before_tool_callbacks.append(block_telegram_tool_retry)
        after_agent_callbacks.append(clear_telegram_tool_failure)

    telegram_tool_error_callback = (
        recover_telegram_tool_error if include_user_scoped_tools else None
    )

    return LlmAgent(
        name="blacki",
        description=return_description_root(),
        before_agent_callback=logging_callbacks.before_agent,
        after_agent_callback=after_agent_callbacks,
        model=build_model(),
        instruction=return_instruction_root(),
        tools=agent_tools,
        before_model_callback=logging_callbacks.before_model,
        after_model_callback=after_model_callbacks,
        before_tool_callback=before_tool_callbacks,
        after_tool_callback=logging_callbacks.after_tool,
        on_tool_error_callback=telegram_tool_error_callback,
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

    from blacki.declarative_db.plugin import DeclarativeDbPlugin
    from blacki.sandbox.config import sandbox_enabled
    from blacki.sandbox.images import SandboxMultimodalToolResultsPlugin
    from blacki.user_files import UserFilesPromptPlugin, user_files_enabled
    from blacki.wiki.tools import WikiMemoryPlugin, WikiPolicyPlugin, WikiToolset

    wiki_tools_enabled = any(isinstance(tool, WikiToolset) for tool in agent.tools)
    openai_root = agent.model if isinstance(agent.model, OpenAIResponsesLlm) else None

    sandbox_tools_enabled = sandbox_enabled()
    plugins: list[BasePlugin] = [
        *(
            [SandboxMultimodalToolResultsPlugin(name="sandbox_multimodal_results")]
            if sandbox_tools_enabled
            else []
        ),
        TelegramModelOverridePlugin(
            name="telegram_model_override",
            root_is_openai=openai_root is not None,
        ),
        *([WikiPolicyPlugin(name="private_wiki")] if wiki_tools_enabled else []),
        DeclarativeDbPlugin(name="declarative_db"),
        *([UserFilesPromptPlugin(name="user_files")] if user_files_enabled() else []),
        *([WikiMemoryPlugin(name="owner_memory")] if wiki_tools_enabled else []),
        TemporalContextPlugin(),
    ]

    return App(
        name="blacki",
        root_agent=agent,
        plugins=plugins,
        events_compaction_config=EventsCompactionConfig(
            summarizer=(
                OpenAIEventsSummarizer(llm=openai_root)
                if openai_root is not None
                else None
            ),
            compaction_interval=_AUTO_COMPACTION_INTERVAL_SENTINEL,
            overlap_size=0,
            token_threshold=AUTO_COMPACTION_TOKEN_THRESHOLD,
            event_retention_size=AUTO_COMPACTION_EVENT_RETENTION_SIZE,
        ),
        context_cache_config=None,
        resumability_config=None,
    )


_find_and_load_dotenv()

root_agent = create_agent(include_user_scoped_tools=False)

app = create_app(root_agent)

# ADK requires module-level `root_agent` and `app` globals for agent discovery.
# These are created at import time to support the ADK runtime's module scanning.
# For explicit initialization control, use `create_agent()` and `create_app()`
# directly from an entry point.
