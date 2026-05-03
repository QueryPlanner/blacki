"""ADK LlmAgent configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.apps import App
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

if TYPE_CHECKING:
    from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger(__name__)

logging_callbacks = LoggingCallbacks()


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


_find_and_load_dotenv()

# Determine model configuration
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")
model: str | LiteLlm = model_name

# Determine whether the (possibly normalized) model requires LiteLlm.
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
        model = LiteLlm(model=model_name, **litellm_kwargs)
    except ImportError:
        logger.warning(
            "LiteLlm not available, falling back to string model name. "
            "OpenRouter models may not work."
        )

skills_dir = Path(__file__).parent / "skills"


# Build the list of tools (list[Any] due to heterogeneous conditional imports)
agent_tools: list[Any] = []

# Add Brave Search tool if API key is available
brave_search_api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
if brave_search_api_key:
    try:
        from .tools import brave_search

        agent_tools.append(brave_search)
        logger.info("Brave Search tool enabled")
    except ImportError as e:
        logger.warning("Failed to load Brave Search tool: %s", e)

# Add Reminder tools if database is configured
database_url = os.getenv("DATABASE_URL", "").strip()
if database_url:
    try:
        from .reminders import cancel_reminder, list_reminders, schedule_reminder

        agent_tools.extend([schedule_reminder, list_reminders, cancel_reminder])
        logger.info("Reminder tools enabled")
    except ImportError as e:
        logger.warning("Failed to load Reminder tools: %s", e)

# Add Calorie tools if database is configured
if database_url:
    try:
        from .calories import (
            delete_meal,
            edit_meal,
            get_calorie_summary,
            log_meal,
            set_calorie_goal,
        )

        agent_tools.extend(
            [
                log_meal,
                get_calorie_summary,
                edit_meal,
                delete_meal,
                set_calorie_goal,
            ]
        )
        logger.info("Calorie tracking tools enabled")
    except ImportError as e:
        logger.warning("Failed to load Calorie tools: %s", e)

# Add Workout tools if database is configured
if database_url:
    try:
        from .workouts import (
            delete_workout,
            get_exercise_progress,
            get_last_workout,
            get_todays_workout,
            list_recent_workouts,
            log_workout,
            set_workout_split,
        )

        agent_tools.extend(
            [
                log_workout,
                get_last_workout,
                get_exercise_progress,
                list_recent_workouts,
                delete_workout,
                set_workout_split,
                get_todays_workout,
            ]
        )
        logger.info("Workout tracking tools enabled")
    except ImportError as e:
        logger.warning("Failed to load Workout tools: %s", e)

# Add Skills toolset (explore_repo skill)
try:
    from .skills import load_skill_from_dir
    from .skills.mcp_skill_toolset import McpSkillToolset

    explore_repo_skill = load_skill_from_dir(skills_dir / "explore_repo")
    if explore_repo_skill:
        agent_tools.append(McpSkillToolset(skills=[(explore_repo_skill, None)]))
        logger.info("Explore repo skill enabled")
except ImportError as e:
    logger.warning("Failed to load skills toolset: %s", e)

# Add Sandbox tools if enabled
sandbox_enabled = os.getenv("SANDBOX_ENABLED", "false").strip().lower() in (
    "true",
    "1",
    "yes",
)
if sandbox_enabled:
    try:
        from .sandbox import (
            sandbox_list_files,
            sandbox_read_file,
            sandbox_run_command,
            sandbox_send_file_to_user,
            sandbox_write_file,
        )

        agent_tools.extend(
            [
                sandbox_run_command,
                sandbox_write_file,
                sandbox_read_file,
                sandbox_list_files,
                sandbox_send_file_to_user,
            ]
        )
        logger.info("Sandbox tools enabled")
    except ImportError as e:
        logger.warning("Failed to load Sandbox tools: %s", e)

# Add Memory tools. The tools initialize Mem0 lazily and return structured
# configuration errors instead of disappearing from ADK's tool registry.
try:
    from .memory import (
        delete_all_memories,
        delete_memory,
        get_all_memories,
        get_memory,
        save_memory,
        search_memory,
        update_memory,
    )

    agent_tools.extend(
        [
            save_memory,
            search_memory,
            get_all_memories,
            get_memory,
            update_memory,
            delete_memory,
            delete_all_memories,
        ]
    )
    logger.info("Memory tools enabled")
except ImportError as e:
    logger.warning("Failed to load Memory tools: %s", e)

# Build before_tool_callback with optional telegram notifications
before_tool_callbacks: list[Any] = [logging_callbacks.before_tool]
after_model_callbacks: list[Any] = [logging_callbacks.after_model]
if telegram_tool_notifications_enabled():
    logger.info(
        "Telegram tool notifications enabled; "
        "registering before_tool and after_model callbacks"
    )
    before_tool_callbacks.append(notify_telegram_before_tool)
    after_model_callbacks.append(notify_telegram_after_model)

root_agent = LlmAgent(
    name="blacki",
    description=return_description_root(),
    before_agent_callback=logging_callbacks.before_agent,
    after_agent_callback=logging_callbacks.after_agent,
    model=model,
    instruction=return_instruction_root(),
    tools=agent_tools,
    before_model_callback=logging_callbacks.before_model,
    after_model_callback=after_model_callbacks,
    before_tool_callback=before_tool_callbacks,
    after_tool_callback=logging_callbacks.after_tool,
)

# Optional App configs explicitly set to None for template documentation
app = App(
    name="blacki",
    root_agent=root_agent,
    plugins=[
        GlobalInstructionPlugin(return_global_instruction),
        LoggingPlugin(),
        DeepSeekReasoningPlugin(name="deepseek_reasoning"),
    ],
    events_compaction_config=None,
    context_cache_config=None,
    resumability_config=None,
)
