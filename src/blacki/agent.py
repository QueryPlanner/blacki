"""ADK LlmAgent configuration."""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.plugins.logging_plugin import LoggingPlugin

from .callbacks import (
    LoggingCallbacks,
    notify_telegram_before_tool,
    telegram_tool_notifications_enabled,
)
from .prompt import (
    return_description_root,
    return_global_instruction,
    return_instruction_root,
)
from .tools import (
    browser_list_profiles,
    browser_stop_session,
    browser_task,
    example_tool,
)

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
google_api_key = os.getenv("GOOGLE_API_KEY")

model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")
model: Any = model_name

use_litellm = False

# OpenRouter-only: never use native Gemini (requires GOOGLE_API_KEY).
if openrouter_api_key and not google_api_key:
    model_name = _normalize_model_for_openrouter(model_name)
    use_litellm = True
elif model_name.lower().startswith("openrouter/") or "/" in model_name:
    use_litellm = True

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


def _create_skill_tuples() -> list[tuple[Any, Any]]:
    """Create skill tuples for McpSkillToolset.

    Returns a list of (Skill, McpToolset | None) tuples for all available
    skills with their optional MCP toolsets.
    """
    skills_list: list[tuple[Any, Any]] = []

    notion_token = os.getenv("NOTION_TOKEN", "").strip()
    if notion_token:
        try:
            from google.adk.tools.mcp_tool.mcp_session_manager import (
                StdioConnectionParams,
            )
            from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
            from mcp import StdioServerParameters

            from .skills import load_skill_from_dir

            skill = load_skill_from_dir(skills_dir / "notion")
            if skill:
                notion_mcp = McpToolset(
                    connection_params=StdioConnectionParams(
                        server_params=StdioServerParameters(
                            command="npx",
                            args=["-y", "@notionhq/notion-mcp-server"],
                            env={"NOTION_TOKEN": notion_token},
                        ),
                        timeout=30.0,
                    ),
                )
                skills_list.append((skill, notion_mcp))
                logger.info("Notion MCP skill enabled")
        except ImportError as e:
            logger.warning("MCP dependencies not available for Notion skill: %s", e)
        except Exception as e:
            logger.warning("Failed to create Notion MCP skill: %s", e)

    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    if github_token:
        try:
            from google.adk.tools.mcp_tool.mcp_session_manager import (
                StreamableHTTPConnectionParams,
            )
            from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

            from .skills import load_skill_from_dir

            skill = load_skill_from_dir(skills_dir / "github")
            if skill:
                github_mcp = McpToolset(
                    connection_params=StreamableHTTPConnectionParams(
                        url="https://api.githubcopilot.com/mcp/",
                        headers={
                            "Authorization": f"Bearer {github_token}",
                            "Content-Type": "application/json",
                        },
                        timeout=30.0,
                    ),
                )
                skills_list.append((skill, github_mcp))
                logger.info("GitHub MCP skill enabled")
        except ImportError as e:
            logger.warning("MCP dependencies not available for GitHub skill: %s", e)
        except Exception as e:
            logger.warning("Failed to create GitHub MCP skill: %s", e)

    return skills_list


# Build the list of tools
agent_tools: list[Any] = [
    example_tool,
    browser_task,
    browser_stop_session,
    browser_list_profiles,
]

# Add Brave Search tool if API key is available
brave_search_api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
if brave_search_api_key:
    try:
        from .tools import brave_search

        agent_tools.append(brave_search)
        logger.info("Brave Search tool enabled")
    except ImportError as e:
        logger.warning("Failed to load Brave Search tool: %s", e)

# Add MCP skills if available
skill_tuples = _create_skill_tuples()
if skill_tuples:
    from .skills.mcp_skill_toolset import McpSkillToolset

    agent_tools.append(McpSkillToolset(skills=skill_tuples))

# Build before_tool_callback with optional telegram notifications
before_tool_callbacks: list[Any] = [logging_callbacks.before_tool]
if telegram_tool_notifications_enabled():
    logger.info("Telegram tool notifications enabled; registering before_tool callback")
    before_tool_callbacks.append(notify_telegram_before_tool)

root_agent = LlmAgent(
    name="root_agent",
    description=return_description_root(),
    before_agent_callback=logging_callbacks.before_agent,
    after_agent_callback=logging_callbacks.after_agent,
    model=model,
    instruction=return_instruction_root(),
    tools=agent_tools,
    before_model_callback=logging_callbacks.before_model,
    after_model_callback=logging_callbacks.after_model,
    before_tool_callback=before_tool_callbacks,
    after_tool_callback=logging_callbacks.after_tool,
)

# Optional App configs explicitly set to None for template documentation
app = App(
    name="agent",
    root_agent=root_agent,
    plugins=[
        GlobalInstructionPlugin(return_global_instruction),
        LoggingPlugin(),
    ],
    events_compaction_config=None,
    context_cache_config=None,
    resumability_config=None,
)
