"""Tool registry for building agent tools from explicit configuration.

This module provides factory-based tool registration, replacing the module-level
tool building pattern with explicit dependency injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    """Configuration for building agent tools.

    All fields are optional and tools are only enabled when their
    corresponding configuration is provided.

    Attributes:
        exa_api_key: API key for Exa Search.
        brave_search_api_key: API key for Brave Search.
        sqlite_path: Path to SQLite database for storage-backed tools.
        sandbox_enabled: Whether to enable sandbox tools.
        skills_dir: Directory containing skill definitions.
        legacy_workout_tools_enabled: Whether to expose weekly split fallbacks.
        kokoro_tts_base_url: Private Kokoro API base URL.
        kokoro_tts_voice: Default Kokoro voice ID.
        google_health_enabled: Whether the private Google Health reader is enabled.
    """

    exa_api_key: str | None = None
    brave_search_api_key: str | None = None
    sqlite_path: str | None = None
    sandbox_enabled: bool = False
    skills_dir: Path | None = None
    weather_enabled: bool = True
    legacy_workout_tools_enabled: bool = False
    kokoro_tts_base_url: str | None = None
    kokoro_tts_voice: str = "af_heart"
    google_health_enabled: bool = False
    zepto_mcp_enabled: bool = False
    zepto_mcp_config_dir: Path = Path("data/credentials/zepto-mcp-remote")
    zepto_mcp_allowed_chat_ids: frozenset[str] = frozenset()
    r2_files_enabled: bool = False


def build_tools(
    config: ToolConfig, *, include_user_scoped_tools: bool = False
) -> list[Any]:
    """Build tools based on explicit configuration.

    Args:
        config: Tool configuration specifying which tools to enable.

    Returns:
        List of tool instances ready for use by the agent.
    """
    tools: list[Any] = []

    if config.exa_api_key:
        tools.extend(_build_exa_search_tools())
        logger.info("Exa Search tool enabled")

    if config.brave_search_api_key:
        tools.extend(_build_brave_search_tools())
        logger.info("Brave Search tool enabled")

    if config.sqlite_path:
        tools.extend(_build_reminder_tools())
        tools.extend(_build_calorie_tools())
        tools.extend(_build_workout_tools(config.legacy_workout_tools_enabled))
        tools.extend(_build_declarative_db_tools())
        logger.info("Database-backed tools enabled")

    if config.sandbox_enabled:
        tools.extend(_build_sandbox_tools())
        logger.info("Sandbox tools enabled")

    if config.skills_dir:
        tools.extend(
            _build_skill_tools(
                config.skills_dir,
                config=config,
                include_user_scoped_tools=include_user_scoped_tools,
            )
        )

    if config.weather_enabled:
        tools.extend(_build_weather_tools())
        logger.info("Weather tools enabled")

    if include_user_scoped_tools and config.kokoro_tts_base_url:
        tools.extend(
            _build_tts_tools(
                base_url=config.kokoro_tts_base_url,
                voice=config.kokoro_tts_voice,
            )
        )

    if include_user_scoped_tools and config.google_health_enabled:
        tools.extend(_build_health_tools())

    if include_user_scoped_tools and config.r2_files_enabled:
        tools.extend(_build_user_file_tools())

    tools.extend(_build_memory_tools())

    return tools


def _build_exa_search_tools() -> list[Any]:
    """Build Exa Search tools."""
    try:
        from blacki.search import exa_search

        return [exa_search]
    except ImportError as e:
        logger.warning("Failed to load Exa Search tool: %s", e)
        return []


def _build_brave_search_tools() -> list[Any]:
    """Build Brave Search tools."""
    try:
        from blacki.tools import brave_search

        return [brave_search]
    except ImportError as e:
        logger.warning("Failed to load Brave Search tool: %s", e)
        return []


def _build_reminder_tools() -> list[Any]:
    """Build reminder tools."""
    try:
        from blacki.reminders import cancel_reminder, list_reminders, schedule_reminder

        return [schedule_reminder, list_reminders, cancel_reminder]
    except ImportError as e:  # pragma: no cover
        logger.warning("Failed to load Reminder tools: %s", e)
        return []


def _build_calorie_tools() -> list[Any]:
    """Build calorie tracking tools."""
    try:
        from blacki.calories import (
            delete_meal,
            edit_meal,
            get_calorie_summary,
            log_meal,
            set_calorie_goal,
        )

        return [log_meal, get_calorie_summary, edit_meal, delete_meal, set_calorie_goal]
    except ImportError as e:  # pragma: no cover
        logger.warning("Failed to load Calorie tools: %s", e)
        return []


def _build_workout_tools(include_legacy: bool = False) -> list[Any]:
    """Build canonical training tools and optional legacy split fallbacks."""
    try:
        from blacki.workouts import (
            advance_training_cycle,
            delete_workout,
            get_last_workout,
            get_todays_training,
            get_todays_workout,
            get_training_history,
            get_training_metrics,
            log_training,
            log_workout,
            set_training_program,
            set_workout_split,
            update_training_metrics,
        )

        tools = [
            set_training_program,
            get_todays_training,
            log_training,
            advance_training_cycle,
            get_training_history,
            get_training_metrics,
            update_training_metrics,
            delete_workout,
        ]
        if include_legacy:
            tools.extend(
                [
                    log_workout,
                    get_last_workout,
                    set_workout_split,
                    get_todays_workout,
                ]
            )
        return tools
    except ImportError as e:  # pragma: no cover
        logger.warning("Failed to load Workout tools: %s", e)
        return []


def _build_sandbox_tools() -> list[Any]:
    """Build sandbox tools."""
    try:
        from blacki.sandbox import (
            sandbox_execute_code,
            sandbox_list_files,
            sandbox_read_file,
            sandbox_run_command,
            sandbox_send_file_to_user,
            sandbox_write_file,
        )

        return [
            sandbox_run_command,
            sandbox_write_file,
            sandbox_read_file,
            sandbox_list_files,
            sandbox_send_file_to_user,
            sandbox_execute_code,
        ]
    except ImportError as e:  # pragma: no cover
        logger.warning("Failed to load Sandbox tools: %s", e)
        return []


def _build_skill_tools(
    skills_dir: Path,
    *,
    config: ToolConfig | None = None,
    include_user_scoped_tools: bool = False,
) -> list[Any]:
    """Build skill tools from a directory."""
    config = config or ToolConfig()
    try:
        from google.adk.skills.models import Skill
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

        from blacki.skills import load_skill_from_dir
        from blacki.skills.mcp_skill_toolset import McpSkillToolset

        skills_to_load = ["explore_repo", "agent_browser"]
        loaded_skills: list[tuple[Skill, McpToolset | None]] = []
        for skill_name in skills_to_load:
            skill = load_skill_from_dir(skills_dir / skill_name)
            if skill:
                logger.info("%s skill enabled", skill_name)
                loaded_skills.append((skill, None))

        if include_user_scoped_tools and config.zepto_mcp_enabled:
            from blacki.zepto import ZeptoCredentialError, create_zepto_toolset

            try:
                zepto_toolset = create_zepto_toolset(
                    config_dir=config.zepto_mcp_config_dir,
                    allowed_chat_ids=config.zepto_mcp_allowed_chat_ids,
                )
                zepto_skill = load_skill_from_dir(skills_dir / "zepto")
                if zepto_skill:
                    logger.info("Zepto MCP skill enabled for the root agent")
                    loaded_skills.append((zepto_skill, zepto_toolset))
            except ZeptoCredentialError as exc:
                logger.warning("Zepto MCP disabled: %s", exc)

        if loaded_skills:
            return [McpSkillToolset(skills=loaded_skills)]
    except ImportError as e:  # pragma: no cover
        logger.warning("Failed to load skills toolset: %s", e)
    return []


def _build_weather_tools() -> list[Any]:
    """Build weather tools."""
    try:
        from blacki.weather import get_current_weather, get_weather_forecast

        return [get_current_weather, get_weather_forecast]
    except ImportError as e:  # pragma: no cover
        logger.warning("Failed to load Weather tools: %s", e)
        return []


def _build_memory_tools() -> list[Any]:
    """Build memory tools."""
    try:
        from blacki.memory import (
            delete_memory,
            get_all_memories,
            get_memory,
            save_memory,
            search_memory,
            update_memory,
        )

        return [
            save_memory,
            search_memory,
            get_all_memories,
            get_memory,
            update_memory,
            delete_memory,
        ]
    except ImportError as e:  # pragma: no cover
        logger.warning("Failed to load Memory tools: %s", e)
        return []


def _build_tts_tools(*, base_url: str, voice: str) -> list[Any]:
    """Build the private Telegram speech-delivery tool."""
    try:
        from blacki.tts import KokoroTtsConfig, create_send_text_to_speech_tool

        config = KokoroTtsConfig(base_url=base_url, voice=voice)
        logger.info("Kokoro TTS tool enabled for the Telegram root agent")
        return [create_send_text_to_speech_tool(config)]
    except (ImportError, ValueError) as exc:
        logger.warning("Kokoro TTS disabled: %s", exc)
        return []


def _build_user_file_tools() -> list[Any]:
    """Build private Telegram sender-scoped durable file tools."""
    try:
        from blacki.user_files import create_user_file_tools

        logger.info("Durable R2 file tools enabled for the Telegram root agent")
        return create_user_file_tools()
    except (ImportError, ValueError) as exc:
        logger.warning("Durable R2 file tools disabled: %s", exc)
        return []


def _build_health_tools() -> list[Any]:
    """Build the private, read-only Google Health tool."""
    try:
        from blacki.health.tools import get_health_summary

        logger.info("Google Health summary tool enabled for the Telegram root agent")
        return [get_health_summary]
    except ImportError as exc:  # pragma: no cover
        logger.warning("Google Health tool disabled: %s", exc)
        return []


def _build_declarative_db_tools() -> list[Any]:
    """Build declarative database tools."""
    try:
        from blacki.declarative_db.tools import (
            create_custom_table,
            create_query_template,
            delete_custom_instruction_override,
            delete_custom_table,
            execute_query_template,
            list_custom_tables_and_templates,
            set_custom_instruction_override,
        )

        return [
            create_custom_table,
            delete_custom_table,
            create_query_template,
            execute_query_template,
            list_custom_tables_and_templates,
            set_custom_instruction_override,
            delete_custom_instruction_override,
        ]
    except ImportError as e:  # pragma: no cover
        logger.warning("Failed to load Declarative DB tools: %s", e)
        return []


def build_tool_config_from_env() -> ToolConfig:
    """Build tool configuration from environment variables.

    Returns:
        ToolConfig populated from os.environ.
    """
    import os

    skills_dir = Path(__file__).parent / "skills"
    # Match server.py AGENT_DIR calculation (points to src/, not project root)
    agent_dir = os.getenv("AGENT_DIR", str(Path(__file__).resolve().parent.parent))
    default_sqlite_path = str(Path(agent_dir) / ".adk" / "tools.db")

    sqlite_path = os.getenv("SQLITE_PATH", "").strip() or default_sqlite_path
    allowed_zepto_chat_ids = frozenset(
        item.strip()
        for item in os.getenv("ZEPTO_MCP_ALLOWED_TELEGRAM_CHAT_IDS", "").split(",")
        if item.strip()
    )

    return ToolConfig(
        exa_api_key=os.getenv("EXA_API_KEY", "").strip() or None,
        brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or None,
        sqlite_path=sqlite_path,
        sandbox_enabled=os.getenv("SANDBOX_ENABLED", "false").strip().lower()
        in ("true", "1", "yes"),
        skills_dir=skills_dir,
        legacy_workout_tools_enabled=os.getenv("LEGACY_WORKOUT_TOOLS_ENABLED", "false")
        .strip()
        .lower()
        in ("true", "1", "yes"),
        kokoro_tts_base_url=os.getenv("KOKORO_TTS_BASE_URL", "").strip() or None,
        kokoro_tts_voice=os.getenv("KOKORO_TTS_VOICE", "af_heart").strip()
        or "af_heart",
        google_health_enabled=_google_health_configured(),
        zepto_mcp_enabled=os.getenv("ZEPTO_MCP_ENABLED", "false").strip().lower()
        in ("true", "1", "yes"),
        zepto_mcp_config_dir=Path(
            os.getenv(
                "ZEPTO_MCP_CONFIG_DIR",
                "data/credentials/zepto-mcp-remote",
            ).strip()
        ),
        zepto_mcp_allowed_chat_ids=allowed_zepto_chat_ids,
        r2_files_enabled=os.getenv("R2_FILES_ENABLED", "false").strip().lower()
        in ("true", "1", "yes"),
    )


def _google_health_configured() -> bool:
    """Return whether the optional Google Health configuration is complete."""
    try:
        from blacki.health.config import google_health_configured_from_environment

        return google_health_configured_from_environment()
    except ImportError:  # pragma: no cover
        return False
