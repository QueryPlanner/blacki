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
    """

    exa_api_key: str | None = None
    brave_search_api_key: str | None = None
    sqlite_path: str | None = None
    sandbox_enabled: bool = False
    skills_dir: Path | None = None
    weather_enabled: bool = True


def build_tools(config: ToolConfig) -> list[Any]:
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
        tools.extend(_build_workout_tools())
        tools.extend(_build_declarative_db_tools())
        logger.info("Database-backed tools enabled")

    if config.sandbox_enabled:
        tools.extend(_build_sandbox_tools())
        logger.info("Sandbox tools enabled")

    if config.skills_dir:
        tools.extend(_build_skill_tools(config.skills_dir))

    if config.weather_enabled:
        tools.extend(_build_weather_tools())
        logger.info("Weather tools enabled")

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


def _build_workout_tools() -> list[Any]:
    """Build workout tracking tools."""
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

        return [
            set_training_program,
            get_todays_training,
            log_training,
            advance_training_cycle,
            get_training_history,
            get_training_metrics,
            update_training_metrics,
            log_workout,
            get_last_workout,
            delete_workout,
            set_workout_split,
            get_todays_workout,
        ]
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


def _build_skill_tools(skills_dir: Path) -> list[Any]:
    """Build skill tools from a directory."""
    try:
        from google.adk.skills.models import Skill
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

        from blacki.skills import load_skill_from_dir
        from blacki.skills.mcp_skill_toolset import McpSkillToolset

        skills_to_load = ["explore_repo", "gemini_cli", "agent_browser"]
        loaded_skills: list[tuple[Skill, McpToolset | None]] = []
        for skill_name in skills_to_load:
            skill = load_skill_from_dir(skills_dir / skill_name)
            if skill:
                logger.info("%s skill enabled", skill_name)
                loaded_skills.append((skill, None))

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

    return ToolConfig(
        exa_api_key=os.getenv("EXA_API_KEY", "").strip() or None,
        brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or None,
        sqlite_path=sqlite_path,
        sandbox_enabled=os.getenv("SANDBOX_ENABLED", "false").strip().lower()
        in ("true", "1", "yes"),
        skills_dir=skills_dir,
    )
