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
        brave_search_api_key: API key for Brave Search.
        database_url: Postgres connection string for storage-backed tools.
        sandbox_enabled: Whether to enable sandbox tools.
        skills_dir: Directory containing skill definitions.
    """

    brave_search_api_key: str | None = None
    database_url: str | None = None
    sandbox_enabled: bool = False
    skills_dir: Path | None = None


def build_tools(config: ToolConfig) -> list[Any]:
    """Build tools based on explicit configuration.

    Args:
        config: Tool configuration specifying which tools to enable.

    Returns:
        List of tool instances ready for use by the agent.
    """
    tools: list[Any] = []

    if config.brave_search_api_key:
        tools.extend(_build_brave_search_tools())
        logger.info("Brave Search tool enabled")

    if config.database_url:
        tools.extend(_build_reminder_tools())
        tools.extend(_build_calorie_tools())
        tools.extend(_build_workout_tools())
        logger.info("Database-backed tools enabled")

    if config.sandbox_enabled:
        tools.extend(_build_sandbox_tools())
        logger.info("Sandbox tools enabled")

    if config.skills_dir:
        tools.extend(_build_skill_tools(config.skills_dir))

    tools.extend(_build_memory_tools())

    return tools


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
    except ImportError as e:
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
    except ImportError as e:
        logger.warning("Failed to load Calorie tools: %s", e)
        return []


def _build_workout_tools() -> list[Any]:
    """Build workout tracking tools."""
    try:
        from blacki.workouts import (
            delete_workout,
            get_exercise_progress,
            get_last_workout,
            get_todays_workout,
            list_recent_workouts,
            log_workout,
            set_workout_split,
        )

        return [
            log_workout,
            get_last_workout,
            get_exercise_progress,
            list_recent_workouts,
            delete_workout,
            set_workout_split,
            get_todays_workout,
        ]
    except ImportError as e:
        logger.warning("Failed to load Workout tools: %s", e)
        return []


def _build_sandbox_tools() -> list[Any]:
    """Build sandbox tools."""
    try:
        from blacki.sandbox import (
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
        ]
    except ImportError as e:
        logger.warning("Failed to load Sandbox tools: %s", e)
        return []


def _build_skill_tools(skills_dir: Path) -> list[Any]:
    """Build skill tools from a directory."""
    try:
        from blacki.skills import load_skill_from_dir
        from blacki.skills.mcp_skill_toolset import McpSkillToolset

        explore_repo_skill = load_skill_from_dir(skills_dir / "explore_repo")
        if explore_repo_skill:
            logger.info("Explore repo skill enabled")
            return [McpSkillToolset(skills=[(explore_repo_skill, None)])]
    except ImportError as e:
        logger.warning("Failed to load skills toolset: %s", e)
    return []


def _build_memory_tools() -> list[Any]:
    """Build memory tools."""
    try:
        from blacki.memory import (
            delete_all_memories,
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
            delete_all_memories,
        ]
    except ImportError as e:
        logger.warning("Failed to load Memory tools: %s", e)
        return []


def build_tool_config_from_env() -> ToolConfig:
    """Build tool configuration from environment variables.

    Returns:
        ToolConfig populated from os.environ.
    """
    import os

    skills_dir = Path(__file__).parent / "skills"

    return ToolConfig(
        brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or None,
        database_url=os.getenv("DATABASE_URL", "").strip() or None,
        sandbox_enabled=os.getenv("SANDBOX_ENABLED", "false").strip().lower()
        in ("true", "1", "yes"),
        skills_dir=skills_dir,
    )
