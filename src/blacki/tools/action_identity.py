"""Lightweight action identifiers shared by dispatch and transport hooks."""

from typing import Any

MEAL_ACTION_NAMES = (
    "log_meal",
    "get_calorie_summary",
    "edit_meal",
    "delete_meal",
    "set_calorie_goal",
    "get_meal_sync_status",
    "retry_meal_sync",
)


def meal_action_name(tool_name: str, args: dict[str, Any]) -> str:
    """Resolve a registered action without importing its implementation."""
    action = args.get("action")
    if tool_name == "meals" and isinstance(action, str) and action in MEAL_ACTION_NAMES:
        return action
    return tool_name
