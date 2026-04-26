import logging
from datetime import timedelta
from typing import Any

import dateparser  # type: ignore[import-untyped]
from google.adk.tools import ToolContext

from blacki.utils.preferences import get_preferences_storage
from blacki.utils.timezone import get_app_timezone, now_utc

from .storage import CalorieEntry, get_storage

logger = logging.getLogger(__name__)

DEFAULT_CALORIE_GOAL = 2000


def _parse_date(date_str: str | None) -> str:
    """Parse a natural language date to YYYY-MM-DD local time."""
    tz = get_app_timezone()
    if not date_str or date_str.lower() in ("today", "now"):
        return now_utc().astimezone(tz).strftime("%Y-%m-%d")

    dt = dateparser.parse(
        date_str,
        settings={"TIMEZONE": str(tz), "RETURN_AS_TIMEZONE_AWARE": True},
    )
    if not dt:
        # Fallback to today if unparseable
        return now_utc().astimezone(tz).strftime("%Y-%m-%d")

    return str(dt.strftime("%Y-%m-%d"))


async def log_meal(
    tool_context: ToolContext,
    description: str,
    estimated_calories: int,
    meal_type: str | None = None,
    protein_g: int | None = None,
    carbs_g: int | None = None,
    fat_g: int | None = None,
) -> dict[str, Any]:
    """Log a meal and estimate its calories and macros.

    The LLM MUST estimate the calories before calling this tool
    based on the food description.
    """
    if estimated_calories <= 0:
        return {"status": "error", "message": "estimated_calories must be > 0"}
    if not description.strip():
        return {"status": "error", "message": "description cannot be empty"}

    valid_meal_types = {"breakfast", "lunch", "dinner", "snack"}
    if meal_type and meal_type.lower() not in valid_meal_types:
        return {
            "status": "error",
            "message": f"meal_type must be one of {valid_meal_types}",
        }

    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    now = now_utc()
    tz = get_app_timezone()
    local_date = now.astimezone(tz).strftime("%Y-%m-%d")

    entry = CalorieEntry(
        user_id=user_id,
        description=description,
        calories=estimated_calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        meal_type=meal_type.lower() if meal_type else None,
        logged_at=now.isoformat(timespec="seconds"),
        logged_date=local_date,
    )

    storage = get_storage()
    entry_id = await storage.add_entry(entry)

    # Get running daily total
    summary = await storage.get_daily_summary(user_id, local_date)

    # Get user goal
    pref_storage = get_preferences_storage()
    goal = await pref_storage.get(user_id, "calorie_goal", DEFAULT_CALORIE_GOAL)

    remaining = goal - summary.total_calories

    return {
        "status": "success",
        "entry_id": entry_id,
        "message": f"Logged {estimated_calories} kcal for '{description}'.",
        "daily_total": summary.total_calories,
        "calorie_goal": goal,
        "remaining": remaining,
    }


async def get_calorie_summary(
    tool_context: ToolContext,
    date: str | None = None,
    days: int = 1,
) -> dict[str, Any]:
    """Get calorie intake summary for a specific day or date range.

    Use days=1 for a single day detailed view.
    Use days>1 for multi-day aggregates (up to 30 days).
    """
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    storage = get_storage()
    pref_storage = get_preferences_storage()
    goal = await pref_storage.get(user_id, "calorie_goal", DEFAULT_CALORIE_GOAL)

    target_date = _parse_date(date)

    if days <= 1:
        summary = await storage.get_daily_summary(user_id, target_date)
        return {
            "status": "success",
            "calorie_goal": goal,
            "summary": summary.model_dump(),
        }
    else:
        # Multi-day aggregate
        days = min(days, 30)  # Cap at 30 days
        tz = get_app_timezone()
        dt_end = dateparser.parse(
            target_date,
            settings={"TIMEZONE": str(tz), "RETURN_AS_TIMEZONE_AWARE": True},
        )
        if not dt_end:
            dt_end = now_utc().astimezone(tz)

        dt_start = dt_end - timedelta(days=days - 1)
        start_date = dt_start.strftime("%Y-%m-%d")

        summaries = await storage.get_date_range_summary(
            user_id, start_date, target_date
        )
        return {
            "status": "success",
            "calorie_goal": goal,
            "summaries": [s.model_dump() for s in summaries],
        }


async def edit_meal(
    tool_context: ToolContext,
    entry_id: int,
    description: str | None = None,
    estimated_calories: int | None = None,
    meal_type: str | None = None,
    protein_g: int | None = None,
    carbs_g: int | None = None,
    fat_g: int | None = None,
) -> dict[str, Any]:
    """Edit an existing meal entry."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    updates: dict[str, Any] = {}
    if description is not None:
        updates["description"] = description
    if estimated_calories is not None:
        updates["calories"] = estimated_calories
    if meal_type is not None:
        updates["meal_type"] = meal_type.lower()
    if protein_g is not None:
        updates["protein_g"] = protein_g
    if carbs_g is not None:
        updates["carbs_g"] = carbs_g
    if fat_g is not None:
        updates["fat_g"] = fat_g

    if not updates:
        return {"status": "error", "message": "No fields provided to update"}

    storage = get_storage()
    updated = await storage.update_entry(entry_id, user_id, **updates)

    if updated:
        return {"status": "success", "message": f"Updated entry {entry_id}"}
    else:
        return {
            "status": "error",
            "message": f"Entry {entry_id} not found or you don't have permission",
        }


async def delete_meal(
    tool_context: ToolContext,
    entry_id: int,
) -> dict[str, Any]:
    """Delete a mis-logged meal entry."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    storage = get_storage()
    deleted = await storage.delete_entry(entry_id, user_id)

    if deleted:
        return {"status": "success", "message": f"Deleted entry {entry_id}"}
    else:
        return {
            "status": "error",
            "message": f"Entry {entry_id} not found or you don't have permission",
        }


async def set_calorie_goal(
    tool_context: ToolContext,
    daily_calories: int,
) -> dict[str, Any]:
    """Set the daily calorie goal."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    if not (500 <= daily_calories <= 10000):
        return {
            "status": "error",
            "message": "Calorie goal must be between 500 and 10000",
        }

    pref_storage = get_preferences_storage()
    await pref_storage.set(user_id, "calorie_goal", daily_calories)

    return {
        "status": "success",
        "message": f"Daily calorie goal set to {daily_calories} kcal",
        "new_goal": daily_calories,
    }
