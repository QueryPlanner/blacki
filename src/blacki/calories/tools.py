import logging
import math
from datetime import timedelta
from typing import Any

import dateparser  # type: ignore[import-untyped]
from google.adk.tools import ToolContext
from pydantic import ValidationError

from blacki.container import get_container
from blacki.health.config import (
    GOOGLE_HEALTH_NUTRITION_SCOPES,
    health_user_id_for_telegram_user,
)
from blacki.utils.dates import parse_date
from blacki.utils.preferences import get_preferences_storage
from blacki.utils.timezone import get_app_timezone, now_utc

from .service import VALID_MEAL_TYPES, get_meal_service, validate_meal
from .storage import CalorieEntry, get_storage

logger = logging.getLogger(__name__)

DEFAULT_CALORIE_GOAL = 2000


async def log_meal(
    tool_context: ToolContext,
    description: str,
    estimated_calories: int,
    date: str | None = None,
    meal_type: str | None = None,
    protein_g: float | None = None,
    carbs_g: float | None = None,
    fat_g: float | None = None,
) -> dict[str, Any]:
    """Log a meal and estimate its calories and macros.

    The LLM MUST estimate the calories before calling this tool
    based on the food description.

    Args:
        date: Optional date for the meal (natural language like "yesterday",
              "last Monday", or "2024-01-15"). Defaults to today.
    """
    if estimated_calories <= 0:
        return {"status": "error", "message": "estimated_calories must be > 0"}
    if not description.strip():
        return {"status": "error", "message": "description cannot be empty"}

    if meal_type and meal_type.lower() not in VALID_MEAL_TYPES:
        return {
            "status": "error",
            "message": f"meal_type must be one of {set(VALID_MEAL_TYPES)}",
        }

    user_id = tool_context.user_id
    if not user_id:  # pragma: no cover
        return {"status": "error", "message": "Missing user_id in tool_context"}

    try:
        now = now_utc()
        local_date = parse_date(date)
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
        validate_meal(entry)

        service = _try_get_meal_service()
        if service is None:
            storage = get_storage()
            entry_id = await storage.add_entry(entry)
            google_health_sync = "not_enabled"
        else:
            entry_id, google_health_sync = await service.mutate(
                user_id,
                private=_is_private_tool_context(tool_context),
                entry=entry,
            )

        storage = get_storage()

        # Get running daily total
        result: dict[str, Any] = {
            "status": "success",
            "entry_id": entry_id,
            "message": _meal_saved_message(
                f"Logged {estimated_calories} kcal for '{description}'.",
                google_health_sync,
            ),
            "google_health_sync": google_health_sync,
        }
        try:
            summary = await storage.get_daily_summary(user_id, local_date)
            result["daily_total"] = summary.total_calories
        except Exception:
            logger.exception("Failed to read meal summary after local commit")
            result["message"] += (
                " The meal was saved, but the daily summary is unavailable."
            )
        try:
            pref_storage = get_preferences_storage()
            goal = await pref_storage.get(user_id, "calorie_goal", DEFAULT_CALORIE_GOAL)
            result["calorie_goal"] = goal
            if "daily_total" in result:
                result["remaining"] = goal - result["daily_total"]
        except Exception:
            logger.exception("Failed to read calorie goal after local commit")
        return result
    except ValidationError as e:
        return {"status": "error", "message": f"Validation failed: {str(e)}"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("Failed to log meal")
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


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
    if not user_id:  # pragma: no cover
        return {"status": "error", "message": "Missing user_id in tool_context"}

    try:
        target_date = parse_date(date)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    storage = get_storage()
    pref_storage = get_preferences_storage()
    goal = await pref_storage.get(user_id, "calorie_goal", DEFAULT_CALORIE_GOAL)

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
        if not dt_end:  # pragma: no cover
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
    date: str | None = None,
    meal_type: str | None = None,
    protein_g: float | None = None,
    carbs_g: float | None = None,
    fat_g: float | None = None,
) -> dict[str, Any]:
    """Edit an existing meal entry.

    Args:
        date: Optional new date for the meal (natural language like
              "yesterday", "last Monday", or "2024-01-15").
    """
    user_id = tool_context.user_id
    if not user_id:  # pragma: no cover
        return {"status": "error", "message": "Missing user_id in tool_context"}

    updates: dict[str, Any] = {}
    if description is not None:  # pragma: no cover
        updates["description"] = description
    if estimated_calories is not None:  # pragma: no cover
        updates["calories"] = estimated_calories
    if date is not None:
        try:
            updates["logged_date"] = parse_date(date)
        except ValueError as e:
            return {"status": "error", "message": str(e)}
    if meal_type is not None:  # pragma: no cover
        if meal_type.lower() not in VALID_MEAL_TYPES:
            return {
                "status": "error",
                "message": f"meal_type must be one of {set(VALID_MEAL_TYPES)}",
            }
        updates["meal_type"] = meal_type.lower()
    if protein_g is not None:  # pragma: no cover
        updates["protein_g"] = protein_g
    if carbs_g is not None:  # pragma: no cover
        updates["carbs_g"] = carbs_g
    if fat_g is not None:  # pragma: no cover
        updates["fat_g"] = fat_g

    if description is not None and not description.strip():
        return {"status": "error", "message": "description cannot be empty"}
    if estimated_calories is not None and estimated_calories <= 0:
        return {"status": "error", "message": "estimated_calories must be > 0"}
    for value in (protein_g, carbs_g, fat_g):
        if value is not None and (
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        ):
            return {
                "status": "error",
                "message": "macros must be finite and nonnegative",
            }

    if not updates:  # pragma: no cover
        return {"status": "error", "message": "No fields provided to update"}

    try:
        service = _try_get_meal_service()
        if service is None:
            storage = get_storage()
            updated = await storage.update_entry(entry_id, user_id, **updates)
            sync_status = "not_enabled"
        else:
            _, sync_status = await service.mutate(
                user_id,
                private=_is_private_tool_context(tool_context),
                entry_id=entry_id,
                updates=updates,
            )
            updated = True

        if updated:
            return {
                "status": "success",
                "message": _meal_saved_message(
                    f"Updated entry {entry_id}", sync_status
                ),
                "google_health_sync": sync_status,
            }
        else:  # pragma: no cover
            return {
                "status": "error",
                "message": f"Entry {entry_id} not found or you don't have permission",
            }
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception:
        logger.exception(f"Failed to edit meal entry {entry_id}")
        return {"status": "error", "message": "An unexpected error occurred"}


async def delete_meal(
    tool_context: ToolContext,
    entry_id: int,
) -> dict[str, Any]:
    """Delete a mis-logged meal entry."""
    user_id = tool_context.user_id
    if not user_id:  # pragma: no cover
        return {"status": "error", "message": "Missing user_id in tool_context"}

    try:
        service = _try_get_meal_service()
        if service is None:
            storage = get_storage()
            deleted = await storage.delete_entry(entry_id, user_id)
            sync_status = "not_enabled"
        else:
            _, sync_status = await service.mutate(
                user_id,
                private=_is_private_tool_context(tool_context),
                entry_id=entry_id,
            )
            deleted = True

        if deleted:
            return {
                "status": "success",
                "message": _meal_saved_message(
                    f"Deleted entry {entry_id}", sync_status
                ),
                "google_health_sync": sync_status,
            }
        else:  # pragma: no cover
            return {
                "status": "error",
                "message": f"Entry {entry_id} not found or you don't have permission",
            }
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception:
        logger.exception(f"Failed to delete meal entry {entry_id}")
        return {"status": "error", "message": "An unexpected error occurred"}


async def set_calorie_goal(
    tool_context: ToolContext,
    daily_calories: int,
) -> dict[str, Any]:
    """Set the daily calorie goal."""
    user_id = tool_context.user_id
    if not user_id:  # pragma: no cover
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


async def get_meal_sync_status(tool_context: ToolContext) -> dict[str, Any]:
    """Read the current Google Health meal-export status for this private chat."""
    user_id = tool_context.user_id
    if not user_id:  # pragma: no cover
        return {"status": "error", "message": "Missing user_id in tool_context"}
    if not _is_private_tool_context(tool_context):
        return {
            "status": "error",
            "message": "Google Health meal export is available only in private chats",
        }

    health_user_id = health_user_id_for_telegram_user(user_id)
    if health_user_id is None:
        return {"status": "error", "message": "Invalid private Telegram identity"}

    try:
        health = get_container().google_health_storage
        await health.initialize()
        connection = await health.get_connection(health_user_id)
        counts = await health.nutrition.counts(health_user_id)
    except RuntimeError:
        return {"status": "error", "message": "Health storage is not initialized"}
    except Exception:
        logger.exception("Failed to read Google Health meal-export status")
        return {"status": "error", "message": "Could not read meal export status"}

    return {
        "status": "success",
        "google_health_connection": (
            connection.status if connection is not None else "not_connected"
        ),
        "nutrition_permissions": _has_nutrition_scopes(connection),
        "google_health_sync": counts,
    }


async def retry_meal_sync(tool_context: ToolContext) -> dict[str, Any]:
    """Retry failed Google Health meal exports for this private chat."""
    user_id = tool_context.user_id
    if not user_id:  # pragma: no cover
        return {"status": "error", "message": "Missing user_id in tool_context"}
    if not _is_private_tool_context(tool_context):
        return {
            "status": "error",
            "message": "Google Health meal export is available only in private chats",
        }

    health_user_id = health_user_id_for_telegram_user(user_id)
    if health_user_id is None:
        return {"status": "error", "message": "Invalid private Telegram identity"}

    try:
        container = get_container()
        health = container.google_health_storage
        await health.initialize()
        connection = await health.get_connection(health_user_id)
        if connection is None:
            return {
                "status": "not_connected",
                "requeued": 0,
                "message": "Connect Google Health before retrying meal exports.",
            }
        if not _health_connection_can_export(connection):
            return {
                "status": "authorization_required",
                "requeued": 0,
                "message": (
                    "Both Google Health nutrition permissions are required before "
                    "failed meal exports can be retried."
                ),
            }

        requeued = await health.nutrition.retry_failed(
            health_user_id, connection.health_user_id
        )
        if requeued and container.nutrition_export_worker is not None:
            container.nutrition_export_worker.wake()
        counts = await health.nutrition.counts(health_user_id)
        message = (
            f"Queued {requeued} failed meal export(s) for retry."
            if requeued
            else "There are no failed meal exports to retry."
        )
        return {
            "status": "success",
            "requeued": requeued,
            "message": message,
            "google_health_sync": counts,
        }
    except RuntimeError:
        return {"status": "error", "message": "Health storage is not initialized"}
    except Exception:
        logger.exception("Failed to retry Google Health meal exports")
        return {"status": "error", "message": "Could not retry meal exports"}


def _try_get_meal_service() -> Any | None:
    """Use the atomic service when the application container is available.

    The fallback keeps direct unit-level tool use compatible with the storage
    singleton; production startup always initializes the application container.
    """
    try:
        return get_meal_service()
    except RuntimeError:
        return None


def _is_private_tool_context(tool_context: ToolContext) -> bool:
    state = getattr(tool_context, "state", None)
    if state is None:
        return False
    getter = getattr(state, "get", None)
    return bool(getter("telegram_chat_type") == "private") if getter else False


def _meal_saved_message(message: str, sync_status: str) -> str:
    if sync_status == "pending":
        return f"{message} Saved in Blacki."
    if sync_status == "authorization_required":
        return f"{message} Saved in Blacki; reconnect Google Health to sync it."
    if sync_status == "failed":
        return (
            f"{message} Saved in Blacki; Google Health export failed. "
            "Ask me to retry failed meal exports."
        )
    return f"{message} Saved in Blacki."


def _has_nutrition_scopes(connection: Any | None) -> bool:
    """Return whether a connection grants both nutrition permissions."""
    return bool(
        connection is not None
        and set(GOOGLE_HEALTH_NUTRITION_SCOPES) <= set(connection.scopes)
    )


def _health_connection_can_export(connection: Any | None) -> bool:
    """Return whether a connected account can retry provider writes."""
    return bool(
        connection is not None
        and connection.status == "connected"
        and connection.encrypted_refresh_token is not None
        and _has_nutrition_scopes(connection)
    )
