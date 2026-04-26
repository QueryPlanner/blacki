import logging
from typing import Any

import dateparser  # type: ignore[import-untyped]
from google.adk.tools import ToolContext

from blacki.utils.preferences import get_preferences_storage
from blacki.utils.timezone import get_app_timezone, now_utc

from .storage import SetDetail, WorkoutExercise, WorkoutSession, get_storage

logger = logging.getLogger(__name__)


def _parse_date(date_str: str | None) -> str:
    tz = get_app_timezone()
    if not date_str or date_str.lower() in ("today", "now"):  # pragma: no cover
        return now_utc().astimezone(tz).strftime("%Y-%m-%d")

    dt = dateparser.parse(  # pragma: no cover
        date_str,
        settings={"TIMEZONE": str(tz), "RETURN_AS_TIMEZONE_AWARE": True},
    )
    if not dt:  # pragma: no cover
        return now_utc().astimezone(tz).strftime("%Y-%m-%d")

    return str(dt.strftime("%Y-%m-%d"))  # pragma: no cover


async def log_workout(
    tool_context: ToolContext,
    split_name: str,
    exercises: list[dict[str, Any]],
    workout_date: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Start or complete a full workout session."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    parsed_date = _parse_date(workout_date)

    # Parse exercises
    parsed_exercises = []
    for i, ex_dict in enumerate(exercises):
        if "name" not in ex_dict or "sets" not in ex_dict:
            return {
                "status": "error",
                "message": "Each exercise must have 'name' and 'sets' keys",
            }

        sets_data = ex_dict["sets"]
        sets_list: list[dict[str, Any]] = []

        if isinstance(sets_data, int):
            # Shorthand: sets=3, reps=8, weight=100
            reps = ex_dict.get("reps", 0)
            weight = ex_dict.get("weight_kg") or ex_dict.get("weight", 0)
            sets_list = [{"weight_kg": weight, "reps": reps} for _ in range(sets_data)]
        elif isinstance(sets_data, dict):  # pragma: no cover
            sets_list = [sets_data]
        elif isinstance(sets_data, list):
            sets_list = sets_data
        else:  # pragma: no cover
            return {
                "status": "error",
                "message": "'sets' must be a list of dictionaries or an integer",
            }

        sets: list[SetDetail] = []
        for set_dict in sets_list:
            if "weight_kg" not in set_dict and "weight" not in set_dict:
                return {
                    "status": "error",
                    "message": "Each set must have 'weight_kg' (or 'weight')",
                }
            if "reps" not in set_dict:  # pragma: no cover
                return {
                    "status": "error",
                    "message": "Each set must have 'reps'",
                }

            weight_val = set_dict.get("weight_kg") or set_dict.get("weight", 0)
            sets.append(
                SetDetail(
                    set_num=set_dict.get("set_num", len(sets) + 1),
                    weight_kg=float(weight_val),
                    reps=int(set_dict["reps"]),
                    is_warmup=bool(set_dict.get("is_warmup", False)),
                )
            )

        parsed_exercises.append(
            WorkoutExercise(
                exercise_name=ex_dict["name"].lower(),
                sets=sets,
                exercise_order=i,
                notes=ex_dict.get("notes"),
            )
        )

    session = WorkoutSession(
        user_id=user_id,
        workout_date=parsed_date,
        split_name=split_name,
        notes=notes,
        created_at=now_utc().isoformat(timespec="seconds"),
        exercises=parsed_exercises,
    )

    storage = get_storage()

    # Check if we have a previous session for comparison BEFORE saving the new one
    last_session = await storage.get_latest_split_session(user_id, split_name)

    result: dict[str, Any]
    if last_session and last_session.workout_date == parsed_date:
        # User is logging more exercises for today's session, append them
        session_id = last_session.id
        if session_id is None:  # pragma: no cover
            raise ValueError(
                "Session ID cannot be None when appending to an existing session."
            )
        for exercise in parsed_exercises:
            # Update order to be after existing exercises
            exercise.exercise_order += len(last_session.exercises)
            await storage.add_exercise(session_id, exercise)

        result = {
            "status": "success",
            "session_id": session_id,
            "message": (
                f"Added {len(exercises)} exercises to today's '{split_name}' workout."
            ),
        }
        # For comparison, we want the session *before* today's, but since we
        # don't fetch that easily, we skip comparison for appends.
        last_session = None
    else:
        session_id = await storage.create_session(session)
        result = {
            "status": "success",
            "session_id": session_id,
            "message": (
                f"Logged '{split_name}' workout with {len(exercises)} exercises."
            ),
        }

    if last_session:
        result["comparison"] = {
            "last_workout_date": last_session.workout_date,
            # the LLM will parse exercises and show it
            "message": "Compared to last session: ...",
            "last_session": last_session.model_dump(),
        }

    return result


async def get_last_workout(
    tool_context: ToolContext,
    split_name: str,
) -> dict[str, Any]:
    """Get the most recent session for that split with full exercise data."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    storage = get_storage()
    session = await storage.get_latest_split_session(user_id, split_name)

    if session:
        return {"status": "success", "session": session.model_dump()}
    else:  # pragma: no cover
        return {
            "status": "not_found",
            "message": f"No previous '{split_name}' workout found",
        }


async def get_exercise_progress(
    tool_context: ToolContext,
    exercise_name: str,
    weeks: int = 4,
) -> dict[str, Any]:
    """Progressive overload view for a single exercise over N weeks."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    storage = get_storage()
    # cap at 8 entries as per instructions
    limit = min(weeks * 2, 8)

    history = await storage.get_exercise_history(user_id, exercise_name, limit=limit)

    return {
        "status": "success",
        "exercise_name": exercise_name.lower(),
        "history": [h.model_dump() for h in history],
    }


async def list_recent_workouts(
    tool_context: ToolContext,
    limit: int = 10,
) -> dict[str, Any]:
    """Overview of recent sessions (lightweight)."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    storage = get_storage()
    sessions = await storage.get_recent_sessions(user_id, limit=limit)

    return {
        "status": "success",
        "sessions": [s.model_dump() for s in sessions],
    }


async def delete_workout(
    tool_context: ToolContext,
    session_id: int,
) -> dict[str, Any]:
    """Remove a mis-logged session."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    storage = get_storage()
    deleted = await storage.delete_session(session_id, user_id)

    if deleted:
        return {"status": "success", "message": f"Deleted session {session_id}"}
    else:  # pragma: no cover
        return {
            "status": "error",
            "message": f"Session {session_id} not found or you don't have permission",
        }


async def set_workout_split(
    tool_context: ToolContext,
    split_config: dict[str, str],
) -> dict[str, Any]:
    """Configure the weekly split mapping."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    days_of_week = {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }

    # lower keys
    split_config = {k.lower(): str(v) for k, v in split_config.items()}

    if set(split_config.keys()) != days_of_week:
        return {
            "status": "error",
            "message": "split_config must contain exactly the 7 days of the week",
        }

    for day, split in split_config.items():
        if not split.strip():
            return {"status": "error", "message": f"Split for {day} cannot be empty"}

    pref_storage = get_preferences_storage()
    await pref_storage.set(user_id, "workout_split", split_config)

    return {
        "status": "success",
        "message": "Workout split updated successfully",
        "split": split_config,
    }


async def get_todays_workout(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """What should I do today?"""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    pref_storage = get_preferences_storage()
    split_config = await pref_storage.get(user_id, "workout_split")

    if not split_config:  # pragma: no cover
        return {
            "status": "not_configured",
            "message": (
                "No workout split configured. "
                "Please use set_workout_split to configure one."
            ),
        }

    tz = get_app_timezone()
    now = now_utc().astimezone(tz)
    day_name = now.strftime("%A").lower()

    todays_split = split_config.get(day_name, "Rest")

    if todays_split.lower() == "rest":
        return {
            "status": "rest_day",
            "message": f"Today is {day_name.capitalize()}, which is a Rest day.",
        }

    storage = get_storage()
    last_session = await storage.get_latest_split_session(user_id, todays_split)

    if last_session:
        return {
            "status": "success",
            "day": day_name.capitalize(),
            "split": todays_split,
            "message": f"Today is {todays_split} day.",
            "last_session": last_session.model_dump(),
        }
    else:  # pragma: no cover
        return {
            "status": "success",
            "day": day_name.capitalize(),
            "split": todays_split,
            "message": (
                f"Today is {todays_split} day, but no previous session was found."
            ),
        }
