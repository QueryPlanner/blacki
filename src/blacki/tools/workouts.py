import logging
from typing import Any

from google.adk.tools import ToolContext

from blacki.utils.dates import parse_date
from blacki.utils.preferences import get_preferences_storage
from blacki.utils.timezone import get_app_timezone, now_utc
from blacki.workouts.storage import (
    COMPLETION_STATUSES,
    SESSION_TYPES,
    SetDetail,
    TrainingMetric,
    TrainingProgram,
    TrainingProgramDay,
    WorkoutExercise,
    WorkoutSession,
    get_storage,
)

logger = logging.getLogger(__name__)

LOWER_BACK_SWAP_STATUSES = {"tight", "sore", "pain", "avoid_hinge", "strained"}


def _parse_workout_exercises(
    exercises: list[dict[str, Any]] | None,
) -> tuple[list[WorkoutExercise], str | None]:
    if not exercises:
        return [], None

    if not isinstance(exercises, list):
        return [], "Exercises must be a list of dictionaries"  # type: ignore[unreachable]

    parsed_exercises = []
    for i, ex_dict in enumerate(exercises):
        if not isinstance(ex_dict, dict):
            return [], "Each exercise item must be a dictionary"  # type: ignore[unreachable]

        if "name" not in ex_dict or "sets" not in ex_dict:
            return [], "Each exercise must have 'name' and 'sets' keys"

        sets_data = ex_dict["sets"]
        sets_list: list[dict[str, Any]]

        if isinstance(sets_data, int):
            reps = ex_dict.get("reps", 0)
            weight = ex_dict.get("weight_kg") or ex_dict.get("weight", 0)
            sets_list = [{"weight_kg": weight, "reps": reps} for _ in range(sets_data)]
        elif isinstance(sets_data, dict):
            sets_list = [sets_data]
        elif isinstance(sets_data, list):
            for s in sets_data:
                if not isinstance(s, dict):
                    return [], "Each set item in sets list must be a dictionary"
            sets_list = sets_data
        else:
            return [], "'sets' must be a list of dictionaries or an integer"

        sets: list[SetDetail] = []
        for set_dict in sets_list:
            if "weight_kg" not in set_dict and "weight" not in set_dict:
                return [], "Each set must have 'weight_kg' (or 'weight')"
            if "reps" not in set_dict:
                return [], "Each set must have 'reps'"

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
                exercise_name=str(ex_dict["name"]).lower(),
                sets=sets,
                exercise_order=i,
                notes=ex_dict.get("notes"),
            )
        )

    return parsed_exercises, None


def _infer_metric_unit(metric_name: str) -> str:
    if metric_name.endswith("_kg") or "1rm" in metric_name or "load" in metric_name:
        return "kg"
    if metric_name.endswith("_bpm") or "heart_rate" in metric_name:
        return "bpm"
    if metric_name.endswith("_km") or "distance" in metric_name:
        return "km"
    if metric_name.endswith("_min") or "duration" in metric_name:
        return "min"
    return "unitless"


def _parse_training_metrics(
    user_id: str, metrics: dict[str, Any], recorded_at: str
) -> tuple[list[TrainingMetric], str | None]:
    if not isinstance(metrics, dict):
        return [], "metrics must be a dictionary"  # type: ignore[unreachable]

    if not metrics:
        return [], "metrics cannot be empty"

    parsed = []
    for raw_name, raw_value in metrics.items():
        metric_name = str(raw_name).strip().lower()
        if not metric_name:
            return [], "metric names cannot be empty"

        notes = None
        metric_recorded_at = recorded_at
        if isinstance(raw_value, dict):
            if "value" not in raw_value:
                return [], f"Metric '{metric_name}' must include a value"
            value_obj = raw_value["value"]
            unit = str(raw_value.get("unit") or _infer_metric_unit(metric_name))
            notes = raw_value.get("notes")
            metric_recorded_at = str(raw_value.get("recorded_at") or recorded_at)
        else:
            value_obj = raw_value
            unit = _infer_metric_unit(metric_name)

        try:
            value = float(value_obj)
        except (TypeError, ValueError):
            return [], f"Metric '{metric_name}' value must be numeric"

        parsed.append(
            TrainingMetric(
                user_id=user_id,
                metric_name=metric_name,
                value=value,
                unit=unit,
                recorded_at=metric_recorded_at,
                notes=notes,
            )
        )

    return parsed, None


def _matching_program_day(
    program: TrainingProgram, cycle_day: int
) -> TrainingProgramDay | None:
    for day in program.days:
        if day.cycle_day == cycle_day:
            return day
    return None


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

    try:
        parsed_date = parse_date(workout_date)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    parsed_exercises, parse_error = _parse_workout_exercises(exercises)
    if parse_error:
        return {"status": "error", "message": parse_error}

    session = WorkoutSession(
        user_id=user_id,
        workout_date=parsed_date,
        split_name=split_name,
        notes=notes,
        created_at=now_utc().isoformat(timespec="seconds"),
        exercises=parsed_exercises,
        session_type="resistance",
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


async def set_training_program(
    tool_context: ToolContext,
    program_config: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a rotating, multi-modal training program and its cycle pointer."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    if not isinstance(program_config, dict):
        return {"status": "error", "message": "program_config must be a dictionary"}  # type: ignore[unreachable]

    if baseline_metrics is not None and not isinstance(baseline_metrics, dict):
        return {"status": "error", "message": "baseline_metrics must be a dictionary"}  # type: ignore[unreachable]

    days_config = program_config.get("days")
    if not isinstance(days_config, list) or not days_config:
        return {"status": "error", "message": "program_config.days must be a list"}

    cycle_length_days = int(program_config.get("cycle_length_days") or len(days_config))
    current_cycle_day = int(program_config.get("current_cycle_day") or 1)
    current_mesocycle_week = int(program_config.get("current_mesocycle_week") or 1)
    if cycle_length_days < 1:
        return {"status": "error", "message": "cycle_length_days must be positive"}
    if not (1 <= current_cycle_day <= cycle_length_days):
        return {
            "status": "error",
            "message": "current_cycle_day must fit within the cycle length",
        }
    if current_mesocycle_week < 1:
        return {"status": "error", "message": "current_mesocycle_week must be positive"}

    program_days = []
    seen_days: set[int] = set()
    for index, day_config in enumerate(days_config, start=1):
        if not isinstance(day_config, dict):
            return {"status": "error", "message": "Each program day must be an object"}
        cycle_day = int(day_config.get("cycle_day") or index)
        if not (1 <= cycle_day <= cycle_length_days):
            return {"status": "error", "message": "cycle_day is outside cycle length"}
        if cycle_day in seen_days:
            return {"status": "error", "message": f"Duplicate cycle_day {cycle_day}"}
        seen_days.add(cycle_day)

        session_type = str(day_config.get("session_type") or "other").lower()
        if session_type not in SESSION_TYPES:
            return {
                "status": "error",
                "message": f"session_type must be one of {sorted(SESSION_TYPES)}",
            }

        exercises = day_config.get("exercises") or []
        if not isinstance(exercises, list):
            return {"status": "error", "message": "day exercises must be a list"}
        rules = day_config.get("rules") or {}
        if not isinstance(rules, dict):
            return {"status": "error", "message": "day rules must be an object"}

        target_duration = day_config.get("target_duration_min")
        program_days.append(
            TrainingProgramDay(
                cycle_day=cycle_day,
                focus=str(day_config.get("focus") or session_type),
                session_type=session_type,
                prescription=day_config.get("prescription"),
                modality=day_config.get("modality"),
                target_zone=day_config.get("target_zone"),
                target_duration_min=int(target_duration)
                if target_duration is not None
                else None,
                exercises=exercises,
                rules=rules,
                notes=day_config.get("notes"),
            )
        )

    now = now_utc().isoformat(timespec="seconds")
    try:
        starts_on = parse_date(program_config.get("starts_on"))
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    program = TrainingProgram(
        user_id=user_id,
        name=str(program_config.get("name") or "Training Program"),
        cycle_length_days=cycle_length_days,
        mesocycle_length_days=int(program_config.get("mesocycle_length_days") or 28),
        deload_week_interval=int(program_config.get("deload_week_interval") or 5),
        starts_on=starts_on,
        notes=program_config.get("notes"),
        created_at=now,
        updated_at=now,
        days=program_days,
    )

    storage = get_storage()
    program_id = await storage.create_training_program(
        program,
        current_cycle_day=current_cycle_day,
        current_mesocycle_week=current_mesocycle_week,
    )

    metrics_config = baseline_metrics or program_config.get("baseline_metrics")
    metric_ids: list[int] = []
    if isinstance(metrics_config, dict):
        parsed_metrics, metric_error = _parse_training_metrics(
            user_id, metrics_config, now
        )
        if metric_error:
            return {"status": "error", "message": metric_error}
        metric_ids = await storage.add_training_metrics(parsed_metrics)

    return {
        "status": "success",
        "message": (
            f"Stored training program '{program.name}' with {len(program_days)} days."
        ),
        "program_id": program_id,
        "cycle_length_days": cycle_length_days,
        "current_cycle_day": current_cycle_day,
        "current_mesocycle_week": current_mesocycle_week,
        "metric_ids": metric_ids,
    }


async def get_todays_training(tool_context: ToolContext) -> dict[str, Any]:
    """Return today's training from the active rotating program pointer."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    storage = get_storage()
    program = await storage.get_active_training_program(user_id)
    if program is None or program.state is None:
        return {
            "status": "not_configured",
            "message": "No active training program configured.",
        }

    cycle_day = program.state.current_cycle_day
    program_day = _matching_program_day(program, cycle_day)
    if program_day is None:
        return {
            "status": "error",
            "message": f"No program day found for cycle day {cycle_day}",
        }

    last_sessions = await storage.get_training_history(
        user_id,
        cycle_day=cycle_day,
        session_type=program_day.session_type,
        limit=1,
    )
    is_deload = program.state.current_mesocycle_week == program.deload_week_interval

    recommendations = []
    if cycle_day == 6:
        day_four_sessions = await storage.get_training_history(
            user_id,
            cycle_day=4,
            session_type="resistance",
            limit=1,
        )
        if day_four_sessions:
            lower_back_status = str(
                day_four_sessions[0].metrics.get("lower_back_status", "")
            ).lower()
            if lower_back_status in LOWER_BACK_SWAP_STATUSES:
                recommendations.append(
                    {
                        "type": "conditional_swap",
                        "message": (
                            "Lower back status after Day 4 suggests swapping "
                            "Day 6 and Day 7."
                        ),
                        "lower_back_status": lower_back_status,
                    }
                )

    return {
        "status": "success",
        "program": {
            "id": program.id,
            "name": program.name,
            "version": program.version,
            "cycle_length_days": program.cycle_length_days,
        },
        "state": program.state.model_dump(),
        "training_day": program_day.model_dump(),
        "is_deload": is_deload,
        "deload_message": "Use 50% sets and 50% load." if is_deload else None,
        "last_comparable_session": last_sessions[0].model_dump()
        if last_sessions
        else None,
        "recommendations": recommendations,
    }


async def log_training(
    tool_context: ToolContext,
    session_type: str,
    cycle_day: int | None = None,
    workout_date: str | None = None,
    exercises: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str | None = None,
    completion_status: str = "completed",
    advance_day: bool = False,
) -> dict[str, Any]:
    """Log a resistance, conditioning, recovery, or rest training session."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    normalized_session_type = session_type.lower()
    if normalized_session_type not in SESSION_TYPES:
        return {
            "status": "error",
            "message": f"session_type must be one of {sorted(SESSION_TYPES)}",
        }
    normalized_completion = completion_status.lower()
    if normalized_completion not in COMPLETION_STATUSES:
        allowed_statuses = sorted(COMPLETION_STATUSES)
        return {
            "status": "error",
            "message": f"completion_status must be one of {allowed_statuses}",
        }

    if metrics is not None and not isinstance(metrics, dict):
        return {  # type: ignore[unreachable]
            "status": "error",
            "message": "metrics must be a dictionary",
        }

    parsed_exercises, parse_error = _parse_workout_exercises(exercises)
    if parse_error:
        return {"status": "error", "message": parse_error}

    try:
        parsed_workout_date = parse_date(workout_date)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    storage = get_storage()
    program = await storage.get_active_training_program(user_id)
    if cycle_day is None and program and program.state:
        cycle_day = program.state.current_cycle_day

    program_day = (
        _matching_program_day(program, cycle_day) if program and cycle_day else None
    )
    split_name = program_day.focus if program_day else normalized_session_type
    previous_sessions = await storage.get_training_history(
        user_id,
        cycle_day=cycle_day,
        session_type=normalized_session_type,
        limit=1,
    )
    now = now_utc().isoformat(timespec="seconds")

    session = WorkoutSession(
        user_id=user_id,
        workout_date=parsed_workout_date,
        split_name=split_name,
        notes=notes,
        created_at=now,
        exercises=parsed_exercises,
        program_id=program.id if program else None,
        program_version=program.version if program else None,
        cycle_day=cycle_day,
        session_type=normalized_session_type,
        completion_status=normalized_completion,
        metrics=metrics or {},
    )
    session_id = await storage.create_session(session)

    advanced_state = None
    if advance_day:
        advanced_state = await storage.advance_training_state(user_id, 1, now)

    return {
        "status": "success",
        "session_id": session_id,
        "message": f"Logged {normalized_session_type} training session.",
        "previous_comparable_session": previous_sessions[0].model_dump()
        if previous_sessions
        else None,
        "advanced_state": advanced_state.model_dump() if advanced_state else None,
    }


async def advance_training_cycle(
    tool_context: ToolContext,
    days: int = 1,
) -> dict[str, Any]:
    """Explicitly advance the active training program pointer."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}
    if not (1 <= days <= 28):
        return {"status": "error", "message": "days must be between 1 and 28"}

    storage = get_storage()
    updated_at = now_utc().isoformat(timespec="seconds")
    state = await storage.advance_training_state(user_id, days, updated_at)
    if state is None:
        return {
            "status": "not_configured",
            "message": "No active training program configured.",
        }

    return {"status": "success", "state": state.model_dump()}


async def get_training_history(
    tool_context: ToolContext,
    cycle_day: int | None = None,
    session_type: str | None = None,
    exercise_name: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Get comparable training sessions by cycle day, type, or exercise."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    normalized_session_type = session_type.lower() if session_type else None
    if normalized_session_type and normalized_session_type not in SESSION_TYPES:
        return {
            "status": "error",
            "message": f"session_type must be one of {sorted(SESSION_TYPES)}",
        }

    storage = get_storage()
    sessions = await storage.get_training_history(
        user_id,
        cycle_day=cycle_day,
        session_type=normalized_session_type,
        exercise_name=exercise_name,
        limit=limit,
    )
    return {"status": "success", "sessions": [s.model_dump() for s in sessions]}


async def get_training_metrics(
    tool_context: ToolContext,
    metric_names: list[str] | None = None,
) -> dict[str, Any]:
    """Get latest training metrics like 1RMs, max HR, or ruck load."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    normalized_names = [name.lower() for name in metric_names] if metric_names else None
    storage = get_storage()
    metrics = await storage.get_latest_training_metrics(user_id, normalized_names)
    return {"status": "success", "metrics": [m.model_dump() for m in metrics]}


async def update_training_metrics(
    tool_context: ToolContext,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Record training metric history such as 1RMs or max heart rate."""
    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "message": "Missing user_id in tool_context"}

    now = now_utc().isoformat(timespec="seconds")
    parsed_metrics, metric_error = _parse_training_metrics(user_id, metrics, now)
    if metric_error:
        return {"status": "error", "message": metric_error}

    storage = get_storage()
    metric_ids = await storage.add_training_metrics(parsed_metrics)
    return {
        "status": "success",
        "message": f"Recorded {len(metric_ids)} training metrics.",
        "metric_ids": metric_ids,
    }


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
