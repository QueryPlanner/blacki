# mypy: disable-error-code="no-untyped-def"
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from google.adk.tools import ToolContext

from blacki.workouts.storage import WorkoutSession, WorkoutSessionSummary
from blacki.workouts.tools import (
    delete_workout,
    get_exercise_progress,
    get_last_workout,
    get_todays_workout,
    list_recent_workouts,
    log_workout,
    set_workout_split,
)


@pytest.fixture
def mock_tool_context():
    mock_context = create_autospec(ToolContext, spec_set=True, instance=True)
    mock_context.state = {}
    mock_context.user_id = "user1"
    return mock_context


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_storage")
async def test_log_workout_success(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.get_latest_split_session.return_value = None
    mock_storage.create_session.return_value = 1

    result = await log_workout(
        mock_tool_context,
        split_name="push",
        exercises=[{"name": "bench press", "sets": [{"weight_kg": 100, "reps": 10}]}],
    )

    assert result["status"] == "success"
    assert result["session_id"] == 1
    mock_storage.create_session.assert_called_once()


@pytest.mark.asyncio
async def test_log_workout_validation(mock_tool_context) -> None:
    result = await log_workout(
        mock_tool_context, split_name="push", exercises=[{"invalid": "format"}]
    )
    assert result["status"] == "error"

    result = await log_workout(
        mock_tool_context,
        split_name="push",
        exercises=[{"name": "bench press", "sets": [{"invalid": "format"}]}],
    )
    assert result["status"] == "error"


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_storage")
async def test_get_last_workout(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage

    session = WorkoutSession(
        id=1,
        user_id="user1",
        workout_date="2026-04-26",
        split_name="push",
        created_at="2026-04-26T10:00:00",
    )
    mock_storage.get_latest_split_session.return_value = session

    result = await get_last_workout(mock_tool_context, "push")

    assert result["status"] == "success"
    assert result["session"]["id"] == 1


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_storage")
async def test_get_exercise_progress(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.get_exercise_history.return_value = []

    result = await get_exercise_progress(mock_tool_context, "bench press")

    assert result["status"] == "success"
    assert result["exercise_name"] == "bench press"
    mock_storage.get_exercise_history.assert_called_once()


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_storage")
async def test_list_recent_workouts(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.get_recent_sessions.return_value = [
        WorkoutSessionSummary(
            id=1, workout_date="2026-04-26", split_name="push", exercise_count=5
        )
    ]

    result = await list_recent_workouts(mock_tool_context)

    assert result["status"] == "success"
    assert len(result["sessions"]) == 1


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_storage")
async def test_delete_workout(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.delete_session.return_value = True

    result = await delete_workout(mock_tool_context, 1)

    assert result["status"] == "success"
    mock_storage.delete_session.assert_called_once_with(1, "user1")


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_preferences_storage")
async def test_set_workout_split(mock_get_pref, mock_tool_context) -> None:
    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref

    split = {
        "monday": "push",
        "tuesday": "pull",
        "wednesday": "legs",
        "thursday": "push",
        "friday": "pull",
        "saturday": "legs",
        "sunday": "rest",
    }

    result = await set_workout_split(mock_tool_context, split)

    assert result["status"] == "success"
    mock_pref.set.assert_called_once_with("user1", "workout_split", split)


@pytest.mark.asyncio
async def test_set_workout_split_validation(mock_tool_context) -> None:
    # Missing days
    split = {"monday": "push"}
    result = await set_workout_split(mock_tool_context, split)
    assert result["status"] == "error"

    # Empty split
    split = {
        "monday": "push",
        "tuesday": "pull",
        "wednesday": "legs",
        "thursday": "push",
        "friday": "pull",
        "saturday": "legs",
        "sunday": "",
    }
    result = await set_workout_split(mock_tool_context, split)
    assert result["status"] == "error"


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_storage")
@patch("blacki.workouts.tools.get_preferences_storage")
@patch("blacki.workouts.tools.now_utc")
async def test_get_todays_workout(
    mock_now, mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    import datetime

    # Mock Monday
    mock_now.return_value = datetime.datetime(
        2026, 4, 20, 12, 0, 0, tzinfo=datetime.UTC
    )

    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = {"monday": "push"}

    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage

    session = WorkoutSession(
        id=1,
        user_id="user1",
        workout_date="2026-04-13",
        split_name="push",
        created_at="2026-04-13T10:00:00",
    )
    mock_storage.get_latest_split_session.return_value = session

    result = await get_todays_workout(mock_tool_context)

    assert result["status"] == "success"
    assert result["split"] == "push"
    assert result["last_session"]["id"] == 1


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_preferences_storage")
@patch("blacki.workouts.tools.now_utc")
async def test_get_todays_workout_rest_day(
    mock_now, mock_get_pref, mock_tool_context
) -> None:
    import datetime

    # Mock Sunday
    mock_now.return_value = datetime.datetime(
        2026, 4, 26, 12, 0, 0, tzinfo=datetime.UTC
    )

    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = {"sunday": "rest"}

    result = await get_todays_workout(mock_tool_context)

    assert result["status"] == "rest_day"
    assert "rest" in result["message"].lower()

@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_storage")
async def test_log_workout_shorthand_sets(mock_get_storage, mock_tool_context):
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage

    exercises = [
        {
            "name": "bench press",
            "sets": 3,
            "reps": 8,
            "weight": 100,
        }
    ]

    result = await log_workout(mock_tool_context, "Push", exercises)

    assert result["status"] == "success"
    session_arg = mock_storage.create_session.call_args[0][0]
    assert len(session_arg.exercises) == 1
    
    bench = session_arg.exercises[0]
    assert bench.exercise_name == "bench press"
    assert len(bench.sets) == 3
    for s in bench.sets:
        assert s.weight_kg == 100
        assert s.reps == 8
