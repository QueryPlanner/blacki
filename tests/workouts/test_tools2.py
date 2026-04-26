# mypy: disable-error-code="no-untyped-def"
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from google.adk.tools import ToolContext

from blacki.workouts.storage import WorkoutExercise, WorkoutSession
from blacki.workouts.tools import log_workout


@pytest.fixture
def mock_tool_context():
    mock_context = create_autospec(ToolContext, spec_set=True, instance=True)
    mock_context.state = {}
    mock_context.user_id = "user1"
    return mock_context


@pytest.mark.asyncio
@patch("blacki.workouts.tools.get_storage")
@patch("blacki.workouts.tools._parse_date")
async def test_log_workout_append(
    mock_parse_date, mock_get_storage, mock_tool_context
) -> None:
    mock_parse_date.return_value = "2026-04-26"
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage

    last_session = WorkoutSession(
        id=1,
        user_id="user1",
        workout_date="2026-04-26",
        split_name="push",
        created_at="2026-04-26T10:00:00",
        exercises=[WorkoutExercise(exercise_name="bench press", sets=[])],
    )
    mock_storage.get_latest_split_session.return_value = last_session

    result = await log_workout(
        mock_tool_context,
        split_name="push",
        exercises=[{"name": "squat", "sets": [{"weight_kg": 100, "reps": 10}]}],
        workout_date="2026-04-26",
    )

    assert result["status"] == "success"
    assert result["session_id"] == 1
    mock_storage.add_exercise.assert_called_once()
