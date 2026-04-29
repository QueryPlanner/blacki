# mypy: disable-error-code="no-untyped-def"
from unittest.mock import AsyncMock, MagicMock

import pytest

from blacki.workouts.storage import (
    PostgresWorkoutStorage,
    SetDetail,
    WorkoutExercise,
    WorkoutSession,
    close_workout_storage,
    get_storage,
    init_workout_storage,
)


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=tx)
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.execute = AsyncMock()
    pool.fetch = AsyncMock()
    pool.fetchval = AsyncMock()
    pool.fetchrow = AsyncMock()
    return pool


@pytest.fixture
async def workout_storage(mock_pool):
    storage = PostgresWorkoutStorage(mock_pool)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(mock_pool) -> None:
    storage = PostgresWorkoutStorage(mock_pool)
    await storage.initialize()

    conn = mock_pool.acquire.return_value.__aenter__.return_value
    assert conn.execute.call_count == 5
    assert storage._schema_ready is True


@pytest.mark.asyncio
async def test_create_session(workout_storage, mock_pool) -> None:
    conn = mock_pool.acquire.return_value.__aenter__.return_value
    conn.fetchval.return_value = 123

    session = WorkoutSession(
        user_id="user1",
        workout_date="2026-04-26",
        split_name="push",
        created_at="2026-04-26T10:00:00",
        exercises=[
            WorkoutExercise(
                exercise_name="bench press",
                sets=[SetDetail(set_num=1, weight_kg=100, reps=10)],
            )
        ],
    )

    conn.execute.reset_mock()
    session_id = await workout_storage.create_session(session)

    assert session_id == 123
    conn.fetchval.assert_called_once()
    assert conn.execute.call_count == 1
    args = conn.execute.call_args[0]
    assert args[1] == 123
    assert args[2] == "bench press"


@pytest.mark.asyncio
async def test_add_exercise(workout_storage, mock_pool) -> None:
    mock_pool.fetchval.return_value = 456

    exercise = WorkoutExercise(
        exercise_name="squat", sets=[SetDetail(set_num=1, weight_kg=120, reps=8)]
    )

    exercise_id = await workout_storage.add_exercise(123, exercise)

    assert exercise_id == 456
    mock_pool.fetchval.assert_called_once()


@pytest.mark.asyncio
async def test_update_exercise(workout_storage, mock_pool) -> None:
    mock_pool.fetchval.return_value = "user1"  # ownership check
    mock_pool.execute.return_value = "UPDATE 1"

    result = await workout_storage.update_exercise(456, "user1", notes="form felt good")

    assert result is True
    mock_pool.execute.assert_called_once()


@pytest.mark.asyncio
async def test_delete_exercise(workout_storage, mock_pool) -> None:
    mock_pool.fetchval.return_value = "user1"
    mock_pool.execute.return_value = "DELETE 1"

    result = await workout_storage.delete_exercise(456, "user1")

    assert result is True
    mock_pool.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_session(workout_storage, mock_pool) -> None:
    mock_pool.fetchrow.return_value = {
        "id": 1,
        "user_id": "user1",
        "workout_date": "2026-04-26",
        "split_name": "push",
        "notes": None,
        "created_at": "2026-04-26T10:00:00",
    }

    mock_pool.fetch.return_value = [
        {
            "id": 10,
            "session_id": 1,
            "exercise_name": "bench press",
            "sets": (
                '[{"set_num": 1, "weight_kg": 100, "reps": 10, "is_warmup": false}]'
            ),
            "exercise_order": 0,
            "notes": None,
        }
    ]

    session = await workout_storage.get_session(1, "user1")

    assert session is not None
    assert session.id == 1
    assert session.split_name == "push"
    assert len(session.exercises) == 1
    assert session.exercises[0].exercise_name == "bench press"
    assert session.exercises[0].sets[0].weight_kg == 100


@pytest.mark.asyncio
async def test_get_recent_sessions(workout_storage, mock_pool) -> None:
    mock_pool.fetch.return_value = [
        {
            "id": 1,
            "workout_date": "2026-04-26",
            "split_name": "push",
            "exercise_count": 5,
        }
    ]

    sessions = await workout_storage.get_recent_sessions("user1")

    assert len(sessions) == 1
    assert sessions[0].id == 1
    assert sessions[0].exercise_count == 5


@pytest.mark.asyncio
async def test_get_exercise_history(workout_storage, mock_pool) -> None:
    mock_pool.fetch.return_value = [
        {
            "workout_date": "2026-04-26",
            "split_name": "push",
            "sets": (
                '[{"set_num": 1, "weight_kg": 100, "reps": 10, "is_warmup": false}]'
            ),
        }
    ]

    history = await workout_storage.get_exercise_history("user1", "bench press")

    assert len(history) == 1
    assert history[0].best_set_weight_kg == 100
    assert history[0].best_set_reps == 10
    assert history[0].total_volume_kg == 1000


@pytest.mark.asyncio
async def test_delete_session(workout_storage, mock_pool) -> None:
    mock_pool.execute.return_value = "DELETE 1"

    result = await workout_storage.delete_session(1, "user1")

    assert result is True


@pytest.mark.asyncio
async def test_singleton(mock_pool) -> None:
    import blacki.workouts.storage as storage

    storage._storage = None

    with pytest.raises(RuntimeError):
        get_storage()

    instance = await init_workout_storage(mock_pool)
    assert get_storage() is instance

    await close_workout_storage()
    with pytest.raises(RuntimeError):
        get_storage()


@pytest.mark.asyncio
async def test_reinit_workout_storage_closes_existing(mock_pool) -> None:
    """init_workout_storage closes existing storage before replacing."""
    import blacki.workouts.storage as storage

    existing = PostgresWorkoutStorage(mock_pool)
    existing.close = AsyncMock()  # type: ignore[method-assign]
    storage._storage = existing

    new = await init_workout_storage(mock_pool)

    existing.close.assert_awaited_once()
    assert storage._storage is new

    storage._storage = None


@pytest.mark.asyncio
async def test_get_latest_split_session(workout_storage, mock_pool) -> None:
    mock_pool.fetchrow.return_value = {
        "id": 1,
        "user_id": "user1",
        "workout_date": "2026-04-26",
        "split_name": "push",
        "notes": None,
        "created_at": "2026-04-26T10:00:00",
    }
    mock_pool.fetch.return_value = []

    session = await workout_storage.get_latest_split_session("user1", "push")
    assert session is not None
    assert session.id == 1
