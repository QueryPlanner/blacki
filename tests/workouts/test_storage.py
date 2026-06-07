# mypy: disable-error-code="no-untyped-def"
"""Unit tests for workout storage."""

import asyncio

import aiosqlite
import pytest

from blacki.workouts.storage import (
    SetDetail,
    SqliteWorkoutStorage,
    WorkoutExercise,
    WorkoutSession,
)


@pytest.fixture
async def conn():
    """Create an in-memory SQLite connection for testing."""
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


@pytest.fixture
def lock():
    """Create a lock for write operations."""
    return asyncio.Lock()


@pytest.fixture
async def storage(conn, lock):
    """Create a storage instance with the test connection."""
    storage = SqliteWorkoutStorage(conn, lock)
    await storage.initialize()
    yield storage
    await storage.close()


class TestSqliteWorkoutStorage:
    """Tests for SqliteWorkoutStorage."""

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, conn, lock) -> None:
        """Should create tables on initialization."""
        storage = SqliteWorkoutStorage(conn, lock)
        await storage.initialize()

        assert storage.is_initialized is True

        cursor = await conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='workout_sessions'
            """
        )
        row = await cursor.fetchone()
        assert row is not None

        cursor = await conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='workout_exercises'
            """
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_create_session(self, storage) -> None:
        """Should create a session with exercises."""
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

        session_id = await storage.create_session(session)

        assert session_id == 1

        saved = await storage.get_session(session_id, "user1")
        assert saved is not None
        assert saved.split_name == "push"
        assert len(saved.exercises) == 1
        assert saved.exercises[0].exercise_name == "bench press"
        assert saved.exercises[0].sets[0].weight_kg == 100

    @pytest.mark.asyncio
    async def test_create_session_multiple_exercises(self, storage) -> None:
        """Should create a session with multiple exercises."""
        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-04-26",
            split_name="push",
            created_at="2026-04-26T10:00:00",
            exercises=[
                WorkoutExercise(
                    exercise_name="bench press",
                    sets=[
                        SetDetail(set_num=1, weight_kg=100, reps=10),
                        SetDetail(set_num=2, weight_kg=105, reps=8),
                    ],
                    exercise_order=0,
                ),
                WorkoutExercise(
                    exercise_name="shoulder press",
                    sets=[SetDetail(set_num=1, weight_kg=60, reps=12)],
                    exercise_order=1,
                ),
            ],
        )

        session_id = await storage.create_session(session)

        saved = await storage.get_session(session_id, "user1")
        assert len(saved.exercises) == 2
        assert saved.exercises[0].sets[0].weight_kg == 100
        assert saved.exercises[1].exercise_name == "shoulder press"

    @pytest.mark.asyncio
    async def test_add_exercise(self, storage) -> None:
        """Should add an exercise to an existing session."""
        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-04-26",
            split_name="push",
            created_at="2026-04-26T10:00:00",
            exercises=[],
        )
        session_id = await storage.create_session(session)

        exercise = WorkoutExercise(
            exercise_name="squat",
            sets=[SetDetail(set_num=1, weight_kg=120, reps=8)],
        )

        exercise_id = await storage.add_exercise(session_id, exercise)

        assert exercise_id == 1

        saved = await storage.get_session(session_id, "user1")
        assert len(saved.exercises) == 1
        assert saved.exercises[0].exercise_name == "squat"

    @pytest.mark.asyncio
    async def test_update_exercise(self, storage) -> None:
        """Should update an exercise's sets and notes."""
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
        session_id = await storage.create_session(session)
        saved = await storage.get_session(session_id, "user1")
        exercise_id = saved.exercises[0].id

        result = await storage.update_exercise(
            exercise_id,
            "user1",
            sets=[SetDetail(set_num=1, weight_kg=110, reps=8)],
            notes="felt strong",
        )

        assert result is True

        updated = await storage.get_session(session_id, "user1")
        assert updated.exercises[0].sets[0].weight_kg == 110
        assert updated.exercises[0].notes == "felt strong"

    @pytest.mark.asyncio
    async def test_update_exercise_wrong_user(self, storage) -> None:
        """Should not update exercise belonging to different user."""
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
        session_id = await storage.create_session(session)
        saved = await storage.get_session(session_id, "user1")
        exercise_id = saved.exercises[0].id

        result = await storage.update_exercise(
            exercise_id,
            "user2",
            notes="should not work",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_exercise(self, storage) -> None:
        """Should delete an exercise."""
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
        session_id = await storage.create_session(session)
        saved = await storage.get_session(session_id, "user1")
        exercise_id = saved.exercises[0].id

        result = await storage.delete_exercise(exercise_id, "user1")

        assert result is True

        updated = await storage.get_session(session_id, "user1")
        assert len(updated.exercises) == 0

    @pytest.mark.asyncio
    async def test_delete_exercise_wrong_user(self, storage) -> None:
        """Should not delete exercise belonging to different user."""
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
        session_id = await storage.create_session(session)
        saved = await storage.get_session(session_id, "user1")
        exercise_id = saved.exercises[0].id

        result = await storage.delete_exercise(exercise_id, "user2")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, storage) -> None:
        """Should return None for non-existent session."""
        result = await storage.get_session(999, "user1")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_wrong_user(self, storage) -> None:
        """Should return None for session belonging to different user."""
        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-04-26",
            split_name="push",
            created_at="2026-04-26T10:00:00",
            exercises=[],
        )
        session_id = await storage.create_session(session)

        result = await storage.get_session(session_id, "user2")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_session(self, storage) -> None:
        """Should delete a session and cascade to exercises."""
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
        session_id = await storage.create_session(session)

        result = await storage.delete_session(session_id, "user1")

        assert result is True

        saved = await storage.get_session(session_id, "user1")
        assert saved is None

    @pytest.mark.asyncio
    async def test_delete_session_wrong_user(self, storage) -> None:
        """Should not delete session belonging to different user."""
        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-04-26",
            split_name="push",
            created_at="2026-04-26T10:00:00",
            exercises=[],
        )
        session_id = await storage.create_session(session)

        result = await storage.delete_session(session_id, "user2")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_latest_split_session(self, storage) -> None:
        """Should get the most recent session for a split."""
        for i in range(3):
            session = WorkoutSession(
                user_id="user1",
                workout_date=f"2026-04-{20 + i:02d}",
                split_name="push" if i < 2 else "pull",
                created_at=f"2026-04-{20 + i:02d}T10:00:00",
                exercises=[
                    WorkoutExercise(
                        exercise_name=f"exercise {i}",
                        sets=[SetDetail(set_num=1, weight_kg=100, reps=10)],
                    )
                ],
            )
            await storage.create_session(session)

        result = await storage.get_latest_split_session("user1", "push")

        assert result is not None
        assert result.workout_date == "2026-04-21"
        assert result.split_name == "push"

    @pytest.mark.asyncio
    async def test_get_latest_split_session_not_found(self, storage) -> None:
        """Should return None if no session for split."""
        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-04-26",
            split_name="push",
            created_at="2026-04-26T10:00:00",
            exercises=[],
        )
        await storage.create_session(session)

        result = await storage.get_latest_split_session("user1", "legs")

        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_users_isolated(self, storage) -> None:
        """Should isolate data between users."""
        session1 = WorkoutSession(
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
        session2 = WorkoutSession(
            user_id="user2",
            workout_date="2026-04-26",
            split_name="pull",
            created_at="2026-04-26T11:00:00",
            exercises=[
                WorkoutExercise(
                    exercise_name="deadlift",
                    sets=[SetDetail(set_num=1, weight_kg=200, reps=5)],
                )
            ],
        )
        id1 = await storage.create_session(session1)
        id2 = await storage.create_session(session2)

        s1 = await storage.get_session(id1, "user1")
        s2 = await storage.get_session(id2, "user2")

        assert s1.split_name == "push"
        assert s2.split_name == "pull"

        sessions1 = await storage.get_training_history("user1")
        sessions2 = await storage.get_training_history("user2")

        assert len(sessions1) == 1
        assert len(sessions2) == 1

    @pytest.mark.asyncio
    async def test_update_exercise_no_updates_returns_false(self, storage) -> None:
        """Should return False when no updates provided."""
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
        session_id = await storage.create_session(session)
        saved = await storage.get_session(session_id, "user1")
        exercise_id = saved.exercises[0].id

        result = await storage.update_exercise(exercise_id, "user1")

        assert result is False


class TestGetStorage:
    """Tests for get_storage function."""

    @pytest.mark.asyncio
    async def test_get_storage_raises_when_not_initialized(self, conn, lock) -> None:
        """Should raise RuntimeError when storage is not initialized."""
        from blacki.container import (
            reset_container_for_tests,
            set_container_from_connection,
        )
        from blacki.workouts.storage import get_storage

        set_container_from_connection(conn, lock)

        with pytest.raises(RuntimeError, match="Workout storage not initialized"):
            get_storage()

        reset_container_for_tests()

    @pytest.mark.asyncio
    async def test_get_storage_returns_storage_when_initialized(
        self, conn, lock
    ) -> None:
        """Should return storage when initialized."""
        from blacki.container import (
            reset_container_for_tests,
            set_container_from_connection,
        )
        from blacki.workouts.storage import get_storage

        container = set_container_from_connection(conn, lock)
        await container.workout_storage.initialize()

        result = get_storage()

        assert result is container.workout_storage

        reset_container_for_tests()


class TestCreateSessionEdgeCases:
    """Tests for create_session edge cases."""

    @pytest.mark.asyncio
    async def test_create_session_raises_when_lastrowid_none(self, conn, lock) -> None:
        """Should raise RuntimeError when lastrowid is None after session insert."""
        from unittest.mock import AsyncMock

        import aiosqlite

        mock_conn = AsyncMock(spec=aiosqlite.Connection)
        mock_cursor = AsyncMock()
        mock_cursor.lastrowid = None
        mock_conn.execute.return_value = mock_cursor

        storage = SqliteWorkoutStorage(mock_conn, lock)
        storage._schema_ready = True

        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-04-26",
            split_name="push",
            created_at="2026-04-26T10:00:00",
            exercises=[],
        )

        with pytest.raises(
            RuntimeError, match="Failed to get lastrowid after session insert"
        ):
            await storage.create_session(session)

        mock_conn.execute.assert_called()
        call_args = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert "ROLLBACK" in call_args

    @pytest.mark.asyncio
    async def test_create_session_rollback_on_exception(self, conn, lock) -> None:
        """Should rollback transaction on exception during session creation."""
        from unittest.mock import AsyncMock

        import aiosqlite

        mock_conn = AsyncMock(spec=aiosqlite.Connection)

        async def execute_side_effect(query, *args):
            if "COMMIT" in query:
                raise Exception("commit failed")
            mock_cursor = AsyncMock()
            mock_cursor.lastrowid = 1
            return mock_cursor

        mock_conn.execute.side_effect = execute_side_effect

        storage = SqliteWorkoutStorage(mock_conn, lock)
        storage._schema_ready = True

        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-04-26",
            split_name="push",
            created_at="2026-04-26T10:00:00",
            exercises=[],
        )

        with pytest.raises(Exception, match="commit failed"):
            await storage.create_session(session)

        call_args = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert "ROLLBACK" in call_args
