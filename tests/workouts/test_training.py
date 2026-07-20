# mypy: disable-error-code="no-untyped-def"
# ruff: noqa: E501
"""Unit tests for training-program storage, models, and tools."""

import asyncio
from unittest.mock import AsyncMock, create_autospec, patch

import aiosqlite
import pytest
from google.adk.tools import ToolContext

from blacki.workouts.storage import (
    SetDetail,
    SqliteWorkoutStorage,
    TrainingMetric,
    TrainingProgram,
    TrainingProgramDay,
    TrainingProgramState,
    WorkoutExercise,
    WorkoutSession,
)
from blacki.workouts.tools import (
    advance_training_cycle,
    get_todays_training,
    get_training_history,
    get_training_metrics,
    log_training,
    set_training_program,
    update_training_metrics,
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


@pytest.fixture
def mock_tool_context():
    mock_context = create_autospec(ToolContext, spec_set=True, instance=True)
    mock_context.state = {}
    mock_context.user_id = "user1"
    return mock_context


class TestSqliteTrainingStorage:
    """Tests for the SQLite training-program storage methods."""

    @pytest.mark.asyncio
    async def test_ensure_columns_idempotent(self, conn, lock) -> None:
        """Should handle existing columns and ignore them during ALTER."""
        storage = SqliteWorkoutStorage(conn, lock)
        await storage.initialize()
        # Call it again to prove safety/idempotency
        await storage.initialize()
        assert storage.is_initialized is True

    @pytest.mark.asyncio
    async def test_create_and_get_training_program(self, storage) -> None:
        """Should save a program with days and retrieve it as active."""
        program = TrainingProgram(
            user_id="user1",
            name="Test Program",
            cycle_length_days=14,
            starts_on="2026-06-07",
            created_at="2026-06-07T12:00:00",
            updated_at="2026-06-07T12:00:00",
            days=[
                TrainingProgramDay(
                    cycle_day=1,
                    focus="Legs",
                    session_type="resistance",
                    prescription="3x5 Squats",
                    exercises=[{"name": "squat", "sets": 3}],
                )
            ],
        )

        program_id = await storage.create_training_program(program, 1, 1)
        assert program_id == 1

        active = await storage.get_active_training_program("user1")
        assert active is not None
        assert active.name == "Test Program"
        assert len(active.days) == 1
        assert active.days[0].focus == "Legs"
        assert active.days[0].exercises == [{"name": "squat", "sets": 3}]
        assert active.state is not None
        assert active.state.current_cycle_day == 1
        assert active.state.current_mesocycle_week == 1

    @pytest.mark.asyncio
    async def test_advance_training_state_calculations(self, storage) -> None:
        """Should correctly advance cycle days and weeks, wrapping 14 to 1 and weeks to week 1."""
        program = TrainingProgram(
            user_id="user1",
            name="Rotating Program",
            cycle_length_days=14,
            mesocycle_length_days=28,
            deload_week_interval=5,
            starts_on="2026-06-07",
            created_at="2026-06-07T12:00:00",
            updated_at="2026-06-07T12:00:00",
            days=[
                TrainingProgramDay(cycle_day=i, focus="Rest", session_type="rest")
                for i in range(1, 15)
            ],
        )
        await storage.create_training_program(
            program, current_cycle_day=1, current_mesocycle_week=1
        )

        # Advance by 6 days (to day 7)
        state = await storage.advance_training_state("user1", 6, "2026-06-13T12:00:00")
        assert state is not None
        assert state.current_cycle_day == 7
        assert state.current_mesocycle_week == 1

        # Advance by 1 more day (to day 8). Day 7 was the end of week 1, so week advances to 2.
        state = await storage.advance_training_state("user1", 1, "2026-06-14T12:00:00")
        assert state.current_cycle_day == 8
        assert state.current_mesocycle_week == 2

        # Advance by 7 days (to day 15 -> day 1). Day 14 was the end of week 2, so week advances to 3.
        state = await storage.advance_training_state("user1", 7, "2026-06-21T12:00:00")
        assert state.current_cycle_day == 1
        assert state.current_mesocycle_week == 3

        # Advance state to deload week 5 (week 4 ends at day 28 which is cycle day 14 of second loop)
        # Week 3 ends at cycle day 7 of third loop, week 4 ends at cycle day 14 of third loop.
        # Let's verify by advancing to week 5
        # Current state is Day 1, Week 3.
        # Advance by 13 days to get to Day 14, Week 4
        state = await storage.advance_training_state("user1", 13, "2026-07-04T12:00:00")
        assert state.current_cycle_day == 14
        assert state.current_mesocycle_week == 4

        # Advance 1 day to Day 1, Week 5 (Deload week)
        state = await storage.advance_training_state("user1", 1, "2026-07-05T12:00:00")
        assert state.current_cycle_day == 1
        assert state.current_mesocycle_week == 5

        # Day 7 is end of week 5. Advance 6 days to Day 7, Week 5.
        state = await storage.advance_training_state("user1", 6, "2026-07-11T12:00:00")
        assert state.current_cycle_day == 7
        assert state.current_mesocycle_week == 5

        # Advance 1 day. End of deload interval (5), week wraps back to 1. Cycle day becomes 8.
        state = await storage.advance_training_state("user1", 1, "2026-07-12T12:00:00")
        assert state.current_cycle_day == 8
        assert state.current_mesocycle_week == 1

    @pytest.mark.asyncio
    async def test_get_training_history_filters(self, storage) -> None:
        """Should filter history by cycle day, session type, or exercise name."""
        now = "2026-06-07T12:00:00"
        session1 = WorkoutSession(
            user_id="user1",
            workout_date="2026-06-07",
            split_name="Legs",
            created_at=now,
            cycle_day=1,
            session_type="resistance",
            exercises=[
                WorkoutExercise(
                    exercise_name="squat",
                    sets=[SetDetail(set_num=1, weight_kg=150.0, reps=5)],
                )
            ],
        )
        session2 = WorkoutSession(
            user_id="user1",
            workout_date="2026-06-08",
            split_name="Elliptical",
            created_at=now,
            cycle_day=2,
            session_type="zone2",
            metrics={"duration_min": 45, "avg_hr_bpm": 135},
        )
        await storage.create_session(session1)
        await storage.create_session(session2)

        # Filter by cycle day
        history_day1 = await storage.get_training_history("user1", cycle_day=1)
        assert len(history_day1) == 1
        assert history_day1[0].split_name == "Legs"

        # Filter by type
        history_zone2 = await storage.get_training_history(
            "user1", session_type="zone2"
        )
        assert len(history_zone2) == 1
        assert history_zone2[0].split_name == "Elliptical"

        # Filter by exercise
        history_squat = await storage.get_training_history(
            "user1", exercise_name="squat"
        )
        assert len(history_squat) == 1
        assert history_squat[0].exercises[0].exercise_name == "squat"

        # Filter by missing exercise
        history_bench = await storage.get_training_history(
            "user1", exercise_name="bench"
        )
        assert len(history_bench) == 0

    @pytest.mark.asyncio
    async def test_training_metrics_history(self, storage) -> None:
        """Should store metrics history and return the single latest point for each metric."""
        metric1 = TrainingMetric(
            user_id="user1",
            metric_name="squat_1rm",
            value=150.0,
            unit="kg",
            recorded_at="2026-06-07T12:00:00",
        )
        metric2 = TrainingMetric(
            user_id="user1",
            metric_name="squat_1rm",
            value=152.5,
            unit="kg",
            recorded_at="2026-06-08T12:00:00",
        )
        metric3 = TrainingMetric(
            user_id="user1",
            metric_name="bench_1rm",
            value=100.0,
            unit="kg",
            recorded_at="2026-06-07T12:00:00",
        )

        ids = await storage.add_training_metrics([metric1, metric2, metric3])
        assert len(ids) == 3

        # Retrieve all latest metrics
        latest = await storage.get_latest_training_metrics("user1")
        assert len(latest) == 2

        # Verify order and values (latest squat_1rm should be 152.5)
        bench = next(m for m in latest if m.metric_name == "bench_1rm")
        squat = next(m for m in latest if m.metric_name == "squat_1rm")
        assert bench.value == 100.0
        assert squat.value == 152.5
        assert squat.recorded_at == "2026-06-08T12:00:00"

        # Retrieve filtered list of names
        filtered = await storage.get_latest_training_metrics("user1", ["bench_1rm"])
        assert len(filtered) == 1
        assert filtered[0].metric_name == "bench_1rm"


class TestTrainingTools:
    """Tests for the high-level training tools and workflows."""

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_set_training_program_success(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Should validate and create a program config and baseline metrics."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.create_training_program.return_value = 1
        mock_storage.add_training_metrics.return_value = [1, 2]

        config = {
            "name": "Rotating Plan",
            "cycle_length_days": 14,
            "starts_on": "today",
            "days": [
                {
                    "cycle_day": 1,
                    "focus": "Legs Strength",
                    "session_type": "resistance",
                    "prescription": "3x5 Squats",
                    "exercises": [{"name": "squat", "sets": 3}],
                },
                {
                    "cycle_day": 2,
                    "focus": "Cardio Base",
                    "session_type": "zone2",
                    "modality": "elliptical",
                    "target_duration_min": 45,
                },
            ],
            "baseline_metrics": {
                "squat_1rm": 150.0,
                "deadlift_1rm": 200.0,
            },
        }

        result = await set_training_program(mock_tool_context, config)
        assert result["status"] == "success"
        assert result["program_id"] == 1
        assert len(result["metric_ids"]) == 2
        mock_storage.create_training_program.assert_called_once()
        mock_storage.add_training_metrics.assert_called_once()

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_set_training_program_validations(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Should enforce validation rules on keys, indexes, and session types."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage

        # Missing days
        assert (await set_training_program(mock_tool_context, {}))["status"] == "error"

        # Cycle day outside bounds
        config = {
            "days": [{"cycle_day": 20, "session_type": "rest"}],
            "cycle_length_days": 14,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # Invalid explicit start date must not silently become today
        config = {
            "days": [{"cycle_day": 1, "session_type": "rest"}],
            "cycle_length_days": 1,
            "starts_on": "definitely-not-a-real-date",
        }
        result = await set_training_program(mock_tool_context, config)
        assert result["status"] == "error"
        assert "Could not understand date" in result["message"]

        # Duplicate cycle day
        config = {
            "days": [
                {"cycle_day": 1, "session_type": "rest"},
                {"cycle_day": 1, "session_type": "rest"},
            ],
            "cycle_length_days": 14,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # Invalid session type
        config = {
            "days": [{"cycle_day": 1, "session_type": "invalid_type"}],
            "cycle_length_days": 14,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # Invalid day config format
        config = {
            "days": [None],
            "cycle_length_days": 14,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # Invalid metrics structure
        config = {
            "days": [{"cycle_day": 1, "session_type": "rest"}],
            "cycle_length_days": 14,
            "baseline_metrics": {"": 100},
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_get_todays_training_no_program(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Should return not_configured when no active program is set."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.get_active_training_program.return_value = None

        result = await get_todays_training(mock_tool_context)
        assert result["status"] == "not_configured"

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_get_todays_training_day_six_swap(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Should recommend swapping Day 6 and Day 7 if Day 4 deadlifts logged back strain."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage

        day = TrainingProgramDay(
            cycle_day=6,
            focus="VO2 Max",
            session_type="vo2",
            prescription="4x4 Rower",
        )
        program = TrainingProgram(
            id=1,
            user_id="user1",
            name=" Rotating",
            starts_on="2026-06-07",
            created_at="2026-06-07T12:00:00",
            updated_at="2026-06-07T12:00:00",
            days=[day],
            state=TrainingProgramState(
                user_id="user1",
                program_id=1,
                current_cycle_day=6,
                current_mesocycle_week=1,
                updated_at="2026-06-12T12:00:00",
            ),
        )
        mock_storage.get_active_training_program.return_value = program

        # Mock Day 4 log with back strain
        day_four_session = WorkoutSession(
            user_id="user1",
            workout_date="2026-06-10",
            split_name="Pull",
            created_at="2026-06-10T12:00:00",
            cycle_day=4,
            metrics={"lower_back_status": "pain"},
        )

        async def history_side_effect(
            user_id, cycle_day, session_type, limit=1, exercise_name=None
        ):
            if cycle_day == 6:
                return []
            if cycle_day == 4:
                return [day_four_session]
            return []

        mock_storage.get_training_history.side_effect = history_side_effect

        result = await get_todays_training(mock_tool_context)
        assert result["status"] == "success"
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["type"] == "conditional_swap"
        assert "swapping Day 6 and Day 7" in result["recommendations"][0]["message"]

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_log_training_success_and_advancement(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Should save a session with multi-modal metrics and advance state pointer if flag is set."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.create_session.return_value = 123

        program = TrainingProgram(
            id=1,
            user_id="user1",
            name="Rotating Plan",
            starts_on="2026-06-07",
            created_at="2026-06-07T12:00:00",
            updated_at="2026-06-07T12:00:00",
            days=[
                TrainingProgramDay(cycle_day=1, focus="Legs", session_type="resistance")
            ],
            state=TrainingProgramState(
                user_id="user1",
                program_id=1,
                current_cycle_day=1,
                current_mesocycle_week=1,
                updated_at="2026-06-07T12:00:00",
            ),
        )
        mock_storage.get_active_training_program.return_value = program

        advanced_state = TrainingProgramState(
            user_id="user1",
            program_id=1,
            current_cycle_day=2,
            current_mesocycle_week=1,
            updated_at="2026-06-07T13:00:00",
        )
        mock_storage.advance_training_state.return_value = advanced_state
        mock_storage.get_training_history.return_value = []

        result = await log_training(
            mock_tool_context,
            session_type="resistance",
            cycle_day=1,
            exercises=[{"name": "squat", "sets": [{"weight_kg": 150.0, "reps": 5}]}],
            metrics={"lower_back_status": "ok"},
            advance_day=True,
        )

        assert result["status"] == "success"
        assert result["session_id"] == 123
        assert result["advanced_state"] is not None
        assert result["advanced_state"]["current_cycle_day"] == 2
        mock_storage.create_session.assert_called_once()
        mock_storage.advance_training_state.assert_called_once()

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_advance_training_cycle_errors(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Should validate advance steps and error when no program exists."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage

        # Invalid days input
        result = await advance_training_cycle(mock_tool_context, days=0)
        assert result["status"] == "error"

        # Active program missing
        mock_storage.advance_training_state.return_value = None
        result = await advance_training_cycle(mock_tool_context, days=1)
        assert result["status"] == "not_configured"

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_get_training_history_tool(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Should filter and validate parameters in history tools."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.get_training_history.return_value = []

        result = await get_training_history(mock_tool_context, session_type="invalid")
        assert result["status"] == "error"

        result = await get_training_history(mock_tool_context, session_type="zone2")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_get_and_update_metrics_tools(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Should record metric history and retrieve filtered values."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.add_training_metrics.return_value = [10]
        mock_storage.get_latest_training_metrics.return_value = [
            TrainingMetric(
                user_id="user1",
                metric_name="squat_1rm",
                value=150.0,
                unit="kg",
                recorded_at="2026-06-07T12:00:00",
            )
        ]

        # Update metrics
        result = await update_training_metrics(mock_tool_context, {"squat_1rm": 150.0})
        assert result["status"] == "success"
        assert result["metric_ids"] == [10]

        # Get metrics
        result = await get_training_metrics(mock_tool_context, ["squat_1rm"])
        assert result["status"] == "success"
        assert len(result["metrics"]) == 1
        assert result["metrics"][0]["metric_name"] == "squat_1rm"

    @pytest.mark.asyncio
    async def test_missing_user_id_error_paths(self) -> None:
        """Should handle missing user_id in tool contexts gracefully across all new paths."""
        mock_context = create_autospec(ToolContext, spec_set=True, instance=True)
        mock_context.user_id = None

        assert (await set_training_program(mock_context, {}))["status"] == "error"
        assert (await get_todays_training(mock_context))["status"] == "error"
        assert (await log_training(mock_context, "rest"))["status"] == "error"
        assert (await advance_training_cycle(mock_context))["status"] == "error"
        assert (await get_training_history(mock_context))["status"] == "error"
        assert (await get_training_metrics(mock_context))["status"] == "error"
        assert (await update_training_metrics(mock_context, {}))["status"] == "error"


class TestTrainingRegistryAndPrompt:
    """Tests confirming tools are correctly integrated into the registry and exposed by prompt."""

    def test_training_tools_built_by_registry(self) -> None:
        """Registry build_tools must include all 7 new tools."""
        from blacki.registry import ToolConfig, build_tools

        config = ToolConfig(sqlite_path="/tmp/test_tools.db")
        tools = build_tools(config)
        tool_names = {
            getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools
        }

        new_tool_names = {
            "set_training_program",
            "get_todays_training",
            "log_training",
            "advance_training_cycle",
            "get_training_history",
            "get_training_metrics",
            "update_training_metrics",
        }
        for name in new_tool_names:
            assert name in tool_names

    def test_prompt_guidance_for_training(self) -> None:
        """A workout request should load the training-program policy."""
        from blacki.prompt import build_domain_instruction

        tool_names = {
            "set_training_program",
            "get_todays_training",
            "log_training",
            "advance_training_cycle",
            "get_training_history",
            "get_training_metrics",
            "update_training_metrics",
        }
        prompt = build_domain_instruction("Show today's workout", tool_names)
        assert "training-program" in prompt
        assert "set_training_program" in prompt
        assert "get_todays_training" in prompt
        assert "log_training" in prompt
        assert "advance_training_cycle" in prompt
        assert "get_training_metrics" in prompt
        assert "update_training_metrics" in prompt


class TestTrainingEdgeCasesAndCoverage:
    """Additional tests to reach 100% test coverage across training storage and tools."""

    @pytest.mark.asyncio
    async def test_ensure_columns_idempotence_branches(self, conn, lock) -> None:
        """Force table initialization when tables exist to cover False branch of column check."""
        storage = SqliteWorkoutStorage(conn, lock)
        await storage.initialize()
        # Reset schema_ready to check existing table columns path
        storage._schema_ready = False
        await storage.initialize()
        assert storage.is_initialized is True

    @pytest.mark.asyncio
    async def test_storage_create_program_insert_failure_rollback(
        self, storage
    ) -> None:
        """Prove transactions roll back correctly if an error is thrown in create_training_program."""
        orig_execute = storage._conn.execute

        class MockAiosqliteHelper:
            def __init__(self, ctx, force_fail=False):
                self.ctx = ctx
                self.force_fail = force_fail

            def __await__(self):
                return self._await_impl().__await__()

            async def _await_impl(self):
                if self.force_fail:
                    raise Exception("mock commit fail")
                return await self.ctx

            async def __aenter__(self):
                if self.force_fail:
                    raise Exception("mock commit fail")
                return await self.ctx.__aenter__()

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await self.ctx.__aexit__(exc_type, exc_val, exc_tb)

        def mock_execute(query, *args, **kwargs):
            force_fail = "INSERT INTO training_programs" in query
            ctx = orig_execute(query, *args, **kwargs)
            return MockAiosqliteHelper(ctx, force_fail=force_fail)

        with patch.object(storage._conn, "execute", side_effect=mock_execute):
            program = TrainingProgram(
                user_id="user1",
                name="Fail Program",
                starts_on="2026-06-07",
                created_at="2026-06-07T12:00:00",
                updated_at="2026-06-07T12:00:00",
            )

            with pytest.raises(Exception, match="mock commit fail"):
                await storage.create_training_program(program, 1, 1)

    @pytest.mark.asyncio
    async def test_storage_create_program_missing_lastrowid(self, storage) -> None:
        """RuntimeError should be raised if lastrowid is None after program insertion."""
        orig_execute = storage._conn.execute

        class MockCursor:
            def __init__(self, orig_cursor):
                self._orig = orig_cursor

            @property
            def lastrowid(self):
                return None

            def __getattr__(self, name):
                return getattr(self._orig, name)

        class MockAiosqliteHelper:
            def __init__(self, ctx, force_none_id=False):
                self.ctx = ctx
                self.force_none_id = force_none_id

            def __await__(self):
                return self._await_impl().__await__()

            async def _await_impl(self):
                cursor = await self.ctx
                if self.force_none_id:
                    return MockCursor(cursor)
                return cursor

            async def __aenter__(self):
                cursor = await self.ctx.__aenter__()
                if self.force_none_id:
                    return MockCursor(cursor)
                return cursor

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await self.ctx.__aexit__(exc_type, exc_val, exc_tb)

        def mock_execute(query, *args, **kwargs):
            force_none_id = "INSERT INTO training_programs" in query
            ctx = orig_execute(query, *args, **kwargs)
            return MockAiosqliteHelper(ctx, force_none_id=force_none_id)

        with patch.object(storage._conn, "execute", side_effect=mock_execute):
            program = TrainingProgram(
                user_id="user1",
                name="Fail Program",
                starts_on="2026-06-07",
                created_at="2026-06-07T12:00:00",
                updated_at="2026-06-07T12:00:00",
            )

            with pytest.raises(
                RuntimeError, match="Failed to get lastrowid after program insert"
            ):
                await storage.create_training_program(program, 1, 1)

    @pytest.mark.asyncio
    async def test_storage_create_metrics_missing_lastrowid(self, storage) -> None:
        """RuntimeError should be raised if lastrowid is None after metric insertion."""
        orig_execute = storage._conn.execute

        class MockCursor:
            def __init__(self, orig_cursor):
                self._orig = orig_cursor

            @property
            def lastrowid(self):
                return None

            def __getattr__(self, name):
                return getattr(self._orig, name)

        class MockAiosqliteHelper:
            def __init__(self, ctx, force_none_id=False):
                self.ctx = ctx
                self.force_none_id = force_none_id

            def __await__(self):
                return self._await_impl().__await__()

            async def _await_impl(self):
                cursor = await self.ctx
                if self.force_none_id:
                    return MockCursor(cursor)
                return cursor

            async def __aenter__(self):
                cursor = await self.ctx.__aenter__()
                if self.force_none_id:
                    return MockCursor(cursor)
                return cursor

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await self.ctx.__aexit__(exc_type, exc_val, exc_tb)

        def mock_execute(query, *args, **kwargs):
            force_none_id = "INSERT INTO training_metrics" in query
            ctx = orig_execute(query, *args, **kwargs)
            return MockAiosqliteHelper(ctx, force_none_id=force_none_id)

        with patch.object(storage._conn, "execute", side_effect=mock_execute):
            metric = TrainingMetric(
                user_id="user1",
                metric_name="bench",
                value=100.0,
                unit="kg",
                recorded_at="2026-06-07T12:00:00",
            )

            with pytest.raises(
                RuntimeError, match="Failed to get lastrowid after metric insert"
            ):
                await storage.add_training_metrics([metric])

    @pytest.mark.asyncio
    async def test_storage_get_active_program_not_found(self, storage) -> None:
        """Querying active program when missing should return None."""
        program = await storage.get_active_training_program("nonexistent")
        assert program is None

    @pytest.mark.asyncio
    async def test_storage_advance_state_no_program(self, storage) -> None:
        """Advancing training state when program is missing should return None."""
        state = await storage.advance_training_state("user1", 1, "2026-06-07T12:00:00")
        assert state is None

    @pytest.mark.asyncio
    async def test_storage_get_training_history_handles_deleted_sessions(
        self, storage
    ) -> None:
        """Verify get_training_history ignores sessions that return None from get_session."""
        # Insert a raw session that we will fail to load to hit session is None check
        await storage._conn.execute(
            """
            INSERT INTO workout_sessions (user_id, workout_date, split_name, created_at)
            VALUES ('user1', '2026-06-07', 'Legs', '2026-06-07T12:00:00')
            """
        )
        # Session ID is 1. If we query user2, get_session returns None because of user isolation.
        history = await storage.get_training_history("user2", limit=1)
        assert len(history) == 0

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_tool_log_training_parsing_shorthands(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Test optional and shorthand exercise formats in log_training."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.create_session.return_value = 1
        mock_storage.get_active_training_program.return_value = None
        mock_storage.get_training_history.return_value = []

        # Empty exercises
        result = await log_training(mock_tool_context, "rest", exercises=None)
        assert result["status"] == "success"

        # Shorthand sets list with single set dict
        exercises = [{"name": "bench", "sets": {"weight_kg": 100, "reps": 10}}]
        result = await log_training(
            mock_tool_context, "resistance", exercises=exercises
        )
        assert result["status"] == "success"

        # Invalid sets type
        exercises_invalid = [{"name": "bench", "sets": "invalid"}]
        result = await log_training(
            mock_tool_context, "resistance", exercises=exercises_invalid
        )
        assert result["status"] == "error"

        # Missing reps in sets
        exercises_missing = [{"name": "bench", "sets": [{"weight_kg": 100}]}]
        result = await log_training(
            mock_tool_context, "resistance", exercises=exercises_missing
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_tool_log_training_metrics_parsing_branches(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Test unit inferring and parsing dictionaries with units/notes in log_training/metrics."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.add_training_metrics.return_value = [1]

        # Metric units by name suffix
        metrics = {
            "cardio_bpm": 145.0,
            "hike_km": 5.2,
            "ruck_min": 90.0,
            "squat_1rm": 150.0,
            "random_metric": 42.0,
        }
        result = await update_training_metrics(mock_tool_context, metrics)
        assert result["status"] == "success"

        # Dictionary format with notes and unit override
        metrics_dict = {
            "bench_1rm": {"value": 105.0, "unit": "lbs", "notes": "sore shoulder"}
        }
        result = await update_training_metrics(mock_tool_context, metrics_dict)
        assert result["status"] == "success"

        # Metric parsing validations
        # Missing value
        assert (await update_training_metrics(mock_tool_context, {"bench_1rm": {}}))[
            "status"
        ] == "error"
        # Non-numeric
        assert (
            await update_training_metrics(mock_tool_context, {"bench_1rm": "invalid"})
        )["status"] == "error"
        # Empty metric dict
        assert (await update_training_metrics(mock_tool_context, {}))[
            "status"
        ] == "error"
        # Empty metric name
        assert (await update_training_metrics(mock_tool_context, {"": 10}))[
            "status"
        ] == "error"

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_get_todays_training_day_six_swap_branches(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Test day 6 branches where Day 4 log doesn't exist, or has different status, or day is missing."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage

        # Scenario A: Day 6 is current cycle day, but Day 4 has NO recorded session
        day = TrainingProgramDay(cycle_day=6, focus="VO2 Max", session_type="vo2")
        program = TrainingProgram(
            id=1,
            user_id="user1",
            name=" Rotating",
            starts_on="2026-06-07",
            created_at="2026-06-07T12:00:00",
            updated_at="2026-06-07T12:00:00",
            days=[day],
            state=TrainingProgramState(
                user_id="user1",
                program_id=1,
                current_cycle_day=6,
                current_mesocycle_week=1,
                updated_at="2026-06-12T12:00:00",
            ),
        )
        mock_storage.get_active_training_program.return_value = program
        mock_storage.get_training_history.return_value = []  # Day 4 returns empty list

        result = await get_todays_training(mock_tool_context)
        assert result["status"] == "success"
        assert len(result["recommendations"]) == 0

        # Scenario B: Day 4 log exists but back status is OK (no swap recommended)
        day_four_session = WorkoutSession(
            user_id="user1",
            workout_date="2026-06-10",
            split_name="Pull",
            created_at="2026-06-10T12:00:00",
            cycle_day=4,
            metrics={"lower_back_status": "ok"},
        )

        async def history_side_effect(
            user_id, cycle_day, session_type, limit=1, exercise_name=None
        ):
            if cycle_day == 4:
                return [day_four_session]
            return []

        mock_storage.get_training_history.side_effect = history_side_effect

        result = await get_todays_training(mock_tool_context)
        assert result["status"] == "success"
        assert len(result["recommendations"]) == 0

        # Scenario C: Active program exists but no matching day config for today's cycle day
        assert program.state is not None
        program.state.current_cycle_day = 10
        result = await get_todays_training(mock_tool_context)
        assert result["status"] == "error"

        # Scenario D: Active program day config exists for day 1 (non-6 day config to cover cycle_day != 6 branch on line 368)
        day_one = TrainingProgramDay(
            cycle_day=1, focus="Legs", session_type="resistance"
        )
        program.days.append(day_one)
        assert program.state is not None
        program.state.current_cycle_day = 1
        mock_storage.get_training_history.side_effect = None
        mock_storage.get_training_history.return_value = []
        result = await get_todays_training(mock_tool_context)
        assert result["status"] == "success"
        assert result["training_day"]["cycle_day"] == 1


class TestFullTestCoverageFillers:
    """Explicitly tests missing branch coverage to reach 100% codebase wide."""

    @pytest.mark.asyncio
    async def test_storage_ensure_columns_actually_upgrades(self, conn, lock) -> None:
        """Create a table without the new columns first, then ensure _ensure_columns adds them."""
        # Create workout_sessions table with ONLY the legacy columns
        await conn.execute("""
            CREATE TABLE workout_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                workout_date TEXT NOT NULL,
                split_name TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
        """)
        storage = SqliteWorkoutStorage(conn, lock)
        # Manually invoke ensure_columns (normally called inside _create_tables which would do IF NOT EXISTS)
        # This will meet the "not in existing" condition and trigger the ALTER TABLE statement (Line 300)
        await storage._ensure_columns(
            "workout_sessions",
            {
                "program_id": "INTEGER",
                "session_type": "TEXT NOT NULL DEFAULT 'resistance'",
            },
        )

        # Verify the columns are indeed added
        cursor = await conn.execute("PRAGMA table_info(workout_sessions)")
        rows = await cursor.fetchall()
        existing = {row[1] for row in rows}
        assert "program_id" in existing
        assert "session_type" in existing

    @pytest.mark.asyncio
    async def test_storage_get_training_history_no_session_ids(self, storage) -> None:
        """Ensure get_training_history handles sessions with None IDs correctly."""
        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-06-07",
            split_name="Legs",
            created_at="2026-06-07T12:00:00",
            cycle_day=1,
            session_type="resistance",
        )
        await storage.create_session(session)
        original_row_to_session = storage._row_to_session

        def mock_row_to_session(row):
            s = original_row_to_session(row)
            s.id = None
            return s

        with patch.object(storage, "_row_to_session", side_effect=mock_row_to_session):
            sessions = await storage.get_training_history("user1", limit=1)
            assert len(sessions) == 1
            assert sessions[0].id is None

    @pytest.mark.asyncio
    async def test_storage_get_training_history_mismatched_exercise_session_id(
        self, storage
    ) -> None:
        """Ensure get_training_history filters out exercises with mismatched session_ids."""
        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-06-07",
            split_name="Legs",
            created_at="2026-06-07T12:00:00",
            cycle_day=1,
            session_type="resistance",
        )
        await storage.create_session(session)
        original_fetch_all = storage._fetch_all

        async def mock_fetch_all(query, values):
            if "workout_exercises" in query:
                return [
                    {
                        "id": 99,
                        "session_id": 9999,
                        "exercise_name": "bench press",
                        "exercise_order": 0,
                        "sets": "[]",
                    }
                ]
            return await original_fetch_all(query, values)

        with patch.object(storage, "_fetch_all", side_effect=mock_fetch_all):
            sessions = await storage.get_training_history("user1", limit=1)
            assert len(sessions) == 1
            assert len(sessions[0].exercises) == 0

    @pytest.mark.asyncio
    async def test_storage_get_training_history_session_id_not_in_dict(
        self, storage
    ) -> None:
        """Cover session.id not in exercises_by_session dictionary."""
        session = WorkoutSession(
            user_id="user1",
            workout_date="2026-06-07",
            split_name="Legs",
            created_at="2026-06-07T12:00:00",
            cycle_day=1,
            session_type="resistance",
        )
        await storage.create_session(session)
        original_row_to_session = storage._row_to_session
        sessions_ref = []

        def mock_row_to_session(row):
            s = original_row_to_session(row)
            sessions_ref.append(s)
            return s

        original_fetch_all = storage._fetch_all

        async def mock_fetch_all(query, values):
            if "workout_exercises" in query:
                for s in sessions_ref:
                    s.id = 9999
                return []
            return await original_fetch_all(query, values)

        with (
            patch.object(storage, "_row_to_session", side_effect=mock_row_to_session),
            patch.object(storage, "_fetch_all", side_effect=mock_fetch_all),
        ):
            await storage.get_training_history("user1", limit=1)

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_defensive_type_validations(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Test defensive type checking added for LLM-provided arguments in tools."""
        from blacki.workouts.tools import (
            log_training,
            set_training_program,
            update_training_metrics,
        )

        # 1. exercises is not a list in log_training
        res = await log_training(
            mock_tool_context,
            "resistance",
            exercises="not a list",  # type: ignore[arg-type]
        )
        assert res["status"] == "error"
        assert "must be a list" in res["message"]

        # 2. exercise item is not a dict in log_training
        res = await log_training(
            mock_tool_context,
            "resistance",
            exercises=["not a dict"],  # type: ignore[list-item]
        )
        assert res["status"] == "error"
        assert "must be a dictionary" in res["message"]

        # 3. set detail is not a dict in sets list in log_training
        res = await log_training(
            mock_tool_context,
            "resistance",
            exercises=[{"name": "bench press", "sets": ["not a dict"]}],
        )
        assert res["status"] == "error"
        assert "must be a dictionary" in res["message"]

        # 4. metrics in log_training is not a dict
        res = await log_training(mock_tool_context, "resistance", metrics="not a dict")  # type: ignore[arg-type]
        assert res["status"] == "error"
        assert "must be a dictionary" in res["message"]

        # 5. metrics in update_training_metrics is not a dict
        res = await update_training_metrics(mock_tool_context, "not a dict")  # type: ignore[arg-type]
        assert res["status"] == "error"
        assert "must be a dictionary" in res["message"]

        # 6. program_config in set_training_program is not a dict
        res = await set_training_program(mock_tool_context, "not a dict")  # type: ignore[arg-type]
        assert res["status"] == "error"
        assert "must be a dictionary" in res["message"]

        # 7. baseline_metrics in set_training_program is not a dict
        res = await set_training_program(
            mock_tool_context,
            {},
            baseline_metrics="not a dict",  # type: ignore[arg-type]
        )
        assert res["status"] == "error"
        assert "must be a dictionary" in res["message"]

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_set_training_program_additional_validations(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Cover additional validation branches in set_training_program."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage

        # cycle_length_days < 1
        config = {
            "days": [{"cycle_day": 1, "session_type": "rest"}],
            "cycle_length_days": -5,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # current_cycle_day out of bounds
        config = {
            "days": [{"cycle_day": 1, "session_type": "rest"}],
            "cycle_length_days": 1,
            "current_cycle_day": 5,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # current_mesocycle_week < 1
        config = {
            "days": [{"cycle_day": 1, "session_type": "rest"}],
            "cycle_length_days": 1,
            "current_cycle_day": 1,
            "current_mesocycle_week": -5,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # exercises is not a list
        config = {
            "days": [{"cycle_day": 1, "session_type": "rest", "exercises": "not_list"}],
            "cycle_length_days": 1,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # rules is not a dict
        config = {
            "days": [{"cycle_day": 1, "session_type": "rest", "rules": "not_dict"}],
            "cycle_length_days": 1,
        }
        assert (await set_training_program(mock_tool_context, config))[
            "status"
        ] == "error"

        # metrics_config is None / empty dict (Line 318->326 False branch)
        config = {
            "days": [{"cycle_day": 1, "session_type": "rest"}],
            "cycle_length_days": 1,
            "baseline_metrics": None,
        }
        result = await set_training_program(mock_tool_context, config)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_log_training_additional_validations(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Cover validation and defaulting branches in log_training."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.create_session.return_value = 1
        mock_storage.get_training_history.return_value = []

        # session_type invalid
        assert (await log_training(mock_tool_context, "invalid_type"))[
            "status"
        ] == "error"

        # completion_status invalid
        assert (
            await log_training(mock_tool_context, "rest", completion_status="invalid")
        )["status"] == "error"

        invalid_date_result = await log_training(
            mock_tool_context,
            "rest",
            workout_date="definitely-not-a-real-date",
        )
        assert invalid_date_result["status"] == "error"
        assert "Could not understand date" in invalid_date_result["message"]

        # cycle_day is None, should default to active program state (Line 448)
        program = TrainingProgram(
            id=1,
            user_id="user1",
            name="Rotating Plan",
            starts_on="2026-06-07",
            created_at="2026-06-07T12:00:00",
            updated_at="2026-06-07T12:00:00",
            state=TrainingProgramState(
                user_id="user1",
                program_id=1,
                current_cycle_day=3,
                current_mesocycle_week=1,
                updated_at="2026-06-07T12:00:00",
            ),
        )
        mock_storage.get_active_training_program.return_value = program
        result = await log_training(mock_tool_context, "rest", cycle_day=None)
        assert result["status"] == "success"
        # Verify the logged session was created with the default cycle day 3
        logged_session = mock_storage.create_session.call_args[0][0]
        assert logged_session.cycle_day == 3

    @pytest.mark.asyncio
    @patch("blacki.workouts.tools.get_storage")
    async def test_advance_training_cycle_tool_success(
        self, mock_get_storage, mock_tool_context
    ) -> None:
        """Verify successful explicit cycle advancement tool execution."""
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        state = TrainingProgramState(
            user_id="user1",
            program_id=1,
            current_cycle_day=5,
            current_mesocycle_week=1,
            updated_at="2026-06-07T12:00:00",
        )
        mock_storage.advance_training_state.return_value = state

        result = await advance_training_cycle(mock_tool_context, days=2)
        assert result["status"] == "success"
        assert result["state"]["current_cycle_day"] == 5
        mock_storage.advance_training_state.assert_called_once()
