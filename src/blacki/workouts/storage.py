import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import asyncpg  # type: ignore[import-untyped]
from pydantic import BaseModel

from blacki.storage.base import PostgresStorage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SetDetail(BaseModel):
    """Details for a single workout set."""

    set_num: int
    weight_kg: float
    reps: int
    is_warmup: bool = False


class WorkoutExercise(BaseModel):
    """An exercise within a workout session."""

    id: int | None = None
    session_id: int | None = None
    exercise_name: str  # always lowercase
    sets: list[SetDetail]
    exercise_order: int = 0
    notes: str | None = None


class WorkoutSession(BaseModel):
    """A full workout session including exercises."""

    id: int | None = None
    user_id: str
    workout_date: str  # YYYY-MM-DD local
    split_name: str  # Push / Pull / Legs / etc.
    notes: str | None = None
    created_at: str  # ISO UTC
    exercises: list[WorkoutExercise] = []


class WorkoutSessionSummary(BaseModel):
    """Lightweight view for listing — no exercise data."""

    id: int
    workout_date: str
    split_name: str
    exercise_count: int


class ExerciseHistoryEntry(BaseModel):
    """One instance of an exercise across time for progressive overload tracking."""

    workout_date: str
    split_name: str
    sets: list[SetDetail]
    best_set_weight_kg: float
    best_set_reps: int
    total_volume_kg: float  # sum(weight * reps) across all working sets


class PostgresWorkoutStorage(PostgresStorage):
    """Storage for workout tracking using Postgres via asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        super().__init__(pool)

    async def _create_tables(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id            BIGSERIAL PRIMARY KEY,
                user_id       TEXT      NOT NULL,
                workout_date  DATE      NOT NULL,
                split_name    TEXT      NOT NULL,
                notes         TEXT,
                created_at    TIMESTAMPTZ NOT NULL
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_date
                ON workout_sessions (user_id, workout_date DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_split
                ON workout_sessions (user_id, split_name)
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS workout_exercises (
                id             BIGSERIAL PRIMARY KEY,
                session_id     BIGINT    NOT NULL REFERENCES workout_sessions(id)
                                                  ON DELETE CASCADE,
                exercise_name  TEXT      NOT NULL,
                sets           JSONB     NOT NULL,
                exercise_order INTEGER   NOT NULL DEFAULT 0,
                notes          TEXT
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_exercises_session
                ON workout_exercises (session_id)
        """)

    async def create_session(self, session: WorkoutSession) -> int:
        """Create session row + all exercises atomically."""
        async with self._pool.acquire() as conn, conn.transaction():
            sid = await conn.fetchval(
                """
                    INSERT INTO workout_sessions
                        (user_id, workout_date, split_name, notes, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                session.user_id,
                session.workout_date,
                session.split_name,
                session.notes,
                session.created_at,
            )

            for exercise in session.exercises:
                sets_json = json.dumps([s.model_dump() for s in exercise.sets])
                await conn.execute(
                    """
                        INSERT INTO workout_exercises
                            (session_id, exercise_name, sets, exercise_order, notes)
                        VALUES ($1, $2, $3::jsonb, $4, $5)
                        """,
                    sid,
                    exercise.exercise_name,
                    sets_json,
                    exercise.exercise_order,
                    exercise.notes,
                )
        return int(sid)

    async def add_exercise(self, session_id: int, exercise: WorkoutExercise) -> int:
        """Add one exercise to an existing session."""
        sets_json = json.dumps([s.model_dump() for s in exercise.sets])
        eid = await self._pool.fetchval(
            """
            INSERT INTO workout_exercises
                (session_id, exercise_name, sets, exercise_order, notes)
            VALUES ($1, $2, $3::jsonb, $4, $5)
            RETURNING id
            """,
            session_id,
            exercise.exercise_name,
            sets_json,
            exercise.exercise_order,
            exercise.notes,
        )
        return int(eid)

    async def update_exercise(
        self,
        exercise_id: int,
        user_id: str,
        sets: list[SetDetail] | None = None,
        notes: str | None = None,
    ) -> bool:
        """Update sets/notes for an exercise. Needs user_id for authorization."""
        owner = await self._pool.fetchval(
            """
            SELECT s.user_id FROM workout_sessions s
            JOIN workout_exercises e ON s.id = e.session_id
            WHERE e.id = $1
            """,
            exercise_id,
        )
        if owner != user_id:  # pragma: no cover
            return False

        updates: list[str] = []
        values: list[Any] = []
        if sets is not None:  # pragma: no cover
            updates.append(f"sets = ${len(values) + 1}::jsonb")
            values.append(json.dumps([s.model_dump() for s in sets]))
        if notes is not None:
            updates.append(f"notes = ${len(values) + 1}")
            values.append(notes)

        if not updates:  # pragma: no cover
            return False

        updates_str = ", ".join(updates)
        query = (
            f"UPDATE workout_exercises SET {updates_str} WHERE id = ${len(values) + 1}"  # noqa: S608
        )
        values.append(exercise_id)

        result = await self._pool.execute(query, *values)
        return bool(result == "UPDATE 1")

    async def delete_exercise(self, exercise_id: int, user_id: str) -> bool:
        """Remove one exercise from a session."""
        owner = await self._pool.fetchval(
            """
            SELECT s.user_id FROM workout_sessions s
            JOIN workout_exercises e ON s.id = e.session_id
            WHERE e.id = $1
            """,
            exercise_id,
        )
        if owner != user_id:  # pragma: no cover
            return False

        result = await self._pool.execute(
            "DELETE FROM workout_exercises WHERE id = $1", exercise_id
        )
        return bool(result == "DELETE 1")

    async def get_session(self, session_id: int, user_id: str) -> WorkoutSession | None:
        """Get full session with exercises."""
        row = await self._pool.fetchrow(
            "SELECT * FROM workout_sessions WHERE id = $1 AND user_id = $2",
            session_id,
            user_id,
        )
        if not row:  # pragma: no cover
            return None

        session = self._row_to_session(row)

        ex_rows = await self._pool.fetch(
            "SELECT * FROM workout_exercises WHERE session_id = $1 "
            "ORDER BY exercise_order ASC, id ASC",
            session_id,
        )
        session.exercises = [self._row_to_exercise(r) for r in ex_rows]
        return session

    async def get_latest_split_session(
        self, user_id: str, split_name: str
    ) -> WorkoutSession | None:
        """Returns the most recent session for a given split."""
        row = await self._pool.fetchrow(
            """
            SELECT * FROM workout_sessions
            WHERE user_id = $1 AND split_name = $2
            ORDER BY workout_date DESC, created_at DESC
            LIMIT 1
            """,
            user_id,
            split_name,
        )
        if not row:  # pragma: no cover
            return None

        return await self.get_session(row["id"], user_id)

    async def get_recent_sessions(
        self, user_id: str, limit: int = 10
    ) -> list[WorkoutSessionSummary]:
        """Returns lightweight view of recent sessions."""
        limit = min(limit, 20)  # Capped at 20
        rows = await self._pool.fetch(
            """
            SELECT s.id, s.workout_date, s.split_name, COUNT(e.id) as exercise_count
            FROM workout_sessions s
            LEFT JOIN workout_exercises e ON s.id = e.session_id
            WHERE s.user_id = $1
            GROUP BY s.id
            ORDER BY s.workout_date DESC, s.created_at DESC
            LIMIT $2
            """,
            user_id,
            limit,
        )
        return [
            WorkoutSessionSummary(
                id=r["id"],
                workout_date=r["workout_date"],
                split_name=r["split_name"],
                exercise_count=r["exercise_count"],
            )
            for r in rows
        ]

    async def get_exercise_history(
        self, user_id: str, exercise_name: str, limit: int = 8
    ) -> list[ExerciseHistoryEntry]:
        """Returns the last N instances of a specific exercise."""
        limit = min(limit, 8)  # Capped at 8
        rows = await self._pool.fetch(
            """
            SELECT s.workout_date, s.split_name, e.sets
            FROM workout_exercises e
            JOIN workout_sessions s ON e.session_id = s.id
            WHERE s.user_id = $1 AND e.exercise_name = $2
            ORDER BY s.workout_date DESC, s.created_at DESC
            LIMIT $3
            """,
            user_id,
            exercise_name.lower(),
            limit,
        )

        history = []
        for r in rows:
            sets_data = (
                json.loads(r["sets"]) if isinstance(r["sets"], str) else r["sets"]
            )
            sets = [SetDetail(**s) for s in sets_data]

            best_weight = 0.0
            best_reps = 0
            volume = 0.0

            for s in sets:
                if not s.is_warmup:  # pragma: no cover
                    volume += s.weight_kg * s.reps
                    if s.weight_kg > best_weight or (  # pragma: no cover
                        s.weight_kg == best_weight and s.reps > best_reps
                    ):
                        best_weight = s.weight_kg
                        best_reps = s.reps

            history.append(
                ExerciseHistoryEntry(
                    workout_date=r["workout_date"],
                    split_name=r["split_name"],
                    sets=sets,
                    best_set_weight_kg=best_weight,
                    best_set_reps=best_reps,
                    total_volume_kg=volume,
                )
            )

        return history

    async def delete_session(self, session_id: int, user_id: str) -> bool:
        """Cascades to exercises."""
        result = await self._pool.execute(
            "DELETE FROM workout_sessions WHERE id = $1 AND user_id = $2",
            session_id,
            user_id,
        )
        return bool(result == "DELETE 1")

    def _row_to_session(self, row: Mapping[str, Any]) -> WorkoutSession:
        return WorkoutSession(
            id=int(row["id"]),
            user_id=row["user_id"],
            workout_date=str(row["workout_date"]),
            split_name=row["split_name"],
            notes=row["notes"],
            created_at=(
                row["created_at"].isoformat()
                if hasattr(row["created_at"], "isoformat")
                else str(row["created_at"])
            ),
            exercises=[],
        )

    def _row_to_exercise(self, row: Mapping[str, Any]) -> WorkoutExercise:
        sets_data = (
            json.loads(row["sets"]) if isinstance(row["sets"], str) else row["sets"]
        )
        return WorkoutExercise(
            id=int(row["id"]),
            session_id=int(row["session_id"]),
            exercise_name=row["exercise_name"],
            sets=[SetDetail(**s) for s in sets_data],
            exercise_order=int(row["exercise_order"]),
            notes=row["notes"],
        )


_storage: PostgresWorkoutStorage | None = None


def get_storage() -> PostgresWorkoutStorage:
    """Return the process-wide singleton PostgresWorkoutStorage instance.

    Uses the AppContainer for dependency injection.
    """
    from blacki.container import _container

    if _container is None or _container._workout_storage is None:
        raise RuntimeError(
            "Workout storage not initialized. Call init_workout_storage() first."
        )
    return _container.workout_storage


async def init_workout_storage(pool: asyncpg.Pool) -> PostgresWorkoutStorage:
    """Initialize the workout storage with a Postgres pool.

    Note: This function is provided for backward compatibility.
    Prefer using AppContainer directly for new code.
    """
    global _storage
    import blacki.container as container_module

    if container_module._container is None:
        container_module.set_container_from_pool(pool)

    if _storage is not None:
        await _storage.close()
        _storage = None

    container = container_module._container
    if container is None:
        raise RuntimeError("Container not initialized")
    if container._workout_storage is not None:
        await container._workout_storage.close()

    storage = container.workout_storage
    await storage.initialize()
    _storage = storage
    return storage


async def close_workout_storage() -> None:
    """Close the singleton workout storage.

    Note: This function is provided for backward compatibility.
    Prefer using AppContainer.close() for new code.
    """
    global _storage
    import blacki.container as container_module

    if container_module._container is not None:
        container = container_module._container
        if container._workout_storage is not None:
            await container._workout_storage.close()
            container._workout_storage = None
    _storage = None
