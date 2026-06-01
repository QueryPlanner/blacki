"""Persistent storage for workout tracking backed by SQLite."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from blacki.storage.base import SqlStorage

if TYPE_CHECKING:
    import asyncio

    import aiosqlite

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
    exercise_name: str
    sets: list[SetDetail]
    exercise_order: int = 0
    notes: str | None = None


class WorkoutSession(BaseModel):
    """A full workout session including exercises."""

    id: int | None = None
    user_id: str
    workout_date: str
    split_name: str
    notes: str | None = None
    created_at: str
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
    total_volume_kg: float


class SqliteWorkoutStorage(SqlStorage):
    """Storage for workout tracking using SQLite via aiosqlite."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        super().__init__(conn, lock)

    async def _create_tables(self) -> None:
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                workout_date TEXT NOT NULL,
                split_name TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_date
                ON workout_sessions (user_id, workout_date DESC)
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_split
                ON workout_sessions (user_id, split_name)
        """)

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS workout_exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                exercise_name TEXT NOT NULL,
                sets TEXT NOT NULL,
                exercise_order INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (session_id)
                    REFERENCES workout_sessions(id) ON DELETE CASCADE
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_exercises_session
                ON workout_exercises (session_id)
        """)

    async def create_session(self, session: WorkoutSession) -> int:
        """Create session row + all exercises atomically."""
        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                cursor = await self._conn.execute(
                    """
                    INSERT INTO workout_sessions
                        (user_id, workout_date, split_name, notes, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session.user_id,
                        session.workout_date,
                        session.split_name,
                        session.notes,
                        session.created_at,
                    ),
                )
                sid = cursor.lastrowid
                if sid is None:
                    raise RuntimeError("Failed to get lastrowid after session insert")

                for exercise in session.exercises:
                    sets_json = json.dumps([s.model_dump() for s in exercise.sets])
                    await self._conn.execute(
                        """
                        INSERT INTO workout_exercises
                            (session_id, exercise_name, sets, exercise_order, notes)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            sid,
                            exercise.exercise_name,
                            sets_json,
                            exercise.exercise_order,
                            exercise.notes,
                        ),
                    )
                await self._conn.execute("COMMIT")
                return sid
            except Exception:
                await self._conn.execute("ROLLBACK")
                raise

    async def add_exercise(self, session_id: int, exercise: WorkoutExercise) -> int:
        """Add one exercise to an existing session."""
        sets_json = json.dumps([s.model_dump() for s in exercise.sets])
        eid = await self._execute(
            """
            INSERT INTO workout_exercises
                (session_id, exercise_name, sets, exercise_order, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                exercise.exercise_name,
                sets_json,
                exercise.exercise_order,
                exercise.notes,
            ),
        )
        return eid

    async def update_exercise(
        self,
        exercise_id: int,
        user_id: str,
        sets: list[SetDetail] | None = None,
        notes: str | None = None,
    ) -> bool:
        """Update sets/notes for an exercise. Needs user_id for authorization."""
        owner_row = await self._fetch_one(
            """
            SELECT s.user_id FROM workout_sessions s
            JOIN workout_exercises e ON s.id = e.session_id
            WHERE e.id = ?
            """,
            (exercise_id,),
        )
        if owner_row is None or owner_row["user_id"] != user_id:
            return False

        updates: list[str] = []
        values: list[Any] = []
        if sets is not None:
            updates.append("sets = ?")
            values.append(json.dumps([s.model_dump() for s in sets]))
        if notes is not None:
            updates.append("notes = ?")
            values.append(notes)

        if not updates:
            return False

        updates_str = ", ".join(updates)
        values.append(exercise_id)
        query = f"UPDATE workout_exercises SET {updates_str} WHERE id = ?"  # noqa: S608

        async with self._lock:
            cursor = await self._conn.execute(query, values)
            return cursor.rowcount > 0

    async def delete_exercise(self, exercise_id: int, user_id: str) -> bool:
        """Remove one exercise from a session."""
        owner_row = await self._fetch_one(
            """
            SELECT s.user_id FROM workout_sessions s
            JOIN workout_exercises e ON s.id = e.session_id
            WHERE e.id = ?
            """,
            (exercise_id,),
        )
        if owner_row is None or owner_row["user_id"] != user_id:
            return False

        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM workout_exercises WHERE id = ?", (exercise_id,)
            )
            return cursor.rowcount > 0

    async def get_session(self, session_id: int, user_id: str) -> WorkoutSession | None:
        """Get full session with exercises."""
        row = await self._fetch_one(
            "SELECT * FROM workout_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        if not row:
            return None

        session = self._row_to_session(row)

        ex_rows = await self._fetch_all(
            """
            SELECT * FROM workout_exercises WHERE session_id = ?
            ORDER BY exercise_order ASC, id ASC
            """,
            (session_id,),
        )
        session.exercises = [self._row_to_exercise(r) for r in ex_rows]
        return session

    async def get_latest_split_session(
        self, user_id: str, split_name: str
    ) -> WorkoutSession | None:
        """Returns the most recent session for a given split."""
        row = await self._fetch_one(
            """
            SELECT * FROM workout_sessions
            WHERE user_id = ? AND split_name = ?
            ORDER BY workout_date DESC, created_at DESC
            LIMIT 1
            """,
            (user_id, split_name),
        )
        if not row:
            return None

        return await self.get_session(row["id"], user_id)

    async def get_recent_sessions(
        self, user_id: str, limit: int = 10
    ) -> list[WorkoutSessionSummary]:
        """Returns lightweight view of recent sessions."""
        limit = min(limit, 20)
        rows = await self._fetch_all(
            """
            SELECT s.id, s.workout_date, s.split_name, COUNT(e.id) as exercise_count
            FROM workout_sessions s
            LEFT JOIN workout_exercises e ON s.id = e.session_id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY s.workout_date DESC, s.created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
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
        limit = min(limit, 8)
        rows = await self._fetch_all(
            """
            SELECT s.workout_date, s.split_name, e.sets
            FROM workout_exercises e
            JOIN workout_sessions s ON e.session_id = s.id
            WHERE s.user_id = ? AND e.exercise_name = ?
            ORDER BY s.workout_date DESC, s.created_at DESC
            LIMIT ?
            """,
            (user_id, exercise_name.lower(), limit),
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
                if not s.is_warmup:
                    volume += s.weight_kg * s.reps
                    if s.weight_kg > best_weight or (
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
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM workout_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            return cursor.rowcount > 0

    def _row_to_session(self, row: dict[str, Any]) -> WorkoutSession:
        return WorkoutSession(
            id=int(row["id"]),
            user_id=row["user_id"],
            workout_date=str(row["workout_date"]),
            split_name=row["split_name"],
            notes=row["notes"],
            created_at=row["created_at"],
            exercises=[],
        )

    def _row_to_exercise(self, row: dict[str, Any]) -> WorkoutExercise:
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


_storage: SqliteWorkoutStorage | None = None


def get_storage() -> SqliteWorkoutStorage:
    """Return the process-wide singleton SqliteWorkoutStorage instance.

    Uses the AppContainer for dependency injection.
    """
    from blacki.container import get_container

    container = get_container()
    storage = container.workout_storage
    if not storage.is_initialized:
        raise RuntimeError(
            "Workout storage not initialized. Call storage.initialize() first."
        )
    return storage
