"""Persistent storage for workout tracking backed by SQLite."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from blacki.storage.base import SqlStorage

if TYPE_CHECKING:
    import asyncio

    import aiosqlite

logger = logging.getLogger(__name__)

SESSION_TYPES = frozenset(
    {
        "resistance",
        "zone2",
        "vo2",
        "sugarcane",
        "ruck",
        "recovery",
        "rest",
        "mobility",
        "other",
    }
)
COMPLETION_STATUSES = frozenset({"planned", "completed", "partial", "skipped"})


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
    sets: list[SetDetail] = Field(default_factory=list)
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
    exercises: list[WorkoutExercise] = Field(default_factory=list)
    program_id: int | None = None
    program_version: int | None = None
    cycle_day: int | None = None
    session_type: str = "resistance"
    completion_status: str = "completed"
    metrics: dict[str, Any] = Field(default_factory=dict)


class WorkoutSessionSummary(BaseModel):
    """Lightweight view for listing — no exercise data."""

    id: int
    workout_date: str
    split_name: str
    exercise_count: int
    cycle_day: int | None = None
    session_type: str | None = None
    completion_status: str = "completed"


class ExerciseHistoryEntry(BaseModel):
    """One instance of an exercise across time for progressive overload tracking."""

    workout_date: str
    split_name: str
    sets: list[SetDetail]
    best_set_weight_kg: float
    best_set_reps: int
    total_volume_kg: float


class TrainingProgramDay(BaseModel):
    """One scheduled day in a rotating training program."""

    id: int | None = None
    program_id: int | None = None
    cycle_day: int
    focus: str
    session_type: str
    prescription: str | None = None
    modality: str | None = None
    target_zone: str | None = None
    target_duration_min: int | None = None
    exercises: list[dict[str, Any]] = Field(default_factory=list)
    rules: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class TrainingProgramState(BaseModel):
    """Current pointer for a user's active training program."""

    user_id: str
    program_id: int
    current_cycle_day: int
    current_mesocycle_week: int
    updated_at: str


class TrainingProgram(BaseModel):
    """A rotating training program and its scheduled days."""

    id: int | None = None
    user_id: str
    name: str
    cycle_length_days: int = 14
    mesocycle_length_days: int = 28
    deload_week_interval: int = 5
    starts_on: str
    version: int = 1
    is_active: bool = True
    notes: str | None = None
    created_at: str
    updated_at: str
    days: list[TrainingProgramDay] = Field(default_factory=list)
    state: TrainingProgramState | None = None


class TrainingMetric(BaseModel):
    """A user training metric measurement, stored as history."""

    id: int | None = None
    user_id: str
    metric_name: str
    value: float
    unit: str
    recorded_at: str
    notes: str | None = None


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
                created_at TEXT NOT NULL,
                program_id INTEGER,
                program_version INTEGER,
                cycle_day INTEGER,
                session_type TEXT NOT NULL DEFAULT 'resistance',
                completion_status TEXT NOT NULL DEFAULT 'completed',
                metrics TEXT NOT NULL DEFAULT '{}'
            )
        """)
        await self._ensure_columns(
            "workout_sessions",
            {
                "program_id": "INTEGER",
                "program_version": "INTEGER",
                "cycle_day": "INTEGER",
                "session_type": "TEXT NOT NULL DEFAULT 'resistance'",
                "completion_status": "TEXT NOT NULL DEFAULT 'completed'",
                "metrics": "TEXT NOT NULL DEFAULT '{}'",
            },
        )
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_date
                ON workout_sessions (user_id, workout_date DESC)
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_split
                ON workout_sessions (user_id, split_name)
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_cycle
                ON workout_sessions (user_id, cycle_day, session_type)
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

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS training_programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                cycle_length_days INTEGER NOT NULL,
                mesocycle_length_days INTEGER NOT NULL,
                deload_week_interval INTEGER NOT NULL,
                starts_on TEXT NOT NULL,
                version INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_programs_user_active
                ON training_programs (user_id, is_active, version DESC)
        """)

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS training_program_days (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_id INTEGER NOT NULL,
                cycle_day INTEGER NOT NULL,
                focus TEXT NOT NULL,
                session_type TEXT NOT NULL,
                prescription TEXT,
                modality TEXT,
                target_zone TEXT,
                target_duration_min INTEGER,
                exercises TEXT NOT NULL DEFAULT '[]',
                rules TEXT NOT NULL DEFAULT '{}',
                notes TEXT,
                UNIQUE(program_id, cycle_day),
                FOREIGN KEY (program_id)
                    REFERENCES training_programs(id) ON DELETE CASCADE
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_program_days_program_cycle
                ON training_program_days (program_id, cycle_day)
        """)

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS training_program_state (
                user_id TEXT PRIMARY KEY,
                program_id INTEGER NOT NULL,
                current_cycle_day INTEGER NOT NULL,
                current_mesocycle_week INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (program_id)
                    REFERENCES training_programs(id) ON DELETE CASCADE
            )
        """)

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                notes TEXT
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_metrics_user_name_time
                ON training_metrics (user_id, metric_name, recorded_at DESC)
        """)

    async def _ensure_columns(self, table_name: str, columns: dict[str, str]) -> None:
        cursor = await self._conn.execute(f"PRAGMA table_info({table_name})")  # noqa: S608
        rows = await cursor.fetchall()
        existing = {row[1] for row in rows}
        for column_name, column_sql in columns.items():
            if column_name not in existing:
                await self._conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"  # noqa: S608
                )

    async def create_session(self, session: WorkoutSession) -> int:
        """Create session row + all exercises atomically."""
        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                cursor = await self._conn.execute(
                    """
                    INSERT INTO workout_sessions
                        (
                            user_id, workout_date, split_name, notes, created_at,
                            program_id, program_version, cycle_day, session_type,
                            completion_status, metrics
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.user_id,
                        session.workout_date,
                        session.split_name,
                        session.notes,
                        session.created_at,
                        session.program_id,
                        session.program_version,
                        session.cycle_day,
                        session.session_type,
                        session.completion_status,
                        json.dumps(session.metrics),
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
            SELECT
                s.id,
                s.workout_date,
                s.split_name,
                s.cycle_day,
                s.session_type,
                s.completion_status,
                COUNT(e.id) as exercise_count
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
                cycle_day=r["cycle_day"],
                session_type=r["session_type"],
                completion_status=r["completion_status"],
            )
            for r in rows
        ]

    async def create_training_program(
        self,
        program: TrainingProgram,
        current_cycle_day: int,
        current_mesocycle_week: int,
    ) -> int:
        """Create a new active training program and its cycle pointer."""
        version_row = await self._fetch_one(
            """
            SELECT COALESCE(MAX(version), 0) + 1 as next_version
            FROM training_programs
            WHERE user_id = ?
            """,
            (program.user_id,),
        )
        version = int(version_row["next_version"]) if version_row else 1

        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                await self._conn.execute(
                    """
                    UPDATE training_programs
                    SET is_active = 0, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (program.updated_at, program.user_id),
                )
                cursor = await self._conn.execute(
                    """
                    INSERT INTO training_programs
                        (
                            user_id, name, cycle_length_days, mesocycle_length_days,
                            deload_week_interval, starts_on, version, is_active,
                            notes, created_at, updated_at
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        program.user_id,
                        program.name,
                        program.cycle_length_days,
                        program.mesocycle_length_days,
                        program.deload_week_interval,
                        program.starts_on,
                        version,
                        1,
                        program.notes,
                        program.created_at,
                        program.updated_at,
                    ),
                )
                program_id = cursor.lastrowid
                if program_id is None:
                    raise RuntimeError("Failed to get lastrowid after program insert")

                for day in program.days:
                    await self._conn.execute(
                        """
                        INSERT INTO training_program_days
                            (
                                program_id, cycle_day, focus, session_type,
                                prescription, modality, target_zone,
                                target_duration_min, exercises, rules, notes
                            )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            program_id,
                            day.cycle_day,
                            day.focus,
                            day.session_type,
                            day.prescription,
                            day.modality,
                            day.target_zone,
                            day.target_duration_min,
                            json.dumps(day.exercises),
                            json.dumps(day.rules),
                            day.notes,
                        ),
                    )

                await self._conn.execute(
                    """
                    INSERT INTO training_program_state
                        (
                            user_id, program_id, current_cycle_day,
                            current_mesocycle_week, updated_at
                        )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        program_id = excluded.program_id,
                        current_cycle_day = excluded.current_cycle_day,
                        current_mesocycle_week = excluded.current_mesocycle_week,
                        updated_at = excluded.updated_at
                    """,
                    (
                        program.user_id,
                        program_id,
                        current_cycle_day,
                        current_mesocycle_week,
                        program.updated_at,
                    ),
                )
                await self._conn.execute("COMMIT")
                return int(program_id)
            except Exception:
                await self._conn.execute("ROLLBACK")
                raise

    async def get_active_training_program(self, user_id: str) -> TrainingProgram | None:
        """Return the active training program with days and state."""
        row = await self._fetch_one(
            """
            SELECT * FROM training_programs
            WHERE user_id = ? AND is_active = 1
            ORDER BY version DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        )
        if row is None:
            return None

        program = self._row_to_training_program(row)
        if program.id is None:  # pragma: no cover
            return program

        day_rows = await self._fetch_all(
            """
            SELECT * FROM training_program_days
            WHERE program_id = ?
            ORDER BY cycle_day ASC
            """,
            (program.id,),
        )
        program.days = [self._row_to_training_program_day(r) for r in day_rows]
        program.state = await self.get_training_state(user_id)
        return program

    async def get_training_state(self, user_id: str) -> TrainingProgramState | None:
        """Return the current training cycle pointer for a user."""
        row = await self._fetch_one(
            "SELECT * FROM training_program_state WHERE user_id = ?",
            (user_id,),
        )
        return self._row_to_training_state(row) if row else None

    async def advance_training_state(
        self, user_id: str, days: int, updated_at: str
    ) -> TrainingProgramState | None:
        """Advance the active program pointer by a number of calendar days."""
        program = await self.get_active_training_program(user_id)
        if program is None or program.state is None or program.id is None:
            return None

        current_day = program.state.current_cycle_day
        current_week = program.state.current_mesocycle_week
        for _ in range(days):
            if current_day % 7 == 0:
                current_week += 1
                if current_week > program.deload_week_interval:
                    current_week = 1
            current_day = (
                current_day + 1 if current_day < program.cycle_length_days else 1
            )

        async with self._lock:
            await self._conn.execute(
                """
                UPDATE training_program_state
                SET current_cycle_day = ?, current_mesocycle_week = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (current_day, current_week, updated_at, user_id),
            )

        return TrainingProgramState(
            user_id=user_id,
            program_id=program.id,
            current_cycle_day=current_day,
            current_mesocycle_week=current_week,
            updated_at=updated_at,
        )

    async def get_training_history(
        self,
        user_id: str,
        cycle_day: int | None = None,
        session_type: str | None = None,
        exercise_name: str | None = None,
        limit: int = 8,
    ) -> list[WorkoutSession]:
        """Return comparable training sessions by cycle day, type, or exercise."""
        limit = min(max(limit, 1), 20)
        joins = ""
        where = ["s.user_id = ?"]
        values: list[Any] = [user_id]
        if exercise_name:
            joins = "JOIN workout_exercises e ON e.session_id = s.id"
            where.append("e.exercise_name = ?")
            values.append(exercise_name.lower())
        if cycle_day is not None:
            where.append("s.cycle_day = ?")
            values.append(cycle_day)
        if session_type is not None:
            where.append("s.session_type = ?")
            values.append(session_type)

        values.append(limit)
        query = f"""
            SELECT DISTINCT s.*
            FROM workout_sessions s
            {joins}
            WHERE {" AND ".join(where)}
            ORDER BY s.workout_date DESC, s.created_at DESC
            LIMIT ?
        """  # noqa: S608
        rows = await self._fetch_all(query, tuple(values))
        sessions = []
        for row in rows:
            session = await self.get_session(int(row["id"]), user_id)
            if session is not None:
                sessions.append(session)
        return sessions

    async def add_training_metrics(self, metrics: list[TrainingMetric]) -> list[int]:
        """Insert metric history rows and return their IDs."""
        ids = []
        async with self._lock:
            for metric in metrics:
                cursor = await self._conn.execute(
                    """
                    INSERT INTO training_metrics
                        (user_id, metric_name, value, unit, recorded_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metric.user_id,
                        metric.metric_name,
                        metric.value,
                        metric.unit,
                        metric.recorded_at,
                        metric.notes,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("Failed to get lastrowid after metric insert")
                ids.append(cursor.lastrowid)
        return ids

    async def get_latest_training_metrics(
        self, user_id: str, metric_names: list[str] | None = None
    ) -> list[TrainingMetric]:
        """Return the latest row for each requested metric name."""
        values: list[Any] = [user_id]
        where = "WHERE user_id = ?"
        if metric_names:
            placeholders = ", ".join("?" for _ in metric_names)
            where = f"{where} AND metric_name IN ({placeholders})"
            values.extend(metric_names)

        rows = await self._fetch_all(
            f"""
            SELECT * FROM training_metrics
            {where}
            ORDER BY metric_name ASC, recorded_at DESC, id DESC
            """,  # noqa: S608
            tuple(values),
        )
        latest_by_name: dict[str, TrainingMetric] = {}
        for row in rows:
            name = row["metric_name"]
            if name not in latest_by_name:
                latest_by_name[name] = self._row_to_training_metric(row)
        return list(latest_by_name.values())

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
        metrics_data = row.get("metrics") or "{}"
        metrics = (
            json.loads(metrics_data) if isinstance(metrics_data, str) else metrics_data
        )
        return WorkoutSession(
            id=int(row["id"]),
            user_id=row["user_id"],
            workout_date=str(row["workout_date"]),
            split_name=row["split_name"],
            notes=row["notes"],
            created_at=row["created_at"],
            exercises=[],
            program_id=row.get("program_id"),
            program_version=row.get("program_version"),
            cycle_day=row.get("cycle_day"),
            session_type=row.get("session_type") or "resistance",
            completion_status=row.get("completion_status") or "completed",
            metrics=metrics,
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

    def _row_to_training_program(self, row: dict[str, Any]) -> TrainingProgram:
        return TrainingProgram(
            id=int(row["id"]),
            user_id=row["user_id"],
            name=row["name"],
            cycle_length_days=int(row["cycle_length_days"]),
            mesocycle_length_days=int(row["mesocycle_length_days"]),
            deload_week_interval=int(row["deload_week_interval"]),
            starts_on=str(row["starts_on"]),
            version=int(row["version"]),
            is_active=bool(row["is_active"]),
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            days=[],
            state=None,
        )

    def _row_to_training_program_day(self, row: dict[str, Any]) -> TrainingProgramDay:
        exercises_data = row.get("exercises") or "[]"
        rules_data = row.get("rules") or "{}"
        exercises = (
            json.loads(exercises_data)
            if isinstance(exercises_data, str)
            else exercises_data
        )
        rules = json.loads(rules_data) if isinstance(rules_data, str) else rules_data
        return TrainingProgramDay(
            id=int(row["id"]),
            program_id=int(row["program_id"]),
            cycle_day=int(row["cycle_day"]),
            focus=row["focus"],
            session_type=row["session_type"],
            prescription=row["prescription"],
            modality=row["modality"],
            target_zone=row["target_zone"],
            target_duration_min=row["target_duration_min"],
            exercises=exercises,
            rules=rules,
            notes=row["notes"],
        )

    def _row_to_training_state(self, row: dict[str, Any]) -> TrainingProgramState:
        return TrainingProgramState(
            user_id=row["user_id"],
            program_id=int(row["program_id"]),
            current_cycle_day=int(row["current_cycle_day"]),
            current_mesocycle_week=int(row["current_mesocycle_week"]),
            updated_at=row["updated_at"],
        )

    def _row_to_training_metric(self, row: dict[str, Any]) -> TrainingMetric:
        return TrainingMetric(
            id=int(row["id"]),
            user_id=row["user_id"],
            metric_name=row["metric_name"],
            value=float(row["value"]),
            unit=row["unit"],
            recorded_at=row["recorded_at"],
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
