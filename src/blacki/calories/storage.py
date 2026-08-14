"""Persistent storage for calorie tracking backed by SQLite."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from blacki.storage.base import SqlStorage

if TYPE_CHECKING:
    import asyncio

    import aiosqlite

_ALLOWED_UPDATE_COLUMNS = frozenset(
    {
        "description",
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "meal_type",
        "logged_at",
        "logged_date",
    }
)

logger = logging.getLogger(__name__)


class CalorieEntry(BaseModel):
    """A single calorie log entry."""

    id: int | None = None
    user_id: str
    description: str
    calories: int
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    meal_type: str | None = None
    logged_at: str
    logged_date: str


class DailySummary(BaseModel):
    """Summary of calorie intake for a specific date."""

    date: str
    total_calories: int = 0
    total_protein_g: float | None = None
    total_carbs_g: float | None = None
    total_fat_g: float | None = None
    entry_count: int = 0
    entries: list[CalorieEntry] = []


class SqliteCalorieStorage(SqlStorage):
    """Storage for calorie tracking using SQLite via aiosqlite."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        super().__init__(conn, lock)

    async def _create_tables(self) -> None:
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS calorie_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                description TEXT NOT NULL,
                calories INTEGER NOT NULL,
                protein_g REAL,
                carbs_g REAL,
                fat_g REAL,
                meal_type TEXT,
                logged_at TEXT NOT NULL,
                logged_date TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calorie_logs_user_date
                ON calorie_logs (user_id, logged_date)
        """)

    async def add_entry(self, entry: CalorieEntry) -> int:
        """Insert a calorie entry and return its new row ID."""
        rid = await self._execute(
            """
            INSERT INTO calorie_logs
                (
                    user_id, description, calories, protein_g, carbs_g, fat_g,
                    meal_type, logged_at, logged_date
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.user_id,
                entry.description,
                entry.calories,
                entry.protein_g,
                entry.carbs_g,
                entry.fat_g,
                entry.meal_type,
                entry.logged_at,
                entry.logged_date,
            ),
        )
        return rid

    async def get_daily_summary(self, user_id: str, date_str: str) -> DailySummary:
        """Get summary and up to 50 entries for a specific day."""
        rows = await self._fetch_all(
            """
            SELECT * FROM calorie_logs
            WHERE user_id = ? AND logged_date = ?
            ORDER BY logged_at ASC
            """,
            (user_id, date_str),
        )

        entries = [self._row_to_entry(r) for r in rows]

        summary = DailySummary(date=date_str, entry_count=len(entries), entries=entries)

        has_protein = False
        has_carbs = False
        has_fat = False

        summary.total_protein_g = 0
        summary.total_carbs_g = 0
        summary.total_fat_g = 0

        for e in entries:
            summary.total_calories += e.calories
            if e.protein_g is not None:
                has_protein = True
                summary.total_protein_g += e.protein_g
            if e.carbs_g is not None:
                has_carbs = True
                summary.total_carbs_g += e.carbs_g
            if e.fat_g is not None:
                has_fat = True
                summary.total_fat_g += e.fat_g

        if not has_protein:
            summary.total_protein_g = None
        if not has_carbs:
            summary.total_carbs_g = None
        if not has_fat:
            summary.total_fat_g = None

        return summary

    async def get_date_range_summary(
        self, user_id: str, start_date: str, end_date: str
    ) -> list[DailySummary]:
        """Get summaries for a date range, capped at 30 days (no individual entries)."""
        rows = await self._fetch_all(
            """
            SELECT
                logged_date,
                COUNT(*) as entry_count,
                SUM(calories) as total_calories,
                SUM(protein_g) as total_protein_g,
                SUM(carbs_g) as total_carbs_g,
                SUM(fat_g) as total_fat_g
            FROM calorie_logs
            WHERE user_id = ? AND logged_date >= ? AND logged_date <= ?
            GROUP BY logged_date
            ORDER BY logged_date DESC
            LIMIT 30
            """,
            (user_id, start_date, end_date),
        )

        summaries = []
        for r in rows:
            summaries.append(
                DailySummary(
                    date=str(r["logged_date"]),
                    total_calories=int(r["total_calories"])
                    if r["total_calories"] is not None
                    else 0,
                    total_protein_g=float(r["total_protein_g"])
                    if r["total_protein_g"] is not None
                    else None,
                    total_carbs_g=float(r["total_carbs_g"])
                    if r["total_carbs_g"] is not None
                    else None,
                    total_fat_g=float(r["total_fat_g"])
                    if r["total_fat_g"] is not None
                    else None,
                    entry_count=int(r["entry_count"]),
                    entries=[],
                )
            )
        return summaries

    async def update_entry(self, entry_id: int, user_id: str, **fields: Any) -> bool:
        """Update a specific calorie entry."""
        if not fields:
            return False

        set_clauses = []
        values: list[Any] = []

        for key, value in fields.items():
            if key not in _ALLOWED_UPDATE_COLUMNS:
                raise ValueError(
                    f"Column '{key}' is not allowed in calorie_logs UPDATE"
                )
            set_clauses.append(f"{key} = ?")
            values.append(value)

        values.extend([entry_id, user_id])
        updates_str = ", ".join(set_clauses)
        query = f"UPDATE calorie_logs SET {updates_str} WHERE id = ? AND user_id = ?"  # noqa: S608

        async with self._lock:
            cursor = await self._conn.execute(query, values)
            return cursor.rowcount > 0

    async def delete_entry(self, entry_id: int, user_id: str) -> bool:
        """Delete a calorie entry."""
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM calorie_logs WHERE id = ? AND user_id = ?",
                (entry_id, user_id),
            )
            return cursor.rowcount > 0

    def _row_to_entry(self, row: dict[str, Any]) -> CalorieEntry:
        return CalorieEntry(
            id=int(row["id"]),
            user_id=row["user_id"],
            description=row["description"],
            calories=int(row["calories"]),
            protein_g=float(row["protein_g"]) if row["protein_g"] is not None else None,
            carbs_g=float(row["carbs_g"]) if row["carbs_g"] is not None else None,
            fat_g=float(row["fat_g"]) if row["fat_g"] is not None else None,
            meal_type=row["meal_type"],
            logged_at=row["logged_at"],
            logged_date=str(row["logged_date"]),
        )


def get_storage() -> SqliteCalorieStorage:
    """Return the process-wide singleton SqliteCalorieStorage instance.

    Uses the AppContainer for dependency injection.
    """
    from blacki.container import get_container

    container = get_container()
    storage = container.calorie_storage
    if not storage.is_initialized:
        raise RuntimeError(
            "Calorie storage not initialized. Call storage.initialize() first."
        )
    return storage
