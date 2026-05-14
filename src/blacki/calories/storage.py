import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import asyncpg  # type: ignore[import-untyped]
from pydantic import BaseModel

from blacki.storage.base import PostgresStorage

if TYPE_CHECKING:
    pass

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
    meal_type: str | None = None  # breakfast/lunch/dinner/snack
    logged_at: str  # UTC ISO
    logged_date: str  # YYYY-MM-DD local


class DailySummary(BaseModel):
    """Summary of calorie intake for a specific date."""

    date: str  # YYYY-MM-DD
    total_calories: int = 0
    total_protein_g: float | None = None
    total_carbs_g: float | None = None
    total_fat_g: float | None = None
    entry_count: int = 0
    entries: list[CalorieEntry] = []  # populated only in single-day queries


class PostgresCalorieStorage(PostgresStorage):
    """Storage for calorie tracking using Postgres via asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        super().__init__(pool)

    async def _create_tables(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS calorie_logs (
                id            BIGSERIAL PRIMARY KEY,
                user_id       TEXT      NOT NULL,
                description   TEXT      NOT NULL,
                calories      INTEGER   NOT NULL,
                protein_g     REAL,
                carbs_g       REAL,
                fat_g         REAL,
                meal_type     TEXT,
                logged_at     TIMESTAMPTZ NOT NULL,
                logged_date   DATE      NOT NULL
            )
        """)
        column_type = await conn.fetchval("""
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = 'calorie_logs' AND column_name = 'protein_g'
        """)
        if column_type == "integer":
            await conn.execute("""
                ALTER TABLE calorie_logs
                ALTER COLUMN protein_g TYPE REAL USING protein_g::REAL,
                ALTER COLUMN carbs_g TYPE REAL USING carbs_g::REAL,
                ALTER COLUMN fat_g TYPE REAL USING fat_g::REAL;
            """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calorie_logs_user_date
                ON calorie_logs (user_id, logged_date)
        """)

    async def add_entry(self, entry: CalorieEntry) -> int:
        """Insert a calorie entry and return its new row ID."""
        rid = await self._pool.fetchval(
            """
            INSERT INTO calorie_logs
                (
                    user_id, description, calories, protein_g, carbs_g, fat_g,
                    meal_type, logged_at, logged_date
                )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id

            """,
            entry.user_id,
            entry.description,
            entry.calories,
            entry.protein_g,
            entry.carbs_g,
            entry.fat_g,
            entry.meal_type,
            entry.logged_at,
            entry.logged_date,
        )
        return int(rid)

    async def get_daily_summary(self, user_id: str, date_str: str) -> DailySummary:
        """Get summary and up to 50 entries for a specific day."""
        rows = await self._pool.fetch(
            """
            SELECT * FROM calorie_logs
            WHERE user_id = $1 AND logged_date = $2
            ORDER BY logged_at ASC
            """,
            user_id,
            date_str,
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

        if not has_protein:  # pragma: no cover
            summary.total_protein_g = None
        if not has_carbs:  # pragma: no cover
            summary.total_carbs_g = None
        if not has_fat:  # pragma: no cover
            summary.total_fat_g = None

        return summary

    async def get_date_range_summary(
        self, user_id: str, start_date: str, end_date: str
    ) -> list[DailySummary]:
        """Get summaries for a date range, capped at 30 days (no individual entries)."""
        rows = await self._pool.fetch(
            """
            SELECT
                logged_date,
                COUNT(*) as entry_count,
                SUM(calories) as total_calories,
                SUM(protein_g) as total_protein_g,
                SUM(carbs_g) as total_carbs_g,
                SUM(fat_g) as total_fat_g
            FROM calorie_logs
            WHERE user_id = $1 AND logged_date >= $2 AND logged_date <= $3
            GROUP BY logged_date
            ORDER BY logged_date DESC
            LIMIT 30
            """,
            user_id,
            start_date,
            end_date,
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
        if not fields:  # pragma: no cover
            return False

        set_clauses = []
        values: list[Any] = [entry_id, user_id]

        for i, (key, value) in enumerate(fields.items(), start=3):
            if key not in _ALLOWED_UPDATE_COLUMNS:
                raise ValueError(
                    f"Column '{key}' is not allowed in calorie_logs UPDATE"
                )
            set_clauses.append(f"{key} = ${i}")
            values.append(value)

        updates_str = ", ".join(set_clauses)
        query = f"UPDATE calorie_logs SET {updates_str} WHERE id = $1 AND user_id = $2"  # noqa: S608

        result = await self._pool.execute(query, *values)
        return bool(result == "UPDATE 1")

    async def delete_entry(self, entry_id: int, user_id: str) -> bool:
        """Delete a calorie entry."""
        result = await self._pool.execute(
            "DELETE FROM calorie_logs WHERE id = $1 AND user_id = $2",
            entry_id,
            user_id,
        )
        return bool(result == "DELETE 1")

    def _row_to_entry(self, row: Mapping[str, Any]) -> CalorieEntry:
        return CalorieEntry(
            id=int(row["id"]),
            user_id=row["user_id"],
            description=row["description"],
            calories=int(row["calories"]),
            protein_g=float(row["protein_g"]) if row["protein_g"] is not None else None,
            carbs_g=float(row["carbs_g"]) if row["carbs_g"] is not None else None,
            fat_g=float(row["fat_g"]) if row["fat_g"] is not None else None,
            meal_type=row["meal_type"],
            logged_at=(
                row["logged_at"].isoformat()
                if hasattr(row["logged_at"], "isoformat")
                else str(row["logged_at"])
            ),
            logged_date=str(row["logged_date"]),
        )


_storage: PostgresCalorieStorage | None = None


def get_storage() -> PostgresCalorieStorage:
    """Return the process-wide singleton PostgresCalorieStorage instance.

    Uses the AppContainer for dependency injection.
    """
    from blacki.container import get_container

    container = get_container()
    storage = container.calorie_storage
    if not storage.is_initialized:
        raise RuntimeError(
            "Calorie storage not initialized. Call init_calorie_storage() first."
        )
    return storage


async def init_calorie_storage(pool: asyncpg.Pool) -> PostgresCalorieStorage:
    """Initialize the calorie storage with a Postgres pool.

    Note: This function is provided for backward compatibility.
    Prefer using AppContainer directly for new code.
    """
    global _storage
    import blacki.container as container_module

    if container_module._container is None:  # pragma: no cover
        container_module.set_container_from_pool(pool)

    if _storage is not None:
        await _storage.close()
        _storage = None

    container = container_module._container
    if container is None:  # pragma: no cover
        raise RuntimeError("Container not initialized")
    if container._calorie_storage is not None:  # pragma: no cover
        await container._calorie_storage.close()

    storage = container.calorie_storage
    await storage.initialize()
    _storage = storage
    return storage


async def close_calorie_storage() -> None:
    """Close the singleton calorie storage.

    Note: This function is provided for backward compatibility.
    Prefer using AppContainer.close() for new code.
    """
    global _storage
    import blacki.container as container_module

    if container_module._container is not None:  # pragma: no cover
        container = container_module._container
        if container._calorie_storage is not None:
            await container._calorie_storage.close()
            container._calorie_storage = None
    _storage = None
