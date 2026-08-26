"""Atomic local meal mutations and optional Google Health enrollment."""

from __future__ import annotations

import contextlib
import math
from datetime import UTC, datetime, time, timedelta
from typing import Any

from blacki.container import AppContainer, get_container
from blacki.health.config import (
    GOOGLE_HEALTH_NUTRITION_SCOPES,
    health_user_id_for_telegram_user,
)
from blacki.health.nutrition_storage import NutritionStorage
from blacki.utils.timezone import get_app_timezone

from .storage import CalorieEntry

VALID_MEAL_TYPES = frozenset({"breakfast", "lunch", "dinner", "snack"})


def validate_meal(entry: CalorieEntry) -> None:
    """Validate fields shared by create and edit mutations."""
    if not entry.description.strip():
        raise ValueError("description cannot be empty")
    if entry.calories <= 0:
        raise ValueError("estimated_calories must be > 0")
    if entry.meal_type not in {None, *VALID_MEAL_TYPES}:
        raise ValueError("meal_type must be breakfast, lunch, dinner, or snack")
    for value in (entry.protein_g, entry.carbs_g, entry.fat_g):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("macros must be finite and nonnegative")


def nutrition_payload(entry: CalorieEntry) -> dict[str, Any]:
    """Map a meal to the verified Google Health NutritionLog wire shape.

    Unknown nutrients are omitted. ``serving`` is intentionally omitted: the
    Health API requires a verified ``foodMeasurementUnit`` when that object is
    present, and Blacki has no source for a truthful unit.
    """
    tz = get_app_timezone()
    logged = datetime.fromisoformat(entry.logged_at)
    if logged.tzinfo is None:
        logged = logged.replace(tzinfo=tz)
    else:
        logged = logged.astimezone(tz)
    meal_date = datetime.strptime(entry.logged_date, "%Y-%m-%d").date()
    start = (
        logged
        if logged.date() == meal_date
        else datetime.combine(meal_date, time(12), tzinfo=tz)
    )
    end = start + timedelta(seconds=1)
    start_offset = start.utcoffset()
    end_offset = end.utcoffset()
    if start_offset is None or end_offset is None:  # pragma: no cover
        raise ValueError("meal timezone offset is unavailable")
    nutrition: dict[str, Any] = {
        "interval": {
            "startTime": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "endTime": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "startUtcOffset": f"{int(start_offset.total_seconds())}s",
            "endUtcOffset": f"{int(end_offset.total_seconds())}s",
        },
        "foodDisplayName": entry.description,
        "energy": {"kcal": entry.calories},
    }
    if entry.meal_type:
        nutrition["mealType"] = entry.meal_type.upper()
    nutrients: list[dict[str, Any]] = []
    if entry.protein_g is not None:
        nutrients.append(
            {"nutrient": "PROTEIN", "quantity": {"grams": entry.protein_g}}
        )
    if nutrients:
        nutrition["nutrients"] = nutrients
    if entry.carbs_g is not None:
        nutrition["totalCarbohydrate"] = {"grams": entry.carbs_g}
    if entry.fat_g is not None:
        nutrition["totalFat"] = {"grams": entry.fat_g}
    return {"nutritionLog": nutrition}


class MealService:
    """Commit a calorie mutation and its export intent in one transaction."""

    def __init__(self, container: AppContainer) -> None:
        self.container = container
        self._nutrition: NutritionStorage | Any | None = None

    async def _get_nutrition_storage(self) -> NutritionStorage | Any:
        health = self.container.google_health_storage
        nutrition = getattr(health, "nutrition", None)
        if nutrition is None:
            nutrition = NutritionStorage(self.container.conn, self.container.lock)
            self._nutrition = nutrition
        await nutrition.initialize()
        return nutrition

    async def mutate(
        self,
        user_id: str,
        *,
        private: bool = False,
        entry: CalorieEntry | None = None,
        entry_id: int | None = None,
        updates: dict[str, Any] | None = None,
    ) -> tuple[int, str]:
        """Create, edit, or delete a meal and return its sync status.

        ``entry`` selects create, a non-empty ``updates`` mapping selects edit,
        and ``entry=None`` with ``entry_id`` selects delete. All local writes
        and export revisions happen under the same SQLite transaction.
        """
        if entry_id is None and entry is None:
            raise ValueError("A new meal is required")
        if entry_id is not None and entry is not None:
            raise ValueError("entry and entry_id cannot both select a new meal")

        container = self.container
        calorie_storage = container.calorie_storage
        health = container.google_health_storage
        nutrition = await self._get_nutrition_storage()
        await calorie_storage.initialize()
        await health.initialize()

        async with container.lock:
            await container.conn.execute("BEGIN")
            try:
                original: CalorieEntry | None = None
                if entry_id is not None:
                    async with container.conn.execute(
                        "SELECT * FROM calorie_logs WHERE id = ? AND user_id = ?",
                        (entry_id, user_id),
                    ) as cursor:
                        row = await cursor.fetchone()
                    if row is None:
                        raise ValueError("Meal not found or you do not have permission")
                    original = CalorieEntry.model_validate(dict(row))

                if entry is not None:
                    if entry.user_id != user_id:
                        raise ValueError("Meal owner does not match the current user")
                    validate_meal(entry)
                    created = True
                elif updates:
                    if original is None:  # pragma: no cover
                        raise RuntimeError("edit requires the original meal row")
                    entry = CalorieEntry.model_validate(
                        {**original.model_dump(), **updates}
                    )
                    validate_meal(entry)
                    created = False
                elif entry_id is not None:
                    created = False
                else:  # pragma: no cover
                    raise ValueError("A new meal is required")

                if entry_id is None:
                    if entry is None:  # pragma: no cover
                        raise RuntimeError("create requires a meal entry")
                    async with container.conn.execute(
                        """
                        INSERT INTO calorie_logs
                            (user_id, description, calories, protein_g, carbs_g,
                             fat_g, meal_type, logged_at, logged_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            entry.description,
                            entry.calories,
                            entry.protein_g,
                            entry.carbs_g,
                            entry.fat_g,
                            entry.meal_type,
                            entry.logged_at,
                            entry.logged_date,
                        ),
                    ) as cursor:
                        inserted_id = cursor.lastrowid
                    if inserted_id is None:  # pragma: no cover
                        raise RuntimeError("Meal insert did not return an identifier")
                    entry_id = int(inserted_id)
                elif entry is None:
                    await container.conn.execute(
                        "DELETE FROM calorie_logs WHERE id = ? AND user_id = ?",
                        (entry_id, user_id),
                    )
                else:
                    await container.conn.execute(
                        """
                        UPDATE calorie_logs
                        SET description = ?, calories = ?, protein_g = ?,
                            carbs_g = ?, fat_g = ?, meal_type = ?,
                            logged_at = ?, logged_date = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (
                            entry.description,
                            entry.calories,
                            entry.protein_g,
                            entry.carbs_g,
                            entry.fat_g,
                            entry.meal_type,
                            entry.logged_at,
                            entry.logged_date,
                            entry_id,
                            user_id,
                        ),
                    )

                sync_status = await self._enqueue_export(
                    nutrition=nutrition,
                    health=health,
                    user_id=user_id,
                    private=private,
                    entry_id=entry_id,
                    entry=entry,
                    original=original,
                    created=created,
                    updates=updates,
                )
                await container.conn.execute("COMMIT")
                if sync_status == "pending" and container.nutrition_export_worker:
                    container.nutrition_export_worker.wake()
                return entry_id, sync_status
            except BaseException:
                with contextlib.suppress(Exception):
                    await container.conn.execute("ROLLBACK")
                raise

    async def _enqueue_export(
        self,
        *,
        nutrition: NutritionStorage | Any,
        health: Any,
        user_id: str,
        private: bool,
        entry_id: int,
        entry: CalorieEntry | None,
        original: CalorieEntry | None,
        created: bool,
        updates: dict[str, Any] | None,
    ) -> str:
        canonical = health_user_id_for_telegram_user(user_id) if private else None
        connection = (
            await health.get_connection(canonical) if canonical is not None else None
        )
        previous = await nutrition.meal(entry_id)
        if previous is not None and str(previous["status"]) == "cancelled":
            return "not_enabled"

        eligible = _nutrition_authorized(connection)
        existing_account = (
            str(previous["health_user_id"])
            if previous is not None and previous.get("health_user_id")
            else None
        )
        connection_account = (
            connection.health_user_id if connection is not None else existing_account
        )
        account_matches = (
            existing_account is None
            or connection_account is None
            or existing_account == connection_account
        )

        # Only a newly created private meal with both nutrition scopes enrolls.
        # A connection added later must not backfill meals that predate consent.
        should_enqueue = (
            private
            and account_matches
            and ((created and eligible) or (not created and previous is not None))
        )
        if not should_enqueue:
            if created and connection is not None and not eligible:
                return "authorization_required"
            return "not_enabled"

        health_user_id = existing_account or (
            connection.health_user_id if connection is not None else ""
        )
        if not health_user_id:
            return "authorization_required"

        if entry is None:
            target = await self._latest_remote_resource(nutrition, entry_id)
            await nutrition.enqueue(
                meal_id=entry_id,
                owner_id=user_id,
                telegram_user_id=canonical or user_id,
                health_user_id=health_user_id,
                payload=None,
                operation="delete",
                target_resource_name=target,
            )
        else:
            payload = nutrition_payload(entry)
            if previous is not None and (
                updates is None or "logged_date" not in updates
            ):
                prior_payload = await nutrition.latest_payload(entry_id)
                prior_interval = _interval_from_payload(prior_payload)
                if prior_interval is not None:
                    payload["nutritionLog"]["interval"] = prior_interval

            # Google Health's anonymous data points cannot be edited. Delete a
            # previously reconciled/in-flight point before creating its revision.
            if (
                previous is not None
                and str(previous.get("desired_operation")) == "upsert"
            ):
                target = await self._latest_remote_resource(nutrition, entry_id)
                if target is not None:
                    await nutrition.enqueue(
                        meal_id=entry_id,
                        owner_id=user_id,
                        telegram_user_id=canonical or user_id,
                        health_user_id=health_user_id,
                        payload=None,
                        operation="delete",
                        target_resource_name=target,
                    )
            await nutrition.enqueue(
                meal_id=entry_id,
                owner_id=user_id,
                telegram_user_id=canonical or user_id,
                health_user_id=health_user_id,
                payload=payload,
                operation="upsert",
            )

        if eligible:
            return "pending"
        await container_execute(
            self.container.conn,
            "UPDATE nutrition_exports SET status = ? WHERE meal_id = ?",
            ("authorization_required", entry_id),
        )
        return "authorization_required"

    async def _latest_remote_resource(
        self, nutrition: NutritionStorage | Any, meal_id: int
    ) -> str | None:
        revisions = await nutrition.revisions(meal_id)
        for revision in reversed(revisions):
            if revision.get("operation", "upsert") != "upsert":
                continue
            state = str(revision.get("state", "queued"))
            if state in {"synced", "in_flight", "uncertain"}:
                return str(revision["resource_name"])
        return None


async def container_execute(conn: Any, query: str, params: tuple[Any, ...]) -> None:
    """Execute a mutation using the caller's already-held transaction."""
    await conn.execute(query, params)


def _nutrition_authorized(connection: Any | None) -> bool:
    return bool(
        connection is not None
        and connection.status == "connected"
        and set(GOOGLE_HEALTH_NUTRITION_SCOPES) <= set(connection.scopes)
    )


def _interval_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    nutrition = payload.get("nutritionLog")
    if not isinstance(nutrition, dict):
        return None
    interval = nutrition.get("interval")
    return dict(interval) if isinstance(interval, dict) else None


def get_meal_service() -> MealService:
    """Return a meal service bound to the process container."""
    return MealService(get_container())
