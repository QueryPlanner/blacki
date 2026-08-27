"""One-time historical Google Health nutrition export coordination."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from blacki.calories.service import nutrition_payload
from blacki.calories.storage import SqliteCalorieStorage

from .config import (
    GOOGLE_HEALTH_NUTRITION_SCOPES,
    health_user_id_for_telegram_user,
    telegram_chat_id_for_health_user,
)
from .nutrition_storage import (
    BACKFILL_BATCH_SIZE,
    BACKFILL_LEASE_SECONDS,
    BACKFILL_VERSION,
)
from .storage import HealthConnection, SqliteGoogleHealthStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NutritionBackfillResult:
    """Safe local outcome for one backfill coordination attempt."""

    status: str
    telegram_user_id: str
    health_user_id: str | None = None
    queued_count: int = 0
    skipped_count: int = 0


class NutritionBackfillCoordinator:
    """Queue historical meals without making provider calls."""

    def __init__(
        self,
        health_storage: SqliteGoogleHealthStorage,
        calorie_storage: SqliteCalorieStorage,
        *,
        wake: Callable[[], None] | None = None,
    ) -> None:
        self.health_storage = health_storage
        self.calorie_storage = calorie_storage
        self.wake = wake

    async def run_all_eligible(self) -> list[NutritionBackfillResult]:
        """Sweep connected accounts that granted both nutrition scopes."""
        results: list[NutritionBackfillResult] = []
        for connection in await self.health_storage.list_active_connections():
            if not _nutrition_authorized(connection):
                continue
            try:
                results.append(await self.run_user(connection.telegram_user_id))
            except Exception:
                logger.exception(
                    "Google Health nutrition backfill failed for one account"
                )
                results.append(
                    NutritionBackfillResult(
                        status="failed",
                        telegram_user_id=connection.telegram_user_id,
                        health_user_id=connection.health_user_id,
                    )
                )
        return results

    async def run_user(self, telegram_user_id: str) -> NutritionBackfillResult:
        """Claim and advance one private user's account-bound backfill."""
        health_user_id = health_user_id_for_telegram_user(telegram_user_id)
        if health_user_id is None or not _is_private_health_user_id(health_user_id):
            return NutritionBackfillResult(
                status="skipped", telegram_user_id=telegram_user_id
            )

        connection = await self.health_storage.get_connection(health_user_id)
        if connection is None:
            return NutritionBackfillResult(
                status="not_connected", telegram_user_id=health_user_id
            )
        if not _nutrition_authorized(connection):
            return NutritionBackfillResult(
                status="not_eligible",
                telegram_user_id=health_user_id,
                health_user_id=connection.health_user_id,
            )

        nutrition = self.health_storage.nutrition
        high_water = await self.calorie_storage.health_backfill_high_water(
            health_user_id
        )
        claimed = await nutrition.claim_backfill(
            health_user_id,
            connection.health_user_id,
            high_water,
            version=BACKFILL_VERSION,
            lease_seconds=BACKFILL_LEASE_SECONDS,
        )
        if claimed is None:
            existing = await nutrition.get_backfill(
                health_user_id, connection.health_user_id, BACKFILL_VERSION
            )
            return NutritionBackfillResult(
                status=str(existing["status"]) if existing is not None else "running",
                telegram_user_id=health_user_id,
                health_user_id=connection.health_user_id,
                queued_count=(
                    int(existing["queued_count"]) if existing is not None else 0
                ),
                skipped_count=(
                    int(existing["skipped_count"]) if existing is not None else 0
                ),
            )

        queued_total = 0
        skipped_total = 0
        cursor = int(claimed["cursor_meal_id"])
        high_water = int(claimed["high_water_meal_id"])
        try:
            if high_water == 0:
                await nutrition.advance_backfill(
                    health_user_id,
                    connection.health_user_id,
                    0,
                    queued_count=0,
                    skipped_count=0,
                    version=BACKFILL_VERSION,
                    lease_seconds=BACKFILL_LEASE_SECONDS,
                )
                return NutritionBackfillResult(
                    status="completed",
                    telegram_user_id=health_user_id,
                    health_user_id=connection.health_user_id,
                )

            while cursor < high_water:
                (
                    entries,
                    batch_cursor,
                ) = await self.calorie_storage.health_backfill_batch(
                    health_user_id,
                    after_id=cursor,
                    through_id=high_water,
                    limit=BACKFILL_BATCH_SIZE,
                )
                next_cursor = batch_cursor if batch_cursor is not None else high_water
                batch_queued = 0
                batch_skipped = 0

                async with nutrition._lock:
                    await nutrition.conn.execute("BEGIN")
                    try:
                        for entry in entries:
                            result = await nutrition.ensure_backfill_export(
                                meal_id=int(entry.id or 0),
                                owner_id=entry.user_id,
                                telegram_user_id=health_user_id,
                                health_user_id=connection.health_user_id,
                                payload=nutrition_payload(entry),
                            )
                            if result == "queued":
                                batch_queued += 1
                            else:
                                batch_skipped += 1
                        updated = await nutrition._advance_backfill_unlocked(
                            health_user_id,
                            connection.health_user_id,
                            next_cursor,
                            queued_count=batch_queued,
                            skipped_count=batch_skipped,
                            version=BACKFILL_VERSION,
                            now=time.time(),
                            lease_seconds=BACKFILL_LEASE_SECONDS,
                        )
                        if not updated:
                            raise RuntimeError("nutrition backfill lease was lost")
                        await nutrition.conn.execute("COMMIT")
                    except BaseException:
                        await nutrition._rollback()
                        raise

                cursor = next_cursor
                queued_total += batch_queued
                skipped_total += batch_skipped
                if batch_queued and self.wake is not None:
                    self.wake()
                if batch_cursor is None or cursor >= high_water:
                    break

            return NutritionBackfillResult(
                status="completed",
                telegram_user_id=health_user_id,
                health_user_id=connection.health_user_id,
                queued_count=queued_total,
                skipped_count=skipped_total,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            await nutrition.fail_backfill(
                health_user_id,
                connection.health_user_id,
                "backfill_queue_error",
                version=BACKFILL_VERSION,
            )
            logger.exception(
                "Google Health nutrition backfill could not queue local meals"
            )
            return NutritionBackfillResult(
                status="failed",
                telegram_user_id=health_user_id,
                health_user_id=connection.health_user_id,
                queued_count=queued_total,
                skipped_count=skipped_total,
            )


def _nutrition_authorized(connection: HealthConnection) -> bool:
    """Return whether the connection can export nutrition data."""
    return bool(
        connection.status == "connected"
        and connection.encrypted_refresh_token is not None
        and set(GOOGLE_HEALTH_NUTRITION_SCOPES) <= set(connection.scopes)
    )


def _is_private_health_user_id(health_user_id: str) -> bool:
    """Reject negative Telegram group IDs from historical export."""
    chat_id = telegram_chat_id_for_health_user(health_user_id)
    return chat_id is not None and chat_id > 0
