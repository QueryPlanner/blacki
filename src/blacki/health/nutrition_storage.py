"""Durable, account-bound meal export state.

Calorie rows remain the local source of truth.  This module stores the latest
desired revision plus immutable create revisions for Google Health.  A delete
sets ``desired_revision`` to NULL while retaining the revisions the worker must
remove remotely, so local deletion never loses reconciliation intent.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from blacki.storage.base import SqlStorage

_MISSING = object()


class NutritionStorage(SqlStorage):
    """Store desired meals and immutable remote create revisions."""

    async def _create_tables(self) -> None:
        await self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nutrition_exports (
                meal_id INTEGER PRIMARY KEY,
                owner_id TEXT NOT NULL,
                telegram_user_id TEXT NOT NULL,
                health_user_id TEXT NOT NULL,
                desired_revision TEXT,
                desired_operation TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0,
                error_code TEXT
            );
            CREATE TABLE IF NOT EXISTS nutrition_revisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                meal_id INTEGER NOT NULL REFERENCES nutrition_exports(meal_id),
                resource_name TEXT,
                operation TEXT NOT NULL,
                payload_json TEXT,
                state TEXT NOT NULL DEFAULT 'queued'
            );
            CREATE INDEX IF NOT EXISTS nutrition_due
                ON nutrition_exports(status, next_attempt);
            CREATE INDEX IF NOT EXISTS nutrition_meal_revisions
                ON nutrition_revisions(meal_id, sequence);
            """
        )

    async def enqueue(
        self,
        *,
        meal_id: int,
        owner_id: str,
        telegram_user_id: str,
        health_user_id: str,
        payload: dict[str, Any] | None,
        operation: str,
        target_resource_name: str | None = None,
    ) -> None:
        """Persist the newest desired create or deletion in the caller tx.

        ``operation`` is ``"upsert"`` (mints a fresh opaque Google data-point
        resource name and requires ``payload``) or ``"delete"`` (removes
        ``target_resource_name``, which may be ``None`` when nothing was ever
        dispatched remotely). The worker resolves prior in-flight/uncertain
        revisions before allowing a replacement create.
        """
        if operation not in {"upsert", "delete"}:
            raise ValueError(f"unsupported nutrition export operation: {operation}")

        existing = await self._fetch_one(
            "SELECT owner_id, telegram_user_id, health_user_id "
            "FROM nutrition_exports WHERE meal_id = ?",
            (meal_id,),
        )
        if existing is not None:
            if str(existing["owner_id"]) != owner_id:
                raise ValueError("nutrition export owner cannot change")
            if str(existing["telegram_user_id"]) != telegram_user_id:
                raise ValueError("nutrition export identity cannot change")
            old_health_user_id = str(existing["health_user_id"])
            if old_health_user_id and old_health_user_id != health_user_id:
                raise ValueError("nutrition export account cannot change")

        if operation == "upsert":
            if payload is None:
                raise ValueError("upsert requires a payload")
            resource_name: str | None = (
                f"users/{health_user_id}/dataTypes/nutrition-log/dataPoints/"
                f"blacki-{uuid4()}"
            )
        else:
            resource_name = target_resource_name

        await self._conn.execute(
            """
            INSERT INTO nutrition_exports
                (meal_id, owner_id, telegram_user_id, health_user_id,
                 desired_revision, desired_operation, status, attempts,
                 next_attempt, error_code)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, 0, NULL)
            ON CONFLICT(meal_id) DO UPDATE SET
                desired_revision = excluded.desired_revision,
                desired_operation = excluded.desired_operation,
                status = 'pending', attempts = 0, next_attempt = 0,
                error_code = NULL
            """,
            (
                meal_id,
                owner_id,
                telegram_user_id,
                health_user_id,
                resource_name,
                operation,
            ),
        )
        if resource_name is not None:
            await self._conn.execute(
                """
                INSERT INTO nutrition_revisions
                    (meal_id, resource_name, operation, payload_json, state)
                VALUES (?, ?, ?, ?, 'queued')
                """,
                (
                    meal_id,
                    resource_name,
                    operation,
                    json.dumps(payload, allow_nan=False)
                    if payload is not None
                    else None,
                ),
            )

    async def meal(self, meal_id: int) -> dict[str, Any] | None:
        """Return the desired export row for one meal."""
        return await self._fetch_one(
            "SELECT * FROM nutrition_exports WHERE meal_id = ?", (meal_id,)
        )

    async def due(self, now: float) -> list[dict[str, Any]]:
        """Return a bounded batch of pending desired exports."""
        return await self._fetch_all(
            """
            SELECT * FROM nutrition_exports
            WHERE status = 'pending' AND next_attempt <= ?
            ORDER BY next_attempt, meal_id
            LIMIT 10
            """,
            (now,),
        )

    async def revisions(self, meal_id: int) -> list[dict[str, Any]]:
        """Return all immutable revisions, including terminal states."""
        return await self._fetch_all(
            """
            SELECT * FROM nutrition_revisions
            WHERE meal_id = ?
            ORDER BY sequence
            """,
            (meal_id,),
        )

    async def latest_payload(self, meal_id: int) -> dict[str, Any] | None:
        """Return the newest decodable payload for preserving its interval."""
        rows = await self._fetch_all(
            """
            SELECT payload_json FROM nutrition_revisions
            WHERE meal_id = ?
            ORDER BY sequence DESC
            """,
            (meal_id,),
        )
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                return payload
        return None

    async def revision_state(self, sequence: int, state: str) -> None:
        """Record provider progress for one immutable revision.

        Keyed by ``sequence`` rather than ``resource_name``: a delete revision
        deliberately reuses the resource name of the create it targets, so
        the name alone cannot identify a single row.
        """
        await self._conn.execute(
            "UPDATE nutrition_revisions SET state = ? WHERE sequence = ?",
            (state, sequence),
        )

    async def result(
        self,
        meal_id: int,
        status: str,
        *,
        error: str | None = None,
        next_attempt: float = 0,
        expected_revision: str | None | object = _MISSING,
    ) -> bool:
        """Record a result without clobbering a newer desired edit.

        ``IS`` is used for the guard so deletion rows with a NULL desired
        revision can be guarded as well.  Omitting the guard preserves the
        pre-feature calling convention.
        """
        params: list[Any] = [status, error, next_attempt, meal_id]
        where = "meal_id = ? AND status != 'cancelled'"
        if expected_revision is not _MISSING:
            if expected_revision is None:
                where += " AND desired_revision IS NULL"
            else:
                where += " AND desired_revision IS ?"
                params.append(expected_revision)
        cursor = await self._conn.execute(
            f"""
            UPDATE nutrition_exports
            SET status = ?, error_code = ?, attempts = attempts + 1,
                next_attempt = ?
            WHERE {where}
            """,  # noqa: S608
            tuple(params),
        )
        return cursor.rowcount > 0

    async def counts(self, user_id: str) -> dict[str, int]:
        """Return durable non-cancelled export counts for one identity."""
        rows = await self._fetch_all(
            """
            SELECT status, COUNT(*) AS count
            FROM nutrition_exports
            WHERE telegram_user_id = ? AND status != 'cancelled'
            GROUP BY status
            """,
            (user_id,),
        )
        return {str(row["status"]): int(row["count"]) for row in rows}

    async def cancel(self, user_id: str) -> None:
        """Cancel dispatch and purge locally stored provider payloads/IDs."""
        await self._conn.execute(
            """
            UPDATE nutrition_exports
            SET status = 'cancelled', health_user_id = '',
                desired_revision = NULL, error_code = NULL
            WHERE telegram_user_id = ?
            """,
            (user_id,),
        )
        await self._conn.execute(
            """
            DELETE FROM nutrition_revisions
            WHERE meal_id IN (
                SELECT meal_id FROM nutrition_exports WHERE telegram_user_id = ?
            )
            """,
            (user_id,),
        )

    async def resume(self, user_id: str, health_user_id: str) -> None:
        """Resume only authorization-paused work for the same account."""
        await self._conn.execute(
            """
            UPDATE nutrition_exports
            SET status = 'pending', next_attempt = 0, error_code = NULL
            WHERE telegram_user_id = ? AND health_user_id = ?
              AND status = 'authorization_required'
            """,
            (user_id, health_user_id),
        )
