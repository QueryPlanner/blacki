"""Durable, account-bound meal export state.

Calorie rows remain the local source of truth. This module stores the latest
desired revision plus immutable revisions for Google Health. A separate
history table records terminal provider results so disconnecting and
reconnecting the same Google account cannot create duplicate data points.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any
from uuid import uuid4

from blacki.storage.base import SqlStorage

BACKFILL_VERSION = 1
BACKFILL_BATCH_SIZE = 50
BACKFILL_LEASE_SECONDS = 300

_MISSING = object()


class NutritionStorage(SqlStorage):
    """Store desired meals, immutable revisions, and export history."""

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
            CREATE TABLE IF NOT EXISTS nutrition_export_history (
                meal_id INTEGER NOT NULL,
                telegram_user_id TEXT NOT NULL,
                health_user_id TEXT NOT NULL,
                resource_name TEXT NOT NULL,
                operation TEXT NOT NULL,
                payload_json TEXT,
                state TEXT NOT NULL,
                backfill_version INTEGER NOT NULL DEFAULT 1,
                updated_at REAL NOT NULL,
                PRIMARY KEY (meal_id, health_user_id, resource_name, operation)
            );
            CREATE TABLE IF NOT EXISTS google_health_nutrition_backfills (
                telegram_user_id TEXT NOT NULL,
                health_user_id TEXT NOT NULL,
                backfill_version INTEGER NOT NULL,
                high_water_meal_id INTEGER NOT NULL DEFAULT 0,
                cursor_meal_id INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                lease_expires_at REAL,
                queued_count INTEGER NOT NULL DEFAULT 0,
                skipped_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                started_at REAL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                PRIMARY KEY (telegram_user_id, health_user_id, backfill_version)
            );
            CREATE INDEX IF NOT EXISTS nutrition_due
                ON nutrition_exports(status, next_attempt);
            CREATE INDEX IF NOT EXISTS nutrition_meal_revisions
                ON nutrition_revisions(meal_id, sequence);
            CREATE INDEX IF NOT EXISTS nutrition_history_account
                ON nutrition_export_history(telegram_user_id, health_user_id);
            CREATE INDEX IF NOT EXISTS nutrition_backfills_status
                ON google_health_nutrition_backfills(status, lease_expires_at);
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
        await self._enqueue_unlocked(
            meal_id=meal_id,
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            health_user_id=health_user_id,
            payload=payload,
            operation=operation,
            target_resource_name=target_resource_name,
        )

    async def _enqueue_unlocked(
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
        """Persist an export while the caller owns the transaction or lock."""
        if operation not in {"upsert", "delete"}:
            raise ValueError(f"unsupported nutrition export operation: {operation}")

        existing = await self._fetch_one(
            "SELECT owner_id, telegram_user_id, health_user_id, status "
            "FROM nutrition_exports WHERE meal_id = ?",
            (meal_id,),
        )
        if existing is not None:
            if str(existing["owner_id"]) != owner_id:
                raise ValueError("nutrition export owner cannot change")
            if str(existing["telegram_user_id"]) != telegram_user_id:
                raise ValueError("nutrition export identity cannot change")
            old_health_user_id = str(existing["health_user_id"])
            if (
                old_health_user_id
                and old_health_user_id != health_user_id
                and str(existing["status"]) != "cancelled"
            ):
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
                health_user_id = excluded.health_user_id,
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
        payload = _first_dict_payload(rows)
        if payload is not None:
            return payload

        history = await self._fetch_all(
            """
            SELECT payload_json FROM nutrition_export_history
            WHERE meal_id = ? AND operation = 'upsert' AND state = 'synced'
            ORDER BY updated_at DESC
            """,
            (meal_id,),
        )
        return _first_dict_payload(history)

    async def latest_remote_resource(
        self, meal_id: int, health_user_id: str | None = None
    ) -> str | None:
        """Return the latest known remote create resource for one meal."""
        revisions = await self.revisions(meal_id)
        for revision in reversed(revisions):
            if revision.get("operation", "upsert") != "upsert":
                continue
            resource_name = revision.get("resource_name")
            if resource_name is None:
                continue
            if health_user_id is not None and not _resource_belongs_to_health_user(
                str(resource_name), health_user_id
            ):
                continue
            if str(revision.get("state", "queued")) in {
                "synced",
                "in_flight",
                "uncertain",
            }:
                return str(resource_name)

        if health_user_id is not None:
            row = await self._fetch_one(
                """
                SELECT resource_name FROM nutrition_export_history
                WHERE meal_id = ? AND health_user_id = ?
                  AND operation = 'upsert' AND state = 'synced'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (meal_id, health_user_id),
            )
        else:
            row = await self._fetch_one(
                """
                SELECT resource_name FROM nutrition_export_history
                WHERE meal_id = ? AND operation = 'upsert' AND state = 'synced'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (meal_id,),
            )
        return str(row["resource_name"]) if row is not None else None

    async def revision_state(self, sequence: int, state: str) -> None:
        """Record provider progress for one immutable revision."""
        async with self._write_transaction():
            await self._revision_state_unlocked(sequence, state)

    async def _revision_state_unlocked(self, sequence: int, state: str) -> None:
        """Update revision state while the caller owns the lock or tx."""
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
        """Record a result without clobbering a newer desired edit."""
        async with self._write_transaction():
            return await self._result_unlocked(
                meal_id,
                status,
                error=error,
                next_attempt=next_attempt,
                expected_revision=expected_revision,
            )

    async def _result_unlocked(
        self,
        meal_id: int,
        status: str,
        *,
        error: str | None = None,
        next_attempt: float = 0,
        expected_revision: str | None | object = _MISSING,
    ) -> bool:
        """Update export status while the caller owns the lock or tx."""
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

    async def record_remote_result(self, sequence: int) -> None:
        """Persist a terminal provider result for reconnect idempotency."""
        async with self._write_transaction():
            row = await self._fetch_one(
                """
                SELECT r.meal_id, r.resource_name, r.operation, r.payload_json,
                       r.state, e.telegram_user_id
                FROM nutrition_revisions AS r
                JOIN nutrition_exports AS e ON e.meal_id = r.meal_id
                WHERE r.sequence = ?
                """,
                (sequence,),
            )
            if row is None or str(row["state"]) not in {"synced", "deleted"}:
                return
            resource_name = row["resource_name"]
            if resource_name is None:
                return
            health_user_id = _health_user_id_from_resource(str(resource_name))
            if health_user_id is None:
                return
            row["health_user_id"] = health_user_id
            await self._record_remote_result_unlocked(row)

    async def _record_remote_result_unlocked(self, row: dict[str, Any]) -> None:
        """Record a terminal revision while the caller owns the lock or tx."""
        await self._conn.execute(
            """
            INSERT INTO nutrition_export_history
                (meal_id, telegram_user_id, health_user_id, resource_name,
                 operation, payload_json, state, backfill_version, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (meal_id, health_user_id, resource_name, operation)
            DO UPDATE SET
                telegram_user_id = excluded.telegram_user_id,
                operation = excluded.operation,
                payload_json = excluded.payload_json,
                state = excluded.state,
                backfill_version = excluded.backfill_version,
                updated_at = excluded.updated_at
            """,
            (
                int(row["meal_id"]),
                str(row["telegram_user_id"]),
                str(row["health_user_id"]),
                str(row["resource_name"]),
                str(row["operation"]),
                row["payload_json"],
                str(row["state"]),
                BACKFILL_VERSION,
                time.time(),
            ),
        )

    async def ensure_backfill_export(
        self,
        *,
        meal_id: int,
        owner_id: str,
        telegram_user_id: str,
        health_user_id: str,
        payload: dict[str, Any],
    ) -> str:
        """Ensure one historical meal has an idempotent export intent.

        The caller must hold the shared lock and an open transaction. Existing
        pending, failed, or terminal current-account rows are left untouched.
        A previously synced resource with an unchanged payload is also left
        untouched. A changed payload gets a delete for the old resource and a
        new upsert in order.
        """
        existing = await self._fetch_one(
            "SELECT * FROM nutrition_exports WHERE meal_id = ?", (meal_id,)
        )
        history: dict[str, Any] | None = None
        if existing is not None:
            if str(existing["owner_id"]) != owner_id:
                raise ValueError("nutrition export owner cannot change")
            if str(existing["telegram_user_id"]) != telegram_user_id:
                raise ValueError("nutrition export identity cannot change")
            old_health_user_id = str(existing["health_user_id"])
            if (
                old_health_user_id
                and old_health_user_id != health_user_id
                and str(existing["status"]) != "cancelled"
            ):
                raise ValueError("nutrition export account cannot change")
            if str(existing["status"]) != "cancelled":
                return "existing"

            if not old_health_user_id or old_health_user_id == health_user_id:
                history = await self._latest_synced_history_unlocked(
                    meal_id, health_user_id
                )
                if history is not None and _payload_json_matches(
                    history["payload_json"], payload
                ):
                    await self._restore_synced_history_unlocked(
                        existing,
                        history,
                        owner_id=owner_id,
                        telegram_user_id=telegram_user_id,
                        health_user_id=health_user_id,
                    )
                    return "history"
                restored = await self._restore_cancelled_revision_unlocked(
                    existing, health_user_id=health_user_id, payload=payload
                )
                if restored is not None:
                    return restored

            if old_health_user_id and old_health_user_id != health_user_id:
                await self._cancel_unresolved_revisions_unlocked(
                    meal_id, old_health_user_id
                )

        if history is None:
            history = await self._latest_synced_history_unlocked(
                meal_id, health_user_id
            )
        if history is not None and _payload_json_matches(
            history["payload_json"], payload
        ):
            await self._restore_synced_history_unlocked(
                existing,
                history,
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                health_user_id=health_user_id,
            )
            return "history"

        if history is not None:
            await self._enqueue_unlocked(
                meal_id=meal_id,
                owner_id=owner_id,
                telegram_user_id=telegram_user_id,
                health_user_id=health_user_id,
                payload=None,
                operation="delete",
                target_resource_name=str(history["resource_name"]),
            )
        await self._enqueue_unlocked(
            meal_id=meal_id,
            owner_id=owner_id,
            telegram_user_id=telegram_user_id,
            health_user_id=health_user_id,
            payload=payload,
            operation="upsert",
        )
        return "queued"

    async def _restore_synced_history_unlocked(
        self,
        existing: dict[str, Any] | None,
        history: dict[str, Any],
        *,
        owner_id: str,
        telegram_user_id: str,
        health_user_id: str,
    ) -> None:
        """Restore a cancelled meal from a known synced remote point."""
        resource_name = str(history["resource_name"])
        if existing is None:
            await self._conn.execute(
                """
                INSERT INTO nutrition_exports
                    (meal_id, owner_id, telegram_user_id, health_user_id,
                     desired_revision, desired_operation, status, attempts,
                     next_attempt, error_code)
                VALUES (?, ?, ?, ?, ?, 'upsert', 'synced', 0, 0, NULL)
                """,
                (
                    int(history["meal_id"]),
                    owner_id,
                    telegram_user_id,
                    health_user_id,
                    resource_name,
                ),
            )
            return
        await self._conn.execute(
            """
            UPDATE nutrition_exports
            SET health_user_id = ?, desired_revision = ?,
                desired_operation = 'upsert', status = 'synced', attempts = 0,
                next_attempt = 0, error_code = NULL
            WHERE meal_id = ? AND status = 'cancelled'
            """,
            (health_user_id, resource_name, int(existing["meal_id"])),
        )

    async def _restore_cancelled_revision_unlocked(
        self,
        existing: dict[str, Any],
        *,
        health_user_id: str,
        payload: dict[str, Any],
    ) -> str | None:
        """Restore one unresolved same-account upsert after disconnect."""
        revisions = await self._fetch_all(
            """
            SELECT * FROM nutrition_revisions
            WHERE meal_id = ?
            ORDER BY sequence DESC
            """,
            (int(existing["meal_id"]),),
        )
        for revision in revisions:
            resource_name = revision["resource_name"]
            state = str(revision["state"])
            if (
                str(revision["operation"]) != "upsert"
                or resource_name is None
                or not _resource_belongs_to_health_user(
                    str(resource_name), health_user_id
                )
                or state == "cancelled"
                or not _payload_json_matches(revision["payload_json"], payload)
            ):
                continue
            if state == "synced":
                row = dict(existing)
                row.update(
                    {
                        "resource_name": resource_name,
                        "operation": "upsert",
                        "payload_json": revision["payload_json"],
                        "state": "synced",
                        "health_user_id": health_user_id,
                    }
                )
                await self._record_remote_result_unlocked(row)
                await self._restore_synced_history_unlocked(
                    existing,
                    {
                        "meal_id": int(existing["meal_id"]),
                        "resource_name": resource_name,
                    },
                    owner_id=str(existing["owner_id"]),
                    telegram_user_id=str(existing["telegram_user_id"]),
                    health_user_id=health_user_id,
                )
                return "history"
            status = "failed" if state == "failed" else "pending"
            await self._conn.execute(
                """
                UPDATE nutrition_exports
                SET health_user_id = ?, desired_revision = ?,
                    desired_operation = 'upsert', status = ?, attempts = 0,
                    next_attempt = 0, error_code = CASE
                        WHEN ? = 'failed' THEN error_code ELSE NULL END
                WHERE meal_id = ? AND status = 'cancelled'
                """,
                (
                    health_user_id,
                    str(resource_name),
                    status,
                    status,
                    int(existing["meal_id"]),
                ),
            )
            return "existing"
        return None

    async def _cancel_unresolved_revisions_unlocked(
        self, meal_id: int, health_user_id: str
    ) -> None:
        """Prevent old-account work from running after an account switch."""
        await self._conn.execute(
            """
            UPDATE nutrition_revisions
            SET state = 'cancelled'
            WHERE meal_id = ? AND state IN ('queued', 'in_flight', 'uncertain')
              AND resource_name LIKE ?
            """,
            (meal_id, f"users/{health_user_id}/dataTypes/nutrition-log/dataPoints/%"),
        )

    async def _latest_synced_history_unlocked(
        self, meal_id: int, health_user_id: str
    ) -> dict[str, Any] | None:
        """Return the newest known synced resource for one account."""
        return await self._fetch_one(
            """
            SELECT * FROM nutrition_export_history
            WHERE meal_id = ? AND health_user_id = ?
              AND operation = 'upsert' AND state = 'synced'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (meal_id, health_user_id),
        )

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

    async def has_other_account(self, user_id: str, health_user_id: str) -> bool:
        """Detect retained meal state bound to a different Health account."""
        row = await self._fetch_one(
            """
            SELECT 1 FROM nutrition_exports
            WHERE telegram_user_id = ? AND health_user_id != ?
            LIMIT 1
            """,
            (user_id, health_user_id),
        )
        return row is not None

    async def cancel(self, user_id: str, *, cancel_revisions: bool = False) -> None:
        """Cancel dispatch while retaining revisions for safe reconnects."""
        await self._conn.execute(
            """
            UPDATE nutrition_exports
            SET status = 'cancelled', desired_revision = NULL
            WHERE telegram_user_id = ?
            """,
            (user_id,),
        )
        await self._conn.execute(
            """
            UPDATE google_health_nutrition_backfills
            SET status = 'pending', high_water_meal_id = 0, cursor_meal_id = 0,
                lease_expires_at = NULL,
                queued_count = 0, skipped_count = 0, last_error = NULL,
                started_at = NULL, updated_at = ?, completed_at = NULL
            WHERE telegram_user_id = ?
            """,
            (time.time(), user_id),
        )
        if cancel_revisions:
            await self._conn.execute(
                """
                UPDATE nutrition_revisions
                SET state = 'cancelled'
                WHERE meal_id IN (
                    SELECT meal_id FROM nutrition_exports WHERE telegram_user_id = ?
                ) AND state IN ('queued', 'in_flight', 'uncertain')
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

    async def retry_failed(self, user_id: str, health_user_id: str) -> int:
        """Requeue only current-account failures with a matching failed revision."""
        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                rows = await self._fetch_all(
                    """
                    SELECT meal_id, desired_revision
                    FROM nutrition_exports
                    WHERE telegram_user_id = ? AND health_user_id = ?
                      AND status = 'failed' AND desired_revision IS NOT NULL
                    """,
                    (user_id, health_user_id),
                )
                requeued = 0
                for row in rows:
                    revision = await self._fetch_one(
                        """
                        SELECT sequence
                        FROM nutrition_revisions
                        WHERE meal_id = ? AND resource_name = ? AND state = 'failed'
                        ORDER BY sequence DESC
                        LIMIT 1
                        """,
                        (int(row["meal_id"]), row["desired_revision"]),
                    )
                    if revision is None:
                        continue
                    await self._conn.execute(
                        "UPDATE nutrition_revisions SET state = 'queued' "
                        "WHERE sequence = ?",
                        (int(revision["sequence"]),),
                    )
                    cursor = await self._conn.execute(
                        """
                        UPDATE nutrition_exports
                        SET status = 'pending', attempts = 0, next_attempt = 0,
                            error_code = NULL
                        WHERE meal_id = ? AND status = 'failed'
                          AND desired_revision = ?
                        """,
                        (int(row["meal_id"]), row["desired_revision"]),
                    )
                    requeued += cursor.rowcount
                await self._conn.execute("COMMIT")
                return requeued
            except BaseException:
                await self._rollback()
                raise

    async def get_backfill(
        self, user_id: str, health_user_id: str, version: int = BACKFILL_VERSION
    ) -> dict[str, Any] | None:
        """Return one durable backfill ledger row."""
        return await self._fetch_one(
            """
            SELECT * FROM google_health_nutrition_backfills
            WHERE telegram_user_id = ? AND health_user_id = ?
              AND backfill_version = ?
            """,
            (user_id, health_user_id, version),
        )

    async def claim_backfill(
        self,
        user_id: str,
        health_user_id: str,
        high_water_meal_id: int,
        *,
        version: int = BACKFILL_VERSION,
        now: float | None = None,
        lease_seconds: int = BACKFILL_LEASE_SECONDS,
    ) -> dict[str, Any] | None:
        """Claim a pending or expired backfill lease under the write lock."""
        reference_time = time.time() if now is None else now
        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                row = await self._fetch_one(
                    """
                    SELECT * FROM google_health_nutrition_backfills
                    WHERE telegram_user_id = ? AND health_user_id = ?
                      AND backfill_version = ?
                    """,
                    (user_id, health_user_id, version),
                )
                if row is not None:
                    lease_expires_at = row["lease_expires_at"]
                    if str(row["status"]) == "completed" or (
                        str(row["status"]) == "running"
                        and lease_expires_at is not None
                        and float(lease_expires_at) > reference_time
                    ):
                        await self._conn.execute("COMMIT")
                        return None
                    stored_high_water = int(row["high_water_meal_id"])
                    if stored_high_water > 0:
                        high_water_meal_id = stored_high_water
                    else:
                        high_water_meal_id = max(0, high_water_meal_id)
                    await self._conn.execute(
                        """
                        UPDATE google_health_nutrition_backfills
                        SET high_water_meal_id = ?, status = 'running',
                            lease_expires_at = ?,
                            last_error = NULL, updated_at = ?,
                            started_at = COALESCE(started_at, ?)
                        WHERE telegram_user_id = ? AND health_user_id = ?
                          AND backfill_version = ?
                        """,
                        (
                            high_water_meal_id,
                            reference_time + lease_seconds,
                            reference_time,
                            reference_time,
                            user_id,
                            health_user_id,
                            version,
                        ),
                    )
                else:
                    await self._conn.execute(
                        """
                        INSERT INTO google_health_nutrition_backfills
                            (telegram_user_id, health_user_id, backfill_version,
                             high_water_meal_id, status, lease_expires_at,
                             updated_at, started_at)
                        VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                        """,
                        (
                            user_id,
                            health_user_id,
                            version,
                            max(0, high_water_meal_id),
                            reference_time + lease_seconds,
                            reference_time,
                            reference_time,
                        ),
                    )
                await self._conn.execute("COMMIT")
                return await self.get_backfill(user_id, health_user_id, version)
            except BaseException:
                await self._rollback()
                raise

    async def advance_backfill(
        self,
        user_id: str,
        health_user_id: str,
        cursor_meal_id: int,
        *,
        queued_count: int,
        skipped_count: int,
        version: int = BACKFILL_VERSION,
        now: float | None = None,
        lease_seconds: int = BACKFILL_LEASE_SECONDS,
    ) -> bool:
        """Advance a claimed backfill in a short transaction."""
        reference_time = time.time() if now is None else now
        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                updated = await self._advance_backfill_unlocked(
                    user_id,
                    health_user_id,
                    cursor_meal_id,
                    queued_count=queued_count,
                    skipped_count=skipped_count,
                    version=version,
                    now=reference_time,
                    lease_seconds=lease_seconds,
                )
                await self._conn.execute("COMMIT")
                return updated
            except BaseException:
                await self._rollback()
                raise

    async def _advance_backfill_unlocked(
        self,
        user_id: str,
        health_user_id: str,
        cursor_meal_id: int,
        *,
        queued_count: int,
        skipped_count: int,
        version: int,
        now: float,
        lease_seconds: int,
    ) -> bool:
        """Advance a claimed backfill while the caller owns the lock or tx."""
        row = await self._fetch_one(
            """
            SELECT high_water_meal_id FROM google_health_nutrition_backfills
            WHERE telegram_user_id = ? AND health_user_id = ?
              AND backfill_version = ? AND status = 'running'
            """,
            (user_id, health_user_id, version),
        )
        if row is None:
            return False
        completed = cursor_meal_id >= int(row["high_water_meal_id"])
        await self._conn.execute(
            """
            UPDATE google_health_nutrition_backfills
            SET cursor_meal_id = ?, status = ?, lease_expires_at = ?,
                queued_count = queued_count + ?, skipped_count = skipped_count + ?,
                updated_at = ?, completed_at = CASE WHEN ? THEN ? ELSE completed_at END
            WHERE telegram_user_id = ? AND health_user_id = ?
              AND backfill_version = ? AND status = 'running'
            """,
            (
                cursor_meal_id,
                "completed" if completed else "running",
                None if completed else now + lease_seconds,
                queued_count,
                skipped_count,
                now,
                completed,
                now if completed else None,
                user_id,
                health_user_id,
                version,
            ),
        )
        return True

    async def fail_backfill(
        self,
        user_id: str,
        health_user_id: str,
        error_code: str,
        *,
        version: int = BACKFILL_VERSION,
        now: float | None = None,
    ) -> None:
        """Release a backfill lease with a safe resumable error code."""
        safe_error = _safe_error_code(error_code)
        reference_time = time.time() if now is None else now
        async with self._write_transaction():
            await self._conn.execute(
                """
                UPDATE google_health_nutrition_backfills
                SET status = 'pending', lease_expires_at = NULL,
                    last_error = ?, updated_at = ?
                WHERE telegram_user_id = ? AND health_user_id = ?
                  AND backfill_version = ? AND status = 'running'
                """,
                (safe_error, reference_time, user_id, health_user_id, version),
            )

    @asynccontextmanager
    async def _write_transaction(self) -> AsyncIterator[None]:
        """Run one short nutrition state update transaction under the lock."""
        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                yield
            except BaseException:
                await self._rollback()
                raise
            else:
                await self._conn.execute("COMMIT")

    async def _rollback(self) -> None:
        """Roll back a transaction without hiding the original exception."""
        with suppress(Exception):
            await self._conn.execute("ROLLBACK")


def _first_dict_payload(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first JSON object from newest-first storage rows."""
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _payload_json_matches(payload_json: Any, payload: dict[str, Any]) -> bool:
    """Compare stored and intended payloads without trusting malformed JSON."""
    try:
        stored = json.loads(payload_json)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(stored, dict) and stored == payload


def _safe_error_code(error_code: str) -> str:
    """Keep ledger errors bounded and free of provider payloads."""
    if error_code.isascii() and error_code.isprintable():
        return error_code[:80]
    return "backfill_error"


def _resource_belongs_to_health_user(resource_name: str, health_user_id: str) -> bool:
    """Match a resource to one exact Google Health account."""
    return resource_name.startswith(
        f"users/{health_user_id}/dataTypes/nutrition-log/dataPoints/"
    )


def _health_user_id_from_resource(resource_name: str) -> str | None:
    """Extract an account ID from a canonical nutrition resource name."""
    prefix = "users/"
    suffix = "/dataTypes/nutrition-log/dataPoints/"
    if not resource_name.startswith(prefix) or suffix not in resource_name:
        return None
    health_user_id = resource_name[len(prefix) :].split(suffix, 1)[0]
    return health_user_id or None
