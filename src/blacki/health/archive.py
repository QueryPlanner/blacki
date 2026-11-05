"""Private, account-bound archive of provider observations and ingestion progress.

This is an archive, not a mirror: provider deletions do not erase observations.
No payloads are exposed to the model, sandbox, Telegram, or R2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from blacki.storage.base import SqlStorage
from blacki.utils.timezone import now_utc


class HealthArchiveStorage(SqlStorage):
    """Persist bounded pages and their checkpoints in the same transaction."""

    async def _create_tables(self) -> None:
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS google_health_archive (
                owner TEXT NOT NULL, account TEXT NOT NULL, stream TEXT NOT NULL,
                digest TEXT NOT NULL, payload TEXT NOT NULL,
                first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
                PRIMARY KEY (owner, account, stream, digest)
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS google_health_imports (
                owner TEXT NOT NULL, stream TEXT NOT NULL, generation TEXT NOT NULL,
                checkpoint TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (owner, stream)
            )
        """)

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS google_health_import_tokens (
                owner TEXT NOT NULL, stream TEXT NOT NULL, scan TEXT NOT NULL,
                digest TEXT NOT NULL, PRIMARY KEY (owner, stream, scan, digest)
            )
        """)

    async def cursor_seen(
        self, owner: str, stream: str, scan: str, token: str, request: str = ""
    ) -> bool:
        row = await self._fetch_one(
            "SELECT 1 FROM google_health_import_tokens "
            "WHERE owner = ? AND stream = ? AND scan = ? AND digest = ?",
            (
                owner,
                stream,
                scan,
                hashlib.sha256(f"{request}:{token}".encode()).hexdigest(),
            ),
        )
        return row is not None

    async def clear(self, owner: str, *, records: bool = True) -> None:
        """Clear within the caller's connection lifecycle transaction."""
        await self._conn.execute(
            "DELETE FROM google_health_imports WHERE owner = ?", (owner,)
        )
        await self._conn.execute(
            "DELETE FROM google_health_import_tokens WHERE owner = ?", (owner,)
        )
        if records:
            await self._conn.execute(
                "DELETE FROM google_health_archive WHERE owner = ?", (owner,)
            )

    async def progress(self, owner: str) -> dict[str, dict[str, Any]]:
        rows = await self._fetch_all(
            "SELECT stream, checkpoint FROM google_health_imports WHERE owner = ?",
            (owner,),
        )
        return {str(row["stream"]): json.loads(row["checkpoint"]) for row in rows}

    async def membership_start(self, owner: str, account: str) -> str | None:
        row = await self._fetch_one(
            "SELECT payload FROM google_health_archive WHERE owner = ? AND account = ? "
            "AND stream = 'profile' ORDER BY last_seen DESC LIMIT 1",
            (owner, account),
        )
        if row is None:
            return None
        from datetime import date

        value = (
            json.loads(row["payload"]).get("data", {}).get("membershipStartDate", {})
        )
        try:
            return date(value["year"], value["month"], value["day"]).isoformat()
        except (KeyError, TypeError, ValueError):
            return None

    async def commit_page(
        self,
        *,
        owner: str,
        account: str,
        generation: str,
        stream: str,
        expected: Mapping[str, Any] | None,
        checkpoint: Mapping[str, Any],
        records: list[dict[str, Any]],
    ) -> bool:
        """Reject stale work after disconnect, reconsent, or another page commit."""
        timestamp = now_utc().isoformat()
        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                connection = await self._fetch_one(
                    "SELECT health_user_id, generation, status "
                    "FROM google_health_connections "
                    "WHERE telegram_user_id = ?",
                    (owner,),
                )
                previous = await self._fetch_one(
                    "SELECT checkpoint FROM google_health_imports "
                    "WHERE owner = ? AND stream = ?",
                    (owner, stream),
                )
                actual = json.loads(previous["checkpoint"]) if previous else None
                if (
                    connection is None
                    or connection["health_user_id"] != account
                    or connection["generation"] != generation
                    or connection["status"] != "connected"
                    or actual != expected
                ):
                    await self._conn.execute("ROLLBACK")
                    return False
                if checkpoint.get("scanId"):
                    await self._conn.execute(
                        "DELETE FROM google_health_import_tokens "
                        "WHERE owner = ? AND stream = ? AND scan != ?",
                        (owner, stream, checkpoint["scanId"]),
                    )
                if checkpoint.get("pageToken"):
                    await self._conn.execute(
                        "INSERT OR IGNORE INTO google_health_import_tokens "
                        "VALUES (?, ?, ?, ?)",
                        (
                            owner,
                            stream,
                            checkpoint["scanId"],
                            hashlib.sha256(
                                (
                                    str(checkpoint.get("requestSignature", ""))
                                    + ":"
                                    + str(checkpoint["pageToken"])
                                ).encode()
                            ).hexdigest(),
                        ),
                    )
                for record in records:
                    payload = json.dumps(
                        record, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    digest = hashlib.sha256(payload.encode()).hexdigest()
                    await self._conn.execute(
                        "INSERT INTO google_health_archive "
                        "VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(owner, account, stream, digest) DO UPDATE "
                        "SET last_seen = excluded.last_seen",
                        (owner, account, stream, digest, payload, timestamp, timestamp),
                    )
                await self._conn.execute(
                    "INSERT INTO google_health_imports VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(owner, stream) DO UPDATE "
                    "SET generation = excluded.generation, "
                    "checkpoint = excluded.checkpoint, "
                    "updated_at = excluded.updated_at",
                    (
                        owner,
                        stream,
                        generation,
                        json.dumps(dict(checkpoint), sort_keys=True),
                        timestamp,
                    ),
                )
                await self._conn.execute("COMMIT")
                return True
            except BaseException:
                await self._conn.execute("ROLLBACK")
                raise

    async def coverage(self, owner: str) -> dict[str, Any]:
        """Return status metadata without health payloads or provider identifiers."""
        from .catalog import HEALTH_STREAMS, UNREADABLE_DATA_TYPES

        progress = await self.progress(owner)
        count = await self._fetch_all(
            "SELECT count(*) AS observations FROM google_health_archive "
            "WHERE owner = ?",
            (owner,),
        )
        statuses = {
            spec.key: {
                key: value
                for key, value in progress.get(spec.key, {"status": "pending"}).items()
                if key
                in {
                    "status",
                    "representation",
                    "completed",
                    "attempted",
                    "error",
                    "historyStart",
                    "rangeStart",
                    "rangeEnd",
                    "recordsSeen",
                    "mediaStatus",
                }
            }
            for spec in HEALTH_STREAMS
        }
        return {
            "observations_stored": int(count[0]["observations"]),
            "storage": "private_host_database",
            "semantics": "observation_archive",
            "streams": statuses,
            "unreadable_types": list(UNREADABLE_DATA_TYPES),
            "complete": all(
                value.get("status") == "complete"
                and value.get("mediaStatus")
                not in {"unavailable", "permission_required"}
                for value in statuses.values()
            ),
        }
