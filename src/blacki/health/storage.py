"""SQLite persistence for Google Health credentials and normalized records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from blacki.storage.base import SqlStorage
from blacki.utils.timezone import now_utc

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class HealthConnection:
    """Stored Google Health connection metadata."""

    telegram_user_id: str
    encrypted_refresh_token: str | None
    health_user_id: str
    legacy_fitbit_user_id: str | None
    scopes: tuple[str, ...]
    status: str
    connected_at: str
    last_synced_at: str | None
    last_refresh_requested_at: str | None
    last_sync_error: str | None


class SqliteGoogleHealthStorage(SqlStorage):
    """Store encrypted credentials, OAuth state, and normalized daily records."""

    async def _create_tables(self) -> None:
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS google_health_connections (
                telegram_user_id TEXT PRIMARY KEY,
                encrypted_refresh_token TEXT,
                health_user_id TEXT NOT NULL,
                legacy_fitbit_user_id TEXT,
                scopes_json TEXT NOT NULL,
                status TEXT NOT NULL,
                connected_at TEXT NOT NULL,
                last_synced_at TEXT,
                last_refresh_requested_at TEXT,
                last_sync_error TEXT
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS google_health_oauth_states (
                state_hash TEXT PRIMARY KEY,
                telegram_user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS google_health_daily_summaries (
                telegram_user_id TEXT NOT NULL,
                summary_date TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (telegram_user_id, summary_date)
            )
        """)
        await self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_google_health_states_expiry
            ON google_health_oauth_states (expires_at)
            """
        )

    async def store_oauth_state(
        self,
        state_hash: str,
        telegram_user_id: str,
        *,
        expires_at: datetime,
    ) -> None:
        """Store a hashed, short-lived, single-use OAuth state value."""
        now = now_utc()
        async with self._lock:
            await self._conn.execute(
                "DELETE FROM google_health_oauth_states WHERE expires_at <= ?",
                (_iso(now),),
            )
            await self._conn.execute(
                """
                INSERT INTO google_health_oauth_states
                    (state_hash, telegram_user_id, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (state_hash, telegram_user_id, _iso(expires_at), _iso(now)),
            )

    async def consume_oauth_state(
        self, state_hash: str, *, now: datetime | None = None
    ) -> str | None:
        """Consume one valid state and return its bound Telegram identity."""
        reference_time = now or now_utc()
        async with self._lock:
            row = await self._fetch_one(
                """
                SELECT telegram_user_id, expires_at
                FROM google_health_oauth_states
                WHERE state_hash = ?
                """,
                (state_hash,),
            )
            if row is None:
                return None
            await self._conn.execute(
                "DELETE FROM google_health_oauth_states WHERE state_hash = ?",
                (state_hash,),
            )
            expires_at = _parse_timestamp(row["expires_at"])
            if expires_at <= reference_time.astimezone(UTC):
                return None
            return str(row["telegram_user_id"])

    async def upsert_connection(
        self,
        *,
        telegram_user_id: str,
        encrypted_refresh_token: str,
        health_user_id: str,
        legacy_fitbit_user_id: str | None,
        scopes: tuple[str, ...],
    ) -> None:
        """Create or replace a connected account without storing plaintext tokens."""
        connected_at = _iso(now_utc())
        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO google_health_connections (
                    telegram_user_id, encrypted_refresh_token, health_user_id,
                    legacy_fitbit_user_id, scopes_json, status, connected_at,
                    last_synced_at, last_refresh_requested_at, last_sync_error
                ) VALUES (?, ?, ?, ?, ?, 'connected', ?, NULL, NULL, NULL)
                ON CONFLICT (telegram_user_id) DO UPDATE SET
                    encrypted_refresh_token = excluded.encrypted_refresh_token,
                    health_user_id = excluded.health_user_id,
                    legacy_fitbit_user_id = excluded.legacy_fitbit_user_id,
                    scopes_json = excluded.scopes_json,
                    status = 'connected',
                    connected_at = excluded.connected_at,
                    last_sync_error = NULL
                """,
                (
                    telegram_user_id,
                    encrypted_refresh_token,
                    health_user_id,
                    legacy_fitbit_user_id,
                    json.dumps(sorted(scopes)),
                    connected_at,
                ),
            )

    async def get_connection(self, telegram_user_id: str) -> HealthConnection | None:
        """Return one connection without decrypting its token."""
        row = await self._fetch_one(
            """
            SELECT telegram_user_id, encrypted_refresh_token, health_user_id,
                   legacy_fitbit_user_id, scopes_json, status, connected_at,
                   last_synced_at, last_refresh_requested_at, last_sync_error
            FROM google_health_connections
            WHERE telegram_user_id = ?
            """,
            (telegram_user_id,),
        )
        return _connection_from_row(row) if row is not None else None

    async def list_active_connections(self) -> list[HealthConnection]:
        """Return only connections that still have usable local credentials."""
        rows = await self._fetch_all(
            """
            SELECT telegram_user_id, encrypted_refresh_token, health_user_id,
                   legacy_fitbit_user_id, scopes_json, status, connected_at,
                   last_synced_at, last_refresh_requested_at, last_sync_error
            FROM google_health_connections
            WHERE encrypted_refresh_token IS NOT NULL AND status = 'connected'
            """
        )
        return [_connection_from_row(row) for row in rows]

    async def mark_synced(self, telegram_user_id: str) -> None:
        """Record a successful sync without storing raw provider responses."""
        async with self._lock:
            await self._conn.execute(
                """
                UPDATE google_health_connections
                SET last_synced_at = ?, last_sync_error = NULL
                WHERE telegram_user_id = ?
                """,
                (_iso(now_utc()), telegram_user_id),
            )

    async def mark_reauthorization_required(
        self, telegram_user_id: str, error_code: str = "authorization_required"
    ) -> None:
        """Disable scheduled pulls while retaining only safe status metadata."""
        safe_error = (
            error_code
            if error_code.isascii()
            and error_code.isprintable()
            and len(error_code) <= 80
            else "authorization_required"
        )
        async with self._lock:
            await self._conn.execute(
                """
                UPDATE google_health_connections
                SET encrypted_refresh_token = NULL,
                    status = 'reauthorization_required',
                    last_sync_error = ?
                WHERE telegram_user_id = ?
                """,
                (safe_error, telegram_user_id),
            )

    async def claim_manual_refresh(
        self,
        telegram_user_id: str,
        *,
        cooldown_seconds: int,
        now: datetime | None = None,
    ) -> tuple[bool, datetime | None]:
        """Claim the manual refresh slot for a user under the shared write lock."""
        reference_time = now or now_utc()
        async with self._lock:
            row = await self._fetch_one(
                """
                SELECT encrypted_refresh_token, status, last_refresh_requested_at
                FROM google_health_connections
                WHERE telegram_user_id = ?
                """,
                (telegram_user_id,),
            )
            if row is None or row["encrypted_refresh_token"] is None:
                return False, None
            if row["status"] != "connected":
                return False, None
            last_requested = row["last_refresh_requested_at"]
            if isinstance(last_requested, str):
                next_allowed = _parse_timestamp(last_requested) + timedelta(
                    seconds=cooldown_seconds
                )
                if next_allowed > reference_time.astimezone(UTC):
                    return False, next_allowed
            await self._conn.execute(
                """
                UPDATE google_health_connections
                SET last_refresh_requested_at = ?
                WHERE telegram_user_id = ?
                """,
                (_iso(reference_time), telegram_user_id),
            )
            return True, None

    async def upsert_daily_summaries(
        self,
        telegram_user_id: str,
        summaries: list[Mapping[str, Any]],
    ) -> None:
        """Upsert normalized days by Telegram identity and civil date."""
        now = _iso(now_utc())
        rows: list[tuple[str, str, str, str]] = []
        for summary in summaries:
            summary_date = summary.get("date")
            if not isinstance(summary_date, str) or not summary_date:
                continue
            rows.append(
                (telegram_user_id, summary_date, json.dumps(dict(summary)), now)
            )
        if not rows:
            return
        async with self._lock:
            await self._conn.executemany(
                """
                INSERT INTO google_health_daily_summaries
                    (telegram_user_id, summary_date, summary_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (telegram_user_id, summary_date) DO UPDATE SET
                    summary_json = excluded.summary_json,
                    updated_at = excluded.updated_at
                """,
                rows,
            )

    async def get_daily_summaries(
        self,
        telegram_user_id: str,
        *,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Read only normalized records in a closed-open date range."""
        rows = await self._fetch_all(
            """
            SELECT summary_json
            FROM google_health_daily_summaries
            WHERE telegram_user_id = ? AND summary_date >= ? AND summary_date < ?
            ORDER BY summary_date ASC
            """,
            (telegram_user_id, start_date, end_date),
        )
        summaries: list[dict[str, Any]] = []
        for row in rows:
            try:
                decoded = json.loads(row["summary_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(decoded, dict):
                summaries.append(decoded)
        return summaries

    async def delete_connection(self, telegram_user_id: str) -> bool:
        """Delete credentials, OAuth state, and normalized health data."""
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM google_health_connections WHERE telegram_user_id = ?",
                (telegram_user_id,),
            )
            await self._conn.execute(
                "DELETE FROM google_health_oauth_states WHERE telegram_user_id = ?",
                (telegram_user_id,),
            )
            await self._conn.execute(
                "DELETE FROM google_health_daily_summaries WHERE telegram_user_id = ?",
                (telegram_user_id,),
            )
            return cursor.rowcount > 0


def _connection_from_row(row: Mapping[str, Any]) -> HealthConnection:
    try:
        scopes_value = json.loads(row["scopes_json"])
    except (TypeError, json.JSONDecodeError):
        scopes_value = []
    scopes = (
        tuple(item for item in scopes_value if isinstance(item, str))
        if isinstance(scopes_value, list)
        else ()
    )
    return HealthConnection(
        telegram_user_id=str(row["telegram_user_id"]),
        encrypted_refresh_token=(
            str(row["encrypted_refresh_token"])
            if row["encrypted_refresh_token"] is not None
            else None
        ),
        health_user_id=str(row["health_user_id"]),
        legacy_fitbit_user_id=(
            str(row["legacy_fitbit_user_id"])
            if row["legacy_fitbit_user_id"] is not None
            else None
        ),
        scopes=scopes,
        status=str(row["status"]),
        connected_at=str(row["connected_at"]),
        last_synced_at=(
            str(row["last_synced_at"]) if row["last_synced_at"] is not None else None
        ),
        last_refresh_requested_at=(
            str(row["last_refresh_requested_at"])
            if row["last_refresh_requested_at"] is not None
            else None
        ),
        last_sync_error=(
            str(row["last_sync_error"]) if row["last_sync_error"] is not None else None
        ),
    )


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
