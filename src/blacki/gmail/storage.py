"""SQLite persistence for Gmail OAuth state and encrypted connections."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from blacki.storage.base import SqlStorage

from .config import GMAIL_SCOPE, canonical_gmail_user_id
from .errors import GmailAlreadyConnectedError, GmailCredentialError


@dataclass(frozen=True, slots=True)
class GmailConnection:
    """Stored metadata for one canonical Telegram Gmail connection."""

    telegram_user_id: str
    encrypted_refresh_token: str
    scopes: tuple[str, ...]
    status: str
    connected_at: float


class SqliteGmailStorage(SqlStorage):
    """Store hashed OAuth state and encrypted refresh tokens in shared SQLite."""

    async def _create_tables(self) -> None:
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gmail_connections (
                telegram_user_id TEXT PRIMARY KEY,
                encrypted_refresh_token TEXT,
                scopes_json TEXT NOT NULL,
                status TEXT NOT NULL,
                connected_at REAL NOT NULL
            )
            """
        )
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gmail_oauth_states (
                state_hash TEXT PRIMARY KEY,
                telegram_user_id TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        await self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_gmail_oauth_states_expires_at
            ON gmail_oauth_states (expires_at)
            """
        )

    async def store_oauth_state(
        self,
        state_hash: str,
        telegram_user_id: str,
        *,
        expires_at: float,
        now: float | None = None,
    ) -> None:
        """Store only a hash of a short-lived, single-use OAuth state."""
        user_id = _require_user_id(telegram_user_id)
        if len(state_hash) != 64 or any(
            char not in "0123456789abcdef" for char in state_hash
        ):
            raise GmailCredentialError("Gmail OAuth state hash is invalid")
        reference_time = time.time() if now is None else now
        if expires_at <= reference_time:
            raise GmailCredentialError("Gmail OAuth state expiry is invalid")
        async with self._lock:
            await self._conn.execute(
                "DELETE FROM gmail_oauth_states WHERE expires_at <= ?",
                (reference_time,),
            )
            await self._conn.execute(
                """
                INSERT INTO gmail_oauth_states
                    (state_hash, telegram_user_id, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (state_hash, user_id, expires_at, reference_time),
            )

    async def consume_oauth_state(
        self,
        state_hash: str,
        *,
        now: float | None = None,
    ) -> str | None:
        """Consume one valid state atomically and return its bound user ID."""
        reference_time = time.time() if now is None else now
        async with self._lock:
            row = await self._fetch_one(
                """
                SELECT telegram_user_id, expires_at
                FROM gmail_oauth_states
                WHERE state_hash = ?
                """,
                (state_hash,),
            )
            if row is None:
                return None
            await self._conn.execute(
                "DELETE FROM gmail_oauth_states WHERE state_hash = ?",
                (state_hash,),
            )
            if float(row["expires_at"]) <= reference_time:
                return None
            return _require_user_id(str(row["telegram_user_id"]))

    async def get_connection(self, telegram_user_id: str) -> GmailConnection | None:
        """Return one connection without decrypting its refresh token."""
        user_id = _require_user_id(telegram_user_id)
        row = await self._fetch_one(
            """
            SELECT telegram_user_id, encrypted_refresh_token, scopes_json,
                   status, connected_at
            FROM gmail_connections
            WHERE telegram_user_id = ?
            """,
            (user_id,),
        )
        return _connection_from_row(row) if row is not None else None

    async def has_connection(self, telegram_user_id: str) -> bool:
        """Return whether a user has a usable connected Gmail account."""
        connection = await self.get_connection(telegram_user_id)
        return bool(
            connection
            and connection.status == "connected"
            and connection.encrypted_refresh_token
        )

    async def save_connection(
        self,
        *,
        telegram_user_id: str,
        encrypted_refresh_token: str,
        scopes: tuple[str, ...],
        connected_at: float | None = None,
    ) -> None:
        """Save a connection while refusing silent account replacement."""
        user_id = _require_user_id(telegram_user_id)
        if not encrypted_refresh_token:
            raise GmailCredentialError("Gmail refresh token is empty")
        if GMAIL_SCOPE not in scopes:
            raise GmailCredentialError("Gmail connection is missing the required scope")
        timestamp = time.time() if connected_at is None else connected_at
        async with self._lock:
            existing = await self._fetch_one(
                """
                SELECT encrypted_refresh_token, status
                FROM gmail_connections
                WHERE telegram_user_id = ?
                """,
                (user_id,),
            )
            if existing is not None and (
                existing["status"] == "connected" or existing["encrypted_refresh_token"]
            ):
                raise GmailAlreadyConnectedError(
                    "Gmail is already connected. Disconnect it before connecting "
                    "another account."
                )
            await self._conn.execute(
                """
                INSERT INTO gmail_connections (
                    telegram_user_id, encrypted_refresh_token, scopes_json,
                    status, connected_at
                ) VALUES (?, ?, ?, 'connected', ?)
                ON CONFLICT (telegram_user_id) DO UPDATE SET
                    encrypted_refresh_token = excluded.encrypted_refresh_token,
                    scopes_json = excluded.scopes_json,
                    status = 'connected',
                    connected_at = excluded.connected_at
                """,
                (
                    user_id,
                    encrypted_refresh_token,
                    json.dumps(sorted(scopes)),
                    timestamp,
                ),
            )

    async def replace_refresh_token(
        self,
        telegram_user_id: str,
        encrypted_refresh_token: str,
    ) -> None:
        """Replace a rotated refresh token without exposing its plaintext."""
        user_id = _require_user_id(telegram_user_id)
        if not encrypted_refresh_token:
            raise GmailCredentialError("Gmail refresh token is empty")
        async with self._lock:
            await self._conn.execute(
                """
                UPDATE gmail_connections
                SET encrypted_refresh_token = ?, status = 'connected'
                WHERE telegram_user_id = ? AND status = 'connected'
                """,
                (encrypted_refresh_token, user_id),
            )

    async def mark_reauthorization_required(self, telegram_user_id: str) -> None:
        """Disable a connection after invalid_grant or a missing scope."""
        user_id = _require_user_id(telegram_user_id)
        async with self._lock:
            await self._conn.execute(
                """
                UPDATE gmail_connections
                SET encrypted_refresh_token = NULL,
                    status = 'reauthorization_required'
                WHERE telegram_user_id = ?
                """,
                (user_id,),
            )

    async def mark_revocation_required(self, telegram_user_id: str) -> None:
        """Disable a connection while retaining its token for revoke retry."""
        user_id = _require_user_id(telegram_user_id)
        async with self._lock:
            await self._conn.execute(
                """
                UPDATE gmail_connections
                SET status = 'revocation_required'
                WHERE telegram_user_id = ?
                """,
                (user_id,),
            )

    async def remove_connection(self, telegram_user_id: str) -> GmailConnection | None:
        """Remove exactly one user's connection and return its prior metadata."""
        user_id = _require_user_id(telegram_user_id)
        async with self._lock:
            connection = await self.get_connection(user_id)
            await self._conn.execute(
                "DELETE FROM gmail_connections WHERE telegram_user_id = ?",
                (user_id,),
            )
            return connection


def _require_user_id(user_id: str) -> str:
    canonical = canonical_gmail_user_id(user_id)
    if canonical is None:
        raise GmailCredentialError("Gmail requires a private Telegram user identity")
    return canonical


def _connection_from_row(row: dict[str, Any]) -> GmailConnection:
    raw_scopes = row.get("scopes_json")
    if not isinstance(raw_scopes, str):
        raise GmailCredentialError("Stored Gmail connection metadata is corrupt")
    try:
        scopes_value = json.loads(raw_scopes)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GmailCredentialError(
            "Stored Gmail connection metadata is corrupt"
        ) from exc
    if not isinstance(scopes_value, list) or not all(
        isinstance(scope, str) for scope in scopes_value
    ):
        raise GmailCredentialError("Stored Gmail connection metadata is corrupt")
    token = row.get("encrypted_refresh_token")
    if token is not None and not isinstance(token, str):
        raise GmailCredentialError("Stored Gmail connection metadata is corrupt")
    status = row.get("status")
    if not isinstance(status, str):
        raise GmailCredentialError("Stored Gmail connection metadata is corrupt")
    return GmailConnection(
        telegram_user_id=_require_user_id(str(row["telegram_user_id"])),
        encrypted_refresh_token=token or "",
        scopes=tuple(scopes_value),
        status=status,
        connected_at=float(row["connected_at"]),
    )
