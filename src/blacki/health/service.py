"""OAuth, synchronization, and summary orchestration for Google Health."""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any

from blacki.utils.timezone import now_utc

from .client import (
    GoogleHealthApiError,
    GoogleHealthAuthError,
    GoogleHealthClient,
)
from .config import (
    GOOGLE_HEALTH_SCOPES,
    GoogleHealthConfig,
    TokenEncryptionError,
    health_user_id_for_telegram_user,
)
from .normalize import normalize_data_points
from .storage import HealthConnection, SqliteGoogleHealthStorage

logger = logging.getLogger(__name__)

SYNC_DATA_TYPES = (
    "steps",
    "distance",
    "active-energy-burned",
    "active-minutes",
    "active-zone-minutes",
    "exercise",
    "sleep",
    "daily-resting-heart-rate",
    "daily-heart-rate-zones",
    "time-in-heart-rate-zone",
    "weight",
    "body-fat",
)


class GoogleHealthOAuthError(ValueError):
    """Raised when an OAuth state or authorization response is unusable."""


@dataclass(frozen=True, slots=True)
class OAuthCompletion:
    """Safe result of a callback, without tokens or provider identifiers."""

    telegram_user_id: str
    connected: bool


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Safe synchronization result suitable for a Telegram status message."""

    status: str
    telegram_user_id: str
    days_upserted: int = 0
    records_fetched: int = 0
    unavailable_data_types: tuple[str, ...] = ()
    next_allowed_at: str | None = None


class GoogleHealthService:
    """Coordinate the Google Health client and application storage."""

    def __init__(
        self,
        config: GoogleHealthConfig,
        storage: SqliteGoogleHealthStorage,
        *,
        client: GoogleHealthClient | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.client = client or GoogleHealthClient(config)

    async def close(self) -> None:
        """Close the owned Google Health HTTP client."""
        await self.client.close()

    async def begin_authorization(self, telegram_user_id: str) -> str:
        """Create a single-use state and return the Google authorization URL."""
        health_user_id = _canonical_user_id(telegram_user_id)
        state = secrets.token_urlsafe(32)
        await self.storage.store_oauth_state(
            hashlib.sha256(state.encode("utf-8")).hexdigest(),
            health_user_id,
            expires_at=now_utc()
            + timedelta(seconds=self.config.oauth_state_ttl_seconds),
        )
        return self.config.authorization_url(state)

    async def complete_authorization(
        self,
        *,
        state: str,
        code: str | None,
        error: str | None,
    ) -> OAuthCompletion:
        """Validate state, exchange code, resolve identity, and store credentials."""
        if not state:
            raise GoogleHealthOAuthError("OAuth state is missing")
        user_id = await self.storage.consume_oauth_state(
            hashlib.sha256(state.encode("utf-8")).hexdigest()
        )
        if user_id is None:
            raise GoogleHealthOAuthError("OAuth state is invalid or expired")
        if error:
            return OAuthCompletion(telegram_user_id=user_id, connected=False)
        if not code:
            raise GoogleHealthOAuthError("OAuth authorization code is missing")

        token_response = await self.client.exchange_code(code)
        refresh_token = token_response.refresh_token
        if refresh_token is None:
            existing = await self.storage.get_connection(user_id)
            refresh_token = self._decrypt_existing_token(existing)
        if refresh_token is None:
            raise GoogleHealthOAuthError("Google did not return a refresh token")

        identity = await self.client.get_identity(token_response.access_token)
        await self.storage.upsert_connection(
            telegram_user_id=user_id,
            encrypted_refresh_token=self.config.cipher.encrypt(refresh_token),
            health_user_id=identity.health_user_id,
            legacy_fitbit_user_id=identity.legacy_fitbit_user_id,
            scopes=token_response.scopes or GOOGLE_HEALTH_SCOPES,
        )
        return OAuthCompletion(telegram_user_id=user_id, connected=True)

    async def connection_status(self, telegram_user_id: str) -> dict[str, Any]:
        """Return a safe connection status without account IDs or token material."""
        user_id = _canonical_user_id(telegram_user_id)
        connection = await self.storage.get_connection(user_id)
        if connection is None:
            return {"status": "not_connected"}
        result: dict[str, Any] = {
            "status": connection.status,
            "last_synced_at": connection.last_synced_at,
        }
        if connection.last_sync_error:
            result["last_sync_error"] = connection.last_sync_error
        return result

    async def disconnect(self, telegram_user_id: str) -> bool:
        """Revoke the remote token best-effort, then delete local health data."""
        user_id = _canonical_user_id(telegram_user_id)
        connection = await self.storage.get_connection(user_id)
        if connection is not None and connection.encrypted_refresh_token is not None:
            try:
                refresh_token = self.config.cipher.decrypt(
                    connection.encrypted_refresh_token
                )
            except TokenEncryptionError:
                refresh_token = None
            if refresh_token is not None:
                try:
                    await self.client.revoke_token(refresh_token)
                except GoogleHealthApiError:
                    logger.warning("Google Health remote token revocation failed")
        return await self.storage.delete_connection(user_id)

    async def refresh_user(
        self, telegram_user_id: str, *, days: int = 14
    ) -> SyncResult:
        """Perform one rate-limited on-demand synchronization."""
        user_id = _canonical_user_id(telegram_user_id)
        allowed, next_allowed_at = await self.storage.claim_manual_refresh(
            user_id,
            cooldown_seconds=self.config.manual_refresh_cooldown_seconds,
        )
        if not allowed:
            status = await self.connection_status(user_id)
            if status["status"] != "connected":
                return SyncResult(status=status["status"], telegram_user_id=user_id)
            return SyncResult(
                status="rate_limited",
                telegram_user_id=user_id,
                next_allowed_at=next_allowed_at.isoformat()
                if next_allowed_at
                else None,
            )
        return await self.sync_user(user_id, days=days)

    async def sync_user(self, telegram_user_id: str, *, days: int = 14) -> SyncResult:
        """Refresh tokens, fetch a bounded window, normalize, and upsert records."""
        user_id = _canonical_user_id(telegram_user_id)
        connection = await self.storage.get_connection(user_id)
        if connection is None:
            return SyncResult(status="not_connected", telegram_user_id=user_id)
        if (
            connection.status != "connected"
            or connection.encrypted_refresh_token is None
        ):
            return SyncResult(status=connection.status, telegram_user_id=user_id)

        try:
            refresh_token = self.config.cipher.decrypt(
                connection.encrypted_refresh_token
            )
        except TokenEncryptionError:
            await self.storage.mark_reauthorization_required(
                user_id, "stored_token_invalid"
            )
            return SyncResult(
                status="reauthorization_required", telegram_user_id=user_id
            )

        try:
            token = await self.client.refresh_access_token(refresh_token)
        except GoogleHealthAuthError as exc:
            logger.warning(
                "Google Health token refresh requires authorization: "
                "status_code=%s error_code=%s",
                exc.status_code,
                exc.error_code,
            )
            await self.storage.mark_reauthorization_required(
                user_id, exc.error_code or "authorization_required"
            )
            return SyncResult(
                status="reauthorization_required", telegram_user_id=user_id
            )
        except GoogleHealthApiError as exc:
            if exc.error_code == "invalid_grant":
                await self.storage.mark_reauthorization_required(
                    user_id, "invalid_grant"
                )
                return SyncResult(
                    status="reauthorization_required", telegram_user_id=user_id
                )
            logger.warning(
                "Google Health token refresh failed: status_code=%s error_code=%s",
                exc.status_code,
                exc.error_code,
            )
            return SyncResult(status="failed", telegram_user_id=user_id)

        start_time, end_time = _sync_window(days)
        data_by_type: dict[str, list[dict[str, Any]]] = {}
        unavailable: list[str] = []
        records_fetched = 0
        for data_type in SYNC_DATA_TYPES:
            try:
                points = await self.client.list_data_points(
                    token.access_token,
                    data_type,
                    start_time=start_time,
                    end_time=end_time,
                )
            except GoogleHealthAuthError as exc:
                logger.warning(
                    "Google Health data fetch requires authorization: "
                    "data_type=%s status_code=%s error_code=%s",
                    data_type,
                    exc.status_code,
                    exc.error_code,
                )
                if exc.status_code == 403:
                    unavailable.append(data_type)
                    continue
                await self.storage.mark_reauthorization_required(
                    user_id, exc.error_code or "authorization_required"
                )
                return SyncResult(
                    status="reauthorization_required", telegram_user_id=user_id
                )
            except GoogleHealthApiError as exc:
                logger.warning(
                    "Google Health data fetch failed: data_type=%s "
                    "status_code=%s error_code=%s",
                    data_type,
                    exc.status_code,
                    exc.error_code,
                )
                return SyncResult(status="failed", telegram_user_id=user_id)
            data_by_type[data_type] = points
            records_fetched += len(points)

        normalized_days = normalize_data_points(data_by_type)
        await self.storage.upsert_daily_summaries(
            user_id, [day.to_dict() for day in normalized_days]
        )
        await self.storage.mark_synced(user_id)
        logger.info(
            "Google Health sync completed for one user "
            "(%d records, %d days, %d unavailable types)",
            records_fetched,
            len(normalized_days),
            len(unavailable),
        )
        return SyncResult(
            status="success",
            telegram_user_id=user_id,
            days_upserted=len(normalized_days),
            records_fetched=records_fetched,
            unavailable_data_types=tuple(unavailable),
        )

    async def sync_all(self) -> list[SyncResult]:
        """Synchronize every active connection for the background scheduler."""
        connections = await self.storage.list_active_connections()
        return [
            await self.sync_user(connection.telegram_user_id)
            for connection in connections
        ]

    async def summary(self, telegram_user_id: str, *, days: int = 7) -> dict[str, Any]:
        """Return normalized records and deterministic trend inputs."""
        return await summarize_stored_health(self.storage, telegram_user_id, days=days)

    async def summary_for_tool(
        self, telegram_user_id: str, *, days: int
    ) -> dict[str, Any]:
        """Validate a model-requested window before returning a summary."""
        if days < 1 or days > 14:
            return {"status": "error", "error": "days must be between 1 and 14"}
        return await self.summary(telegram_user_id, days=days)

    def _decrypt_existing_token(
        self, connection: HealthConnection | None
    ) -> str | None:
        if connection is None or connection.encrypted_refresh_token is None:
            return None
        try:
            return self.config.cipher.decrypt(connection.encrypted_refresh_token)
        except TokenEncryptionError:
            return None


async def summarize_stored_health(
    storage: SqliteGoogleHealthStorage,
    telegram_user_id: str,
    *,
    days: int = 7,
) -> dict[str, Any]:
    """Read normalized records without requiring an OAuth client instance."""
    user_id = _canonical_user_id(telegram_user_id)
    connection = await storage.get_connection(user_id)
    if connection is None:
        return {"status": "not_connected"}
    status: dict[str, Any] = {"status": connection.status}
    if connection.last_synced_at is not None:
        status["last_synced_at"] = connection.last_synced_at
    if connection.last_sync_error:
        status["last_sync_error"] = connection.last_sync_error
    if connection.status != "connected":
        return status

    start_date, end_date = _date_window(days)
    summaries = await storage.get_daily_summaries(
        user_id, start_date=start_date, end_date=end_date
    )
    return {
        "status": "success",
        "source": "google_health",
        "days": summaries,
        "trends": _build_trends(summaries),
    }


def format_health_summary(summary: dict[str, Any]) -> str:
    """Render a concise, non-diagnostic Telegram summary."""
    status = summary.get("status")
    if status == "not_connected":
        return "Google Health is not connected. Use /connect_health to authorize it."
    if status == "reauthorization_required":
        return (
            "Google Health needs authorization again. Use /connect_health to reconnect."
        )
    if status == "rate_limited":
        return "A health refresh was requested recently. Please try again later."
    if status != "success":
        return (
            "I couldn't read your Google Health data right now. Please try again later."
        )

    days = summary.get("days")
    if not isinstance(days, list) or not days:
        return (
            "Google Health is connected, but no imported records were available for "
            "the selected period. Missing values are omitted; nothing is guessed."
        )

    lines = ["📊 Google Health summary"]
    for day in days:
        if not isinstance(day, dict):
            continue
        date_value = day.get("date")
        if not isinstance(date_value, str):
            continue
        fields = [date_value]
        if isinstance(day.get("steps"), int):
            fields.append(f"{day['steps']:,} steps")
        if isinstance(day.get("distance_meters"), (int, float)):
            fields.append(f"{day['distance_meters'] / 1000:.1f} km")
        if isinstance(day.get("active_minutes"), int):
            fields.append(f"{day['active_minutes']} active min")
        if isinstance(day.get("active_zone_minutes"), int):
            fields.append(f"{day['active_zone_minutes']} zone min")
        if isinstance(day.get("resting_heart_rate_bpm"), int):
            fields.append(f"resting HR {day['resting_heart_rate_bpm']} bpm")
        sleep_minutes = _sleep_minutes_in_day(day)
        if sleep_minutes is not None:
            fields.append(f"sleep {_format_minutes(sleep_minutes)}")
        workouts = day.get("workouts")
        if isinstance(workouts, list) and workouts:
            fields.append(f"{len(workouts)} workout(s)")
        lines.append("• " + " · ".join(fields))

    trends = summary.get("trends")
    if isinstance(trends, dict):
        trend_text = _format_trends(trends)
        if trend_text:
            lines.append("")
            lines.append("7-day data averages: " + trend_text)
    lines.extend(
        [
            "",
            "This is wellness information, not medical advice. Data depends on "
            "what Google Health retained from connected sources.",
        ]
    )
    return "\n".join(lines)


def _canonical_user_id(user_id: str) -> str:
    canonical = health_user_id_for_telegram_user(user_id)
    if canonical is None:
        raise GoogleHealthOAuthError(
            "Google Health is available only in a private Telegram chat"
        )
    return canonical


def _sync_window(days: int) -> tuple[str, str]:
    safe_days = max(1, min(days, 14))
    current = now_utc()
    start = datetime.combine(
        current.date() - timedelta(days=safe_days - 1), time.min, tzinfo=UTC
    )
    end = datetime.combine(current.date() + timedelta(days=1), time.min, tzinfo=UTC)
    return _timestamp(start), _timestamp(end)


def _date_window(days: int) -> tuple[str, str]:
    safe_days = max(1, min(days, 14))
    current = now_utc().date()
    return (
        (current - timedelta(days=safe_days - 1)).isoformat(),
        (current + timedelta(days=1)).isoformat(),
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_trends(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    trends: dict[str, Any] = {}
    for key in (
        "steps",
        "active_minutes",
        "active_zone_minutes",
        "resting_heart_rate_bpm",
    ):
        values = [
            day[key]
            for day in summaries
            if isinstance(day, dict) and isinstance(day.get(key), (int, float))
        ]
        if values:
            trends[key] = {
                "average": round(sum(values) / len(values), 1),
                "latest": values[-1],
                "days_with_data": len(values),
            }
    sleep_values = [
        minutes
        for day in summaries
        if isinstance(day, dict)
        for minutes in [_sleep_minutes_in_day(day)]
        if minutes is not None
    ]
    if sleep_values:
        trends["sleep_minutes"] = {
            "average": round(sum(sleep_values) / len(sleep_values), 1),
            "latest": sleep_values[-1],
            "days_with_data": len(sleep_values),
        }
    return trends


def _sleep_minutes_in_day(day: dict[str, Any]) -> int | None:
    sleeps = day.get("sleep")
    if not isinstance(sleeps, list):
        return None
    values = [
        item["minutes"]
        for item in sleeps
        if isinstance(item, dict) and isinstance(item.get("minutes"), int)
    ]
    return sum(values) if values else None


def _format_minutes(minutes: int) -> str:
    hours, remainder = divmod(minutes, 60)
    return f"{hours}h {remainder:02d}m" if hours else f"{remainder}m"


def _format_trends(trends: dict[str, Any]) -> str:
    parts: list[str] = []
    labels = (
        ("steps", "steps"),
        ("active_minutes", "active min"),
        ("sleep_minutes", "sleep min"),
        ("resting_heart_rate_bpm", "resting HR"),
    )
    for key, label in labels:
        item = trends.get(key)
        if not isinstance(item, dict) or not isinstance(
            item.get("average"), (int, float)
        ):
            continue
        suffix = " bpm" if key == "resting_heart_rate_bpm" else ""
        parts.append(f"{label} {item['average']}{suffix}")
    return ", ".join(parts)
