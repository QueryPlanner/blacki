"""Background dispatch and reconciliation of Google Health nutrition exports.

Runs independently of the read-only health sync scheduler, once a minute.
Each tick advances at most one unresolved revision per due meal so ordering
between an earlier delete and a later create is always respected, and every
transient failure is retried with exponential backoff instead of ever being
dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from .client import GoogleHealthApiError, GoogleHealthAuthError, GoogleHealthClient
from .config import (
    GOOGLE_HEALTH_NUTRITION_SCOPES,
    GoogleHealthConfig,
    TokenEncryptionError,
)
from .storage import SqliteGoogleHealthStorage

logger = logging.getLogger(__name__)

_BASE_BACKOFF_SECONDS = 60.0
_MAX_BACKOFF_SECONDS = 3600.0
_POLL_INTERVAL_SECONDS = 60.0
_TOKEN_EXPIRY_SAFETY_MARGIN_SECONDS = 60.0
_TERMINAL_REVISION_STATES = {"synced", "deleted", "failed", "cancelled"}


class NutritionExportWorker:
    """Dispatch pending nutrition export jobs and reconcile ambiguous ones."""

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
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._wake_event = asyncio.Event()
        self._token_cache: dict[str, tuple[str, float]] = {}

    async def start(self) -> None:
        """Start the dispatch loop: every 60s, or immediately on ``wake()``."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(), name="google_health_nutrition_export"
        )
        logger.info("Google Health nutrition export worker started")

    async def stop(self) -> None:
        """Stop the dispatch loop and wait for a running tick to finish."""
        if not self._running:
            return
        self._running = False
        self._wake_event.set()
        if self._task is not None:
            await self._task
            self._task = None
        logger.info("Google Health nutrition export worker stopped")

    async def close(self) -> None:
        """Close the owned Google Health HTTP client."""
        await self.client.close()

    def wake(self) -> None:
        """Trigger an immediate dispatch tick instead of waiting for the timer.

        Safe to call from ``MealService.mutate()`` right after its commit:
        it only sets a flag the single dispatch loop already checks, so it
        cannot race with or duplicate a tick already in progress.
        """
        self._wake_event.set()

    async def _run_loop(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._wake_event.wait(), timeout=_POLL_INTERVAL_SECONDS
                )
            self._wake_event.clear()
            if not self._running:
                return
            await self._dispatch_due()

    async def _dispatch_due(self) -> None:
        """Process one bounded batch of due export jobs, isolating failures."""
        try:
            due = await self.storage.nutrition.due(datetime.now(UTC).timestamp())
        except Exception:
            logger.exception("Failed to load due Google Health export jobs")
            return
        for row in due:
            try:
                await self._process_meal(row)
            except Exception:
                logger.exception("Google Health export job failed unexpectedly")

    async def _process_meal(self, row: dict[str, Any]) -> None:
        meal_id = int(row["meal_id"])
        telegram_user_id = str(row["telegram_user_id"])
        health_user_id = str(row["health_user_id"])
        attempts = int(row["attempts"])
        desired_revision = row["desired_revision"]
        desired_operation = row["desired_operation"]

        nutrition = self.storage.nutrition
        revisions = await nutrition.revisions(meal_id)
        pending, stale = _select_pending_revision(revisions, desired_revision)
        for revision in stale:
            await nutrition.revision_state(int(revision["sequence"]), "cancelled")

        if pending is None:
            if _desired_revision_state(revisions, desired_revision) == "failed":
                return
            await nutrition.result(
                meal_id,
                _final_status(desired_operation),
                expected_revision=desired_revision,
            )
            return

        connection = await self.storage.get_connection(telegram_user_id)
        if (
            connection is None
            or connection.status != "connected"
            or connection.encrypted_refresh_token is None
        ):
            await self.storage.mark_reauthorization_required(telegram_user_id)
            return
        if connection.health_user_id != health_user_id:
            # A stale due-row snapshot can outlive a Google account switch.
            # Do not mark the replacement account as unauthorized; retire only
            # the old revision, guarded by the snapshot's desired resource.
            await nutrition.result(
                meal_id,
                "cancelled",
                error="account_replaced",
                expected_revision=desired_revision,
            )
            return

        if not set(GOOGLE_HEALTH_NUTRITION_SCOPES) <= set(connection.scopes):
            # The connection itself is fine, only nutrition write scope is
            # missing (e.g. the user reconnected with read-only scopes).
            # Pause just this export instead of nuking the whole connection,
            # which would also disable read-only Health summaries.
            await nutrition.result(
                meal_id,
                "authorization_required",
                error="nutrition_scope_missing",
                expected_revision=desired_revision,
            )
            return

        try:
            refresh_token = self.config.cipher.decrypt(
                connection.encrypted_refresh_token
            )
        except TokenEncryptionError:
            await self.storage.mark_reauthorization_required(
                telegram_user_id, "stored_token_invalid"
            )
            return

        try:
            access_token = await self._get_access_token(health_user_id, refresh_token)
        except GoogleHealthAuthError as exc:
            self._token_cache.pop(health_user_id, None)
            await self.storage.mark_reauthorization_required(
                telegram_user_id, exc.error_code or "authorization_required"
            )
            return
        except GoogleHealthApiError as exc:
            await self._backoff(nutrition, meal_id, attempts, desired_revision, exc)
            return

        try:
            if str(pending["operation"]) == "upsert":
                outcome, error = await self._dispatch_upsert(
                    nutrition, access_token, pending
                )
            else:
                outcome, error = await self._dispatch_delete(
                    nutrition, access_token, pending
                )
        except GoogleHealthAuthError as exc:
            self._token_cache.pop(health_user_id, None)
            await self.storage.mark_reauthorization_required(
                telegram_user_id, exc.error_code or "authorization_required"
            )
            return

        if outcome == "resolved":
            await nutrition.record_remote_result(int(pending["sequence"]))
            remaining = await nutrition.revisions(meal_id)
            still_pending, _ = _select_pending_revision(remaining, desired_revision)
            if still_pending is None:
                await nutrition.result(
                    meal_id,
                    _final_status(desired_operation),
                    expected_revision=desired_revision,
                )
            else:
                await nutrition.result(
                    meal_id,
                    "pending",
                    next_attempt=0,
                    expected_revision=desired_revision,
                )
        elif outcome == "failed":
            safe_error = _safe_error_code(error)
            await nutrition.result(
                meal_id,
                "failed",
                error=safe_error,
                expected_revision=desired_revision,
            )
            logger.warning(
                "Google Health nutrition export failed permanently: error_code=%s",
                safe_error,
            )
        else:
            await self._backoff(nutrition, meal_id, attempts, desired_revision, error)

    async def _get_access_token(self, health_user_id: str, refresh_token: str) -> str:
        """Return a cached access token, refreshing only on miss or expiry.

        Cached per Google account so a batch of due meals for the same user
        costs one token-endpoint call instead of one per meal. Never
        persisted: it is cheap to re-derive from the encrypted-at-rest
        refresh token, and a lost cache on restart costs one extra refresh.
        """
        now = datetime.now(UTC).timestamp()
        cached = self._token_cache.get(health_user_id)
        if cached is not None and cached[1] > now:
            return cached[0]

        token = await self.client.refresh_access_token(refresh_token)
        expires_in = token.expires_in if token.expires_in is not None else 0
        expires_at = now + max(0.0, expires_in - _TOKEN_EXPIRY_SAFETY_MARGIN_SECONDS)
        self._token_cache[health_user_id] = (token.access_token, expires_at)
        return token.access_token

    async def _dispatch_upsert(
        self, nutrition: Any, access_token: str, revision: dict[str, Any]
    ) -> tuple[str, Exception | None]:
        sequence = int(revision["sequence"])
        resource_name = str(revision["resource_name"])
        if str(revision["state"]) in {"uncertain", "in_flight"}:
            # A crash between marking "in_flight" and recording the create's
            # outcome leaves that state persisted across restarts, not just
            # set in memory. Reconcile it the same way as "uncertain" instead
            # of blindly re-POSTing a create that may have already landed.
            return await self._verify_upsert(nutrition, access_token, revision)

        payload = json.loads(revision["payload_json"])
        await nutrition.revision_state(sequence, "in_flight")
        try:
            operation = await self.client.create_nutrition_log(
                access_token, resource_name, payload
            )
        except GoogleHealthAuthError:
            raise
        except GoogleHealthApiError as exc:
            if _is_transient(exc):
                await nutrition.revision_state(sequence, "uncertain")
                return "retry", exc
            await nutrition.revision_state(sequence, "failed")
            return "failed", exc

        if operation.successful:
            await nutrition.revision_state(sequence, "synced")
            return "resolved", None
        if not operation.done:
            await nutrition.revision_state(sequence, "uncertain")
            return "retry", None
        await nutrition.revision_state(sequence, "failed")
        return "failed", _ProviderError(operation.error_code)

    async def _verify_upsert(
        self, nutrition: Any, access_token: str, revision: dict[str, Any]
    ) -> tuple[str, Exception | None]:
        sequence = int(revision["sequence"])
        resource_name = str(revision["resource_name"])
        try:
            point = await self.client.get_data_point(access_token, resource_name)
        except GoogleHealthAuthError:
            raise
        except GoogleHealthApiError as exc:
            if exc.status_code == 404:
                await nutrition.revision_state(sequence, "queued")
                return "retry", None
            if _is_transient(exc):
                return "retry", exc
            await nutrition.revision_state(sequence, "failed")
            return "failed", exc

        payload = json.loads(revision["payload_json"])
        if _nutrition_log_matches(point, payload):
            await nutrition.revision_state(sequence, "synced")
            return "resolved", None
        await nutrition.revision_state(sequence, "failed")
        return "failed", _ProviderError("verification_mismatch")

    async def _dispatch_delete(
        self, nutrition: Any, access_token: str, revision: dict[str, Any]
    ) -> tuple[str, Exception | None]:
        sequence = int(revision["sequence"])
        resource_name = str(revision["resource_name"])
        if str(revision["state"]) in {"uncertain", "in_flight"}:
            # Same restart hazard as the upsert path: a persisted "in_flight"
            # delete must be reconciled with a GET before retrying, since the
            # delete itself may have already succeeded before the crash.
            try:
                await self.client.get_data_point(access_token, resource_name)
            except GoogleHealthAuthError:
                raise
            except GoogleHealthApiError as exc:
                if exc.status_code == 404:
                    await nutrition.revision_state(sequence, "deleted")
                    return "resolved", None
                if _is_transient(exc):
                    return "retry", exc
                await nutrition.revision_state(sequence, "failed")
                return "failed", exc
            await nutrition.revision_state(sequence, "queued")
            return "retry", None

        await nutrition.revision_state(sequence, "in_flight")
        try:
            operation = await self.client.delete_nutrition_log(
                access_token, resource_name
            )
        except GoogleHealthAuthError:
            raise
        except GoogleHealthApiError as exc:
            if exc.status_code == 404:
                await nutrition.revision_state(sequence, "deleted")
                return "resolved", None
            if _is_transient(exc):
                await nutrition.revision_state(sequence, "uncertain")
                return "retry", exc
            await nutrition.revision_state(sequence, "failed")
            return "failed", exc

        if operation.successful:
            await nutrition.revision_state(sequence, "deleted")
            return "resolved", None
        if not operation.done:
            await nutrition.revision_state(sequence, "uncertain")
            return "retry", None
        await nutrition.revision_state(sequence, "failed")
        return "failed", _ProviderError(operation.error_code)

    async def _backoff(
        self,
        nutrition: Any,
        meal_id: int,
        attempts: int,
        expected_revision: str | None,
        error: Exception | None,
    ) -> None:
        delay = min(_BASE_BACKOFF_SECONDS * (2**attempts), _MAX_BACKOFF_SECONDS)
        retry_after = getattr(error, "retry_after_seconds", None)
        if isinstance(retry_after, int | float):
            delay = max(delay, min(float(retry_after), _MAX_BACKOFF_SECONDS))
        next_attempt = datetime.now(UTC).timestamp() + delay
        await nutrition.result(
            meal_id,
            "pending",
            error=_safe_error_code(error),
            next_attempt=next_attempt,
            expected_revision=expected_revision,
        )


class _ProviderError(Exception):
    """Wrap a safe Google Health operation error code for uniform handling."""

    def __init__(self, error_code: str | None) -> None:
        super().__init__(error_code or "provider_error")
        self.error_code = error_code or "provider_error"


def _final_status(desired_operation: Any) -> str:
    return "deleted" if str(desired_operation) == "delete" else "synced"


def _select_pending_revision(
    revisions: list[dict[str, Any]], desired_revision: str | None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Pick the oldest unresolved revision, cancelling never-dispatched creates.

    An ``upsert`` revision whose resource name no longer matches the meal's
    current desired target was superseded by a later edit or deletion. If it
    was never sent to Google (``queued``), it is safe to drop outright — but
    an ``in_flight``/``uncertain`` one must be reconciled first, regardless of
    its position, since Google may already be processing it.

    A ``delete`` revision is never treated as stale by this check: its
    resource name is always the *old* point being removed, not the meal's
    current desired target, so it would otherwise always look superseded.
    Skipping a real, still-queued delete would leave the old point on Google
    forever once its replacement create lands.
    """
    non_terminal = [
        r for r in revisions if str(r["state"]) not in _TERMINAL_REVISION_STATES
    ]
    stale: list[dict[str, Any]] = []
    for revision in non_terminal:
        if (
            str(revision["operation"]) == "upsert"
            and str(revision["resource_name"]) != desired_revision
            and str(revision["state"]) == "queued"
        ):
            stale.append(revision)
            continue
        return revision, stale
    return None, stale


def _desired_revision_state(
    revisions: list[dict[str, Any]], desired_revision: str | None
) -> str | None:
    """Return the latest state for the parent's current desired resource."""
    matching = [
        revision
        for revision in revisions
        if revision.get("resource_name") == desired_revision
    ]
    if not matching:
        return None
    return str(matching[-1]["state"])


def _is_transient(exc: GoogleHealthApiError) -> bool:
    if exc.transport or exc.status_code is None:
        return True
    return exc.status_code == 429 or exc.status_code >= 500


def _safe_error_code(error: Exception | None) -> str | None:
    if error is None:
        return None
    code = getattr(error, "error_code", None)
    if isinstance(code, str) and code.isascii() and code.isprintable():
        return code[:80]
    return type(error).__name__


def _nutrition_log_matches(point: dict[str, Any], payload: dict[str, Any]) -> bool:
    remote = point.get("nutritionLog")
    intended = payload.get("nutritionLog")
    if not isinstance(remote, dict) or not isinstance(intended, dict):
        return False
    if remote.get("foodDisplayName") != intended.get("foodDisplayName"):
        return False
    remote_energy = remote.get("energy") or {}
    intended_energy = intended.get("energy") or {}
    return remote_energy.get("kcal") == intended_energy.get("kcal")
