"""Bounded, resumable full-history ingestion, independent of foreground summaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from blacki.utils.timezone import now_utc

from .catalog import HEALTH_STREAMS, SCOPE_PREFIX, HealthStream
from .client import GoogleHealthApiError, GoogleHealthClient
from .config import TokenEncryptionError
from .normalize import _date_from_value
from .service import GoogleHealthService
from .storage import HealthConnection


class HealthArchiveWorker:
    """Archive at most eight pages per run, fairly across owners and streams.

    Checkpoint compare-and-swap also rejects duplicate responses from another
    process. Completed full scans repeat after 24 hours to observe old edits.
    """

    def __init__(self, service: GoogleHealthService) -> None:
        self.service = service
        self.storage = service.storage.archive
        self.client: GoogleHealthClient = service.client
        self._lock = asyncio.Lock()

    async def run_once(self, *, page_budget: int = 8) -> None:
        async with self._lock:
            candidates = []
            for connection in await self.service.storage.list_active_connections():
                progress = await self.storage.progress(connection.telegram_user_id)
                desired = await self.storage.history_start(
                    connection.telegram_user_id,
                    connection.health_user_id,
                    self.service.config.archive_start_date,
                )
                for spec in HEALTH_STREAMS:
                    previous = progress.get(spec.key)
                    covered = (previous or {}).get(
                        "coveredStart", (previous or {}).get("historyStart")
                    )
                    extending = (
                        spec.method == "rollUp"
                        and previous
                        and previous.get("status") == "complete"
                        and desired
                        and covered
                        and desired < covered
                    )
                    if (
                        previous
                        and previous.get("due", "") > now_utc().isoformat()
                        and not extending
                    ):
                        continue
                    # Profile goes first so aggregate history can use membership date.
                    candidates.append(
                        (
                            previous.get("attempted", "") if previous else "",
                            0 if spec.key == "profile" else 1,
                            connection,
                            spec,
                            previous,
                        )
                    )
            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2].telegram_user_id,
                    item[3].key,
                )
            )
            for _, _, connection, spec, previous in candidates[
                : max(0, min(page_budget, 8))
            ]:
                await self._step(connection, spec, previous)

    async def _step(
        self,
        connection: HealthConnection,
        spec: HealthStream,
        previous: dict[str, Any] | None,
    ) -> None:
        state = dict(previous or {})
        now = now_utc()
        if state.get("status") == "complete":
            covered = state.get("coveredStart", state.get("historyStart"))
            desired = await self.storage.history_start(
                connection.telegram_user_id,
                connection.health_user_id,
                self.service.config.archive_start_date,
            )
            state = {
                key: state[key]
                for key in (
                    "pageSize",
                    "completed",
                    "coveredStart",
                    "earliestObservedDate",
                )
                if key in state
            }
            if spec.method == "rollUp" and desired and covered and desired < covered:
                state["rangeEnd"] = datetime.combine(
                    date.fromisoformat(covered), time.min, UTC
                ).isoformat()

        state.setdefault("scanId", uuid.uuid4().hex)
        state.update(
            attempted=now.isoformat(),
            status="running",
            representation=f"{spec.version}:{spec.method}",
        )
        records: list[dict[str, Any]] = []
        output_stream = spec.key
        try:
            if SCOPE_PREFIX + spec.scope not in connection.scopes:
                state.update(
                    status="permission_required",
                    due=(now + timedelta(hours=24)).isoformat(),
                )
            else:
                token = await self.client.refresh_access_token(
                    self.service.config.cipher.decrypt(
                        connection.encrypted_refresh_token or ""
                    )
                )
                output_stream, records = await self._fetch(
                    connection, spec, state, token.access_token
                )
                state.pop("error", None)
        except TokenEncryptionError:
            state.update(
                status="reauthorization_required",
                due=(now + timedelta(hours=24)).isoformat(),
            )
        except GoogleHealthApiError as exc:
            # Error messages and provider payloads deliberately never reach logs.
            state.update(
                status="permission_required"
                if exc.status_code == 403
                else "reauthorization_required"
                if exc.status_code == 401
                else "failed",
                error=f"http_{exc.status_code}"
                if exc.status_code
                else "provider_error",
                due=(
                    now
                    + timedelta(
                        seconds=max(60, min(exc.retry_after_seconds or 300, 86400))
                    )
                ).isoformat(),
            )
            if exc.error_code == "page_too_large":
                page_size = state.get(
                    "pageSize", 25 if spec.key in {"sleep", "exercise"} else 1000
                )
                if page_size > 1:
                    state["pageSize"] = max(1, page_size // 2)
                    state.pop("pageToken", None)
                    state["scanId"] = uuid.uuid4().hex
                else:
                    state.update(
                        status="record_too_large",
                        due=(now + timedelta(days=1)).isoformat(),
                    )
            if exc.status_code == 400 and state.get("pageToken"):
                # Expired page tokens restart the identical scan idempotently.
                state.pop("pageToken", None)
                state["scanId"] = uuid.uuid4().hex
        await self.storage.commit_page(
            owner=connection.telegram_user_id,
            account=connection.health_user_id,
            generation=connection.generation,
            stream=spec.key,
            expected=previous,
            checkpoint=state,
            records=[
                {"representation": output_stream, "data": record} for record in records
            ],
            observed_start=state.get("earliestObservedDate"),
        )

    async def _fetch(
        self,
        connection: HealthConnection,
        spec: HealthStream,
        state: dict[str, Any],
        access_token: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        # TCX work is bounded to one exercise per turn and shares its checkpoint.
        pending = state.get("mediaPending", [])
        if pending:
            name = pending[0]
            if SCOPE_PREFIX + "location.readonly" not in connection.scopes:
                state["mediaStatus"] = "permission_required"
                state["mediaPending"] = []
            else:
                try:
                    media = await self.client.read_archive_page(
                        access_token,
                        path=f"/v4/{name}:exportExerciseTcx",
                        params={"alt": "media", "partialData": "true"},
                        media=True,
                    )
                except GoogleHealthApiError as exc:
                    if exc.status_code not in {403, 404} and exc.error_code not in {
                        "page_too_large",
                        "invalid_media_type",
                        "invalid_media",
                    }:
                        raise
                    state["mediaStatus"] = "unavailable"
                    media = {"status": "unavailable", "httpStatus": exc.status_code}
                state["mediaPending"] = pending[1:]
                if not state["mediaPending"] and state.pop("jsonFinished", False):
                    self._complete(state)
                return "v4:exercise-tcx", [{"exercise": name, **media}]
            if state.pop("jsonFinished", False):
                self._complete(state)
                return "v4:exercise-tcx", []

        params: dict[str, Any] = {
            "pageSize": state.get(
                "pageSize", 25 if spec.key in {"sleep", "exercise"} else 1000
            )
        }
        if state.get("pageToken"):
            params["pageToken"] = state["pageToken"]
        path = f"/{spec.version}/users/me/dataTypes/{spec.key}/dataPoints"
        method = "GET"
        body = None
        response_key = "dataPoints"
        if spec.method == "resource":
            path = f"/v4/users/me/{spec.key}"
            params = {}
        elif spec.method == "devices":
            path = "/v4/users/me/pairedDevices"
            params["pageSize"] = 100
            response_key = "pairedDevices"
        elif spec.method == "reconcile":
            path += ":reconcile"
        elif spec.method == "rollUp":
            lower = await self.storage.history_start(
                connection.telegram_user_id,
                connection.health_user_id,
                self.service.config.archive_start_date,
            )
            if lower is None or date.fromisoformat(lower) > now_utc().date():
                state.update(
                    status="blocked_membership_date",
                    due=(now_utc() + timedelta(minutes=5)).isoformat(),
                )
                return f"{spec.version}:{spec.method}", []
            state["historyStart"] = min(state.get("historyStart", lower), lower)
            state.setdefault(
                "rangeEnd",
                datetime.combine(
                    now_utc().date() + timedelta(days=1), time.min, UTC
                ).isoformat(),
            )
            end = datetime.fromisoformat(state["rangeEnd"])
            start = max(
                datetime.combine(
                    date.fromisoformat(state["historyStart"]), time.min, UTC
                ),
                end - timedelta(days=1),
            )
            if state.get("pageToken"):
                start = datetime.fromisoformat(state["rangeStart"])
            state["rangeStart"] = start.isoformat()
            path += ":rollUp"
            method = "POST"
            body = {
                **params,
                "range": {"startTime": start.isoformat(), "endTime": end.isoformat()},
                "windowSize": "60s",
            }
            params = {}
            response_key = "rollupDataPoints"
        state["requestSignature"] = hashlib.sha256(
            json.dumps(
                {
                    "path": path,
                    "method": method,
                    "params": {k: v for k, v in params.items() if k != "pageToken"},
                    "body": {k: v for k, v in (body or {}).items() if k != "pageToken"},
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        payload = await self.client.read_archive_page(
            access_token, path=path, method=method, params=params, body=body
        )
        if spec.method == "resource":
            records = [payload]
        else:
            records = payload.get(response_key, [])
            if not isinstance(records, list) or any(
                not isinstance(item, dict) for item in records
            ):
                raise GoogleHealthApiError(
                    "Invalid archive page", error_code="invalid_page"
                )
        next_token = payload.get("nextPageToken", "")
        if not isinstance(next_token, str):
            raise GoogleHealthApiError(
                "Invalid archive cursor", error_code="invalid_page"
            )
        # Persist token hashes so arbitrarily long cycles are detected without
        # accumulating the full token history in memory or imposing a history cap.
        if next_token and await self.storage.cursor_seen(
            connection.telegram_user_id,
            spec.key,
            state["scanId"],
            next_token,
            state["requestSignature"],
        ):
            raise GoogleHealthApiError(
                "Repeated archive cursor", error_code="repeated_cursor"
            )
        if next_token:
            state.update(pageToken=next_token)
        else:
            state.pop("pageToken", None)
            if spec.method == "rollUp":
                state["coveredStart"] = min(
                    state.get("coveredStart", state["rangeStart"][:10]),
                    state["rangeStart"][:10],
                )
            if (
                spec.method == "rollUp"
                and state["rangeStart"][:10] > state["historyStart"]
            ):
                state["rangeEnd"] = state["rangeStart"]
            else:
                self._complete(state)
        if spec.key == "exercise":
            names = []
            for item in records:
                name = item.get("name", "")
                if isinstance(name, str) and re.fullmatch(
                    rf"users/(?:me|{re.escape(connection.health_user_id)})/dataTypes/exercise/dataPoints/[A-Za-z0-9-]+",
                    name,
                ):
                    names.append(name)
            if names:
                state["mediaPending"] = names
                if state["status"] == "complete":
                    state.update(status="running", jsonFinished=True)
                    state.pop("due", None)
        if spec.method in {"list", "reconcile"}:
            observed = earliest_observation_date(spec.key, records)
            if observed:
                state["earliestObservedDate"] = min(
                    state.get("earliestObservedDate", observed), observed
                )
        state["pagesSeen"] = state.get("pagesSeen", 0) + 1
        state["recordsSeen"] = state.get("recordsSeen", 0) + len(records)
        return f"{spec.version}:{spec.method}", records

    @staticmethod
    def _complete(state: dict[str, Any]) -> None:
        state.update(
            status="complete",
            completed=now_utc().isoformat(),
            due=(now_utc() + timedelta(hours=24)).isoformat(),
        )


def earliest_observation_date(
    data_type: str, records: list[dict[str, Any]]
) -> str | None:
    """Inspect only a known data component's documented observation fields."""
    parts = data_type.split("-")
    field = parts[0] + "".join(part.title() for part in parts[1:])
    dates = []
    for record in records:
        component = record.get(field)
        if not isinstance(component, dict):
            continue
        daily = _date_from_value(component.get("date"))
        if daily:
            dates.append(daily)
        for group, physical, civil in (
            ("sampleTime", ("physicalTime",), ("civilTime",)),
            ("interval", ("startTime", "endTime"), ("civilStartTime", "civilEndTime")),
        ):
            times = component.get(group)
            if not isinstance(times, dict):
                continue
            for key in civil:
                day = _date_from_value(times.get(key))
                if day:
                    dates.append(day)
            for key in physical:
                value = times.get(key)
                if not isinstance(value, str):
                    continue
                try:
                    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if instant.tzinfo is not None:
                        dates.append(instant.astimezone(UTC).date().isoformat())
                except ValueError:
                    continue
    return min(dates) if dates else None
