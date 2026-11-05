"""Bounded structured queries over one private account's host-side archive.

All SQL is generated here. Model inputs are data, never SQL or executable code.
Each request uses its own read-only connection and execution budget.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any

from blacki.utils.timezone import now_utc

from .catalog import HEALTH_STREAMS, HealthStream
from .config import (
    GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION,
    health_user_id_for_telegram_user,
)
from .normalize import _date_from_value
from .storage import SqliteGoogleHealthStorage

MAX_BYTES = 32768
CATALOG_TYPES = {
    "food",
    "food-measurement-unit",
    "profile",
    "settings",
    "irnProfile",
    "pairedDevices",
}
# Field names and int64 string encodings verified against Google's v4 discovery.
METRICS: dict[str, tuple[str, ...]] = {
    "steps": ("steps.count",),
    "distance": ("distance.millimeters",),
    "active-energy-burned": ("activeEnergyBurned.kcal",),
    "daily-resting-heart-rate": ("dailyRestingHeartRate.beatsPerMinute",),
    "heart-rate": ("heartRate.beatsPerMinute",),
    "daily-heart-rate-variability": (
        "dailyHeartRateVariability.averageHeartRateVariabilityMilliseconds",
        "dailyHeartRateVariability.entropy",
    ),
    "heart-rate-variability": (
        "heartRateVariability.rootMeanSquareOfSuccessiveDifferencesMilliseconds",
        "heartRateVariability.standardDeviationMilliseconds",
    ),
    "daily-oxygen-saturation": (
        "dailyOxygenSaturation.averagePercentage",
        "dailyOxygenSaturation.lowerBoundPercentage",
        "dailyOxygenSaturation.upperBoundPercentage",
    ),
    "daily-respiratory-rate": ("dailyRespiratoryRate.breathsPerMinute",),
    "sleep": ("sleep.summary.minutesAsleep", "sleep.summary.minutesAwake"),
    "weight": ("weight.weightGrams",),
    "body-fat": ("bodyFat.percentage",),
    "total-calories": ("totalCalories.kcalSum",),
}
INT64_FIELDS = {
    "steps.count",
    "distance.millimeters",
    "dailyRestingHeartRate.beatsPerMinute",
    "heartRate.beatsPerMinute",
    "sleep.summary.minutesAsleep",
    "sleep.summary.minutesAwake",
}


def _component(data_type: str) -> str:
    parts = data_type.split("-")
    return parts[0] + "".join(part.title() for part in parts[1:])


def _day(payload: str, spec: HealthStream) -> str | None:
    record = json.loads(payload)["data"]
    if spec.method == "rollUp":
        physical = record.get("startTime")
        civil = None
    else:
        component = record.get(_component(spec.key), {})
        if not isinstance(component, dict):
            return None
        daily = _date_from_value(component.get("date"))
        if daily:
            return daily
        sample = component.get("sampleTime", {})
        interval = component.get("interval", {})
        sample = sample if isinstance(sample, dict) else {}
        interval = interval if isinstance(interval, dict) else {}
        physical = sample.get("physicalTime") or interval.get(
            "endTime" if spec.key == "sleep" else "startTime"
        )
        civil = sample.get("civilTime") or interval.get(
            "civilEndTime" if spec.key == "sleep" else "civilStartTime"
        )
    if isinstance(physical, str):
        try:
            instant = datetime.fromisoformat(physical.replace("Z", "+00:00"))
            if instant.tzinfo is not None:
                return instant.astimezone(UTC).date().isoformat()
        except ValueError:
            pass
    return _date_from_value(civil)


def _number(value: Any, kind: str | None, allow_string: int) -> float | None:
    if kind not in {"integer", "real"} and not (allow_string and kind == "text"):
        return None
    if kind == "text":
        if not re.fullmatch(r"-?[0-9]{1,19}", value):
            return None
        integer = int(value)
        if not -(2**63) <= integer <= 2**63 - 1:
            return None
        value = integer
    # SQLite json_type guarantees numeric values here, or the validated int64.
    result = float(value)
    return result if math.isfinite(result) else None


def _field(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _read(
    path: str,
    owner: str,
    account: str,
    spec: HealthStream,
    start: str | None,
    end: str | None,
    fields: list[str],
    limit: int,
    after: int,
    fingerprint: str,
    operation: str,
    aggregation: str,
    metric: str | None,
    basis: str,
) -> dict[str, Any]:
    deadline = monotonic() + 2.0
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    ) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(lambda: int(monotonic() > deadline), 1000)
        conn.create_function("health_day", 1, lambda payload: _day(payload, spec))
        conn.create_function("health_number", 3, _number)
        # Owner/account/type/representation filtering precedes version selection.
        sql = """
            WITH scoped AS (
                SELECT rowid AS rid, payload, digest, last_seen,
                    coalesce(nullif(json_extract(payload, '$.data.name'), ''),
                             nullif(json_extract(payload, '$.data.dataPointName'), '')
                    ) AS provider_id
                FROM google_health_archive
                WHERE owner=? AND account=? AND stream=?
                  AND json_extract(payload, '$.representation')=?
            ), ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY coalesce(provider_id, digest)
                    ORDER BY last_seen DESC, rid DESC
                ) AS version FROM scoped
            ), matched AS (
                SELECT *, health_day(payload) AS day FROM ranked WHERE version=1
            )
        """
        params: list[Any] = [owner, account, spec.key, f"{spec.version}:{spec.method}"]
        condition = " WHERE (? IS NULL OR (day>=? AND day<?))"
        date_params = [start, start, end]
        if operation == "aggregate":
            field_path = "$.data." + metric if metric else "$.data"
            query = "".join(
                (
                    sql,
                    """
                , numbers AS (
                    SELECT provider_id, health_number(json_extract(payload, ?),
                        json_type(payload, ?), ?) AS value FROM matched
            """,
                    condition,
                    """
                ) SELECT count(*) AS matched, count(value) AS numeric_inputs,
                    coalesce(sum(provider_id IS NULL), 0) AS anonymous,
                    sum(value) AS sum_value, avg(value) AS avg_value,
                    min(value) AS min_value, max(value) AS max_value,
                    count(value) AS count_value FROM numbers
            """,
                )
            )
            row = conn.execute(
                query,
                [
                    *params,
                    field_path,
                    field_path,
                    int(metric in INT64_FIELDS),
                    *date_params,
                ],
            ).fetchone()
            result = dict(row)
            if metric is None:
                result["value"] = result["matched"]
            if (
                result["anonymous"]
                and basis == "latest_records"
                and aggregation != "count"
            ):
                return {
                    "status": "ambiguous_identity",
                    "matched_records": result["matched"],
                    "anonymous_inputs": result["anonymous"],
                    "message": (
                        "Some inputs lack stable provider IDs. Use "
                        "aggregation_basis=observations only to explicitly aggregate "
                        "archived observations, which can overlap sources or versions."
                    ),
                }
            value = result.get("value", result[aggregation + "_value"])
            if value is not None and not math.isfinite(value):
                return {
                    "status": "numeric_overflow",
                    "message": "The aggregate exceeds finite numeric range.",
                }
            return {
                "status": "success",
                "value": value,
                "matched_records": result["matched"],
                "numeric_inputs": result["numeric_inputs"],
                "anonymous_inputs": result["anonymous"],
            }
        query = "".join(
            (
                sql,
                " SELECT rid,payload,provider_id,day FROM matched",
                condition,
                " AND rid>? ORDER BY rid LIMIT ?",
            )
        )
        cursor_rows = conn.execute(query, [*params, *date_params, after, limit + 1])
        records: list[dict[str, Any]] = []
        used = 0
        has_more = False
        last_id = after
        for row in cursor_rows:
            if len(records) == limit:
                has_more = True
                break
            data = json.loads(row["payload"])["data"]
            selected = (
                {field: _field(data, field) for field in fields}
                if fields
                else data.get(
                    _component(spec.key),
                    {
                        k: v
                        for k, v in data.items()
                        if k not in {"name", "dataPointName"}
                    },
                )
            )
            item = {
                "date": row["day"],
                "data": selected,
                "identity": "provider_record"
                if row["provider_id"]
                else "anonymous_observation",
            }
            size = len(json.dumps(item, allow_nan=False).encode())
            if used + size > MAX_BYTES - 2048:
                if not records:
                    return {
                        "status": "record_too_large",
                        "message": (
                            "Select scalar fields to keep results within 32 KiB."
                        ),
                        "available_fields": [name for name in data if len(name) <= 160][
                            :20
                        ],
                    }
                has_more = True
                break
            records.append(item)
            last_id = row["rid"]
            used += size
        next_cursor = None
        if has_more:
            next_cursor = base64.urlsafe_b64encode(
                json.dumps({"query": fingerprint, "after": last_id}).encode()
            ).decode()
        return {
            "status": "success",
            "records": records,
            "has_more": has_more,
            "next_cursor": next_cursor,
        }


async def query_archive(
    storage: SqliteGoogleHealthStorage,
    user_id: str,
    *,
    data_type: str,
    start_date: str | None = None,
    end_date: str | None = None,
    fields: list[str] | None = None,
    limit: int = 50,
    cursor: str | None = None,
    operation: str = "records",
    aggregation: str = "count",
    metric: str | None = None,
    aggregation_basis: str = "latest_records",
) -> dict[str, Any]:
    """Validate structured inputs and enforce account identity around a local query."""
    owner = health_user_id_for_telegram_user(user_id)
    if owner is None:
        return {"status": "error", "error": "Private Telegram identity required"}
    connection = await storage.get_connection(owner)
    if connection is None:
        return {"status": "not_connected"}
    if connection.status != "connected":
        return {"status": connection.status}
    if connection.archive_consent_version < GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION:
        return {
            "status": "archive_consent_required",
            "error": (
                "Reconnect Google Health to accept archival retention "
                "before querying the archive"
            ),
        }
    spec = next((item for item in HEALTH_STREAMS if item.key == data_type), None)
    if spec is None:
        return {
            "status": "error",
            "error": "Unknown readable data type",
            "available_types": [item.key for item in HEALTH_STREAMS],
        }
    fields = fields or []
    if (
        type(limit) is not int
        or not 1 <= limit <= 100
        or not isinstance(fields, list)
        or len(fields) > 8
        or any(
            not isinstance(field, str)
            or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*){0,5}", field
            )
            or len(field) > 160
            for field in fields
        )
    ):
        return {
            "status": "error",
            "error": "Use a limit of 1-100 and at most eight simple JSON field paths",
        }
    if (
        operation not in {"records", "aggregate"}
        or aggregation not in {"count", "sum", "avg", "min", "max"}
        or aggregation_basis not in {"latest_records", "observations"}
    ):
        return {
            "status": "error",
            "error": "Unsupported query operation, aggregation, or basis",
        }
    if operation == "aggregate" and (
        data_type in CATALOG_TYPES
        or (aggregation != "count" and metric is None)
        or (metric is not None and metric not in METRICS.get(data_type, ()))
    ):
        return {
            "status": "error",
            "error": "Select a supported numeric metric for this time-series type",
            "supported_metrics": list(METRICS.get(data_type, ())),
        }
    start = end = None
    if data_type in CATALOG_TYPES:
        if start_date or end_date:
            return {
                "status": "error",
                "error": "Catalog and profile records have no observation-date range",
            }
    else:
        if bool(start_date) != bool(end_date):
            return {"status": "error", "error": "Supply both start_date and end_date"}
        try:
            today = now_utc().date()
            first = (
                date.fromisoformat(start_date)
                if start_date
                else today - timedelta(days=6)
            )
            last = date.fromisoformat(end_date) if end_date else today
            if (
                first > last
                or (last - first).days > 30
                or (start_date and first.isoformat() != start_date)
                or (end_date and last.isoformat() != end_date)
            ):
                raise ValueError("invalid range")
            start, end = first.isoformat(), (last + timedelta(days=1)).isoformat()
        except (ValueError, OverflowError):
            return {
                "status": "error",
                "error": "Use inclusive YYYY-MM-DD dates spanning at most 31 days",
            }
    fingerprint = hashlib.sha256(
        json.dumps(
            [
                owner,
                connection.health_user_id,
                connection.generation,
                data_type,
                start,
                end,
                fields,
                operation,
                aggregation,
                metric,
                aggregation_basis,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    after = 0
    if cursor:
        try:
            if len(cursor) > 1000 or operation != "records":
                raise ValueError("invalid cursor")
            value = json.loads(base64.urlsafe_b64decode(cursor.encode()))
            after = value["after"]
            if (
                value["query"] != fingerprint
                or type(after) is not int
                or not 0 <= after <= 9223372036854775807
            ):
                raise ValueError("invalid cursor")
        except (ValueError, TypeError, KeyError, UnicodeError):
            return {
                "status": "error",
                "error": "Cursor does not match this query and connection",
            }
    databases = await storage.archive.database_path()
    if not databases:
        return {
            "status": "error",
            "error": "Queries require a persistent host database",
        }
    try:
        result = await asyncio.to_thread(
            _read,
            databases,
            owner,
            connection.health_user_id,
            spec,
            start,
            end,
            fields,
            limit,
            after,
            fingerprint,
            operation,
            aggregation,
            metric,
            aggregation_basis,
        )
    except sqlite3.OperationalError:
        return {
            "status": "query_unavailable",
            "error": (
                "Query hit its two-second budget, storage was busy, or a record "
                "was malformed. Retry when ingestion is quieter. Large archives "
                "can hit this budget even with a narrow date range."
            ),
        }
    current = await storage.get_connection(owner)
    if (
        current is None
        or current.status != "connected"
        or current.generation != connection.generation
    ):
        return {
            "status": "connection_changed",
            "error": "The Health connection changed during the query",
        }
    result.update(
        data_type=data_type,
        representation=f"{spec.version}:{spec.method}",
        start_date=start,
        end_date=end_date or (now_utc().date().isoformat() if start else None),
        date_convention=(
            "Inclusive requested dates; physical timestamps use UTC. "
            "Daily/civil-only records use provider calendar dates. "
            "Sleep anchors at end; other intervals at start."
        ),
        semantics=(
            "Latest observed version per provider ID; anonymous payloads "
            "remain separate observations. Provider deletion does not erase "
            "observations. Ingestion may change results between pages."
        ),
        coverage=(await storage.archive.coverage(owner))["streams"][data_type],
    )
    return result
