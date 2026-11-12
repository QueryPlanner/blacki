"""Host-only Health query contracts using real SQLite and ADK contexts."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import ToolContext

from blacki.container import AppContainer, reset_container_for_tests, set_container
from blacki.health import query as module
from blacki.health.catalog import HEALTH_STREAMS
from blacki.health.config import GOOGLE_HEALTH_SCOPES
from blacki.health.query import query_archive
from blacki.health.storage import SqliteGoogleHealthStorage
from blacki.tools.health import query_health_data


def context(owner: str = "telegram-chat-42", chat_type: str = "private") -> ToolContext:
    return ToolContext(
        InvocationContext(
            session_service=InMemorySessionService(),
            invocation_id="query-test",
            session=Session(
                id="session",
                app_name="blacki",
                user_id=owner,
                state={"telegram_chat_type": chat_type},
            ),
        )
    )


@dataclass
class Queries:
    storage: SqliteGoogleHealthStorage
    conn: aiosqlite.Connection

    async def connect(
        self, owner: str = "telegram-chat-42", account: str = "account"
    ) -> None:
        await self.storage.upsert_connection(
            telegram_user_id=owner,
            health_user_id=account,
            encrypted_refresh_token="encrypted-test-value",
            legacy_fitbit_user_id=None,
            scopes=GOOGLE_HEALTH_SCOPES,
            archive_consent_version=1,
        )

    async def add(
        self,
        data_type: str,
        data: list[dict[str, Any]],
        owner: str = "telegram-chat-42",
        representation: str | None = None,
    ) -> None:
        c = await self.storage.get_connection(owner)
        assert c is not None
        spec = next(s for s in HEALTH_STREAMS if s.key == data_type)
        previous = (await self.storage.archive.progress(owner)).get(data_type)
        assert await self.storage.archive.commit_page(
            owner=owner,
            account=c.health_user_id,
            generation=c.generation,
            stream=data_type,
            expected=previous,
            checkpoint={"status": "running"},
            records=[
                {
                    "representation": representation or f"{spec.version}:{spec.method}",
                    "data": item,
                }
                for item in data
            ],
        )

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("data_type", "daily-resting-heart-rate")
        if kwargs["data_type"] not in module.CATALOG_TYPES:
            kwargs.setdefault("start_date", "2026-11-04")
            kwargs.setdefault("end_date", "2026-11-04")
        return await query_health_data(context(), **kwargs)


@pytest.fixture
async def queries(tmp_path: Path) -> AsyncIterator[Queries]:
    reset_container_for_tests()
    conn = await aiosqlite.connect(tmp_path / "host.db", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    container = AppContainer(conn=conn)
    await container.google_health_storage.initialize()
    set_container(container)
    q = Queries(container.google_health_storage, conn)
    await q.connect()
    yield q
    reset_container_for_tests()
    await container.close()


def resting(
    value: Any = 60,
    name: str
    | None = "users/account/dataTypes/daily-resting-heart-rate/dataPoints/one",
    day: int = 4,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "dailyRestingHeartRate": {
            "date": {"year": 2026, "month": 11, "day": day},
            "beatsPerMinute": value,
        }
    }
    if name is not None:
        record["name"] = name
    return record


async def test_bot_reads_yesterday_directly_without_daily_summaries(
    queries: Queries,
) -> None:
    await queries.add("daily-resting-heart-rate", [resting("61")])
    result = await queries.run(fields=["dailyRestingHeartRate.beatsPerMinute"])
    assert result["status"] == "success"
    assert result["records"] == [
        {
            "date": "2026-11-04",
            "data": {"dailyRestingHeartRate.beatsPerMinute": "61"},
            "identity": "provider_record",
        }
    ]
    assert result["coverage"]["status"] == "running"
    assert (
        await queries.storage.get_daily_summaries(
            "telegram-chat-42", start_date="2026-11-01", end_date="2026-11-06"
        )
        == []
    )
    assert "account" not in json.dumps(result)


async def test_owner_account_and_latest_version_isolation(queries: Queries) -> None:
    await queries.add("daily-resting-heart-rate", [resting(60)])
    await queries.connect(owner="telegram-chat-99", account="other")
    await queries.add(
        "daily-resting-heart-rate", [resting(999)], owner="telegram-chat-99"
    )
    await queries.add("daily-resting-heart-rate", [resting(62)])
    result = await queries.run()
    assert [r["data"]["beatsPerMinute"] for r in result["records"]] == [62]
    # Replacing the Google account removes the previous account's observations.
    await queries.connect(account="replacement")
    assert (await queries.run())["records"] == []


async def test_reconciled_identity_and_media_filtering(queries: Queries) -> None:
    point: dict[str, Any] = {
        "dataPointName": "users/account/dataTypes/floors/dataPoints/1",
        "floors": {"interval": {"startTime": "2026-11-04T12:00:00Z"}, "count": 1},
    }
    await queries.add("floors", [point])
    point["floors"]["count"] = 2
    await queries.add("floors", [point])
    result = await queries.run(data_type="floors")
    assert len(result["records"]) == 1 and result["records"][0]["data"]["count"] == 2
    await queries.add(
        "exercise", [{"exercise": {"interval": {"startTime": "2026-11-04T12:00:00Z"}}}]
    )
    await queries.add(
        "exercise",
        [{"exercise": "opaque-id", "tcx": "<xml/>"}],
        representation="v4:exercise-tcx",
    )
    assert len((await queries.run(data_type="exercise"))["records"]) == 1


async def test_keyset_pagination_and_query_binding(queries: Queries) -> None:
    await queries.add(
        "daily-resting-heart-rate",
        [resting(60, "first"), resting(61, "second"), resting(62, "third")],
    )
    first = await queries.run(limit=1)
    assert first["has_more"]
    second = await queries.run(limit=1, cursor=first["next_cursor"])
    assert second["records"][0]["data"]["beatsPerMinute"] == 61
    final = await queries.run(limit=1, cursor=second["next_cursor"])
    assert final["records"][0]["data"]["beatsPerMinute"] == 62
    assert final["has_more"] is False and final["next_cursor"] is None
    assert (await queries.run(start_date="2026-11-03", cursor=first["next_cursor"]))[
        "status"
    ] == "error"
    await queries.connect(owner="telegram-chat-99", account="other")
    other = await query_health_data(
        context("telegram-chat-99"),
        data_type="daily-resting-heart-rate",
        start_date="2026-11-04",
        end_date="2026-11-04",
        cursor=first["next_cursor"],
    )
    assert other["status"] == "error"
    await queries.connect()
    assert (await queries.run(cursor=first["next_cursor"]))["status"] == "error"


async def test_ingestion_between_pages_is_a_live_view(queries: Queries) -> None:
    await queries.add(
        "daily-resting-heart-rate", [resting(60, "first"), resting(61, "second")]
    )
    first = await queries.run(limit=1)
    await queries.add("daily-resting-heart-rate", [resting(62, "third")])
    page = await queries.run(cursor=first["next_cursor"])
    assert [r["data"]["beatsPerMinute"] for r in page["records"]] == [61, 62]
    assert "change results between pages" in page["semantics"]


@pytest.mark.parametrize(
    "aggregation,expected",
    [("sum", 123.0), ("avg", 61.5), ("min", 60.0), ("max", 63.0), ("count", 2)],
)
async def test_numeric_aggregations_use_latest_identifiable_inputs(
    queries: Queries, aggregation: str, expected: float
) -> None:
    await queries.add(
        "daily-resting-heart-rate", [resting("59", "one"), resting("63", "two")]
    )
    await queries.add("daily-resting-heart-rate", [resting("60", "one")])
    result = await queries.run(
        operation="aggregate",
        aggregation=aggregation,
        metric="dailyRestingHeartRate.beatsPerMinute",
    )
    assert result["status"] == "success" and result["value"] == expected
    assert result["matched_records"] == 2 and result["numeric_inputs"] == 2


async def test_anonymous_inputs_are_separate_and_require_explicit_basis(
    queries: Queries,
) -> None:
    await queries.add(
        "daily-resting-heart-rate", [resting(60, None), resting(64, None)]
    )
    blocked = await queries.run(
        operation="aggregate",
        aggregation="avg",
        metric="dailyRestingHeartRate.beatsPerMinute",
    )
    assert (
        blocked["status"] == "ambiguous_identity" and blocked["anonymous_inputs"] == 2
    )
    result = await queries.run(
        operation="aggregate",
        aggregation="avg",
        metric="dailyRestingHeartRate.beatsPerMinute",
        aggregation_basis="observations",
    )
    assert result["value"] == 62.0
    count = await queries.run(operation="aggregate")
    assert count["value"] == 2


async def test_nonnumeric_boolean_and_nonfinite_values_are_excluded(
    queries: Queries,
) -> None:
    values = [
        "60",
        "bad",
        "61.5",
        True,
        None,
        "NaN",
        "Infinity",
        "9223372036854775808",
        62,
    ]
    await queries.add(
        "daily-resting-heart-rate", [resting(v, str(i)) for i, v in enumerate(values)]
    )
    result = await queries.run(
        operation="aggregate",
        aggregation="avg",
        metric="dailyRestingHeartRate.beatsPerMinute",
    )
    assert (
        result["value"] == 61.0
        and result["numeric_inputs"] == 2
        and result["matched_records"] == 9
    )
    await queries.add(
        "weight",
        [
            {
                "name": "one",
                "weight": {
                    "sampleTime": {"physicalTime": "2026-11-04T00:00:00Z"},
                    "weightGrams": "70000",
                },
            }
        ],
    )
    invalid = await queries.run(
        data_type="weight",
        operation="aggregate",
        aggregation="avg",
        metric="weight.weightGrams",
    )
    assert invalid["numeric_inputs"] == 0 and invalid["value"] is None


async def test_nonfinite_numbers_and_aggregate_overflow_are_not_returned(
    queries: Queries,
) -> None:
    # Valid JSON exponent overflow can become infinity in SQLite.
    await queries.add(
        "weight",
        [
            {
                "name": "one",
                "weight": {
                    "sampleTime": {"physicalTime": "2026-11-04T00:00:00Z"},
                    "weightGrams": 1e308,
                },
            },
            {
                "name": "two",
                "weight": {
                    "sampleTime": {"physicalTime": "2026-11-04T01:00:00Z"},
                    "weightGrams": 1e308,
                },
            },
        ],
    )
    assert (
        await queries.run(
            data_type="weight",
            operation="aggregate",
            aggregation="sum",
            metric="weight.weightGrams",
        )
    )["status"] == "numeric_overflow"
    await queries.conn.execute(
        "UPDATE google_health_archive SET payload=replace(payload,'1e+308','1e999') "
        "WHERE stream='weight'"
    )
    result = await queries.run(
        data_type="weight",
        operation="aggregate",
        aggregation="avg",
        metric="weight.weightGrams",
    )
    assert result["numeric_inputs"] == 0 and result["value"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_type": "steps' OR 1=1 --"},
        {"limit": 0},
        {"limit": 101},
        {"limit": True},
        {"fields": ["x"] * 9},
        {"fields": ["steps.count); ATTACH DATABASE x"]},
        {"fields": ["x." * 7 + "x"]},
        {"fields": ["x" * 161]},
        {"operation": "execute"},
        {"aggregation": "delete"},
        {"aggregation_basis": "arbitrary"},
        {"operation": "aggregate", "aggregation": "avg"},
        {"operation": "aggregate", "metric": "unapproved"},
        {"data_type": "profile", "operation": "aggregate"},
        {"start_date": "2026-11-01", "end_date": None},
        {"start_date": "bad"},
        {"start_date": "9999-12-31", "end_date": "9999-12-31"},
        {"start_date": "2026-11-05", "end_date": "2026-11-04"},
        {"start_date": "2026-09-01", "end_date": "2026-11-04"},
        {"start_date": "20261104"},
        {"end_date": "20261104"},
        {"cursor": "not-base64"},
        {"cursor": "x" * 1001},
        {"cursor": base64.urlsafe_b64encode(b"{}").decode()},
        {"cursor": base64.urlsafe_b64encode(b"null").decode()},
        {"cursor": "e30=", "operation": "aggregate"},
    ],
)
async def test_invalid_structured_queries_never_execute_sql(
    queries: Queries, kwargs: dict[str, Any]
) -> None:
    result = await queries.run(**kwargs)
    assert result["status"] == "error"
    assert await queries.storage.get_connection("telegram-chat-42") is not None


@pytest.mark.parametrize(
    "data_type,record,expected",
    [
        (
            "heart-rate",
            {
                "heartRate": {
                    "sampleTime": {"physicalTime": "2026-11-05T00:30:00+02:00"},
                    "beatsPerMinute": "61",
                }
            },
            "2026-11-04",
        ),
        (
            "sleep",
            {
                "sleep": {
                    "interval": {
                        "startTime": "2026-11-03T23:00:00Z",
                        "endTime": "2026-11-04T07:00:00Z",
                    },
                    "summary": {"minutesAsleep": "420"},
                }
            },
            "2026-11-04",
        ),
        (
            "steps",
            {
                "steps": {
                    "interval": {
                        "civilStartTime": {"year": 2026, "month": 11, "day": 4}
                    },
                    "count": 12,
                }
            },
            "2026-11-04",
        ),
        (
            "steps",
            {
                "steps": {
                    "interval": {
                        "startTime": "bad",
                        "civilStartTime": {"year": 2026, "month": 11, "day": 4},
                    },
                    "count": 12,
                }
            },
            "2026-11-04",
        ),
        (
            "steps",
            {
                "steps": {
                    "interval": {
                        "startTime": "2026-11-04T12:00:00",
                        "civilStartTime": {"year": 2026, "month": 11, "day": 4},
                    },
                    "count": 12,
                }
            },
            "2026-11-04",
        ),
        ("steps", {"steps": None}, None),
        ("steps", {"steps": {"sampleTime": None, "interval": None}}, None),
        (
            "total-calories",
            {
                "startTime": "2026-11-04T00:00:00Z",
                "endTime": "2026-11-04T00:01:00Z",
                "totalCalories": {"kcalSum": 2.0},
            },
            "2026-11-04",
        ),
    ],
)
async def test_date_anchors_are_explicit_and_end_date_is_inclusive(
    queries: Queries, data_type: str, record: dict[str, Any], expected: str | None
) -> None:
    await queries.add(data_type, [record])
    result = await queries.run(data_type=data_type)
    assert result["status"] == "success"
    assert [r["date"] for r in result["records"]] == ([expected] if expected else [])


async def test_date_filter_uses_latest_version_even_when_date_moves(
    queries: Queries,
) -> None:
    await queries.add("daily-resting-heart-rate", [resting(60, day=4)])
    await queries.add("daily-resting-heart-rate", [resting(61, day=5)])
    assert (await queries.run())["records"] == []


async def test_metadata_ignores_unrelated_dates_and_rejects_date_filters(
    queries: Queries,
) -> None:
    await queries.add(
        "profile",
        [
            {
                "name": "users/account/profile",
                "age": 30,
                "membershipStartDate": {"year": 1900, "month": 1, "day": 1},
            }
        ],
    )
    result = await queries.run(data_type="profile")
    assert result["records"][0]["data"]["age"] == 30
    assert "name" not in result["records"][0]["data"]
    assert (await queries.run(data_type="profile", start_date="2026-11-04"))[
        "status"
    ] == "error"


async def test_record_and_response_byte_limits_support_safe_field_selection(
    queries: Queries,
) -> None:
    await queries.add(
        "electrocardiogram",
        [
            {
                "electrocardiogram": {
                    "interval": {"startTime": "2026-11-04T12:00:00Z"},
                    "waveformSamples": list(range(20000)),
                }
            }
        ],
    )
    assert (await queries.run(data_type="electrocardiogram"))[
        "status"
    ] == "record_too_large"
    selected = await queries.run(
        data_type="electrocardiogram",
        fields=[
            "electrocardiogram.interval.startTime",
            "electrocardiogram.interval.startTime.invalid",
        ],
    )
    assert (
        selected["records"][0]["data"]["electrocardiogram.interval.startTime"]
        == "2026-11-04T12:00:00Z"
    )
    assert (
        selected["records"][0]["data"]["electrocardiogram.interval.startTime.invalid"]
        is None
    )
    await queries.add(
        "daily-resting-heart-rate",
        [
            {**resting(60, "first"), "extra": "x" * 18000},
            {**resting(61, "second"), "extra": "x" * 18000},
        ],
    )
    first = await queries.run(fields=["dailyRestingHeartRate", "extra"])
    assert len(first["records"]) == 1 and first["has_more"]
    assert len(json.dumps(first).encode()) <= 32768
    second = await queries.run(
        fields=["dailyRestingHeartRate", "extra"], cursor=first["next_cursor"]
    )
    assert len(second["records"]) == 1 and second["has_more"] is False


async def test_cursor_rejects_tampered_offsets_and_wrong_types(
    queries: Queries,
) -> None:
    await queries.add(
        "daily-resting-heart-rate", [resting(60, "one"), resting(61, "two")]
    )
    first = await queries.run(limit=1)
    value = json.loads(base64.urlsafe_b64decode(first["next_cursor"]))
    for after in [-1, True, "1", 10**30]:
        value["after"] = after
        cursor = base64.urlsafe_b64encode(json.dumps(value).encode()).decode()
        assert (await queries.run(cursor=cursor))["status"] == "error"


@pytest.mark.parametrize(
    "owner,chat_type", [("", "private"), ("telegram-chat-42", "group")]
)
async def test_real_context_requires_private_user(
    queries: Queries, owner: str, chat_type: str
) -> None:
    result = await query_health_data(context(owner, chat_type), data_type="steps")
    assert result["status"] == "error"


async def test_disconnected_uninitialized_and_invalid_identities_are_explicit(
    queries: Queries,
) -> None:
    assert (await query_archive(queries.storage, "not-private", data_type="steps"))[
        "status"
    ] == "error"
    await queries.storage.mark_reauthorization_required(
        "telegram-chat-42", "invalid_grant"
    )
    assert (await queries.run())["status"] == "reauthorization_required"
    await queries.storage.delete_connection("telegram-chat-42")
    assert (await queries.run())["status"] == "not_connected"
    reset_container_for_tests()
    assert (await query_health_data(context(), data_type="steps"))[
        "error"
    ] == "Health storage is not initialized"


async def test_in_memory_storage_has_no_readonly_file(queries: Queries) -> None:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    storage = SqliteGoogleHealthStorage(conn, asyncio.Lock())
    try:
        await storage.initialize()
        await storage.upsert_connection(
            telegram_user_id="telegram-chat-42",
            health_user_id="account",
            encrypted_refresh_token="opaque",
            legacy_fitbit_user_id=None,
            scopes=(),
            archive_consent_version=1,
        )
        result = await query_archive(storage, "telegram-chat-42", data_type="steps")
        assert result["status"] == "error" and "persistent host" in result["error"]
    finally:
        await storage.close()
        await conn.close()


async def test_query_budget_does_not_interrupt_storage_writes(
    queries: Queries, monkeypatch: pytest.MonkeyPatch
) -> None:
    await queries.add(
        "daily-resting-heart-rate", [resting(60, str(i)) for i in range(1000)]
    )
    clock = iter([0.0, 10.0])
    original = module.monotonic
    monkeypatch.setattr(module, "monotonic", lambda: next(clock))
    result = await queries.run()
    assert result["status"] == "query_unavailable"
    monkeypatch.setattr(module, "monotonic", original)
    await queries.add("daily-resting-heart-rate", [resting(62, "later")])
    assert (await queries.run(limit=1))["status"] == "success"


@pytest.mark.parametrize("event", ["disconnect", "reconnect", "reauthorize"])
async def test_connection_change_during_query_discards_results(
    queries: Queries, monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    await queries.add("daily-resting-heart-rate", [resting()])
    started = threading.Event()
    release = threading.Event()
    original = module.monotonic

    def clock() -> float:
        started.set()
        release.wait(timeout=3)
        return original()

    monkeypatch.setattr(module, "monotonic", clock)
    task = asyncio.create_task(queries.run())
    assert await asyncio.to_thread(started.wait, 3)
    if event == "disconnect":
        await queries.storage.delete_connection("telegram-chat-42")
    elif event == "reconnect":
        await queries.connect()
    else:
        await queries.storage.mark_reauthorization_required(
            "telegram-chat-42", "invalid_grant"
        )
    release.set()
    assert (await task)["status"] == "connection_changed"


async def test_default_dates_and_empty_aggregates_are_useful(queries: Queries) -> None:
    result = await query_health_data(
        context(), data_type="steps", operation="aggregate"
    )
    assert result["status"] == "success" and result["value"] == 0
    assert result["start_date"] and result["end_date"]


async def test_large_unknown_field_names_cannot_expand_error_responses(
    queries: Queries,
) -> None:
    await queries.add(
        "electrocardiogram",
        [
            {
                "electrocardiogram": {
                    "interval": {"startTime": "2026-11-04T12:00:00Z"},
                    "waveformSamples": list(range(20000)),
                },
                "unknown" * 10000: True,
            }
        ],
    )
    result = await queries.run(data_type="electrocardiogram")
    assert result["status"] == "record_too_large"
    assert len(json.dumps(result).encode()) <= 32768
    assert all(len(name) <= 160 for name in result["available_fields"])


async def test_legacy_consent_cannot_query_archival_records(queries: Queries) -> None:
    await queries.add("daily-resting-heart-rate", [resting()])
    await queries.conn.execute(
        "UPDATE google_health_connections SET archive_consent_version=0"
    )
    result = await queries.run()
    assert result["status"] == "archive_consent_required"
    assert "records" not in result
