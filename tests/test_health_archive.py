"""Archive contracts through real HTTP clients, workers, and file-backed SQLite."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
import pytest
from cryptography.fernet import Fernet

from blacki.health import archive_worker
from blacki.health.archive_worker import HealthArchiveWorker
from blacki.health.catalog import SCOPE_PREFIX
from blacki.health.client import GoogleHealthClient
from blacki.health.config import GOOGLE_HEALTH_SCOPES, GoogleHealthConfig
from blacki.health.service import GoogleHealthService
from blacki.health.storage import SqliteGoogleHealthStorage


@dataclass
class ArchiveHarness:
    storage: SqliteGoogleHealthStorage
    conn: aiosqlite.Connection
    worker: HealthArchiveWorker
    config: GoogleHealthConfig
    responses: dict[str, list[Any]]
    requests: list[httpx.Request]
    clock: list[datetime]

    async def connect(
        self,
        owner: str = "telegram-chat-42",
        account: str = "account",
        scopes: tuple[str, ...] = GOOGLE_HEALTH_SCOPES,
    ) -> None:
        await self.storage.upsert_connection(
            telegram_user_id=owner,
            health_user_id=account,
            encrypted_refresh_token=self.config.cipher.encrypt("refresh"),
            legacy_fitbit_user_id=None,
            scopes=scopes,
        )

    async def records(self, owner: str = "telegram-chat-42") -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT payload FROM google_health_archive WHERE owner = ?", (owner,)
        ) as cursor:
            return [json.loads(row[0]) for row in await cursor.fetchall()]

    async def run(self, turns: int = 1) -> None:
        for _ in range(turns):
            await self.worker.run_once()
            self.clock[0] += timedelta(seconds=1)


@pytest.fixture
async def archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[ArchiveHarness]:
    clock = [datetime(2026, 11, 5, tzinfo=UTC)]
    monkeypatch.setattr(archive_worker, "now_utc", lambda: clock[0])
    responses: dict[str, list[Any]] = {}
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/token":
            return httpx.Response(
                200, json={"access_token": "access", "expires_in": 3600}
            )
        queue = responses.get(request.url.path, [])
        if queue:
            value = queue.pop(0)
            if callable(value):
                value = await value()
            return (
                value
                if isinstance(value, httpx.Response)
                else httpx.Response(200, json=value)
            )
        if request.url.path.endswith("/profile"):
            return httpx.Response(
                200,
                json={
                    "membershipStartDate": {"year": 2026, "month": 11, "day": 3},
                    "unknown": {"retain": True},
                },
            )
        return httpx.Response(200, json={})

    config = GoogleHealthConfig(
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.test/callback",
        token_encryption_key=Fernet.generate_key().decode(),
    )
    conn = await aiosqlite.connect(tmp_path / "archive.db", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    storage = SqliteGoogleHealthStorage(conn, asyncio.Lock())
    await storage.initialize()
    client = GoogleHealthClient(
        config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    service = GoogleHealthService(config, storage, client=client)
    harness = ArchiveHarness(
        storage, conn, HealthArchiveWorker(service), config, responses, requests, clock
    )
    await harness.connect()
    yield harness
    await service.close()
    await storage.close()
    await conn.close()


async def test_full_history_pages_and_unknown_fields_survive_restart(
    archive: ArchiveHarness,
) -> None:
    path = "/v4/users/me/dataTypes/heart-rate/dataPoints"
    archive.responses[path] = [
        {
            "dataPoints": [
                {
                    "name": "old",
                    "heartRate": {
                        "sampleTime": {"physicalTime": "2010-01-01T00:00:00Z"},
                        "beatsPerMinute": 61,
                    },
                    "futureField": [1, {"x": True}],
                }
            ],
            "nextPageToken": "older",
        },
        {"dataPoints": [{"name": "older", "heartRate": {"beatsPerMinute": 62}}]},
    ]
    await archive.run(6)
    # Reinitialize the schema and worker to prove durable restart behavior.
    await archive.storage.close()
    await archive.storage.initialize()
    archive.worker = HealthArchiveWorker(archive.worker.service)
    await archive.run(6)
    calls = [r for r in archive.requests if r.url.path == path]
    assert len(calls) == 2
    assert "filter" not in calls[0].url.params
    assert calls[1].url.params["pageToken"] == "older"
    stored = [x["data"] for x in await archive.records() if "heartRate" in x["data"]]
    assert len(stored) == 2
    assert next(item for item in stored if item["name"] == "old")["futureField"] == [
        1,
        {"x": True},
    ]
    coverage = await archive.storage.archive.coverage("telegram-chat-42")
    assert coverage["streams"]["heart-rate"]["status"] == "complete"
    assert "older" not in json.dumps(coverage)


async def test_correct_representations_and_finite_minute_rollups(
    archive: ArchiveHarness,
) -> None:
    await archive.run(15)
    paths = [r.url.path for r in archive.requests]
    assert "/v4/users/me/dataTypes/floors/dataPoints:reconcile" in paths
    assert "/v4beta/users/me/dataTypes/skin-temperature-sensors/dataPoints" in paths
    assert "/v4/users/me/settings" in paths
    assert "/v4/users/me/irnProfile" in paths
    assert "/v4/users/me/pairedDevices" in paths
    rollups = [
        r
        for r in archive.requests
        if r.url.path.endswith("/total-calories/dataPoints:rollUp")
    ]
    assert len(rollups) == 3
    bodies = [json.loads(r.content) for r in rollups]
    assert all(r.method == "POST" for r in rollups)
    assert bodies[0]["windowSize"] == "60s"
    assert bodies[-1]["range"]["startTime"] == "2026-11-03T00:00:00+00:00"
    assert all(
        datetime.fromisoformat(b["range"]["endTime"])
        - datetime.fromisoformat(b["range"]["startTime"])
        == timedelta(days=1)
        for b in bodies
    )
    coverage = await archive.storage.archive.coverage("telegram-chat-42")
    assert coverage["complete"] is True
    assert coverage["unreadable_types"] == [
        "menstrual-period",
        "moods",
        "ovulation-test",
        "symptoms",
    ]
    before = len(archive.requests)
    await archive.run()
    assert len(archive.requests) == before


@pytest.mark.parametrize("event", ["disconnect", "replace", "reconsent"])
async def test_inflight_page_cannot_resurrect_old_connection(
    archive: ArchiveHarness, event: str
) -> None:
    path = "/v4/users/me/dataTypes/active-energy-burned/dataPoints"

    async def invalidate() -> dict[str, Any]:
        if event == "disconnect":
            await archive.storage.delete_connection("telegram-chat-42")
        else:
            await archive.connect(
                account="replacement" if event == "replace" else "account"
            )
        return {"dataPoints": [{"private": "stale"}]}

    archive.responses[path] = [invalidate]
    await archive.run()
    assert all(x["data"] != {"private": "stale"} for x in await archive.records())
    if event == "disconnect":
        assert await archive.records() == []
        assert await archive.storage.archive.progress("telegram-chat-42") == {}


async def test_owner_isolation_and_disconnect_cleanup(archive: ArchiveHarness) -> None:
    await archive.connect(owner="telegram-chat-99", account="other")
    await archive.run(12)
    assert await archive.records("telegram-chat-99")
    await archive.storage.delete_connection("telegram-chat-42")
    assert await archive.records() == []
    assert await archive.storage.archive.progress("telegram-chat-42") == {}
    assert await archive.records("telegram-chat-99")


@pytest.mark.parametrize(
    "response,status",
    [
        (
            httpx.Response(403, json={"error": {"message": "sensitive"}}),
            "permission_required",
        ),
        (
            httpx.Response(401, json={"error": {"message": "sensitive"}}),
            "reauthorization_required",
        ),
        (
            httpx.Response(429, headers={"Retry-After": "600"}, json={"error": {}}),
            "failed",
        ),
        (httpx.Response(500, json={"error": {}}), "failed"),
        ({"dataPoints": "bad"}, "failed"),
        ({"dataPoints": [42]}, "failed"),
        ({"nextPageToken": 42}, "failed"),
        (httpx.Response(200, content="not json"), "failed"),
    ],
)
async def test_failures_are_visible_private_and_do_not_block_other_types(
    archive: ArchiveHarness, response: Any, status: str
) -> None:
    archive.responses["/v4/users/me/dataTypes/active-energy-burned/dataPoints"] = [
        response
    ]
    await archive.run(6)
    progress = await archive.storage.archive.progress("telegram-chat-42")
    assert progress["active-energy-burned"]["status"] == status
    assert progress["steps"]["status"] == "complete"
    assert "sensitive" not in json.dumps(progress)
    if isinstance(response, httpx.Response) and response.status_code == 429:
        assert progress["active-energy-burned"]["due"] == "2026-11-05T00:10:00+00:00"


async def test_failed_last_page_and_repeated_cursor_never_claim_completion(
    archive: ArchiveHarness,
) -> None:
    path = "/v4/users/me/dataTypes/active-energy-burned/dataPoints"
    archive.responses[path] = [
        {"dataPoints": [{"keep": True}], "nextPageToken": "again"},
        {"dataPoints": [{"reject": True}], "nextPageToken": "again"},
    ]
    await archive.run(12)
    progress = await archive.storage.archive.progress("telegram-chat-42")
    assert progress["active-energy-burned"]["status"] == "failed"
    records = await archive.records()
    assert any(x["data"] == {"keep": True} for x in records)
    assert all(x["data"] != {"reject": True} for x in records)


async def test_missing_permission_and_membership_date_are_explicit(
    archive: ArchiveHarness,
) -> None:
    await archive.connect(scopes=(SCOPE_PREFIX + "activity_and_fitness.readonly",))
    await archive.run(6)
    progress = await archive.storage.archive.progress("telegram-chat-42")
    assert progress["profile"]["status"] == "permission_required"
    assert progress["total-calories"]["status"] == "blocked_membership_date"
    assert not any(r.url.path.endswith("/profile") for r in archive.requests)
    assert not (await archive.storage.archive.coverage("telegram-chat-42"))["complete"]


async def test_rescan_preserves_versions_and_deduplicates_replay(
    archive: ArchiveHarness,
) -> None:
    path = "/v4/users/me/dataTypes/active-energy-burned/dataPoints"
    archive.responses[path] = [{"dataPoints": [{"name": "same", "value": 1}]}]
    await archive.run(12)
    for value in [1, 2]:
        archive.clock[0] += timedelta(days=2)
        archive.responses[path] = [{"dataPoints": [{"name": "same", "value": value}]}]
        await archive.run(12)
    records = [x for x in await archive.records() if x["data"].get("name") == "same"]
    assert len(records) == 2


async def test_tcx_is_archived_separately_and_never_executes(
    archive: ArchiveHarness,
) -> None:
    archive.responses["/v4/users/me/dataTypes/exercise/dataPoints"] = [
        {
            "dataPoints": [
                {
                    "name": "users/account/dataTypes/exercise/dataPoints/123",
                    "exercise": {"unknown": True},
                }
            ]
        }
    ]
    archive.responses[
        "/v4/users/account/dataTypes/exercise/dataPoints/123:exportExerciseTcx"
    ] = [
        httpx.Response(
            200,
            headers={"content-type": "application/xml"},
            content="<TrainingCenterDatabase />",
        )
    ]
    await archive.run(12)
    records = await archive.records()
    assert any(
        x["representation"] == "v4:exercise-tcx"
        and x["data"]["tcx"] == "<TrainingCenterDatabase />"
        for x in records
    )
    assert any(x["data"].get("exercise") == {"unknown": True} for x in records)
    media_request = next(
        r for r in archive.requests if r.url.path.endswith(":exportExerciseTcx")
    )
    assert media_request.url.params["alt"] == "media"


async def test_long_history_has_no_lifetime_page_cap(archive: ArchiveHarness) -> None:
    path = "/v4/users/me/dataTypes/active-energy-burned/dataPoints"
    archive.responses[path] = [{"dataPoints": [{"page": 1}], "nextPageToken": "second"}]
    await archive.run()
    state = (await archive.storage.archive.progress("telegram-chat-42"))[
        "active-energy-burned"
    ]
    state["pagesSeen"] = 100000
    await archive.conn.execute(
        "UPDATE google_health_imports SET checkpoint = ? "
        "WHERE owner = ? AND stream = ?",
        (json.dumps(state), "telegram-chat-42", "active-energy-burned"),
    )
    archive.responses[path] = [
        {"dataPoints": [{"page": 100001}], "nextPageToken": "third"},
        {"dataPoints": [{"page": 100002}]},
    ]
    await archive.run(12)
    stored = [
        item["data"]["page"]
        for item in await archive.records()
        if "page" in item["data"]
    ]
    assert sorted(stored) == [1, 100001, 100002]
    assert (await archive.storage.archive.progress("telegram-chat-42"))[
        "active-energy-burned"
    ]["status"] == "complete"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, json={"error": {}}),
        httpx.Response(200, headers={"content-type": "text/html"}, content="<html/>"),
        httpx.Response(
            200, headers={"content-type": "application/xml"}, content=b"\xff"
        ),
        httpx.Response(
            200,
            headers={"content-type": "application/xml"},
            content=b"x" * (8 * 1024 * 1024 + 1),
        ),
    ],
)
async def test_bad_tcx_never_blocks_next_exercise_page(
    archive: ArchiveHarness, response: httpx.Response
) -> None:
    path = "/v4/users/me/dataTypes/exercise/dataPoints"
    archive.responses[path] = [
        {
            "dataPoints": [{"name": "users/account/dataTypes/exercise/dataPoints/123"}],
            "nextPageToken": "next",
        },
        {"dataPoints": [{"exercise": {"fromSecondPage": True}}]},
    ]
    archive.responses[
        "/v4/users/account/dataTypes/exercise/dataPoints/123:exportExerciseTcx"
    ] = [response]
    await archive.run(18)
    assert any(
        item["data"].get("exercise", {}).get("fromSecondPage")
        for item in await archive.records()
        if isinstance(item["data"].get("exercise", {}), dict)
    )
    assert not (await archive.storage.archive.coverage("telegram-chat-42"))["complete"]


@pytest.mark.parametrize(
    "membership,override,expected",
    [
        ({}, "2026-11-01", "2026-11-01"),
        ({"year": 2026, "month": 11, "day": 3}, "2026-11-04", "2026-11-03"),
        ({"year": 2026, "month": 11, "day": 3}, "2026-11-01", "2026-11-01"),
        ({}, "2099-01-01", None),
    ],
)
async def test_history_override_can_only_extend_history(
    archive: ArchiveHarness,
    membership: dict[str, Any],
    override: str,
    expected: str | None,
) -> None:
    from dataclasses import replace

    archive.worker.service.config = replace(archive.config, archive_start_date=override)
    archive.responses["/v4/users/me/profile"] = [{"membershipStartDate": membership}]
    await archive.run(18)
    state = (await archive.storage.archive.progress("telegram-chat-42"))[
        "total-calories"
    ]
    if expected:
        assert state["historyStart"] == expected
        assert state["status"] == "complete"
    else:
        assert state["status"] == "blocked_membership_date"


async def test_checkpoint_cas_and_transaction_rollback(archive: ArchiveHarness) -> None:
    connection = await archive.storage.get_connection("telegram-chat-42")
    assert connection is not None
    args: dict[str, Any] = {
        "owner": "telegram-chat-42",
        "account": "account",
        "generation": connection.generation,
        "stream": "steps",
        "expected": None,
        "checkpoint": {"status": "complete"},
        "records": [{"all": "fields"}],
    }
    await archive.conn.execute(
        "CREATE TRIGGER fail_import BEFORE INSERT ON google_health_imports "
        "BEGIN SELECT RAISE(FAIL, 'test failure'); END"
    )
    with pytest.raises(aiosqlite.IntegrityError, match="test failure"):
        await archive.storage.archive.commit_page(**args)
    assert await archive.records() == []
    assert await archive.storage.archive.progress("telegram-chat-42") == {}
    await archive.conn.execute("DROP TRIGGER fail_import")
    assert await archive.storage.archive.commit_page(**args)
    assert not await archive.storage.archive.commit_page(**args)
    assert len(await archive.records()) == 1


async def test_summary_exposes_only_coverage_and_reports_import_progress(
    archive: ArchiveHarness,
) -> None:
    from blacki.health.service import format_health_summary

    await archive.run(12)
    summary = await archive.worker.service.summary("telegram-chat-42")
    text = format_health_summary(summary)
    assert "Full-history archive:" in text
    assert summary["archive"]["complete"] is True
    assert summary["archive"]["observations_stored"] == 3
    assert "membershipStartDate" not in json.dumps(summary)
    assert "unknown" not in json.dumps(summary)


async def test_corrupt_refresh_token_reports_reauthorization(
    archive: ArchiveHarness,
) -> None:
    await archive.conn.execute(
        "UPDATE google_health_connections SET encrypted_refresh_token = 'invalid'"
    )
    await archive.run()
    state = await archive.storage.archive.progress("telegram-chat-42")
    assert state["profile"]["status"] == "reauthorization_required"
    assert archive.requests == []


async def test_expired_pagination_restarts_without_duplicate_observations(
    archive: ArchiveHarness,
) -> None:
    path = "/v4/users/me/dataTypes/active-energy-burned/dataPoints"
    archive.responses[path] = [
        {"dataPoints": [{"retained": True}], "nextPageToken": "expired"},
        httpx.Response(400, json={"error": {}}),
        {"dataPoints": [{"retained": True}], "nextPageToken": "replacement"},
        {"dataPoints": [{"last": True}]},
    ]
    await archive.run(12)
    state = (await archive.storage.archive.progress("telegram-chat-42"))[
        "active-energy-burned"
    ]
    assert state["status"] == "failed"
    assert "pageToken" not in state
    archive.clock[0] += timedelta(minutes=10)
    await archive.run(12)
    assert (
        sum(item["data"] == {"retained": True} for item in await archive.records()) == 1
    )
    assert (await archive.storage.archive.progress("telegram-chat-42"))[
        "active-energy-burned"
    ]["status"] == "complete"


@pytest.mark.parametrize("next_page", [False, True])
async def test_missing_location_permission_does_not_stop_exercise_json(
    archive: ArchiveHarness, next_page: bool
) -> None:
    await archive.connect(
        scopes=tuple(
            s for s in GOOGLE_HEALTH_SCOPES if s != SCOPE_PREFIX + "location.readonly"
        )
    )
    page: dict[str, Any] = {
        "dataPoints": [{"name": "users/account/dataTypes/exercise/dataPoints/1"}]
    }
    if next_page:
        page["nextPageToken"] = "later"
    archive.responses["/v4/users/me/dataTypes/exercise/dataPoints"] = [
        page,
        {"dataPoints": [{"exercise": {"later": True}}]},
    ]
    await archive.run(18)
    state = (await archive.storage.archive.progress("telegram-chat-42"))["exercise"]
    assert state["status"] == "complete"
    assert state["mediaStatus"] == "permission_required"
    assert not any(r.url.path.endswith(":exportExerciseTcx") for r in archive.requests)


async def test_transient_tcx_error_retries_remaining_media(
    archive: ArchiveHarness,
) -> None:
    archive.responses["/v4/users/me/dataTypes/exercise/dataPoints"] = [
        {
            "dataPoints": [
                {"name": "users/account/dataTypes/exercise/dataPoints/1"},
                {"name": "users/account/dataTypes/exercise/dataPoints/2"},
            ]
        }
    ]
    archive.responses[
        "/v4/users/account/dataTypes/exercise/dataPoints/1:exportExerciseTcx"
    ] = [
        httpx.Response(503, json={"error": {}}),
        httpx.Response(200, headers={"content-type": "text/xml"}, content="<first/>"),
    ]
    archive.responses[
        "/v4/users/account/dataTypes/exercise/dataPoints/2:exportExerciseTcx"
    ] = [httpx.Response(200, headers={"content-type": "text/xml"}, content="<second/>")]
    await archive.run(12)
    archive.clock[0] += timedelta(minutes=10)
    await archive.run(12)
    values = [
        item["data"].get("tcx")
        for item in await archive.records()
        if item["representation"] == "v4:exercise-tcx"
    ]
    assert set(values) == {"<first/>", "<second/>"}


async def test_transport_failure_is_bounded_and_visible(
    archive: ArchiveHarness,
) -> None:
    async def fail() -> None:
        raise httpx.ConnectError("sensitive connection information")

    archive.responses["/v4/users/me/dataTypes/active-energy-burned/dataPoints"] = [fail]
    await archive.run()
    state = await archive.storage.archive.progress("telegram-chat-42")
    assert state["active-energy-burned"]["status"] == "failed"
    assert "sensitive" not in json.dumps(state)


@pytest.mark.parametrize("value,valid", [("2000-01-01", True), ("invalid", False)])
def test_configured_archive_boundary_is_validated(value: str, valid: bool) -> None:
    from blacki.health.config import GoogleHealthConfigurationError

    values = {
        "GOOGLE_HEALTH_CLIENT_ID": "client",
        "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
        "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
        "GOOGLE_HEALTH_ARCHIVE_START_DATE": value,
    }
    if valid:
        config = GoogleHealthConfig.from_environment(values)
        assert config is not None and config.archive_start_date == value
    else:
        with pytest.raises(GoogleHealthConfigurationError, match="YYYY-MM-DD"):
            GoogleHealthConfig.from_environment(values)


async def test_oversized_ecg_page_restarts_with_smaller_page_and_keeps_waveform(
    archive: ArchiveHarness,
) -> None:
    path = "/v4/users/me/dataTypes/electrocardiogram/dataPoints"
    archive.responses[path] = [
        httpx.Response(200, content=b"x" * (8 * 1024 * 1024 + 1)),
        {
            "dataPoints": [
                {"electrocardiogram": {"waveformSamples": [1, -2, 3], "unknown": 17}}
            ]
        },
    ]
    await archive.run(12)
    state = (await archive.storage.archive.progress("telegram-chat-42"))[
        "electrocardiogram"
    ]
    assert state["pageSize"] == 500
    archive.clock[0] += timedelta(minutes=10)
    await archive.run(12)
    calls = [r for r in archive.requests if r.url.path == path]
    assert calls[0].url.params["pageSize"] == "1000"
    assert calls[1].url.params["pageSize"] == "500"
    assert any(
        item["data"].get("electrocardiogram", {}).get("waveformSamples") == [1, -2, 3]
        for item in await archive.records()
    )
    archive.clock[0] += timedelta(days=2)
    await archive.run(6)
    assert [r for r in archive.requests if r.url.path == path][-1].url.params[
        "pageSize"
    ] == "500"


async def test_single_oversized_record_is_an_explicit_limitation(
    archive: ArchiveHarness,
) -> None:
    path = "/v4/users/me/dataTypes/electrocardiogram/dataPoints"
    archive.responses[path] = [
        httpx.Response(200, content=b"x" * (8 * 1024 * 1024 + 1))
    ]
    await archive.run(6)
    state = (await archive.storage.archive.progress("telegram-chat-42"))[
        "electrocardiogram"
    ]
    state["pageSize"] = 1
    await archive.conn.execute(
        "UPDATE google_health_imports SET checkpoint = ? WHERE stream = ?",
        (json.dumps(state), "electrocardiogram"),
    )
    archive.clock[0] += timedelta(minutes=10)
    archive.responses[path] = [
        httpx.Response(200, content=b"x" * (8 * 1024 * 1024 + 1))
    ]
    await archive.run(6)
    assert (await archive.storage.archive.progress("telegram-chat-42"))[
        "electrocardiogram"
    ]["status"] == "record_too_large"


async def test_compressed_provider_page_is_decoded_once(
    archive: ArchiveHarness,
) -> None:
    import gzip

    payload = {"dataPoints": [{"steps": {"count": 1234}, "futureField": True}]}
    archive.responses["/v4/users/me/dataTypes/active-energy-burned/dataPoints"] = [
        httpx.Response(
            200,
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
            content=gzip.compress(json.dumps(payload).encode()),
        )
    ]
    await archive.run()
    assert any(
        item["data"] == payload["dataPoints"][0] for item in await archive.records()
    )
    state = (await archive.storage.archive.progress("telegram-chat-42"))[
        "active-energy-burned"
    ]
    assert state["status"] == "complete"


async def test_older_observation_reopens_only_uncovered_aggregate_history(
    archive: ArchiveHarness,
) -> None:
    await archive.run(15)
    assert (await archive.storage.archive.coverage("telegram-chat-42"))["complete"]
    # Only the next HRV scan is due; older evidence must reopen other streams.
    future = (archive.clock[0] + timedelta(days=3)).isoformat()
    states = await archive.storage.archive.progress("telegram-chat-42")
    for key, state in states.items():
        state["due"] = "" if key == "daily-heart-rate-variability" else future
        await archive.conn.execute(
            "UPDATE google_health_imports SET checkpoint=? WHERE owner=? AND stream=?",
            (json.dumps(state), "telegram-chat-42", key),
        )
    archive.responses[
        "/v4/users/me/dataTypes/daily-heart-rate-variability/dataPoints"
    ] = [
        {
            "dataPoints": [
                {
                    "dailyHeartRateVariability": {
                        "date": {"year": 2026, "month": 11, "day": 1}
                    }
                }
            ]
        }
    ]
    start = len(archive.requests)
    await archive.worker.run_once(page_budget=1)
    coverage = await archive.storage.archive.coverage("telegram-chat-42")
    assert coverage["complete"] is False
    assert coverage["streams"]["total-calories"]["status"] == "extension_pending"
    assert (
        coverage["streams"]["total-calories"]["completed"]
        == states["total-calories"]["completed"]
    )
    # Restart and reconsent retain the historical evidence even when progress resets.
    archive.worker = HealthArchiveWorker(archive.worker.service)
    await archive.run(8)
    rollups = [
        json.loads(r.content)
        for r in archive.requests[start:]
        if r.url.path.endswith("/total-calories/dataPoints:rollUp")
    ]
    assert rollups[0]["range"]["endTime"] == "2026-11-03T00:00:00+00:00"
    assert rollups[-1]["range"]["startTime"] == "2026-11-01T00:00:00+00:00"
    assert (await archive.storage.archive.coverage("telegram-chat-42"))["complete"]
    await archive.connect()
    assert (
        await archive.storage.archive.observed_start("telegram-chat-42") == "2026-11-01"
    )
    await archive.connect(account="new-account")
    assert await archive.storage.archive.observed_start("telegram-chat-42") is None


@pytest.mark.parametrize(
    "component,expected",
    [
        ({"date": {"year": 2026, "month": 10, "day": 1}}, "2026-10-01"),
        ({"sampleTime": {"physicalTime": "2026-10-01T01:00:00+02:00"}}, "2026-09-30"),
        (
            {
                "sampleTime": {
                    "civilTime": {"year": 2026, "month": 9, "day": 29},
                    "physicalTime": "2026-10-01T00:00:00Z",
                }
            },
            "2026-09-29",
        ),
        (
            {
                "interval": {
                    "startTime": "2026-10-01T00:00:00Z",
                    "civilStartTime": {"year": 2026, "month": 9, "day": 30},
                }
            },
            "2026-09-30",
        ),
        ({"sampleTime": {"physicalTime": "bad"}}, None),
        ({"sampleTime": {"physicalTime": "2026-10-01T00:00:00"}}, None),
        ({"sampleTime": {"physicalTime": 42}}, None),
    ],
)
async def test_observation_dates_extend_history_without_using_unrelated_dates(
    archive: ArchiveHarness, component: dict[str, Any], expected: str | None
) -> None:
    archive.responses["/v4/users/me/dataTypes/active-energy-burned/dataPoints"] = [
        {
            "dataPoints": [
                {
                    "activeEnergyBurned": component,
                    "metadata": {"date": {"year": 1900, "month": 1, "day": 1}},
                }
            ]
        }
    ]
    await archive.run()
    assert await archive.storage.archive.observed_start("telegram-chat-42") == expected


async def test_aggregate_pagination_keeps_window_when_older_data_arrives(
    archive: ArchiveHarness,
) -> None:
    path = "/v4/users/me/dataTypes/total-calories/dataPoints:rollUp"
    archive.responses[path] = [
        {"rollupDataPoints": [], "nextPageToken": "continue"},
        {"rollupDataPoints": []},
    ]
    await archive.run(6)
    state = (await archive.storage.archive.progress("telegram-chat-42"))[
        "total-calories"
    ]
    assert state["pageToken"] == "continue"
    # An independently committed recorded page extends the desired historical bound.
    connection = await archive.storage.get_connection("telegram-chat-42")
    assert connection is not None
    await archive.storage.archive.commit_page(
        owner="telegram-chat-42",
        account="account",
        generation=connection.generation,
        stream="test-recorded-evidence",
        expected=None,
        checkpoint={"status": "complete"},
        records=[],
        observed_start="2026-11-01",
    )
    await archive.run(6)
    bodies = [json.loads(r.content) for r in archive.requests if r.url.path == path]
    assert bodies[0]["range"] == bodies[1]["range"]
    assert bodies[1]["pageToken"] == "continue"
    assert (await archive.storage.archive.progress("telegram-chat-42"))[
        "total-calories"
    ]["historyStart"] == "2026-11-01"
