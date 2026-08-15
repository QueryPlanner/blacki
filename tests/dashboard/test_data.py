"""Focused contract and security tests for the local dashboard store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, cast

import pytest

from blacki.dashboard import DashboardStore
from blacki.dashboard.data import (
    _activity_series,
    _allowlisted_attributes,
    _attachment_item,
    _event_is_error,
    _event_messages,
    _event_stats,
    _event_usage,
    _EventRow,
    _in_window,
    _iso_timestamp,
    _iter_raw_jsonl_lines,
    _JsonlScan,
    _load_snapshot_sync,
    _log_item,
    _normalize_epoch_seconds,
    _number,
    _parse_json_object,
    _parse_timestamp,
    _sanitize_json,
    _sanitize_text,
    _scan_jsonl_sync,
    _scan_warning,
    _select_table_rows,
    _session_version,
    _tool_status,
    _trace_groups,
    _trace_item,
    _trace_latency_stats,
    _trace_matches,
    _trace_summary,
    _window_bounds,
)
from blacki.dashboard.models import clamp_limit, clamp_offset


def _make_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE sessions (
            app_name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            id TEXT NOT NULL,
            state TEXT,
            create_time TEXT,
            update_time TEXT,
            PRIMARY KEY (app_name, user_id, id)
        );
        CREATE TABLE events (
            id TEXT NOT NULL,
            app_name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            invocation_id TEXT,
            timestamp REAL,
            event_data TEXT
        );
        """
    )
    connection.commit()
    connection.close()


def _insert_session(
    path: Path,
    user_id: str,
    session_id: str,
    *,
    created: str = "2026-08-15T10:00:00+00:00",
    updated: str = "2026-08-15T10:01:00+00:00",
) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
        ("blacki", user_id, session_id, "{}", created, updated),
    )
    connection.commit()
    connection.close()


def _insert_event(
    path: Path,
    event_id: str,
    user_id: str,
    session_id: str,
    timestamp: float,
    data: dict[str, object],
) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            "blacki",
            user_id,
            session_id,
            str(data.get("invocation_id", "")),
            timestamp,
            json.dumps(data),
        ),
    )
    connection.commit()
    connection.close()


def _text_event(
    event_id: str,
    role: str,
    text: str,
    timestamp: float,
    *,
    invocation_id: str = "inv-1",
) -> dict[str, object]:
    return {
        "id": event_id,
        "timestamp": timestamp,
        "invocation_id": invocation_id,
        "author": "user" if role == "user" else "root_agent",
        "content": {"role": role, "parts": [{"text": text}]},
    }


@pytest.mark.asyncio
async def test_sessions_users_replay_versions_and_overview(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _make_db(db_path)
    _insert_session(db_path, "user-1", "telegram-chat-1-v1")
    _insert_session(db_path, "user-1", "telegram-chat-1-v2")
    _insert_session(db_path, "user-2", "other-session")
    human_event = _text_event(
        "human", "user", "<script>alert(1)</script>", 1_781_000_000.0
    )
    human_event["content"] = {
        "role": "user",
        "parts": [
            {"text": "<script>alert(1)</script>"},
            {"inlineData": {"mimeType": "image/png", "data": "binary-secret"}},
            {"file_data": {"mime_type": "application/pdf", "file_uri": "secret"}},
        ],
    }
    _insert_event(
        db_path,
        "human",
        "user-1",
        "telegram-chat-1-v1",
        1_781_000_000.0,
        human_event,
    )
    _insert_event(
        db_path,
        "model",
        "user-1",
        "telegram-chat-1-v1",
        1_781_000_001.0,
        {
            **_text_event("model", "model", "A safe answer", 1_781_000_001.0),
            "usage_metadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 7,
                "totalTokenCount": 11,
            },
        },
    )
    _insert_event(
        db_path,
        "thought-tools",
        "user-1",
        "telegram-chat-1-v1",
        1_781_000_002.0,
        {
            "id": "thought-tools",
            "timestamp": 1_781_000_002.0,
            "invocation_id": "inv-1",
            "author": "root_agent",
            "content": {
                "role": "model",
                "parts": [
                    {"text": "private thought", "thought": True},
                    {
                        "function_call": {
                            "name": "dangerous_tool",
                            "args": {"secret": "do-not-return"},
                        }
                    },
                    {
                        "function_response": {
                            "name": "dangerous_tool",
                            "response": {"result": "do-not-return"},
                        }
                    },
                    {"inlineData": {"data": "binary"}},
                ],
            },
        },
    )
    _insert_event(
        db_path,
        "reset-human",
        "user-1",
        "telegram-chat-1-v2",
        1_781_000_003.0,
        _text_event("reset-human", "user", "new generation", 1_781_000_003.0),
    )

    store = DashboardStore(db_path, tmp_path)
    users = await store.list_users("user-1", 10, 0)
    assert users["total"] == 1
    assert users["items"][0]["session_count"] == 2
    assert users["items"][0]["reset_count"] == 1
    assert users["items"][0]["retained_history"] is True

    sessions = await store.list_sessions("user-1", 10, 0)
    assert [item["version"] for item in sessions["items"]] == [2, 1]
    assert sessions["items"][0]["is_current"] is True
    assert sessions["items"][0]["retained_session_ids"] == ["telegram-chat-1-v1"]

    replay = await store.get_session("user-1", "telegram-chat-1-v1")
    assert replay is not None
    assert [message["role"] for message in replay["messages"] if "role" in message] == [
        "human",
        "model",
    ]
    assert replay["messages"][0]["text"] == "<script>alert(1)</script>"
    assert [
        message["mime_type"]
        for message in replay["messages"]
        if message.get("type") == "attachment"
    ] == ["image/png", "application/pdf", "unknown"]
    assert "private thought" not in json.dumps(replay)
    assert "do-not-return" not in json.dumps(replay)
    assert "binary-secret" not in json.dumps(replay)
    assert "file_uri" not in json.dumps(replay)
    assert any(
        message.get("name") == "dangerous_tool" for message in replay["messages"]
    )
    assert replay["reset_generation"] == 0

    overview = await store.get_overview("all")
    assert overview["stats"]["users"] == 2
    assert overview["stats"]["sessions"] == 3
    assert overview["stats"]["messages"] == 3
    assert overview["stats"]["tokens"] == 11
    assert overview["activity"]


@pytest.mark.asyncio
async def test_overview_error_rate_uses_failed_invocations_only(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _make_db(db_path)
    _insert_session(db_path, "user-1", "session-1")
    for event_id in ("failed-start", "failed-end"):
        _insert_event(
            db_path,
            event_id,
            "user-1",
            "session-1",
            1_781_000_000.0,
            {
                **_text_event(
                    event_id,
                    "model",
                    "failed",
                    1_781_000_000.0,
                    invocation_id="failed-invocation",
                ),
                "error": "provider failure",
            },
        )
    (tmp_path / "blacki-telemetry.log").write_text(
        json.dumps(
            {
                "timestamp": 1_781_000_000.0,
                "level": "ERROR",
                "message": "request failed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "blacki-traces.log").write_text(
        json.dumps(
            {
                "name": "failed-request",
                "context": {"trace_id": "trace-1", "span_id": "span-1"},
                "start_time": 1_781_000_000.0,
                "end_time": 1_781_000_001.0,
                "status": "ERROR",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    overview = await DashboardStore(db_path, tmp_path).get_overview("all")

    assert overview["stats"]["requests"] == 1
    assert overview["stats"]["errors"] == 1
    assert overview["stats"]["request_errors"] == 1
    assert overview["stats"]["log_errors"] == 1
    assert overview["stats"]["trace_errors"] == 1
    assert overview["stats"]["error_rate"] == 1.0


@pytest.mark.asyncio
async def test_read_only_snapshot_can_read_during_wal_writer(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _make_db(db_path)
    _insert_session(db_path, "writer-user", "writer-v1")
    writer = sqlite3.connect(db_path, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "UPDATE sessions SET state = ? WHERE user_id = ?",
        ('{"pending":true}', "writer-user"),
    )
    try:
        result = await DashboardStore(db_path, tmp_path).list_users("", 50, 0)
        assert result["items"][0]["user_id"] == "writer-user"
        assert not result["degraded"]
    finally:
        writer.rollback()
        writer.close()


@pytest.mark.asyncio
async def test_missing_and_corrupt_database_degrade_without_paths(
    tmp_path: Path,
) -> None:
    missing = await DashboardStore(tmp_path / "missing.db", tmp_path).list_users(
        "", 50, 0
    )
    assert missing["degraded"] is True
    assert all("missing.db" not in warning for warning in missing["warnings"])

    corrupt_path = tmp_path / "corrupt.db"
    corrupt_path.write_bytes(b"not a sqlite database")
    corrupt = await DashboardStore(corrupt_path, tmp_path).list_sessions("u", 50, 0)
    assert corrupt["degraded"] is True
    assert all("not a sqlite" not in warning for warning in corrupt["warnings"])
    assert await DashboardStore(corrupt_path, tmp_path).get_session("u", "s") is None


@pytest.mark.asyncio
async def test_logs_stream_caps_and_redacts_secrets(tmp_path: Path) -> None:
    log_path = tmp_path / "blacki-telemetry.log"
    rows = [
        {
            "timestamp": "2026-08-15T10:00:00Z",
            "level": "error",
            "name": "test",
            "message": (
                "Bearer abc.def and api_key=super-secret "
                "telegram token 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij "
                "at /Users/private/user/project/file.py"
            ),
            "exception": "ValueError: /private/secret/path",
            "cookie": "session-cookie",
        },
        {"timestamp": "2026-08-15T10:01:00Z", "level": "INFO", "message": "needle"},
        {"level": "INFO", "message": "final line"},
    ]
    log_path.write_bytes(
        b"\xff\xfe malformed\n"
        + json.dumps(rows[0]).encode()
        + b"\n"
        + b"{"
        + b"x" * (1024 * 1024 + 10)
        + b"}\n"
        + json.dumps(rows[1]).encode()
        + b"\n"
        + json.dumps(rows[2]).encode()
    )
    store = DashboardStore(tmp_path / "missing.db", tmp_path)
    logs = await store.list_logs("ERROR", "", 10)
    assert logs["total"] == 1
    serialized = json.dumps(logs["items"][0])
    assert "super-secret" not in serialized
    assert "abc.def" not in serialized
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij" not in serialized
    assert "/Users/private" not in serialized
    assert "ValueError" not in serialized
    assert any("malformed" in warning for warning in logs["warnings"])
    assert any("1 MiB" in warning for warning in logs["warnings"])
    final = await store.list_logs(None, "final", 10)
    assert final["items"][0]["message"] == "final line"


@pytest.mark.asyncio
async def test_trace_projection_allowlist_filters_content(tmp_path: Path) -> None:
    trace_path = tmp_path / "blacki-traces.log"
    trace_path.write_text(
        json.dumps(
            {
                "name": "agent.invoke",
                "context": {"trace_id": "trace-1", "span_id": "span-1"},
                "start_time": 1_781_000_000,
                "end_time": 1_781_000_001,
                "status": {"status_code": "ERROR", "description": "secret prompt"},
                "attributes": {
                    "service.name": "blacki",
                    "gen_ai.request.model": "model-x",
                    "gcp.vertex.agent.session_id": "session-1",
                    "gcp.vertex.agent.invocation_id": "invocation-1",
                    "gen_ai.conversation.id": "conversation-1",
                    "gen_ai.tool.name": "search",
                    "agent.name": "root-agent",
                    "llm.model_name": "model-x",
                    "gen_ai.usage.prompt_tokens": 3,
                    "llm.token_count.input": "not-numeric",
                    "gen_ai.prompt": "do-not-return",
                    "gen_ai.input": "do-not-return",
                    "gen_ai.usage.input_tokens": 3,
                    "tool.args": "do-not-return",
                },
                "events": [
                    {
                        "name": "model.response",
                        "timestamp": 1_781_000_001,
                        "attributes": {
                            "http.status_code": 500,
                            "gen_ai.output": "do-not-return",
                        },
                    }
                ],
            }
        )
        + "\n"
        + json.dumps({"name": "without-context"})
        + "\n",
        encoding="utf-8",
    )
    store = DashboardStore(tmp_path / "missing.db", tmp_path)
    traces = await store.list_traces("error", "model-x", 10)
    assert traces["total"] == 1
    trace = traces["items"][0]
    assert trace["trace_id"] == "trace-1"
    assert trace["duration_ms"] == 1000.0
    assert trace["attributes"] == {
        "service.name": "blacki",
        "gen_ai.request.model": "model-x",
        "gcp.vertex.agent.session_id": "session-1",
        "gcp.vertex.agent.invocation_id": "invocation-1",
        "gen_ai.conversation.id": "conversation-1",
        "gen_ai.tool.name": "search",
        "agent.name": "root-agent",
        "llm.model_name": "model-x",
        "gen_ai.usage.input_tokens": 3,
        "gen_ai.usage.prompt_tokens": 3,
    }
    assert "llm.token_count.input" not in trace["attributes"]
    serialized = json.dumps(trace)
    assert "do-not-return" not in serialized
    assert "secret prompt" not in serialized
    assert "description" not in serialized
    found = await store.get_trace("trace-1")
    assert found is not None
    assert found["trace_id"] == "trace-1"
    assert await store.get_trace("not-present") is None


@pytest.mark.asyncio
async def test_trace_groups_are_newest_first_and_windowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "blacki-traces.log"
    trace_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "name": "root",
                    "context": {"trace_id": "trace-a", "span_id": "a1"},
                    "start_time": 100,
                    "end_time": 105,
                    "status": "OK",
                },
                {
                    "name": "child",
                    "context": {"trace_id": "trace-a", "span_id": "a2"},
                    "start_time": 102,
                    "end_time": 110,
                    "status": "ERROR",
                },
                {
                    "name": "newer-trace",
                    "context": {"trace_id": "trace-b", "span_id": "b1"},
                    "start_time": 200,
                    "end_time": 202,
                    "status": "OK",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "blacki-telemetry.log").write_text(
        json.dumps({"timestamp": 50, "level": "INFO", "message": "old"})
        + "\n"
        + json.dumps({"timestamp": 150, "level": "ERROR", "message": "new"})
        + "\n",
        encoding="utf-8",
    )
    store = DashboardStore(tmp_path / "missing.db", tmp_path)

    traces = await store.list_traces(None, "", 10)
    assert [item["trace_id"] for item in traces["items"]] == [
        "trace-b",
        "trace-a",
    ]
    assert traces["items"][1]["span_count"] == 2
    assert traces["items"][1]["start_time"] == "1970-01-01T00:01:40+00:00"
    assert traces["items"][1]["end_time"] == "1970-01-01T00:01:50+00:00"
    assert traces["items"][1]["duration_ms"] == 10_000.0
    assert traces["items"][1]["status"] == "ERROR"
    assert (await store.list_traces(None, "child", 10))["total"] == 1
    assert (await store.list_traces("ERROR", "", 10))["total"] == 1

    detail = await store.get_trace("trace-a")
    assert detail is not None
    assert detail["span_count"] == 2
    assert len(detail["spans"]) == 2
    assert detail["spans"][0]["span_id"] == "a2"

    monkeypatch.setattr(
        "blacki.dashboard.data._window_bounds",
        lambda _window: ("1h", 90.0, 150.0, []),
    )
    overview = await store.get_overview("1h")
    assert overview["stats"]["log_records"] == 1
    assert overview["stats"]["trace_spans"] == 2
    assert overview["stats"]["traces"] == 1
    assert overview["stats"]["avg_latency_ms"] == 10_000.0
    assert overview["stats"]["p95_latency_ms"] == 10_000.0


@pytest.mark.asyncio
async def test_extreme_trace_timestamps_degrade_without_endpoint_errors(
    tmp_path: Path,
) -> None:
    (tmp_path / "blacki-traces.log").write_text(
        json.dumps(
            {
                "name": "damaged-span",
                "context": {"trace_id": "damaged-trace"},
                "start_time": 10**100,
                "end_time": -(10**100),
                "status": "ERROR",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = DashboardStore(tmp_path / "missing.db", tmp_path)

    traces = await store.list_traces(None, "", 5)
    assert traces["total"] == 1
    assert traces["items"][0]["trace_id"] == "damaged-trace"
    assert traces["items"][0]["start_time"] is None
    assert traces["items"][0]["end_time"] is None

    overview = await store.get_overview("all")
    assert overview["degraded"] is True
    assert overview["stats"]["traces"] == 0
    assert "damaged-trace" not in json.dumps(overview["activity"])


@pytest.mark.asyncio
async def test_limits_missing_logs_and_unknown_window(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path / "missing.db", tmp_path)
    logs = await store.list_logs(None, "x" * 500, 9999)
    traces = await store.list_traces(None, "x" * 500, 9999)
    assert logs["limit"] == 200
    assert traces["limit"] == 200
    assert logs["degraded"] is True
    assert traces["degraded"] is True
    overview = await store.get_overview("bad-window")
    assert overview["window"] == "24h"
    assert any("Unknown window" in warning for warning in overview["warnings"])


@pytest.mark.asyncio
async def test_old_adk_columns_and_malformed_events_are_adapted(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            app_name TEXT, user_id TEXT, id TEXT, state TEXT
        );
        CREATE TABLE events (
            id TEXT, app_name TEXT, user_id TEXT, session_id TEXT,
            invocation_id TEXT, timestamp REAL, author TEXT, content TEXT,
            usage_metadata TEXT, error_code TEXT, error_message TEXT,
            partial INTEGER, turn_complete INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("blacki", "legacy", "legacy-v1", "{}"),
    )
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-event",
            "blacki",
            "legacy",
            "legacy-v1",
            "",
            1_781_000_100,
            "user",
            json.dumps({"role": "user", "parts": [{"text": "legacy"}]}),
            json.dumps({"promptTokenCount": "2"}),
            "E_TEST",
            "failed",
            0,
            1,
        ),
    )
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "bad-event",
            "blacki",
            "legacy",
            "legacy-v1",
            "inv",
            None,
            "agent",
            "not-json",
            "not-json",
            None,
            None,
            None,
            None,
        ),
    )
    connection.commit()
    connection.close()

    session = await DashboardStore(db_path, tmp_path).get_session("legacy", "legacy-v1")
    assert session is not None
    assert session["messages"][0]["text"] == "legacy"
    assert session["input_tokens"] == 2
    assert session["error_count"] == 1
    assert any("decoded" in warning for warning in session["warnings"])


def test_defensive_adapters_cover_scalars_windows_and_trace_shapes() -> None:
    assert _number(True) == 0
    assert _number(-2) == 0
    assert _number("bad") == 0
    assert _number(object()) == 0
    assert _parse_json_object({"x": 1}) == ({"x": 1}, False)
    assert _parse_json_object(b'{"x": 1}') == ({"x": 1}, False)
    assert _parse_json_object("[]")[1] is True
    assert _parse_json_object(3)[1] is True
    assert _parse_json_object(None) == ({}, False)
    assert _parse_timestamp(True) is None
    assert _parse_timestamp(object()) is None
    assert _parse_timestamp("   ") is None
    assert _parse_timestamp("5.5") == 5.5
    assert _parse_timestamp(1_781_000_000_000) == 1_781_000_000
    assert _parse_timestamp(1_781_000_000_000_000) == 1_781_000_000
    assert _parse_timestamp(1_781_000_000_000_000_000) == 1_781_000_000
    assert _parse_timestamp(10**100) is None
    assert _parse_timestamp("1e100") is None
    assert _iso_timestamp(10**100) is None
    assert _parse_timestamp(datetime(2026, 1, 1)) is not None
    assert _parse_timestamp("2026-01-01T00:00:00") is not None
    assert _window_bounds(None)[0] == "24h"
    assert _window_bounds("1w")[1] is not None
    assert _window_bounds("9999d")[1] is not None
    assert _in_window(None, None, 10) is False
    assert _in_window(5, 10, 20) is False
    assert _in_window(15, 10, 20) is True
    assert _session_version("plain") == ("plain", None)
    assert _session_version("chat-v2") == ("chat", 2)
    assert _tool_status({"status": "failed"}) == "error"
    assert _tool_status({"error": "hidden"}) == "error"
    assert _tool_status({"response": {"error": "hidden"}}) == "error"
    assert _tool_status({}) == "completed"
    assert _allowlisted_attributes(None) == {}
    assert _allowlisted_attributes({"gen_ai.response.model": "m"}) == {
        "gen_ai.response.model": "m"
    }
    attrs = _allowlisted_attributes(
        {
            "service.version": float("nan"),
            "service.name": object(),
            "gen_ai.prompt": "hidden",
        }
    )
    assert attrs["service.version"] is None
    assert str(attrs["service.name"]).startswith("<object object")
    assert _trace_item({"context": {"trace_id": "t"}}) is not None
    assert _trace_item({"name": "no-id"}) is None
    typed_trace = _trace_item(
        {
            "context": {"trace_id": "typed"},
            "kind": 1,
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2025-01-01T00:00:00Z",
        }
    )
    assert typed_trace is not None
    assert typed_trace["kind"] == "1"
    none_kind_trace = _trace_item({"context": {"trace_id": "none-kind"}, "kind": None})
    assert none_kind_trace is not None
    assert none_kind_trace["kind"] is None
    string_kind_trace = _trace_item(
        {"context": {"trace_id": "string-kind"}, "kind": "internal"}
    )
    assert string_kind_trace is not None
    assert string_kind_trace["kind"] == "internal"
    assert _trace_matches({"trace_id": "trace"}, "") is True
    assert _trace_groups([{"trace_id": ""}]) == []
    empty_summary = _trace_summary([{"trace_id": "manual", "attributes": None}])
    assert empty_summary["duration_ms"] == 0.0
    assert _trace_latency_stats([]) == {"average": 0.0, "p95": 0.0, "max": 0.0}


def test_stats_and_warning_edges() -> None:
    event = _EventRow("e", "u", "s", "", 3.0, {"error": "x"})
    assert _event_is_error(event) is True
    assert _event_is_error(_EventRow("e2", "u", "s", "", None, {"error_message": "x"}))
    assert _event_usage(event) == (0, 0, 0)
    stats = _event_stats([event])
    assert stats["latency_ms"]["max"] == 0.0
    assert _scan_warning(_JsonlScan((), 0, 1, 1), missing="missing")
    assert _scan_warning(_JsonlScan((), 10_000, 0, 0), missing="missing")
    assert _scan_warning(_JsonlScan((), 1, 0, 0, truncated=True), missing="missing")
    assert clamp_limit(cast(Any, "bad")) == 50
    assert clamp_offset(cast(Any, "bad")) == 0
    assert _activity_series(
        [
            _EventRow("out", "u", "s", "", 3.0, {}),
            _EventRow("in", "u", "s", "", 5.0, {}),
        ],
        since=4.0,
        until=6.0,
        window="1w",
    )


def test_timestamp_conversion_failures_are_quarantined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenDatetime(datetime):
        def timestamp(self) -> float:
            raise OSError("damaged timestamp")

    assert _parse_timestamp(10**1000) is None
    assert _parse_timestamp(BrokenDatetime(2026, 1, 1)) is None
    assert _normalize_epoch_seconds(cast(Any, object())) is None

    class FailingDateFactory(datetime):
        def timestamp(self) -> float:
            raise OSError("damaged timestamp")

        @classmethod
        def fromisoformat(cls, _value: str) -> FailingDateFactory:
            return cls(2026, 1, 1)

        @classmethod
        def fromtimestamp(
            cls, _timestamp: float, tz: tzinfo | None = None
        ) -> FailingDateFactory:
            raise OSError("damaged timestamp")

    monkeypatch.setattr("blacki.dashboard.data.datetime", FailingDateFactory)
    assert _parse_timestamp("2026-01-01T00:00:00Z") is None
    assert _iso_timestamp(0) is None


def test_canonical_helpers_tolerate_unknown_and_timestamp_values() -> None:
    assert _parse_timestamp("not-a-date") is None
    assert _parse_timestamp(float("nan")) is None
    event = _EventRow(
        event_id="e",
        user_id="u",
        session_id="s",
        invocation_id="i",
        timestamp=None,
        data={"content": {"parts": [{"unknown": True}]}},
    )
    assert _event_messages(event) == []
    attachment = _attachment_item({"inlineData": "opaque"}, event=event, index=0)
    assert attachment is not None
    assert attachment["mime_type"] == "unknown"
    redacted = _sanitize_text("Authorization: Bearer abc123 at /tmp/file")
    assert "abc123" not in redacted
    assert "/tmp/file" not in redacted


def test_bounded_jsonl_iterator_and_json_sanitizer_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ChunkedHandle:
        def __init__(self, chunks: list[bytes]) -> None:
            self.chunks = iter(chunks)

        def read(self, _size: int) -> bytes:
            return next(self.chunks, b"")

    oversized = list(
        _iter_raw_jsonl_lines(
            ChunkedHandle(
                [
                    b"x" * (1024 * 1024 + 1),
                    b"y" * (64 * 1024),
                    b"z" * (64 * 1024),
                    b'\n{"ok": true}\n',
                ]
            )
        )
    )
    assert oversized == [(b"", True), (b'{"ok": true}', False)]
    assert list(_iter_raw_jsonl_lines(ChunkedHandle([b"x" * (1024 * 1024 + 1)]))) == [
        (b"", True)
    ]
    assert list(_iter_raw_jsonl_lines(ChunkedHandle([b'{"x": 1}']))) == [
        (b'{"x": 1}', False)
    ]
    scalar_path = tmp_path / "scalar.jsonl"
    scalar_path.write_text("[]\n{}\n", encoding="utf-8")
    scanned = _scan_jsonl_sync(scalar_path)
    assert scanned.accepted == 1
    assert scanned.invalid == 1
    capped_path = tmp_path / "capped.jsonl"
    capped_path.write_text(
        "".join(json.dumps({"index": index}) + "\n" for index in range(10_001)),
        encoding="utf-8",
    )
    capped = _scan_jsonl_sync(capped_path)
    assert capped.accepted == 10_000
    assert capped.records[0]["index"] == 1
    assert capped.records[-1]["index"] == 10_000

    tail_path = tmp_path / "tail.jsonl"
    tail_path.write_text(
        "".join(json.dumps({"index": index}) + "\n" for index in range(5)),
        encoding="utf-8",
    )
    tail_scan = _scan_jsonl_sync(tail_path)
    assert tail_scan.truncated is False
    monkeypatch.setattr("blacki.dashboard.data.MAX_JSONL_TAIL_BYTES", 25)
    tail_scan = _scan_jsonl_sync(tail_path)
    assert tail_scan.truncated is True
    assert tail_scan.records[-1]["index"] == 4
    no_newline = tmp_path / "no-newline.jsonl"
    no_newline.write_bytes(b"x" * 40)
    assert _scan_jsonl_sync(no_newline).truncated is True

    sanitized = _sanitize_json(
        {
            "token": "hidden",
            "exception": "stack",
            "values": [float("nan"), object()],
        }
    )
    assert sanitized["token"] == "[REDACTED]"  # noqa: S105
    assert sanitized["exception"] == "[REDACTED_EXCEPTION]"
    assert sanitized["values"][0] is None
    assert sanitized["values"][1].startswith("<object object")
    assert sanitized["values"][1].endswith(">")
    assert {
        "token": "[REDACTED]",
        "exception": "[REDACTED_EXCEPTION]",
    }.items() <= sanitized.items()
    assert _sanitize_text("x" * (64 * 1024 + 1)).endswith("…")
    assert _log_item({"level": 10, "message": "numeric"})["message"] == "numeric"


def test_snapshot_schema_and_event_projection_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    snapshot = _load_snapshot_sync(empty, "blacki")
    assert snapshot.sessions == ()
    assert snapshot.warnings

    malformed_schema = tmp_path / "malformed.db"
    connection = sqlite3.connect(malformed_schema)
    connection.execute("CREATE TABLE sessions (user_id TEXT)")
    connection.execute("CREATE TABLE events (app_name TEXT)")
    connection.commit()
    connection.close()
    assert _load_snapshot_sync(malformed_schema, "blacki").warnings

    rows_db = tmp_path / "bad-rows.db"
    connection = sqlite3.connect(rows_db)
    connection.execute(
        "CREATE TABLE sessions (app_name TEXT, user_id, id, create_time, update_time)"
    )
    connection.execute(
        "CREATE TABLE events (app_name TEXT, user_id, session_id, id, timestamp)"
    )
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
        ("blacki", 4, 5, None, None),
    )
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        ("blacki", 4, 5, None, None),
    )
    connection.commit()
    connection.close()
    assert _load_snapshot_sync(rows_db, "blacki").sessions == ()

    event_db = tmp_path / "event-fallback.db"
    connection = sqlite3.connect(event_db)
    connection.execute(
        "CREATE TABLE sessions (app_name TEXT, user_id, id, create_time, update_time)"
    )
    connection.execute(
        "CREATE TABLE events (app_name TEXT, user_id, session_id, id, timestamp)"
    )
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
        ("blacki", "u", "s-v1", None, None),
    )
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        ("blacki", "u", "s-v1", None, 4.0),
    )
    connection.commit()
    connection.close()
    fallback = _load_snapshot_sync(event_db, "blacki")
    assert fallback.events[0].event_id == ""

    def fail_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("hidden")

    monkeypatch.setattr("blacki.dashboard.data.sqlite3.connect", fail_connect)
    assert _load_snapshot_sync(event_db, "blacki").warnings
    monkeypatch.undo()

    connection = sqlite3.connect(":memory:")
    assert (
        _select_table_rows(
            connection,
            table="sessions",
            columns=set(),
            where="1 = 1",
            parameters=(),
        )
        == []
    )
    connection.close()

    event = _EventRow(
        "edge",
        "u",
        "s",
        "i",
        4.0,
        {
            "content": {
                "parts": [
                    4,
                    {"function_call": {}},
                    {"function_response": {}},
                    {"function_call": {"name": 1}},
                    {"function_response": {"name": 1}},
                    {
                        "function_response": {
                            "name": "failed-tool",
                            "response": {"status": "failed"},
                        }
                    },
                    {
                        "function_response": {
                            "name": "tool",
                            "response": {"error": 1},
                        }
                    },
                ]
            }
        },
    )
    projected = _event_messages(event)
    assert projected[-1]["status"] == "error"
    latency_events = [
        _EventRow("one", "u", "s", "run", 1.0, {}),
        _EventRow("two", "u", "s", "run", 2.0, {}),
    ]
    assert _event_stats(latency_events)["latency_ms"]["max"] == 1000.0
    assert _activity_series([], since=None, until=2.0, window="all")


@pytest.mark.asyncio
async def test_missing_tables_and_trace_filter_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_is_file(_path: Path) -> bool:
        raise OSError("hidden")

    monkeypatch.setattr(Path, "is_file", fail_is_file)
    broken = _load_snapshot_sync(tmp_path / "broken.db", "blacki")
    monkeypatch.undo()
    assert broken.warnings

    path = tmp_path / "blacki-traces.log"
    path.write_text(
        json.dumps(
            {
                "name": "other",
                "context": {"trace_id": "other"},
                "status": "OK",
                "events": [None, {}, {"name": 3}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = DashboardStore(tmp_path / "missing.db", tmp_path)
    assert (await store.list_traces("ERROR", "", 10))["items"] == []
    assert (await store.list_traces(None, "", 10))["items"]
    assert (await store.list_traces(None, "needle", 10))["items"] == []
