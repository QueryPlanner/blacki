"""Integration tests for the dashboard's private data views."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from blacki.dashboard.routes import create_dashboard_router
from blacki.utils.config import ServerEnv
from blacki.wiki.storage import WikiStore

USER = "telegram-chat-12"
OTHER = "telegram-chat-34"
TRACE = "a" * 32
PROMPT = "Use the captured instructions.\n" * 300 + "password: example-secret"


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("blacki.dashboard.routes.get_log_dir", lambda: tmp_path)
    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "true")
    state = tmp_path / ".adk"
    state.mkdir()
    with closing(sqlite3.connect(state / "sessions.db")) as db, db:
        db.executescript(
            """
            CREATE TABLE sessions (id TEXT, app_name TEXT, user_id TEXT);
            CREATE TABLE events (
                id TEXT, app_name TEXT, user_id TEXT, session_id TEXT,
                invocation_id TEXT
            );
            """
        )
        db.executemany(
            "INSERT INTO sessions VALUES ('session-v1','blacki',?)", [(USER,), (OTHER,)]
        )
        db.executemany(
            "INSERT INTO events VALUES (?,'blacki',?,'session-v1',?)",
            [
                ("first", USER, "inv-a"),
                ("second", OTHER, "inv-b"),
                ("collision1", USER, "collision"),
                ("collision2", OTHER, "collision"),
            ],
        )
    return tmp_path


def _client(path: Path) -> TestClient:
    env = ServerEnv.model_validate(
        {
            "AGENT_NAME": "test",
            "AGENT_DIR": str(path),
            "SQLITE_PATH": str(path / "tools.db"),
        }
    )
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    app.include_router(create_dashboard_router(env))
    return TestClient(app, base_url="http://localhost:8080")


@pytest.fixture
def client(files: Path) -> Iterator[TestClient]:
    with _client(files) as browser:
        yield browser


def _span(
    index: int,
    attributes: dict[str, object],
    parent: int | None = None,
    *,
    name: str = "generate_content model-x",
    status: str = "UNSET",
) -> dict[str, object]:
    """Build a span in the OpenTelemetry SDK's JSON format."""
    return {
        "name": name,
        "context": {"trace_id": "0x" + TRACE, "span_id": f"0x{index:016x}"},
        "kind": "SpanKind.INTERNAL",
        "parent_id": f"0x{parent:016x}" if parent is not None else None,
        "start_time": f"2026-08-15T10:00:{index:02d}.000000Z",
        "end_time": f"2026-08-15T10:00:{index:02d}.250000Z",
        "status": {"status_code": status},
        "attributes": attributes,
        "events": [],
    }


def _model_call(prompt: str = PROMPT, **attributes: object) -> dict[str, object]:
    """Return the experimental GenAI attributes ADK writes on a model call."""
    return {
        "gen_ai.request.model": "model-x",
        "gen_ai.system_instructions": json.dumps([{"type": "text", "content": prompt}]),
        "gen_ai.tool.definitions": json.dumps(
            [
                {
                    "name": "add",
                    "description": "Add two numbers.",
                    "parameters": {"type": "object"},
                    "type": "function",
                }
            ]
        ),
        "gen_ai.input.messages": json.dumps(
            [
                {"role": "user", "parts": [{"type": "text", "content": "1+2?"}]},
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": "call-1",
                            "name": "add",
                            "arguments": {"a": 1, "b": 2},
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": "call-1",
                            "response": {"result": 3},
                        }
                    ],
                },
            ]
        ),
        "gen_ai.output.messages": json.dumps(
            [
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "text", "content": "It is 3."},
                        {
                            "type": "tool_call",
                            "id": "call-2",
                            "name": "add",
                            "arguments": {"a": 3, "b": 4},
                        },
                    ],
                    "finish_reason": "stop",
                }
            ]
        ),
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 3,
        **attributes,
    }


def _write_spans(path: Path, spans: list[dict[str, object]]) -> None:
    (path / "blacki-traces.log").write_text(
        "\n".join(map(json.dumps, spans)) + "\n", encoding="utf-8"
    )


def test_context_matches_invocation_ancestry_and_omits_unverified_spans(
    client: TestClient, files: Path
) -> None:
    _write_spans(
        files,
        [
            {"context": {"trace_id": "invalid", "span_id": "invalid"}},
            _span(1, {"gcp.vertex.agent.invocation_id": "inv-a", "user.id": USER}),
            _span(2, _model_call(), 1),
            _span(3, _model_call(**{"user.id": OTHER}), 1),
            _span(4, _model_call(**{"session.id": "session-v2"}), 1),
            _span(5, _model_call(**{"blacki.invocation_id": "inv-b"}), 1),
            _span(6, _model_call(**{"blacki.invocation_id": "collision"})),
            _span(7, _model_call()),
            _span(8, _model_call(), 1),
            _span(8, _model_call(), 1),
            _span(9, _model_call(**{"blacki.invocation_id": "inv-a"}), 9),
            _span(10, {**_model_call(), "blacki.invocation_id": 7}),
            _span(11, {"blacki.invocation_id": "inv-a"}, name="call_llm"),
        ],
    )
    params = {"user_id": USER, "session_id": "session-v1"}
    listing = client.get("/dashboard/api/context", params=params)
    assert listing.status_code == 200
    assert listing.headers["cache-control"] == "no-store"
    body = listing.json()
    assert body["total"] == 1
    assert PROMPT not in listing.text
    span_id = f"{2:016x}"
    assert body["items"][0]["span_id"] == span_id
    assert body["items"][0]["trace_id"] == TRACE
    assert body["items"][0]["model"] == "model-x"

    selector = {"trace_id": TRACE, "span_id": span_id}
    assert client.get(
        "/dashboard/api/context", params=params | selector | {"user_id": OTHER}
    ).json() == {"status": "capture_missing"}
    assert client.get(
        "/dashboard/api/context", params=params | selector | {"session_id": "absent"}
    ).json() == {"status": "session_missing"}


def test_context_capture_shows_the_whole_model_call(
    client: TestClient, files: Path
) -> None:
    """Every tab comes from the experimental GenAI attributes, unfiltered."""
    _write_spans(
        files,
        [
            _span(1, {"gcp.vertex.agent.invocation_id": "inv-a"}),
            _span(2, _model_call(), 1),
            _span(
                3,
                {"gen_ai.tool.call.id": "call-2", "gen_ai.tool.name": "add"},
                2,
                name="execute_tool add",
                status="OK",
            ),
            _span(
                4,
                _model_call(
                    **{
                        "gen_ai.input.messages": json.dumps(
                            [
                                {
                                    "role": "user",
                                    "parts": [
                                        {
                                            "type": "tool_call_response",
                                            "id": "call-2",
                                            "response": {"result": 7},
                                        }
                                    ],
                                }
                            ]
                        )
                    }
                ),
                1,
            ),
        ],
    )
    detail = client.get(
        "/dashboard/api/context",
        params={
            "user_id": USER,
            "session_id": "session-v1",
            "trace_id": TRACE,
            "span_id": f"{2:016x}",
        },
    )
    capture = detail.json()["capture"]
    assert capture["system_prompt"] == PROMPT
    assert capture["tools"] == [
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {"type": "object"},
        }
    ]
    assert capture["partial_tools"] is False
    assert [message["role"] for message in capture["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert capture["messages"][2]["parts"] == [
        {
            "type": "function_response",
            "name": "add",
            "response": {"result": 3},
            "call_id": "call-1",
        }
    ]
    assert capture["response"]["text"] == "It is 3."
    assert capture["response"]["finish_reason"] == "stop"
    assert capture["response"]["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert capture["tool_calls"] == [
        {
            "name": "add",
            "call_id": "call-2",
            "args": {"a": 3, "b": 4},
            "response": {"result": 7},
            "status": "OK",
            "duration_ms": 250,
        }
    ]
    raw = json.loads(capture["raw_span"])
    assert raw["attributes"]["gen_ai.output.messages"][0]["finish_reason"] == "stop"
    assert "password: example-secret" in capture["system_prompt"]


def test_trace_views_return_full_content(client: TestClient, files: Path) -> None:
    _write_spans(files, [_span(1, _model_call())])
    traces = client.get("/dashboard/api/traces").json()
    assert traces["items"][0]["trace_id"] == TRACE
    trace = client.get("/dashboard/api/trace", params={"trace_id": TRACE})
    assert trace.status_code == 200
    span = trace.json()["spans"][0]
    assert span["attributes"]["gen_ai.system_instructions"] == json.dumps(
        [{"type": "text", "content": PROMPT}]
    )


def test_context_distinguishes_missing_trace_from_unreadable_session_store(
    client: TestClient, files: Path
) -> None:
    url = "/dashboard/api/context"
    params = {"user_id": USER, "session_id": "session-v1"}
    assert client.get(url, params=params).json() == {"status": "trace_missing"}

    sessions = files / ".adk" / "sessions.db"
    sessions.rename(sessions.with_suffix(".bak"))
    assert client.get(url, params=params).json() == {"status": "unavailable"}

    sessions.write_text("not a database", encoding="utf-8")
    _write_spans(files, [_span(1, _model_call())])
    assert client.get(url, params=params).json() == {"status": "unavailable"}


@pytest.mark.parametrize(
    "host",
    (
        "localhost:8080",
        "127.0.0.1",
        "[::1]:8080",
        "100.64.0.1:8080",
        "[fd7a:115c:a1e0::1]",
        "blacki.example-tailnet.ts.net",
    ),
)
def test_private_routes_available_by_default_on_supported_hosts(
    client: TestClient, host: str
) -> None:
    headers = {"Host": host, "Sec-Fetch-Site": "same-origin"}
    prompt = client.get(
        "/dashboard/api/context",
        params={"user_id": USER, "session_id": "session-v1"},
        headers=headers,
    )
    wiki = client.get("/dashboard/api/wiki", params={"user_id": USER}, headers=headers)
    assert prompt.status_code == wiki.status_code == 200
    assert prompt.json() == {"status": "trace_missing"}
    assert wiki.json() == {"status": "missing"}


def test_private_routes_enforce_host_and_browser_boundary(client: TestClient) -> None:
    for path in ("context", "wiki", "logs", "traces", "trace"):
        url = "/dashboard/api/" + path
        for headers in (
            {"Host": "foreign.example", "X-Forwarded-Host": "localhost"},
            {"Origin": "https://foreign.example"},
            {"Origin": "null"},
            {"Host": "[malformed"},
            {"Host": "localhost:0"},
            {"Host": "localhost/path"},
            {"Host": "user@localhost"},
            {"Host": "blacki.example-tailnet.ts.net.foreign.example"},
            {"Host": "blacki.-example-tailnet.ts.net"},
            {"Host": "100.128.0.1"},
            {"Host": "[fd7a:115c:a1e1::1]"},
            {"Sec-Fetch-Site": "cross-site"},
            {"Sec-Fetch-Site": "same-site"},
            [("Host", "localhost"), ("Host", "foreign.example")],
        ):
            assert client.get(url, headers=headers).status_code == 403
    assert (
        client.get("/dashboard/api/context", params={"user_id": USER}).status_code
        == 422
    )
    assert (
        client.get(
            "/dashboard/api/context",
            params={"user_id": USER, "session_id": "session-v1", "trace_id": TRACE},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/dashboard/api/wiki", params={"user_id": USER, "page": "0"}
        ).status_code
        == 422
    )


def test_wiki_lists_owner_pages_and_reads_the_selected_page(
    client: TestClient, files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "/dashboard/api/wiki"
    path = files / "wiki.db"
    params = {"user_id": USER}
    assert client.get(url, params=params).json() == {"status": "missing"}
    assert not path.exists()

    wiki = WikiStore(path)
    wiki.edit(12, "profile/preferences.md", "Use bullet points.", "", read=False)
    wiki.edit(34, "profile/private.md", "Use prose.", "", read=False)
    original = path.read_bytes()

    result = client.get(url, params=params)
    data = result.json()
    assert data["pages"] == ["profile/preferences.md"]
    assert data["page"] == 0
    assert data["selected"]["path"] == "profile/preferences.md"
    assert data["selected"]["markdown"] == "Use bullet points."
    assert data["selected"]["chars"] == len("Use bullet points.")
    assert "profile/private.md" not in result.text
    assert "Use prose." not in result.text
    assert client.get(
        url, params=params | {"page": "1", "revision": str(data["revision"])}
    ).json() == {"status": "page_missing"}
    assert client.get(url, params={"user_id": USER + "-thread-4"}).json() == {
        "status": "unsupported_user"
    }
    assert path.read_bytes() == original

    wiki.set_enabled(12, False)
    assert client.get(url, params=params).json() == {"status": "wiki_disabled"}
    wiki.set_enabled(12, True)
    assert client.get(
        url, params=params | {"page": "0", "revision": str(data["revision"])}
    ).json() == {"status": "wiki_changed"}
    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "false")
    assert client.get(url, params=params).json() == {"status": "wiki_disabled"}


def test_wiki_markdown_is_returned_as_data(client: TestClient, files: Path) -> None:
    secret = "- note: sk-or-v1-" + "a" * 24
    WikiStore(files / "wiki.db").edit(
        12,
        "notes.md",
        "<script>alert('x')</script>\n" + secret,
        "",
        read=False,
    )
    data = client.get("/dashboard/api/wiki", params={"user_id": USER}).json()
    markdown = data["selected"]["markdown"]
    assert "<script>alert('x')</script>" in markdown
    assert secret in markdown


def test_wiki_from_a_claim_based_database_reads_as_empty(
    client: TestClient, files: Path
) -> None:
    with closing(sqlite3.connect(files / "wiki.db")) as db, db:
        db.executescript(
            """
            CREATE TABLE wiki_owners (
                owner INTEGER PRIMARY KEY, enabled INTEGER NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE wiki_claims (
                owner INTEGER NOT NULL, path TEXT NOT NULL, fact TEXT NOT NULL,
                claim TEXT NOT NULL, source TEXT NOT NULL, excerpt TEXT NOT NULL,
                source_date TEXT NOT NULL, disputed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (owner, path, fact, source)
            );
            INSERT INTO wiki_owners VALUES (12, 1, 3);
            INSERT INTO wiki_claims VALUES
                (12, 'old.md', 'k', 'Old claim.', 'tg:1', 'Old claim.',
                 '2026-01-01', 0);
            """
        )
    data = client.get("/dashboard/api/wiki", params={"user_id": USER}).json()
    assert data == {"status": "empty"}


def test_wiki_distinguishes_empty_unknown_owner_and_unreadable_data(
    client: TestClient, files: Path
) -> None:
    url = "/dashboard/api/wiki"
    path = files / "wiki.db"
    WikiStore(path).set_enabled(34, True)
    assert client.get(url, params={"user_id": USER}).json() == {"status": "empty"}
    assert client.get(url, params={"user_id": "telegram-chat-99"}).json() == {
        "status": "user_missing"
    }
    path.write_text("not a database", encoding="utf-8")
    assert client.get(url, params={"user_id": USER}).json() == {"status": "unavailable"}


def test_parts_map_every_genai_part_type() -> None:
    from blacki.dashboard.private_data import _normalize_part

    names = {"call-1": "add"}
    assert _normalize_part({"type": "text", "content": "hi"}, names) == {
        "type": "text",
        "text": "hi",
    }
    assert _normalize_part({"type": "reasoning", "content": 3}, names) == {
        "type": "thought",
        "text": "",
    }
    assert _normalize_part({"type": "tool_call", "id": 5}, names) == {
        "type": "function_call",
        "name": "",
        "args": {},
        "call_id": "",
    }
    assert _normalize_part({"type": "tool_call_response", "id": "call-1"}, names) == {
        "type": "function_response",
        "name": "add",
        "response": {},
        "call_id": "call-1",
    }
    assert _normalize_part({"type": "blob", "mime_type": "image/png"}, names) == {
        "type": "inline_data",
        "mime_type": "image/png",
    }
    assert _normalize_part(
        {"type": "file_data", "mime_type": "text/plain", "uri": "file:///x"}, names
    ) == {"type": "file_data", "mime_type": "text/plain", "file_uri": "file:///x"}
    assert _normalize_part({"type": "audio", "z": 1, "a": 2}, names) == {
        "type": "other",
        "keys": ["a", "type", "z"],
    }


@pytest.mark.parametrize("value", [None, "not JSON", "{}", json.dumps([1, None])])
def test_messages_and_response_tolerate_unusable_values(value: object) -> None:
    from blacki.dashboard.private_data import (
        _messages,
        _prompt,
        _response,
        _tool_calls,
    )

    attributes = {
        "gen_ai.input.messages": value,
        "gen_ai.output.messages": value,
        "gen_ai.system_instructions": value,
    }
    messages = _messages(attributes)
    assert all(message["parts"] == [] for message in messages)
    assert _prompt(attributes) == ""
    assert _tool_calls(attributes, {}, {}) == []
    response = _response(attributes)
    assert response is None or response == {"text": "", "parts": []}


def test_response_without_usage_or_finish_reason() -> None:
    from blacki.dashboard.private_data import _response

    output = [{"role": 7, "parts": [{"type": "text", "content": "ok"}]}]
    assert _response({"gen_ai.output.messages": json.dumps(output)}) == {
        "text": "ok",
        "parts": [{"type": "text", "text": "ok"}],
    }


def test_tools_mark_unnamed_definitions_as_partial() -> None:
    from blacki.dashboard.private_data import _tools

    definitions = [{"name": "read", "description": 3}, {"name": ""}, "bad"]
    assert _tools({"gen_ai.tool.definitions": json.dumps(definitions)}) == (
        [{"name": "read", "description": "", "parameters": {}}],
        True,
    )


def test_tool_call_without_span_or_result_and_unusable_records() -> None:
    from blacki.dashboard.private_data import (
        _duration_ms,
        _tool_calls,
        _tool_results,
        _tool_span_index,
    )

    output = [
        {"parts": [{"type": "text", "content": "x"}, {"type": "tool_call", "id": "a"}]}
    ]
    assert _tool_calls({"gen_ai.output.messages": json.dumps(output)}, {}, {}) == [
        {
            "name": "",
            "call_id": "a",
            "args": {},
            "response": None,
            "status": None,
            "duration_ms": None,
        }
    ]
    valid = {"attributes": {"gen_ai.tool.call.id": "call-1"}}
    assert _tool_span_index(
        [{}, {"attributes": {"gen_ai.tool.call.id": ""}}, valid]
    ) == {"call-1": valid}
    messages = [{"parts": [{"type": "tool_call_response"}, {"type": "text"}]}]
    assert (
        _tool_results(
            [{}, {"attributes": {"gen_ai.input.messages": json.dumps(messages)}}]
        )
        == {}
    )
    assert _duration_ms(None) is None
    assert _duration_ms({"start_time": 1}) is None
    assert _duration_ms({"start_time": 2, "end_time": 1}) is None


def test_raw_span_decodes_json_attributes_only() -> None:
    from blacki.dashboard.private_data import _raw_span

    raw = json.loads(
        _raw_span({"name": "s", "attributes": {"a": '{"x": 1}', "b": "plain", "c": 2}})
    )
    assert raw == {"name": "s", "attributes": {"a": {"x": 1}, "b": "plain", "c": 2}}


def test_invocation_rejects_long_ancestry() -> None:
    from blacki.dashboard.private_data import _invocation

    spans = {
        (TRACE, f"{index:016x}"): {
            "attributes": {},
            "parent_id": f"0x{index + 1:016x}",
        }
        for index in range(70)
    }
    assert _invocation((TRACE, f"{0:016x}"), spans, USER, "session-v1") is None
    orphan = {
        (TRACE, f"{1:016x}"): {
            "attributes": {"blacki.invocation_id": "inv-a"},
            "parent_id": f"0x{99:016x}",
        }
    }
    assert _invocation((TRACE, f"{1:016x}"), orphan, USER, "session-v1") == "inv-a"
