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
PROMPT = "Use the captured instructions.\n" * 300 + "End of recorded prompt."


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
    index: int, attributes: dict[str, object], parent: int | None = None
) -> dict[str, object]:
    return {
        "name": "generate_content",
        "context": {"trace_id": TRACE, "span_id": f"{index:016x}"},
        "parent_id": f"{parent:016x}" if parent is not None else None,
        "start_time": 1730000000 + index,
        "attributes": attributes,
    }


def _old_prompt(text: str, **attributes: object) -> dict[str, object]:
    return {
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": text,
        "llm.input_messages.1.message.content": "Raw user message",
        "llm.tools.0.tool.json_schema": json.dumps(
            {
                "name": "example_tool",
                "description": "Read example records.",
                "parameters": {"description": "Internal schema details"},
            }
        ),
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
            _span(1, {"gcp.vertex.agent.invocation_id": "inv-a", "user.id": USER}),
            _span(2, _old_prompt(PROMPT), 1),
            _span(3, _old_prompt("Foreign user", **{"user.id": OTHER}), 1),
            _span(4, _old_prompt("Wrong session", **{"session.id": "session-v2"}), 1),
            _span(5, _old_prompt("Conflict", **{"blacki.invocation_id": "inv-b"}), 1),
            _span(6, _old_prompt("Collision", **{"blacki.invocation_id": "collision"})),
            _span(7, _old_prompt("Sibling with no ancestry")),
            _span(8, _old_prompt("Duplicated span"), 1),
            _span(8, _old_prompt("Duplicated span"), 1),
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

    selector = {"trace_id": TRACE, "span_id": span_id}
    detail = client.get("/dashboard/api/context", params=params | selector)
    capture = detail.json()["capture"]
    assert capture["system_prompt"] == PROMPT
    assert capture["tools"] == [
        {"name": "example_tool", "description": "Read example records."}
    ]
    assert "Raw user message" not in detail.text
    assert "Internal schema details" not in detail.text
    assert client.get(
        "/dashboard/api/context", params=params | selector | {"user_id": OTHER}
    ).json() == {"status": "capture_missing"}
    assert client.get(
        "/dashboard/api/context", params=params | selector | {"session_id": "absent"}
    ).json() == {"status": "session_missing"}
    assert (
        PROMPT
        not in client.get("/dashboard/api/trace", params={"trace_id": TRACE}).text
    )


def test_structured_capture_and_incomplete_tool_list(
    client: TestClient, files: Path
) -> None:
    _write_spans(
        files,
        [
            _span(
                1,
                {
                    "blacki.invocation_id": "inv-a",
                    "gen_ai.system_instructions": json.dumps(
                        [
                            {"type": "text", "content": "System rule."},
                            {"type": "text", "content": "password: example-secret"},
                        ]
                    ),
                    "gen_ai.tool.definitions": json.dumps(
                        [
                            {"name": "read", "description": "Read one record."},
                            {"name": "partial"},
                        ]
                    ),
                },
            )
        ],
    )
    response = client.get(
        "/dashboard/api/context",
        params={
            "user_id": USER,
            "session_id": "session-v1",
            "trace_id": TRACE,
            "span_id": f"{1:016x}",
        },
    )
    capture = response.json()["capture"]
    assert capture["system_prompt"] == "System rule.\npassword: [REDACTED]"
    assert capture["tools"] == [{"name": "read", "description": "Read one record."}]
    assert capture["partial_tools"] is True


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
    for path in ("context", "wiki"):
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


def test_wiki_markdown_is_returned_as_data_and_redacted(
    client: TestClient, files: Path
) -> None:
    WikiStore(files / "wiki.db").edit(
        12,
        "notes.md",
        "<script>alert('x')</script>\n- note: sk-or-v1-" + "a" * 24,
        "",
        read=False,
    )
    data = client.get("/dashboard/api/wiki", params={"user_id": USER}).json()
    markdown = data["selected"]["markdown"]
    assert "<script>alert('x')</script>" in markdown
    assert "sk-or-v1-" not in markdown
    assert "[REDACTED]" in markdown


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


def test_trace_tail_can_read_multipart_prompt_but_rejects_broken_ancestry(
    client: TestClient, files: Path
) -> None:
    params = {"user_id": USER, "session_id": "session-v1"}
    _write_spans(
        files,
        [
            {"context": {"trace_id": "invalid", "span_id": "invalid"}},
            _span(
                1,
                {
                    "blacki.invocation_id": "inv-a",
                    "llm.model_name": "example-model",
                    "llm.input_messages.0.message.role": "system",
                    "llm.input_messages.0.message.contents.1.message_content.text": (
                        "Last."
                    ),
                    "llm.input_messages.0.message.contents.0.message_content.text": (
                        "First."
                    ),
                    "llm.tools.1.tool.json_schema": "not JSON",
                    "gen_ai.system_instructions": "[]",
                },
                99,  # Older parent outside the retained trace file.
            ),
            _span(
                2, _old_prompt("Cyclic span", **{"blacki.invocation_id": "inv-a"}), 2
            ),
            _span(3, _old_prompt("Unusable ID", **{"blacki.invocation_id": 7})),
            _span(4, _old_prompt("<elided>", **{"blacki.invocation_id": "inv-a"})),
        ],
    )
    listing = client.get("/dashboard/api/context", params=params).json()
    assert listing["total"] == 1
    assert listing["items"][0]["model"] == "example-model"
    detail = client.get(
        "/dashboard/api/context",
        params=params | {"trace_id": TRACE, "span_id": f"{1:016x}"},
    ).json()["capture"]
    assert detail["system_prompt"] == "First.\nLast."
    assert detail["partial_tools"] is True
    assert detail["tools"] == []


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
    _write_spans(files, [_span(1, _old_prompt(PROMPT))])
    assert client.get(url, params=params).json() == {"status": "unavailable"}


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
