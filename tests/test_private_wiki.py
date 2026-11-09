"""Private wiki behavior through SQLite and the real ADK runner."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types

from blacki.agent import create_agent, create_app
from blacki.runtime.adk import AdkRuntime, SessionLocator
from blacki.wiki.history import search_history
from blacki.wiki.storage import Source, WikiStore
from blacki.wiki.tools import (
    TOOL_NAMES,
    WikiTurn,
    review_reminder_owner,
    wiki_turn,
)


def test_conflicts_isolation_and_forgetting(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki.db")
    path = "projects/blacki.md"
    store.set_enabled(12, True)
    store.set_enabled(34, True)
    first = Source("tg:1", "Use SQLite.", "2026-11-09T01:00:00Z")
    second = Source("tg:2", "We use Postgres.", "2026-11-09T02:00:00Z")
    correction = Source("tg:3", "Actually, use SQLite.", "2026-11-09T03:00:00Z")
    assert store.write(12, path, "database", "Use SQLite.", first, 0, False) == "saved"
    assert store.read(34, "search", "SQLite")["pages"] == []
    assert (
        store.write(12, path, "database", "Use SQLite.", first, 0, False) == "unchanged"
    )
    assert (
        store.write(12, path, "database", "Use Postgres.", second, 0, False)
        == "stale_revision"
    )
    assert (
        store.write(12, path, "database", "Use Postgres.", second, 1, False)
        == "disputed"
    )
    assert store.read(12, "search", "SQLite")["pages"] == []
    assert "(disputed)" in store.read(12, "page", path)["markdown"]
    assert (
        store.write(12, path, "database", "Use SQLite.", correction, 2, False)
        == "saved"
    )
    assert store.read(12, "prefix", "projects/")["pages"] == [path]
    assert store.read(12, "source", "tg:3")["source"]["excerpt"] == correction.text
    store.set_enabled(12, False)
    revision, entries = store.inspect(12)
    assert entries == [(path, "database")]
    assert store.forget(12, path, "database", revision)
    assert "tg:3" in store.suppressed_sources(12)
    store.set_enabled(12, True)
    assert (
        store.write(12, path, "database", "Use SQLite.", first, 6, False) == "forgotten"
    )


def _call(name: str, index: int, args: dict[str, Any]) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"test-wiki-{index}",
                        name=name,
                        args=args,
                    )
                )
            ],
        )
    )


class WikiModel(BaseLlm):
    calls: int = 0

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        assert set(llm_request.tools_dict) >= TOOL_NAMES
        if self.calls == 1:
            yield _call(
                "wiki_read", 1, {"mode": "search", "value": "wiki-private-query"}
            )
            return
        replies = [
            part.function_response
            for content in llm_request.contents
            for part in content.parts or []
            if part.function_response
        ]
        if self.calls == 2:
            assert replies[-1].response == {
                "pages": [],
                "revision": 0,
                "truncated": False,
            }
            yield _call(
                "wiki_write",
                2,
                {
                    "path": "projects/blacki.md",
                    "fact": "storage",
                    "claim": "Store the wiki in SQLite.",
                    "source_ref": "current",
                    "revision": 0,
                },
            )
            return
        assert replies[-1].response == {"status": "saved"}
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Saved.")],
            )
        )


async def test_private_wiki_adk_events_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "1")
    db_path = tmp_path / "sessions.db"
    session_service = DatabaseSessionService(f"sqlite+aiosqlite:///{db_path}")
    store = WikiStore(tmp_path / "wiki.db")
    store.set_enabled(12, True)
    model = WikiModel(model="test")
    agent = create_agent(include_user_scoped_tools=True)
    agent.model = model
    runtime = AdkRuntime(session_service, agent_app=create_app(agent))
    locator = SessionLocator("telegram-chat-12", "telegram-chat-12")
    source = Source("tg:99", "Let's keep the wiki in SQLite.", "2026-11-09T01:00:00Z")
    try:
        with wiki_turn(WikiTurn(store, db_path, 12, locator.user_id, source)):
            response = await runtime.run_user_turn(
                locator=locator, message_text=source.text
            )
        assert response == "Saved."
        session = await runtime.get_or_create_session(locator=locator)
        persisted = str([event.model_dump() for event in session.events])
        assert "wiki-private-query" not in persisted
        assert "Store the wiki in SQLite." not in persisted
        assert store.read(12, "page", "projects/blacki.md")["markdown"].endswith(
            "[tg:99]"
        )
        matches = search_history(db_path, 12, "SQLite", "", "", None, set())
        assert [(item.ref, item.text, item.writable) for item in matches] == [
            ("tg:99", source.text, True)
        ]
        assert search_history(db_path, 34, "SQLite", "", "", None, set()) == []
    finally:
        await session_service.close()


async def test_review_grant_requires_private_owner(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki.db")
    source = Source("tg:1", "Schedule wiki review every Friday.", "2026-11-09")
    turn = WikiTurn(store, None, 12, "telegram-chat-12", source, request=source.text)
    with wiki_turn(turn):
        assert await review_reminder_owner(turn.locator) is None
        store.set_enabled(12, True)
        assert await review_reminder_owner("telegram-chat-34") is None
        assert await review_reminder_owner(turn.locator) == 12
    assert await review_reminder_owner(turn.locator) is None


class HistoryModel(BaseLlm):
    calls: int = 0
    expected_ref: str = "tg:45"

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        assert set(llm_request.tools_dict) == TOOL_NAMES
        if self.calls == 1:
            yield _call(
                "search_past_sessions",
                1,
                {
                    "query": "vault-backed",
                    "topic": 2,
                    "since": "2026-01-01",
                    "until": "2027-01-01",
                },
            )
            return
        replies = [
            part.function_response
            for content in llm_request.contents
            for part in content.parts or []
            if part.function_response
        ]
        assert replies[-1].response is not None
        matches = replies[-1].response["matches"]
        assert [match["ref"] for match in matches] == [self.expected_ref]
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Found [{self.expected_ref}].")],
            )
        )


async def test_authorized_reminder_searches_past_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime
    from unittest.mock import create_autospec

    from google.adk.events import Event

    from blacki.reminders.storage import Reminder
    from blacki.telegram import TelegramConfig
    from blacki.telegram.api import TelegramApiClient
    from blacki.telegram.bot import TelegramBot

    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "1")
    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    db_path = tmp_path / "sessions.db"
    session_service = DatabaseSessionService(f"sqlite+aiosqlite:///{db_path}")
    store = WikiStore(tmp_path / "wiki.db")
    store.set_enabled(12, True)
    agent = create_agent(include_user_scoped_tools=True)
    agent.model = HistoryModel(model="test")
    runtime = AdkRuntime(session_service, agent_app=create_app(agent))
    bot = TelegramBot(
        TelegramConfig.model_validate({"TELEGRAM_ENABLED": False}),
        runtime,
        wiki_store=store,
    )
    api = create_autospec(TelegramApiClient, instance=True, spec_set=True)
    bot._api = api
    try:
        locator = SessionLocator(
            "telegram-chat-12-thread-2", "telegram-chat-12-thread-2"
        )
        session = await runtime.get_or_create_session(locator=locator)
        timestamp = datetime(2026, 11, 9, tzinfo=UTC).timestamp()
        await session_service.append_event(
            session,
            Event(
                author="user",
                timestamp=timestamp,
                custom_metadata={"blacki_wiki": {"owner": 12, "message_id": 45}},
                content=types.Content(
                    role="user",
                    parts=[types.Part(text="Use a vault-backed deployment.")],
                ),
            ),
        )
        await runtime.create_next_session(locator=locator)
        reminder = Reminder(
            user_id="telegram-chat-12",
            wiki_owner=12,
            message="Review the deployment wiki.",
            trigger_time="2026-11-09T18:00:00Z",
            created_at="2026-11-09T12:00:00Z",
        )
        await bot.handle_scheduled_reminder(reminder)
        assert "Found" in api.send_message.call_args.kwargs["text"]
        store.set_enabled(12, False)
        await bot.handle_scheduled_reminder(reminder)
        assert "could not access memory" in api.send_message.call_args.kwargs["text"]
    finally:
        await session_service.close()


async def test_memory_controls_through_private_telegram(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import create_autospec

    from google.adk.sessions import InMemorySessionService

    from blacki.telegram import TelegramConfig
    from blacki.telegram.api import TelegramApiClient
    from blacki.telegram.bot import TelegramBot
    from blacki.telegram.types import Update

    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    store = WikiStore(tmp_path / "wiki.db")
    runtime = AdkRuntime(InMemorySessionService())
    bot = TelegramBot(TelegramConfig.model_validate({}), runtime, wiki_store=store)
    api = create_autospec(TelegramApiClient, instance=True, spec_set=True)
    bot._api = api

    async def send(
        text: str,
        message_id: int,
        chat_type: str = "private",
        *,
        forwarded: bool = False,
    ) -> None:
        update = Update.model_validate(
            {
                "update_id": message_id,
                "message": {
                    "message_id": message_id,
                    "date": "2026-11-09T01:00:00Z",
                    "chat": {"id": 12, "type": chat_type},
                    "from": {"id": 12, "first_name": "User"},
                    "text": text,
                    "forward_origin": {"type": "hidden_user"} if forwarded else None,
                },
            }
        )
        await bot._safe_handle_update(update)

    await send("/memory on", 1, "group")
    assert not store.enabled(12)
    await send("/memory on", 2)
    assert store.enabled(12)
    assert (
        store.write(
            12,
            "preferences/format.md",
            "tone",
            "Short replies.",
            Source("tg:3", "Use short replies.", "2026-11-09"),
            0,
            False,
        )
        == "saved"
    )
    await send("/memory forget preferences/format.md unknown", 3)
    assert api.send_message.call_args.kwargs["text"] == "No matching wiki claim."
    await send("/memory off", 4)
    assert not store.enabled(12)
    await send("/memory list", 5)
    assert api.send_message.call_args.kwargs["text"] == "preferences/format.md tone"
    await send("/memory forget preferences/format.md tone", 6)
    confirmation = api.send_message.call_args.kwargs["text"]
    token = confirmation.split("/memory confirm ", 1)[1].split()[0]
    await send("/memory confirm wrong", 7)
    assert store.inspect(12)[1]
    await send(f"/memory confirm {token}", 8)
    assert store.inspect(12)[1] == []
    assert "tg:3" in store.suppressed_sources(12)
    await send("/memory on", 9, forwarded=True)
    assert not store.enabled(12)
    await send("/memory", 10)
    assert "Use /memory on" in api.send_message.call_args.kwargs["text"]
    bot.wiki_store = WikiStore(tmp_path)
    await send("/memory on", 11)
    assert api.send_message.call_args.kwargs["text"] == "Memory storage is unavailable."


async def test_tool_authorization_and_input_limits(tmp_path: Path) -> None:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions import InMemorySessionService, Session
    from google.adk.tools import ToolContext

    from blacki.wiki.tools import (
        WikiToolset,
        search_past_sessions,
        wiki_read,
        wiki_write,
    )

    context = ToolContext(
        InvocationContext(
            invocation_id="test",
            session=Session(id="fake", app_name="blacki", user_id="telegram-chat-12"),
            session_service=InMemorySessionService(),
        ),
        function_call_id="read",
    )
    store = WikiStore(tmp_path / "wiki.db")
    path = "preferences/editor.md"
    assert await WikiToolset().get_tools() == []
    assert await wiki_read(context) == {"status": "unavailable"}
    assert await wiki_write(path, "editor", "Use vim.", "current", 0) == {
        "status": "unavailable"
    }
    assert await search_past_sessions(context, "vim") == {"status": "unavailable"}
    store.set_enabled(12, True)
    turn = WikiTurn(
        store,
        tmp_path / "missing.db",
        12,
        "telegram-chat-12",
        Source("tg:1", "Use vim.", "2026-11-09"),
    )
    with wiki_turn(turn):
        assert await wiki_read(context, "unsafe") == {"status": "invalid_request"}
        assert await wiki_write(path, "editor", "Use vim.", "forged", 0) == {
            "status": "invalid_source"
        }
        assert await wiki_write(path, "editor", "", "current", 0) == {
            "status": "invalid_request"
        }
        assert await wiki_write(
            path, "editor", "Use vim.", "current", 0, ["x" * 161]
        ) == {"status": "invalid_request"}
        assert await wiki_write(path, "editor", "My password is x.", "current", 0) == {
            "status": "ineligible_content"
        }
        assert await wiki_write("../unsafe.md", "editor", "Use vim.", "current", 0) == {
            "status": "invalid_path_or_fact"
        }
        assert await search_past_sessions(context, "") == {"status": "invalid_request"}
        assert await search_past_sessions(context, "vim", since="bad") == {
            "status": "unavailable"
        }
        assert await search_past_sessions(context, "vim") == {"status": "unavailable"}
        assert await search_past_sessions(context, "vim", since="20261109") == {
            "status": "invalid_date"
        }
        for n in range(16):
            context.function_call_id = f"read-{n}"
            assert await wiki_read(context) == {"status": "ok"}
        assert await wiki_read(context) == {"status": "limit_reached"}
        store.set_enabled(12, False)
        assert await WikiToolset().get_tools() == []


def test_links_and_suppression_survive_correction(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki.db")
    store.set_enabled(12, True)
    path = "projects/deploy.md"
    first = Source("tg:1", "Use provider A.", "2026-11-08")
    second = Source("tg:2", "Actually, use provider B.", "2026-11-09")
    fresh = Source("tg:3", "Remember we use provider B.", "2026-11-10")
    assert store.write(12, path, "provider", first.text, first, 0, False) == "saved"
    assert (
        store.write(
            12, "projects/main.md", "provider", first.text, first, 1, False, (path,)
        )
        == "saved"
    )
    assert (
        "[[projects/deploy.md]]"
        in store.read(12, "page", "projects/main.md")["markdown"]
    )
    assert store.write(12, path, "provider", second.text, second, 2, False) == "saved"
    assert not store.forget(12, path, "provider", 0)
    assert not store.forget(12, path, "missing", 3)
    assert store.forget(12, path, "provider", 3)
    assert store.suppressed_sources(12) == {"tg:1", "tg:2"}
    assert store.read(12, "source", "tg:1")["source"] is None
    assert (
        "[[projects/deploy.md]]"
        not in store.read(12, "page", "projects/main.md")["markdown"]
    )
    assert store.write(12, path, "provider", fresh.text, fresh, 4, True) == "saved"


def test_legacy_history_is_read_only_and_filters_internal_events(
    tmp_path: Path,
) -> None:
    import json
    import sqlite3
    from contextlib import closing

    path = tmp_path / "sessions.db"
    owner = "telegram-chat-12"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE sessions (app_name, user_id, id, state);
            CREATE TABLE events (app_name, user_id, session_id, id,
                                 timestamp, author, content);
            """)
        private = json.dumps(
            {"telegram_chat_type": "private", "telegram_chat_id": "12"}
        )
        cases = [
            (
                owner,
                f"{owner}-v1",
                private,
                "one",
                "user",
                {"role": "user", "parts": [{"text": "legacy SQLite note"}]},
            ),
            (
                owner,
                f"{owner}-v1",
                private,
                "synthetic",
                "user",
                {"role": "user", "parts": [{"text": "[Scheduled Event] SQLite"}]},
            ),
            (
                owner,
                f"{owner}-v1",
                private,
                "tool",
                "blacki",
                {"role": "model", "parts": [{"function_call": {"name": "SQLite"}}]},
            ),
            (
                owner,
                f"{owner}-v1",
                private,
                "reply",
                "blacki",
                {"role": "model", "parts": [{"text": "SQLite confirmed"}]},
            ),
            (
                owner,
                f"{owner}-v2",
                "{}",
                "unverified",
                "user",
                {"role": "user", "parts": [{"text": "SQLite unverified"}]},
            ),
            (
                owner,
                f"{owner}-foo",
                private,
                "unknown",
                "user",
                {"role": "user", "parts": [{"text": "SQLite invalid session"}]},
            ),
            (
                f"{owner}-thread-foo",
                f"{owner}-thread-foo-v1",
                private,
                "bad-topic",
                "user",
                {"role": "user", "parts": [{"text": "SQLite invalid topic"}]},
            ),
        ]
        db.executemany(
            "INSERT INTO sessions VALUES ('blacki', ?, ?, ?)",
            {(user, session, state) for user, session, state, *_ in cases},
        )
        db.executemany(
            "INSERT INTO events VALUES ('blacki', ?, ?, ?, '2026-11-09', ?, ?)",
            [
                (user, session, ref, role, json.dumps(content))
                for user, session, _, ref, role, content in cases
            ],
        )
        db.execute(
            "INSERT INTO events VALUES "
            "('blacki', ?, ?, 'broken', '2026-11-09', 'user', 'bad')",
            (owner, f"{owner}-v1"),
        )
    hits = search_history(path, 12, "sqlite", "", "", None, set())
    assert {hit.text for hit in hits} == {"legacy SQLite note", "SQLite confirmed"}
    assert all(not hit.writable for hit in hits)
    assert search_history(path, 12, "sqlite", "", "", 2, set()) == []
    assert (
        search_history(path, 12, "sqlite", "", "", None, {hit.ref for hit in hits})
        == []
    )
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE events")
    with pytest.raises(ValueError, match="unsupported event schema"):
        search_history(path, 12, "sqlite", "", "", None, set())


async def test_broken_wiki_storage_fails_closed(tmp_path: Path) -> None:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions import InMemorySessionService, Session
    from google.adk.tools import ToolContext

    from blacki.wiki.tools import WikiToolset, wiki_origin, wiki_read, wiki_write

    directory = WikiStore(tmp_path)
    with wiki_turn(WikiTurn(directory, None, 12, "telegram-chat-12")):
        assert await WikiToolset().get_tools() == []
        assert wiki_origin() == {"blacki_wiki": {"owner": 12}}

    store = WikiStore(tmp_path / "wiki.db")
    assert store.read(12, "search", "") == {"status": "disabled"}
    source = Source("tg:1", "Use vim.", "2026-11-09")
    assert (
        store.write(12, "preferences.md", "editor", source.text, source, 0, False)
        == "disabled"
    )
    store.set_enabled(12, True)
    with store.connect() as db:
        db.execute("DROP TABLE wiki_claims")
        db.execute("CREATE TABLE wiki_claims (unrecognized_schema INTEGER)")
    invocation = InvocationContext(
        invocation_id="broken",
        session=Session(id="test", app_name="blacki", user_id="telegram-chat-12"),
        session_service=InMemorySessionService(),
    )
    context = ToolContext(invocation, function_call_id="read")
    with wiki_turn(WikiTurn(store, None, 12, "telegram-chat-12", source)):
        assert await wiki_read(context, "page", "preferences.md") == {
            "status": "unavailable"
        }
        assert await wiki_write(
            "preferences.md", "editor", source.text, "current", 0
        ) == {"status": "unavailable"}
        await WikiToolset().close()


async def test_wiki_policy_filters_and_bounds_reminder_requests(tmp_path: Path) -> None:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    from google.adk.sessions import InMemorySessionService, Session
    from google.adk.tools import FunctionTool, ToolContext

    from blacki.wiki.tools import WikiPolicyPlugin, wiki_read

    store = WikiStore(tmp_path / "wiki.db")
    store.set_enabled(12, True)
    invocation = InvocationContext(
        invocation_id="review",
        session=Session(id="test", app_name="blacki", user_id="telegram-chat-12"),
        session_service=InMemorySessionService(),
    )
    context = ToolContext(invocation)
    plugin = WikiPolicyPlugin("wiki")
    await plugin.before_run_callback(invocation_context=invocation)
    event = Event(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="Preparing to review."),
                types.Part(
                    function_call=types.FunctionCall(
                        name="wiki_read",
                        id="read",
                        args={"value": "private query"},
                    )
                ),
            ],
        )
    )
    redacted = await plugin.on_event_callback(
        invocation_context=invocation, event=event
    )
    assert redacted and redacted.get_function_calls()[0].args == {}
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="read",
                            name="wiki_read",
                            response={"status": "ok"},
                        )
                    )
                ],
            ),
        ]
    )
    turn = WikiTurn(store, None, 12, "telegram-chat-12", review=True)
    turn.results["read"] = {"oversize": "a" * 12000}
    with wiki_turn(turn):
        assert await wiki_read(context) == {"status": "limit_reached"}
        await plugin.before_model_callback(
            callback_context=CallbackContext(invocation),
            llm_request=request,
        )
        assert "oversize" not in str(request.contents)
        tool = FunctionTool(lambda: "blocked")
        result = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=context
        )
        assert result == {"status": "unavailable"}
        store.set_enabled(12, False)
        await plugin.before_model_callback(
            callback_context=CallbackContext(invocation),
            llm_request=LlmRequest(),
        )


async def test_history_excludes_synthetic_and_caps_results(tmp_path: Path) -> None:
    import json
    import sqlite3
    from contextlib import closing

    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions import InMemorySessionService, Session
    from google.adk.tools import ToolContext

    from blacki.wiki.tools import search_past_sessions

    path = tmp_path / "sessions.db"
    locator = "telegram-chat-12"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE sessions (app_name, user_id, id, state);
            CREATE TABLE events (app_name, user_id, session_id, id,
                                 timestamp, event_data);
            """)
        db.execute(
            "INSERT INTO sessions VALUES ('blacki', ?, ?, '{}')",
            (locator, f"{locator}-v1"),
        )
        content = {"role": "user", "parts": [{"text": "Project discussion"}]}
        events = [
            {
                "author": "user",
                "content": content,
                "custom_metadata": {"blacki_wiki": {"owner": 12, "message_id": index}},
            }
            for index in range(10)
        ] + [
            {
                "author": "user",
                "content": content,
                "custom_metadata": {"blacki_wiki": {"kind": "synthetic"}},
            },
            {
                "author": "user",
                "content": content,
                "partial": True,
                "custom_metadata": {"blacki_wiki": {"owner": 12, "message_id": 80}},
            },
        ]
        db.executemany(
            "INSERT INTO events VALUES ('blacki', ?, ?, ?, '2026-11-09', ?)",
            [
                (locator, f"{locator}-v1", index, json.dumps(event))
                for index, event in enumerate(events)
            ],
        )
    results = search_history(path, 12, "Project", "", "", None, set())
    assert len(results) == 8
    assert all(result.writable and result.ref != "tg:80" for result in results)
    store = WikiStore(tmp_path / "wiki.db")
    store.set_enabled(12, True)
    context = ToolContext(
        InvocationContext(
            invocation_id="history",
            session=Session(id="current", app_name="blacki", user_id=locator),
            session_service=InMemorySessionService(),
        ),
        function_call_id="search",
    )
    turn = WikiTurn(store, path, 12, locator)
    turn.sources.update(
        (f"tg:{index}", Source(f"tg:{index}", "", "2026-11-09")) for index in range(64)
    )
    with wiki_turn(turn):
        assert await search_past_sessions(context, "Project") == {
            "status": "limit_reached"
        }


async def test_existing_reminder_schema_and_in_memory_history(tmp_path: Path) -> None:
    import asyncio

    from blacki.reminders.storage import SqliteReminderStorage
    from blacki.storage.sqlite import create_connection

    db = await create_connection(tmp_path / "tools.db")
    memory_session_service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        lock = asyncio.Lock()
        await SqliteReminderStorage(db, lock).initialize()
        # A new process can initialize the same schema without re-adding columns.
        await SqliteReminderStorage(db, lock).initialize()
        assert AdkRuntime(memory_session_service).wiki_history_path() is None
    finally:
        await db.close()
        await memory_session_service.close()
