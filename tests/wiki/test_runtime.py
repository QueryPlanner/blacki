"""Private wiki through the real ADK runner, session events, and history search."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types

from blacki.agent import create_agent, create_app
from blacki.runtime.adk import AdkRuntime, SessionLocator
from blacki.wiki.history import list_history, read_history, search_history
from blacki.wiki.storage import WikiStore
from blacki.wiki.tools import TOOL_NAMES, WikiTurn, wiki_turn

PAGE = "projects/blacki.md"


def _call(name: str, index: int, args: dict[str, Any]) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"wiki-{index}", name=name, args=args
                    )
                )
            ],
        )
    )


def _last_response(request: LlmRequest) -> dict[str, Any]:
    replies = [
        part.function_response
        for content in request.contents
        for part in content.parts or []
        if part.function_response
    ]
    response = replies[-1].response
    assert response is not None
    return response


class ScriptedWikiModel(BaseLlm):
    """Creates a page, reads it in full, then appends in the same turn."""

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
                "wiki_write",
                1,
                {
                    "path": PAGE,
                    "new_text": "# Blacki\n- Store the wiki in SQLite.",
                    "old_text": "",
                },
            )
        elif self.calls == 2:
            assert _last_response(llm_request)["status"] == "saved"
            yield _call("wiki_read", 2, {"path": PAGE})
        elif self.calls == 3:
            read = _last_response(llm_request)
            assert read["status"] == "ok"
            assert "Store the wiki in SQLite." in read["markdown"]
            yield _call(
                "wiki_write",
                3,
                {
                    "path": PAGE,
                    "new_text": "- Owner prefers short replies.",
                    "old_text": "",
                },
            )
        else:
            assert _last_response(llm_request)["status"] == "saved"
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="Saved.")])
            )


async def test_a_turn_reads_and_writes_through_adk_and_history_ignores_tool_traffic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "1")
    db_path = tmp_path / "sessions.db"
    session_service = DatabaseSessionService(f"sqlite+aiosqlite:///{db_path}")
    store = WikiStore(tmp_path / "wiki.db")
    store.set_enabled(12, True)
    agent = create_agent(include_user_scoped_tools=True)
    agent.model = ScriptedWikiModel(model="test")
    runtime = AdkRuntime(session_service, agent_app=create_app(agent))
    locator = SessionLocator("telegram-chat-12", "telegram-chat-12")
    text = "Let's keep the wiki in SQLite."
    try:
        with wiki_turn(WikiTurn(store, db_path, 12, locator.user_id, message_id=99)):
            response = await runtime.run_user_turn(
                locator=locator,
                message_text=text,
                state={"telegram_chat_type": "private", "telegram_chat_id": "12"},
            )
        assert response == "Saved."

        page = store.get_page(12, PAGE)
        assert page is not None
        assert page.markdown == (
            "# Blacki\n- Store the wiki in SQLite.\n- Owner prefers short replies."
        )

        # Tool arguments and results are kept as ordinary session events...
        session = await runtime.get_or_create_session(locator=locator)
        persisted = str([event.model_dump() for event in session.events])
        assert "Store the wiki in SQLite." in persisted

        # ...but history search reads only plain user and reply text.
        matches = search_history(db_path, 12, "SQLite", "", "", None, set())
        assert [(match.ref, match.text) for match in matches] == [("tg:99", text)]
        assert search_history(db_path, 12, "Store the wiki", "", "", None, set()) == []
        assert search_history(db_path, 34, "SQLite", "", "", None, set()) == []
        reset = await runtime.create_next_session(
            locator=locator,
            state={"telegram_chat_type": "private", "telegram_chat_id": "12"},
        )
        assert reset.id != session.id
        listed = list_history(db_path, 12, None, "")
        assert {item["session_id"] for item in listed["sessions"]} == {
            session.id,
            reset.id,
        }
        transcript = read_history(db_path, 12, session.id, 0, 0)
        assert [item["text"] for item in transcript["messages"]] == [text, "Saved."]
        assert transcript["next_cursor"] is None
    finally:
        await session_service.close()


async def test_runtime_exposes_only_a_file_backed_sqlite_history_path(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    file_service = DatabaseSessionService(f"sqlite+aiosqlite:///{db_path}")
    memory_service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        assert AdkRuntime(file_service).wiki_history_path() == db_path
        assert AdkRuntime(memory_service).wiki_history_path() is None
        assert AdkRuntime(InMemorySessionService()).wiki_history_path() is None
    finally:
        await file_service.close()
        await memory_service.close()
