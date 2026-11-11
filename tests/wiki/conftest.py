"""Shared fixtures for the private wiki tests."""

import json
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import ToolContext

from blacki.wiki.storage import WikiStore

OWNER = 12
LOCATOR = "telegram-chat-12"
PATH = "people/ana.md"
PRIVATE_STATE = '{"telegram_chat_type": "private", "telegram_chat_id": "12"}'


@pytest.fixture
def store(tmp_path: Path) -> WikiStore:
    """A wiki database with memory enabled for the test owner."""
    wiki = WikiStore(tmp_path / "wiki.db")
    wiki.set_enabled(OWNER, True)
    return wiki


@pytest.fixture
def context() -> ToolContext:
    """A real ADK tool context for the owner's private chat."""
    invocation = InvocationContext(
        invocation_id="test",
        session=Session(id="test", app_name="blacki", user_id=LOCATOR),
        session_service=InMemorySessionService(),
    )
    return ToolContext(invocation, function_call_id="call")


@pytest.fixture
def make_history(tmp_path: Path) -> Callable[[Iterable[dict[str, Any]]], Path]:
    """Build a Telegram ADK events database from event_data dictionaries.

    Every event belongs to one private chat session for the test owner.
    """

    def build(events: Iterable[dict[str, Any]]) -> Path:
        path = tmp_path / "sessions.db"
        session = f"{LOCATOR}-v1"
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE sessions (app_name, user_id, id, state);
                CREATE TABLE events (app_name, user_id, session_id, id,
                                     timestamp, event_data);
                """)
            db.execute(
                "INSERT INTO sessions VALUES ('blacki', ?, ?, ?)",
                (LOCATOR, session, PRIVATE_STATE),
            )
            db.executemany(
                "INSERT INTO events VALUES ('blacki', ?, ?, ?, '2026-11-09', ?)",
                [
                    (LOCATOR, session, str(index), json.dumps(event))
                    for index, event in enumerate(events)
                ],
            )
        return path

    return build
