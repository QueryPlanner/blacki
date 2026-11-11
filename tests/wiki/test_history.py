"""Past-session search over Telegram ADK events."""

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from blacki.wiki.history import list_history, read_history, search_history

OWNER_CHAT = "telegram-chat-12"


def _text(
    text: str, *, role: str = "user", author: str = "user", **extra: Any
) -> dict[str, Any]:
    return {
        "author": author,
        "content": {"role": role, "parts": [{"text": text}]},
        **extra,
    }


def _origin(owner: int, message_id: int) -> dict[str, Any]:
    return {
        "custom_metadata": {"blacki_wiki": {"owner": owner, "message_id": message_id}}
    }


def test_only_plain_user_and_reply_text_is_searchable(
    make_history: Callable[[Any], Path],
) -> None:
    path = make_history(
        [
            _text("vault user note", **_origin(12, 1)),
            _text("vault reply", role="model", author="blacki", **_origin(12, 1)),
            {
                "author": "blacki",
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": "wiki_write",
                                "args": {"new_text": "vault call"},
                            }
                        }
                    ],
                },
            },
            {
                "author": "blacki",
                "content": {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": "wiki_read",
                                "response": {"markdown": "vault response"},
                            }
                        }
                    ],
                },
            },
            {
                "author": "user",
                "content": {
                    "role": "user",
                    "parts": [
                        {"function_response": {"name": "wiki_read", "response": {}}}
                    ],
                },
            },
            {
                "author": "blacki",
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "vault mixed"},
                        {"function_call": {"name": "wiki_write", "args": {}}},
                    ],
                },
            },
            _text("[Scheduled Event] vault reminder"),
            _text("vault partial", partial=True),
            {
                "author": "blacki",
                "content": {
                    "role": "model",
                    "parts": [{"text": "vault thought", "thought": True}],
                },
            },
            _text(
                "vault synthetic",
                custom_metadata={"blacki_wiki": {"kind": "synthetic"}},
            ),
            _text("vault other owner", **_origin(34, 2)),
        ]
    )
    hits = search_history(path, 12, "vault", "", "", None, set())
    assert sorted(hit.text for hit in hits) == ["vault reply", "vault user note"]


def test_user_messages_keep_a_message_ref_and_replies_do_not(
    make_history: Callable[[Any], Path],
) -> None:
    path = make_history(
        [
            _text("vault question", **_origin(12, 45)),
            _text("vault answer", role="model", author="blacki", **_origin(12, 45)),
        ]
    )
    refs = {
        hit.role: hit.ref
        for hit in search_history(path, 12, "vault", "", "", None, set())
    }
    assert refs["user"] == "tg:45"
    assert refs["model"].startswith("adk:")


def test_excluded_refs_are_skipped(make_history: Callable[[Any], Path]) -> None:
    path = make_history([_text("vault question", **_origin(12, 45))])
    assert search_history(path, 12, "vault", "", "", None, {"tg:45"}) == []


def test_history_caps_matches_and_excludes_partial_and_synthetic_events(
    make_history: Callable[[Any], Path],
) -> None:
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
    path = make_history(events)
    results = search_history(path, 12, "Project", "", "", None, set())
    assert len(results) == 8
    assert all(result.ref != "tg:80" for result in results)


def test_topic_filter_limits_matches_to_one_thread(
    make_history: Callable[[Any], Path],
) -> None:
    path = make_history([_text("vault question", **_origin(12, 45))])
    assert search_history(path, 12, "vault", "", "", 2, set()) == []


def test_legacy_history_is_read_only_and_filters_internal_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.db"
    owner = OWNER_CHAT
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
    assert read_history(path, 12, f"{owner}-v1", 0, 0)["status"] == "ok"
    hits = search_history(path, 12, "sqlite", "", "", None, set())
    assert {hit.text for hit in hits} == {"legacy SQLite note", "SQLite confirmed"}
    assert search_history(path, 12, "sqlite", "", "", 2, set()) == []
    assert (
        search_history(path, 12, "sqlite", "", "", None, {hit.ref for hit in hits})
        == []
    )
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE events")
    with pytest.raises(ValueError, match="unsupported event schema"):
        search_history(path, 12, "sqlite", "", "", None, set())


def test_list_history_includes_previous_versions_and_paginates(
    make_history: Callable[[Any], Path],
) -> None:
    path = make_history([_text("first session")])
    topic = f"{OWNER_CHAT}-thread-9"
    with closing(sqlite3.connect(path)) as db, db:
        state = db.execute("SELECT state FROM sessions").fetchone()[0]
        sessions = [(OWNER_CHAT, f"{OWNER_CHAT}-v{i}", state) for i in range(2, 22)]
        sessions += [
            (topic, f"{topic}-v1", state),
            (topic, f"{topic}-invalid", state),
            (topic, f"{OWNER_CHAT}-v30", state),
            (topic, f"{topic}-v2", "invalid JSON"),
            ("telegram-chat-123", "telegram-chat-123-v1", state),
        ]
        db.executemany("INSERT INTO sessions VALUES ('blacki', ?, ?, ?)", sessions)

    first = list_history(path, 12, None, "")
    second = list_history(path, 12, None, first["next_cursor"])
    listed = first["sessions"] + second["sessions"]
    assert len(first["sessions"]) == 20
    assert second["next_cursor"] is None
    assert {item["session_id"] for item in listed} == {
        *(f"{OWNER_CHAT}-v{i}" for i in range(1, 22)),
        f"{topic}-v1",
    }
    original = next(item for item in listed if item["session_id"] == f"{OWNER_CHAT}-v1")
    assert original["preview"] == "first session"
    assert original["first_at"] == original["last_at"] == "2026-11-09"
    assert [item["topic"] for item in list_history(path, 12, 9, "")["sessions"]] == [9]


def test_read_history_reassembles_original_messages_and_captions(
    make_history: Callable[[Any], Path],
) -> None:
    long_message = "x" * 16_001
    path = make_history(
        [
            _text(long_message),
            {
                "author": "user",
                "content": {
                    "role": "user",
                    "parts": [
                        {"text": "a caption"},
                        {"inline_data": {"data": "ignored"}},
                    ],
                },
            },
            _text("the reply", role="model", author="blacki"),
            _text("other owner's text", **_origin(23, 1)),
            _text("summary", actions={"compaction": {"start_timestamp": 1}}),
        ]
    )
    session = f"{OWNER_CHAT}-v1"
    first = read_history(path, 12, session, 0, 0)
    assert "compacted" in first["history_note"]
    assert read_history(path, 23, session, 0, 0) == {"status": "not_found"}
    assert read_history(path, 12, f"{OWNER_CHAT}-v9", 0, 0) == {"status": "not_found"}
    assert first["next_cursor"] == "0:8000"
    assert first["messages"][0]["text"] == long_message[:8000]
    second = read_history(path, 12, session, 0, 8000)
    assert second["next_cursor"] == "0:16000"
    third = read_history(path, 12, session, 0, 16000)
    assert third["next_cursor"] is None
    assert [item["text"] for item in third["messages"]] == [
        "x",
        "a caption",
        "the reply",
    ]


def test_read_history_pages_by_message_count_and_skips_invalid_events(
    make_history: Callable[[Any], Path],
) -> None:
    path = make_history(
        [{"author": "user", "content": {"role": "user"}}, *[_text("hello")] * 31]
    )
    session = f"{OWNER_CHAT}-v1"
    assert list_history(path, 12, None, "")["sessions"][0]["preview"] == "hello"
    first = read_history(path, 12, session, 0, 0)
    assert len(first["messages"]) == 30
    offset, chars = map(int, first["next_cursor"].split(":"))
    last = read_history(path, 12, session, offset, chars)
    assert len(last["messages"]) == 1
    assert last["next_cursor"] is None
    assert read_history(path, 12, session, offset, 5)["messages"] == []
