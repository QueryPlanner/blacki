"""Read-only access to the owner's Telegram session history."""
# ruff: noqa: S608

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_MAX_MATCHES = 8
_SESSION_PAGE_SIZE = 20
_TRANSCRIPT_PAGE_CHARS = 8_000
_TRANSCRIPT_PAGE_MESSAGES = 30


@dataclass(frozen=True)
class HistoryMatch:
    """One user or assistant text message that matched a past-session search."""

    ref: str
    text: str
    date: str
    role: str
    session: str


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def search_history(
    path: Path,
    owner: int,
    query: str,
    since: str,
    until: str,
    topic: int | None,
    exclude: set[str],
) -> list[HistoryMatch]:
    """Return plain-text chat messages for owner that contain query.

    Only events whose parts are all text and whose author and role are a
    user message or a Blacki reply can match. Function calls and function
    responses, including wiki tool traffic, are never indexed. Refs in exclude,
    such as the message being answered, are skipped.
    """
    prefix = f"telegram-chat-{owner}"
    locator = re.compile(re.escape(prefix) + r"(?:-thread-[0-9]+)?")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        fields = _event_fields(db)
        # Only interpolate the two schema-dependent, constant column lists.
        rows = db.execute(
            f"""
            SELECT e.id, e.session_id, e.timestamp, e.user_id, s.state, {fields}
            FROM events e JOIN sessions s ON s.app_name=e.app_name
                AND s.user_id=e.user_id AND s.id=e.session_id
            WHERE e.app_name='blacki'
                AND (e.user_id=? OR e.user_id GLOB ?)
                AND (?='' OR date(e.timestamp)>=?)
                AND (?='' OR date(e.timestamp)<=?)
            ORDER BY e.timestamp DESC, e.id DESC LIMIT 2000
            """,
            (prefix, prefix + "-thread-*", since, since, until, until),
        )
        found: list[HistoryMatch] = []
        for row in rows:
            if not locator.fullmatch(row["user_id"]) or not re.fullmatch(
                re.escape(row["user_id"]) + r"-v[0-9]+", row["session_id"]
            ):
                continue
            if topic is not None and row["user_id"] != f"{prefix}-thread-{topic}":
                continue
            item = _message(row, owner)
            if item is None:
                continue
            ref, message = item.ref, item.text
            if ref in exclude or ref in {item.ref for item in found}:
                continue
            pos = message.casefold().find(query.casefold())
            if pos < 0:
                continue
            # Long messages stay searchable around the match.
            start = max(0, pos - 150)
            found.append(
                HistoryMatch(
                    ref,
                    message[start : start + 600],
                    str(row["timestamp"]),
                    item.role,
                    item.session,
                )
            )
            if len(found) == _MAX_MATCHES:
                break
        return found


def _event_fields(db: sqlite3.Connection) -> str:
    columns = {row[1] for row in db.execute("PRAGMA table_info(events)")}
    if "event_data" in columns:
        return "e.event_data, NULL AS author, NULL AS content"
    if {"author", "content"} <= columns:
        return "NULL AS event_data, e.author, e.content"
    raise ValueError("unsupported event schema")


def _private_session(value: Any, owner: int) -> bool:
    try:
        state = _object(value)
    except (ValueError, TypeError):
        return False
    return state.get("telegram_chat_type") == "private" and str(
        state.get("telegram_chat_id")
    ) == str(owner)


def _message(
    row: sqlite3.Row, owner: int, *, include_media: bool = False
) -> HistoryMatch | None:
    try:
        event = _object(row["event_data"])
        origin = _object(_object(event.get("custom_metadata")).get("blacki_wiki"))
        content = _object(event.get("content") or row["content"])
    except (ValueError, TypeError):
        return None
    verified = origin.get("owner") == owner
    if origin and not verified:
        return None
    if not verified and not _private_session(row["state"], owner):
        return None
    role = content.get("role")
    author = event.get("author") or row["author"]
    if (role, author) not in (("user", "user"), ("model", "blacki")):
        return None
    actions = event.get("actions") or {}
    if event.get("partial") or (
        isinstance(actions, dict) and actions.get("compaction")
    ):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    text_parts: list[str] = []
    for part in parts:
        if (
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and not part.get("thought")
        ):
            text_parts.append(part["text"])
        elif not include_media or (
            isinstance(part, dict)
            and (part.get("function_call") or part.get("function_response"))
        ):
            return None
    message = "\n".join(text_parts) if include_media else " ".join(text_parts).strip()
    if not message.strip() or message.lstrip().startswith("[Scheduled Event]"):
        return None
    msg_id = origin.get("message_id")
    # Replies share the run's metadata, so only user messages get a
    # message ref. Everything else is keyed by its event ID.
    ref = (
        f"tg:{msg_id}"
        if role == "user" and verified and isinstance(msg_id, int)
        else "adk:" + hashlib.sha256(str(row["id"]).encode()).hexdigest()[:24]
    )
    return HistoryMatch(
        ref, message, str(row["timestamp"]), str(role), row["session_id"]
    )


def _session_locator(session_id: str, owner: int) -> re.Match[str] | None:
    return re.fullmatch(
        rf"(telegram-chat-{owner}(?:-thread-([0-9]+))?)-v[0-9]+", session_id
    )


def _session_rows(
    db: sqlite3.Connection, user_id: str, session_id: str, offset: int = 0
) -> sqlite3.Cursor:
    fields = _event_fields(db)
    return db.execute(
        f"""
        SELECT e.id, e.session_id, e.timestamp, s.state, {fields}
        FROM events e JOIN sessions s ON s.app_name=e.app_name
            AND s.user_id=e.user_id AND s.id=e.session_id
        WHERE e.app_name='blacki' AND e.user_id=? AND e.session_id=?
        ORDER BY e.timestamp, e.id LIMIT -1 OFFSET ?
        """,
        (user_id, session_id, offset),
    )


def list_history(
    path: Path, owner: int, topic: int | None, cursor: str
) -> dict[str, Any]:
    """List private Telegram sessions, including older versions after a reset."""
    prefix = f"telegram-chat-{owner}"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT s.id, s.user_id, s.state,
                MIN(e.timestamp) AS first_at, MAX(e.timestamp) AS last_at
            FROM sessions s LEFT JOIN events e ON s.app_name=e.app_name
                AND s.user_id=e.user_id AND s.id=e.session_id
            WHERE s.app_name='blacki' AND (s.user_id=? OR s.user_id GLOB ?)
                AND s.id>?
            GROUP BY s.id, s.user_id, s.state ORDER BY s.id
            """,
            (prefix, prefix + "-thread-*", cursor),
        )
        sessions: list[dict[str, Any]] = []
        for row in rows:
            locator = _session_locator(row["id"], owner)
            if (
                locator is None
                or locator[1] != row["user_id"]
                or not _private_session(row["state"], owner)
            ):
                continue
            thread = int(locator[2]) if locator[2] is not None else None
            if topic is not None and topic != thread:
                continue
            if len(sessions) == _SESSION_PAGE_SIZE:
                return {
                    "status": "ok",
                    "sessions": sessions,
                    "next_cursor": sessions[-1]["session_id"],
                }
            preview = ""
            for event in _session_rows(db, row["user_id"], row["id"]):
                message = _message(event, owner, include_media=True)
                if message is not None:
                    preview = message.text[:160]
                    break
            sessions.append(
                {
                    "session_id": row["id"],
                    "topic": thread,
                    "first_at": row["first_at"],
                    "last_at": row["last_at"],
                    "preview": preview,
                }
            )
    return {"status": "ok", "sessions": sessions, "next_cursor": None}


def read_history(
    path: Path, owner: int, session_id: str, row_offset: int, char_offset: int
) -> dict[str, Any]:
    """Read original user and assistant text, including captions, in event order."""
    locator = _session_locator(session_id, owner)
    if locator is None:
        return {"status": "not_found"}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        session = db.execute(
            "SELECT state FROM sessions WHERE app_name='blacki' AND user_id=? AND id=?",
            (locator[1], session_id),
        ).fetchone()
        if session is None or not _private_session(session["state"], owner):
            return {"status": "not_found"}
        messages: list[dict[str, Any]] = []
        result: dict[str, Any] = {
            "status": "ok",
            "session_id": session_id,
            "messages": messages,
            "next_cursor": None,
        }
        if _event_fields(db).startswith("e.event_data"):
            compacted = db.execute(
                """
                SELECT 1 FROM events e WHERE e.app_name='blacki' AND e.user_id=?
                    AND e.session_id=? AND json_extract(
                        CASE WHEN json_valid(e.event_data) THEN e.event_data END,
                        '$.actions.compaction'
                    ) IS NOT NULL LIMIT 1
                """,
                (locator[1], session_id),
            ).fetchone()
            if compacted is not None:
                result["history_note"] = (
                    "This session was compacted. Only original text still stored "
                    "is included. If only a summary remains, the original text "
                    "for that portion is unavailable."
                )
        remaining = _TRANSCRIPT_PAGE_CHARS
        for index, row in enumerate(
            _session_rows(db, locator[1], session_id, row_offset), start=row_offset
        ):
            message = _message(row, owner, include_media=True)
            if message is None:
                continue
            start = char_offset if index == row_offset else 0
            if remaining == 0 or len(messages) == _TRANSCRIPT_PAGE_MESSAGES:
                result["next_cursor"] = f"{index}:{start}"
                break
            text = message.text[start:]
            if not text:
                continue
            messages.append({**asdict(message), "text": text[:remaining]})
            if len(text) > remaining:
                result["next_cursor"] = f"{index}:{start + remaining}"
                break
            remaining -= len(text)
    return result
