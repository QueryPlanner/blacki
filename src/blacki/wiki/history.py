"""Bounded, read-only search of Telegram's ADK SQLite sessions."""
# ruff: noqa: S608

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_MATCHES = 8


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
        columns = {row[1] for row in db.execute("PRAGMA table_info(events)")}
        if "event_data" in columns:
            fields = "e.event_data, NULL AS author, NULL AS content"
        elif {"author", "content"} <= columns:
            fields = "NULL AS event_data, e.author, e.content"
        else:
            raise ValueError("unsupported event schema")
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
            try:
                event = _object(row["event_data"])
                state = _object(row["state"])
                origin = _object(
                    _object(event.get("custom_metadata")).get("blacki_wiki")
                )
                content = _object(event.get("content") or row["content"])
            except (ValueError, TypeError):
                continue
            verified = origin.get("owner") == owner
            if origin and not verified:
                continue
            if not verified and (
                state.get("telegram_chat_type") != "private"
                or str(state.get("telegram_chat_id")) != str(owner)
            ):
                continue
            role = content.get("role")
            author = event.get("author") or row["author"]
            if (role, author) not in (("user", "user"), ("model", "blacki")):
                continue
            actions = event.get("actions") or {}
            if event.get("partial") or (
                isinstance(actions, dict) and actions.get("compaction")
            ):
                continue
            parts = content.get("parts")
            if (
                not isinstance(parts, list)
                or not parts
                or any(
                    not isinstance(part, dict)
                    or not isinstance(part.get("text"), str)
                    or part.get("thought")
                    for part in parts
                )
            ):
                continue
            message = " ".join(part["text"] for part in parts).strip()
            if not message or message.startswith("[Scheduled Event]"):
                continue
            msg_id = origin.get("message_id")
            # Replies share the run's metadata, so only user messages get a
            # message ref. Everything else is keyed by its event ID.
            ref = (
                f"tg:{msg_id}"
                if role == "user" and verified and isinstance(msg_id, int)
                else "adk:" + hashlib.sha256(str(row["id"]).encode()).hexdigest()[:24]
            )
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
                    str(role),
                    row["session_id"],
                )
            )
            if len(found) == _MAX_MATCHES:
                break
        return found
