"""Read-only prompt and wiki data for the private operator dashboard."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from ..wiki import wiki_configured
from .data import (
    _as_dict,
    _iso_timestamp,
    _otel_id,
    _trace_time,
    scan_traces_sync,
)
from .models import JsonObject

_TRACE_ID = re.compile(r"[0-9a-f]{32}")
_SPAN_ID = re.compile(r"[0-9a-f]{16}")
_INVOCATION_KEYS = ("blacki.invocation_id", "gcp.vertex.agent.invocation_id")
_SESSION_KEYS = (
    "blacki.session_id",
    "gcp.vertex.agent.session_id",
    "gen_ai.conversation.id",
    "session.id",
)
_USER_KEYS = ("blacki.user_id", "user.id")
_DIRECT_USER = re.compile(r"telegram-chat-([1-9][0-9]{0,18})")

# Experimental OpenTelemetry GenAI attributes written by ADK on model calls.
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"
TOOL_CALL_ID = "gen_ai.tool.call.id"
_USAGE_KEYS = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.reasoning.output_tokens",
)


def valid_capture_id(trace_id: str, span_id: str) -> bool:
    """Accept lowercase hex trace and span IDs."""
    return bool(_TRACE_ID.fullmatch(trace_id) and _SPAN_ID.fullmatch(span_id))


def _decode(value: Any) -> Any:
    """Decode an attribute that the SDK stored as a JSON string."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _as_list(value: Any) -> list[Any]:
    decoded = _decode(value)
    return decoded if isinstance(decoded, list) else []


def _prompt(attributes: JsonObject) -> str:
    """Join the text of the system instructions sent with a model call."""
    return "\n".join(
        part["content"]
        for part in _as_list(attributes.get(SYSTEM_INSTRUCTIONS))
        if isinstance(part, Mapping) and isinstance(part.get("content"), str)
    )


def _tools(attributes: JsonObject) -> tuple[list[JsonObject], bool]:
    """Return the tool definitions and whether any could not be read."""
    tools: list[JsonObject] = []
    partial = False
    for definition in _as_list(attributes.get(TOOL_DEFINITIONS)):
        item = _as_dict(definition)
        name = item.get("name")
        if not isinstance(name, str) or not name:
            partial = True
            continue
        description = item.get("description")
        tools.append(
            {
                "name": name,
                "description": description if isinstance(description, str) else "",
                "parameters": item.get("parameters") or {},
            }
        )
    return tools, partial


def _part_id(part: Mapping[str, Any]) -> str:
    value = part.get("id")
    return value if isinstance(value, str) else ""


def _normalize_part(part: Mapping[str, Any], names: Mapping[str, str]) -> JsonObject:
    """Map one GenAI message part onto the inspector's part shape."""
    kind = part.get("type")
    if kind in {"text", "reasoning"}:
        content = part.get("content")
        return {
            "type": "thought" if kind == "reasoning" else "text",
            "text": content if isinstance(content, str) else "",
        }
    call_id = _part_id(part)
    if kind == "tool_call":
        return {
            "type": "function_call",
            "name": str(part.get("name") or ""),
            "args": part.get("arguments") or {},
            "call_id": call_id,
        }
    if kind == "tool_call_response":
        return {
            "type": "function_response",
            "name": names.get(call_id, ""),
            "response": part.get("response") or {},
            "call_id": call_id,
        }
    if kind == "blob":
        return {"type": "inline_data", "mime_type": str(part.get("mime_type") or "")}
    if kind == "file_data":
        return {
            "type": "file_data",
            "mime_type": str(part.get("mime_type") or ""),
            "file_uri": str(part.get("uri") or ""),
        }
    return {"type": "other", "keys": sorted(str(key) for key in part)}


def _tool_names(messages: Iterable[Any]) -> dict[str, str]:
    """Map tool call IDs to tool names so results can be labeled."""
    names: dict[str, str] = {}
    for message in messages:
        for part in _as_list(_as_dict(message).get("parts")):
            item = _as_dict(part)
            if item.get("type") == "tool_call" and (call_id := _part_id(item)):
                names[call_id] = str(item.get("name") or "")
    return names


def _normalize_message(message: Any, names: Mapping[str, str]) -> JsonObject:
    item = _as_dict(message)
    parts = [
        _normalize_part(part, names)
        for part in _as_list(item.get("parts"))
        if isinstance(part, Mapping)
    ]
    role = item.get("role")
    return {
        "role": role if isinstance(role, str) else "",
        "text": "\n".join(
            part["text"]
            for part in parts
            if part["type"] in {"text", "thought"} and part["text"]
        ),
        "parts": parts,
    }


def _messages(attributes: JsonObject) -> list[JsonObject]:
    """Return the conversation sent to the model, oldest first."""
    messages = _as_list(attributes.get(INPUT_MESSAGES))
    names = _tool_names(messages)
    return [_normalize_message(message, names) for message in messages]


def _response(attributes: JsonObject) -> JsonObject | None:
    """Return the model's output messages, finish reasons, and token usage."""
    outputs = _as_list(attributes.get(OUTPUT_MESSAGES))
    if not outputs:
        return None
    names = _tool_names(outputs)
    parts: list[JsonObject] = []
    reasons: list[str] = []
    for output in outputs:
        message = _normalize_message(output, names)
        parts.extend(message["parts"])
        reason = _as_dict(output).get("finish_reason")
        if isinstance(reason, str) and reason:
            reasons.append(reason)
    result: JsonObject = {
        "text": "\n".join(
            part["text"]
            for part in parts
            if part["type"] in {"text", "thought"} and part["text"]
        ),
        "parts": parts,
    }
    if reasons:
        result["finish_reason"] = ", ".join(reasons)
    usage = {
        key.removeprefix("gen_ai.usage.").replace(".", "_"): attributes[key]
        for key in _USAGE_KEYS
        if key in attributes
    }
    if usage:
        result["usage"] = usage
    return result


def _duration_ms(record: JsonObject | None) -> int | None:
    """Return a span's wall-clock duration in whole milliseconds."""
    if record is None:
        return None
    start = _trace_time(record, "start_time")
    end = _trace_time(record, "end_time")
    if start is None or end is None or end < start:
        return None
    return round((end - start) * 1000)


def _tool_span_index(records: Iterable[JsonObject]) -> dict[str, JsonObject]:
    """Index tool execution spans by their tool call ID."""
    index: dict[str, JsonObject] = {}
    for record in records:
        call_id = _as_dict(record.get("attributes")).get(TOOL_CALL_ID)
        if isinstance(call_id, str) and call_id:
            index[call_id] = record
    return index


def _tool_results(records: Iterable[JsonObject]) -> dict[str, Any]:
    """Index tool results by call ID from the messages sent to later calls."""
    results: dict[str, Any] = {}
    for record in records:
        for message in _as_list(_as_dict(record.get("attributes")).get(INPUT_MESSAGES)):
            for part in _as_list(_as_dict(message).get("parts")):
                item = _as_dict(part)
                if item.get("type") == "tool_call_response" and (
                    call_id := _part_id(item)
                ):
                    results[call_id] = item.get("response")
    return results


def _tool_calls(
    attributes: JsonObject,
    spans: Mapping[str, JsonObject],
    results: Mapping[str, Any],
) -> list[JsonObject]:
    """Pair each tool call in a response with its execution span and result."""
    calls: list[JsonObject] = []
    for output in _as_list(attributes.get(OUTPUT_MESSAGES)):
        for part in _as_list(_as_dict(output).get("parts")):
            item = _as_dict(part)
            if item.get("type") != "tool_call":
                continue
            call_id = _part_id(item)
            record = spans.get(call_id)
            status = (
                _as_dict(record.get("status")).get("status_code") if record else None
            )
            calls.append(
                {
                    "name": str(item.get("name") or ""),
                    "call_id": call_id,
                    "args": item.get("arguments") or {},
                    "response": results.get(call_id),
                    "status": status if isinstance(status, str) else None,
                    "duration_ms": _duration_ms(record),
                }
            )
    return calls


def _raw_span(record: JsonObject) -> str:
    """Render the span exactly as it was recorded, with JSON attributes decoded."""
    attributes = {
        key: _decode(value) for key, value in _as_dict(record.get("attributes")).items()
    }
    return json.dumps(
        {**record, "attributes": attributes},
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _span_key(record: JsonObject) -> tuple[str, str] | None:
    ctx = _as_dict(record.get("context"))
    trace, span = _otel_id(ctx.get("trace_id")), _otel_id(ctx.get("span_id"))
    if (
        isinstance(trace, str)
        and isinstance(span, str)
        and valid_capture_id(trace, span)
    ):
        return trace, span
    return None


def _invocation(
    key: tuple[str, str],
    spans: dict[tuple[str, str], JsonObject],
    user_id: str,
    session_id: str,
) -> str | None:
    trace_id, span_id = key
    seen: set[str] = set()
    invocation_ids: set[str] = set()
    while (trace_id, span_id) in spans:
        record = spans[(trace_id, span_id)]
        if not record or span_id in seen or len(seen) >= 64:
            return None
        seen.add(span_id)
        attributes = _as_dict(record.get("attributes"))
        for keys, expected in ((_USER_KEYS, user_id), (_SESSION_KEYS, session_id)):
            if any(k in attributes and attributes[k] != expected for k in keys):
                return None
        for name in _INVOCATION_KEYS:
            if name in attributes:
                value = attributes[name]
                if not isinstance(value, str) or not value:
                    return None
                invocation_ids.add(value)
        parent = _otel_id(record.get("parent_id"))
        if not isinstance(parent, str):
            break
        span_id = parent
    return next(iter(invocation_ids)) if len(invocation_ids) == 1 else None


@contextmanager
def _read_only(path: Path) -> Iterator[sqlite3.Connection]:
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=1)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        yield db


def _has_table(db: sqlite3.Connection, name: str) -> bool:
    return (
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


class DashboardPrivateStore:
    """Expose scoped views without changing trace capture or database state."""

    def __init__(self, sessions: Path, traces: Path, wiki: Path) -> None:
        self.sessions = sessions
        self.traces = traces
        self.wiki = wiki

    async def get_context(
        self,
        user_id: str,
        session_id: str,
        limit: int,
        offset: int,
        key: tuple[str, str] | None,
    ) -> JsonObject:
        return await asyncio.to_thread(
            self._get_context, user_id, session_id, limit, offset, key
        )

    def _get_context(
        self,
        user_id: str,
        session_id: str,
        limit: int,
        offset: int,
        key: tuple[str, str] | None,
    ) -> JsonObject:
        if not self.sessions.is_file():
            return {"status": "unavailable"}
        scan = scan_traces_sync(self.traces)
        if scan.unavailable:
            return {"status": "trace_missing"}

        spans: dict[tuple[str, str], JsonObject] = {}
        for record in scan.records:
            span_key = _span_key(record)
            if span_key:
                spans[span_key] = {} if span_key in spans else record
        keys = [key] if key else list(spans)
        candidates: dict[str, list[tuple[tuple[str, str], JsonObject]]] = defaultdict(
            list
        )
        for span_key in keys:
            attributes = _as_dict(spans.get(span_key, {}).get("attributes"))
            if INPUT_MESSAGES not in attributes:
                continue
            invocation = _invocation(span_key, spans, user_id, session_id)
            if invocation is not None:
                candidates[invocation].append((span_key, attributes))

        captures = []
        try:
            with _read_only(self.sessions) as db:
                exists = db.execute(
                    "SELECT 1 FROM sessions "
                    "WHERE app_name='blacki' AND user_id=? AND id=?",
                    (user_id, session_id),
                ).fetchone()
                if exists is None:
                    return {"status": "session_missing"}
                # Check candidate IDs against all events, including events
                # outside the dashboard's bounded transcript snapshot.
                verified = db.execute(
                    """
                    SELECT invocation_id FROM events
                    WHERE app_name='blacki'
                      AND invocation_id IN (SELECT value FROM json_each(?))
                    GROUP BY invocation_id
                    HAVING COUNT(CASE WHEN user_id=? AND session_id=? THEN 1 END)
                           = COUNT(*)
                    """,
                    (json.dumps(list(candidates)), user_id, session_id),
                )
                for (invocation,) in verified:
                    for span_key, attributes in candidates[invocation]:
                        model = attributes.get("gen_ai.request.model")
                        item = {
                            "trace_id": span_key[0],
                            "span_id": span_key[1],
                            "timestamp": _iso_timestamp(
                                _trace_time(spans[span_key], "start_time")
                            ),
                            "model": model if isinstance(model, str) else None,
                        }
                        if key:
                            tools, partial = _tools(attributes)
                            return {
                                "status": "available",
                                "capture": {
                                    **item,
                                    "system_prompt": _prompt(attributes),
                                    "tools": tools,
                                    "partial_tools": partial,
                                    "messages": _messages(attributes),
                                    "response": _response(attributes),
                                    "tool_calls": _tool_calls(
                                        attributes,
                                        _tool_span_index(scan.records),
                                        _tool_results(scan.records),
                                    ),
                                    "raw_span": _raw_span(spans[span_key]),
                                },
                            }
                        captures.append(item)
        except sqlite3.Error:
            return {"status": "unavailable"}
        if key:
            return {"status": "capture_missing"}
        captures.sort(
            key=lambda item: (item["timestamp"] or "", item["span_id"]), reverse=True
        )
        return {
            "status": "available",
            "items": captures[offset : offset + limit],
            "total": len(captures),
            "limited_scan": bool(scan.truncated or scan.oversized or scan.invalid),
        }

    async def get_wiki(
        self,
        user_id: str,
        page: int | None,
        revision: int | None,
        offset: int,
    ) -> JsonObject:
        return await asyncio.to_thread(self._get_wiki, user_id, page, revision, offset)

    def _get_wiki(
        self,
        user_id: str,
        page: int | None,
        revision: int | None,
        offset: int,
    ) -> JsonObject:
        match = _DIRECT_USER.fullmatch(user_id)
        if not match or (owner := int(match[1])) > 2**63 - 1:
            return {"status": "unsupported_user"}
        if not wiki_configured():
            return {"status": "wiki_disabled"}
        if not self.wiki.is_file():
            return {"status": "missing"}
        try:
            with _read_only(self.sessions) as sessions:
                if (
                    sessions.execute(
                        "SELECT 1 FROM sessions "
                        "WHERE app_name='blacki' AND user_id=? LIMIT 1",
                        (user_id,),
                    ).fetchone()
                    is None
                ):
                    return {"status": "user_missing"}
            with _read_only(self.wiki) as db:
                state = db.execute(
                    "SELECT enabled, revision FROM wiki_owners WHERE owner=?", (owner,)
                ).fetchone()
                if state is not None and not state["enabled"]:
                    return {"status": "wiki_disabled"}
                current_revision = state["revision"] if state is not None else 0
                if revision is not None and revision != current_revision:
                    return {"status": "wiki_changed"}
                if not _has_table(db, "wiki_pages"):
                    # A claim-based database has no pages until the next wiki
                    # operation migrates its schema.
                    return {"status": "empty"}
                pages = db.execute(
                    """
                    SELECT path FROM wiki_pages
                    WHERE owner=? ORDER BY path LIMIT 51 OFFSET ?
                    """,
                    (owner, offset),
                ).fetchall()
                if not pages:
                    return {"status": "empty"}
                # A revision-bound position keeps page names out of access logs
                # and prevents writes from moving the browser's selection.
                page_index = offset if page is None else page
                if not offset <= page_index < offset + min(50, len(pages)):
                    return {"status": "page_missing"}
                # The read-only transaction keeps this page consistent with the
                # listing above.
                selected = db.execute(
                    """
                    SELECT path, markdown, updated_at FROM wiki_pages
                    WHERE owner=? AND path=?
                    """,
                    (owner, pages[page_index - offset]["path"]),
                ).fetchone()
                return {
                    "status": "available",
                    "pages": [row["path"] for row in pages[:50]],
                    "more_pages": len(pages) > 50,
                    "page": page_index,
                    "revision": current_revision,
                    "selected": {
                        "path": selected["path"],
                        "markdown": selected["markdown"],
                        "chars": len(selected["markdown"]),
                        "updated_at": selected["updated_at"],
                    },
                }
        except sqlite3.Error:
            return {"status": "unavailable"}
