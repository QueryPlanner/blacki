"""Read-only prompt and wiki data for the private operator dashboard."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from ..observability.redaction import redact_secrets
from ..wiki import wiki_configured
from .data import (
    _as_dict,
    _iso_timestamp,
    _JsonlScan,
    _scan_jsonl_sync,
    _trace_time,
)
from .models import MAX_ACCEPTED_JSONL_RECORDS, JsonObject

_TRACE_ID = re.compile(r"(?:0x)?[0-9a-f]{32}")
_SPAN_ID = re.compile(r"(?:0x)?[0-9a-f]{16}")
_DIRECT_USER = re.compile(r"telegram-chat-([1-9][0-9]{0,18})")
_TOOL_SCHEMA = re.compile(r"llm\.tools\.([0-9]{1,5})\.tool\.json_schema")
_SYSTEM_PART = re.compile(
    r"llm\.input_messages\.0\.message\.contents\.([0-9]{1,5})\.message_content\.text"
)
_INVOCATION_KEYS = ("blacki.invocation_id", "gcp.vertex.agent.invocation_id")
_SESSION_KEYS = (
    "blacki.session_id",
    "gcp.vertex.agent.session_id",
    "gen_ai.conversation.id",
    "session.id",
)
_USER_KEYS = ("blacki.user_id", "user.id")
_SECRET_KEY = re.compile(
    r"(?i)^(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"client[_-]?secret|password|passwd|secret|authorization|cookie|"
    r"telegram[_-]?bot[_-]?token|bot[_-]?token|api[_-]?token|token)$"
)


def valid_capture_id(trace_id: str, span_id: str) -> bool:
    """Accept the IDs written by the local exporter and older hex serializers."""
    return bool(_TRACE_ID.fullmatch(trace_id) and _SPAN_ID.fullmatch(span_id))


def _redact(text: str) -> str:
    return redact_secrets(text)


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, RecursionError):
            return None
    return value


def _content_text(value: Any) -> str:
    """Extract text from a Gemini content value or a list of them."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        parts = value.get("parts")
        if isinstance(parts, list):
            texts = [
                part["text"]
                for part in parts
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            ]
            if texts:
                return "\n".join(texts)
        text = value.get("text")
        if isinstance(text, str):
            return text
    if isinstance(value, list):
        return "\n".join(
            text for item in value if (text := _content_text(item)).strip()
        )
    return ""


def _adk_request(attributes: JsonObject) -> Mapping[str, Any] | None:
    request = _decode(attributes.get("gcp.vertex.agent.llm_request"))
    return request if isinstance(request, Mapping) else None


def _prompt(attributes: JsonObject) -> str:
    request = _adk_request(attributes)
    if request is not None:
        config = _as_dict(request.get("config"))
        prompt = _content_text(config.get("system_instruction"))
        if prompt.strip():
            return prompt
        contents = request.get("contents")
        if isinstance(contents, list):
            system_text = "\n".join(
                text
                for content in contents
                if isinstance(content, Mapping) and content.get("role") == "system"
                if (text := _content_text(content)).strip()
            )
            if system_text:
                return system_text

    parts = _decode(attributes.get("gen_ai.system_instructions"))
    if isinstance(parts, list):
        texts = [
            part["content"]
            for part in parts
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("content"), str)
        ]
        if texts:
            return "\n".join(texts)
    if attributes.get("llm.input_messages.0.message.role") != "system":
        return ""
    content = attributes.get("llm.input_messages.0.message.content")
    if isinstance(content, str) and content:
        return content
    entries = sorted(
        (int(match[1]), text)
        for key, text in attributes.items()
        if isinstance(text, str) and (match := _SYSTEM_PART.fullmatch(key))
    )
    return "\n".join(text for _, text in entries)


def _tools(attributes: JsonObject) -> tuple[list[JsonObject], bool]:
    request = _adk_request(attributes)
    if request is not None:
        config = _as_dict(request.get("config"))
        definitions = config.get("tools")
        if isinstance(definitions, list):
            tools: list[JsonObject] = []
            partial = False
            for definition in definitions:
                declaration_list = _as_dict(definition).get("function_declarations")
                if not isinstance(declaration_list, list):
                    partial = True
                    continue
                for declaration in declaration_list:
                    item = _as_dict(declaration)
                    name, description = item.get("name"), item.get("description")
                    if isinstance(name, str) and name:
                        tools.append(
                            {
                                "name": _redact(name),
                                "description": (
                                    _redact(description)
                                    if isinstance(description, str)
                                    else ""
                                ),
                                "parameters": item.get("parameters") or {},
                            }
                        )
                    else:
                        partial = True
            return tools, partial

    entries = sorted(
        (int(match[1]), _decode(value))
        for key, value in attributes.items()
        if (match := _TOOL_SCHEMA.fullmatch(key))
    )
    definitions = _decode(attributes.get("gen_ai.tool.definitions"))
    if not isinstance(definitions, list):
        definitions = [value for _, value in entries]
    tools = []
    for definition in definitions:
        tool = _as_dict(definition)
        name, description = tool.get("name"), tool.get("description")
        if (
            isinstance(name, str)
            and name
            and isinstance(description, str)
            and description
        ):
            tools.append({"name": _redact(name), "description": _redact(description)})
    gaps = bool(entries) and [i for i, _ in entries] != list(range(len(entries)))
    return tools, len(tools) != len(definitions) or gaps


def _redact_value(value: Any) -> Any:
    """Redact credential-shaped strings inside a decoded JSON value."""
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]" if _SECRET_KEY.fullmatch(str(key)) else _redact_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _normalize_part(part: Mapping[str, Any]) -> JsonObject:
    """Describe one Gemini content part without exposing large binary blobs."""
    text = part.get("text")
    if isinstance(text, str):
        return {
            "type": "thought" if part.get("thought") is True else "text",
            "text": _redact(text),
        }
    call = part.get("function_call")
    if isinstance(call, Mapping):
        return {
            "type": "function_call",
            "name": _redact(str(call.get("name") or "")),
            "args": _redact_value(call.get("args") or {}),
            "call_id": _redact(str(call.get("id") or "")),
        }
    response = part.get("function_response")
    if isinstance(response, Mapping):
        return {
            "type": "function_response",
            "name": _redact(str(response.get("name") or "")),
            "response": _redact_value(response.get("response") or {}),
            "call_id": _redact(str(response.get("id") or "")),
        }
    inline = part.get("inline_data")
    if isinstance(inline, Mapping):
        mime_type = inline.get("mime_type")
        return {
            "type": "inline_data",
            "mime_type": _redact(mime_type) if isinstance(mime_type, str) else "",
        }
    file_data = part.get("file_data")
    if isinstance(file_data, Mapping):
        mime_type, uri = file_data.get("mime_type"), file_data.get("file_uri")
        return {
            "type": "file_data",
            "mime_type": _redact(mime_type) if isinstance(mime_type, str) else "",
            "file_uri": _redact(uri) if isinstance(uri, str) else "",
        }
    return {"type": "other", "keys": sorted(str(key) for key in part)}


def _normalize_content(content: Mapping[str, Any]) -> JsonObject:
    """Normalize one content object into a role, text summary, and parts."""
    parts = content.get("parts")
    normalized = (
        [_normalize_part(part) for part in parts if isinstance(part, Mapping)]
        if isinstance(parts, list)
        else []
    )
    texts = [
        part["text"]
        for part in normalized
        if part["type"] in {"text", "thought"} and part.get("text")
    ]
    role = content.get("role")
    return {
        "role": role if isinstance(role, str) else "",
        "text": "\n".join(texts),
        "parts": normalized,
    }


def _messages(attributes: JsonObject) -> list[JsonObject]:
    """Return the conversation messages included in the model request."""
    request = _adk_request(attributes)
    if request is None:
        return []
    contents = request.get("contents")
    if not isinstance(contents, list):
        return []
    return [
        _normalize_content(content)
        for content in contents
        if isinstance(content, Mapping)
    ]


def _response(attributes: JsonObject) -> JsonObject | None:
    """Normalize the model response captured on an ADK span."""
    response = _decode(attributes.get("gcp.vertex.agent.llm_response"))
    if not isinstance(response, Mapping):
        return None
    content = _as_dict(response.get("content"))
    normalized = (
        _normalize_content(content)
        if content
        else {"role": "model", "text": "", "parts": []}
    )
    result: JsonObject = {
        "text": normalized["text"],
        "parts": normalized["parts"],
    }
    for field in (
        "finish_reason",
        "error_code",
        "error_message",
        "interrupted",
        "turn_complete",
        "partial",
        "model_version",
    ):
        value = response.get(field)
        if value is not None:
            result[field] = _redact(value) if isinstance(value, str) else value
    usage = response.get("usage_metadata")
    if isinstance(usage, Mapping):
        result["usage"] = _redact_value(usage)
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
    """Index tool execution spans by their provider call ID."""
    index: dict[str, JsonObject] = {}
    for record in records:
        attributes = _as_dict(record.get("attributes"))
        if attributes.get("gen_ai.operation.name") != "execute_tool":
            continue
        call_id = attributes.get("gen_ai.tool.call.id")
        if isinstance(call_id, str) and call_id:
            index[call_id] = record
    return index


def _tool_calls(
    attributes: JsonObject, index: Mapping[str, JsonObject]
) -> list[JsonObject]:
    """Pair function calls in a model response with their execution spans."""
    response = _decode(attributes.get("gcp.vertex.agent.llm_response"))
    if not isinstance(response, Mapping):
        return []
    parts = _as_dict(response.get("content")).get("parts")
    if not isinstance(parts, list):
        return []
    calls: list[JsonObject] = []
    for part in parts:
        call = _as_dict(part).get("function_call")
        if not isinstance(call, Mapping):
            continue
        raw_call_id = call.get("id")
        call_id = raw_call_id if isinstance(raw_call_id, str) else ""
        record = index.get(call_id)
        tool_attributes = _as_dict(record.get("attributes")) if record else {}
        args = call.get("args")
        if args is None:
            args = _decode(tool_attributes.get("gcp.vertex.agent.tool_call_args"))
        result = _decode(tool_attributes.get("gcp.vertex.agent.tool_response"))
        status = _as_dict(record.get("status")).get("status_code") if record else None
        calls.append(
            {
                "name": _redact(str(call.get("name") or "")),
                "call_id": _redact(call_id),
                "args": _redact_value(args) if args is not None else {},
                "response": _redact_value(result) if result is not None else None,
                "status": status if isinstance(status, str) else None,
                "duration_ms": _duration_ms(record),
            }
        )
    return calls


def _raw_request(attributes: JsonObject) -> str | None:
    """Render the captured ADK request as redacted, readable JSON."""
    request = _adk_request(attributes)
    if request is None:
        return None
    try:
        return _redact(json.dumps(request, ensure_ascii=False, indent=2, default=str))
    except (TypeError, ValueError):
        return None


def _scan_trace_dir_sync(path: Path) -> _JsonlScan:
    """Scan the daily trace files, newest data winning inside the record cap."""
    try:
        files = sorted(path.glob("*.jsonl"))
    except OSError:
        return _JsonlScan((), 0, 0, 0, unavailable=True)
    if not files:
        return _JsonlScan((), 0, 0, 0, unavailable=True)

    records: deque[dict[str, Any]] = deque(maxlen=MAX_ACCEPTED_JSONL_RECORDS)
    accepted = invalid = oversized = 0
    truncated = False
    for file_path in files:
        scan = _scan_jsonl_sync(file_path)
        if scan.unavailable:
            continue
        accepted += scan.accepted
        invalid += scan.invalid
        oversized += scan.oversized
        truncated = truncated or scan.truncated
        records.extend(scan.records)
    if accepted == 0 and not records:
        return _JsonlScan((), 0, 0, 0, unavailable=True)
    return _JsonlScan(tuple(records), accepted, invalid, oversized, truncated=truncated)


def _span_key(record: JsonObject) -> tuple[str, str] | None:
    ctx = _as_dict(record.get("context"))
    trace, span = ctx.get("trace_id"), ctx.get("span_id")
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
        parent = record.get("parent_id")
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
        scan = _scan_trace_dir_sync(self.traces)
        if scan.unavailable:
            return {"status": "trace_missing"}

        spans: dict[tuple[str, str], JsonObject] = {}
        for record in scan.records:
            span_key = _span_key(record)
            if span_key:
                spans[span_key] = {} if span_key in spans else record
        keys = [key] if key else list(spans)
        candidates: dict[str, list[tuple[tuple[str, str], JsonObject, str]]] = (
            defaultdict(list)
        )
        for span_key in keys:
            attributes = _as_dict(spans.get(span_key, {}).get("attributes"))
            prompt = _prompt(attributes)
            if not prompt.strip() or prompt.strip().lower() in {
                "[redacted]",
                "<redacted>",
                "[elided]",
                "<elided>",
            }:
                continue
            invocation = _invocation(span_key, spans, user_id, session_id)
            if invocation is not None:
                candidates[invocation].append((span_key, attributes, prompt))

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
                    for span_key, attributes, prompt in candidates[invocation]:
                        model = attributes.get(
                            "gen_ai.request.model"
                        ) or attributes.get("llm.model_name")
                        item = {
                            "trace_id": span_key[0],
                            "span_id": span_key[1],
                            "timestamp": _iso_timestamp(
                                _trace_time(spans[span_key], "start_time")
                            ),
                            "model": _redact(model) if isinstance(model, str) else None,
                        }
                        if key:
                            tools, partial = _tools(attributes)
                            tool_index = _tool_span_index(scan.records)
                            return {
                                "status": "available",
                                "capture": {
                                    **item,
                                    "system_prompt": _redact(prompt),
                                    "tools": tools,
                                    "partial_tools": partial,
                                    "messages": _messages(attributes),
                                    "response": _response(attributes),
                                    "tool_calls": _tool_calls(attributes, tool_index),
                                    "raw_request": _raw_request(attributes),
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
                    "pages": [_redact(row["path"]) for row in pages[:50]],
                    "more_pages": len(pages) > 50,
                    "page": page_index,
                    "revision": current_revision,
                    "selected": {
                        "path": _redact(selected["path"]),
                        "markdown": _redact(selected["markdown"]),
                        "chars": len(selected["markdown"]),
                        "updated_at": selected["updated_at"],
                    },
                }
        except sqlite3.Error:
            return {"status": "unavailable"}
