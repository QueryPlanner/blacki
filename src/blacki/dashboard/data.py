"""Read-only data access for Blacki's local observability dashboard.

The ADK session database is an implementation detail and its schema has
changed over time.  This module therefore reads only the small set of stable
columns needed by the dashboard and adapts both the current JSON event shape
and the older column-based shape.  All database and JSONL work happens in
worker threads so FastAPI's event loop remains responsive.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .models import (
    DEGRADED_DATABASE_WARNING,
    MAX_ACCEPTED_JSONL_RECORDS,
    MAX_DATABASE_ROWS,
    MAX_JSONL_LINE_BYTES,
    MAX_JSONL_TAIL_BYTES,
    MAX_LOG_PAGE_SIZE,
    MAX_TRACE_PAGE_SIZE,
    MISSING_EVENTS_WARNING,
    MISSING_TELEMETRY_WARNING,
    MISSING_TRACES_WARNING,
    TRACE_ATTRIBUTE_ALLOWLIST,
    JsonObject,
    bounded_search,
    clamp_limit,
    clamp_offset,
    page_result,
)

_SESSION_VERSION_RE = re.compile(r"^(?P<prefix>.+)-v(?P<version>[0-9]+)$")
_TELEGRAM_DIRECT_USER_ID_RE = re.compile(r"^telegram-chat-(?P<user_id>[0-9]+)$")
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:Users|home|var|tmp|private|app|workspace|opt|srv|etc)/[^\s\"']+"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_TELEGRAM_TOKEN_RE = re.compile(r"\b[0-9]{6,12}:[A-Za-z0-9_-]{20,}\b")
_KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"client[_-]?secret|password|passwd|secret|authorization|cookie|"
    r"telegram[_-]?bot[_-]?token|bot[_-]?token|api[_-]?token|token)\b\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
)
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api_token",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "password",
        "passwd",
        "secret",
        "authorization",
        "cookie",
        "telegram_bot_token",
        "bot_token",
        "credential",
        "credentials",
    }
)
_WINDOW_RE = re.compile(r"^(?P<count>[0-9]{1,4})(?P<unit>[hdw])$")
_MAX_TEXT_REDACTION_LENGTH = 64 * 1024
_MIN_TIMESTAMP_SECONDS = -62_135_596_800.0  # datetime year 1
_MAX_TIMESTAMP_SECONDS = 253_402_300_799.999999  # datetime year 9999
_TRACE_NUMERIC_ATTRIBUTE_KEYS = frozenset(
    {
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.total_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.prompt_token_count",
        "gen_ai.usage.candidates_token_count",
        "llm.token_count.input",
        "llm.token_count.output",
        "llm.token_count.total",
        "llm.token_count.prompt",
        "llm.token_count.completion",
    }
)


@dataclass(frozen=True, slots=True)
class _SessionRow:
    user_id: str
    session_id: str
    created_at: float | None
    updated_at: float | None


@dataclass(frozen=True, slots=True)
class _EventRow:
    event_id: str
    user_id: str
    session_id: str
    invocation_id: str
    timestamp: float | None
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Snapshot:
    sessions: tuple[_SessionRow, ...]
    events: tuple[_EventRow, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _JsonlScan:
    records: tuple[dict[str, Any], ...]
    accepted: int
    invalid: int
    oversized: int
    unavailable: bool = False
    truncated: bool = False


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _parse_timestamp(value: Any) -> float | None:
    """Parse ADK/SQLite timestamps without raising on damaged rows."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        return _normalize_epoch_seconds(numeric)
    if isinstance(value, datetime):
        parsed = value if value.tzinfo else value.replace(tzinfo=UTC)
        try:
            return _normalize_epoch_seconds(parsed.timestamp())
        except (OSError, OverflowError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    number: float | None
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None:
        return _normalize_epoch_seconds(number)
    try:
        parsed_dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=UTC)
    try:
        return _normalize_epoch_seconds(parsed_dt.timestamp())
    except (OSError, OverflowError, ValueError):
        return None


def _normalize_epoch_seconds(value: float) -> float | None:
    """Normalize common epoch units and reject values outside datetime."""
    try:
        if not math.isfinite(value):
            return None
        magnitude = abs(value)
        if magnitude >= 1e18:
            value /= 1e9
        elif magnitude >= 1e15:
            value /= 1e6
        elif magnitude >= 1e12:
            value /= 1e3
    except (OverflowError, TypeError, ValueError):
        return None
    if not _MIN_TIMESTAMP_SECONDS <= value <= _MAX_TIMESTAMP_SECONDS:
        return None
    return value


def _iso_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    normalized = _normalize_epoch_seconds(timestamp)
    if normalized is None:
        return None
    try:
        return datetime.fromtimestamp(normalized, tz=UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _number(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(float(value)))
        except ValueError:
            return 0
    return 0


def _parse_json_object(value: Any) -> tuple[dict[str, Any], bool]:
    """Return an object and whether the stored value was malformed."""
    if isinstance(value, Mapping):
        return dict(value), False
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="replace")
        except (TypeError, ValueError):  # pragma: no cover - bytes always decode
            return {}, True
    if not isinstance(value, str):
        return {}, value is not None
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        return {}, True
    if not isinstance(decoded, dict):
        return {}, True
    return decoded, False


def _row_value(row: sqlite3.Row, name: str) -> Any:
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def _select_table_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    columns: set[str],
    where: str,
    parameters: tuple[Any, ...],
    order_by: str | None = None,
) -> list[sqlite3.Row]:
    """Select only known columns; table/column names are internal constants."""
    requested = (
        "app_name",
        "user_id",
        "id",
        "session_id",
        "create_time",
        "update_time",
        "invocation_id",
        "timestamp",
        "event_data",
        "author",
        "content",
        "usage_metadata",
        "error_code",
        "error_message",
        "partial",
        "turn_complete",
    )
    selected = [name for name in requested if name in columns]
    if not selected:
        return []
    query = f"SELECT {', '.join(selected)} FROM {table} WHERE {where}"  # noqa: S608
    if order_by and order_by in columns:
        query += f" ORDER BY {order_by} DESC"  # noqa: S608
    query += f" LIMIT {MAX_DATABASE_ROWS}"
    return list(connection.execute(query, parameters).fetchall())


def _load_snapshot_sync(path: Path, app_name: str) -> _Snapshot:
    warnings: list[str] = []
    sessions: list[_SessionRow] = []
    events: list[_EventRow] = []
    malformed_event_count = 0

    try:
        if not path.is_file():
            return _Snapshot((), (), (DEGRADED_DATABASE_WARNING,))
    except OSError:
        return _Snapshot((), (), (DEGRADED_DATABASE_WARNING,))

    connection: sqlite3.Connection | None = None
    try:
        # The URI intentionally uses mode=ro.  No dashboard operation can
        # create tables, run migrations, or mutate the bot's active database.
        uri = f"file:{quote(path.resolve().as_posix(), safe='/')}?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=1.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")

        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "sessions" not in tables:
            warnings.append(DEGRADED_DATABASE_WARNING)
        else:
            session_columns = _table_columns(connection, "sessions")
            required = {"app_name", "user_id", "id"}
            if not required.issubset(session_columns):
                warnings.append(DEGRADED_DATABASE_WARNING)
            else:
                rows = _select_table_rows(
                    connection,
                    table="sessions",
                    columns=session_columns,
                    where="app_name = ?",
                    parameters=(app_name,),
                    order_by="update_time",
                )
                for row in rows:
                    user_id = _row_value(row, "user_id")
                    session_id = _row_value(row, "id")
                    if not isinstance(user_id, str) or not isinstance(session_id, str):
                        continue
                    sessions.append(
                        _SessionRow(
                            user_id=user_id,
                            session_id=session_id,
                            created_at=_parse_timestamp(_row_value(row, "create_time")),
                            updated_at=_parse_timestamp(_row_value(row, "update_time")),
                        )
                    )

        if "events" not in tables:
            warnings.append(MISSING_EVENTS_WARNING)
        else:
            event_columns = _table_columns(connection, "events")
            required = {"app_name", "user_id", "session_id"}
            if not required.issubset(event_columns):
                warnings.append(MISSING_EVENTS_WARNING)
            else:
                rows = _select_table_rows(
                    connection,
                    table="events",
                    columns=event_columns,
                    where="app_name = ?",
                    parameters=(app_name,),
                    order_by="timestamp",
                )
                for row in rows:
                    user_id = _row_value(row, "user_id")
                    session_id = _row_value(row, "session_id")
                    if not isinstance(user_id, str) or not isinstance(session_id, str):
                        continue

                    data, malformed = _parse_json_object(_row_value(row, "event_data"))
                    malformed_event_count += int(malformed)
                    # v0 ADK databases stored these fields as columns rather
                    # than event_data.  Merge only scalar/object fields that
                    # the canonical adapter understands.
                    legacy_values = {
                        "author": _row_value(row, "author"),
                        "content": _row_value(row, "content"),
                        "usage_metadata": _row_value(row, "usage_metadata"),
                        "error_code": _row_value(row, "error_code"),
                        "error_message": _row_value(row, "error_message"),
                        "partial": _row_value(row, "partial"),
                        "turn_complete": _row_value(row, "turn_complete"),
                    }
                    for key, value in legacy_values.items():
                        if key not in data and value is not None:
                            if key in {"content", "usage_metadata"}:
                                parsed_value, legacy_malformed = _parse_json_object(
                                    value
                                )
                                malformed_event_count += int(legacy_malformed)
                                data[key] = parsed_value
                            else:
                                data[key] = value

                    event_id = _row_value(row, "id")
                    if not isinstance(event_id, str):
                        event_id = str(data.get("id") or "")
                    invocation_id = _row_value(row, "invocation_id")
                    if not isinstance(invocation_id, str) or not invocation_id:
                        invocation_id = str(data.get("invocation_id") or "")
                    event_timestamp = _parse_timestamp(data.get("timestamp"))
                    if event_timestamp is None:
                        event_timestamp = _parse_timestamp(_row_value(row, "timestamp"))
                    events.append(
                        _EventRow(
                            event_id=event_id,
                            user_id=user_id,
                            session_id=session_id,
                            invocation_id=invocation_id,
                            timestamp=event_timestamp,
                            data=data,
                        )
                    )
        connection.rollback()
    except (OSError, sqlite3.Error, ValueError):
        # The UI should remain usable when the bot is rotating or repairing a
        # database.  Do not include paths or driver exceptions in API output.
        sessions = []
        events = []
        warnings = [DEGRADED_DATABASE_WARNING]
    finally:
        if connection is not None:
            connection.close()

    if malformed_event_count:
        warnings.append("Some session events could not be decoded.")
    return _Snapshot(
        tuple(sessions),
        tuple(events),
        tuple(dict.fromkeys(warnings)),
    )


def _load_telegram_identities_sync(path: Path) -> dict[int, tuple[str, str | None]]:
    """Read dashboard-only Telegram labels without modifying the tools database."""
    if not path.is_file():
        return {}
    connection: sqlite3.Connection | None = None
    try:
        uri = f"file:{quote(path.resolve().as_posix(), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            """
            SELECT telegram_user_id, display_name, username
            FROM telegram_identities
            """
        ).fetchall()
        return {
            int(row["telegram_user_id"]): (str(row["display_name"]), row["username"])
            for row in rows
            if isinstance(row["telegram_user_id"], int)
            and isinstance(row["display_name"], str)
            and (row["username"] is None or isinstance(row["username"], str))
        }
    except (OSError, sqlite3.Error, ValueError):
        return {}
    finally:
        if connection is not None:
            connection.close()


def _iter_raw_jsonl_lines(handle: Any) -> Iterator[tuple[bytes, bool]]:
    """Yield bounded lines and flag oversized lines without allocating them."""
    buffer = bytearray()
    oversized = False
    while True:
        chunk = handle.read(64 * 1024)
        if not chunk:
            break
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            if oversized:
                oversized = False
                yield b"", True
            elif len(line.rstrip(b"\r")) > MAX_JSONL_LINE_BYTES:
                yield b"", True
            else:
                yield line.rstrip(b"\r"), False
        if not oversized and len(buffer) > MAX_JSONL_LINE_BYTES:
            # Discard the already-read prefix.  The rest is consumed in
            # bounded chunks until the newline marks the end of this record.
            buffer.clear()
            oversized = True
        elif oversized:
            # Keep discarding chunks until the record terminator arrives.
            # Without this branch, one malformed record can grow the buffer
            # without bound after it first crosses the configured limit.
            buffer.clear()
    if oversized:
        yield b"", True
    elif buffer:
        # A final line without a newline is a valid JSONL record when JSON is
        # complete, and malformed JSON is simply counted as invalid below.
        yield bytes(buffer).rstrip(b"\r"), False


def _discard_partial_jsonl_line(handle: Any) -> None:
    """Advance to the first complete line after a bounded tail seek."""
    while True:
        chunk_start = handle.tell()
        chunk = handle.read(64 * 1024)
        newline = chunk.find(b"\n") if chunk else -1
        if newline >= 0:
            handle.seek(chunk_start + newline + 1)
            return
        if not chunk:
            return


def _scan_jsonl_sync(path: Path) -> _JsonlScan:
    records: deque[dict[str, Any]] = deque(maxlen=MAX_ACCEPTED_JSONL_RECORDS)
    invalid = 0
    oversized = 0
    accepted = 0
    truncated = False
    try:
        file_size = path.stat().st_size
        handle = path.open("rb")
    except (OSError, ValueError):
        return _JsonlScan((), 0, 0, 0, unavailable=True)

    with handle:
        tail_offset = max(0, file_size - MAX_JSONL_TAIL_BYTES)
        if tail_offset:
            truncated = True
            handle.seek(tail_offset)
            _discard_partial_jsonl_line(handle)
        for raw_line, was_oversized in _iter_raw_jsonl_lines(handle):
            if was_oversized:
                oversized += 1
                continue
            try:
                decoded = json.loads(raw_line.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
                invalid += 1
                continue
            if not isinstance(decoded, dict):
                invalid += 1
                continue
            accepted = min(MAX_ACCEPTED_JSONL_RECORDS, accepted + 1)
            records.append(decoded)
    return _JsonlScan(tuple(records), accepted, invalid, oversized, truncated=truncated)


def _sanitize_text(value: str) -> str:
    text = value[:_MAX_TEXT_REDACTION_LENGTH]
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _TELEGRAM_TOKEN_RE.sub("[REDACTED]", text)
    text = _KEY_VALUE_SECRET_RE.sub(r"\1[REDACTED]", text)
    text = _ABSOLUTE_PATH_RE.sub("[PATH_REDACTED]", text)
    if len(value) > _MAX_TEXT_REDACTION_LENGTH:
        text += "…"
    return text


def _sanitize_json(value: Any, key: str | None = None) -> Any:
    key_name = (
        re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower().replace("-", "_") if key else ""
    )
    if key_name in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if key_name in {"exception", "exc_info", "traceback", "stack"}:
        return "[REDACTED_EXCEPTION]"
    if isinstance(value, str):
        return _sanitize_text(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _sanitize_json(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return _sanitize_text(str(value))


def _event_content(event: _EventRow) -> dict[str, Any]:
    return _as_dict(event.data.get("content"))


def _tool_status(response: Mapping[str, Any]) -> str:
    status = response.get("status")
    if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
        return "error"
    if any(key in response for key in ("error", "error_code", "errorCode")):
        return "error"
    nested = response.get("response")
    if isinstance(nested, Mapping):
        nested_status = nested.get("status")
        if isinstance(nested_status, str) and nested_status.lower() in {
            "error",
            "failed",
            "failure",
        }:
            return "error"
        if any(key in nested for key in ("error", "error_code", "errorCode")):
            return "error"
    return "completed"


def _attachment_item(
    part: Mapping[str, Any], *, event: _EventRow, index: int
) -> JsonObject | None:
    """Expose only a MIME marker for inline/file data parts."""
    attachment: Any = None
    for key in ("inline_data", "inlineData", "file_data", "fileData"):
        if key in part:
            attachment = part.get(key)
            break
    if attachment is None:
        return None
    mime_type: Any = None
    if isinstance(attachment, Mapping):
        mime_type = attachment.get("mime_type", attachment.get("mimeType"))
    safe_mime = (
        _sanitize_text(mime_type)[:128]
        if isinstance(mime_type, str) and mime_type
        else "unknown"
    )
    return {
        "id": f"{event.event_id}:attachment:{index}",
        "timestamp": _iso_timestamp(event.timestamp),
        "type": "attachment",
        "mime_type": safe_mime,
    }


def _event_messages(event: _EventRow) -> list[JsonObject]:
    """Project an ADK event to safe human/model/tool replay items."""
    content = _event_content(event)
    parts = content.get("parts")
    if not isinstance(parts, list):
        return []
    role_value = content.get("role")
    author = str(event.data.get("author") or "").lower()
    role = "human" if role_value == "user" or author == "user" else "model"
    result: list[JsonObject] = []
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            continue
        if part.get("thought"):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            result.append(
                {
                    "id": event.event_id,
                    "timestamp": _iso_timestamp(event.timestamp),
                    "role": role,
                    "text": text,
                }
            )
        attachment = _attachment_item(part, event=event, index=index)
        if attachment is not None:
            result.append(attachment)
        function_call = part.get("function_call") or part.get("functionCall")
        if isinstance(function_call, Mapping):
            name = function_call.get("name")
            if isinstance(name, str) and name:
                result.append(
                    {
                        "id": f"{event.event_id}:tool:{index}",
                        "timestamp": _iso_timestamp(event.timestamp),
                        "type": "tool",
                        "name": _sanitize_text(name)[:256],
                        "status": "requested",
                    }
                )
        function_response = part.get("function_response") or part.get(
            "functionResponse"
        )
        if isinstance(function_response, Mapping):
            name = function_response.get("name")
            if isinstance(name, str) and name:
                result.append(
                    {
                        "id": f"{event.event_id}:tool:{index}",
                        "timestamp": _iso_timestamp(event.timestamp),
                        "type": "tool",
                        "name": _sanitize_text(name)[:256],
                        "status": _tool_status(function_response),
                    }
                )
    return result


def _event_usage(event: _EventRow) -> tuple[int, int, int]:
    usage = event.data.get("usage_metadata") or event.data.get("usageMetadata")
    if not isinstance(usage, Mapping):
        return 0, 0, 0
    input_tokens = _number(
        usage.get(
            "prompt_token_count",
            usage.get(
                "promptTokenCount",
                usage.get("input_tokens", usage.get("inputTokens", 0)),
            ),
        )
    )
    output_tokens = _number(
        usage.get(
            "candidates_token_count",
            usage.get(
                "candidatesTokenCount",
                usage.get("output_tokens", usage.get("outputTokens", 0)),
            ),
        )
    )
    total_tokens = _number(
        usage.get(
            "total_token_count",
            usage.get(
                "totalTokenCount",
                usage.get("total_tokens", usage.get("totalTokens", 0)),
            ),
        )
    )
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _event_is_error(event: _EventRow) -> bool:
    if event.data.get("error_code") or event.data.get("errorCode"):
        return True
    if event.data.get("error_message") or event.data.get("errorMessage"):
        return True
    error = event.data.get("error")
    return bool(error)


def _event_stats(events: Iterable[_EventRow]) -> dict[str, Any]:
    event_list = list(events)
    message_count = sum(
        sum(1 for item in _event_messages(event) if item.get("role"))
        for event in event_list
    )
    invocation_timestamps: dict[str, list[float]] = defaultdict(list)
    input_tokens = output_tokens = total_tokens = event_errors = 0
    error_invocations: set[str] = set()
    for event in event_list:
        if event.invocation_id and event.timestamp is not None:
            invocation_timestamps[event.invocation_id].append(event.timestamp)
        current_input, current_output, current_total = _event_usage(event)
        input_tokens += current_input
        output_tokens += current_output
        total_tokens += current_total
        if _event_is_error(event):
            event_errors += 1
            if event.invocation_id:
                error_invocations.add(event.invocation_id)
    latencies = [
        (max(timestamps) - min(timestamps)) * 1000
        for timestamps in invocation_timestamps.values()
        if len(timestamps) > 1 and max(timestamps) >= min(timestamps)
    ]
    latencies.sort()
    average = sum(latencies) / len(latencies) if latencies else 0.0
    if latencies:
        p95_index = min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)
        p95 = latencies[p95_index]
        maximum = latencies[-1]
    else:
        p95 = maximum = 0.0
    return {
        "messages": message_count,
        "invocations": len(invocation_timestamps),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": total_tokens,
        "errors": event_errors,
        "request_errors": len(error_invocations),
        "latency_ms": {
            "average": round(average, 2),
            "p95": round(p95, 2),
            "max": round(maximum, 2),
        },
    }


def _session_version(session_id: str) -> tuple[str, int | None]:
    match = _SESSION_VERSION_RE.match(session_id)
    if match is None:
        return session_id, None
    try:
        return match.group("prefix"), int(match.group("version"))
    except (TypeError, ValueError):  # pragma: no cover - regex only captures digits
        return session_id, None


def _session_sort_key(session: _SessionRow) -> tuple[int, float, str]:
    _, version = _session_version(session.session_id)
    return (
        version if version is not None else -1,
        session.updated_at if session.updated_at is not None else 0.0,
        session.session_id,
    )


def _session_events(
    events: Iterable[_EventRow], *, user_id: str, session_id: str
) -> list[_EventRow]:
    selected = [
        event
        for event in events
        if event.user_id == user_id and event.session_id == session_id
    ]
    selected.sort(
        key=lambda event: (
            event.timestamp if event.timestamp is not None else 0.0,
            event.event_id,
        )
    )
    return selected


def _session_item(
    session: _SessionRow,
    events: Iterable[_EventRow],
    *,
    is_current: bool,
    retained_session_ids: list[str],
) -> JsonObject:
    event_list = list(events)
    _, version = _session_version(session.session_id)
    stats = _event_stats(event_list)
    reset_generation = max(0, version - 1) if version is not None else None
    return {
        "id": session.session_id,
        "session_id": session.session_id,
        "user_id": session.user_id,
        "version": version,
        "generation": version,
        "reset_generation": reset_generation,
        "is_current": is_current,
        "retained_history": bool(retained_session_ids),
        "retained_session_ids": retained_session_ids,
        "created_at": _iso_timestamp(session.created_at),
        "updated_at": _iso_timestamp(session.updated_at),
        "event_count": len(event_list),
        "message_count": stats["messages"],
        "invocation_count": stats["invocations"],
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "tokens": stats["tokens"],
        "error_count": stats["errors"],
        "latency_ms": stats["latency_ms"],
    }


def _window_bounds(window: str | None) -> tuple[str, float | None, float, list[str]]:
    now = datetime.now(tz=UTC).timestamp()
    value = str(window or "24h").strip().lower()
    if value == "all":
        return value, None, now, []
    match = _WINDOW_RE.match(value)
    if match is None:
        return "24h", now - 24 * 3600, now, ["Unknown window; using 24h."]
    count = int(match.group("count"))
    unit = match.group("unit")
    seconds = count * {"h": 3600, "d": 86400, "w": 7 * 86400}[unit]
    seconds = min(seconds, 365 * 86400)
    return value, now - seconds, now, []


def _in_window(timestamp: float | None, since: float | None, until: float) -> bool:
    return (
        timestamp is not None
        and (since is None or timestamp >= since)
        and timestamp <= until
    )


def _activity_series(
    events: Iterable[_EventRow], *, since: float | None, until: float, window: str
) -> list[JsonObject]:
    event_list = list(events)
    if since is None:
        timestamps = [
            event.timestamp for event in event_list if event.timestamp is not None
        ]
        start = min(timestamps) if timestamps else until - 24 * 3600
        end = max(until, start + 3600)
        bucket_count = min(24, max(1, math.ceil((end - start) / 3600)))
    else:
        start = since
        if (window.endswith("h") and int(window[:-1] or "0") <= 24) or (
            window.endswith("d") and int(window[:-1] or "0") <= 30
        ):
            bucket_count = max(1, int(window[:-1]))
        else:
            bucket_count = 24
        end = until
    width = max((end - start) / bucket_count, 1.0)
    buckets: list[dict[str, Any]] = [
        {
            "timestamp": _iso_timestamp(start + index * width),
            "messages": 0,
            "invocations": set(),
            "tokens": 0,
            "errors": 0,
        }
        for index in range(bucket_count)
    ]
    for event in event_list:
        if not _in_window(event.timestamp, since, until) or event.timestamp is None:
            continue
        index = min(bucket_count - 1, max(0, int((event.timestamp - start) / width)))
        bucket = buckets[index]
        bucket["messages"] += sum(
            1 for item in _event_messages(event) if item.get("role")
        )
        if event.invocation_id:
            bucket["invocations"].add(event.invocation_id)
        bucket["tokens"] += _event_usage(event)[2]
        bucket["errors"] += int(_event_is_error(event))
    return [
        {
            "timestamp": bucket["timestamp"],
            "messages": bucket["messages"],
            "invocations": len(bucket["invocations"]),
            "tokens": bucket["tokens"],
            "errors": bucket["errors"],
        }
        for bucket in buckets
    ]


def _scan_warning(scan: _JsonlScan, *, missing: str) -> list[str]:
    warnings: list[str] = []
    if scan.unavailable:
        warnings.append(missing)
    if scan.invalid:
        warnings.append("Some log records were malformed and were skipped.")
    if scan.oversized:
        warnings.append("Some log records exceeded the 1 MiB line limit.")
    if scan.truncated:
        warnings.append("Only the most recent log tail was scanned.")
    if scan.accepted >= MAX_ACCEPTED_JSONL_RECORDS:
        warnings.append("The log scan reached its safety limit.")
    return warnings


def _log_item(record: dict[str, Any]) -> JsonObject:
    clean = _sanitize_json(record)
    if not isinstance(clean, dict):  # pragma: no cover - input is always a dict
        clean = {}
    timestamp = _parse_timestamp(record.get("timestamp"))
    if timestamp is not None:
        clean["timestamp"] = _iso_timestamp(timestamp)
    level = record.get("level")
    if isinstance(level, str):
        clean["level"] = level.upper()
    if "name" in record and "logger" not in clean:
        clean["logger"] = clean.get("name")
    return clean


def _log_sort_key(item: Mapping[str, Any]) -> float:
    timestamp = _parse_timestamp(item.get("timestamp"))
    return timestamp if timestamp is not None else float("-inf")


def _trace_status(record: Mapping[str, Any]) -> str:
    status = record.get("status")
    if isinstance(status, Mapping):
        code = status.get("status_code", status.get("statusCode"))
    else:
        code = status
    return str(code or "UNSET").upper()


def _trace_time(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        value = record.get("startTime" if key == "start_time" else "endTime")
    return _parse_timestamp(value)


def _safe_attribute_value(value: Any) -> JsonObject | str | int | float | bool | None:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _sanitize_text(value)[:2048]
    # Arrays are useful for a small number of OTel attributes but nested
    # objects can hide content, so only scalar values are exposed.
    return _sanitize_text(str(value))[:2048]


def _allowlisted_attributes(value: Any) -> JsonObject:
    if not isinstance(value, Mapping):
        return {}
    attributes: JsonObject = {}
    for key, item in value.items():
        name = str(key)
        if name not in TRACE_ATTRIBUTE_ALLOWLIST:
            continue
        if name in _TRACE_NUMERIC_ATTRIBUTE_KEYS and (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or (isinstance(item, float) and not math.isfinite(item))
        ):
            continue
        safe_value = _safe_attribute_value(item)
        attributes[name] = safe_value
    return attributes


def _trace_item(record: dict[str, Any]) -> JsonObject | None:
    context = _as_dict(record.get("context"))
    trace_id = (
        record.get("trace_id")
        or record.get("traceId")
        or context.get("trace_id")
        or context.get("traceId")
    )
    if not isinstance(trace_id, str) or not trace_id:
        return None
    span_id = (
        record.get("span_id")
        or record.get("spanId")
        or context.get("span_id")
        or context.get("spanId")
    )
    parent_id = (
        record.get("parent_id")
        or record.get("parentId")
        or context.get("parent_id")
        or context.get("parentId")
    )
    start = _trace_time(record, "start_time")
    end = _trace_time(record, "end_time")
    duration = (
        (end - start) * 1000
        if start is not None and end is not None and end >= start
        else 0.0
    )
    raw_name = record.get("name")
    name = _sanitize_text(raw_name)[:256] if isinstance(raw_name, str) else "span"
    kind = record.get("kind")
    if not isinstance(kind, str):
        kind = str(kind) if kind is not None else None
    events: list[JsonObject] = []
    raw_events = record.get("events")
    if isinstance(raw_events, list):
        for raw_event in raw_events[:100]:
            if not isinstance(raw_event, Mapping):
                continue
            event_name = raw_event.get("name")
            if not isinstance(event_name, str):
                continue
            event_item: JsonObject = {
                "name": _sanitize_text(event_name)[:256],
                "timestamp": _iso_timestamp(
                    _parse_timestamp(raw_event.get("timestamp"))
                ),
                "attributes": _allowlisted_attributes(raw_event.get("attributes")),
            }
            events.append(event_item)
    return {
        "trace_id": trace_id,
        "span_id": span_id if isinstance(span_id, str) else None,
        "parent_span_id": parent_id if isinstance(parent_id, str) else None,
        "name": name,
        "kind": kind,
        "start_time": _iso_timestamp(start),
        "end_time": _iso_timestamp(end),
        "duration_ms": round(duration, 2),
        "status": _trace_status(record),
        "span_count": 1,
        "spans": 1,
        "attributes": _allowlisted_attributes(record.get("attributes")),
        "events": events,
    }


def _trace_matches(item: JsonObject, search: str) -> bool:
    if not search:
        return True
    serialized = json.dumps(item, ensure_ascii=False, sort_keys=True).lower()
    return search.lower() in serialized


def _trace_item_timestamp(item: Mapping[str, Any]) -> float | None:
    """Use a span start time for overview windows, then its end time."""
    start = _parse_timestamp(item.get("start_time"))
    if start is not None:
        return start
    return _parse_timestamp(item.get("end_time"))


def _trace_sort_key(item: Mapping[str, Any]) -> float:
    timestamp = _trace_item_timestamp(item)
    return timestamp if timestamp is not None else float("-inf")


def _trace_status_for_spans(spans: list[JsonObject]) -> str:
    statuses = [str(span.get("status", "UNSET")).upper() for span in spans]
    if any(status in {"ERROR", "FAILED", "FAILURE"} for status in statuses):
        return "ERROR"
    return statuses[0] if statuses else "UNSET"


def _trace_summary(spans: list[JsonObject]) -> JsonObject:
    """Collapse spans for one trace into a safe operator-facing summary."""
    ordered = sorted(spans, key=_trace_sort_key, reverse=True)
    latest = ordered[0]
    starts = [
        timestamp
        for span in spans
        if (timestamp := _parse_timestamp(span.get("start_time"))) is not None
    ]
    ends = [
        timestamp
        for span in spans
        if (timestamp := _parse_timestamp(span.get("end_time"))) is not None
    ]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    duration = (end - start) * 1000 if start is not None and end is not None else 0.0
    attributes: JsonObject = {}
    for span in reversed(ordered):
        raw_attributes = span.get("attributes")
        if isinstance(raw_attributes, Mapping):
            attributes.update(raw_attributes)
    return {
        "trace_id": latest.get("trace_id"),
        "span_id": latest.get("span_id"),
        "parent_span_id": latest.get("parent_span_id"),
        "name": latest.get("name", "span"),
        "kind": latest.get("kind"),
        "start_time": _iso_timestamp(start),
        "end_time": _iso_timestamp(end),
        "duration_ms": round(max(0.0, duration), 2),
        "status": _trace_status_for_spans(ordered),
        "span_count": len(spans),
        "spans": len(spans),
        "attributes": attributes,
        "events": [],
    }


def _trace_groups(
    spans: Iterable[JsonObject],
) -> list[tuple[JsonObject, list[JsonObject]]]:
    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for span in spans:
        trace_id = span.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            grouped[trace_id].append(span)
    groups = [(_trace_summary(members), members) for members in grouped.values()]
    groups.sort(key=lambda group: _trace_sort_key(group[0]), reverse=True)
    return groups


def _trace_latency_stats(
    groups: Iterable[tuple[JsonObject, list[JsonObject]]],
) -> dict[str, float]:
    durations = sorted(
        float(summary.get("duration_ms", 0.0))
        for summary, _spans in groups
        if isinstance(summary.get("duration_ms"), (int, float))
        and float(summary.get("duration_ms", 0.0)) >= 0
    )
    if not durations:
        return {"average": 0.0, "p95": 0.0, "max": 0.0}
    p95_index = min(len(durations) - 1, math.ceil(len(durations) * 0.95) - 1)
    return {
        "average": round(sum(durations) / len(durations), 2),
        "p95": round(durations[p95_index], 2),
        "max": round(durations[-1], 2),
    }


class DashboardStore:
    """Read-only adapter over ADK sessions and local JSONL telemetry."""

    def __init__(
        self,
        session_db_path: Path,
        log_dir: Path,
        app_name: str = "blacki",
        identity_db_path: Path | None = None,
    ) -> None:
        self.session_db_path = Path(session_db_path)
        self.log_dir = Path(log_dir)
        self.app_name = str(app_name)
        self.identity_db_path = Path(identity_db_path) if identity_db_path else None

    async def _snapshot(self) -> _Snapshot:
        return await asyncio.to_thread(
            _load_snapshot_sync,
            self.session_db_path,
            self.app_name,
        )

    def _events_by_session(
        self, snapshot: _Snapshot
    ) -> dict[tuple[str, str], list[_EventRow]]:
        grouped: dict[tuple[str, str], list[_EventRow]] = defaultdict(list)
        for event in snapshot.events:
            grouped[(event.user_id, event.session_id)].append(event)
        for selected in grouped.values():
            selected.sort(
                key=lambda event: (
                    event.timestamp if event.timestamp is not None else 0.0,
                    event.event_id,
                )
            )
        return grouped

    async def get_overview(self, window: str) -> JsonObject:
        snapshot, log_scan, trace_scan = await asyncio.gather(
            self._snapshot(),
            asyncio.to_thread(_scan_jsonl_sync, self.log_dir / "blacki-telemetry.log"),
            asyncio.to_thread(_scan_jsonl_sync, self.log_dir / "blacki-traces.log"),
        )
        log_items = [_log_item(record) for record in log_scan.records]
        trace_span_items = [
            item for record in trace_scan.records if (item := _trace_item(record))
        ]
        normalized_window, since, until, window_warnings = _window_bounds(window)
        if since is None:
            known_times = [
                value
                for value in (
                    [event.timestamp for event in snapshot.events]
                    + [session.updated_at for session in snapshot.sessions]
                    + [_parse_timestamp(item.get("timestamp")) for item in log_items]
                    + [_trace_item_timestamp(item) for item in trace_span_items]
                )
                if value is not None
            ]
            until = max([until, *known_times])
        selected_events = [
            event
            for event in snapshot.events
            if _in_window(event.timestamp, since, until)
        ]
        selected_sessions = [
            session
            for session in snapshot.sessions
            if session.updated_at is None
            or _in_window(session.updated_at, since, until)
        ]
        selected_users = {session.user_id for session in selected_sessions}
        selected_users.update(event.user_id for event in selected_events)
        stats = _event_stats(selected_events)
        selected_log_items = [
            item
            for item in log_items
            if _in_window(
                _parse_timestamp(item.get("timestamp")),
                since,
                until,
            )
        ]
        selected_trace_spans = [
            item
            for item in trace_span_items
            if _in_window(_trace_item_timestamp(item), since, until)
        ]
        trace_groups = _trace_groups(selected_trace_spans)
        log_errors = sum(
            1
            for item in selected_log_items
            if str(item.get("level", "")).upper() in {"ERROR", "CRITICAL"}
        )
        trace_errors = sum(
            1
            for summary, _spans in trace_groups
            if str(summary.get("status", "")).upper() == "ERROR"
        )
        trace_latency = _trace_latency_stats(trace_groups)
        stats["users"] = len(selected_users)
        stats["active_users"] = len(selected_users)
        stats["user_count"] = len(selected_users)
        stats["sessions"] = len(selected_sessions)
        stats["session_count"] = len(selected_sessions)
        stats["reset_sessions"] = sum(
            1
            for session in selected_sessions
            if (_session_version(session.session_id)[1] or 0) > 1
        )
        stats["requests"] = stats["invocations"]
        stats["request_count"] = stats["invocations"]
        stats["log_records"] = len(selected_log_items)
        stats["trace_spans"] = len(selected_trace_spans)
        stats["traces"] = len(trace_groups)
        stats["trace_count"] = len(trace_groups)
        stats["avg_latency_ms"] = trace_latency["average"]
        stats["average_latency_ms"] = trace_latency["average"]
        stats["p95_latency_ms"] = trace_latency["p95"]
        stats["trace_latency_ms"] = trace_latency
        stats["log_errors"] = log_errors
        stats["trace_errors"] = trace_errors
        stats["errors"] = stats["request_errors"]
        stats["error_rate"] = (
            stats["errors"] / stats["requests"] if stats["requests"] else 0.0
        )
        warnings = list(snapshot.warnings)
        warnings.extend(_scan_warning(log_scan, missing=MISSING_TELEMETRY_WARNING))
        warnings.extend(_scan_warning(trace_scan, missing=MISSING_TRACES_WARNING))
        warnings.extend(window_warnings)
        warnings = list(dict.fromkeys(warnings))
        result: JsonObject = {
            "window": normalized_window,
            "from": _iso_timestamp(since),
            "to": _iso_timestamp(until),
            "stats": stats,
            "counts": stats,
            "activity": _activity_series(
                selected_events,
                since=since,
                until=until,
                window=normalized_window,
            ),
            "degraded": bool(warnings),
            "warnings": warnings,
        }
        # These top-level aliases keep the response convenient for tiny UIs
        # while the grouped stats/counts object remains the stable contract.
        for key in (
            "users",
            "sessions",
            "traces",
            "messages",
            "invocations",
            "requests",
            "tokens",
            "errors",
        ):
            result[key] = stats[key]
        result["error_rate"] = stats["error_rate"]
        result["metrics"] = stats
        return result

    async def list_users(self, search: str, limit: int, offset: int) -> JsonObject:
        snapshot = await self._snapshot()
        identities = (
            await asyncio.to_thread(
                _load_telegram_identities_sync, self.identity_db_path
            )
            if self.identity_db_path is not None
            else {}
        )
        query = bounded_search(search).lower()
        page_limit = clamp_limit(limit)
        page_offset = clamp_offset(offset)
        user_ids = sorted(
            {session.user_id for session in snapshot.sessions}
            | {event.user_id for event in snapshot.events}
        )
        items: list[JsonObject] = []
        for user_id in user_ids:
            identity_match = _TELEGRAM_DIRECT_USER_ID_RE.match(user_id)
            identity = (
                identities.get(int(identity_match.group("user_id")))
                if identity_match is not None
                else None
            )
            display_name = identity[0] if identity else None
            username = identity[1] if identity else None
            searchable_identity = " ".join(
                item for item in (display_name, username) if item
            ).lower()
            if (
                query
                and query not in user_id.lower()
                and query not in searchable_identity
            ):
                continue
            sessions = [
                session for session in snapshot.sessions if session.user_id == user_id
            ]
            sessions.sort(key=_session_sort_key, reverse=True)
            events = [event for event in snapshot.events if event.user_id == user_id]
            stats = _event_stats(events)
            latest = sessions[0] if sessions else None
            versions = [
                version
                for _, version in (
                    _session_version(item.session_id) for item in sessions
                )
                if version is not None
            ]
            items.append(
                {
                    "user_id": user_id,
                    "display_name": display_name,
                    "username": username,
                    "session_count": len(sessions),
                    "reset_count": max(0, max(versions, default=1) - 1),
                    "retained_history": len(sessions) > 1,
                    "latest_session_id": latest.session_id if latest else None,
                    "latest_update_at": (
                        _iso_timestamp(latest.updated_at) if latest else None
                    ),
                    "message_count": stats["messages"],
                    "invocation_count": stats["invocations"],
                    "tokens": stats["tokens"],
                    "error_count": stats["errors"],
                }
            )
        warnings = list(snapshot.warnings)
        return page_result(
            items[page_offset : page_offset + page_limit],
            total=len(items),
            limit=page_limit,
            offset=page_offset,
            warnings=warnings,
        )

    async def list_sessions(self, user_id: str, limit: int, offset: int) -> JsonObject:
        snapshot = await self._snapshot()
        page_limit = clamp_limit(limit)
        page_offset = clamp_offset(offset)
        selected = [
            session for session in snapshot.sessions if session.user_id == user_id
        ]
        selected.sort(key=_session_sort_key, reverse=True)
        grouped = self._events_by_session(snapshot)
        latest = selected[0].session_id if selected else None
        retained_ids = [session.session_id for session in selected[1:]]
        items = [
            _session_item(
                session,
                grouped.get((user_id, session.session_id), []),
                is_current=session.session_id == latest,
                retained_session_ids=retained_ids,
            )
            for session in selected
        ]
        return page_result(
            items[page_offset : page_offset + page_limit],
            total=len(items),
            limit=page_limit,
            offset=page_offset,
            warnings=list(snapshot.warnings),
        )

    async def get_session(self, user_id: str, session_id: str) -> JsonObject | None:
        snapshot = await self._snapshot()
        session = next(
            (
                item
                for item in snapshot.sessions
                if item.user_id == user_id and item.session_id == session_id
            ),
            None,
        )
        if session is None:
            return None
        all_user_sessions = [
            item for item in snapshot.sessions if item.user_id == user_id
        ]
        all_user_sessions.sort(key=_session_sort_key, reverse=True)
        latest = all_user_sessions[0].session_id if all_user_sessions else None
        retained_ids = [
            item.session_id
            for item in all_user_sessions
            if item.session_id != session_id
        ]
        selected_events = _session_events(
            snapshot.events, user_id=user_id, session_id=session_id
        )
        item = _session_item(
            session,
            selected_events,
            is_current=session.session_id == latest,
            retained_session_ids=retained_ids,
        )
        item["messages"] = [
            message for event in selected_events for message in _event_messages(event)
        ]
        item["degraded"] = bool(snapshot.warnings)
        item["warnings"] = list(snapshot.warnings)
        return item

    async def list_logs(self, level: str | None, search: str, limit: int) -> JsonObject:
        page_limit = clamp_limit(limit, maximum=MAX_LOG_PAGE_SIZE)
        query = bounded_search(search).lower()
        scan = await asyncio.to_thread(
            _scan_jsonl_sync, self.log_dir / "blacki-telemetry.log"
        )
        items: list[JsonObject] = []
        wanted_level = str(level).upper() if level else ""
        for record in scan.records:
            item = _log_item(record)
            item_level = str(item.get("level", "")).upper()
            if wanted_level and item_level != wanted_level:
                continue
            if (
                query
                and query
                not in json.dumps(item, ensure_ascii=False, sort_keys=True).lower()
            ):
                continue
            items.append(item)
        items.sort(key=_log_sort_key, reverse=True)
        warnings = _scan_warning(scan, missing=MISSING_TELEMETRY_WARNING)
        return page_result(
            items[:page_limit],
            total=len(items),
            limit=page_limit,
            offset=0,
            warnings=warnings,
        )

    async def list_traces(
        self, status: str | None, search: str, limit: int
    ) -> JsonObject:
        page_limit = clamp_limit(limit, maximum=MAX_TRACE_PAGE_SIZE)
        query = bounded_search(search)
        wanted_status = str(status).upper() if status else ""
        scan = await asyncio.to_thread(
            _scan_jsonl_sync, self.log_dir / "blacki-traces.log"
        )
        span_items: list[JsonObject] = []
        for record in scan.records:
            item = _trace_item(record)
            if item is None:
                continue
            span_items.append(item)
        items: list[JsonObject] = []
        for summary, spans in _trace_groups(span_items):
            if (
                wanted_status
                and str(summary.get("status", "")).upper() != wanted_status
            ):
                continue
            if query and not (
                _trace_matches(summary, query)
                or any(_trace_matches(span, query) for span in spans)
            ):
                continue
            items.append(summary)
        warnings = _scan_warning(scan, missing=MISSING_TRACES_WARNING)
        return page_result(
            items[:page_limit],
            total=len(items),
            limit=page_limit,
            offset=0,
            warnings=warnings,
        )

    async def get_trace(self, trace_id: str) -> JsonObject | None:
        scan = await asyncio.to_thread(
            _scan_jsonl_sync, self.log_dir / "blacki-traces.log"
        )
        requested = str(trace_id)
        matches: list[JsonObject] = []
        for record in scan.records:
            item = _trace_item(record)
            if item is not None and item.get("trace_id") == requested:
                matches.append(item)
        groups = _trace_groups(matches)
        if not groups:
            return None
        result, ordered_spans = groups[0]
        ordered_spans = sorted(ordered_spans, key=_trace_sort_key, reverse=True)
        result = dict(result)
        result["spans"] = ordered_spans
        result["span_count"] = len(ordered_spans)
        warnings = _scan_warning(scan, missing=MISSING_TRACES_WARNING)
        result["degraded"] = bool(warnings)
        result["warnings"] = warnings
        return result
