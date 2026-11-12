"""Small, JSON-friendly types and limits used by the local dashboard.

The dashboard deliberately does not expose ADK/Pydantic objects.  Keeping the
public values as ordinary dictionaries also makes the FastAPI boundary stable
when ADK adds fields to its persisted event model.
"""

from __future__ import annotations

from typing import Any, TypeAlias

# ``Any`` is deliberate here: these values are handed to FastAPI's JSON
# encoder, and recursive aliases make mypy reject useful nested responses.
JsonValue: TypeAlias = Any
JsonObject: TypeAlias = dict[str, JsonValue]

# These are intentionally conservative.  The dashboard is an operator view,
# not an export API, and a bounded result protects a small VPS from a large log
# file or a very active bot.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
MAX_LOG_PAGE_SIZE = 200
MAX_TRACE_PAGE_SIZE = 200
MAX_SEARCH_LENGTH = 128
MAX_JSONL_LINE_BYTES = 1 * 1024 * 1024
# A model request (system prompt, tool declarations, and full history) is stored
# on a single trace line, so the private trace view needs a larger bound than
# application logs to keep long sessions inspectable.
MAX_TRACE_JSONL_LINE_BYTES = 8 * 1024 * 1024
# The trace file is never trimmed, and each model-call span repeats the whole
# conversation.  The dashboard reads only this much of its end, so a page load
# stays bounded while recent model calls remain visible.
MAX_TRACE_JSONL_TAIL_BYTES = 128 * 1024 * 1024
# A bounded tail keeps scans recent and predictable even when the process has
# accumulated months of local telemetry.  The scanner retains only the last
# 10,000 valid records from this tail.
MAX_JSONL_TAIL_BYTES = 16 * 1024 * 1024
MAX_ACCEPTED_JSONL_RECORDS = 10_000
MAX_DATABASE_ROWS = 100_000

DEGRADED_DATABASE_WARNING = "Session data is temporarily unavailable."
MISSING_EVENTS_WARNING = "Session event history is unavailable."
MISSING_TELEMETRY_WARNING = "Telemetry log is unavailable."
MISSING_TRACES_WARNING = "Trace log is unavailable."


def page_result(
    items: list[JsonObject],
    *,
    total: int,
    limit: int,
    offset: int,
    warnings: list[str] | None = None,
) -> JsonObject:
    """Build a consistent paginated response without leaking internals."""
    safe_warnings = list(warnings or [])
    return {
        "items": items,
        "total": max(0, int(total)),
        "limit": max(0, int(limit)),
        "offset": max(0, int(offset)),
        "degraded": bool(safe_warnings),
        "warnings": safe_warnings,
    }


def clamp_limit(value: int, *, maximum: int = MAX_PAGE_SIZE) -> int:
    """Clamp a caller-supplied page size to a safe positive range."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_PAGE_SIZE
    return min(max(1, parsed), maximum)


def clamp_offset(value: int) -> int:
    """Clamp an offset and prevent pathological integer values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return min(max(0, parsed), MAX_DATABASE_ROWS)


def bounded_search(value: str | None) -> str:
    """Normalize a plain substring search to its documented maximum length."""
    if not value:
        return ""
    return str(value)[:MAX_SEARCH_LENGTH]
