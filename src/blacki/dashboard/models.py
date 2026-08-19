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
# A bounded tail keeps scans recent and predictable even when the process has
# accumulated months of local telemetry.  The scanner retains only the last
# 10,000 valid records from this tail.
MAX_JSONL_TAIL_BYTES = 16 * 1024 * 1024
MAX_ACCEPTED_JSONL_RECORDS = 10_000
MAX_DATABASE_ROWS = 100_000

# Only these OpenTelemetry attributes are useful for an operator dashboard.
# In particular, no prompt/content/tool argument attributes are included.
TRACE_ATTRIBUTE_ALLOWLIST = frozenset(
    {
        "service.name",
        "service.namespace",
        "service.version",
        "service.instance.id",
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.operation.name",
        "gen_ai.agent.name",
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
        "http.method",
        "http.route",
        "http.status_code",
        "rpc.system",
        "error.type",
        "error.code",
        "blacki.user_id",
        "blacki.session_id",
        "blacki.invocation_id",
        "gcp.vertex.agent.session_id",
        "gcp.vertex.agent.invocation_id",
        "gen_ai.conversation.id",
        "gen_ai.tool.name",
        "agent.name",
        "llm.model_name",
        "session.id",
        "user.id",
    }
)

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
