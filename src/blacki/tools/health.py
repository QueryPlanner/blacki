"""Read-only Google Health tools exposed to the authenticated Telegram agent."""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from blacki.container import get_container
from blacki.health.service import summarize_stored_health


async def get_health_summary(
    tool_context: ToolContext,
    days: int = 7,
) -> dict[str, Any]:
    """Read a user's normalized Google Health summary.

    This tool is intentionally read-only. It never calls Google directly; the
    scheduled or explicit refresh path owns provider access and persistence.
    The archive field reports acquisition status and stored observation counts,
    not archived values. Empty days means no normalized summaries in that date
    window; it does not mean the full-history archive is empty. This tool cannot
    search historical archive payloads or report values from those payloads.
    Use query_health_data for those values, even while imports are in progress.
    """
    if days < 1 or days > 14:
        return {"status": "error", "error": "days must be between 1 and 14"}

    user_id = getattr(tool_context, "user_id", None)
    if not isinstance(user_id, str) or not user_id:
        return {"status": "error", "error": "Missing Telegram user identity"}

    state = getattr(tool_context, "state", None)
    chat_type = state.get("telegram_chat_type") if state is not None else None
    if chat_type != "private":
        return {
            "status": "error",
            "error": (
                "Google Health summaries are available only in private Telegram chats"
            ),
        }

    try:
        storage = get_container().google_health_storage
    except RuntimeError:
        return {"status": "error", "error": "Health storage is not initialized"}
    return await summarize_stored_health(storage, user_id, days=days)


async def query_health_data(
    tool_context: ToolContext,
    data_type: str,
    start_date: str | None = None,
    end_date: str | None = None,
    fields: list[str] | None = None,
    limit: int = 50,
    cursor: str | None = None,
    operation: str = "records",
    aggregation: str = "count",
    metric: str | None = None,
    aggregation_basis: str = "latest_records",
) -> dict[str, Any]:
    """Query the user's stored Health archive directly with structured filters.

    Use this for specific dates or metrics, including when get_health_summary
    has no days. Available records can be read while historical imports run.
    Report available values with partial-coverage caveats; do not claim that
    all daily data must finish building before any record can be read.

    One readable Google Health data_type per call, e.g. sleep,
    daily-resting-heart-rate, daily-heart-rate-variability, steps, or weight.
    Dates are inclusive YYYY-MM-DD, default to the latest seven UTC days, and
    must span at most 31 days. Metadata such as profile accepts no date range.
    Physical timestamps use UTC dates; daily/civil-only records keep their
    provider calendar date. Sleep uses session end; other intervals use start.

    fields selects at most eight JSON paths relative to the provider record,
    e.g. ["sleep.summary.minutesAsleep"]. Omit it for the data component.
    Results contain at most 100 records and 32 KiB. Use next_cursor unchanged
    with identical filters to read the next page. Ingestion can change results
    between pages. No arbitrary SQL, Python, provider request, or file export.

    operation="aggregate" supports count, sum, avg, min, max. Numeric metric
    names are checked against documented provider fields; an invalid metric
    returns supported_metrics. Example: metric=
    "dailyHeartRateVariability.averageHeartRateVariabilityMilliseconds".
    Protocol int64 strings are accepted only for documented int64 fields.
    Other text, booleans, missing values, and nonfinite numbers are excluded.

    Identifiable records use their latest observed version. Anonymous payloads
    may overlap sources or versions. Their sums/averages require explicitly
    choosing aggregation_basis="observations"; disclose that basis to the user.
    count without a metric counts matched archived observations, not activities.
    The archive preserves observations after provider-side deletions. Results
    remain private to this user's host archive and bounded tool response.
    """
    from blacki.health.query import query_archive

    user_id = tool_context.user_id
    if not user_id:
        return {"status": "error", "error": "Missing Telegram user identity"}
    if tool_context.state.get("telegram_chat_type") != "private":
        return {
            "status": "error",
            "error": "Health queries require a private Telegram chat",
        }
    try:
        storage = get_container().google_health_storage
    except RuntimeError:
        return {"status": "error", "error": "Health storage is not initialized"}
    return await query_archive(
        storage,
        user_id,
        data_type=data_type,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        limit=limit,
        cursor=cursor,
        operation=operation,
        aggregation=aggregation,
        metric=metric,
        aggregation_basis=aggregation_basis,
    )
