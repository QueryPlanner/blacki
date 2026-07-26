"""Privacy helpers for sensitive external-tool data."""

from __future__ import annotations

import os
from typing import Any

ROUTE_TOOL_NAMES = frozenset(
    {
        "get_route_estimate",
        "compare_route_scenarios",
    }
)
REDACTED_ROUTE_DETAILS = "<route details redacted>"


def route_data_redaction_enabled() -> bool:
    """Return whether this process has enabled the Google Routes integration."""
    return bool(os.environ.get("GOOGLE_MAPS_ROUTES_API_KEY", "").strip())


def redact_route_tool_payload(
    tool_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Remove locations, place IDs, and route content from observable payloads."""
    if tool_name not in ROUTE_TOOL_NAMES:
        return payload

    redacted: dict[str, Any] = {"details": REDACTED_ROUTE_DETAILS}
    for key in (
        "status",
        "error_code",
        "scenario_count",
        "successful_scenarios",
        "attribution",
    ):
        if key in payload:
            redacted[key] = payload[key]
    return redacted
