"""Tests for sensitive external-tool privacy helpers."""

from typing import Any

import pytest

from blacki.utils.privacy import (
    REDACTED_ROUTE_DETAILS,
    redact_route_tool_payload,
    route_data_redaction_enabled,
)


@pytest.mark.parametrize(
    ("value", "expected"), [(None, False), ("   ", False), ("key", True)]
)
def test_route_data_redaction_tracks_routes_configuration(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    expected: bool,
) -> None:
    if value is None:
        monkeypatch.delenv("GOOGLE_MAPS_ROUTES_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", value)

    assert route_data_redaction_enabled() is expected


def test_non_route_payload_is_unchanged() -> None:
    payload = {"query": "public information"}

    assert redact_route_tool_payload("exa_search", payload) is payload


def test_route_payload_preserves_only_safe_operational_metadata() -> None:
    payload: dict[str, Any] = {
        "status": "success",
        "origin": "home-address-canary",
        "destination": "office-address-canary",
        "resolved_waypoints": {"origin": {"place_id": "place-id-canary"}},
        "scenario_count": 2,
        "successful_scenarios": 1,
        "attribution": "Google Maps",
    }

    redacted = redact_route_tool_payload("get_route_estimate", payload)

    assert redacted == {
        "details": REDACTED_ROUTE_DETAILS,
        "status": "success",
        "scenario_count": 2,
        "successful_scenarios": 1,
        "attribution": "Google Maps",
    }
    assert "canary" not in str(redacted)


def test_saved_route_payload_is_redacted() -> None:
    payload = {
        "status": "success",
        "route": {"origin_label": "home-canary"},
        "count": 1,
        "cancelled_updates": 2,
    }

    assert redact_route_tool_payload("list_common_routes", payload) == {
        "details": REDACTED_ROUTE_DETAILS,
        "status": "success",
        "count": 1,
        "cancelled_updates": 2,
    }
