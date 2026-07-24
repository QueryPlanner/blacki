"""Tests for the read-only Google Maps Routes tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import create_autospec, patch

import httpx
import pytest
from conftest import MockState, MockToolContext
from google.adk.tools import FunctionTool, ToolContext

from blacki.routes import client as routes_client
from blacki.routes.client import (
    COMPUTE_ROUTES_URL,
    RoutesAPIError,
    reset_routes_client_cache,
)
from blacki.routes.tools import (
    MAX_SCENARIOS,
    RouteScenario,
    RouteValidationError,
    _build_payload,
    _duration_seconds,
    _normalize_response,
    _normalize_route,
    _resolved_waypoint,
    compare_route_scenarios,
    get_route_estimate,
)

FIXED_NOW = datetime(2026, 7, 24, 8, 30, tzinfo=UTC)


def _tool_context() -> ToolContext:
    return cast(ToolContext, MockToolContext(state=MockState({})))


def _strict_client() -> Any:
    client = create_autospec(httpx.AsyncClient, instance=True, spec_set=True)
    routes_client._routes_client = client
    return client


def _response(status_code: int, json: object | None = None) -> httpx.Response:
    request = httpx.Request("POST", COMPUTE_ROUTES_URL)
    if json is None:
        return httpx.Response(status_code, request=request)
    return httpx.Response(status_code, request=request, json=json)


def _route_response(
    *,
    duration: str = "1800s",
    static_duration: str | None = "1200s",
) -> dict[str, object]:
    route: dict[str, object] = {
        "distanceMeters": 12500,
        "duration": duration,
        "description": "Main Road",
        "routeLabels": ["DEFAULT_ROUTE", 7],
        "warnings": ["Road closures may apply.", None],
    }
    if static_duration is not None:
        route["staticDuration"] = static_duration
    return {
        "routes": [route],
        "fallbackInfo": {
            "routingMode": "FALLBACK_TRAFFIC_AWARE",
            "reason": "LATENCY_EXCEEDED",
        },
        "geocodingResults": {
            "origin": {"placeId": "origin-id", "partialMatch": True},
            "destination": {"placeId": "destination-id"},
        },
    }


def _scenario(
    label: str,
    *,
    travel_mode: str = "DRIVE",
    traffic_model: str = "BEST_GUESS",
) -> RouteScenario:
    return RouteScenario(
        label=label,
        travel_mode=travel_mode,
        departure_time="now",
        traffic_model=traffic_model,
        avoid_tolls=False,
        avoid_highways=False,
        avoid_ferries=False,
    )


@pytest.fixture(autouse=True)
async def reset_shared_client() -> AsyncGenerator[None, None]:
    await reset_routes_client_cache()
    yield
    await reset_routes_client_cache()


class TestGetRouteEstimate:
    """Public estimate-tool behavior and normalized output."""

    @pytest.mark.asyncio
    async def test_missing_api_key_is_safe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_MAPS_ROUTES_API_KEY", raising=False)

        result = await get_route_estimate(
            "A",
            "B",
            "DRIVE",
            "now",
            "BEST_GUESS",
            False,
            False,
            False,
            False,
            _tool_context(),
        )

        assert result["status"] == "error"
        assert result["error_code"] == "not_configured"
        assert result["routes"] == []
        assert result["attribution"] == "Google Maps"

    @pytest.mark.asyncio
    async def test_success_normalizes_traffic_without_exposing_place_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", "test-key")
        client = _strict_client()
        client.post.return_value = _response(200, _route_response())

        with patch("blacki.routes.tools.now_utc", return_value=FIXED_NOW):
            result = await get_route_estimate(
                " place_id:origin-id ",
                " Destination address ",
                "driving",
                "2026-07-24T14:00:00+05:30",
                "best guess",
                True,
                False,
                True,
                True,
                _tool_context(),
            )

        assert result["status"] == "success"
        assert "origin" not in result
        assert "destination" not in result
        assert result["travel_mode"] == "DRIVE"
        assert result["departure_time"] == "2026-07-24T08:30:00Z"
        assert result["computed_at"] == "2026-07-24T08:30:00+00:00"
        assert result["attribution"] == "Google Maps"
        assert result["modifier_warning"] == (
            "Avoid options are preferences, not guarantees."
        )
        assert result["mode_warning"] is None
        assert result["fallback"] == {
            "routing_mode": "FALLBACK_TRAFFIC_AWARE",
            "reason": "LATENCY_EXCEEDED",
        }
        assert "resolved_waypoints" not in result
        assert "origin-id" not in str(result)
        assert "destination-id" not in str(result)

        route = result["routes"][0]
        assert route["route_labels"] == ["DEFAULT_ROUTE"]
        assert route["warnings"] == ["Road closures may apply."]
        assert route["distance_meters"] == 12500
        assert route["distance_kilometers"] == 12.5
        assert route["duration_minutes"] == 30.0
        assert route["static_duration_minutes"] == 20.0
        assert route["traffic_delay_minutes"] == 10.0
        assert route["traffic_delay_percent"] == 50.0

        request_payload = client.post.await_args.kwargs["json"]
        assert request_payload == {
            "origin": {"placeId": "origin-id"},
            "destination": {"address": "Destination address"},
            "travelMode": "DRIVE",
            "computeAlternativeRoutes": True,
            "departureTime": "2026-07-24T08:30:00Z",
            "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
            "trafficModel": "BEST_GUESS",
            "routeModifiers": {
                "avoidTolls": True,
                "avoidHighways": False,
                "avoidFerries": True,
            },
        }

    @pytest.mark.asyncio
    async def test_api_error_uses_stable_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", "bad-key")
        client = _strict_client()
        client.post.return_value = _response(403)

        result = await get_route_estimate(
            "A",
            "B",
            "DRIVE",
            "now",
            "BEST_GUESS",
            False,
            False,
            False,
            False,
            _tool_context(),
        )

        assert result["status"] == "error"
        assert result["error_code"] == "authentication_failed"
        assert "bad-key" not in str(result)


class TestPayloadValidation:
    """Local validation prevents avoidable and invalid billable requests."""

    @pytest.mark.parametrize(
        ("travel_mode", "traffic_model", "expected_mode", "routing_preference"),
        [
            ("driving", "BEST_GUESS", "DRIVE", "TRAFFIC_AWARE_OPTIMAL"),
            ("DRIVE", "NONE", "DRIVE", "TRAFFIC_AWARE"),
            ("motorcycle", "NONE", "TWO_WHEELER", "TRAFFIC_AWARE"),
            ("walking", "NONE", "WALK", None),
            ("biking", "NONE", "BICYCLE", None),
            ("TRANSIT", "NONE", "TRANSIT", None),
        ],
    )
    def test_supported_modes_build_expected_routing(
        self,
        travel_mode: str,
        traffic_model: str,
        expected_mode: str,
        routing_preference: str | None,
    ) -> None:
        payload, mode, departure = _build_payload(
            origin="A",
            destination="B",
            travel_mode=travel_mode,
            departure_time="now",
            traffic_model=traffic_model,
            avoid_tolls=False,
            avoid_highways=False,
            avoid_ferries=False,
            include_alternatives=False,
        )

        assert mode == expected_mode
        assert departure == "now"
        assert payload.get("routingPreference") == routing_preference
        assert "departureTime" not in payload

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"travel_mode": "spaceship"}, "Travel mode must be"),
            ({"departure_time": ""}, "Departure time must"),
            ({"departure_time": "tomorrow morning"}, "Departure time must"),
            (
                {"departure_time": "2026-07-24T08:00:00"},
                "must include a timezone",
            ),
            ({"traffic_model": "average"}, "Traffic model must"),
            (
                {"travel_mode": "WALK", "traffic_model": "BEST_GUESS"},
                "only when travel mode is DRIVE",
            ),
            (
                {
                    "travel_mode": "BICYCLE",
                    "traffic_model": "NONE",
                    "avoid_tolls": True,
                },
                "require DRIVE or TWO_WHEELER",
            ),
            ({"origin": ""}, "Origin cannot be empty"),
            ({"destination": ""}, "Destination cannot be empty"),
            ({"origin": "x" * 513}, "Origin is too long"),
            ({"origin": "place_id:  "}, "Origin place ID cannot be empty"),
        ],
    )
    def test_invalid_inputs_are_rejected_locally(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        request: dict[str, object] = {
            "origin": "A",
            "destination": "B",
            "travel_mode": "DRIVE",
            "departure_time": "now",
            "traffic_model": "BEST_GUESS",
            "avoid_tolls": False,
            "avoid_highways": False,
            "avoid_ferries": False,
            "include_alternatives": False,
        }
        request.update(kwargs)

        with pytest.raises(RouteValidationError, match=message):
            _build_payload(**request)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_public_tool_returns_validation_error_without_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", "test-key")
        client = _strict_client()

        result = await get_route_estimate(
            "",
            "B",
            "DRIVE",
            "now",
            "BEST_GUESS",
            False,
            False,
            False,
            False,
            _tool_context(),
        )

        assert result["error_code"] == "invalid_input"
        client.post.assert_not_awaited()


class TestResponseNormalization:
    """Response validation, optional fields, and warning behavior."""

    @pytest.mark.parametrize("value", [None, 10, "10"])
    def test_duration_requires_protobuf_duration(self, value: object) -> None:
        with pytest.raises(RoutesAPIError, match="valid duration"):
            _duration_seconds(value, "duration")

    def test_duration_rejects_non_numeric_seconds(self) -> None:
        with pytest.raises(RoutesAPIError, match="invalid duration"):
            _duration_seconds("tenseconds", "duration")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ({}, None),
            ({"placeId": 3}, None),
            (
                {"placeId": "abc", "partialMatch": False},
                {"place_id": "abc", "partial_match": False},
            ),
        ],
    )
    def test_resolved_waypoint_validation(
        self, value: object, expected: dict[str, object] | None
    ) -> None:
        assert _resolved_waypoint(value) == expected

    @pytest.mark.parametrize(
        ("route", "message"),
        [
            (None, "invalid route"),
            ({"distanceMeters": True, "duration": "1s"}, "route distance"),
            ({"distanceMeters": 1, "duration": "bad"}, "route duration"),
        ],
    )
    def test_invalid_routes_are_rejected(self, route: object, message: str) -> None:
        with pytest.raises(RoutesAPIError, match=message):
            _normalize_route(route, 0)

    def test_optional_route_fields_and_zero_static_duration(self) -> None:
        normalized = _normalize_route(
            {
                "distanceMeters": 10,
                "duration": "3.5s",
                "staticDuration": "0s",
                "description": 4,
                "routeLabels": "DEFAULT_ROUTE",
                "warnings": "warning",
            },
            2,
        )

        assert normalized["route_index"] == 2
        assert normalized["description"] == ""
        assert normalized["route_labels"] == []
        assert normalized["warnings"] == []
        assert normalized["duration_seconds"] == 3.5
        assert normalized["static_duration_seconds"] == 0.0
        assert normalized["traffic_delay_seconds"] == 3.5
        assert normalized["traffic_delay_percent"] is None

    def test_missing_static_duration_has_no_traffic_delta(self) -> None:
        normalized = _normalize_route(
            {
                "distanceMeters": 100,
                "duration": "60s",
                "routeLabels": ["DEFAULT_ROUTE", 3],
                "warnings": ["Use caution", 4],
            },
            0,
        )

        assert normalized["static_duration_seconds"] is None
        assert normalized["static_duration_minutes"] is None
        assert normalized["traffic_delay_seconds"] is None
        assert normalized["traffic_delay_minutes"] is None
        assert normalized["route_labels"] == ["DEFAULT_ROUTE"]
        assert normalized["warnings"] == ["Use caution"]

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            ({}, "invalid routes collection"),
            ({"routes": "invalid"}, "invalid routes collection"),
            ({"routes": []}, "could not find a route"),
        ],
    )
    def test_invalid_route_collections(
        self, data: dict[str, object], message: str
    ) -> None:
        with pytest.raises(RoutesAPIError, match=message):
            _normalize_response(
                data,
                origin="A",
                destination="B",
                travel_mode="DRIVE",
                departure_time="now",
                used_route_modifiers=False,
            )

    @pytest.mark.parametrize("mode", ["WALK", "BICYCLE", "TWO_WHEELER"])
    def test_beta_modes_include_warning(self, mode: str) -> None:
        result = _normalize_response(
            {
                "routes": [
                    {
                        "distanceMeters": 100,
                        "duration": "60s",
                    }
                ],
                "geocodingResults": "invalid",
                "fallbackInfo": "invalid",
            },
            origin="A",
            destination="B",
            travel_mode=mode,
            departure_time="now",
            used_route_modifiers=False,
        )

        assert "beta" in result["mode_warning"]
        assert result["resolved_waypoints"] == {
            "origin": None,
            "destination": None,
        }
        assert result["fallback"] is None
        assert result["modifier_warning"] is None


class TestCompareRouteScenarios:
    """Scenario limits, partial failures, and concurrency."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("count", [0, MAX_SCENARIOS + 1])
    async def test_scenario_count_is_bounded(self, count: int) -> None:
        result = await compare_route_scenarios(
            "A",
            "B",
            [_scenario(str(index)) for index in range(count)],
            _tool_context(),
        )

        assert result["status"] == "error"
        assert result["error_code"] == "invalid_input"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scenarios",
        [
            [_scenario(" ")],
            [_scenario("Morning"), _scenario(" morning ")],
        ],
    )
    async def test_labels_must_be_nonempty_and_unique(
        self, scenarios: list[RouteScenario]
    ) -> None:
        result = await compare_route_scenarios("A", "B", scenarios, _tool_context())

        assert result["status"] == "error"
        assert result["error_code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_missing_api_key_is_safe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_MAPS_ROUTES_API_KEY", raising=False)

        result = await compare_route_scenarios(
            "A", "B", [_scenario("Now")], _tool_context()
        )

        assert result["error_code"] == "not_configured"

    @pytest.mark.asyncio
    async def test_partial_and_all_failed_statuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", "test-key")
        client = _strict_client()
        client.post.return_value = _response(200, _route_response())

        partial = await compare_route_scenarios(
            "A",
            "B",
            [
                _scenario("Drive"),
                _scenario("Invalid walk", travel_mode="WALK"),
            ],
            _tool_context(),
        )
        failed = await compare_route_scenarios(
            "A",
            "B",
            [
                _scenario("Walk", travel_mode="WALK"),
                _scenario("Transit", travel_mode="TRANSIT"),
            ],
            _tool_context(),
        )

        assert partial["status"] == "partial"
        assert partial["successful_scenarios"] == 1
        assert partial["scenarios"][0]["label"] == "Drive"
        assert partial["scenarios"][1]["error_code"] == "invalid_input"
        assert failed["status"] == "error"
        assert failed["successful_scenarios"] == 0
        assert client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_success_preserves_order_and_bounds_concurrency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", "test-key")
        client = _strict_client()
        active = 0
        max_active = 0
        three_started = asyncio.Event()

        async def post(*_args: object, **_kwargs: object) -> httpx.Response:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            if active == 3:
                three_started.set()
            await three_started.wait()
            await asyncio.sleep(0)
            active -= 1
            return _response(200, _route_response())

        client.post.side_effect = post
        labels = ["Now", "Later", "Optimistic", "Pessimistic", "No tolls"]

        result = await compare_route_scenarios(
            " A ",
            " B ",
            [_scenario(label) for label in labels],
            _tool_context(),
        )

        assert result["status"] == "success"
        assert "origin" not in result
        assert "destination" not in result
        assert result["scenario_count"] == 5
        assert result["successful_scenarios"] == 5
        assert [scenario["label"] for scenario in result["scenarios"]] == labels
        assert max_active == 3
        assert "resolved_waypoints" not in str(result)

    def test_adk_declaration_keeps_structured_scenario_schema(self) -> None:
        declaration = FunctionTool(compare_route_scenarios)._get_declaration()

        assert declaration is not None
        schema = declaration.parameters_json_schema
        assert schema is not None
        assert schema["properties"]["scenarios"]["type"] == "array"
        assert "RouteScenario" in schema["$defs"]
