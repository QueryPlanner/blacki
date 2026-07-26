"""Read-only tools backed by the Google Maps Routes API."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

from google.adk.tools import ToolContext
from pydantic import BaseModel, ConfigDict, Field

from blacki.utils.timezone import now_utc, utc_iso_seconds

from .client import RoutesAPIError, compute_routes

SUPPORTED_TRAVEL_MODES = frozenset(
    {"DRIVE", "WALK", "BICYCLE", "TWO_WHEELER", "TRANSIT"}
)
SUPPORTED_TRAFFIC_MODELS = frozenset({"BEST_GUESS", "OPTIMISTIC", "PESSIMISTIC"})
MODE_ALIASES = {
    "DRIVING": "DRIVE",
    "WALKING": "WALK",
    "BIKING": "BICYCLE",
    "CYCLING": "BICYCLE",
    "MOTORCYCLE": "TWO_WHEELER",
}
MAX_SCENARIOS = 5
SCENARIO_CONCURRENCY = 3
GOOGLE_MAPS_ATTRIBUTION = "Google Maps"


class RouteScenario(BaseModel):
    """One explicitly named route-comparison scenario."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Short unique label for this scenario.")
    travel_mode: str = Field(
        description=(
            "DRIVE, WALK, BICYCLE, TWO_WHEELER, or TRANSIT. Common aliases such "
            "as driving and walking are accepted."
        )
    )
    departure_time: str = Field(
        description="Use 'now' or an RFC 3339 timestamp with a timezone offset."
    )
    traffic_model: str = Field(
        description=(
            "BEST_GUESS, OPTIMISTIC, or PESSIMISTIC for DRIVE; use NONE for "
            "other travel modes."
        )
    )
    avoid_tolls: bool = Field(description="Prefer routes without tolls.")
    avoid_highways: bool = Field(description="Prefer routes without highways.")
    avoid_ferries: bool = Field(description="Prefer routes without ferries.")


class RouteValidationError(ValueError):
    """Invalid user-controlled route input."""


def _error_result(
    code: str,
    message: str,
    origin: str,
    destination: str,
) -> dict[str, Any]:
    """Build the stable error contract returned to the agent."""
    return {
        "status": "error",
        "error_code": code,
        "error": message,
        "origin": origin,
        "destination": destination,
        "routes": [],
        "attribution": GOOGLE_MAPS_ATTRIBUTION,
    }


def _normalize_travel_mode(value: str) -> str:
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    normalized = MODE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_TRAVEL_MODES:
        supported = ", ".join(sorted(SUPPORTED_TRAVEL_MODES))
        raise RouteValidationError(f"Travel mode must be one of: {supported}.")
    return normalized


def _normalize_departure_time(value: str) -> str | None:
    normalized = value.strip()
    if normalized.lower() == "now":
        return None
    if not normalized:
        raise RouteValidationError(
            "Departure time must be 'now' or an RFC 3339 timestamp."
        )

    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RouteValidationError(
            "Departure time must be 'now' or an RFC 3339 timestamp."
        ) from exc
    if parsed.tzinfo is None:
        raise RouteValidationError("Departure time must include a timezone offset.")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_traffic_model(value: str, travel_mode: str) -> str | None:
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized == "NONE":
        return None
    if travel_mode != "DRIVE":
        raise RouteValidationError(
            "Traffic models are available only when travel mode is DRIVE."
        )
    if normalized not in SUPPORTED_TRAFFIC_MODELS:
        supported = ", ".join(sorted(SUPPORTED_TRAFFIC_MODELS))
        raise RouteValidationError(
            f"Traffic model must be NONE or one of: {supported}."
        )
    return normalized


def _waypoint(value: str, label: str) -> dict[str, str]:
    normalized = value.strip()
    if not normalized:
        raise RouteValidationError(f"{label} cannot be empty.")
    if len(normalized) > 512:
        raise RouteValidationError(f"{label} is too long.")
    prefix, separator, place_id = normalized.partition(":")
    if prefix.lower() == "place_id" and separator:
        if not place_id.strip():
            raise RouteValidationError(f"{label} place ID cannot be empty.")
        return {"placeId": place_id.strip()}
    return {"address": normalized}


def _build_payload(
    *,
    origin: str,
    destination: str,
    travel_mode: str,
    departure_time: str,
    traffic_model: str,
    avoid_tolls: bool,
    avoid_highways: bool,
    avoid_ferries: bool,
    include_alternatives: bool,
) -> tuple[dict[str, Any], str, str]:
    mode = _normalize_travel_mode(travel_mode)
    normalized_departure = _normalize_departure_time(departure_time)
    normalized_traffic_model = _normalize_traffic_model(traffic_model, mode)

    has_route_modifiers = avoid_tolls or avoid_highways or avoid_ferries
    if has_route_modifiers and mode not in {"DRIVE", "TWO_WHEELER"}:
        raise RouteValidationError(
            "Toll, highway, and ferry avoidance require DRIVE or TWO_WHEELER."
        )

    payload: dict[str, Any] = {
        "origin": _waypoint(origin, "Origin"),
        "destination": _waypoint(destination, "Destination"),
        "travelMode": mode,
        "computeAlternativeRoutes": include_alternatives,
    }
    if normalized_departure is not None:
        payload["departureTime"] = normalized_departure

    if mode == "DRIVE":
        if normalized_traffic_model is None:
            payload["routingPreference"] = "TRAFFIC_AWARE"
        else:
            payload["routingPreference"] = "TRAFFIC_AWARE_OPTIMAL"
            payload["trafficModel"] = normalized_traffic_model
    elif mode == "TWO_WHEELER":
        payload["routingPreference"] = "TRAFFIC_AWARE"

    if has_route_modifiers:
        payload["routeModifiers"] = {
            "avoidTolls": avoid_tolls,
            "avoidHighways": avoid_highways,
            "avoidFerries": avoid_ferries,
        }

    requested_departure = normalized_departure or "now"
    return payload, mode, requested_departure


def _duration_seconds(value: object, field_name: str) -> float:
    if not isinstance(value, str) or not value.endswith("s"):
        raise RoutesAPIError(
            "invalid_response",
            f"Google Maps Routes omitted a valid {field_name}.",
        )
    try:
        return float(value[:-1])
    except ValueError as exc:
        raise RoutesAPIError(
            "invalid_response",
            f"Google Maps Routes returned an invalid {field_name}.",
        ) from exc


def _resolved_waypoint(data: object) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    place_id = data.get("placeId")
    if not isinstance(place_id, str):
        return None
    return {
        "place_id": place_id,
        "partial_match": data.get("partialMatch") is True,
    }


def _normalize_route(
    route: object,
    route_index: int,
) -> dict[str, Any]:
    if not isinstance(route, dict):
        raise RoutesAPIError(
            "invalid_response",
            "Google Maps Routes returned an invalid route.",
        )

    distance = route.get("distanceMeters")
    if not isinstance(distance, int) or isinstance(distance, bool):
        raise RoutesAPIError(
            "invalid_response",
            "Google Maps Routes omitted a valid route distance.",
        )
    duration = _duration_seconds(route.get("duration"), "route duration")
    raw_static_duration = route.get("staticDuration")
    static_duration = (
        _duration_seconds(raw_static_duration, "static route duration")
        if raw_static_duration is not None
        else None
    )
    traffic_delay = duration - static_duration if static_duration is not None else None
    traffic_delay_percent = (
        round((traffic_delay / static_duration) * 100, 1)
        if (
            traffic_delay is not None
            and static_duration is not None
            and static_duration > 0
        )
        else None
    )

    labels = route.get("routeLabels")
    warnings = route.get("warnings")
    description = route.get("description")
    return {
        "route_index": route_index,
        "route_labels": (
            [label for label in labels if isinstance(label, str)]
            if isinstance(labels, list)
            else []
        ),
        "description": description if isinstance(description, str) else "",
        "distance_meters": distance,
        "distance_kilometers": round(distance / 1000, 2),
        "duration_seconds": duration,
        "duration_minutes": round(duration / 60, 1),
        "static_duration_seconds": static_duration,
        "static_duration_minutes": (
            round(static_duration / 60, 1) if static_duration is not None else None
        ),
        "traffic_delay_seconds": traffic_delay,
        "traffic_delay_minutes": (
            round(traffic_delay / 60, 1) if traffic_delay is not None else None
        ),
        "traffic_delay_percent": traffic_delay_percent,
        "warnings": (
            [warning for warning in warnings if isinstance(warning, str)]
            if isinstance(warnings, list)
            else []
        ),
    }


def _normalize_response(
    data: dict[str, Any],
    *,
    origin: str,
    destination: str,
    travel_mode: str,
    departure_time: str,
    used_route_modifiers: bool,
) -> dict[str, Any]:
    raw_routes = data.get("routes")
    if not isinstance(raw_routes, list):
        raise RoutesAPIError(
            "invalid_response",
            "Google Maps Routes returned an invalid routes collection.",
        )
    if not raw_routes:
        raise RoutesAPIError(
            "no_route",
            "Google Maps Routes could not find a route for these locations.",
        )

    geocoding = data.get("geocodingResults")
    geocoding = geocoding if isinstance(geocoding, dict) else {}
    fallback = data.get("fallbackInfo")
    fallback = fallback if isinstance(fallback, dict) else {}

    mode_warning = (
        "Walking, bicycling, and two-wheeler routes are beta; use caution."
        if travel_mode in {"WALK", "BICYCLE", "TWO_WHEELER"}
        else None
    )
    return {
        "status": "success",
        "origin": origin,
        "destination": destination,
        "travel_mode": travel_mode,
        "departure_time": departure_time,
        "computed_at": utc_iso_seconds(now_utc()),
        "routes": [
            _normalize_route(route, index) for index, route in enumerate(raw_routes)
        ],
        "resolved_waypoints": {
            "origin": _resolved_waypoint(geocoding.get("origin")),
            "destination": _resolved_waypoint(geocoding.get("destination")),
        },
        "fallback": (
            {
                "routing_mode": fallback.get("routingMode"),
                "reason": fallback.get("reason"),
            }
            if fallback
            else None
        ),
        "mode_warning": mode_warning,
        "modifier_warning": (
            "Avoid options are preferences, not guarantees."
            if used_route_modifiers
            else None
        ),
        "attribution": GOOGLE_MAPS_ATTRIBUTION,
    }


def _public_route_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep exact endpoints and provider place IDs out of public tool results."""
    for key in ("origin", "destination", "resolved_waypoints"):
        result.pop(key, None)
    return result


async def _estimate_route(
    *,
    api_key: str,
    origin: str,
    destination: str,
    travel_mode: str,
    departure_time: str,
    traffic_model: str,
    avoid_tolls: bool,
    avoid_highways: bool,
    avoid_ferries: bool,
    include_alternatives: bool,
) -> dict[str, Any]:
    try:
        payload, mode, requested_departure = _build_payload(
            origin=origin,
            destination=destination,
            travel_mode=travel_mode,
            departure_time=departure_time,
            traffic_model=traffic_model,
            avoid_tolls=avoid_tolls,
            avoid_highways=avoid_highways,
            avoid_ferries=avoid_ferries,
            include_alternatives=include_alternatives,
        )
        response = await compute_routes(payload, api_key)
        return _normalize_response(
            response,
            origin=origin.strip(),
            destination=destination.strip(),
            travel_mode=mode,
            departure_time=requested_departure,
            used_route_modifiers=avoid_tolls or avoid_highways or avoid_ferries,
        )
    except RouteValidationError as exc:
        return _error_result("invalid_input", str(exc), origin, destination)
    except RoutesAPIError as exc:
        return _error_result(exc.code, str(exc), origin, destination)


async def get_route_estimate(
    origin: str,
    destination: str,
    travel_mode: str,
    departure_time: str,
    traffic_model: str,
    avoid_tolls: bool,
    avoid_highways: bool,
    avoid_ferries: bool,
    include_alternatives: bool,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Get a fresh route distance, ETA, traffic delay, and optional alternatives.

    Use this for one route between an origin and destination. For current
    driving traffic, set travel_mode to DRIVE, departure_time to now, and
    traffic_model to BEST_GUESS. Use NONE as the traffic model for non-driving
    modes. Prefix a Google place ID with ``place_id:``; otherwise locations are
    treated as addresses. Avoid options express preferences, not guarantees.

    Args:
        origin: Starting address or a place ID prefixed with ``place_id:``.
        destination: Ending address or a place ID prefixed with ``place_id:``.
        travel_mode: DRIVE, WALK, BICYCLE, TWO_WHEELER, or TRANSIT.
        departure_time: ``now`` or an RFC 3339 timestamp with timezone offset.
        traffic_model: BEST_GUESS, OPTIMISTIC, PESSIMISTIC, or NONE.
        avoid_tolls: Whether to prefer routes without tolls.
        avoid_highways: Whether to prefer routes without highways.
        avoid_ferries: Whether to prefer routes without ferries.
        include_alternatives: Whether to request alternate routes.

    Returns:
        A dictionary with normalized routes, durations, traffic delay, warnings,
        and Google Maps attribution.
    """
    _ = tool_context
    api_key = os.environ.get("GOOGLE_MAPS_ROUTES_API_KEY", "").strip()
    if not api_key:
        return _public_route_result(
            _error_result(
                "not_configured",
                "GOOGLE_MAPS_ROUTES_API_KEY is not configured.",
                origin,
                destination,
            )
        )
    result = await _estimate_route(
        api_key=api_key,
        origin=origin,
        destination=destination,
        travel_mode=travel_mode,
        departure_time=departure_time,
        traffic_model=traffic_model,
        avoid_tolls=avoid_tolls,
        avoid_highways=avoid_highways,
        avoid_ferries=avoid_ferries,
        include_alternatives=include_alternatives,
    )
    return _public_route_result(result)


async def compare_route_scenarios(
    origin: str,
    destination: str,
    scenarios: list[RouteScenario],
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Compare up to five fresh route scenarios for the same endpoints.

    Use this only when the user asks to compare departure times, travel modes,
    traffic assumptions, or avoid options. Each scenario must have a short
    unique label and all of its route settings. The tool returns the primary
    route for each scenario and preserves individual failures.

    Args:
        origin: Starting address or a place ID prefixed with ``place_id:``.
        destination: Ending address or a place ID prefixed with ``place_id:``.
        scenarios: One to five explicitly named route scenarios.

    Returns:
        A dictionary with overall status and one normalized result per scenario.
    """
    _ = tool_context
    if not 1 <= len(scenarios) <= MAX_SCENARIOS:
        return _public_route_result(
            _error_result(
                "invalid_input",
                f"Provide between 1 and {MAX_SCENARIOS} route scenarios.",
                origin,
                destination,
            )
        )

    labels = [scenario.label.strip() for scenario in scenarios]
    if any(not label for label in labels):
        return _public_route_result(
            _error_result(
                "invalid_input",
                "Every route scenario requires a non-empty label.",
                origin,
                destination,
            )
        )
    normalized_labels = [label.casefold() for label in labels]
    if len(set(normalized_labels)) != len(normalized_labels):
        return _public_route_result(
            _error_result(
                "invalid_input",
                "Route scenario labels must be unique.",
                origin,
                destination,
            )
        )

    api_key = os.environ.get("GOOGLE_MAPS_ROUTES_API_KEY", "").strip()
    if not api_key:
        return _public_route_result(
            _error_result(
                "not_configured",
                "GOOGLE_MAPS_ROUTES_API_KEY is not configured.",
                origin,
                destination,
            )
        )

    semaphore = asyncio.Semaphore(SCENARIO_CONCURRENCY)

    async def run_scenario(
        scenario: RouteScenario,
        label: str,
    ) -> dict[str, Any]:
        async with semaphore:
            result = await _estimate_route(
                api_key=api_key,
                origin=origin,
                destination=destination,
                travel_mode=scenario.travel_mode,
                departure_time=scenario.departure_time,
                traffic_model=scenario.traffic_model,
                avoid_tolls=scenario.avoid_tolls,
                avoid_highways=scenario.avoid_highways,
                avoid_ferries=scenario.avoid_ferries,
                include_alternatives=False,
            )
        return {"label": label, **_public_route_result(result)}

    results = await asyncio.gather(
        *(
            run_scenario(scenario, label)
            for scenario, label in zip(scenarios, labels, strict=True)
        )
    )
    success_count = sum(result.get("status") == "success" for result in results)
    if success_count == len(results):
        status = "success"
    elif success_count:
        status = "partial"
    else:
        status = "error"

    return {
        "status": status,
        "scenario_count": len(results),
        "successful_scenarios": success_count,
        "scenarios": results,
        "attribution": GOOGLE_MAPS_ATTRIBUTION,
    }
