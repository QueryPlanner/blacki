"""ADK tools for user-owned common routes and scheduled traffic checks."""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

from google.adk.tools import ToolContext
from pydantic import BaseModel, ConfigDict, Field

from blacki.reminders import get_scheduler
from blacki.reminders.recurrence import get_next_trigger_time
from blacki.reminders.tools import _build_reminder_schedule
from blacki.utils.timezone import now_utc, utc_iso_seconds

from .scheduling import encode_route_update_event, parse_route_update_event
from .storage import (
    DuplicateRouteNameError,
    SavedRoute,
    SavedRouteLimitError,
    get_saved_route_storage,
    normalize_route_name,
)
from .tools import (
    GOOGLE_MAPS_ATTRIBUTION,
    _estimate_route,
    _normalize_travel_mode,
)

DEFAULT_SAVED_ROUTE_LIMIT = 20
DEFAULT_ROUTE_UPDATE_LIMIT = 10
MINIMUM_ROUTE_UPDATE_INTERVAL = timedelta(minutes=15)
MAX_ROUTE_NAME_LENGTH = 80
MAX_ROUTE_LABEL_LENGTH = 100

logger = logging.getLogger(__name__)


class CommonRouteChanges(BaseModel):
    """Fields that can be changed on a saved common route."""

    model_config = ConfigDict(extra="forbid")

    new_name: str | None = Field(
        default=None,
        description="New user-visible route name, or null to keep it unchanged.",
    )
    origin: str | None = Field(
        default=None,
        description="New origin address or place_id value, or null if unchanged.",
    )
    destination: str | None = Field(
        default=None,
        description="New destination address or place_id value, or null if unchanged.",
    )
    origin_label: str | None = Field(
        default=None,
        description="New user-authored origin label, or null if unchanged.",
    )
    destination_label: str | None = Field(
        default=None,
        description="New user-authored destination label, or null if unchanged.",
    )
    travel_mode: str | None = Field(
        default=None,
        description="New travel mode, or null if unchanged.",
    )
    avoid_tolls: bool | None = Field(
        default=None,
        description="New toll preference, or null if unchanged.",
    )
    avoid_highways: bool | None = Field(
        default=None,
        description="New highway preference, or null if unchanged.",
    )
    avoid_ferries: bool | None = Field(
        default=None,
        description="New ferry preference, or null if unchanged.",
    )


def _configured_limit(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def _owner_from_context(
    tool_context: ToolContext,
) -> tuple[str | None, dict[str, Any] | None]:
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    telegram_chat_id = tool_context.state.get("telegram_chat_id")
    telegram_chat_type = tool_context.state.get("telegram_chat_type")
    if telegram_chat_id and telegram_chat_type != "private":
        return None, {
            "status": "error",
            "error_code": "unsupported_context",
            "message": (
                "Saved routes are available only in a private Telegram chat "
                "because group chats share one conversation identity."
            ),
        }
    if not user_id:
        return None, {
            "status": "error",
            "error_code": "user_not_identified",
            "message": "Cannot access saved routes without a user identity.",
        }
    return str(user_id), None


def _validated_text(value: str, label: str, maximum: int) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{label} cannot be empty.")
    if len(normalized) > maximum:
        raise ValueError(f"{label} is too long (max {maximum} characters).")
    return normalized


def _explicit_place_id(value: str) -> str | None:
    prefix, separator, place_id = value.strip().partition(":")
    if prefix.casefold() == "place_id" and separator and place_id.strip():
        return place_id.strip()
    return None


def _resolved_place_id(
    value: str,
    result: dict[str, Any],
    endpoint: str,
) -> str | None:
    explicit = _explicit_place_id(value)
    if explicit:
        return explicit
    resolved = result.get("resolved_waypoints", {}).get(endpoint)
    if not isinstance(resolved, dict) or resolved.get("partial_match") is True:
        return None
    place_id = resolved.get("place_id")
    return place_id if isinstance(place_id, str) and place_id else None


def _route_settings(route: SavedRoute) -> dict[str, Any]:
    return {
        "travel_mode": route.travel_mode,
        "avoid_tolls": route.avoid_tolls,
        "avoid_highways": route.avoid_highways,
        "avoid_ferries": route.avoid_ferries,
    }


def _route_listing(route: SavedRoute) -> dict[str, Any]:
    return {
        "name": route.name,
        "origin_label": route.origin_label,
        "destination_label": route.destination_label,
        **_route_settings(route),
    }


def _route_summary(route: SavedRoute, result: dict[str, Any]) -> str:
    primary = result["routes"][0]
    delay = primary.get("traffic_delay_minutes")
    delay_text = (
        f", including about {delay:g} minutes of traffic delay"
        if isinstance(delay, int | float)
        else ""
    )
    return (
        f"{route.name}: {primary['duration_minutes']:g} minutes for "
        f"{primary['distance_kilometers']:g} km from {route.origin_label} to "
        f"{route.destination_label}{delay_text}. "
        f"{GOOGLE_MAPS_ATTRIBUTION} · updated {result['computed_at']}."
    )


async def _fresh_saved_route_estimate(route: SavedRoute) -> dict[str, Any]:
    api_key = os.environ.get("GOOGLE_MAPS_ROUTES_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "error",
            "error_code": "not_configured",
            "error": "GOOGLE_MAPS_ROUTES_API_KEY is not configured.",
        }
    return await _estimate_route(
        api_key=api_key,
        origin=f"place_id:{route.origin_place_id}",
        destination=f"place_id:{route.destination_place_id}",
        travel_mode=route.travel_mode,
        departure_time="now",
        traffic_model="BEST_GUESS" if route.travel_mode == "DRIVE" else "NONE",
        avoid_tolls=route.avoid_tolls,
        avoid_highways=route.avoid_highways,
        avoid_ferries=route.avoid_ferries,
        include_alternatives=False,
    )


async def save_common_route(
    route_name: str,
    origin: str,
    destination: str,
    origin_label: str,
    destination_label: str,
    travel_mode: str,
    avoid_tolls: bool,
    avoid_highways: bool,
    avoid_ferries: bool,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Save a reusable route after resolving both endpoints to Google place IDs.

    Use only when the user explicitly asks to save a route. Labels must be the
    user's own non-sensitive descriptions, such as Home and Office. Raw
    addresses and traffic results are never persisted.
    """
    owner_id, error = _owner_from_context(tool_context)
    if error:
        return error
    try:
        name = _validated_text(route_name, "Route name", MAX_ROUTE_NAME_LENGTH)
        start_label = _validated_text(
            origin_label, "Origin label", MAX_ROUTE_LABEL_LENGTH
        )
        end_label = _validated_text(
            destination_label, "Destination label", MAX_ROUTE_LABEL_LENGTH
        )
        api_key = os.environ.get("GOOGLE_MAPS_ROUTES_API_KEY", "").strip()
        if not api_key:
            return {
                "status": "error",
                "error_code": "not_configured",
                "message": "Google Maps Routes is not configured.",
            }
        estimate = await _estimate_route(
            api_key=api_key,
            origin=origin,
            destination=destination,
            travel_mode=travel_mode,
            departure_time="now",
            traffic_model=(
                "BEST_GUESS"
                if _normalize_travel_mode(travel_mode) == "DRIVE"
                else "NONE"
            ),
            avoid_tolls=avoid_tolls,
            avoid_highways=avoid_highways,
            avoid_ferries=avoid_ferries,
            include_alternatives=False,
        )
        if estimate["status"] != "success":
            return {
                "status": "error",
                "error_code": estimate.get("error_code", "route_unavailable"),
                "message": "The route could not be verified, so it was not saved.",
                "attribution": GOOGLE_MAPS_ATTRIBUTION,
            }
        origin_place_id = _resolved_place_id(origin, estimate, "origin")
        destination_place_id = _resolved_place_id(destination, estimate, "destination")
        if not origin_place_id or not destination_place_id:
            return {
                "status": "error",
                "error_code": "ambiguous_location",
                "message": (
                    "One or both locations were ambiguous. Provide a more precise "
                    "address or a Google place ID."
                ),
                "attribution": GOOGLE_MAPS_ATTRIBUTION,
            }
        timestamp = utc_iso_seconds(now_utc())
        saved = await get_saved_route_storage().create_route(
            SavedRoute(
                user_id=owner_id or "",
                name=name,
                normalized_name=normalize_route_name(name),
                origin_place_id=origin_place_id,
                destination_place_id=destination_place_id,
                origin_label=start_label,
                destination_label=end_label,
                travel_mode=estimate["travel_mode"],
                avoid_tolls=avoid_tolls,
                avoid_highways=avoid_highways,
                avoid_ferries=avoid_ferries,
                created_at=timestamp,
                updated_at=timestamp,
            ),
            _configured_limit(
                "GOOGLE_MAPS_SAVED_ROUTE_LIMIT", DEFAULT_SAVED_ROUTE_LIMIT
            ),
        )
        return {
            "status": "success",
            "route": _route_listing(saved),
            "message": f"Saved common route '{saved.name}'.",
            "attribution": GOOGLE_MAPS_ATTRIBUTION,
        }
    except (DuplicateRouteNameError, SavedRouteLimitError, ValueError) as exc:
        return {
            "status": "error",
            "error_code": "invalid_saved_route",
            "message": str(exc),
        }


async def list_common_routes(tool_context: ToolContext) -> dict[str, Any]:
    """List the current user's saved routes without returning place IDs."""
    owner_id, error = _owner_from_context(tool_context)
    if error:
        return error
    routes = await get_saved_route_storage().list_routes(owner_id or "")
    return {
        "status": "success",
        "routes": [_route_listing(route) for route in routes],
        "count": len(routes),
    }


async def check_common_route(
    route_reference: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Get a fresh Google Maps estimate for a saved route name or ``id:N``."""
    owner_id, error = _owner_from_context(tool_context)
    if error:
        return error
    route = await get_saved_route_storage().get_route(owner_id or "", route_reference)
    if route is None:
        return {
            "status": "error",
            "error_code": "route_not_found",
            "message": "That saved route was not found.",
        }
    estimate = await _fresh_saved_route_estimate(route)
    if estimate["status"] != "success":
        return {
            "status": "error",
            "error_code": estimate.get("error_code", "route_unavailable"),
            "message": "A fresh route estimate is unavailable right now.",
            "attribution": GOOGLE_MAPS_ATTRIBUTION,
        }
    return {
        "status": "success",
        "route": _route_listing(route),
        "estimate": {
            key: value
            for key, value in estimate.items()
            if key not in {"origin", "destination", "resolved_waypoints"}
        },
        "summary": _route_summary(route, estimate),
        "attribution": GOOGLE_MAPS_ATTRIBUTION,
    }


async def update_common_route(
    route_reference: str,
    changes: CommonRouteChanges,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Update a saved route only when the user explicitly requests the change."""
    owner_id, error = _owner_from_context(tool_context)
    if error:
        return error
    storage = get_saved_route_storage()
    route = await storage.get_route(owner_id or "", route_reference)
    if route is None or route.id is None:
        return {
            "status": "error",
            "error_code": "route_not_found",
            "message": "That saved route was not found.",
        }
    try:
        specified_changes = changes.model_dump(exclude_none=True)
        if not specified_changes:
            raise ValueError("Provide at least one saved-route field to update.")
        name = (
            _validated_text(changes.new_name, "Route name", MAX_ROUTE_NAME_LENGTH)
            if changes.new_name is not None
            else route.name
        )
        start_label = (
            _validated_text(
                changes.origin_label, "Origin label", MAX_ROUTE_LABEL_LENGTH
            )
            if changes.origin_label is not None
            else route.origin_label
        )
        end_label = (
            _validated_text(
                changes.destination_label,
                "Destination label",
                MAX_ROUTE_LABEL_LENGTH,
            )
            if changes.destination_label is not None
            else route.destination_label
        )
        origin_place_id = route.origin_place_id
        destination_place_id = route.destination_place_id
        normalized_mode = route.travel_mode
        avoid_tolls = (
            changes.avoid_tolls
            if changes.avoid_tolls is not None
            else route.avoid_tolls
        )
        avoid_highways = (
            changes.avoid_highways
            if changes.avoid_highways is not None
            else route.avoid_highways
        )
        avoid_ferries = (
            changes.avoid_ferries
            if changes.avoid_ferries is not None
            else route.avoid_ferries
        )
        route_fields_changed = bool(
            {
                "origin",
                "destination",
                "travel_mode",
                "avoid_tolls",
                "avoid_highways",
                "avoid_ferries",
            }
            & specified_changes.keys()
        )
        if route_fields_changed:
            origin = changes.origin or f"place_id:{route.origin_place_id}"
            destination = (
                changes.destination or f"place_id:{route.destination_place_id}"
            )
            normalized_mode = _normalize_travel_mode(
                changes.travel_mode or route.travel_mode
            )
            api_key = os.environ.get("GOOGLE_MAPS_ROUTES_API_KEY", "").strip()
            if not api_key:
                return {
                    "status": "error",
                    "error_code": "not_configured",
                    "message": "Google Maps Routes is not configured.",
                }
            estimate = await _estimate_route(
                api_key=api_key,
                origin=origin,
                destination=destination,
                travel_mode=normalized_mode,
                departure_time="now",
                traffic_model=("BEST_GUESS" if normalized_mode == "DRIVE" else "NONE"),
                avoid_tolls=avoid_tolls,
                avoid_highways=avoid_highways,
                avoid_ferries=avoid_ferries,
                include_alternatives=False,
            )
            if estimate["status"] != "success":
                return {
                    "status": "error",
                    "error_code": estimate.get("error_code", "route_unavailable"),
                    "message": (
                        "The proposed route could not be verified; no changes saved."
                    ),
                    "attribution": GOOGLE_MAPS_ATTRIBUTION,
                }
            resolved_origin_place_id = _resolved_place_id(origin, estimate, "origin")
            resolved_destination_place_id = _resolved_place_id(
                destination, estimate, "destination"
            )
            if not resolved_origin_place_id or not resolved_destination_place_id:
                return {
                    "status": "error",
                    "error_code": "ambiguous_location",
                    "message": (
                        "One or both locations were ambiguous. No changes were saved."
                    ),
                    "attribution": GOOGLE_MAPS_ATTRIBUTION,
                }
            origin_place_id = resolved_origin_place_id
            destination_place_id = resolved_destination_place_id
        updated = await storage.update_route(
            owner_id or "",
            route.id,
            {
                "name": name,
                "normalized_name": normalize_route_name(name),
                "origin_place_id": origin_place_id,
                "destination_place_id": destination_place_id,
                "origin_label": start_label,
                "destination_label": end_label,
                "travel_mode": normalized_mode,
                "avoid_tolls": int(avoid_tolls),
                "avoid_highways": int(avoid_highways),
                "avoid_ferries": int(avoid_ferries),
                "updated_at": utc_iso_seconds(now_utc()),
            },
        )
        if updated is None:
            return {
                "status": "error",
                "error_code": "route_not_found",
                "message": "That saved route was not found.",
            }
        return {
            "status": "success",
            "route": _route_listing(updated),
            "message": f"Updated common route '{updated.name}'.",
            "attribution": GOOGLE_MAPS_ATTRIBUTION,
        }
    except (DuplicateRouteNameError, ValueError) as exc:
        return {
            "status": "error",
            "error_code": "invalid_saved_route",
            "message": str(exc),
        }


async def delete_common_route(
    route_reference: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Delete a saved route and its active traffic-update reminders."""
    owner_id, error = _owner_from_context(tool_context)
    if error:
        return error
    storage = get_saved_route_storage()
    route = await storage.get_route(owner_id or "", route_reference)
    if route is None or route.id is None:
        return {
            "status": "error",
            "error_code": "route_not_found",
            "message": "That saved route was not found.",
        }
    scheduler = get_scheduler()
    reminders = await scheduler.get_user_reminders(owner_id or "")
    cancelled = 0
    for reminder in reminders:
        event = parse_route_update_event(reminder.message)
        if (
            event
            and event.route_id == route.id
            and reminder.id is not None
            and await scheduler.delete_reminder(reminder.id, owner_id or "")
        ):
            cancelled += 1
    deleted = await storage.delete_route(owner_id or "", route.id)
    if not deleted:
        return {
            "status": "error",
            "error_code": "route_not_found",
            "message": "That saved route was not found.",
        }
    return {
        "status": "success",
        "message": f"Deleted common route '{route.name}'.",
        "cancelled_updates": cancelled,
    }


async def schedule_common_route_update(
    route_reference: str,
    recurrence: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Schedule recurring fresh traffic checks for a saved route.

    ``recurrence`` must be a five-field cron expression in the application
    timezone. Checks must be at least 15 minutes apart.
    """
    owner_id, error = _owner_from_context(tool_context)
    if error:
        return error
    route = await get_saved_route_storage().get_route(owner_id or "", route_reference)
    if route is None or route.id is None:
        return {
            "status": "error",
            "error_code": "route_not_found",
            "message": "That saved route was not found.",
        }
    try:
        schedule = _build_reminder_schedule(
            reminder_datetime=None,
            recurrence=recurrence,
        )
        next_after = get_next_trigger_time(
            schedule["recurrence_rule"],
            schedule["timezone_name"],
            reference_time=schedule["trigger_time"] + timedelta(seconds=1),
        )
    except ValueError as exc:
        return {
            "status": "error",
            "error_code": "invalid_schedule",
            "message": str(exc),
        }
    if next_after - schedule["trigger_time"] < MINIMUM_ROUTE_UPDATE_INTERVAL:
        return {
            "status": "error",
            "error_code": "schedule_too_frequent",
            "message": "Route updates must be scheduled at least 15 minutes apart.",
        }
    scheduler = get_scheduler()
    reminders = await scheduler.get_user_reminders(owner_id or "")
    route_events = [
        (reminder, event)
        for reminder in reminders
        if (event := parse_route_update_event(reminder.message)) is not None
    ]
    if any(
        event.route_id == route.id
        and reminder.recurrence_rule == schedule["recurrence_rule"]
        for reminder, event in route_events
    ):
        return {
            "status": "error",
            "error_code": "duplicate_schedule",
            "message": "That traffic-update schedule already exists.",
        }
    limit = _configured_limit(
        "GOOGLE_MAPS_ROUTE_UPDATE_LIMIT", DEFAULT_ROUTE_UPDATE_LIMIT
    )
    if len(route_events) >= limit:
        return {
            "status": "error",
            "error_code": "schedule_limit_reached",
            "message": f"You can have at most {limit} active route updates.",
        }
    try:
        reminder_id = await scheduler.schedule_reminder(
            user_id=owner_id or "",
            message=encode_route_update_event(route.id),
            trigger_time=schedule["trigger_time"],
            recurrence_rule=schedule["recurrence_rule"],
            recurrence_text=schedule["recurrence_text"],
            timezone_name=schedule["timezone_name"],
        )
    except Exception:
        logger.exception("Failed to create a scheduled route update")
        return {
            "status": "error",
            "error_code": "schedule_failed",
            "message": (
                "The route is still saved, but its traffic update could not be "
                "scheduled. Please try again."
            ),
        }
    return {
        "status": "success",
        "reminder_id": reminder_id,
        "route_name": route.name,
        "recurrence": schedule["recurrence_text"],
        "next_update": utc_iso_seconds(schedule["trigger_time"]),
        "message": f"Scheduled traffic updates for '{route.name}'.",
    }
