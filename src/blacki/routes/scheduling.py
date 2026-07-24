"""Versioned reminder envelopes for scheduled saved-route checks."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

ROUTE_UPDATE_EVENT_KIND: Literal["blacki.route_traffic_update"] = (
    "blacki.route_traffic_update"
)


class ScheduledRouteUpdate(BaseModel):
    """A minimal persisted event that contains no address or traffic result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["blacki.route_traffic_update"] = ROUTE_UPDATE_EVENT_KIND
    version: Literal[1] = 1
    route_id: PositiveInt


def encode_route_update_event(route_id: int) -> str:
    """Serialize a stable scheduled-route event."""
    return ScheduledRouteUpdate(route_id=route_id).model_dump_json()


def parse_route_update_event(value: str) -> ScheduledRouteUpdate | None:
    """Parse a route event, returning ``None`` for normal reminder messages."""
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(decoded, dict) or decoded.get("kind") != ROUTE_UPDATE_EVENT_KIND:
        return None
    try:
        return ScheduledRouteUpdate.model_validate(decoded)
    except ValidationError:
        return None


def build_scheduled_route_prompt(event: ScheduledRouteUpdate) -> str:
    """Build a controlled agent instruction from a validated event."""
    return (
        "[Scheduled Route Update]\n"
        f'Call check_common_route with route_reference "id:{event.route_id}". '
        "If it succeeds, send the returned summary verbatim. If the route no "
        "longer exists or the lookup fails, explain that briefly without "
        "guessing traffic, distance, or travel time."
    )
