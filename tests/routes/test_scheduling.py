"""Tests for versioned route-update reminder envelopes."""

import pytest
from pydantic import ValidationError

from blacki.routes.scheduling import (
    ScheduledRouteUpdate,
    build_scheduled_route_prompt,
    encode_route_update_event,
    parse_route_update_event,
)


def test_event_round_trip_and_prompt() -> None:
    encoded = encode_route_update_event(17)
    event = parse_route_update_event(encoded)

    assert event == ScheduledRouteUpdate(route_id=17)
    assert '"route_id":17' in encoded
    prompt = build_scheduled_route_prompt(event)
    assert 'route_reference "id:17"' in prompt
    assert "summary verbatim" in prompt


@pytest.mark.parametrize(
    "value",
    [
        "ordinary reminder",
        "[]",
        '{"kind":"something_else","route_id":1}',
        '{"kind":"blacki.route_traffic_update","version":2,"route_id":1}',
        '{"kind":"blacki.route_traffic_update","version":1,"route_id":0}',
        '{"kind":"blacki.route_traffic_update","version":1,"route_id":1,"x":2}',
    ],
)
def test_non_events_and_invalid_events_are_rejected(value: str) -> None:
    assert parse_route_update_event(value) is None


def test_event_requires_positive_route_id() -> None:
    with pytest.raises(ValidationError):
        encode_route_update_event(0)
