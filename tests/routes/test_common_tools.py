# mypy: disable-error-code="no-untyped-def"
"""Integration tests for saved-route ADK tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock, create_autospec, patch

import aiosqlite
import pytest
from conftest import MockState, MockToolContext
from google.adk.tools import FunctionTool, ToolContext

from blacki.container import (
    get_container,
    reset_container_for_tests,
    set_container_from_connection,
)
from blacki.reminders import scheduler as scheduler_module
from blacki.reminders.scheduler import ReminderScheduler
from blacki.reminders.storage import Reminder
from blacki.routes.client import RoutesAPIError
from blacki.routes.common_tools import (
    CommonRouteChanges,
    _configured_limit,
    check_common_route,
    delete_common_route,
    list_common_routes,
    save_common_route,
    schedule_common_route_update,
    update_common_route,
)


def _context(
    user_id: str | None = "user-1",
    state: dict[str, str] | None = None,
) -> ToolContext:
    return cast(
        ToolContext,
        MockToolContext(user_id=user_id, state=MockState(state or {})),
    )


def _routes_response(
    *,
    origin_place_id: str = "origin-place",
    destination_place_id: str = "destination-place",
    partial_origin: bool = False,
    static_duration: str | None = "1500s",
) -> dict[str, object]:
    route: dict[str, object] = {
        "distanceMeters": 12000,
        "duration": "1800s",
    }
    if static_duration is not None:
        route["staticDuration"] = static_duration
    return {
        "routes": [route],
        "geocodingResults": {
            "origin": {
                "placeId": origin_place_id,
                "partialMatch": partial_origin,
            },
            "destination": {"placeId": destination_place_id},
        },
    }


@pytest.fixture(autouse=True)
async def route_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[None, None]:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    container = set_container_from_connection(conn, asyncio.Lock())
    await container.saved_route_storage.initialize()
    await container.reminder_storage.initialize()
    scheduler_module._scheduler = None
    monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", "test-routes-key")
    yield
    scheduler_module._scheduler = None
    reset_container_for_tests()
    await conn.close()


async def _save(route_name: str = "Commute", user_id: str = "user-1"):
    with patch(
        "blacki.routes.tools.compute_routes",
        new=AsyncMock(return_value=_routes_response()),
    ):
        return await save_common_route(
            route_name,
            "1 Home Street, Pune",
            "2 Office Road, Pune",
            "Home",
            "Office",
            "DRIVE",
            False,
            False,
            False,
            _context(user_id),
        )


class TestSavedRouteAccess:
    def test_adk_declarations_support_saved_route_schemas(self) -> None:
        for tool in (
            save_common_route,
            list_common_routes,
            check_common_route,
            update_common_route,
            delete_common_route,
            schedule_common_route_update,
        ):
            declaration = FunctionTool(tool)._get_declaration()
            assert declaration is not None
            if tool is list_common_routes:
                assert declaration.parameters_json_schema is None
            else:
                assert declaration.parameters_json_schema is not None

        update_schema = FunctionTool(update_common_route)._get_declaration()
        assert update_schema is not None
        parameters = update_schema.parameters_json_schema
        assert parameters is not None
        assert "CommonRouteChanges" in parameters["$defs"]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("", 20), ("invalid", 20), ("0", 20), ("7", 7)],
    )
    def test_configured_limit_uses_safe_positive_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
        expected: int,
    ) -> None:
        monkeypatch.setenv("TEST_ROUTE_LIMIT", value)
        assert _configured_limit("TEST_ROUTE_LIMIT", 20) == expected

    @pytest.mark.asyncio
    async def test_save_persists_place_ids_but_not_addresses(self) -> None:
        result = await _save()

        assert result["status"] == "success"
        assert "place" not in str(result)
        routes = await list_common_routes(_context())
        assert routes["routes"] == [
            {
                "name": "Commute",
                "origin_label": "Home",
                "destination_label": "Office",
                "travel_mode": "DRIVE",
                "avoid_tolls": False,
                "avoid_highways": False,
                "avoid_ferries": False,
            }
        ]
        rows = list(
            await get_container().conn.execute_fetchall(
                "SELECT origin_place_id, destination_place_id FROM saved_routes"
            )
        )
        assert tuple(rows[0]) == ("origin-place", "destination-place")

    @pytest.mark.asyncio
    async def test_owner_isolation_and_private_telegram_requirement(self) -> None:
        await _save()

        assert (await list_common_routes(_context("other-user")))["routes"] == []
        group = _context(
            "telegram-chat--99",
            {"telegram_chat_id": "-99", "telegram_chat_type": "group"},
        )
        result = await list_common_routes(group)
        assert result["error_code"] == "unsupported_context"

        missing = await list_common_routes(_context(None))
        assert missing["error_code"] == "user_not_identified"

    @pytest.mark.asyncio
    async def test_mutating_and_check_tools_reject_group_context(self) -> None:
        group = _context(
            "telegram-chat--99",
            {"telegram_chat_id": "-99", "telegram_chat_type": "supergroup"},
        )
        changes = CommonRouteChanges()

        results = [
            await save_common_route(
                "Route", "A", "B", "A", "B", "DRIVE", False, False, False, group
            ),
            await check_common_route("Route", group),
            await update_common_route("Route", changes, group),
            await delete_common_route("Route", group),
            await schedule_common_route_update("Route", "0 8 * * *", group),
        ]

        assert {result["error_code"] for result in results} == {"unsupported_context"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kwargs", "error_code"),
        [
            ({"route_name": " "}, "invalid_saved_route"),
            ({"origin_label": "x" * 101}, "invalid_saved_route"),
            ({"destination_label": ""}, "invalid_saved_route"),
        ],
    )
    async def test_save_validates_user_fields(
        self, kwargs: dict[str, str], error_code: str
    ) -> None:
        request = {
            "route_name": "Commute",
            "origin": "Home, Pune",
            "destination": "Office, Pune",
            "origin_label": "Home",
            "destination_label": "Office",
        }
        request.update(kwargs)
        result = await save_common_route(
            request["route_name"],
            request["origin"],
            request["destination"],
            request["origin_label"],
            request["destination_label"],
            "DRIVE",
            False,
            False,
            False,
            _context(),
        )
        assert result["error_code"] == error_code

    @pytest.mark.asyncio
    async def test_save_rejects_missing_key_provider_error_and_ambiguity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_MAPS_ROUTES_API_KEY")
        missing = await save_common_route(
            "Commute",
            "A",
            "B",
            "Home",
            "Office",
            "DRIVE",
            False,
            False,
            False,
            _context(),
        )
        assert missing["error_code"] == "not_configured"

        monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", "key")
        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(side_effect=RoutesAPIError("quota_exceeded", "quota")),
        ):
            provider = await save_common_route(
                "Commute",
                "A",
                "B",
                "Home",
                "Office",
                "DRIVE",
                False,
                False,
                False,
                _context(),
            )
        assert provider["error_code"] == "quota_exceeded"

        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(return_value=_routes_response(partial_origin=True)),
        ):
            ambiguous = await save_common_route(
                "Commute",
                "A",
                "B",
                "Home",
                "Office",
                "DRIVE",
                False,
                False,
                False,
                _context(),
            )
        assert ambiguous["error_code"] == "ambiguous_location"

    @pytest.mark.asyncio
    async def test_save_duplicate_and_configured_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (await _save())["status"] == "success"
        duplicate = await _save(" COMMUTE ")
        assert duplicate["error_code"] == "invalid_saved_route"

        monkeypatch.setenv("GOOGLE_MAPS_SAVED_ROUTE_LIMIT", "1")
        limited = await _save("Gym")
        assert "at most 1" in limited["message"]


class TestCheckAndUpdate:
    @pytest.mark.asyncio
    async def test_check_returns_fresh_attributed_summary(self) -> None:
        await _save()
        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(return_value=_routes_response()),
        ):
            result = await check_common_route("commute", _context())

        assert result["status"] == "success"
        assert result["attribution"] == "Google Maps"
        assert "30 minutes" in result["summary"]
        assert "5 minutes of traffic delay" in result["summary"]
        assert "place" not in str(result)

        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(return_value=_routes_response(static_duration=None)),
        ):
            no_delay = await check_common_route("id:1", _context())
        assert "traffic delay" not in no_delay["summary"]

    @pytest.mark.asyncio
    async def test_check_handles_missing_key_failure_and_missing_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (await check_common_route("missing", _context()))["error_code"] == (
            "route_not_found"
        )
        await _save()
        monkeypatch.delenv("GOOGLE_MAPS_ROUTES_API_KEY")
        unavailable = await check_common_route("commute", _context())
        assert unavailable["error_code"] == "not_configured"

    @pytest.mark.asyncio
    async def test_update_changes_route_and_resolves_new_endpoint(self) -> None:
        await _save()
        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(
                return_value=_routes_response(destination_place_id="new-destination")
            ),
        ):
            result = await update_common_route(
                "Commute",
                CommonRouteChanges(
                    new_name="Weekday commute",
                    destination="3 New Office Road, Pune",
                    destination_label="New Office",
                    travel_mode="WALK",
                    avoid_tolls=False,
                    avoid_highways=False,
                    avoid_ferries=False,
                ),
                _context(),
            )

        assert result["status"] == "success"
        assert result["route"]["name"] == "Weekday commute"
        assert result["route"]["destination_label"] == "New Office"
        assert result["route"]["travel_mode"] == "WALK"

    @pytest.mark.asyncio
    async def test_update_name_and_label_does_not_call_provider(self) -> None:
        await _save()
        provider = AsyncMock(return_value=_routes_response())
        with patch("blacki.routes.tools.compute_routes", new=provider):
            result = await update_common_route(
                "Commute",
                CommonRouteChanges(
                    new_name="Weekday commute",
                    origin_label="My home",
                ),
                _context(),
            )

        assert result["status"] == "success"
        assert result["route"]["name"] == "Weekday commute"
        assert result["route"]["origin_label"] == "My home"
        provider.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_missing_duplicate_invalid_and_unverified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = await update_common_route("missing", CommonRouteChanges(), _context())
        assert missing["error_code"] == "route_not_found"

        await _save()
        await _save("Gym")
        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(return_value=_routes_response()),
        ):
            duplicate = await update_common_route(
                "Gym",
                CommonRouteChanges(new_name="COMMUTE"),
                _context(),
            )
            invalid = await update_common_route(
                "Gym",
                CommonRouteChanges(origin_label=""),
                _context(),
            )
        assert duplicate["error_code"] == "invalid_saved_route"
        assert invalid["error_code"] == "invalid_saved_route"

        monkeypatch.delenv("GOOGLE_MAPS_ROUTES_API_KEY")
        no_key = await update_common_route(
            "Gym", CommonRouteChanges(travel_mode="WALK"), _context()
        )
        assert no_key["error_code"] == "not_configured"

        monkeypatch.setenv("GOOGLE_MAPS_ROUTES_API_KEY", "key")
        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(side_effect=RoutesAPIError("no_route", "none")),
        ):
            unavailable = await update_common_route(
                "Gym", CommonRouteChanges(avoid_tolls=True), _context()
            )
        assert unavailable["error_code"] == "no_route"

        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(return_value=_routes_response(partial_origin=True)),
        ):
            ambiguous = await update_common_route(
                "Gym",
                CommonRouteChanges(origin="Ambiguous origin"),
                _context(),
            )
        assert ambiguous["error_code"] == "ambiguous_location"

    @pytest.mark.asyncio
    async def test_update_handles_route_deleted_during_provider_lookup(self) -> None:
        await _save()

        async def delete_during_lookup(*_args, **_kwargs):
            from blacki.routes.storage import get_saved_route_storage

            route = await get_saved_route_storage().get_route("user-1", "Commute")
            assert route is not None and route.id is not None
            await get_saved_route_storage().delete_route("user-1", route.id)
            return _routes_response()

        with patch(
            "blacki.routes.tools.compute_routes",
            new=AsyncMock(side_effect=delete_during_lookup),
        ):
            result = await update_common_route(
                "Commute",
                CommonRouteChanges(destination="New office, Pune"),
                _context(),
            )
        assert result["error_code"] == "route_not_found"

    @pytest.mark.asyncio
    async def test_update_rejects_empty_change_set_without_provider_call(self) -> None:
        await _save()
        provider = AsyncMock(return_value=_routes_response())
        with patch("blacki.routes.tools.compute_routes", new=provider):
            result = await update_common_route(
                "Commute", CommonRouteChanges(), _context()
            )

        assert result["error_code"] == "invalid_saved_route"
        provider.assert_not_awaited()


class TestScheduling:
    @pytest.mark.asyncio
    async def test_schedule_and_delete_route_cancels_update(self) -> None:
        await _save()
        scheduled = await schedule_common_route_update(
            "Commute", "0 8 * * 1-5", _context()
        )
        assert scheduled["status"] == "success"
        assert scheduled["route_name"] == "Commute"

        deleted = await delete_common_route("Commute", _context())
        assert deleted["status"] == "success"
        assert deleted["cancelled_updates"] == 1
        assert (await list_common_routes(_context()))["count"] == 0

    @pytest.mark.asyncio
    async def test_schedule_validates_route_frequency_duplicate_and_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (await schedule_common_route_update("missing", "0 8 * * *", _context()))[
            "error_code"
        ] == "route_not_found"
        await _save()

        invalid = await schedule_common_route_update("Commute", "not cron", _context())
        frequent = await schedule_common_route_update(
            "Commute", "* * * * *", _context()
        )
        assert invalid["error_code"] == "invalid_schedule"
        assert frequent["error_code"] == "schedule_too_frequent"

        first = await schedule_common_route_update("Commute", "0 8 * * *", _context())
        duplicate = await schedule_common_route_update(
            "Commute", "0 8 * * *", _context()
        )
        assert first["status"] == "success"
        assert duplicate["error_code"] == "duplicate_schedule"

        monkeypatch.setenv("GOOGLE_MAPS_ROUTE_UPDATE_LIMIT", "1")
        limited = await schedule_common_route_update("Commute", "0 9 * * *", _context())
        assert limited["error_code"] == "schedule_limit_reached"

    @pytest.mark.asyncio
    async def test_schedule_failure_is_safe_and_keeps_saved_route(self) -> None:
        await _save()
        scheduler = create_autospec(ReminderScheduler, instance=True, spec_set=True)
        scheduler.get_user_reminders = AsyncMock(return_value=[])
        scheduler.schedule_reminder = AsyncMock(side_effect=RuntimeError("db failed"))

        with patch("blacki.routes.common_tools.get_scheduler", return_value=scheduler):
            result = await schedule_common_route_update(
                "Commute", "0 8 * * *", _context()
            )

        assert result["error_code"] == "schedule_failed"
        assert (await list_common_routes(_context()))["count"] == 1

    @pytest.mark.asyncio
    async def test_delete_missing_route_is_safe(self) -> None:
        result = await delete_common_route("missing", _context())
        assert result["error_code"] == "route_not_found"

    @pytest.mark.asyncio
    async def test_delete_ignores_unrelated_or_uncancellable_reminders(self) -> None:
        await _save()
        scheduler = create_autospec(ReminderScheduler, instance=True, spec_set=True)
        scheduler.get_user_reminders = AsyncMock(
            return_value=[
                Reminder(
                    id=1,
                    user_id="user-1",
                    message="ordinary",
                    trigger_time="2026-08-01T00:00:00+00:00",
                    created_at="2026-07-24T00:00:00+00:00",
                ),
                Reminder(
                    id=2,
                    user_id="user-1",
                    message=(
                        '{"kind":"blacki.route_traffic_update",'
                        '"version":1,"route_id":999}'
                    ),
                    trigger_time="2026-08-01T00:00:00+00:00",
                    created_at="2026-07-24T00:00:00+00:00",
                ),
                Reminder(
                    user_id="user-1",
                    message=(
                        '{"kind":"blacki.route_traffic_update",'
                        '"version":1,"route_id":1}'
                    ),
                    trigger_time="2026-08-01T00:00:00+00:00",
                    created_at="2026-07-24T00:00:00+00:00",
                ),
                Reminder(
                    id=3,
                    user_id="user-1",
                    message=(
                        '{"kind":"blacki.route_traffic_update",'
                        '"version":1,"route_id":1}'
                    ),
                    trigger_time="2026-08-01T00:00:00+00:00",
                    created_at="2026-07-24T00:00:00+00:00",
                ),
            ]
        )
        scheduler.delete_reminder = AsyncMock(return_value=False)

        with patch("blacki.routes.common_tools.get_scheduler", return_value=scheduler):
            result = await delete_common_route("Commute", _context())

        assert result["status"] == "success"
        assert result["cancelled_updates"] == 0
        scheduler.delete_reminder.assert_awaited_once_with(3, "user-1")

    @pytest.mark.asyncio
    async def test_delete_handles_route_removed_concurrently(self) -> None:
        await _save()
        scheduler = create_autospec(ReminderScheduler, instance=True, spec_set=True)

        async def remove_route(_user_id: str):
            from blacki.routes.storage import get_saved_route_storage

            route = await get_saved_route_storage().get_route("user-1", "Commute")
            assert route is not None and route.id is not None
            await get_saved_route_storage().delete_route("user-1", route.id)
            return []

        scheduler.get_user_reminders = AsyncMock(side_effect=remove_route)
        with patch("blacki.routes.common_tools.get_scheduler", return_value=scheduler):
            result = await delete_common_route("Commute", _context())

        assert result["error_code"] == "route_not_found"
