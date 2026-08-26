"""Tests for the Google Health connector and its Telegram-safe boundaries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch
from urllib.parse import parse_qs, urlsplit

import aiosqlite
import httpx
import pytest
from conftest import MockState, MockToolContext
from cryptography.fernet import Fernet

from blacki.container import AppContainer, reset_container_for_tests, set_container
from blacki.health.client import (
    GoogleHealthApiError,
    GoogleHealthAuthError,
    GoogleHealthClient,
    GoogleHealthIdentity,
    GoogleHealthOperation,
    GoogleTokenResponse,
    _filter_for_data_type,
    _json_object,
    _nutrition_parent,
    _parse_operation,
    _parse_token_response,
    _raise_provider_error,
    _retry_after_seconds,
)
from blacki.health.config import (
    GOOGLE_HEALTH_SCOPES,
    GoogleHealthConfig,
    GoogleHealthConfigurationError,
    TokenEncryptionError,
    google_health_configured_from_environment,
    health_user_id_for_telegram_user,
    telegram_chat_id_for_health_user,
)
from blacki.health.models import HealthDay, HealthSleep, HealthWorkout
from blacki.health.normalize import (
    _apply_component,
    _date_for_component,
    _DayBuilder,
    _duration_seconds,
    _interval_seconds,
    normalize_data_points,
)
from blacki.health.scheduler import GoogleHealthScheduler
from blacki.health.service import (
    GoogleHealthOAuthError,
    GoogleHealthService,
    _build_trends,
    _date_window,
    format_health_summary,
)
from blacki.health.storage import (
    HealthConnection,
    SqliteGoogleHealthStorage,
    _parse_timestamp,
)
from blacki.health.tools import get_health_summary


def _config() -> GoogleHealthConfig:
    return GoogleHealthConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/google-health/callback",
        token_encryption_key=Fernet.generate_key().decode(),
        sync_interval_hours=12,
        manual_refresh_cooldown_seconds=3600,
        oauth_state_ttl_seconds=600,
    )


@pytest.fixture
async def health_storage() -> AsyncGenerator[SqliteGoogleHealthStorage, None]:
    """Create an isolated initialized health database."""
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    storage = SqliteGoogleHealthStorage(conn, asyncio.Lock())
    await storage.initialize()
    yield storage
    await storage.close()
    await conn.close()


def test_health_config_and_cipher() -> None:
    """Validate optional settings, exact scopes, URL construction, and encryption."""
    assert GoogleHealthConfig.from_environment({}) is None
    key = Fernet.generate_key().decode()
    environ = {
        "GOOGLE_HEALTH_CLIENT_ID": " id ",
        "GOOGLE_HEALTH_CLIENT_SECRET": " secret ",
        "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": key,
        "GOOGLE_HEALTH_REDIRECT_URI": "http://localhost:8080/callback",
        "GOOGLE_HEALTH_SYNC_INTERVAL_HOURS": "6",
        "GOOGLE_HEALTH_MANUAL_REFRESH_COOLDOWN_SECONDS": "120",
        "GOOGLE_HEALTH_OAUTH_STATE_TTL_SECONDS": "90",
    }
    config = GoogleHealthConfig.from_environment(environ)
    assert config is not None
    assert config.client_id == "id"
    assert config.sync_interval_hours == 6
    assert config.manual_refresh_cooldown_seconds == 120
    assert config.oauth_state_ttl_seconds == 90
    assert GOOGLE_HEALTH_SCOPES == (
        "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
        "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
        "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
        "https://www.googleapis.com/auth/googlehealth.nutrition.readonly",
        "https://www.googleapis.com/auth/googlehealth.nutrition.writeonly",
    )

    query = parse_qs(urlsplit(config.authorization_url("state-value")).query)
    assert query["state"] == ["state-value"]
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    assert query["scope"] == [" ".join(GOOGLE_HEALTH_SCOPES)]

    cipher = config.cipher
    encrypted = cipher.encrypt("refresh-token")
    assert encrypted != "refresh-token"
    assert cipher.decrypt(encrypted) == "refresh-token"
    with pytest.raises(TokenEncryptionError):
        cipher.encrypt("")
    with pytest.raises(TokenEncryptionError):
        cipher.decrypt("not-a-fernet-token")

    secure_config = GoogleHealthConfig.from_environment(
        {
            "GOOGLE_HEALTH_CLIENT_ID": "id",
            "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
            "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": key,
            "GOOGLE_HEALTH_REDIRECT_URI": "https://example.test/callback",
        }
    )
    assert secure_config is not None


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"GOOGLE_HEALTH_CLIENT_ID": "id"}, "configuration is incomplete"),
        (
            {
                "GOOGLE_HEALTH_CLIENT_ID": "id",
                "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
                "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": "bad",
            },
            "invalid",
        ),
        (
            {
                "GOOGLE_HEALTH_CLIENT_ID": "id",
                "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
                "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
                "GOOGLE_HEALTH_REDIRECT_URI": "http://example.test/callback",
            },
            "HTTPS",
        ),
        (
            {
                "GOOGLE_HEALTH_CLIENT_ID": "id",
                "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
                "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
                "GOOGLE_HEALTH_SYNC_INTERVAL_HOURS": "0",
            },
            "positive integer",
        ),
        (
            {
                "GOOGLE_HEALTH_CLIENT_ID": "id",
                "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
                "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
                "GOOGLE_HEALTH_SYNC_INTERVAL_HOURS": "hours",
            },
            "positive integer",
        ),
    ],
)
def test_health_config_rejects_invalid_values(
    environ: dict[str, str], message: str
) -> None:
    """Reject incomplete, unsafe, or invalid connector configuration."""
    with pytest.raises(GoogleHealthConfigurationError, match=message):
        GoogleHealthConfig.from_environment(environ)


def test_health_configured_and_identity_helpers() -> None:
    """Keep configuration detection and private-chat identity scoping narrow."""
    assert google_health_configured_from_environment({}) is False
    key = Fernet.generate_key().decode()
    values = {
        "GOOGLE_HEALTH_CLIENT_ID": "id",
        "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
        "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": key,
    }
    assert google_health_configured_from_environment(values) is True
    assert health_user_id_for_telegram_user("telegram-chat-42-thread-9") == (
        "telegram-chat-42"
    )
    assert health_user_id_for_telegram_user("telegram-chat--42") == (
        "telegram-chat--42"
    )
    assert health_user_id_for_telegram_user("telegram-http-user") is None
    assert telegram_chat_id_for_health_user("telegram-chat-42") == 42
    assert telegram_chat_id_for_health_user("telegram-chat-42-thread-9") is None

    assert (
        google_health_configured_from_environment({"GOOGLE_HEALTH_CLIENT_ID": "id"})
        is False
    )
    assert (
        google_health_configured_from_environment(
            {
                "GOOGLE_HEALTH_CLIENT_ID": "id",
                "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
                "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": "bad",
            }
        )
        is False
    )


def _response(status: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("GET", "https://example.test"),
    )


@pytest.mark.asyncio
async def test_google_health_client_happy_path_and_pagination() -> None:
    """Exchange, refresh, identity, pagination, and revocation use safe requests."""
    config = _config()
    requests: list[httpx.Request] = []
    responses = [
        _response(
            200,
            {"access_token": "access-1", "refresh_token": "refresh-1", "scope": "a b"},
        ),
        _response(200, {"access_token": "access-2", "expires_in": 3600}),
        _response(200, {"healthUserId": "health-id", "legacyUserId": "fitbit-id"}),
        _response(200, {"dataPoints": [{"name": "one"}], "nextPageToken": "next"}),
        _response(200, {"dataPoints": [{"name": "two"}, "ignored"]}),
        _response(200, {}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses.pop(0)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GoogleHealthClient(config, http_client=http_client)
    token = await client.exchange_code("auth-code")
    refreshed = await client.refresh_access_token("refresh-1")
    identity = await client.get_identity("access-2")
    points = await client.list_data_points(
        "access-2",
        "steps",
        start_time="2026-08-10T00:00:00Z",
        end_time="2026-08-17T00:00:00Z",
    )
    await client.revoke_token("refresh-1")
    await client.close()

    assert token == GoogleTokenResponse("access-1", None, "refresh-1", ("a", "b"))
    assert refreshed.access_token == "access-2"  # noqa: S105
    assert identity == GoogleHealthIdentity("health-id", "fitbit-id")
    assert points == [{"name": "one"}, {"name": "two"}]
    assert len(requests) == 6
    assert requests[3].url.params["pageSize"] == "10000"
    assert requests[4].url.params["pageToken"] == "next"
    assert "Bearer access-2" in requests[3].headers["Authorization"]


@pytest.mark.asyncio
async def test_google_health_client_errors_are_safe() -> None:
    """Provider error bodies never become exception text or logs."""
    config = _config()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            return _response(
                400, {"error": "invalid_grant", "error_description": "secret"}
            )
        if request.url.path.endswith("/identity"):
            return _response(200, {})
        if request.url.path.endswith("/dataPoints"):
            return _response(401, {"error": "expired_token", "detail": "private"})
        return _response(400, {"error": "revoked"})

    client = GoogleHealthClient(
        config,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(GoogleHealthApiError) as token_error:
        await client.refresh_access_token("refresh")
    assert token_error.value.error_code == "invalid_grant"
    assert "secret" not in str(token_error.value)
    with pytest.raises(GoogleHealthApiError, match="identity response"):
        await client.get_identity("access")
    with pytest.raises(GoogleHealthAuthError) as api_error:
        await client.list_data_points(
            "access",
            "steps",
            start_time="2026-08-10T00:00:00Z",
            end_time="2026-08-17T00:00:00Z",
        )
    assert api_error.value.status_code == 401
    with pytest.raises(GoogleHealthApiError, match="revocation"):
        await client.revoke_token("refresh")
    await client.close()

    with pytest.raises(GoogleHealthApiError, match="non-JSON"):
        _json_object(httpx.Response(200, text="not-json"))
    with pytest.raises(GoogleHealthApiError, match="unexpected"):
        _json_object(_response(200, []))
    with pytest.raises(GoogleHealthApiError, match="refresh token"):
        _parse_token_response({"access_token": "access"}, require_refresh_token=True)
    with pytest.raises(GoogleHealthApiError, match="access token"):
        _parse_token_response({}, require_refresh_token=False)


@pytest.mark.asyncio
async def test_google_health_token_request_wraps_transport_errors() -> None:
    """A network failure reaching Google's token endpoint is a safe, typed error."""
    config = _config()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = GoogleHealthClient(
        config,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(GoogleHealthApiError) as error:
        await client.refresh_access_token("refresh")
    assert error.value.error_code == "transport_error"
    assert error.value.transport is True
    await client.close()


def test_structured_google_health_error_exposes_only_reason_code() -> None:
    """Extract Google RPC ErrorInfo reasons without exposing provider text."""
    with pytest.raises(GoogleHealthApiError) as error:
        _raise_provider_error(
            {
                "error": {
                    "code": 400,
                    "message": "sensitive provider details",
                    "status": "INVALID_ARGUMENT",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "reason": "ACCOUNT_NOT_LINKED",
                        }
                    ],
                }
            },
            400,
            token_endpoint=False,
        )

    assert error.value.error_code == "ACCOUNT_NOT_LINKED"
    assert "sensitive provider details" not in str(error.value)


@pytest.mark.parametrize(
    ("payload", "expected_error_code"),
    [
        ({"error": {"status": "INVALID_ARGUMENT"}}, "INVALID_ARGUMENT"),
        (
            {
                "error": {
                    "details": [None, {"reason": ""}],
                    "status": "INVALID_ARGUMENT",
                }
            },
            "INVALID_ARGUMENT",
        ),
        ({"error": {"details": [{"reason": ""}]}}, None),
        ({"error": []}, None),
    ],
)
def test_provider_error_parsing_handles_malformed_details(
    payload: dict[str, object], expected_error_code: str | None
) -> None:
    """Provider error parsing falls back without exposing response details."""
    with pytest.raises(GoogleHealthApiError) as error:
        _raise_provider_error(payload, 400, token_endpoint=False)

    assert error.value.error_code == expected_error_code


@pytest.mark.asyncio
async def test_google_health_client_handles_non_list_pages_and_lazy_client() -> None:
    """Handle malformed page collections and construct the client lazily."""
    config = _config()
    client = GoogleHealthClient(
        config,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: _response(200, {"dataPoints": {}}))
        ),
    )
    assert (
        await client.list_data_points(
            "access",
            "steps",
            start_time="2026-08-10T00:00:00Z",
            end_time="2026-08-17T00:00:00Z",
        )
        == []
    )
    await client.close()

    lazy_client = GoogleHealthClient(config)
    assert await lazy_client._ensure_client() is not None
    await lazy_client.close()


_NUTRITION_RESOURCE_NAME = (
    "users/health-user-1/dataTypes/nutrition-log/dataPoints/blacki-test1234"
)


@pytest.mark.asyncio
async def test_google_health_client_nutrition_methods_use_safe_requests() -> None:
    """Create, fetch, and delete a nutrition log through the expected endpoints."""
    config = _config()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/dataPoints"):
            return _response(200, {"done": True, "response": {}})
        if request.method == "GET":
            return _response(200, {"name": _NUTRITION_RESOURCE_NAME})
        if request.method == "POST" and request.url.path.endswith(
            "/dataPoints:batchDelete"
        ):
            return _response(200, {"done": True, "response": {}})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = GoogleHealthClient(
        config,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    created = await client.create_nutrition_log(
        "access", _NUTRITION_RESOURCE_NAME, {"nutritionLog": {"calories": 500}}
    )
    fetched = await client.get_data_point("access", _NUTRITION_RESOURCE_NAME)
    deleted = await client.delete_nutrition_log("access", _NUTRITION_RESOURCE_NAME)
    await client.close()

    assert created == GoogleHealthOperation(done=True, response={})
    assert fetched == {"name": _NUTRITION_RESOURCE_NAME}
    assert deleted == GoogleHealthOperation(done=True, response={})
    assert len(requests) == 3
    assert requests[0].url.path == (
        "/v4/users/health-user-1/dataTypes/nutrition-log/dataPoints"
    )
    assert requests[1].url.path == f"/v4/{_NUTRITION_RESOURCE_NAME}"
    assert requests[2].url.path == (
        "/v4/users/health-user-1/dataTypes/nutrition-log/dataPoints:batchDelete"
    )
    create_body = json.loads(requests[0].content)
    assert create_body["name"] == _NUTRITION_RESOURCE_NAME
    assert create_body["nutritionLog"] == {"calories": 500}
    delete_body = json.loads(requests[2].content)
    assert delete_body == {"names": [_NUTRITION_RESOURCE_NAME]}


@pytest.mark.asyncio
async def test_google_health_client_api_transport_error_is_safe() -> None:
    """A network failure against the Health API never leaks transport details."""
    config = _config()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret transport failure", request=request)

    client = GoogleHealthClient(
        config,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(GoogleHealthApiError) as error:
        await client.get_identity("access")
    assert error.value.transport is True
    assert error.value.error_code == "transport_error"
    assert "secret transport failure" not in str(error.value)
    await client.close()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"done": True, "error": {"status": "NOT_FOUND"}},
            GoogleHealthOperation(done=True, error_code="NOT_FOUND"),
        ),
        (
            {"done": True, "error": {"status": "x" * 90}},
            GoogleHealthOperation(done=True, error_code="x" * 80),
        ),
        (
            {"done": True, "error": {"code": 404}},
            GoogleHealthOperation(done=True, error_code="provider_error_404"),
        ),
        (
            {"done": True, "error": {}},
            GoogleHealthOperation(done=True, error_code="provider_error"),
        ),
        (
            {"done": True, "error": "quota exceeded"},
            GoogleHealthOperation(done=True, error_code="quota exceeded"),
        ),
        (
            {"done": True, "error": "x" * 90},
            GoogleHealthOperation(done=True, error_code="x" * 80),
        ),
        (
            {"done": True, "error": "café"},
            GoogleHealthOperation(done=True, error_code="provider_error"),
        ),
        (
            {"done": True, "error": "\x01bad"},
            GoogleHealthOperation(done=True, error_code="provider_error"),
        ),
        (
            {"done": True, "error": 123},
            GoogleHealthOperation(done=True, error_code="provider_error"),
        ),
        (
            {"done": True, "response": {"ok": True}, "name": "operations/1"},
            GoogleHealthOperation(
                done=True, name="operations/1", response={"ok": True}
            ),
        ),
        (
            {"done": False, "name": ""},
            GoogleHealthOperation(done=False, name=None),
        ),
    ],
)
def test_parse_operation_handles_all_error_shapes(
    payload: dict[str, object], expected: GoogleHealthOperation
) -> None:
    """Every provider-supplied error shape maps to a bounded, safe error code."""
    assert _parse_operation(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"done": "true"},
        {"done": None},
        {"done": True},
        {"done": True, "error": None, "response": None},
    ],
)
def test_parse_operation_rejects_incomplete_payloads(
    payload: dict[str, object],
) -> None:
    """Missing ``done`` or a completed operation without error/response fails closed."""
    with pytest.raises(GoogleHealthApiError, match="incomplete"):
        _parse_operation(payload)


def test_nutrition_parent_validates_resource_name() -> None:
    """Only a well-formed nutrition data point name yields its parent path."""
    assert _nutrition_parent(_NUTRITION_RESOURCE_NAME) == (
        "users/health-user-1/dataTypes/nutrition-log"
    )
    with pytest.raises(ValueError, match="not a valid nutrition data point name"):
        _nutrition_parent("users/health-user-1/dataTypes/nutrition-log/dataPoints/abc")
    with pytest.raises(ValueError, match="not a valid nutrition data point name"):
        _nutrition_parent("users/health-user-1/dataTypes/steps/dataPoints/blacki-1234")


def test_retry_after_seconds_parses_bounded_values() -> None:
    """Numeric, HTTP-date, missing, garbage, and non-finite values are all safe."""
    assert _retry_after_seconds(_response(200, {})) is None

    numeric = httpx.Response(
        200,
        headers={"Retry-After": "120"},
        request=httpx.Request("GET", "https://example.test"),
    )
    assert _retry_after_seconds(numeric) == 120.0

    future = datetime.now(UTC) + timedelta(seconds=90)
    date_response = httpx.Response(
        200,
        headers={"Retry-After": format_datetime(future, usegmt=True)},
        request=httpx.Request("GET", "https://example.test"),
    )
    parsed = _retry_after_seconds(date_response)
    assert parsed is not None
    assert 80.0 <= parsed <= 100.0

    garbage = httpx.Response(
        200,
        headers={"Retry-After": "not-a-number-or-date"},
        request=httpx.Request("GET", "https://example.test"),
    )
    assert _retry_after_seconds(garbage) is None

    for non_finite in ("nan", "inf", "-inf"):
        response = httpx.Response(
            200,
            headers={"Retry-After": non_finite},
            request=httpx.Request("GET", "https://example.test"),
        )
        assert _retry_after_seconds(response) is None

    naive_date_response = httpx.Response(
        200,
        headers={"Retry-After": "Mon, 01 Jan 2077 00:00:00"},
        request=httpx.Request("GET", "https://example.test"),
    )
    naive_parsed = _retry_after_seconds(naive_date_response)
    assert naive_parsed is not None
    assert naive_parsed > 0.0


@pytest.mark.parametrize(
    ("data_type", "fragment"),
    [
        ("daily-resting-heart-rate", "date >="),
        ("daily-heart-rate-zones", "date >="),
        ("weight", "sample_time.physical_time"),
        ("body-fat", "sample_time.physical_time"),
        ("sleep", "interval.end_time"),
        ("exercise", "interval.civil_start_time"),
        ("steps", "interval.start_time"),
    ],
)
def test_google_health_filters(data_type: str, fragment: str) -> None:
    """Use the provider filter field appropriate to each data type."""
    result = _filter_for_data_type(
        data_type,
        "2026-08-10T00:00:00Z",
        "2026-08-17T00:00:00Z",
    )
    assert fragment in result


def test_health_models_omit_missing_fields() -> None:
    """Normalized serialization contains no null or provider-specific fields."""
    workout = HealthWorkout("Weights", 55, calories_kcal=300, active_zone_minutes=12)
    sleep = HealthSleep(
        431,
        start_time="2026-08-15T23:00:00Z",
        end_time="2026-08-16T06:11:00Z",
        stages=({"type": "deep", "minutes": 80},),
    )
    day = HealthDay(
        "2026-08-16",
        steps=8420,
        workouts=(workout,),
        sleep=(sleep,),
        heart_rate_zones={"fatBurn": 20},
        heart_rate_zone_thresholds=({"type": "fatBurn", "min_bpm": 100},),
    ).to_dict()
    empty = HealthDay("2026-08-17").to_dict()
    assert day["steps"] == 8420
    assert day["workouts"] == [workout.to_dict()]
    assert day["sleep"] == [sleep.to_dict()]
    assert "weight_kg" not in day
    assert empty == {"date": "2026-08-17", "source": "google_health"}
    assert HealthWorkout("Run", 10).to_dict() == {"type": "Run", "minutes": 10}
    assert HealthSleep(10).to_dict() == {"minutes": 10}


def test_normalize_google_health_records() -> None:
    """Aggregate supported types, deduplicate records, and omit malformed data."""
    day = "2026-08-16"
    interval = {
        "startTime": f"{day}T08:00:00Z",
        "endTime": f"{day}T09:00:00Z",
        "civilStartTime": f"{day}T08:00:00+00:00",
    }
    exercise = {
        "exercise": {
            "exerciseType": "weightTraining",
            "displayName": "Weights",
            "interval": interval,
            "activeDuration": "3600s",
            "metricsSummary": {"caloriesKcal": 300, "activeZoneMinutes": 12},
        },
        "name": "workout-1",
    }
    sleep = {
        "sleep": {
            "interval": {
                "startTime": f"{day}T23:00:00Z",
                "endTime": "2026-08-17T06:11:00Z",
                "civilEndTime": f"{day}T06:11:00+00:00",
            },
            "summary": {
                "minutesAsleep": 431,
                "stagesSummary": [
                    {"type": "deep", "minutes": 80},
                    {"type": "light", "minutes": 300},
                    {"type": "bad", "minutes": "x"},
                ],
            },
        },
        "name": "sleep-1",
    }
    points: Mapping[str, Sequence[Mapping[str, Any]]] = {
        "steps": [{"steps": {"count": 8000, "interval": interval}}],
        "distance": [{"distance": {"millimeters": 4200000, "interval": interval}}],
        "active-energy-burned": [
            {"activeEnergyBurned": {"kcal": 500, "interval": interval}}
        ],
        "active-minutes": [
            {
                "activeMinutes": {
                    "activeMinutesByActivityLevel": [
                        {"activeMinutes": 20},
                        {"activeMinutes": -3},
                        {"activeMinutes": "bad"},
                    ],
                    "interval": interval,
                }
            }
        ],
        "active-zone-minutes": [
            {"activeZoneMinutes": {"activeZoneMinutes": 22, "interval": interval}}
        ],
        "exercise": [exercise, exercise],
        "sleep": [sleep, sleep],
        "daily-resting-heart-rate": [
            {
                "dailyRestingHeartRate": {
                    "date": {"year": 2026, "month": 8, "day": 16},
                    "beatsPerMinute": 58,
                }
            }
        ],
        "daily-heart-rate-zones": [
            {
                "dailyHeartRateZones": {
                    "date": {"date": {"year": 2026, "month": 8, "day": 16}},
                    "heartRateZones": [
                        {
                            "heartRateZoneType": "fatBurn",
                            "minBeatsPerMinute": 100,
                            "maxBeatsPerMinute": 130,
                        }
                    ],
                }
            }
        ],
        "time-in-heart-rate-zone": [
            {
                "timeInHeartRateZone": {
                    "heartRateZoneType": "fatBurn",
                    "interval": interval,
                }
            }
        ],
        "weight": [
            {
                "weight": {
                    "weightGrams": 80000,
                    "sampleTime": {"physicalTime": f"{day}T07:00:00Z"},
                },
                "name": "2026-08-16T07:00:00Z",
            },
            {
                "weight": {
                    "weightGrams": 81000,
                    "sampleTime": {"physicalTime": f"{day}T08:30:00Z"},
                },
                "name": "2026-08-16T07:30:00Z",
            },
        ],
        "body-fat": [
            {
                "bodyFat": {
                    "percentage": 18.5,
                    "sampleTime": {"physicalTime": f"{day}T07:00:00Z"},
                }
            }
        ],
        "unknown": [{"unknown": {"interval": interval}}],
    }

    result = normalize_data_points(points)
    assert len(result) == 1
    normalized = result[0].to_dict()
    assert normalized["steps"] == 8000
    assert normalized["distance_meters"] == 4200.0
    assert normalized["active_calories_kcal"] == 500.0
    assert normalized["active_minutes"] == 20
    assert normalized["active_zone_minutes"] == 22
    assert normalized["resting_heart_rate_bpm"] == 58
    assert normalized["heart_rate_zones"] == {"fatBurn": 60}
    assert normalized["weight_kg"] == 81.0
    assert normalized["body_fat_percent"] == 18.5
    assert len(normalized["workouts"]) == 1
    assert normalized["sleep"][0]["minutes"] == 431
    assert normalized["heart_rate_zone_thresholds"] == [
        {"type": "fatBurn", "min_bpm": 100, "max_bpm": 130}
    ]

    malformed = normalize_data_points(
        {
            "steps": [
                {"steps": {"count": 1, "interval": {"startTime": "bad"}}},
                {"steps": {"count": 1}},
            ],
            "exercise": [
                {"exercise": {"interval": interval, "activeDuration": "bad"}},
                {"exercise": {"interval": {}}},
            ],
            "sleep": [{"sleep": {"interval": interval}}],
        }
    )
    assert len(malformed) == 1
    assert malformed[0].sleep[0].minutes == 60


def test_normalize_google_health_malformed_components_fail_closed() -> None:
    """Ignore malformed components while retaining usable dated records."""
    day = "2026-08-16"
    interval = {
        "startTime": f"{day}T08:00:00Z",
        "endTime": f"{day}T09:00:00Z",
    }
    result = normalize_data_points(
        {
            "steps": [
                {"steps": []},
                {
                    "steps": {
                        "count": "bad",
                        "interval": {"startTime": f"{day}T00:00:00Z"},
                    }
                },
                {
                    "steps": {
                        "count": 1,
                        "sampleTime": {"civilTime": f"{day}T00:00:00Z"},
                    }
                },
                {
                    "steps": {
                        "count": 1,
                        "sampleTime": {"physicalTime": "bad"},
                    }
                },
            ],
            "distance": [{"distance": {"millimeters": "bad", "interval": interval}}],
            "active-energy-burned": [
                {
                    "activeEnergyBurned": {
                        "kcal": "bad",
                        "interval": interval,
                    }
                }
            ],
            "active-minutes": [
                {
                    "activeMinutes": {
                        "activeMinutesByActivityLevel": "bad",
                        "interval": interval,
                    }
                }
            ],
            "active-zone-minutes": [
                {
                    "activeZoneMinutes": {
                        "activeZoneMinutes": "bad",
                        "interval": interval,
                    }
                }
            ],
            "time-in-heart-rate-zone": [
                {
                    "timeInHeartRateZone": {
                        "heartRateZoneType": 4,
                        "interval": interval,
                    }
                }
            ],
            "daily-resting-heart-rate": [
                {
                    "dailyRestingHeartRate": {
                        "date": {"year": 2026, "month": 99, "day": 16},
                        "interval": {"startTime": f"{day}T00:00:00Z"},
                        "beatsPerMinute": 57,
                    }
                },
                {
                    "dailyRestingHeartRate": {
                        "date": {"date": {"year": 2026, "month": 99, "day": 16}},
                        "interval": {"startTime": f"{day}T00:00:00Z"},
                        "beatsPerMinute": 56,
                    }
                },
                {
                    "dailyRestingHeartRate": {
                        "date": {"year": 2026, "month": 8, "day": 16},
                        "beatsPerMinute": "bad",
                    }
                },
            ],
            "daily-heart-rate-zones": [
                {
                    "dailyHeartRateZones": {
                        "date": {"year": 2026, "month": 8, "day": 16},
                        "heartRateZones": "bad",
                    }
                },
                {
                    "dailyHeartRateZones": {
                        "date": {"year": 2026, "month": 8, "day": 16},
                        "heartRateZones": ["bad", {}],
                    }
                },
            ],
            "exercise": [
                {
                    "exercise": {
                        "interval": [],
                        "sampleTime": {"physicalTime": f"{day}T10:00:00Z"},
                    }
                },
                {
                    "exercise": {
                        "interval": {"startTime": f"{day}T10:00:00Z"},
                        "activeDuration": "bad",
                    }
                },
            ],
            "sleep": [
                {
                    "sleep": {
                        "interval": [],
                        "sampleTime": {"physicalTime": f"{day}T11:00:00Z"},
                    }
                },
                {"sleep": {"interval": {"startTime": f"{day}T23:00:00Z"}}},
                {
                    "sleep": {
                        "interval": interval,
                        "summary": {
                            "stagesSummary": [
                                "bad",
                                {"type": "deep", "minutes": "bad"},
                            ]
                        },
                    }
                },
            ],
            "weight": [
                {
                    "weight": {
                        "weightGrams": "bad",
                        "sampleTime": {"physicalTime": f"{day}T04:00:00Z"},
                    }
                },
                {
                    "weight": {
                        "weightGrams": 80000,
                        "sampleTime": {"physicalTime": f"{day}T08:00:00Z"},
                    },
                    "name": f"{day}T08:00:00Z",
                },
                {
                    "weight": {
                        "weightGrams": 79000,
                        "sampleTime": {"physicalTime": f"{day}T07:00:00Z"},
                    },
                    "name": f"{day}T07:00:00Z",
                },
                {
                    "weight": {
                        "weightGrams": 78000,
                        "sampleTime": "bad",
                        "interval": interval,
                    },
                    "name": f"{day}T06:00:00Z",
                },
                {
                    "weight": {
                        "weightGrams": 77000,
                        "sampleTime": {"civilTime": f"{day}T05:00:00Z"},
                    },
                    "name": f"{day}T05:00:00Z",
                },
            ],
            "body-fat": [
                {
                    "bodyFat": {
                        "percentage": "bad",
                        "sampleTime": {"physicalTime": f"{day}T04:00:00Z"},
                    }
                }
            ],
        }
    )
    assert result and result[0].date == day
    assert result[0].steps == 1

    assert (
        _date_for_component(
            {"sampleTime": {"civilTime": "bad", "physicalTime": "bad"}}, "steps"
        )
        is None
    )
    assert (
        _date_for_component({"date": {"date": {}}}, "daily-resting-heart-rate") is None
    )
    assert (
        _date_for_component({"date": {"date": "bad"}}, "daily-resting-heart-rate")
        is None
    )
    _apply_component(_DayBuilder(), {}, "unknown", {})
    assert _interval_seconds(None) is None
    assert _interval_seconds({"startTime": "now"}) is None
    assert _interval_seconds({"startTime": "bad", "endTime": "bad"}) is None
    assert _duration_seconds(1) is None


@pytest.mark.asyncio
async def test_health_storage_lifecycle(
    health_storage: SqliteGoogleHealthStorage,
) -> None:
    """Persist single-use state, credentials, normalized records, and deletion."""
    now = datetime(2026, 8, 16, tzinfo=UTC)
    await health_storage.store_oauth_state(
        "valid-state", "telegram-chat-42", expires_at=now + timedelta(minutes=5)
    )
    assert (
        await health_storage.consume_oauth_state("valid-state", now=now)
        == "telegram-chat-42"
    )
    assert await health_storage.consume_oauth_state("valid-state", now=now) is None
    await health_storage.store_oauth_state(
        "expired-state", "telegram-chat-42", expires_at=now - timedelta(seconds=1)
    )
    assert await health_storage.consume_oauth_state("expired-state", now=now) is None

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token="encrypted",
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=(GOOGLE_HEALTH_SCOPES[1], GOOGLE_HEALTH_SCOPES[0]),
    )
    connection = await health_storage.get_connection("telegram-chat-42")
    assert connection is not None
    assert connection.scopes == tuple(sorted(GOOGLE_HEALTH_SCOPES[:2]))
    assert await health_storage.list_active_connections() == [connection]

    allowed, next_allowed = await health_storage.claim_manual_refresh(
        "telegram-chat-42", cooldown_seconds=3600, now=now
    )
    assert allowed is True and next_allowed is None
    blocked, next_allowed = await health_storage.claim_manual_refresh(
        "telegram-chat-42", cooldown_seconds=3600, now=now + timedelta(minutes=1)
    )
    assert blocked is False and next_allowed is not None
    allowed_again, _ = await health_storage.claim_manual_refresh(
        "telegram-chat-42", cooldown_seconds=3600, now=now + timedelta(hours=2)
    )
    assert allowed_again is True
    assert await health_storage.claim_manual_refresh(
        "unknown", cooldown_seconds=10, now=now
    ) == (False, None)

    await health_storage.upsert_daily_summaries(
        "telegram-chat-42",
        [
            {"date": "2026-08-16", "steps": 100},
            {"date": "", "steps": 999},
            {"steps": 999},
        ],
    )
    await health_storage.upsert_daily_summaries(
        "telegram-chat-42", [{"date": "2026-08-16", "steps": 200}]
    )
    rows = await health_storage.get_daily_summaries(
        "telegram-chat-42", start_date="2026-08-15", end_date="2026-08-17"
    )
    assert rows == [{"date": "2026-08-16", "steps": 200}]
    await health_storage.conn.execute(
        "INSERT INTO google_health_daily_summaries "
        "(telegram_user_id, summary_date, summary_json, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("telegram-chat-42", "2026-08-15", "not-json", "now"),
    )
    assert (
        len(
            await health_storage.get_daily_summaries(
                "telegram-chat-42", start_date="2026-08-15", end_date="2026-08-17"
            )
        )
        == 1
    )

    await health_storage.mark_synced("telegram-chat-42")
    await health_storage.mark_reauthorization_required("telegram-chat-42", "bad\nerror")
    connection = await health_storage.get_connection("telegram-chat-42")
    assert connection is not None
    assert connection.status == "reauthorization_required"
    assert connection.encrypted_refresh_token is None
    assert connection.last_sync_error == "authorization_required"
    assert await health_storage.list_active_connections() == []
    await health_storage.conn.execute(
        "UPDATE google_health_connections SET encrypted_refresh_token = ? "
        "WHERE telegram_user_id = ?",
        ("encrypted", "telegram-chat-42"),
    )
    assert await health_storage.claim_manual_refresh(
        "telegram-chat-42", cooldown_seconds=10, now=now
    ) == (False, None)
    await health_storage.upsert_daily_summaries(
        "telegram-chat-42", [{"steps": 1}, {"date": ""}]
    )
    await health_storage.conn.execute(
        "UPDATE google_health_connections SET scopes_json = ? "
        "WHERE telegram_user_id = ?",
        ("not-json", "telegram-chat-42"),
    )
    corrupted = await health_storage.get_connection("telegram-chat-42")
    assert corrupted is not None and corrupted.scopes == ()
    await health_storage.conn.execute(
        "INSERT INTO google_health_daily_summaries "
        "(telegram_user_id, summary_date, summary_json, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("telegram-chat-42", "2026-08-14", "[]", "now"),
    )
    assert (
        await health_storage.get_daily_summaries(
            "telegram-chat-42", start_date="2026-08-14", end_date="2026-08-15"
        )
        == []
    )
    assert await health_storage.delete_connection("telegram-chat-42") is True
    assert await health_storage.delete_connection("telegram-chat-42") is False
    assert _parse_timestamp("2026-08-16T00:00:00").tzinfo == UTC


@pytest.mark.asyncio
async def test_delete_connection_rolls_back_on_failure(
    health_storage: SqliteGoogleHealthStorage,
) -> None:
    """A mid-transaction failure must not leave a partially deleted connection."""
    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-99",
        encrypted_refresh_token=_config().cipher.encrypt("refresh"),
        health_user_id="health-id-99",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    original_cancel = health_storage.nutrition.cancel

    async def _boom(user_id: str) -> None:
        raise RuntimeError("simulated failure")

    health_storage.nutrition.cancel = _boom  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="simulated failure"):
            await health_storage.delete_connection("telegram-chat-99")
    finally:
        health_storage.nutrition.cancel = original_cancel  # type: ignore[method-assign]

    connection = await health_storage.get_connection("telegram-chat-99")
    assert connection is not None


@pytest.mark.asyncio
async def test_app_container_initializes_and_closes_google_health_storage() -> None:
    """The shared container owns the health schema and closes it with the app."""
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    container = AppContainer(conn=conn)
    await container.initialize_all_storages()
    assert container.google_health_storage is container.google_health_storage
    await container.close()


@pytest.mark.asyncio
async def test_health_service_oauth_and_summary(
    health_storage: SqliteGoogleHealthStorage,
) -> None:
    """Bind OAuth state to a private chat and preserve an existing refresh token."""
    config = _config()
    client = create_autospec(GoogleHealthClient, instance=True, spec_set=True)
    client.exchange_code = AsyncMock(
        return_value=GoogleTokenResponse("access", 3600, None, ())
    )
    client.get_identity = AsyncMock(
        return_value=GoogleHealthIdentity("health-id", "fitbit-id")
    )
    service = GoogleHealthService(config, health_storage, client=client)
    encrypted = config.cipher.encrypt("old-refresh")
    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=encrypted,
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    url = await service.begin_authorization("telegram-chat-42-thread-5")
    state = parse_qs(urlsplit(url).query)["state"][0]
    completion = await service.complete_authorization(
        state=state, code="code", error=None
    )
    assert completion.connected is True
    assert completion.telegram_user_id == "telegram-chat-42"
    client.exchange_code.assert_awaited_once_with("code")
    stored = await health_storage.get_connection("telegram-chat-42")
    assert stored is not None
    assert config.cipher.decrypt(stored.encrypted_refresh_token or "") == "old-refresh"

    cancel_url = await service.begin_authorization("telegram-chat-42")
    cancel_state = parse_qs(urlsplit(cancel_url).query)["state"][0]
    cancelled = await service.complete_authorization(
        state=cancel_state, code=None, error="access_denied"
    )
    assert cancelled.connected is False
    with pytest.raises(GoogleHealthOAuthError, match="invalid or expired"):
        await service.complete_authorization(
            state=cancel_state, code="code", error=None
        )
    with pytest.raises(GoogleHealthOAuthError, match="state is missing"):
        await service.complete_authorization(state="", code="code", error=None)

    status = await service.connection_status("telegram-chat-42-thread-5")
    assert status["status"] == "connected"
    assert "health-id" not in status
    assert await service.summary_for_tool("telegram-chat-42", days=15) == {
        "status": "error",
        "error": "days must be between 1 and 14",
    }
    assert (await service.summary_for_tool("telegram-chat-42", days=7))["status"] == (
        "success"
    )
    assert await service.connection_status("telegram-chat-99") == {
        "status": "not_connected"
    }

    missing_code_url = await service.begin_authorization("telegram-chat-45")
    missing_code_state = parse_qs(urlsplit(missing_code_url).query)["state"][0]
    with pytest.raises(GoogleHealthOAuthError, match="code is missing"):
        await service.complete_authorization(
            state=missing_code_state, code=None, error=None
        )

    no_refresh_url = await service.begin_authorization("telegram-chat-43")
    no_refresh_state = parse_qs(urlsplit(no_refresh_url).query)["state"][0]
    client.exchange_code.return_value = GoogleTokenResponse("access", 3600, None, ())
    with pytest.raises(GoogleHealthOAuthError, match="refresh token"):
        await service.complete_authorization(
            state=no_refresh_state, code="code", error=None
        )

    new_refresh_url = await service.begin_authorization("telegram-chat-44")
    new_refresh_state = parse_qs(urlsplit(new_refresh_url).query)["state"][0]
    client.exchange_code.return_value = GoogleTokenResponse(
        "access", 3600, "new-refresh", ()
    )
    assert (
        await service.complete_authorization(
            state=new_refresh_state, code="code", error=None
        )
    ).connected is True

    assert service._decrypt_existing_token(None) is None
    empty_connection = HealthConnection(
        telegram_user_id="telegram-chat-46",
        encrypted_refresh_token=None,
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=(),
        status="connected",
        connected_at="now",
        last_synced_at=None,
        last_refresh_requested_at=None,
        last_sync_error=None,
    )
    assert service._decrypt_existing_token(empty_connection) is None
    invalid_connection = HealthConnection(
        telegram_user_id="telegram-chat-46",
        encrypted_refresh_token="invalid",
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=(),
        status="connected",
        connected_at="now",
        last_synced_at=None,
        last_refresh_requested_at=None,
        last_sync_error=None,
    )
    assert service._decrypt_existing_token(invalid_connection) is None
    await service.close()


@pytest.mark.asyncio
async def test_health_service_does_not_reuse_token_for_new_identity(
    health_storage: SqliteGoogleHealthStorage,
) -> None:
    """A different Google account cannot inherit the previous refresh token."""
    config = _config()
    client = create_autospec(GoogleHealthClient, instance=True, spec_set=True)
    client.exchange_code = AsyncMock(
        return_value=GoogleTokenResponse("new-access", 3600, None, ())
    )
    client.get_identity = AsyncMock(
        return_value=GoogleHealthIdentity("new-health-id", None)
    )
    service = GoogleHealthService(config, health_storage, client=client)
    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=config.cipher.encrypt("old-refresh"),
        health_user_id="old-health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    url = await service.begin_authorization("telegram-chat-42")
    state = parse_qs(urlsplit(url).query)["state"][0]

    with pytest.raises(GoogleHealthOAuthError, match="refresh token"):
        await service.complete_authorization(state=state, code="code", error=None)

    client.exchange_code.assert_awaited_once_with("code")
    client.get_identity.assert_awaited_once_with("new-access")
    stored = await health_storage.get_connection("telegram-chat-42")
    assert stored is not None
    assert stored.health_user_id == "old-health-id"
    assert config.cipher.decrypt(stored.encrypted_refresh_token or "") == "old-refresh"
    await service.close()


@pytest.mark.asyncio
async def test_health_storage_replaces_identity_and_window_atomically(
    health_storage: SqliteGoogleHealthStorage,
) -> None:
    """Account replacement clears identity-bound data and metadata safely."""
    now = datetime(2026, 8, 16, tzinfo=UTC)
    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token="old-token",
        health_user_id="old-health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    await health_storage.upsert_daily_summaries(
        "telegram-chat-42",
        [
            {"date": "2026-08-15", "steps": 100},
        ],
    )
    await health_storage.mark_synced("telegram-chat-42")
    assert await health_storage.claim_manual_refresh(
        "telegram-chat-42", cooldown_seconds=3600, now=now
    ) == (True, None)

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token="new-token",
        health_user_id="new-health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    replaced = await health_storage.get_connection("telegram-chat-42")
    assert replaced is not None
    assert replaced.health_user_id == "new-health-id"
    assert replaced.last_synced_at is None
    assert replaced.last_refresh_requested_at is None
    assert (
        await health_storage.get_daily_summaries(
            "telegram-chat-42", start_date="2026-08-15", end_date="2026-08-21"
        )
        == []
    )

    await health_storage.mark_synced("telegram-chat-42")
    assert await health_storage.claim_manual_refresh(
        "telegram-chat-42", cooldown_seconds=3600, now=now
    ) == (True, None)
    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token="same-token",
        health_user_id="new-health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    same_identity = await health_storage.get_connection("telegram-chat-42")
    assert same_identity is not None
    assert same_identity.last_synced_at is not None
    assert same_identity.last_refresh_requested_at is not None

    await health_storage.upsert_daily_summaries(
        "telegram-chat-42",
        [
            {"date": "2026-08-20", "steps": 200},
            {"date": "2026-08-21", "steps": 300},
        ],
    )
    await health_storage.replace_daily_summaries(
        "telegram-chat-42",
        [{"date": "", "steps": 999}],
        start_date="2026-08-20",
        end_date="2026-08-21",
    )
    assert await health_storage.get_daily_summaries(
        "telegram-chat-42", start_date="2026-08-20", end_date="2026-08-22"
    ) == [{"date": "2026-08-21", "steps": 300}]


@pytest.mark.asyncio
async def test_health_storage_replacement_rolls_back_on_failure(
    health_storage: SqliteGoogleHealthStorage,
) -> None:
    """Identity replacement does not partially delete data on write failure."""
    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token="old-token",
        health_user_id="old-health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    await health_storage.upsert_daily_summaries(
        "telegram-chat-42", [{"date": "2026-08-15", "steps": 100}]
    )
    await health_storage.conn.execute(
        """
        CREATE TRIGGER fail_google_health_connection_insert
        BEFORE INSERT ON google_health_connections
        BEGIN
            SELECT RAISE(ABORT, 'connection write failed');
        END
        """
    )
    try:
        with pytest.raises(aiosqlite.IntegrityError, match="connection write failed"):
            await health_storage.upsert_connection(
                telegram_user_id="telegram-chat-42",
                encrypted_refresh_token="new-token",
                health_user_id="new-health-id",
                legacy_fitbit_user_id=None,
                scopes=GOOGLE_HEALTH_SCOPES,
            )
    finally:
        await health_storage.conn.execute(
            "DROP TRIGGER fail_google_health_connection_insert"
        )

    stored = await health_storage.get_connection("telegram-chat-42")
    assert stored is not None and stored.health_user_id == "old-health-id"
    assert await health_storage.get_daily_summaries(
        "telegram-chat-42", start_date="2026-08-15", end_date="2026-08-16"
    ) == [{"date": "2026-08-15", "steps": 100}]


@pytest.mark.asyncio
async def test_health_storage_window_replacement_rolls_back_on_failure(
    health_storage: SqliteGoogleHealthStorage,
) -> None:
    """Window reconciliation restores deleted rows if replacement insertion fails."""
    await health_storage.upsert_daily_summaries(
        "telegram-chat-42", [{"date": "2026-08-15", "steps": 100}]
    )
    await health_storage.conn.execute(
        """
        CREATE TRIGGER fail_google_health_summary_insert
        BEFORE INSERT ON google_health_daily_summaries
        BEGIN
            SELECT RAISE(ABORT, 'summary write failed');
        END
        """
    )
    try:
        with pytest.raises(aiosqlite.IntegrityError, match="summary write failed"):
            await health_storage.replace_daily_summaries(
                "telegram-chat-42",
                [{"date": "2026-08-15", "steps": 200}],
                start_date="2026-08-15",
                end_date="2026-08-16",
            )
    finally:
        await health_storage.conn.execute(
            "DROP TRIGGER fail_google_health_summary_insert"
        )

    assert await health_storage.get_daily_summaries(
        "telegram-chat-42", start_date="2026-08-15", end_date="2026-08-16"
    ) == [{"date": "2026-08-15", "steps": 100}]


@pytest.mark.asyncio
async def test_health_service_sync_failure_modes(
    health_storage: SqliteGoogleHealthStorage, caplog: pytest.LogCaptureFixture
) -> None:
    """Handle absent, invalid, revoked, partially scoped, and failed connections."""
    config = _config()
    client = create_autospec(GoogleHealthClient, instance=True, spec_set=True)
    service = GoogleHealthService(config, health_storage, client=client)
    assert (await service.sync_user("telegram-chat-42")).status == "not_connected"
    assert await service.disconnect("telegram-chat-99") is False

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token="invalid",
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    assert (await service.sync_user("telegram-chat-42")).status == (
        "reauthorization_required"
    )
    assert (await service.refresh_user("telegram-chat-42")).status == (
        "reauthorization_required"
    )
    assert (await service.sync_user("telegram-chat-42")).status == (
        "reauthorization_required"
    )
    reauth_summary = await service.summary("telegram-chat-42")
    assert reauth_summary["status"] == "reauthorization_required"
    assert reauth_summary["last_sync_error"] == "stored_token_invalid"

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=config.cipher.encrypt("refresh"),
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    client.refresh_access_token = AsyncMock(
        side_effect=GoogleHealthAuthError(
            "reauth", status_code=401, error_code="expired"
        )
    )
    assert (await service.sync_user("telegram-chat-42")).status == (
        "reauthorization_required"
    )

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=config.cipher.encrypt("refresh"),
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    client.refresh_access_token = AsyncMock(
        side_effect=GoogleHealthApiError("bad", error_code="invalid_grant")
    )
    assert (await service.sync_user("telegram-chat-42")).status == (
        "reauthorization_required"
    )

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=config.cipher.encrypt("refresh"),
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    client.refresh_access_token = AsyncMock(
        side_effect=GoogleHealthApiError("temporary", error_code="server_error")
    )
    assert (await service.sync_user("telegram-chat-42")).status == "failed"

    client.refresh_access_token = AsyncMock(
        return_value=GoogleTokenResponse("access", 3600, None, GOOGLE_HEALTH_SCOPES)
    )

    async def unauthorized_points(
        *args: object, **kwargs: object
    ) -> list[dict[str, object]]:
        if str(args[1]) == "steps":
            raise GoogleHealthAuthError(
                "expired", status_code=401, error_code="expired_token"
            )
        return []

    client.list_data_points = AsyncMock(side_effect=unauthorized_points)
    assert (await service.sync_user("telegram-chat-42")).status == (
        "reauthorization_required"
    )

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=config.cipher.encrypt("refresh"),
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )

    async def failed_points(*args: object, **kwargs: object) -> list[dict[str, object]]:
        if str(args[1]) == "steps":
            raise GoogleHealthApiError(
                "provider failure", status_code=429, error_code="rateLimitExceeded"
            )
        return []

    client.list_data_points = AsyncMock(side_effect=failed_points)
    assert (await service.sync_user("telegram-chat-42")).status == "failed"
    assert (
        "Google Health data fetch failed: data_type=steps "
        "status_code=429 error_code=rateLimitExceeded"
    ) in caplog.text

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=config.cipher.encrypt("refresh"),
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    client.refresh_access_token = AsyncMock(
        return_value=GoogleTokenResponse("access", 3600, None, GOOGLE_HEALTH_SCOPES)
    )
    stale_date, _ = _date_window(7)
    provider_date = (
        (datetime.fromisoformat(stale_date) + timedelta(days=1)).date().isoformat()
    )

    async def list_points(*args: object, **kwargs: object) -> list[dict[str, object]]:
        data_type = str(args[1])
        if data_type == "exercise":
            raise GoogleHealthAuthError(
                "scope", status_code=403, error_code="forbidden"
            )
        if data_type == "steps":
            return [
                {
                    "steps": {
                        "count": 8420,
                        "interval": {
                            "startTime": f"{provider_date}T00:00:00Z",
                            "endTime": f"{provider_date}T23:59:59Z",
                        },
                    }
                }
            ]
        return []

    client.list_data_points = AsyncMock(side_effect=list_points)
    await health_storage.upsert_daily_summaries(
        "telegram-chat-42", [{"date": stale_date, "steps": 1}]
    )
    result = await service.sync_user("telegram-chat-42", days=7)
    assert result.status == "success"
    assert result.records_fetched == 1
    assert "exercise" in result.unavailable_data_types
    summary = await service.summary("telegram-chat-42", days=7)
    assert summary["status"] == "success"
    assert summary["days"][0]["steps"] == 8420
    assert all(day["date"] != stale_date for day in summary["days"])
    assert (await service.refresh_user("telegram-chat-42")).status == "success"
    assert (await service.refresh_user("telegram-chat-42")).status == "rate_limited"

    assert len(await service.sync_all()) == 1

    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token="invalid",
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    assert await service.disconnect("telegram-chat-42") is True
    await health_storage.upsert_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=config.cipher.encrypt("refresh"),
        health_user_id="health-id",
        legacy_fitbit_user_id=None,
        scopes=GOOGLE_HEALTH_SCOPES,
    )
    client.revoke_token = AsyncMock(
        side_effect=GoogleHealthApiError("revocation failed")
    )
    assert await service.disconnect("telegram-chat-42") is True

    assert await service.summary("telegram-chat-42") == {"status": "not_connected"}
    await service.close()


def test_format_health_summary_omits_missing_values() -> None:
    """Render deterministic wellness text without diagnosing or guessing."""
    assert "/connect_health" in format_health_summary({"status": "not_connected"})
    assert "authorization again" in format_health_summary(
        {"status": "reauthorization_required"}
    )
    assert "recently" in format_health_summary({"status": "rate_limited"})
    assert "couldn't" in format_health_summary({"status": "failed"})
    text = format_health_summary(
        {
            "status": "success",
            "days": [
                {
                    "date": "2026-08-16",
                    "steps": 8420,
                    "distance_meters": 4200,
                    "active_minutes": 42,
                    "active_zone_minutes": 20,
                    "resting_heart_rate_bpm": 58,
                    "sleep": [{"minutes": 431}],
                    "workouts": [{"type": "Weights"}],
                },
                {"date": "2026-08-17"},
                "bad",
            ],
            "trends": {
                "steps": {"average": 7000},
                "sleep_minutes": {"average": 400},
                "resting_heart_rate_bpm": {"average": 60},
            },
        }
    )
    assert "8,420 steps" in text
    assert "7-day data averages" in text
    assert "medical advice" in text
    assert "weight" not in text
    assert "no imported records" in format_health_summary(
        {"status": "success", "days": [], "trends": {}}
    )
    sparse = format_health_summary(
        {
            "status": "success",
            "days": [{"date": 123}],
            "trends": {"unknown": {}},
        }
    )
    assert "medical advice" in sparse
    assert "medical advice" in format_health_summary(
        {"status": "success", "days": [{"date": "2026-08-16"}]}
    )
    assert _build_trends([{"sleep": [{"minutes": 90}]}]) == {
        "sleep_minutes": {"average": 90.0, "latest": 90, "days_with_data": 1}
    }


@pytest.mark.asyncio
async def test_health_service_rejects_non_private_identity() -> None:
    """The service itself also enforces the private Telegram boundary."""
    service = GoogleHealthService(_config(), MagicMock())
    with pytest.raises(GoogleHealthOAuthError, match="private Telegram chat"):
        await service.summary("telegram-http-user")
    await service.close()


@pytest.mark.asyncio
async def test_google_health_scheduler_is_idempotent() -> None:
    """Start and stop one interval job and isolate a failing sync."""
    service = MagicMock()
    service.config.sync_interval_hours = 12
    service.sync_all = AsyncMock(return_value=[])
    scheduler = GoogleHealthScheduler(service)
    with (
        patch.object(scheduler.scheduler, "start") as start,
        patch.object(scheduler.scheduler, "shutdown") as shutdown,
    ):
        await scheduler.start()
        await scheduler.start()
        assert scheduler._running is True
        start.assert_called_once()
        await scheduler._sync_all()
        await scheduler.stop()
        await scheduler.stop()
    service.sync_all.assert_awaited_once()
    shutdown.assert_called_once_with(wait=True)

    service.sync_all = AsyncMock(side_effect=RuntimeError("provider down"))
    await scheduler._sync_all()


@pytest.mark.asyncio
async def test_health_tool_is_private_read_only_and_scoped() -> None:
    """Only private Telegram session state can read stored summaries."""
    reset_container_for_tests()
    storage = MagicMock()
    storage.get_connection = AsyncMock(return_value=None)
    container = MagicMock()
    container.google_health_storage = storage
    set_container(container)
    try:
        assert (
            await get_health_summary(cast(Any, MockToolContext(user_id="x")), days=0)
        )["status"] == ("error")
        assert (await get_health_summary(cast(Any, MockToolContext(user_id=None))))[
            "status"
        ] == ("error")
        group_context = MockToolContext(
            user_id="telegram-chat-42",
            state=MockState({"telegram_chat_type": "group"}),
        )
        assert (await get_health_summary(cast(Any, group_context)))["status"] == "error"
        private_context = MockToolContext(
            user_id="telegram-chat-42",
            state=MockState({"telegram_chat_type": "private"}),
        )
        result = await get_health_summary(cast(Any, private_context))
        assert result == {"status": "not_connected"}
        storage.get_connection.assert_awaited_once_with("telegram-chat-42")
    finally:
        reset_container_for_tests()


@pytest.mark.asyncio
async def test_health_tool_handles_uninitialized_container() -> None:
    """Fail closed when a tool invocation arrives before application startup."""
    reset_container_for_tests()
    context = MockToolContext(
        user_id="telegram-chat-42",
        state=MockState({"telegram_chat_type": "private"}),
    )
    assert (await get_health_summary(cast(Any, context)))["status"] == "error"
