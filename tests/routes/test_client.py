"""Tests for the Google Maps Routes HTTP client."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, create_autospec, patch

import httpx
import pytest

from blacki.routes import client as routes_client
from blacki.routes.client import (
    COMPUTE_ROUTES_URL,
    MAX_ATTEMPTS,
    ROUTES_FIELD_MASK,
    RoutesAPIError,
    _get_shared_routes_client,
    close_shared_routes_client,
    compute_routes,
    reset_routes_client_cache,
)


@pytest.fixture(autouse=True)
async def reset_client() -> AsyncGenerator[None, None]:
    """Keep the process-wide client isolated between tests."""
    await reset_routes_client_cache()
    yield
    await reset_routes_client_cache()


def _strict_client() -> Any:
    return create_autospec(httpx.AsyncClient, instance=True, spec_set=True)


def _response(status_code: int, json: object | None = None) -> httpx.Response:
    request = httpx.Request("POST", COMPUTE_ROUTES_URL)
    if json is None:
        return httpx.Response(status_code, request=request)
    return httpx.Response(status_code, request=request, json=json)


class TestSharedRoutesClient:
    """Shared-client lifecycle behavior."""

    @pytest.mark.asyncio
    async def test_creates_and_reuses_client(self) -> None:
        strict_client = _strict_client()
        with patch(
            "blacki.routes.client.httpx.AsyncClient",
            autospec=True,
            return_value=strict_client,
        ) as client_class:
            first = await _get_shared_routes_client()
            second = await _get_shared_routes_client()

        assert first is strict_client
        assert second is strict_client
        client_class.assert_called_once_with(timeout=15.0)

    @pytest.mark.asyncio
    async def test_close_handles_none_and_clears_client(self) -> None:
        routes_client._routes_client = None

        await close_shared_routes_client()

        strict_client = _strict_client()
        routes_client._routes_client = strict_client

        await close_shared_routes_client()

        strict_client.aclose.assert_awaited_once()
        assert routes_client._routes_client is None

    @pytest.mark.asyncio
    async def test_close_logs_and_clears_after_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        strict_client = _strict_client()
        strict_client.aclose.side_effect = RuntimeError("close failed")
        routes_client._routes_client = strict_client

        await close_shared_routes_client()

        assert routes_client._routes_client is None
        assert "Error while closing shared Google Routes client" in caplog.text


class TestComputeRoutes:
    """HTTP status mapping, field masks, and bounded retry behavior."""

    @pytest.fixture
    def strict_client(self) -> Any:
        client = _strict_client()
        routes_client._routes_client = client
        return client

    @pytest.mark.asyncio
    async def test_success_uses_fixed_headers_and_payload(
        self, strict_client: Any
    ) -> None:
        payload = {"origin": {"address": "A"}, "destination": {"address": "B"}}
        strict_client.post.return_value = _response(200, {"routes": []})

        result = await compute_routes(payload, "secret-key")

        assert result == {"routes": []}
        strict_client.post.assert_awaited_once_with(
            COMPUTE_ROUTES_URL,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": "secret-key",
                "X-Goog-FieldMask": ROUTES_FIELD_MASK,
            },
            json=payload,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [401, 403])
    async def test_authentication_errors_do_not_retry(
        self, strict_client: Any, status_code: int
    ) -> None:
        strict_client.post.return_value = _response(status_code)

        with pytest.raises(RoutesAPIError) as caught:
            await compute_routes({}, "bad-key")

        assert caught.value.code == "authentication_failed"
        assert caught.value.retryable is False
        assert strict_client.post.await_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 404, 422])
    async def test_client_errors_are_normalized(
        self, strict_client: Any, status_code: int
    ) -> None:
        strict_client.post.return_value = _response(status_code)

        with pytest.raises(RoutesAPIError) as caught:
            await compute_routes({}, "key")

        assert caught.value.code == "invalid_request"
        assert strict_client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_unexpected_redirect_is_normalized(self, strict_client: Any) -> None:
        strict_client.post.return_value = _response(302)

        with pytest.raises(RoutesAPIError) as caught:
            await compute_routes({}, "key")

        assert caught.value.code == "unavailable"
        assert caught.value.retryable is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first_status", "expected_delay"),
        [(429, 0.25), (500, 0.25)],
    )
    async def test_retryable_status_recovers(
        self,
        strict_client: Any,
        first_status: int,
        expected_delay: float,
    ) -> None:
        strict_client.post.side_effect = [
            _response(first_status),
            _response(200, {"routes": [{"distanceMeters": 1}]}),
        ]
        sleep = AsyncMock()

        with patch("blacki.routes.client.asyncio.sleep", new=sleep):
            result = await compute_routes({}, "key")

        assert result["routes"] == [{"distanceMeters": 1}]
        sleep.assert_awaited_once_with(expected_delay)
        assert strict_client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_use_exponential_delays(self, strict_client: Any) -> None:
        strict_client.post.side_effect = [
            _response(500),
            _response(503),
            _response(200, {"routes": []}),
        ]
        sleep = AsyncMock()

        with patch("blacki.routes.client.asyncio.sleep", new=sleep):
            await compute_routes({}, "key")

        assert [call.args[0] for call in sleep.await_args_list] == [0.25, 0.5]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status_code", "expected_code"),
        [(429, "rate_limited"), (503, "unavailable")],
    )
    async def test_retryable_status_exhaustion(
        self,
        strict_client: Any,
        status_code: int,
        expected_code: str,
    ) -> None:
        strict_client.post.return_value = _response(status_code)

        with (
            patch("blacki.routes.client.asyncio.sleep", new=AsyncMock()),
            pytest.raises(RoutesAPIError) as caught,
        ):
            await compute_routes({}, "key")

        assert caught.value.code == expected_code
        assert strict_client.post.await_count == MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_network_failure_retries_then_recovers(
        self, strict_client: Any
    ) -> None:
        request = httpx.Request("POST", COMPUTE_ROUTES_URL)
        strict_client.post.side_effect = [
            httpx.ConnectError("offline", request=request),
            _response(200, {"routes": []}),
        ]
        sleep = AsyncMock()

        with patch("blacki.routes.client.asyncio.sleep", new=sleep):
            result = await compute_routes({}, "key")

        assert result == {"routes": []}
        sleep.assert_awaited_once_with(0.25)

    @pytest.mark.asyncio
    async def test_network_failure_exhaustion(self, strict_client: Any) -> None:
        request = httpx.Request("POST", COMPUTE_ROUTES_URL)
        strict_client.post.side_effect = httpx.ConnectError("offline", request=request)

        with (
            patch("blacki.routes.client.asyncio.sleep", new=AsyncMock()),
            pytest.raises(RoutesAPIError) as caught,
        ):
            await compute_routes({}, "key")

        assert caught.value.code == "unavailable"
        assert "could not be reached" in str(caught.value)
        assert strict_client.post.await_count == MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_invalid_json_is_rejected(self, strict_client: Any) -> None:
        request = httpx.Request("POST", COMPUTE_ROUTES_URL)
        strict_client.post.return_value = httpx.Response(
            200, request=request, content=b"not-json"
        )

        with pytest.raises(RoutesAPIError) as caught:
            await compute_routes({}, "key")

        assert caught.value.code == "invalid_response"

    @pytest.mark.asyncio
    async def test_non_object_json_is_rejected(self, strict_client: Any) -> None:
        strict_client.post.return_value = _response(200, [])

        with pytest.raises(RoutesAPIError) as caught:
            await compute_routes({}, "key")

        assert caught.value.code == "invalid_response"
