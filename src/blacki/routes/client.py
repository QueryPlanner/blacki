"""Minimal async client for the Google Maps Routes API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

COMPUTE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
ROUTES_FIELD_MASK = ",".join(
    (
        "routes.distanceMeters",
        "routes.duration",
        "routes.staticDuration",
        "routes.description",
        "routes.routeLabels",
        "routes.warnings",
        "fallbackInfo.routingMode",
        "fallbackInfo.reason",
        "geocodingResults.origin.placeId",
        "geocodingResults.origin.partialMatch",
        "geocodingResults.destination.placeId",
        "geocodingResults.destination.partialMatch",
    )
)
MAX_ATTEMPTS = 3
RETRY_BASE_SECONDS = 0.25

_routes_client_lock = asyncio.Lock()
_routes_client: httpx.AsyncClient | None = None


class RoutesAPIError(RuntimeError):
    """Stable, credential-safe error raised by the Routes client."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


async def reset_routes_client_cache() -> None:
    """Close and clear the shared Routes client between tests."""
    await close_shared_routes_client()


async def close_shared_routes_client() -> None:
    """Close the process-wide Routes client during application shutdown."""
    global _routes_client
    async with _routes_client_lock:
        if _routes_client is not None:
            try:
                await _routes_client.aclose()
            except Exception:
                logger.exception("Error while closing shared Google Routes client")
        _routes_client = None


async def _get_shared_routes_client() -> httpx.AsyncClient:
    """Return a process-wide async client for Google Routes requests."""
    global _routes_client
    async with _routes_client_lock:
        if _routes_client is not None:
            return _routes_client
        _routes_client = httpx.AsyncClient(timeout=15.0)
        return _routes_client


def _response_error(status_code: int) -> RoutesAPIError | None:
    """Map an HTTP status to a stable tool-facing error."""
    if 200 <= status_code < 300:
        return None
    if status_code in (401, 403):
        return RoutesAPIError(
            "authentication_failed",
            "Google Maps Routes authentication failed. Check the configured API key.",
        )
    if status_code == 429:
        return RoutesAPIError(
            "rate_limited",
            "Google Maps Routes rate limit exceeded. Try again later.",
            retryable=True,
        )
    if status_code >= 500:
        return RoutesAPIError(
            "unavailable",
            "Google Maps Routes is temporarily unavailable.",
            retryable=True,
        )
    if 400 <= status_code < 500:
        return RoutesAPIError(
            "invalid_request",
            "Google Maps Routes rejected the route request.",
        )
    return RoutesAPIError(
        "unavailable",
        "Google Maps Routes returned an unexpected HTTP response.",
    )


async def compute_routes(
    payload: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    """Submit one Compute Routes request with bounded retries."""
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ROUTES_FIELD_MASK,
    }
    client = await _get_shared_routes_client()

    for attempt in range(MAX_ATTEMPTS):  # pragma: no branch - always returns or raises
        try:
            response = await client.post(
                COMPUTE_ROUTES_URL,
                headers=headers,
                json=payload,
            )
            error = _response_error(response.status_code)
        except httpx.RequestError:
            logger.warning(
                "Google Routes network request failed on attempt %d",
                attempt + 1,
            )
            error = RoutesAPIError(
                "unavailable",
                "Google Maps Routes could not be reached.",
                retryable=True,
            )

        if error is None:
            try:
                data = response.json()
            except ValueError as exc:
                raise RoutesAPIError(
                    "invalid_response",
                    "Google Maps Routes returned an invalid response.",
                ) from exc
            if not isinstance(data, dict):
                raise RoutesAPIError(
                    "invalid_response",
                    "Google Maps Routes returned an invalid response.",
                )
            return data

        if not error.retryable or attempt == MAX_ATTEMPTS - 1:
            raise error

        await asyncio.sleep(RETRY_BASE_SECONDS * (2**attempt))

    raise AssertionError(  # pragma: no cover - defensive unreachable guard
        "Routes retry loop exhausted without returning"
    )
