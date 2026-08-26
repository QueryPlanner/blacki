"""Small async REST client for the Google Health API and Google OAuth."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import Any

import httpx

from .config import (
    GOOGLE_HEALTH_API_BASE_URL,
    GOOGLE_HEALTH_REVOCATION_URL,
    GOOGLE_HEALTH_TOKEN_URL,
    GoogleHealthConfig,
)

logger = logging.getLogger(__name__)


class GoogleHealthApiError(RuntimeError):
    """A safe, non-payload Google Health or OAuth HTTP error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        retry_after_seconds: float | None = None,
        transport: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after_seconds = retry_after_seconds
        self.transport = transport


class GoogleHealthAuthError(GoogleHealthApiError):
    """Raised when Google requires the user to authorize again."""


@dataclass(frozen=True, slots=True)
class GoogleTokenResponse:
    """Relevant fields from an OAuth token response."""

    access_token: str
    expires_in: int | None
    refresh_token: str | None
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoogleHealthIdentity:
    """Server-side identity mapping returned by the Health API."""

    health_user_id: str
    legacy_fitbit_user_id: str | None


@dataclass(frozen=True, slots=True)
class GoogleHealthOperation:
    """The safe subset of a Health API long-running operation."""

    done: bool
    name: str | None = None
    error_code: str | None = None
    response: dict[str, Any] | None = None

    @property
    def successful(self) -> bool:
        """Return true only for a completed operation without an error."""
        return self.done and self.error_code is None


class GoogleHealthClient:
    """Async client that never logs OAuth tokens or health payloads."""

    def __init__(
        self,
        config: GoogleHealthConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._client = http_client

    async def close(self) -> None:
        """Close the internally owned HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def exchange_code(self, code: str) -> GoogleTokenResponse:
        """Exchange a short-lived authorization code for OAuth tokens."""
        result = await self._token_request(
            {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.config.redirect_uri,
            }
        )
        return _parse_token_response(result, require_refresh_token=False)

    async def refresh_access_token(self, refresh_token: str) -> GoogleTokenResponse:
        """Exchange an encrypted-at-rest refresh token for a short-lived token."""
        result = await self._token_request(
            {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        )
        return _parse_token_response(result, require_refresh_token=False)

    async def revoke_token(self, refresh_token: str) -> None:
        """Ask Google to revoke a refresh token."""
        client = await self._ensure_client()
        response = await client.post(
            GOOGLE_HEALTH_REVOCATION_URL,
            params={"token": refresh_token},
            timeout=30.0,
        )
        if response.status_code >= 400:
            raise GoogleHealthApiError(
                "Google token revocation failed",
                status_code=response.status_code,
            )

    async def get_identity(self, access_token: str) -> GoogleHealthIdentity:
        """Retrieve the Health API identity mapping after user consent."""
        payload = await self._api_request(
            "GET",
            "/v4/users/me/identity",
            access_token=access_token,
        )
        health_user_id = payload.get("healthUserId")
        if not isinstance(health_user_id, str) or not health_user_id:
            raise GoogleHealthApiError("Google Health identity response was incomplete")
        legacy_user_id = payload.get("legacyUserId")
        return GoogleHealthIdentity(
            health_user_id=health_user_id,
            legacy_fitbit_user_id=(
                legacy_user_id if isinstance(legacy_user_id, str) else None
            ),
        )

    async def list_data_points(
        self,
        access_token: str,
        data_type: str,
        *,
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        """List a bounded data type range, following pagination tokens."""
        page_token: str | None = None
        points: list[dict[str, Any]] = []
        filter_expression = _filter_for_data_type(data_type, start_time, end_time)
        page_size = 25 if data_type in {"exercise", "sleep"} else 10000

        while True:
            params: dict[str, Any] = {
                "pageSize": page_size,
                "filter": filter_expression,
            }
            if page_token:
                params["pageToken"] = page_token
            payload = await self._api_request(
                "GET",
                f"/v4/users/me/dataTypes/{data_type}/dataPoints",
                access_token=access_token,
                params=params,
            )
            raw_points = payload.get("dataPoints", [])
            if isinstance(raw_points, list):
                points.extend(point for point in raw_points if isinstance(point, dict))
            next_page_token = payload.get("nextPageToken")
            if not isinstance(next_page_token, str) or not next_page_token:
                return points
            page_token = next_page_token

    async def create_nutrition_log(
        self,
        access_token: str,
        resource_name: str,
        payload: Mapping[str, Any],
    ) -> GoogleHealthOperation:
        """Create one named anonymous nutrition log.

        The API requires the canonical account in both the parent path and the
        DataPoint ``name``. ``payload`` is the DataPoint value, usually
        ``{"nutritionLog": ...}``; this method adds only the immutable name.
        """
        parent = _nutrition_parent(resource_name)
        body = dict(payload)
        body["name"] = resource_name
        result = await self._api_request(
            "POST",
            f"/v4/{parent}/dataPoints",
            access_token=access_token,
            json_body=body,
        )
        return _parse_operation(result)

    async def get_data_point(
        self, access_token: str, resource_name: str
    ) -> dict[str, Any]:
        """Fetch one exact named data point for write reconciliation."""
        _nutrition_parent(resource_name)
        return await self._api_request(
            "GET",
            f"/v4/{resource_name}",
            access_token=access_token,
        )

    async def delete_nutrition_log(
        self, access_token: str, resource_name: str
    ) -> GoogleHealthOperation:
        """Request deletion of one named nutrition log."""
        parent = _nutrition_parent(resource_name)
        result = await self._api_request(
            "POST",
            f"/v4/{parent}/dataPoints:batchDelete",
            access_token=access_token,
            json_body={"names": [resource_name]},
        )
        return _parse_operation(result)

    async def _token_request(self, form: Mapping[str, str]) -> dict[str, Any]:
        client = await self._ensure_client()
        try:
            response = await client.post(
                GOOGLE_HEALTH_TOKEN_URL, data=dict(form), timeout=30.0
            )
        except httpx.RequestError as exc:
            raise GoogleHealthApiError(
                "Google OAuth request could not reach the provider",
                error_code="transport_error",
                transport=True,
            ) from exc
        payload = _json_object(
            response, retry_after_seconds=_retry_after_seconds(response)
        )
        if response.status_code >= 400:
            _raise_provider_error(
                payload,
                response.status_code,
                token_endpoint=True,
                retry_after_seconds=_retry_after_seconds(response),
            )
        return payload

    async def _api_request(
        self,
        method: str,
        path: str,
        *,
        access_token: str,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = await self._ensure_client()
        try:
            response = await client.request(
                method,
                f"{GOOGLE_HEALTH_API_BASE_URL}{path}",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
                params=dict(params) if params is not None else None,
                json=dict(json_body) if json_body is not None else None,
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            raise GoogleHealthApiError(
                "Google Health request could not reach the provider",
                error_code="transport_error",
                transport=True,
            ) from exc
        payload = _json_object(
            response, retry_after_seconds=_retry_after_seconds(response)
        )
        if response.status_code >= 400:
            _raise_provider_error(
                payload,
                response.status_code,
                token_endpoint=False,
                retry_after_seconds=_retry_after_seconds(response),
            )
        return payload

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client


def _parse_token_response(
    payload: Mapping[str, Any], *, require_refresh_token: bool
) -> GoogleTokenResponse:
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    if not isinstance(access_token, str) or not access_token:
        raise GoogleHealthApiError(
            "Google OAuth response did not contain an access token"
        )
    if require_refresh_token and (
        not isinstance(refresh_token, str) or not refresh_token
    ):
        raise GoogleHealthApiError(
            "Google OAuth response did not contain a refresh token"
        )
    expires_in = payload.get("expires_in")
    if not isinstance(expires_in, int):
        expires_in = None
    raw_scope = payload.get("scope", "")
    scopes = (
        tuple(item for item in raw_scope.split() if item)
        if isinstance(raw_scope, str)
        else ()
    )
    return GoogleTokenResponse(
        access_token=access_token,
        expires_in=expires_in,
        refresh_token=refresh_token if isinstance(refresh_token, str) else None,
        scopes=scopes,
    )


def _json_object(
    response: httpx.Response, *, retry_after_seconds: float | None = None
) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise GoogleHealthApiError(
            "Google returned a non-JSON response",
            status_code=response.status_code,
            retry_after_seconds=retry_after_seconds,
        ) from exc
    if not isinstance(payload, dict):
        raise GoogleHealthApiError(
            "Google returned an unexpected response",
            status_code=response.status_code,
            retry_after_seconds=retry_after_seconds,
        )
    return payload


def _raise_provider_error(
    payload: Mapping[str, Any],
    status_code: int,
    *,
    token_endpoint: bool,
    retry_after_seconds: float | None = None,
) -> None:
    raw_error = payload.get("error")
    safe_error_code: str | None = None
    if isinstance(raw_error, str):
        safe_error_code = raw_error
    elif isinstance(raw_error, Mapping):
        details = raw_error.get("details")
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, Mapping):
                    continue
                reason = detail.get("reason")
                if isinstance(reason, str) and reason:
                    safe_error_code = reason
                    break
        if safe_error_code is None:
            status = raw_error.get("status")
            if isinstance(status, str) and status:
                safe_error_code = status
    message = (
        "Google OAuth request failed"
        if token_endpoint
        else "Google Health request failed"
    )
    error_type = (
        GoogleHealthAuthError if status_code in {401, 403} else GoogleHealthApiError
    )
    raise error_type(
        message,
        status_code=status_code,
        error_code=safe_error_code,
        retry_after_seconds=retry_after_seconds,
    )


def _parse_operation(payload: Mapping[str, Any]) -> GoogleHealthOperation:
    """Parse an Operation without exposing provider payloads in exceptions."""
    done = payload.get("done")
    if not isinstance(done, bool):
        raise GoogleHealthApiError("Google Health operation response was incomplete")
    raw_name = payload.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else None
    raw_error = payload.get("error")
    error_code: str | None = None
    if isinstance(raw_error, Mapping):
        status = raw_error.get("status")
        if isinstance(status, str) and status.isascii() and status.isprintable():
            error_code = status[:80]
        elif isinstance(raw_error.get("code"), int):
            error_code = f"provider_error_{raw_error['code']}"
        else:
            error_code = "provider_error"
    elif isinstance(raw_error, str) and raw_error.isascii() and raw_error.isprintable():
        error_code = raw_error[:80]
    elif raw_error is not None:
        error_code = "provider_error"
    raw_response = payload.get("response")
    response = dict(raw_response) if isinstance(raw_response, Mapping) else None
    if done and error_code is None and response is None:
        raise GoogleHealthApiError("Google Health operation response was incomplete")
    return GoogleHealthOperation(
        done=done,
        name=name,
        error_code=error_code,
        response=response,
    )


def _nutrition_parent(resource_name: str) -> str:
    """Validate and return the canonical parent for a nutrition data point."""
    match = re.fullmatch(
        r"users/[^/?#]+/dataTypes/nutrition-log/dataPoints/"
        r"[a-z0-9-]{4,63}",
        resource_name,
    )
    if match is None:
        raise ValueError("resource_name is not a valid nutrition data point name")
    return resource_name.rsplit("/dataPoints/", 1)[0]


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a bounded Retry-After header without trusting arbitrary values."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = retry_at.timestamp() - datetime.now(UTC).timestamp()
    if not isfinite(seconds):
        return None
    return max(0.0, seconds)


def _filter_for_data_type(data_type: str, start_time: str, end_time: str) -> str:
    if data_type in {"daily-resting-heart-rate", "daily-heart-rate-zones"}:
        filter_name = data_type.replace("-", "_")
        return (
            f'{filter_name}.date >= "{start_time[:10]}" '
            f'AND {filter_name}.date < "{end_time[:10]}"'
        )
    if data_type == "exercise":
        return (
            f'exercise.interval.civil_start_time >= "{start_time[:10]}" AND '
            f'exercise.interval.civil_start_time < "{end_time[:10]}"'
        )
    if data_type in {"weight", "body-fat"}:
        filter_name = data_type.replace("-", "_")
        return (
            f'{filter_name}.sample_time.physical_time >= "{start_time}" AND '
            f'{filter_name}.sample_time.physical_time < "{end_time}"'
        )
    if data_type == "sleep":
        return (
            f'sleep.interval.end_time >= "{start_time}" AND '
            f'sleep.interval.end_time < "{end_time}"'
        )
    return (
        f'{data_type.replace("-", "_")}.interval.start_time >= "{start_time}" AND '
        f'{data_type.replace("-", "_")}.interval.start_time < "{end_time}"'
    )
