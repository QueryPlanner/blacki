"""Small async REST client for the Google Health API and Google OAuth."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


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

    async def _token_request(self, form: Mapping[str, str]) -> dict[str, Any]:
        client = await self._ensure_client()
        response = await client.post(
            GOOGLE_HEALTH_TOKEN_URL, data=dict(form), timeout=30.0
        )
        payload = _json_object(response)
        if response.status_code >= 400:
            _raise_provider_error(payload, response.status_code, token_endpoint=True)
        return payload

    async def _api_request(
        self,
        method: str,
        path: str,
        *,
        access_token: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = await self._ensure_client()
        response = await client.request(
            method,
            f"{GOOGLE_HEALTH_API_BASE_URL}{path}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            params=dict(params) if params is not None else None,
            timeout=30.0,
        )
        payload = _json_object(response)
        if response.status_code >= 400:
            _raise_provider_error(payload, response.status_code, token_endpoint=False)
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


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise GoogleHealthApiError(
            "Google returned a non-JSON response", status_code=response.status_code
        ) from exc
    if not isinstance(payload, dict):
        raise GoogleHealthApiError(
            "Google returned an unexpected response", status_code=response.status_code
        )
    return payload


def _raise_provider_error(
    payload: Mapping[str, Any], status_code: int, *, token_endpoint: bool
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
    raise error_type(message, status_code=status_code, error_code=safe_error_code)


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
