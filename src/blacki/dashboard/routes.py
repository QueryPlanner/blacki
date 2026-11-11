"""FastAPI routes for the private, local observability dashboard.

The dashboard is mounted by :mod:`blacki.server` as part of the local server.
This module does not add an application authentication layer: the supported
deployment boundary is the existing loopback bind and an operator-controlled
Tailscale Serve endpoint.

All responses from this router carry the same no-store and browser isolation
headers.  The dashboard contains user conversation data, so it must never be
cached or embedded by another page.
"""

from __future__ import annotations

import logging
import mimetypes
import re
from collections.abc import Mapping
from importlib import resources
from ipaddress import ip_address, ip_network
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from ..observability.ledger import default_usage_ledger_path
from ..observability.setup import get_log_dir
from ..utils.config import ServerEnv
from .data import DashboardStore
from .models import (
    DEFAULT_PAGE_SIZE,
    MAX_DATABASE_ROWS,
    MAX_PAGE_SIZE,
    MAX_SEARCH_LENGTH,
    JsonObject,
)
from .private_data import DashboardPrivateStore, valid_capture_id

logger = logging.getLogger(__name__)

_MAX_LIMIT = MAX_PAGE_SIZE
_MAX_OFFSET = MAX_DATABASE_ROWS
_MAX_QUERY_TEXT = MAX_SEARCH_LENGTH
_DEFAULT_LIMIT = DEFAULT_PAGE_SIZE
_DEFAULT_WINDOW = "24h"
_WINDOW_RE = re.compile(r"^(?:all|[1-9][0-9]{0,3}[hdw])$")
_SERVE_HOST_RE = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){2}ts\.net")
_TAILSCALE_NETWORKS = (
    ip_network("100.64.0.0/10"),
    ip_network("fd7a:115c:a1e0::/48"),
)
_ResponseT = TypeVar("_ResponseT", bound=Response)

_SECURITY_HEADERS: Mapping[str, str] = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
        "frame-src 'none'; frame-ancestors 'none'; object-src 'none'; "
        "base-uri 'self'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


class _InvalidDashboardQueryError(ValueError):
    """Internal marker for a safe, generic query-validation response."""


class DashboardStoreProtocol(Protocol):
    """Async store surface required by the HTTP route layer."""

    async def get_overview(self, window: str) -> JsonObject | None: ...

    async def list_users(
        self, search: str, limit: int, offset: int
    ) -> JsonObject | None: ...

    async def list_sessions(
        self, user_id: str, limit: int, offset: int
    ) -> JsonObject | None: ...

    async def get_session(self, user_id: str, session_id: str) -> JsonObject | None: ...

    async def list_logs(
        self, level: str | None, search: str, limit: int
    ) -> JsonObject | None: ...

    async def list_traces(
        self, status: str | None, search: str, limit: int
    ) -> JsonObject | None: ...

    async def get_trace(self, trace_id: str) -> JsonObject | None: ...


def _with_security_headers(response: _ResponseT) -> _ResponseT:
    """Attach the dashboard's cache and browser isolation policy."""
    for name, value in _SECURITY_HEADERS.items():
        response.headers[name] = value
    return response


def _json_response(content: Any, status_code: int = 200) -> JSONResponse:
    """Return JSON encoded by FastAPI's safe encoder with dashboard headers."""
    return _with_security_headers(
        JSONResponse(content=jsonable_encoder(content), status_code=status_code)
    )


def _error_response(status_code: int, message: str) -> JSONResponse:
    """Return a generic error without exposing paths or backend exceptions."""
    return _json_response({"error": message}, status_code=status_code)


def _invalid_query() -> JSONResponse:
    """Build the common query-validation response."""
    return _error_response(422, "Invalid dashboard query parameters.")


def _parse_int(
    raw: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Parse a bounded integer query value without leaking parser details."""
    value = raw.strip() if isinstance(raw, str) else ""
    if not value:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise _InvalidDashboardQueryError from exc
    if parsed < minimum or parsed > maximum:
        raise _InvalidDashboardQueryError
    return parsed


def _parse_text(
    raw: str | None,
    *,
    required: bool = False,
    max_length: int = _MAX_QUERY_TEXT,
) -> str | None:
    """Normalize a text query and reject unbounded values."""
    value = raw.strip() if raw is not None else ""
    if not value:
        if required:
            raise _InvalidDashboardQueryError
        return None
    if len(value) > max_length:
        raise _InvalidDashboardQueryError
    return value


def _asset_path(asset_name: str) -> Any:
    """Resolve a dashboard package resource after rejecting traversal."""
    relative = PurePosixPath(asset_name)
    if (
        not asset_name
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise FileNotFoundError

    if asset_name == "miniapp.css":
        return resources.files("blacki.miniapp").joinpath("static", "styles.css")

    package_root = resources.files("blacki.dashboard")
    roots = (
        (package_root.joinpath("templates"),)
        if asset_name == "index.html"
        else (package_root.joinpath("static"), package_root)
    )
    for root in roots:
        asset = root.joinpath(*relative.parts)
        if asset.is_file():
            return asset
    raise FileNotFoundError


def _asset_response(asset_name: str) -> Response:
    """Read and return an installed package asset, or a generic 404."""
    try:
        asset = _asset_path(asset_name)
        body = asset.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return _error_response(404, "Dashboard resource not found.")
    except Exception:
        logger.exception("Dashboard resource read failed")
        return _error_response(503, "Dashboard resources are unavailable.")

    media_type = mimetypes.guess_type(asset_name)[0] or "application/octet-stream"
    return _with_security_headers(Response(content=body, media_type=media_type))


async def _store_response(
    store: DashboardStoreProtocol | DashboardPrivateStore | None,
    method_name: str,
    *args: object,
) -> JSONResponse:
    """Call a store method and shield backend errors from the HTTP client."""
    if store is None:
        return _error_response(503, "Dashboard data is temporarily unavailable.")
    try:
        method = getattr(store, method_name)
        result = await method(*args)
    except Exception:
        logger.exception("Dashboard data query failed: %s", method_name)
        return _error_response(503, "Dashboard data is temporarily unavailable.")

    if result is None and method_name in {"get_session", "get_trace"}:
        return _error_response(404, "Dashboard record not found.")
    try:
        return _json_response(result)
    except Exception:
        logger.exception("Dashboard data serialization failed: %s", method_name)
        return _error_response(503, "Dashboard data is temporarily unavailable.")


def _session_db_path(env: ServerEnv) -> Path:
    """Return the ADK SQLite session database used by the dashboard."""
    return Path(env.agent_dir) / ".adk" / "sessions.db"


def _tools_db_path(env: ServerEnv) -> Path:
    """Return the local SQLite database containing dashboard identity labels."""
    return (
        Path(env.sqlite_path)
        if env.sqlite_path
        else Path(env.agent_dir) / ".adk" / "tools.db"
    )


def _private_dashboard_host(hostname: str) -> bool:
    hostname = hostname.removesuffix(".")
    if hostname == "localhost" or _SERVE_HOST_RE.fullmatch(hostname):
        return True
    try:
        address = ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback or any(address in net for net in _TAILSCALE_NETWORKS)


def _private_access(request: Request) -> JSONResponse | None:
    # Loopback or tailnet ACLs remain the access boundary. Check Host and
    # browser headers here for DNS rebinding and cross-origin reads.
    if request.headers.get("origin") is not None or request.headers.get(
        "sec-fetch-site"
    ) in {"cross-site", "same-site"}:
        return _error_response(403, "Open this view from the dashboard directly.")
    hosts = request.headers.getlist("host")
    try:
        # Starlette can fall back to the server address for a malformed Host.
        # Check the raw browser authority before considering any fallback.
        authority = urlsplit(f"//{hosts[0]}") if len(hosts) == 1 else None
        valid_host = (
            authority is not None
            and authority.username is None
            and not (authority.path or authority.query or authority.fragment)
            and authority.port != 0
            and _private_dashboard_host(authority.hostname or "")
        )
    except ValueError:
        valid_host = False
    if not valid_host:
        return _error_response(
            403, "Open private views from a loopback or Tailscale dashboard URL."
        )
    return None


def create_dashboard_router(
    env: ServerEnv,
    store: DashboardStoreProtocol | None = None,
) -> APIRouter:
    """Create routes backed by the local ADK session and observability files.

    A store can be supplied by tests or an embedding application.  Normal
    server construction uses the session database under ``AGENT_DIR`` and the
    log directory selected by :func:`blacki.observability.setup.get_log_dir`.
    """
    if store is not None:
        dashboard_store = store
    else:
        try:
            dashboard_store = DashboardStore(
                _session_db_path(env),
                get_log_dir(),
                "blacki",
                _tools_db_path(env),
                default_usage_ledger_path(env.agent_dir),
            )
        except Exception:
            logger.exception("Dashboard store initialization failed")
            dashboard_store = None
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])
    private_store = DashboardPrivateStore(
        _session_db_path(env),
        get_log_dir() / "blacki-traces.log",
        _tools_db_path(env).with_name("wiki.db"),
    )

    @router.get("")
    async def dashboard_page() -> Response:
        """Serve the dashboard shell from package resources."""
        return _asset_response("index.html")

    @router.get("/")
    async def dashboard_page_with_slash() -> Response:
        """Serve the dashboard shell with an optional trailing slash."""
        return _asset_response("index.html")

    @router.get("/assets/{asset_name:path}")
    async def dashboard_asset(asset_name: str) -> Response:
        """Serve a static dashboard asset from the installed package."""
        return _asset_response(asset_name)

    @router.get("/static/{asset_name:path}")
    async def dashboard_static_asset(asset_name: str) -> Response:
        """Compatibility alias for the dashboard's package asset path."""
        return _asset_response(asset_name)

    @router.get("/api/overview")
    async def dashboard_overview(window: str = _DEFAULT_WINDOW) -> JSONResponse:
        """Return aggregate dashboard metrics for a bounded time window."""
        try:
            window_value = _parse_text(window, required=True, max_length=16)
            if window_value is None or _WINDOW_RE.fullmatch(window_value) is None:
                raise _InvalidDashboardQueryError
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(dashboard_store, "get_overview", window_value)

    @router.get("/api/users")
    async def dashboard_users(
        search: str | None = None,
        limit: str = str(_DEFAULT_LIMIT),
        offset: str = "0",
    ) -> JSONResponse:
        """List users with bounded pagination and optional text filtering."""
        try:
            search_value = _parse_text(search)
            limit_value = _parse_int(
                limit,
                default=_DEFAULT_LIMIT,
                minimum=1,
                maximum=_MAX_LIMIT,
            )
            offset_value = _parse_int(
                offset,
                default=0,
                minimum=0,
                maximum=_MAX_OFFSET,
            )
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(
            dashboard_store,
            "list_users",
            search_value,
            limit_value,
            offset_value,
        )

    @router.get("/api/sessions")
    async def dashboard_sessions(
        user_id: str | None = None,
        limit: str = str(_DEFAULT_LIMIT),
        offset: str = "0",
    ) -> JSONResponse:
        """List sessions for one user with bounded pagination."""
        try:
            user_value = _parse_text(user_id, required=True)
            limit_value = _parse_int(
                limit,
                default=_DEFAULT_LIMIT,
                minimum=1,
                maximum=_MAX_LIMIT,
            )
            offset_value = _parse_int(
                offset,
                default=0,
                minimum=0,
                maximum=_MAX_OFFSET,
            )
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(
            dashboard_store,
            "list_sessions",
            user_value,
            limit_value,
            offset_value,
        )

    @router.get("/api/session")
    async def dashboard_session(
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> JSONResponse:
        """Return one session's conversation events."""
        try:
            user_value = _parse_text(user_id, required=True)
            session_value = _parse_text(session_id, required=True)
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(
            dashboard_store,
            "get_session",
            user_value,
            session_value,
        )

    @router.get("/api/context")
    async def dashboard_context(
        request: Request,
        user_id: str = "",
        session_id: str = "",
        trace_id: str = "",
        span_id: str = "",
        offset: str = "0",
    ) -> JSONResponse:
        """List captures, or read one captured prompt and its tool descriptions."""
        if (denial := _private_access(request)) is not None:
            return denial
        try:
            user = _parse_text(user_id, required=True)
            session = _parse_text(session_id, required=True)
            start = _parse_int(offset, default=0, minimum=0, maximum=_MAX_OFFSET)
            key = None
            if trace_id or span_id:
                if not valid_capture_id(trace_id, span_id):
                    raise _InvalidDashboardQueryError
                key = (trace_id, span_id)
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(
            private_store, "get_context", user, session, _DEFAULT_LIMIT, start, key
        )

    @router.get("/api/wiki")
    async def dashboard_wiki(
        request: Request,
        user_id: str = "",
        page: str = "",
        revision: str = "",
        offset: str = "0",
    ) -> JSONResponse:
        """Read one owner-scoped wiki page list position and its Markdown."""
        if (denial := _private_access(request)) is not None:
            return denial
        try:
            user = _parse_text(user_id, required=True)
            selected = (
                _parse_int(page, default=0, minimum=0, maximum=_MAX_OFFSET)
                if page
                else None
            )
            known_revision = (
                _parse_int(revision, default=0, minimum=0, maximum=2**63 - 1)
                if revision
                else None
            )
            if selected is not None and known_revision is None:
                raise _InvalidDashboardQueryError
            start = _parse_int(offset, default=0, minimum=0, maximum=_MAX_OFFSET)
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(
            private_store,
            "get_wiki",
            user,
            selected,
            known_revision,
            start,
        )

    @router.get("/api/logs")
    async def dashboard_logs(
        level: str | None = None,
        search: str | None = None,
        limit: str = str(_DEFAULT_LIMIT),
    ) -> JSONResponse:
        """List persisted JSON application logs."""
        try:
            level_value = _parse_text(level, max_length=32)
            search_value = _parse_text(search)
            limit_value = _parse_int(
                limit,
                default=_DEFAULT_LIMIT,
                minimum=1,
                maximum=_MAX_LIMIT,
            )
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(
            dashboard_store,
            "list_logs",
            level_value,
            search_value,
            limit_value,
        )

    @router.get("/api/traces")
    async def dashboard_traces(
        status: str | None = None,
        search: str | None = None,
        limit: str = str(_DEFAULT_LIMIT),
    ) -> JSONResponse:
        """List persisted OpenTelemetry spans."""
        try:
            status_value = _parse_text(status, max_length=32)
            search_value = _parse_text(search)
            limit_value = _parse_int(
                limit,
                default=_DEFAULT_LIMIT,
                minimum=1,
                maximum=_MAX_LIMIT,
            )
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(
            dashboard_store,
            "list_traces",
            status_value,
            search_value,
            limit_value,
        )

    @router.get("/api/trace")
    async def dashboard_trace(trace_id: str | None = None) -> JSONResponse:
        """Return one trace by ID."""
        try:
            trace_value = _parse_text(trace_id, required=True)
        except _InvalidDashboardQueryError:
            return _invalid_query()
        return await _store_response(dashboard_store, "get_trace", trace_value)

    @router.api_route(
        "/{unknown_path:path}",
        methods=["GET", "HEAD", "OPTIONS"],
        include_in_schema=False,
    )
    async def dashboard_unknown_path(unknown_path: str) -> JSONResponse:
        """Return a dashboard-scoped 404 with the standard security headers."""
        del unknown_path
        return _error_response(404, "Dashboard resource not found.")

    return router


__all__ = ["create_dashboard_router"]
