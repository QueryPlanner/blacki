"""Static Telegram Mini App and isolated local demo routes."""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import Mapping
from importlib import resources
from pathlib import PurePosixPath
from typing import TypeVar

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from blacki.utils.config import ServerEnv

logger = logging.getLogger(__name__)
_ResponseT = TypeVar("_ResponseT", bound=Response)

_SECURITY_HEADERS: Mapping[str, str] = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; base-uri 'none'; form-action 'self'; "
        "connect-src 'self'; font-src 'self'; img-src 'self' data: blob:; "
        "media-src 'self' blob:; script-src 'self' https://telegram.org; "
        "style-src 'self'; object-src 'none'; frame-src 'none'; "
        "frame-ancestors 'self' https://web.telegram.org https://*.telegram.org; "
        "worker-src 'none'; manifest-src 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def _secure(response: _ResponseT) -> _ResponseT:
    for name, value in _SECURITY_HEADERS.items():
        response.headers[name] = value
    return response


def _error(status_code: int, message: str) -> Response:
    return _secure(JSONResponse({"error": message}, status_code=status_code))


def _asset(asset_name: str) -> Response:
    path = PurePosixPath(asset_name)
    if (
        not asset_name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return _error(404, "Mini App resource not found.")

    try:
        package = resources.files("blacki.miniapp")
        root = (
            package.joinpath("templates")
            if asset_name == "index.html"
            else package.joinpath("static")
        )
        file = root.joinpath(*path.parts)
        if not file.is_file():
            return _error(404, "Mini App resource not found.")
        body = file.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return _error(404, "Mini App resource not found.")
    except Exception:
        logger.exception("Mini App static asset read failed")
        return _error(503, "Mini App resources are unavailable.")

    media_type = mimetypes.guess_type(asset_name)[0] or "application/octet-stream"
    return _secure(Response(body, media_type=media_type))


def create_miniapp_router(env: ServerEnv) -> APIRouter:
    """Create the disabled-by-default Mini App asset router.

    The first release only serves a fixture-only demo. Authenticated live API
    routes are added separately and remain controlled by ``MINIAPP_ENABLED``.
    """
    router = APIRouter()

    @router.get("/miniapp", include_in_schema=False)
    async def miniapp_page() -> Response:
        if not env.miniapp_demo_enabled:
            return _error(404, "Mini App demo is disabled.")
        return _asset("index.html")

    @router.get("/miniapp/static/{asset_name:path}", include_in_schema=False)
    async def miniapp_asset(asset_name: str) -> Response:
        if not env.miniapp_demo_enabled:
            return _error(404, "Mini App demo is disabled.")
        return _asset(asset_name)

    return router
