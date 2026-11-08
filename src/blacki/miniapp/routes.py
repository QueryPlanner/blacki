"""HTTP surface of the Mini App, served as its own small FastAPI app.

It runs on ``MINIAPP_PORT`` (default 8090), separate from the main server, so
exposing it to Telegram through a tunnel exposes only ``/miniapp``.
"""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import Mapping
from importlib import resources
from pathlib import PurePosixPath
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from .auth import InitDataError, SessionStore, verify_init_data
from .config import MiniAppConfig
from .service import ActionError, get_miniapp_service, hooks
from .sources import MiniAppUser

logger = logging.getLogger(__name__)

_CSP = (
    "default-src 'none'; base-uri 'none'; form-action 'none'; object-src 'none'; "
    "script-src 'self' https://telegram.org; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; manifest-src 'self'; "
    "frame-ancestors 'self' https://web.telegram.org https://*.telegram.org"
)
_HEADERS: Mapping[str, str] = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Resource-Policy": "same-origin",
}
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_PROXY_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "forwarded",
    "x-real-ip",
    "cf-connecting-ip",
    "true-client-ip",
    "tailscale-user-login",
    "tailscale-user-name",
)


class SessionRequest(BaseModel):
    init_data: str = Field(default="", max_length=4096)
    dev: bool = False


class ActionRequest(BaseModel):
    action: str = Field(max_length=40)
    page: str = Field(default="", max_length=40)
    card: str = Field(default="", max_length=40)
    args: dict[str, Any] = Field(default_factory=dict)


def _secure(response: Response) -> Response:
    for key, value in _HEADERS.items():
        response.headers[key] = value
    return response


def _json(payload: Any, status: int = 200) -> Response:
    return _secure(JSONResponse(payload, status_code=status))


def _asset(name: str) -> Response:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in {"", ".", ".."} for p in path.parts):
        return _json({"error": "not found"}, 404)
    file = resources.files("blacki.miniapp").joinpath("static", *path.parts)
    try:
        if not file.is_file():
            return _json({"error": "not found"}, 404)
        body = file.read_bytes()
    except OSError:
        return _json({"error": "not found"}, 404)
    media = mimetypes.guess_type(name)[0] or "application/octet-stream"
    if media.startswith("text/") or media in {"application/javascript"}:
        media += "; charset=utf-8"
    return _secure(Response(body, media_type=media))


def _is_strict_loopback(request: Request) -> bool:
    """True only for a direct local request that no proxy or tunnel touched."""
    client = request.scope.get("client")
    host_header = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
    if not client or client[0] not in _LOOPBACK or host_header not in _LOOPBACK:
        return False
    return not any(h in request.headers for h in _PROXY_HEADERS)


def create_miniapp_app(config: MiniAppConfig, bot_token: str | None) -> FastAPI:
    """Build the standalone Mini App web app."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    sessions = SessionStore(config.session_ttl_seconds)
    dev_users: set[int] = set()

    async def current_user(authorization: str | None) -> MiniAppUser:
        token = (authorization or "").removeprefix("Bearer ").strip()
        user_id = sessions.resolve(token)
        if user_id is None:
            raise HTTPException(401, "sign in again")
        # Re-check access every request so a rotated access code revokes sessions.
        if user_id not in dev_users and (
            hooks.authorize is None or not await hooks.authorize(user_id)
        ):
            raise HTTPException(403, "no access")
        return MiniAppUser(user_id)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> Response:
        return _json({"error": exc.detail}, exc.status_code)

    @app.get("/", include_in_schema=False)
    async def root() -> Response:
        return RedirectResponse("/miniapp/", status_code=307)

    @app.get("/miniapp", include_in_schema=False)
    @app.get("/miniapp/", include_in_schema=False)
    async def page() -> Response:
        return _asset("index.html")

    @app.get("/miniapp/static/{name:path}", include_in_schema=False)
    async def static(name: str) -> Response:
        return _asset(name)

    @app.post("/miniapp/api/session")
    async def session(body: SessionRequest, request: Request) -> Response:
        if body.dev:
            if not config.dev_shortcut_enabled or not _is_strict_loopback(request):
                raise HTTPException(403, "the local test sign-in is off")
            user_id = int(config.dev_telegram_user_id or 0)
            dev_users.add(user_id)
            logger.warning("Mini App local test sign-in used for a configured user")
            return _json({"token": sessions.issue(user_id), "dev": True})
        if not bot_token:
            raise HTTPException(503, "Telegram isn't configured")
        try:
            user_id = verify_init_data(
                body.init_data,
                bot_token,
                max_age_seconds=config.init_data_max_age_seconds,
            )
        except InitDataError as exc:
            raise HTTPException(401, str(exc)) from exc
        if hooks.authorize is None or not await hooks.authorize(user_id):
            raise HTTPException(
                403, "this Telegram account doesn't have access to Blacki"
            )
        return _json({"token": sessions.issue(user_id), "dev": False})

    @app.get("/miniapp/api/nav")
    async def nav(authorization: str | None = Header(default=None)) -> Response:
        user = await current_user(authorization)
        service = get_miniapp_service()
        prefs = {
            "theme": await service.store.get_pref(user.chat_user_id, "theme"),
            "targets": await service.store.get_pref(user.chat_user_id, "targets") or {},
        }
        return _json({"pages": await service.nav(user), "prefs": prefs})

    @app.get("/miniapp/api/page/{page_id}")
    async def load_page(
        page_id: str, authorization: str | None = Header(default=None)
    ) -> Response:
        user = await current_user(authorization)
        if len(page_id) > 40:
            raise HTTPException(404, "page not found")
        try:
            return _json(await get_miniapp_service().load(user, page_id))
        except LookupError as exc:
            raise HTTPException(404, "page not found") from exc

    @app.get("/miniapp/api/card/{card_id}")
    async def load_card(
        card_id: str, authorization: str | None = Header(default=None)
    ) -> Response:
        user = await current_user(authorization)
        if len(card_id) > 40:
            raise HTTPException(404, "card not found")
        try:
            return _json(await get_miniapp_service().load_card(user, card_id))
        except LookupError as exc:
            raise HTTPException(404, "card not found or expired") from exc

    @app.post("/miniapp/api/action")
    async def action(
        body: ActionRequest, authorization: str | None = Header(default=None)
    ) -> Response:
        user = await current_user(authorization)
        try:
            result = await get_miniapp_service().run_action(user, body.model_dump())
        except ActionError as exc:
            return _json({"status": "error", "message": str(exc)}, 400)
        except Exception:
            logger.exception("Mini App action %s failed", body.action)
            return _json(
                {
                    "status": "error",
                    "message": "that didn't work; reload the page to see what was saved",  # noqa: E501
                },
                500,
            )
        return _json(result)

    return app
