"""Tests for the private takeover page and WebSocket proxy."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from blacki.browser_takeover.config import BrowserTakeoverConfig
from blacki.browser_takeover.routes import (
    COOKIE_NAME,
    _valid_client_message,
    create_browser_takeover_router,
)


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(create_browser_takeover_router())
    return app


def _service() -> SimpleNamespace:
    return SimpleNamespace(
        config=BrowserTakeoverConfig(
            "http://127.0.0.1/browser-takeover",
            ttl_seconds=60,
        ),
        redeem=AsyncMock(return_value="browser-cookie"),
        complete=AsyncMock(return_value=True),
        authorize=AsyncMock(return_value=None),
    )


def test_page_is_hidden_when_takeover_is_disabled() -> None:
    with patch("blacki.browser_takeover.routes._service", return_value=None):
        response = TestClient(_app()).get("/browser-takeover")

    assert response.status_code == 404


def test_page_has_private_security_headers() -> None:
    service = _service()
    with patch("blacki.browser_takeover.routes._service", return_value=service):
        response = TestClient(_app()).get("/browser-takeover")

    assert response.status_code == 200
    assert "Private browser control" in response.text
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"


def test_redeem_is_single_origin_and_sets_http_only_cookie() -> None:
    service = _service()
    client = TestClient(_app())
    with patch("blacki.browser_takeover.routes._service", return_value=service):
        rejected = client.post(
            "/browser-takeover/redeem",
            json={"token": "one-time"},
            headers={"origin": "https://attacker.test"},
        )
        response = client.post(
            "/browser-takeover/redeem",
            json={"token": "one-time"},
            headers={"origin": "http://127.0.0.1"},
        )

    assert rejected.status_code == 404
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert COOKIE_NAME in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    service.redeem.assert_awaited_once_with("one-time")


def test_redeem_rejects_bad_json_and_expired_token() -> None:
    service = _service()
    service.redeem.return_value = None
    with patch("blacki.browser_takeover.routes._service", return_value=service):
        response = TestClient(_app()).post(
            "/browser-takeover/redeem",
            content="not-json",
            headers={
                "content-type": "application/json",
                "origin": "http://127.0.0.1",
            },
        )

    assert response.status_code == 404
    service.redeem.assert_awaited_once_with("")

    with patch("blacki.browser_takeover.routes._service", return_value=service):
        too_large = TestClient(_app()).post(
            "/browser-takeover/redeem",
            content="x" * 4097,
            headers={"origin": "http://127.0.0.1"},
        )
    assert too_large.status_code == 404


def test_complete_requires_origin_and_valid_cookie() -> None:
    service = _service()
    client = TestClient(_app())
    client.cookies.set(COOKIE_NAME, "browser-cookie")
    with patch("blacki.browser_takeover.routes._service", return_value=service):
        rejected = client.post(
            "/browser-takeover/complete",
            headers={"origin": "https://attacker.test"},
        )
        response = client.post(
            "/browser-takeover/complete",
            headers={"origin": "http://127.0.0.1"},
        )

    assert rejected.status_code == 404
    assert response.status_code == 200
    assert response.json() == {"status": "complete"}
    assert "Max-Age=0" in response.headers["set-cookie"]
    service.complete.assert_awaited_once_with("browser-cookie")


def test_invalid_client_messages_are_not_forwardable() -> None:
    assert _valid_client_message('{"type":"input_keyboard","key":"x"}') is True
    assert _valid_client_message('{"type":"run_command"}') is False
    assert _valid_client_message("not-json") is False
    assert _valid_client_message("x" * (16 * 1024 + 1)) is False


def test_websocket_rejects_missing_cookie() -> None:
    service = _service()
    with (
        patch("blacki.browser_takeover.routes._service", return_value=service),
        pytest.raises(WebSocketDisconnect) as caught,
        TestClient(_app()).websocket_connect("/browser-takeover/ws"),
    ):
        pass

    assert caught.value.code == 4401


class _Upstream:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def __aenter__(self) -> "_Upstream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __aiter__(self) -> "_Upstream":
        return self

    async def __anext__(self) -> str:
        if self.sent:
            raise StopAsyncIteration
        self.sent.append("upstream-started")
        return '{"type":"status","connected":true}'

    async def send(self, message: str) -> None:
        self.sent.append(message)


def test_websocket_proxies_authorized_stream_without_exposing_endpoint() -> None:
    service = _service()
    service.authorize.return_value = SimpleNamespace(
        upstream_url="wss://sandbox.internal/private",
        upstream_headers={"X-Route": "secret"},
    )
    upstream = _Upstream()
    client = TestClient(_app())
    client.cookies.set(COOKIE_NAME, "browser-cookie")
    with (
        patch("blacki.browser_takeover.routes._service", return_value=service),
        patch("blacki.browser_takeover.routes.connect", return_value=upstream) as dial,
        client.websocket_connect(
            "/browser-takeover/ws",
            headers={"origin": "http://127.0.0.1"},
        ) as websocket,
    ):
        assert websocket.receive_text() == '{"type":"status","connected":true}'

    dial.assert_called_once_with(
        "wss://sandbox.internal/private",
        additional_headers={"X-Route": "secret"},
        max_size=8 * 1024 * 1024,
    )
    service.authorize.assert_awaited_once_with("browser-cookie")
