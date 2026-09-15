"""Tests for the private takeover page and WebSocket proxy."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

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
    with patch(
        "blacki.browser_takeover.routes.get_browser_takeover_service",
        return_value=None,
    ):
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


def test_redeem_rejects_non_string_tokens() -> None:
    service = _service()
    service.redeem.return_value = None
    with patch("blacki.browser_takeover.routes._service", return_value=service):
        response = TestClient(_app()).post(
            "/browser-takeover/redeem",
            json={"token": 123},
            headers={"origin": "http://127.0.0.1"},
        )

    assert response.status_code == 404
    service.redeem.assert_awaited_once_with("")


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


def test_websocket_rejects_missing_cookie_with_matching_origin() -> None:
    service = _service()
    with (
        patch("blacki.browser_takeover.routes._service", return_value=service),
        pytest.raises(WebSocketDisconnect) as caught,
        TestClient(_app()).websocket_connect(
            "/browser-takeover/ws",
            headers={"origin": "http://127.0.0.1"},
        ),
    ):
        pass

    assert caught.value.code == 4401
    service.authorize.assert_awaited_once_with(None)


def test_websocket_is_hidden_when_takeover_is_disabled() -> None:
    with (
        patch("blacki.browser_takeover.routes._service", return_value=None),
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


class _InteractiveUpstream:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._message_received = asyncio.Event()
        self._yielded_frame = False

    async def __aenter__(self) -> "_InteractiveUpstream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __aiter__(self) -> "_InteractiveUpstream":
        return self

    async def __anext__(self) -> bytes:
        await self._message_received.wait()
        if self._yielded_frame:
            raise StopAsyncIteration
        self._yielded_frame = True
        return b"frame"

    async def send(self, message: str) -> None:
        self.sent.append(message)
        self._message_received.set()


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
    service.complete.assert_not_awaited()


def test_websocket_forwards_valid_input_and_binary_frames() -> None:
    service = _service()
    service.authorize.return_value = SimpleNamespace(
        upstream_url="wss://sandbox.internal/private",
        upstream_headers={"X-Route": "secret"},
    )
    upstream = _InteractiveUpstream()
    client = TestClient(_app())
    client.cookies.set(COOKIE_NAME, "browser-cookie")
    message = '{"type":"input_keyboard","key":"x"}'
    with (
        patch("blacki.browser_takeover.routes._service", return_value=service),
        patch("blacki.browser_takeover.routes.connect", return_value=upstream),
        client.websocket_connect(
            "/browser-takeover/ws",
            headers={"origin": "http://127.0.0.1"},
        ) as websocket,
    ):
        websocket.send_text('{"type":"run_command"}')
        websocket.send_text(message)
        assert websocket.receive_bytes() == b"frame"

    assert upstream.sent == [message]
    service.complete.assert_not_awaited()


def test_websocket_ignores_upstream_disconnect() -> None:
    service = _service()
    service.authorize.return_value = SimpleNamespace(
        upstream_url="wss://sandbox.internal/private",
        upstream_headers={},
    )
    client = TestClient(_app())
    client.cookies.set(COOKIE_NAME, "browser-cookie")
    with (
        patch("blacki.browser_takeover.routes._service", return_value=service),
        patch(
            "blacki.browser_takeover.routes.connect",
            side_effect=ConnectionClosed(None, None),
        ),
        client.websocket_connect(
            "/browser-takeover/ws",
            headers={"origin": "http://127.0.0.1"},
        ),
    ):
        pass

    service.complete.assert_not_awaited()


def test_websocket_closes_on_unexpected_upstream_error() -> None:
    service = _service()
    service.authorize.return_value = SimpleNamespace(
        upstream_url="wss://sandbox.internal/private",
        upstream_headers={},
    )
    client = TestClient(_app())
    client.cookies.set(COOKIE_NAME, "browser-cookie")
    with (
        patch("blacki.browser_takeover.routes._service", return_value=service),
        patch(
            "blacki.browser_takeover.routes.connect",
            side_effect=RuntimeError("upstream unavailable"),
        ),
        client.websocket_connect(
            "/browser-takeover/ws",
            headers={"origin": "http://127.0.0.1"},
        ) as websocket,
        pytest.raises(WebSocketDisconnect) as caught,
    ):
        websocket.receive_text()

    assert caught.value.code == 1011
    service.complete.assert_not_awaited()
