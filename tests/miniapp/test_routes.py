from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections.abc import AsyncGenerator, Iterator
from typing import Any
from unittest.mock import create_autospec, patch

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

import blacki.miniapp.service as svc_module
from blacki.container import AppContainer
from blacki.miniapp.config import MiniAppConfig
from blacki.miniapp.routes import (
    _CSP,
    _asset,
    _is_strict_loopback,
    create_miniapp_app,
)
from blacki.miniapp.service import (
    MiniAppService,
    hooks,
)
from blacki.miniapp.sources import MiniAppUser
from blacki.miniapp.store import MiniAppStore


async def _allow_user(user_id: int) -> bool:
    """Real authorizer signature; tests stub the return value via autospec."""
    return True


def _authorizer(allowed: bool | None) -> Any:
    """Build an autospec'd authorizer hook that returns ``allowed``."""
    return create_autospec(_allow_user, spec_set=True, return_value=allowed)


@pytest.fixture(autouse=True)
def restore_authorizer() -> Iterator[None]:
    """The Hooks singleton is process-wide; restore it after every test."""
    original = hooks.authorize
    yield
    hooks.authorize = original


def _make_init_data(user_id: int, bot_token: str) -> str:
    now = int(time.time())
    user_json = json.dumps({"id": user_id, "first_name": "Test"})
    fields = {"auth_date": str(now), "user": user_json}
    sorted_fields = sorted(fields.items())
    check_string = "\n".join(f"{k}={v}" for k, v in sorted_fields)
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    sig = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    return "&".join(f"{k}={v}" for k, v in sorted_fields) + f"&hash={sig}"


@pytest.fixture
async def service() -> AsyncGenerator[MiniAppService]:
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    lock = asyncio.Lock()
    container = AppContainer(conn, lock)
    store = MiniAppStore(conn, lock)
    await store.initialize()
    svc = MiniAppService(container=container, store=store)
    svc_module._service = svc
    yield svc
    svc_module._service = None
    await conn.close()


@pytest.fixture
def config() -> MiniAppConfig:
    return MiniAppConfig(
        dev_telegram_user_id=12345,
        session_ttl_seconds=3600,
        init_data_max_age_seconds=86400,
    )


@pytest.mark.asyncio
async def test_asset_loading() -> None:
    # 1. Invalid names / traversal
    res404 = _asset("")
    assert res404.status_code == 404

    res_abs = _asset("/etc/passwd")
    assert res_abs.status_code == 404

    res_dot = _asset("static/../secret")
    assert res_dot.status_code == 404

    res_empty_part = _asset("a//b")
    assert res_empty_part.status_code == 404

    # 2. Non-existent file
    res_missing = _asset("nonexistent_file_abc.txt")
    assert res_missing.status_code == 404

    # 3. Existing file (index.html)
    res_index = _asset("index.html")
    assert res_index.status_code == 200
    assert res_index.media_type is not None and "text/html" in res_index.media_type
    assert res_index.headers["Content-Security-Policy"] == _CSP

    # 4. OSError simulation
    with patch("importlib.resources.files") as mock_files:
        mock_file = mock_files.return_value.joinpath.return_value
        mock_file.is_file.side_effect = OSError("Read failed")
        res_oserr = _asset("index.html")
        assert res_oserr.status_code == 404

    # 5. Javascript charset (mimetypes might guess application/javascript or text/javascript)  # noqa: E501
    with patch("importlib.resources.files") as mock_files:
        mock_file = mock_files.return_value.joinpath.return_value
        mock_file.is_file.return_value = True
        mock_file.read_bytes.return_value = b"console.log('hi')"
        res_js = _asset("app.js")
        assert res_js.status_code == 200
        assert res_js.media_type is not None
        assert "javascript" in res_js.media_type
        assert "charset=utf-8" in res_js.media_type

    # 6. Binary file without text/ or javascript (branch 89->91 false)
    with patch("importlib.resources.files") as mock_files:
        mock_file = mock_files.return_value.joinpath.return_value
        mock_file.is_file.return_value = True
        mock_file.read_bytes.return_value = b"\x89PNG\r\n\x1a\n"
        res_png = _asset("icon.png")
        assert res_png.status_code == 200
        assert res_png.media_type == "image/png"


@pytest.mark.asyncio
async def test_is_strict_loopback() -> None:
    class DummyRequest:
        def __init__(
            self, client: tuple[str, int] | None, headers: dict[str, str]
        ) -> None:
            self.scope = {"client": client} if client else {}
            self.headers = headers

    # 1. No client
    req1: Any = DummyRequest(None, {"host": "localhost:8090"})
    assert not _is_strict_loopback(req1)

    # 2. Non-loopback client
    req2: Any = DummyRequest(("192.168.1.1", 12345), {"host": "localhost"})
    assert not _is_strict_loopback(req2)

    # 3. Non-loopback host
    req3: Any = DummyRequest(("127.0.0.1", 12345), {"host": "example.com"})
    assert not _is_strict_loopback(req3)

    # 4. Proxy header present
    req4: Any = DummyRequest(
        ("127.0.0.1", 12345), {"host": "localhost", "x-forwarded-for": "1.2.3.4"}
    )
    assert not _is_strict_loopback(req4)

    # 5. Strict loopback valid
    req5: Any = DummyRequest(("127.0.0.1", 12345), {"host": "localhost:8090"})
    req6: Any = DummyRequest(("::1", 12345), {"host": "[::1]:8090"})
    assert _is_strict_loopback(req5)
    assert _is_strict_loopback(req6)


@pytest.mark.asyncio
async def test_routes_static_and_root(config: MiniAppConfig) -> None:
    app = create_miniapp_app(config, bot_token="123456:ABC-DEF")
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(
        transport=transport, base_url="http://localhost:8090"
    ) as client:
        # Root redirect
        r_root = await client.get("/", follow_redirects=False)
        assert r_root.status_code == 307
        assert r_root.headers["location"] == "/miniapp/"

        # Miniapp index
        r_page1 = await client.get("/miniapp")
        assert r_page1.status_code == 200

        r_page2 = await client.get("/miniapp/")
        assert r_page2.status_code == 200

        # Static assets
        r_static = await client.get("/miniapp/static/index.html")
        assert r_static.status_code == 200


@pytest.mark.asyncio
async def test_session_dev_and_telegram(config: MiniAppConfig) -> None:
    app = create_miniapp_app(config, bot_token="123456:ABC-DEF")
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(
        transport=transport, base_url="http://localhost:8090"
    ) as client:
        # 1. Dev shortcut success
        r_dev = await client.post("/miniapp/api/session", json={"dev": True})
        assert r_dev.status_code == 200
        data = r_dev.json()
        assert data["dev"] is True
        token = data["token"]
        assert token

        # 2. Dev shortcut failure when disabled
        cfg_no_dev = MiniAppConfig(dev_telegram_user_id=None)
        app_no_dev = create_miniapp_app(cfg_no_dev, bot_token="123456:ABC-DEF")
        transport_no_dev = ASGITransport(app=app_no_dev, client=("127.0.0.1", 12345))
        async with AsyncClient(
            transport=transport_no_dev, base_url="http://localhost:8090"
        ) as client_no_dev:
            r_dev_off = await client_no_dev.post(
                "/miniapp/api/session", json={"dev": True}
            )
            assert r_dev_off.status_code == 403

        # 3. Telegram init_data without bot token configured
        app_no_bot = create_miniapp_app(config, bot_token=None)
        transport_no_bot = ASGITransport(app=app_no_bot, client=("127.0.0.1", 12345))
        async with AsyncClient(
            transport=transport_no_bot, base_url="http://localhost:8090"
        ) as client_no_bot:
            r_no_bot = await client_no_bot.post(
                "/miniapp/api/session", json={"dev": False, "init_data": "some_data"}
            )
            assert r_no_bot.status_code == 503

        # 4. Invalid init_data
        r_bad_init = await client.post(
            "/miniapp/api/session", json={"dev": False, "init_data": "bad_init_data"}
        )
        assert r_bad_init.status_code == 401

        # 5. Valid init_data with unauthorized hook
        valid_init = _make_init_data(999, "123456:ABC-DEF")
        hooks.authorize = _authorizer(False)
        r_unauth = await client.post(
            "/miniapp/api/session", json={"dev": False, "init_data": valid_init}
        )
        assert r_unauth.status_code == 403

        # 6. Valid init_data with authorized hook
        hooks.authorize = _authorizer(True)
        r_auth = await client.post(
            "/miniapp/api/session", json={"dev": False, "init_data": valid_init}
        )
        assert r_auth.status_code == 200
        assert r_auth.json()["dev"] is False


@pytest.mark.asyncio
async def test_auth_headers_and_revocation(
    config: MiniAppConfig, service: MiniAppService
) -> None:
    app = create_miniapp_app(config, bot_token="123456:ABC-DEF")
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(
        transport=transport, base_url="http://localhost:8090"
    ) as client:
        # Missing auth header
        r_no_auth = await client.get("/miniapp/api/nav")
        assert r_no_auth.status_code == 401

        # Invalid token
        r_bad_token = await client.get(
            "/miniapp/api/nav", headers={"Authorization": "Bearer badtoken"}
        )
        assert r_bad_token.status_code == 401

        # Issue non-dev session
        hooks.authorize = _authorizer(True)
        valid_init = _make_init_data(777, "123456:ABC-DEF")
        r_sess = await client.post(
            "/miniapp/api/session", json={"dev": False, "init_data": valid_init}
        )
        token = r_sess.json()["token"]

        # Call with valid token and authorized hook
        r_nav = await client.get(
            "/miniapp/api/nav", headers={"Authorization": f"Bearer {token}"}
        )
        assert r_nav.status_code == 200
        assert "pages" in r_nav.json()

        # Revoke access via hook
        hooks.authorize = _authorizer(False)
        r_revoked = await client.get(
            "/miniapp/api/nav", headers={"Authorization": f"Bearer {token}"}
        )
        assert r_revoked.status_code == 403

        # Set hooks.authorize to None -> also rejected
        hooks.authorize = None
        r_no_hook = await client.get(
            "/miniapp/api/nav", headers={"Authorization": f"Bearer {token}"}
        )
        assert r_no_hook.status_code == 403


@pytest.mark.asyncio
async def test_pages_and_cards_endpoints(
    config: MiniAppConfig, service: MiniAppService
) -> None:
    app = create_miniapp_app(config, bot_token="123456:ABC-DEF")
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(
        transport=transport, base_url="http://localhost:8090"
    ) as client:
        r_sess = await client.post("/miniapp/api/session", json={"dev": True})
        token = r_sess.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Page ID too long (>40)
        long_id = "a" * 41
        r_long = await client.get(f"/miniapp/api/page/{long_id}", headers=headers)
        assert r_long.status_code == 404

        # 2. Page not found
        r_missing = await client.get("/miniapp/api/page/notfound", headers=headers)
        assert r_missing.status_code == 404

        # 3. Builtin page load
        r_today = await client.get("/miniapp/api/page/today", headers=headers)
        assert r_today.status_code == 200
        assert r_today.json()["id"] == "today"

        # 4. Card ID too long (>40)
        r_card_long = await client.get(f"/miniapp/api/card/{long_id}", headers=headers)
        assert r_card_long.status_code == 404

        # 5. Card not found
        r_card_missing = await client.get(
            "/miniapp/api/card/c_nonexistent", headers=headers
        )
        assert r_card_missing.status_code == 404

        # 6. Create and load valid card
        user = MiniAppUser(12345)
        card_res = await service.tool_write(
            user, 'card "Sleep Card"\nroot = Page("Sleep Card", [])', None
        )
        card_id = card_res["results"][0]["card_id"]

        r_card_ok = await client.get(f"/miniapp/api/card/{card_id}", headers=headers)
        assert r_card_ok.status_code == 200
        assert r_card_ok.json()["title"] == "Sleep Card"


@pytest.mark.asyncio
async def test_action_endpoint(config: MiniAppConfig, service: MiniAppService) -> None:
    app = create_miniapp_app(config, bot_token="123456:ABC-DEF")
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(
        transport=transport, base_url="http://localhost:8090"
    ) as client:
        r_sess = await client.post("/miniapp/api/session", json={"dev": True})
        token = r_sess.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Action success
        r_theme = await client.post(
            "/miniapp/api/action",
            headers=headers,
            json={"action": "prefs.theme", "args": {"theme": {"mode": "dark"}}},
        )
        assert r_theme.status_code == 200
        assert r_theme.json()["status"] == "ok"

        # 2. ActionError (400)
        r_act_err = await client.post(
            "/miniapp/api/action",
            headers=headers,
            json={"action": "SaveTarget", "args": {"kind": "invalid", "value": 100}},
        )
        assert r_act_err.status_code == 400
        assert r_act_err.json()["status"] == "error"

        # 3. Unexpected Exception (500)
        with patch.object(
            service,
            "run_action",
            autospec=True,
            side_effect=RuntimeError("Crash"),
        ):
            r_crash = await client.post(
                "/miniapp/api/action",
                headers=headers,
                json={"action": "theme.set", "args": {"theme": "light"}},
            )
        assert r_crash.status_code == 500
        assert "reload the page" in r_crash.json()["message"]
