"""Contract tests for the isolated Telegram Mini App asset routes."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blacki.dashboard.routes import create_dashboard_router
from blacki.miniapp.routes import _asset, create_miniapp_router
from blacki.utils.config import ServerEnv


def _client(tmp_path: Path, *, demo: bool) -> TestClient:
    env = ServerEnv.model_validate(
        {
            "AGENT_NAME": "miniapp-route-test",
            "AGENT_DIR": str(tmp_path),
            "MINIAPP_DEMO_ENABLED": demo,
        }
    )
    app = FastAPI()
    app.include_router(create_miniapp_router(env))
    return TestClient(app)


def _assert_miniapp_headers(response: object) -> None:
    headers = response.headers  # type: ignore[attr-defined]
    assert headers["cache-control"] == "no-store"
    assert (
        "script-src 'self' https://telegram.org" in headers["content-security-policy"]
    )
    assert "connect-src 'self'" in headers["content-security-policy"]
    assert (
        "frame-ancestors 'self' https://web.telegram.org"
        in headers["content-security-policy"]
    )
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["cross-origin-resource-policy"] == "same-origin"


def test_demo_routes_are_disabled_by_default(tmp_path: Path) -> None:
    client = _client(tmp_path, demo=False)

    page = client.get("/miniapp")
    asset = client.get("/miniapp/static/app.js")

    assert page.status_code == 404
    assert page.json() == {"error": "Mini App demo is disabled."}
    assert asset.status_code == 404
    _assert_miniapp_headers(page)
    _assert_miniapp_headers(asset)


def test_enabled_demo_serves_only_local_app_assets(tmp_path: Path) -> None:
    client = _client(tmp_path, demo=True)

    page = client.get("/miniapp")
    app_script = client.get("/miniapp/static/app.js")
    fixture_script = client.get("/miniapp/static/fixtures.js")
    stylesheet = client.get("/miniapp/static/miniapp.css")
    missing = client.get("/miniapp/static/missing.js")
    unknown_api = client.get("/api/miniapp/demo/state")

    assert page.status_code == 200
    assert "Blacki, your little wins club" in page.text
    assert "miniapp-mode" in page.text
    assert "style=" not in page.text
    assert app_script.status_code == 200
    assert "new FixtureAdapter" in app_script.text
    assert "fetch(" not in app_script.text
    assert "XMLHttpRequest" not in app_script.text
    assert "WebSocket" not in app_script.text
    assert fixture_script.status_code == 200
    assert "export class FixtureAdapter" in fixture_script.text
    assert stylesheet.status_code == 200
    assert "blacki-playroom" in stylesheet.text
    assert missing.status_code == 404
    assert unknown_api.status_code == 404
    for response in (page, app_script, fixture_script, stylesheet, missing):
        _assert_miniapp_headers(response)


def test_demo_uses_the_existing_daisyui_dashboard_stylesheet(tmp_path: Path) -> None:
    env = ServerEnv.model_validate(
        {
            "AGENT_NAME": "miniapp-style-test",
            "AGENT_DIR": str(tmp_path),
            "MINIAPP_DEMO_ENABLED": True,
        }
    )
    app = FastAPI()
    app.include_router(create_dashboard_router(env, store=cast(Any, object())))
    app.include_router(create_miniapp_router(env))
    client = TestClient(app)

    page = client.get("/miniapp")
    daisyui = client.get("/dashboard/assets/dashboard.css")

    assert page.status_code == daisyui.status_code == 200
    assert "/dashboard/assets/dashboard.css" in page.text
    assert ".btn{" in daisyui.text
    assert ".range{" in daisyui.text


@pytest.mark.parametrize("asset_name", ["", "../secrets", "/etc/passwd"])
def test_asset_path_rejects_empty_parent_and_absolute_names(asset_name: str) -> None:
    response = _asset(asset_name)

    assert response.status_code == 404
    assert response.body == b'{"error":"Mini App resource not found."}'
    _assert_miniapp_headers(response)


def test_asset_read_errors_are_generic() -> None:
    with patch(
        "blacki.miniapp.routes.resources.files", side_effect=OSError("private path")
    ):
        missing = _asset("app.js")
    with patch(
        "blacki.miniapp.routes.resources.files",
        side_effect=RuntimeError("private implementation detail"),
    ):
        unavailable = _asset("app.js")

    assert missing.status_code == 404
    assert json.loads(missing.body) == {"error": "Mini App resource not found."}
    assert unavailable.status_code == 503
    assert json.loads(unavailable.body) == {
        "error": "Mini App resources are unavailable."
    }
    _assert_miniapp_headers(missing)
    _assert_miniapp_headers(unavailable)
