"""Contract tests for the private dashboard FastAPI routes."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blacki.dashboard.routes import create_dashboard_router
from blacki.observability.ledger import (
    UsageRecord,
    default_usage_ledger_path,
    read_usage_ledger,
    write_usage_record,
)
from blacki.utils.config import ServerEnv


class FakeDashboardStore:
    """Small async store boundary used by the route tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail_method: str | None = None

    async def _result(self, name: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((name, args))
        if name == self.fail_method:
            raise RuntimeError("backend path must not reach the client")
        if name in {"get_session", "get_trace"} and args[-1] == "missing":
            return None
        return {"method": name, "args": list(args)}

    async def get_overview(self, window: str) -> dict[str, Any] | None:
        return await self._result("get_overview", window)

    async def list_users(
        self, search: str, limit: int, offset: int
    ) -> dict[str, Any] | None:
        return await self._result("list_users", search, limit, offset)

    async def list_sessions(
        self, user_id: str | None, limit: int, offset: int
    ) -> dict[str, Any] | None:
        return await self._result("list_sessions", user_id, limit, offset)

    async def get_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        return await self._result("get_session", user_id, session_id)

    async def list_logs(
        self, level: str | None, search: str, limit: int
    ) -> dict[str, Any] | None:
        return await self._result("list_logs", level, search, limit)

    async def list_traces(
        self, status: str | None, search: str, limit: int
    ) -> dict[str, Any] | None:
        return await self._result("list_traces", status, search, limit)

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return await self._result("get_trace", trace_id)


def _env(tmp_path: Path) -> ServerEnv:
    return ServerEnv.model_validate(
        {"AGENT_NAME": "route-test", "AGENT_DIR": str(tmp_path)}
    )


def _client(
    tmp_path: Path,
    store: FakeDashboardStore | None = None,
    *,
    use_default_store: bool = False,
) -> tuple[TestClient, FakeDashboardStore]:
    selected_store = store or FakeDashboardStore()
    app = FastAPI()
    if use_default_store:
        app.include_router(create_dashboard_router(_env(tmp_path)))
    else:
        app.include_router(create_dashboard_router(_env(tmp_path), selected_store))
    return TestClient(app), selected_store


def _assert_dashboard_headers(response: Any) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert "style-src 'self'" in response.headers["content-security-policy"]
    assert "frame-src 'none'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"


def test_page_assets_and_unknown_dashboard_paths_are_local_and_protected(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path)

    page = client.get("/dashboard")
    assert page.status_code == 200
    assert "Blacki observability" in page.text
    _assert_dashboard_headers(page)

    script = client.get("/dashboard/static/dashboard.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    _assert_dashboard_headers(script)

    css = client.get("/dashboard/assets/dashboard.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    _assert_dashboard_headers(css)

    missing = client.get("/dashboard/static/missing.js")
    assert missing.status_code == 404
    assert "path" not in missing.text.lower()
    _assert_dashboard_headers(missing)

    unknown = client.get("/dashboard/api/not-a-route")
    assert unknown.status_code == 404
    _assert_dashboard_headers(unknown)


def test_dashboard_resource_and_data_serialization_failures_are_generic(
    tmp_path: Path,
) -> None:
    store = FakeDashboardStore()
    client, _ = _client(tmp_path, store)

    with patch(
        "blacki.dashboard.routes._asset_path",
        side_effect=RuntimeError("resource implementation detail"),
    ):
        resource_failure = client.get("/dashboard/static/dashboard.js")
    assert resource_failure.status_code == 503
    assert resource_failure.json() == {"error": "Dashboard resources are unavailable."}
    _assert_dashboard_headers(resource_failure)

    with patch(
        "blacki.dashboard.routes.jsonable_encoder",
        side_effect=[
            RuntimeError("serialization implementation detail"),
            {"error": "Dashboard data is temporarily unavailable."},
        ],
    ):
        serialization_failure = client.get("/dashboard/api/overview")
    assert serialization_failure.status_code == 503
    assert serialization_failure.json() == {
        "error": "Dashboard data is temporarily unavailable."
    }
    _assert_dashboard_headers(serialization_failure)


def test_api_routes_validate_and_forward_bounded_queries(tmp_path: Path) -> None:
    client, store = _client(tmp_path)

    requests = (
        ("/dashboard/api/overview?window=7d", "get_overview"),
        ("/dashboard/api/users?search=alice&limit=12&offset=3", "list_users"),
        ("/dashboard/api/sessions?user_id=alice&limit=12&offset=3", "list_sessions"),
        (
            "/dashboard/api/session?user_id=alice&session_id=session-v2",
            "get_session",
        ),
        ("/dashboard/api/logs?level=error&search=timeout&limit=12", "list_logs"),
        (
            "/dashboard/api/traces?status=error&search=llm&limit=12",
            "list_traces",
        ),
        ("/dashboard/api/trace?trace_id=trace-1", "get_trace"),
    )
    for url, method in requests:
        response = client.get(url)
        assert response.status_code == 200
        assert response.json()["method"] == method
        _assert_dashboard_headers(response)

    assert store.calls == [
        ("get_overview", ("7d",)),
        ("list_users", ("alice", 12, 3)),
        ("list_sessions", ("alice", 12, 3)),
        ("get_session", ("alice", "session-v2")),
        ("list_logs", ("error", "timeout", 12)),
        ("list_traces", ("error", "llm", 12)),
        ("get_trace", ("trace-1",)),
    ]

    empty_limit = client.get("/dashboard/api/users?limit=")
    assert empty_limit.status_code == 200
    assert empty_limit.json()["args"] == [None, 50, 0]

    for url in (
        "/dashboard/api/users?limit=not-an-integer",
        "/dashboard/api/users?limit=201",
        "/dashboard/api/users?offset=-1",
        "/dashboard/api/overview?window=not-a-window",
        "/dashboard/api/session?user_id=alice",
        "/dashboard/api/sessions?user_id=alice&limit=not-an-integer",
        "/dashboard/api/trace",
        "/dashboard/api/traces?limit=not-an-integer",
        "/dashboard/api/logs?search=" + ("x" * 129),
    ):
        response = client.get(url)
        assert response.status_code == 422
        assert response.json() == {"error": "Invalid dashboard query parameters."}
        _assert_dashboard_headers(response)


def test_api_routes_shield_store_errors_and_missing_records(tmp_path: Path) -> None:
    store = FakeDashboardStore()
    store.fail_method = "list_logs"
    client, _ = _client(tmp_path, store)

    failed = client.get("/dashboard/api/logs")
    assert failed.status_code == 503
    assert failed.json() == {"error": "Dashboard data is temporarily unavailable."}
    assert "backend path" not in failed.text
    _assert_dashboard_headers(failed)

    missing_session = client.get(
        "/dashboard/api/session?user_id=alice&session_id=missing"
    )
    assert missing_session.status_code == 404
    assert missing_session.json() == {"error": "Dashboard record not found."}
    _assert_dashboard_headers(missing_session)


def test_default_store_uses_session_db_log_dir_and_app_name(tmp_path: Path) -> None:
    fake_store = FakeDashboardStore()
    log_dir = tmp_path / "logs"
    with (
        patch(
            "blacki.dashboard.routes.DashboardStore", return_value=fake_store
        ) as ctor,
        patch("blacki.dashboard.routes.get_log_dir", return_value=log_dir),
    ):
        client, _ = _client(tmp_path, use_default_store=True)

    assert client.get("/dashboard/api/overview").status_code == 200
    ctor.assert_called_once_with(
        tmp_path / ".adk" / "sessions.db",
        log_dir,
        "blacki",
        tmp_path / ".adk" / "tools.db",
        tmp_path / ".adk" / "costs.db",
    )


@pytest.mark.parametrize("configured_override", [None, "override.db"])
def test_dashboard_reads_the_writer_ledger_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_override: str | None,
) -> None:
    """The dashboard and cost writer must resolve one configured ledger path."""
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv("AGENT_DIR", str(agent_dir))
    if configured_override is None:
        monkeypatch.delenv("BLACKI_COST_LEDGER_PATH", raising=False)
    else:
        monkeypatch.setenv(
            "BLACKI_COST_LEDGER_PATH", str(tmp_path / configured_override)
        )

    env = _env(agent_dir)
    writer_path = default_usage_ledger_path()
    dashboard_path = default_usage_ledger_path(env.agent_dir)
    write_usage_record(
        writer_path,
        UsageRecord(
            dedupe_key="dashboard-path",
            observed_at=100.0,
            user_id="user-1",
            session_id="session-1",
            invocation_id="invocation-1",
            model="test-model",
            provider_response_id="response-1",
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
            cost_usd=0.01,
            upstream_cost_usd=None,
            estimated_cost_usd=None,
            cost_kind="reported",
            cost_source="provider_usage",
        ),
    )

    snapshot = read_usage_ledger(
        dashboard_path,
        selected_since=0.0,
        selected_until=200.0,
        month_start=0.0,
        now=200.0,
    )

    assert writer_path == dashboard_path
    assert snapshot.available is True
    assert snapshot.cumulative.records == 1


def test_real_store_degrades_cleanly_when_local_records_are_missing(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    app = FastAPI()
    with patch("blacki.dashboard.routes.get_log_dir", return_value=log_dir):
        app.include_router(create_dashboard_router(_env(tmp_path)))

    response = TestClient(app).get("/dashboard/api/overview")
    assert response.status_code == 200
    assert response.json()["degraded"] is True
    assert "temporarily unavailable" in " ".join(response.json()["warnings"])
    assert str(tmp_path) not in response.text
    _assert_dashboard_headers(response)


def test_store_initialization_failure_keeps_dashboard_shell_available(
    tmp_path: Path,
) -> None:
    with patch(
        "blacki.dashboard.routes.DashboardStore",
        side_effect=OSError("private path"),
    ):
        client, _ = _client(tmp_path, use_default_store=True)

    assert client.get("/dashboard").status_code == 200
    unavailable = client.get("/dashboard/api/overview")
    assert unavailable.status_code == 503
    assert "private path" not in unavailable.text
    _assert_dashboard_headers(unavailable)


@pytest.mark.parametrize("path", ["/dashboard/", "/dashboard/static/"])
def test_dashboard_trailing_and_empty_asset_paths_are_safe(
    tmp_path: Path, path: str
) -> None:
    client, _ = _client(tmp_path)
    response = client.get(path)
    if path == "/dashboard/":
        assert response.status_code == 200
        assert "Blacki observability" in response.text
    else:
        assert response.status_code == 404
    _assert_dashboard_headers(response)
