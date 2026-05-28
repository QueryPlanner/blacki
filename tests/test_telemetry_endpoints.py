"""Tests for telemetry FastAPI endpoints."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blacki.telemetry import queries as queries_module
from blacki.telemetry.queries import get_telemetry_queries


@pytest.fixture
def app() -> FastAPI:
    """Create a minimal FastAPI app with telemetry endpoints."""
    app = FastAPI()

    @app.get("/api/telemetry/stats")
    async def telemetry_stats(hours: int = 24) -> dict:
        queries = get_telemetry_queries()
        return queries.get_summary(hours=hours)

    @app.get("/dashboard")
    async def dashboard() -> Any:
        from fastapi.responses import HTMLResponse

        from blacki.server import DASHBOARD_HTML_PATH

        html_content = DASHBOARD_HTML_PATH.read_text()
        return HTMLResponse(content=html_content)

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Create a temporary log directory with sample log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    telemetry_log = log_dir / "blacki-telemetry.log"
    telemetry_log.write_text(
        '{"timestamp": "2026-05-25T10:00:00", "level": "INFO", '
        '"name": "test", "message": "test", "exception": null}\n'
    )

    traces_log = log_dir / "blacki-traces.log"
    traces_log.write_text(
        '{"name": "test_span", "context": {"trace_id": "abc", "span_id": "def"}, '
        '"start_time": 1748174400000000000, "end_time": 1748174401000000000, '
        '"status": {"status_code": "OK"}, "attributes": {}}\n'
    )

    return log_dir


class TestTelemetryStatsEndpoint:
    """Tests for /api/telemetry/stats endpoint."""

    def test_endpoint_returns_json(
        self, client: TestClient, temp_log_dir: Path
    ) -> None:
        """Test that endpoint returns valid JSON."""
        queries_module._telemetry_queries = None

        with patch("blacki.telemetry.queries.get_log_dir", return_value=temp_log_dir):
            response = client.get("/api/telemetry/stats")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

            data = response.json()
            assert "latency" in data
            assert "tokens" in data
            assert "errors" in data
            assert "tools" in data
            assert "agent" in data
            assert "sre" in data
            assert "log_levels" in data
            assert "span_names" in data
            assert "files" in data

        queries_module._telemetry_queries = None

    def test_endpoint_returns_empty_data_for_missing_files(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Test endpoint returns empty data when log files don't exist."""
        queries_module._telemetry_queries = None

        with patch("blacki.telemetry.queries.get_log_dir", return_value=tmp_path):
            response = client.get("/api/telemetry/stats")

            data = response.json()

            assert data["log_levels"] == []
            assert data["errors"]["recent"] == []
            assert data["latency"]["overall"]["count"] == 0

        queries_module._telemetry_queries = None

    def test_endpoint_accepts_hours_parameter(
        self, client: TestClient, temp_log_dir: Path
    ) -> None:
        """Test that endpoint accepts hours query parameter."""
        queries_module._telemetry_queries = None

        with patch("blacki.telemetry.queries.get_log_dir", return_value=temp_log_dir):
            response = client.get("/api/telemetry/stats?hours=12")

            assert response.status_code == 200
            data = response.json()
            assert data["time_range_hours"] == 12

        queries_module._telemetry_queries = None


class TestDashboardEndpoint:
    """Tests for /dashboard endpoint."""

    def test_endpoint_returns_html(self, client: TestClient) -> None:
        """Test that endpoint returns HTML content."""
        response = client.get("/dashboard")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_html_contains_chart_js(self, client: TestClient) -> None:
        """Test that HTML includes Chart.js CDN reference."""
        response = client.get("/dashboard")

        html_content = response.text
        assert "chart.js" in html_content.lower()
        assert "cdn.jsdelivr.net" in html_content

    def test_html_contains_api_endpoint(self, client: TestClient) -> None:
        """Test that HTML fetches from correct API endpoint."""
        response = client.get("/dashboard")

        html_content = response.text
        assert "/api/telemetry/stats" in html_content

    def test_html_contains_dashboard_title(self, client: TestClient) -> None:
        """Test that HTML has correct title."""
        response = client.get("/dashboard")

        html_content = response.text
        assert "Blacki Telemetry" in html_content

    def test_html_contains_time_filter(self, client: TestClient) -> None:
        """Test that HTML has time filter dropdown."""
        response = client.get("/dashboard")

        html_content = response.text
        assert 'id="timeRange"' in html_content

    def test_html_contains_charts(self, client: TestClient) -> None:
        """Test that HTML has chart canvases."""
        response = client.get("/dashboard")

        html_content = response.text
        assert 'id="latencyChart"' in html_content
        assert 'id="tokenChart"' in html_content
        assert 'id="contextBloatChart"' in html_content

    def test_html_contains_tool_inspector(self, client: TestClient) -> None:
        """Test that HTML has tool inspector table."""
        response = client.get("/dashboard")

        html_content = response.text
        assert 'id="toolInspectorTable"' in html_content
        assert 'id="contextBloatTable"' in html_content
