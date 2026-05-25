"""Tests for telemetry queries module."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from blacki.telemetry import TelemetryQueries, get_telemetry_queries


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Create a temporary log directory with sample log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    telemetry_log = log_dir / "blacki-telemetry.log"
    telemetry_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-25T10:00:00",
                        "level": "INFO",
                        "name": "test.module",
                        "message": "Info message 1",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-25T10:01:00",
                        "level": "ERROR",
                        "name": "test.module",
                        "message": "Error message 1",
                        "exception": "ValueError: test error",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-25T10:02:00",
                        "level": "INFO",
                        "name": "test.other",
                        "message": "Info message 2",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-25T10:03:00",
                        "level": "WARNING",
                        "name": "test.module",
                        "message": "Warning message",
                    }
                ),
            ]
        )
    )

    traces_log = log_dir / "blacki-traces.log"
    now_ns = int(time.time() * 1e9)
    traces_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "name": "test_span_1",
                        "context": {"trace_id": "abc123", "span_id": "def456"},
                        "kind": "INTERNAL",
                        "start_time": now_ns - 3000000000,
                        "end_time": now_ns - 2000000000,
                        "status": {"status_code": "OK"},
                        "attributes": {"llm.token_count": 100},
                    }
                ),
                json.dumps(
                    {
                        "name": "test_span_2",
                        "context": {"trace_id": "ghi789", "span_id": "jkl012"},
                        "kind": "INTERNAL",
                        "start_time": now_ns - 5000000000,
                        "end_time": now_ns - 2000000000,
                        "status": {"status_code": "OK"},
                        "attributes": {"llm.token_count": 200},
                    }
                ),
                json.dumps(
                    {
                        "name": "test_span_1",
                        "context": {"trace_id": "mno345", "span_id": "pqr678"},
                        "kind": "INTERNAL",
                        "start_time": now_ns - 1000000000,
                        "end_time": now_ns,
                        "status": {"status_code": "OK"},
                        "attributes": {},
                    }
                ),
            ]
        )
    )

    return log_dir


@pytest.fixture
def queries(temp_log_dir: Path) -> TelemetryQueries:
    """Create TelemetryQueries instance with temp log directory."""
    return TelemetryQueries(log_dir=temp_log_dir)


@pytest.fixture
def empty_log_dir(tmp_path: Path) -> Path:
    """Create an empty log directory."""
    log_dir = tmp_path / "empty_logs"
    log_dir.mkdir()
    return log_dir


class TestTelemetryQueries:
    """Tests for TelemetryQueries class."""

    def test_init_sets_paths(self, temp_log_dir: Path) -> None:
        """Test that init sets correct log paths."""
        queries = TelemetryQueries(log_dir=temp_log_dir)

        assert queries.telemetry_path == temp_log_dir / "blacki-telemetry.log"
        assert queries.traces_path == temp_log_dir / "blacki-traces.log"

    def test_get_log_level_counts_returns_correct_data(
        self, queries: TelemetryQueries
    ) -> None:
        """Test log level count aggregation."""
        result = queries.get_log_level_counts()

        levels = {item["level"]: item["count"] for item in result}

        assert levels.get("INFO") == 2
        assert levels.get("ERROR") == 1
        assert levels.get("WARNING") == 1

    def test_get_log_level_counts_empty_dir(self, empty_log_dir: Path) -> None:
        """Test log level counts with missing file."""
        queries = TelemetryQueries(log_dir=empty_log_dir)

        result = queries.get_log_level_counts()

        assert result == []

    def test_get_errors_over_time(self, queries: TelemetryQueries) -> None:
        """Test errors over time retrieval."""
        result = queries.get_errors_over_time(hours=24)

        assert isinstance(result, list)

    def test_get_span_latency_stats_returns_correct_data(
        self, queries: TelemetryQueries
    ) -> None:
        """Test span latency statistics calculation."""
        result = queries.get_span_latency_stats(hours=8760)

        assert result["count"] == 3
        assert result["min_ms"] is not None
        assert result["max_ms"] is not None
        assert result["avg_ms"] is not None
        assert result["min_ms"] == 1000.0
        assert result["max_ms"] == 3000.0

    def test_get_span_latency_stats_empty_dir(self, empty_log_dir: Path) -> None:
        """Test span latency with missing file."""
        queries = TelemetryQueries(log_dir=empty_log_dir)

        result = queries.get_span_latency_stats(hours=24)

        assert result["count"] == 0
        assert result["min_ms"] is None
        assert result["max_ms"] is None

    def test_get_span_names_summary_returns_correct_data(
        self, queries: TelemetryQueries
    ) -> None:
        """Test span name frequency summary."""
        result = queries.get_span_names_summary()

        names = {item["name"]: item["count"] for item in result}

        assert names.get("test_span_1") == 2
        assert names.get("test_span_2") == 1

    def test_get_summary_returns_all_data(self, queries: TelemetryQueries) -> None:
        """Test that get_summary returns all expected data."""
        result = queries.get_summary(hours=24)

        assert "latency" in result
        assert "overall" in result["latency"]
        assert "by_operation" in result["latency"]
        assert "tokens" in result
        assert "errors" in result
        assert "tools" in result
        assert "agent" in result
        assert "sre" in result
        assert "log_levels" in result
        assert "span_names" in result
        assert "files" in result

        assert result["files"]["telemetry_log"]["exists"] is True
        assert result["files"]["traces_log"]["exists"] is True

    def test_get_summary_with_missing_files(self, empty_log_dir: Path) -> None:
        """Test summary with missing log files."""
        queries = TelemetryQueries(log_dir=empty_log_dir)

        result = queries.get_summary(hours=24)

        assert result["log_levels"] == []
        assert result["errors"]["recent"] == []
        assert result["latency"]["overall"]["count"] == 0
        assert result["files"]["telemetry_log"]["exists"] is False
        assert result["files"]["traces_log"]["exists"] is False

    def test_get_token_metrics(self, queries: TelemetryQueries) -> None:
        """Test token metrics retrieval."""
        result = queries.get_token_metrics(hours=24)

        assert "total_input" in result
        assert "total_output" in result
        assert "total" in result
        assert isinstance(result["total"], int)

    def test_get_tool_inspector(self, queries: TelemetryQueries) -> None:
        """Test tool inspector retrieval."""
        result = queries.get_tool_inspector(limit=10)

        assert isinstance(result, list)

    def test_get_tool_statistics(self, queries: TelemetryQueries) -> None:
        """Test tool statistics retrieval."""
        result = queries.get_tool_statistics(hours=24)

        assert "total_calls" in result
        assert "success_rate" in result
        assert "failure_rate" in result
        assert "by_tool" in result

    def test_get_agent_metrics(self, queries: TelemetryQueries) -> None:
        """Test agent metrics retrieval."""
        result = queries.get_agent_metrics()

        assert "name" in result
        assert "models" in result
        assert "total_invocations" in result
        assert "total_llm_calls" in result

    def test_get_sre_metrics(self, queries: TelemetryQueries) -> None:
        """Test SRE metrics retrieval."""
        result = queries.get_sre_metrics(hours=24)

        assert "availability" in result
        assert "throughput_per_min" in result
        assert "active_users_24h" in result
        assert "total_spans" in result


class TestGetTelemetryQueries:
    """Tests for singleton get_telemetry_queries function."""

    def test_returns_singleton_instance(self) -> None:
        """Test that get_telemetry_queries returns same instance."""
        import blacki.telemetry.queries as queries_module

        queries_module._telemetry_queries = None

        with patch.object(TelemetryQueries, "__init__", return_value=None) as mock_init:
            instance1 = get_telemetry_queries()
            instance2 = get_telemetry_queries()

            assert mock_init.call_count == 1
            assert instance1 is instance2

        queries_module._telemetry_queries = None
