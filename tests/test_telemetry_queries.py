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


class TestGetTokensPerSecond:
    """Tests for get_tokens_per_second method."""

    def test_empty_traces_file_returns_zeros(self, tmp_path: Path) -> None:
        """Test that empty traces file returns default zeros."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        traces_log.write_text("")

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert result["avg_tokens_per_sec"] == 0.0
        assert result["by_model"] == []
        assert result["by_request"] == []

    def test_missing_traces_file_returns_zeros(self, empty_log_dir: Path) -> None:
        """Test that missing traces file returns default zeros."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert result["avg_tokens_per_sec"] == 0.0
        assert result["by_model"] == []
        assert result["by_request"] == []

    def test_valid_traces_with_token_data(self, tmp_path: Path) -> None:
        """Test valid traces with token/duration data."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace1", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert result["avg_tokens_per_sec"] == 100.0
        assert len(result["by_model"]) == 1
        assert result["by_model"][0]["model"] == "gpt-4"
        assert result["by_model"][0]["avg_tokens_per_sec"] == 100.0
        assert result["by_model"][0]["total_tokens"] == 100
        assert len(result["by_request"]) == 1
        assert result["by_request"][0]["tokens_per_sec"] == 100.0

    def test_multiple_models_aggregation(self, tmp_path: Path) -> None:
        """Test aggregation across multiple models with different tokens/sec."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace1", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 200,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace2", "span_id": "span2"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1500000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace3", "span_id": "span3"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 3000000000,
                            "end_time": now_ns - 2000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "claude-3",
                                "gen_ai.usage.output_tokens": 500,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert result["avg_tokens_per_sec"] == 320.0
        assert len(result["by_model"]) == 2

        model_map = {m["model"]: m for m in result["by_model"]}
        assert model_map["gpt-4"]["total_tokens"] == 300
        assert model_map["gpt-4"]["request_count"] == 2
        assert model_map["claude-3"]["total_tokens"] == 500
        assert model_map["claude-3"]["request_count"] == 1

    def test_zero_duration_excluded(self, tmp_path: Path) -> None:
        """Test that zero duration traces are excluded from results."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace1", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert result["avg_tokens_per_sec"] == 0.0
        assert result["by_model"] == []
        assert result["by_request"] == []

    def test_zero_output_tokens_excluded(self, tmp_path: Path) -> None:
        """Test that zero output_tokens traces are excluded from results."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace1", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 0,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert result["avg_tokens_per_sec"] == 0.0
        assert result["by_model"] == []
        assert result["by_request"] == []

    def test_hours_parameter_filtering(self, tmp_path: Path) -> None:
        """Test that hours parameter filters traces correctly."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        old_trace_time = now_ns - (25 * 3600 * 1_000_000_000)
        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "recent_trace", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "old_trace", "span_id": "span2"},
                            "kind": "INTERNAL",
                            "start_time": old_trace_time,
                            "end_time": old_trace_time + 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 200,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=1)

        assert result["avg_tokens_per_sec"] == 100.0
        assert len(result["by_request"]) == 1
        assert result["by_request"][0]["trace_id"] == "recent_trace"

    def test_by_request_sorted_by_tokens_per_sec(self, tmp_path: Path) -> None:
        """Test that by_request is sorted by tokens_per_sec descending."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "slow_trace", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 10000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "fast_trace", "span_id": "span2"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 1500000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert len(result["by_request"]) == 2
        assert result["by_request"][0]["trace_id"] == "fast_trace"
        assert (
            result["by_request"][0]["tokens_per_sec"]
            > result["by_request"][1]["tokens_per_sec"]
        )

    def test_by_model_sorted_by_avg_tokens_per_sec(self, tmp_path: Path) -> None:
        """Test that by_model is sorted by avg_tokens_per_sec descending."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace1", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "slow-model",
                                "gen_ai.usage.output_tokens": 50,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace2", "span_id": "span2"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 1100000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.request.model": "fast-model",
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert len(result["by_model"]) == 2
        assert result["by_model"][0]["model"] == "fast-model"
        assert (
            result["by_model"][0]["avg_tokens_per_sec"]
            > result["by_model"][1]["avg_tokens_per_sec"]
        )

    def test_by_request_limited_to_50(self, tmp_path: Path) -> None:
        """Test that by_request is limited to 50 items."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces = []
        for i in range(75):
            traces.append(
                json.dumps(
                    {
                        "name": "call_llm",
                        "context": {"trace_id": f"trace_{i}", "span_id": f"span_{i}"},
                        "kind": "INTERNAL",
                        "start_time": now_ns - 2000000000,
                        "end_time": now_ns - 1000000000,
                        "status": {"status_code": "OK"},
                        "attributes": {
                            "gen_ai.request.model": "gpt-4",
                            "gen_ai.usage.output_tokens": 100,
                        },
                    }
                )
            )

        traces_log.write_text("\n".join(traces))

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert len(result["by_request"]) == 50

    def test_missing_model_uses_unknown(self, tmp_path: Path) -> None:
        """Test that missing model attribute is treated as 'unknown'."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "trace1", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert len(result["by_model"]) == 1
        assert result["by_model"][0]["model"] == "unknown"

    def test_non_call_llm_spans_excluded(self, tmp_path: Path) -> None:
        """Test that non-call_llm spans are excluded."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool",
                            "context": {"trace_id": "trace1", "span_id": "span1"},
                            "kind": "INTERNAL",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "gen_ai.usage.output_tokens": 100,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tokens_per_second(hours=24)

        assert result["avg_tokens_per_sec"] == 0.0
        assert result["by_model"] == []
        assert result["by_request"] == []


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


class TestCalculateCost:
    """Tests for _calculate_cost function."""

    def test_calculate_cost_exception_returns_zero(self) -> None:
        """Test that _calculate_cost returns 0.0 when litellm raises exception."""
        from blacki.telemetry.queries import _calculate_cost

        with patch("litellm.completion_cost") as mock_cost:
            mock_cost.side_effect = ValueError("Unknown model")
            result = _calculate_cost("unknown-model", 100, 50)
            assert result == 0.0

    def test_calculate_cost_returns_calculated_cost(self) -> None:
        """Test that _calculate_cost returns the calculated cost."""
        from blacki.telemetry.queries import _calculate_cost

        with patch("litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.00123
            result = _calculate_cost("gpt-4", 100, 50)
            assert result == 0.00123

    def test_calculate_cost_returns_zero_when_litellm_returns_none(self) -> None:
        """Test that _calculate_cost returns 0.0 when litellm returns None."""
        from blacki.telemetry.queries import _calculate_cost

        with patch("litellm.completion_cost") as mock_cost:
            mock_cost.return_value = None
            result = _calculate_cost("gpt-4", 100, 50)
            assert result == 0.0


class TestExecuteToDictEmptyResult:
    """Tests for _execute_to_dict returning empty dict."""

    def test_execute_to_dict_returns_empty_when_no_rows(
        self, empty_log_dir: Path
    ) -> None:
        """Test _execute_to_dict returns {} when query returns no rows."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        result = queries._execute_to_dict("SELECT 1 as x WHERE false", [])
        assert result == {}


class TestMissingFileReturns:
    """Tests for methods returning default values when files are missing."""

    def test_get_errors_over_time_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_errors_over_time returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries.get_errors_over_time(hours=24) == []

    def test_get_latency_by_operation_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_latency_by_operation returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries.get_latency_by_operation(hours=24) == []

    def test_get_token_metrics_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_token_metrics returns default dict when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        result = queries.get_token_metrics(hours=24)
        assert result["total_input"] == 0
        assert result["total_output"] == 0
        assert result["total"] == 0
        assert result["cache_hit_rate"] == 0

    def test_get_token_usage_by_user_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_token_usage_by_user returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries.get_token_usage_by_user(hours=24) == []

    def test_get_cost_metrics_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_cost_metrics returns default dict when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        result = queries.get_cost_metrics(hours=24)
        assert result["total_cost"] == 0.0
        assert result["by_model"] == []
        assert result["cached_savings"] == 0.0

    def test_get_tool_inspector_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_tool_inspector returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries.get_tool_inspector(limit=10) == []

    def test_get_tool_statistics_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_tool_statistics returns default dict when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        result = queries.get_tool_statistics(hours=24)
        assert result["total_calls"] == 0
        assert result["success_rate"] == 0
        assert result["failure_rate"] == 0
        assert result["by_tool"] == []

    def test_get_agent_metrics_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_agent_metrics returns default dict when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        result = queries.get_agent_metrics()
        assert result["name"] is None
        assert result["models"] == []
        assert result["total_invocations"] == 0
        assert result["total_llm_calls"] == 0

    def test_get_sre_metrics_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_sre_metrics returns default dict when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        result = queries.get_sre_metrics(hours=24)
        assert result["availability"] == 0
        assert result["throughput_per_min"] == 0
        assert result["active_users_24h"] == 0
        assert result["total_spans"] == 0

    def test_get_token_usage_over_time_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_token_usage_over_time returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries.get_token_usage_over_time(hours=24) == []

    def test_get_span_names_summary_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_span_names_summary returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries.get_span_names_summary() == []

    def test_get_recent_errors_simple_empty_dir(self, empty_log_dir: Path) -> None:
        """Test _get_recent_errors_simple returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries._get_recent_errors_simple(limit=5) == []

    def test_get_waterfall_traces_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_waterfall_traces returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries.get_waterfall_traces(limit=10) == []

    def test_get_token_accumulation_by_turn_empty_dir(
        self, empty_log_dir: Path
    ) -> None:
        """Test get_token_accumulation_by_turn returns [] when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        assert queries.get_token_accumulation_by_turn(hours=24) == []

    def test_get_cognitive_errors_empty_dir(self, empty_log_dir: Path) -> None:
        """Test get_cognitive_errors returns default dict when file missing."""
        queries = TelemetryQueries(log_dir=empty_log_dir)
        result = queries.get_cognitive_errors(hours=24)
        assert result["cognitive_errors"] == []
        assert result["infra_errors"] == []
        assert result["summary"]["total_cognitive"] == 0
        assert result["summary"]["total_infra"] == 0


class TestGetCostMetricsWithData:
    """Tests for get_cost_metrics with actual data."""

    def test_get_cost_metrics_with_model_data(self, tmp_path: Path) -> None:
        """Test get_cost_metrics calculates costs for models."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                                "gen_ai.usage.input_tokens": 100,
                                "gen_ai.usage.output_tokens": 50,
                                "llm.usage.cached_tokens": 20,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        with patch("blacki.telemetry.queries._calculate_cost") as mock_cost:
            mock_cost.return_value = 0.005
            result = queries.get_cost_metrics(hours=24)

            assert result["total_cost"] == 0.005
            assert len(result["by_model"]) == 1
            assert result["by_model"][0]["model"] == "gpt-4"
            mock_cost.assert_called()

    def test_get_cost_metrics_with_null_model(self, tmp_path: Path) -> None:
        """Test get_cost_metrics handles null model gracefully."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "attributes": {
                                "gen_ai.usage.input_tokens": 100,
                                "gen_ai.usage.output_tokens": 50,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_cost_metrics(hours=24)

        assert len(result["by_model"]) == 1
        assert result["by_model"][0]["model"] == "unknown"


class TestToolInspectorJsonErrors:
    """Tests for get_tool_inspector JSON parsing error handling."""

    def test_tool_inspector_invalid_json_input(self, tmp_path: Path) -> None:
        """Test get_tool_inspector handles invalid JSON in input_value."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool test_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "input.value": "{invalid json",
                                "output.value": '{"result": "ok"}',
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tool_inspector(limit=10)

        assert len(result) == 1
        assert result[0]["input"] == "{invalid json"
        assert result[0]["output"] == '{"result": "ok"}'

    def test_tool_inspector_invalid_json_output(self, tmp_path: Path) -> None:
        """Test get_tool_inspector handles invalid JSON in output_value."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool test_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "input.value": '{"query": "test"}',
                                "output.value": "not valid json at all",
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tool_inspector(limit=10)

        assert len(result) == 1
        assert result[0]["input"] == {"query": "test"}
        assert result[0]["output"] == "not valid json at all"

    def test_tool_inspector_null_name_uses_execute_tool_prefix(
        self, tmp_path: Path
    ) -> None:
        """Test get_tool_inspector handles name with execute_tool prefix."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {},
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tool_inspector(limit=10)

        assert len(result) == 1
        assert result[0]["name"] == "execute_tool"

    def test_tool_inspector_valid_json_both(self, tmp_path: Path) -> None:
        """Test get_tool_inspector parses valid JSON for both input and output."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool test_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {
                                "input.value": '{"query": "test"}',
                                "output.value": '{"result": "ok"}',
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tool_inspector(limit=10)

        assert len(result) == 1
        assert result[0]["input"] == {"query": "test"}
        assert result[0]["output"] == {"result": "ok"}


class TestToolStatisticsNullValues:
    """Tests for get_tool_statistics handling null values."""

    def test_tool_statistics_null_calls_and_successes(self, tmp_path: Path) -> None:
        """Test get_tool_statistics handles null calls/successes values."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool test",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {},
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_tool_statistics(hours=24)

        assert result["total_calls"] >= 0
        assert "by_tool" in result


class TestAgentMetricsWithData:
    """Tests for get_agent_metrics with actual data."""

    def test_agent_metrics_with_agent_name_and_model(self, tmp_path: Path) -> None:
        """Test get_agent_metrics extracts agent name and models."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "agent_run",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "attributes": {
                                "agent.name": "test_agent",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "t2", "span_id": "s2"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "attributes": {
                                "gen_ai.request.model": "gpt-4",
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_agent_metrics()

        assert result["name"] == "test_agent"
        assert "gpt-4" in result["models"]


class TestWaterfallTracesEdgeCases:
    """Tests for get_waterfall_traces edge cases."""

    def test_waterfall_traces_missing_trace_id(self, tmp_path: Path) -> None:
        """Test get_waterfall_traces skips spans with missing trace_id."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "test_span",
                            "context": {"trace_id": None, "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {},
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_waterfall_traces(limit=10)

        assert result == []

    def test_waterfall_traces_missing_span_id(self, tmp_path: Path) -> None:
        """Test get_waterfall_traces skips spans with missing span_id."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "test_span",
                            "context": {"trace_id": "t1", "span_id": None},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {},
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_waterfall_traces(limit=10)

        assert result == []

    def test_waterfall_traces_with_parent_child(self, tmp_path: Path) -> None:
        """Test get_waterfall_traces builds parent-child relationships."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "parent_span",
                            "context": {"trace_id": "t1", "span_id": "parent1"},
                            "parent_id": None,
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {},
                        }
                    ),
                    json.dumps(
                        {
                            "name": "child_span",
                            "context": {"trace_id": "t1", "span_id": "child1"},
                            "parent_id": "parent1",
                            "start_time": now_ns - 1500000000,
                            "end_time": now_ns - 500000000,
                            "status": {"status_code": "OK"},
                            "attributes": {},
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_waterfall_traces(limit=10)

        assert len(result) == 1
        assert result[0]["root"]["name"] == "parent_span"
        assert len(result[0]["root"]["children"]) == 1
        assert result[0]["root"]["children"][0]["name"] == "child_span"

    def test_waterfall_traces_single_span_no_parent(self, tmp_path: Path) -> None:
        """Test get_waterfall_traces handles span with no parent as root."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            json.dumps(
                {
                    "name": "orphan_span",
                    "context": {"trace_id": "t1", "span_id": "span1"},
                    "parent_id": None,
                    "start_time": now_ns - 1000000000,
                    "end_time": now_ns,
                    "status": {"status_code": "OK"},
                    "attributes": {},
                }
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_waterfall_traces(limit=10)

        assert len(result) == 1
        assert result[0]["root"]["name"] == "orphan_span"
        assert result[0]["root"]["children"] == []

    def test_waterfall_traces_trace_id_not_in_traces(self, tmp_path: Path) -> None:
        """Test span with no parent and missing trace_id field is handled."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            json.dumps(
                {
                    "name": "orphan_span",
                    "context": {"trace_id": None, "span_id": "span1"},
                    "parent_id": None,
                    "start_time": now_ns - 1000000000,
                    "end_time": now_ns,
                    "status": {"status_code": "OK"},
                    "attributes": {},
                }
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_waterfall_traces(limit=10)

        assert result == []

    def test_waterfall_traces_orphan_becomes_root(self, tmp_path: Path) -> None:
        """Test span with nonexistent parent_id becomes root."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            json.dumps(
                {
                    "name": "orphan_span",
                    "context": {"trace_id": "t1", "span_id": "span1"},
                    "parent_id": "nonexistent_parent",
                    "start_time": now_ns - 1000000000,
                    "end_time": now_ns,
                    "status": {"status_code": "OK"},
                    "attributes": {},
                }
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_waterfall_traces(limit=10)

        assert len(result) == 1
        assert result[0]["root"]["name"] == "orphan_span"

    def test_waterfall_traces_circular_parent_no_root(self, tmp_path: Path) -> None:
        """Test trace with circular parent refs has no root."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "span_a",
                            "context": {"trace_id": "t1", "span_id": "span_a"},
                            "parent_id": "span_b",
                            "start_time": now_ns - 2000000000,
                            "end_time": now_ns - 1000000000,
                            "status": {"status_code": "OK"},
                            "attributes": {},
                        }
                    ),
                    json.dumps(
                        {
                            "name": "span_b",
                            "context": {"trace_id": "t1", "span_id": "span_b"},
                            "parent_id": "span_a",
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "OK"},
                            "attributes": {},
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_waterfall_traces(limit=10)

        assert result == []


class TestTokenAccumulationByTurn:
    """Tests for get_token_accumulation_by_turn."""

    def test_token_accumulation_growth_rate(self, tmp_path: Path) -> None:
        """Test token accumulation calculates context growth rate."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 5000000000,
                            "end_time": now_ns - 4000000000,
                            "attributes": {
                                "gen_ai.conversation.id": "conv1",
                                "gen_ai.usage.input_tokens": 100,
                                "gen_ai.usage.output_tokens": 50,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "t1", "span_id": "s2"},
                            "start_time": now_ns - 3000000000,
                            "end_time": now_ns - 2000000000,
                            "attributes": {
                                "gen_ai.conversation.id": "conv1",
                                "gen_ai.usage.input_tokens": 200,
                                "gen_ai.usage.output_tokens": 50,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "name": "call_llm",
                            "context": {"trace_id": "t1", "span_id": "s3"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "attributes": {
                                "gen_ai.conversation.id": "conv1",
                                "gen_ai.usage.input_tokens": 300,
                                "gen_ai.usage.output_tokens": 50,
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_token_accumulation_by_turn(hours=24)

        assert len(result) == 1
        assert result[0]["conversation_id"] == "conv1"
        assert result[0]["turn_count"] == 3
        assert result[0]["total_input"] == 600
        assert len(result[0]["input_tokens_by_turn"]) == 3

    def test_token_accumulation_empty_turns(self, tmp_path: Path) -> None:
        """Test token accumulation with empty turns list."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            json.dumps(
                {
                    "name": "call_llm",
                    "context": {"trace_id": "t1", "span_id": "s1"},
                    "start_time": now_ns - 1000000000,
                    "end_time": now_ns,
                    "attributes": {
                        "gen_ai.conversation.id": None,
                    },
                }
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_token_accumulation_by_turn(hours=24)

        assert result == []

    def test_token_accumulation_single_turn_zero_growth(self, tmp_path: Path) -> None:
        """Test token accumulation with single turn has zero growth rate."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            json.dumps(
                {
                    "name": "call_llm",
                    "context": {"trace_id": "t1", "span_id": "s1"},
                    "start_time": now_ns - 1000000000,
                    "end_time": now_ns,
                    "attributes": {
                        "gen_ai.conversation.id": "conv1",
                        "gen_ai.usage.input_tokens": 100,
                        "gen_ai.usage.output_tokens": 50,
                    },
                }
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_token_accumulation_by_turn(hours=8760)

        assert len(result) == 1
        assert result[0]["context_growth_rate"] == 0.0


class TestCognitiveErrors:
    """Tests for get_cognitive_errors."""

    def test_cognitive_errors_with_tool_failures(self, tmp_path: Path) -> None:
        """Test get_cognitive_errors parses tool failure output."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool failing_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "ERROR"},
                            "attributes": {
                                "tool.name": "failing_tool",
                                "input.value": '{"arg": "value"}',
                                "output.value": (
                                    '{"response": {"message": "Tool failed"}}'
                                ),
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_cognitive_errors(hours=8760)

        assert len(result["cognitive_errors"]) == 1
        assert result["cognitive_errors"][0]["type"] == "tool_failure"
        assert result["cognitive_errors"][0]["tool_name"] == "failing_tool"
        assert result["cognitive_errors"][0]["error_message"] == "Tool failed"

    def test_cognitive_errors_invalid_output_json(self, tmp_path: Path) -> None:
        """Test get_cognitive_errors handles invalid JSON in output."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool failing_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "ERROR"},
                            "attributes": {
                                "tool.name": "failing_tool",
                                "output.value": "Raw error string not JSON",
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_cognitive_errors(hours=8760)

        assert len(result["cognitive_errors"]) == 1
        assert (
            result["cognitive_errors"][0]["error_message"]
            == "Raw error string not JSON"
        )

    def test_cognitive_errors_output_with_error_key(self, tmp_path: Path) -> None:
        """Test get_cognitive_errors extracts error from 'error' key."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool failing_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "ERROR"},
                            "attributes": {
                                "tool.name": "failing_tool",
                                "output.value": '{"error": "Something went wrong"}',
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_cognitive_errors(hours=8760)

        assert len(result["cognitive_errors"]) == 1
        assert result["cognitive_errors"][0]["error_message"] == "Something went wrong"

    def test_cognitive_errors_with_infra_errors(self, tmp_path: Path) -> None:
        """Test get_cognitive_errors captures infrastructure errors."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "agent_run",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "ERROR"},
                            "attributes": {
                                "error.type": "NetworkError",
                                "error.message": "Connection refused",
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_cognitive_errors(hours=8760)

        assert len(result["infra_errors"]) == 1
        assert result["infra_errors"][0]["type"] == "infrastructure"
        assert result["infra_errors"][0]["error_type"] == "NetworkError"
        assert result["infra_errors"][0]["error_message"] == "Connection refused"

    def test_cognitive_errors_output_dict_without_message_or_error(
        self, tmp_path: Path
    ) -> None:
        """Test get_cognitive_errors handles dict output without message/error keys."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool failing_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "ERROR"},
                            "attributes": {
                                "tool.name": "failing_tool",
                                "output.value": '{"some_other_key": "value"}',
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_cognitive_errors(hours=8760)

        assert len(result["cognitive_errors"]) == 1
        assert result["cognitive_errors"][0]["error_message"] is None

    def test_cognitive_errors_output_none(self, tmp_path: Path) -> None:
        """Test get_cognitive_errors handles missing output_value."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool failing_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "ERROR"},
                            "attributes": {
                                "tool.name": "failing_tool",
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_cognitive_errors(hours=8760)

        assert len(result["cognitive_errors"]) == 1
        assert result["cognitive_errors"][0]["error_message"] is None

    def test_cognitive_errors_output_is_list(self, tmp_path: Path) -> None:
        """Test get_cognitive_errors handles output that parses to a list."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        traces_log = log_dir / "blacki-traces.log"
        now_ns = int(time.time() * 1e9)

        traces_log.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "name": "execute_tool failing_tool",
                            "context": {"trace_id": "t1", "span_id": "s1"},
                            "start_time": now_ns - 1000000000,
                            "end_time": now_ns,
                            "status": {"status_code": "ERROR"},
                            "attributes": {
                                "tool.name": "failing_tool",
                                "output.value": '["item1", "item2"]',
                            },
                        }
                    ),
                ]
            )
        )

        queries = TelemetryQueries(log_dir=log_dir)
        result = queries.get_cognitive_errors(hours=8760)

        assert len(result["cognitive_errors"]) == 1
        assert result["cognitive_errors"][0]["error_message"] is None
