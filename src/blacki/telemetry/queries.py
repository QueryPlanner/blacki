"""DuckDB-based telemetry query module.

Provides a query interface for local JSON log files using DuckDB.
Designed for both FastAPI endpoints and potential ADK tool integration.
"""

from pathlib import Path
from typing import Any

import duckdb

from blacki.utils.observability import get_log_dir


class TelemetryQueries:
    """Query interface for telemetry and trace logs using DuckDB."""

    def __init__(self, log_dir: Path | None = None):
        self._log_dir = log_dir or get_log_dir()
        self._telemetry_path = self._log_dir / "blacki-telemetry.log"
        self._traces_path = self._log_dir / "blacki-traces.log"
        self._conn = duckdb.connect(":memory:")

    @property
    def telemetry_path(self) -> Path:
        return self._telemetry_path

    @property
    def traces_path(self) -> Path:
        return self._traces_path

    def _file_exists(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    def _execute_to_dicts(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        result = self._conn.execute(query, params)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    def _execute_to_dict(self, query: str, params: list[Any]) -> dict[str, Any]:
        result = self._conn.execute(query, params)
        row = result.fetchone()
        if row is None:
            return {}
        columns = [desc[0] for desc in result.description]
        return dict(zip(columns, row, strict=True))

    DURATION_EXPR = """
        CASE
            WHEN json_type(start_time) IN ('INTEGER', 'UBIGINT') THEN
                (end_time::DOUBLE - start_time::DOUBLE) / 1e6
            WHEN json_type(start_time) = 'VARCHAR' THEN
                EXTRACT(EPOCH FROM end_time::TIMESTAMP) * 1000
                - EXTRACT(EPOCH FROM start_time::TIMESTAMP) * 1000
            ELSE NULL
        END
    """

    EPOCH_EXPR = """
        CASE
            WHEN json_type(start_time) IN ('INTEGER', 'UBIGINT') THEN
                start_time::DOUBLE / 1e9
            WHEN json_type(start_time) = 'VARCHAR' THEN
                EXTRACT(EPOCH FROM start_time::TIMESTAMP)
            ELSE NULL
        END
    """

    def get_log_level_counts(self) -> list[dict[str, Any]]:
        if not self._file_exists(self._telemetry_path):
            return []
        query = """
            SELECT level, COUNT(*) as count
            FROM read_json_auto(?, ignore_errors=true)
            GROUP BY level
            ORDER BY count DESC
        """
        return self._execute_to_dicts(query, [str(self._telemetry_path)])

    def get_errors_over_time(self, hours: int = 24) -> list[dict[str, Any]]:
        if not self._file_exists(self._telemetry_path):
            return []
        query = """
            WITH logs AS (
                SELECT TRY_CAST(timestamp AS TIMESTAMP) as ts, level
                FROM read_json_auto(?, ignore_errors=true)
                WHERE level = 'ERROR' AND timestamp IS NOT NULL
            )
            SELECT DATE_TRUNC('hour', ts) as hour, COUNT(*) as count
            FROM logs
            WHERE ts IS NOT NULL AND ts > now() - INTERVAL '1 hour' * ?
            GROUP BY hour ORDER BY hour
        """
        results = self._execute_to_dicts(query, [str(self._telemetry_path), hours])
        return [
            {"hour": r["hour"].isoformat() if r["hour"] else None, "count": r["count"]}
            for r in results
        ]

    def get_latency_by_operation(self, hours: int = 24) -> list[dict[str, Any]]:
        if not self._file_exists(self._traces_path):
            return []
        query = f"""
            WITH spans AS (
                SELECT name,
                    {self.DURATION_EXPR} as duration_ms,
                    {self.EPOCH_EXPR} as start_epoch
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON'}})
                WHERE end_time IS NOT NULL AND start_time IS NOT NULL
            )
            SELECT
                CASE
                    WHEN name LIKE 'agent_run%' THEN 'agent_run'
                    WHEN name LIKE 'invocation%' THEN 'invocation'
                    WHEN name = 'call_llm' THEN 'call_llm'
                    WHEN name LIKE 'generate_content%' THEN 'generate_content'
                    WHEN name LIKE 'execute_tool%' THEN 'execute_tool'
                    ELSE 'other'
                END as operation,
                COUNT(*) as count, AVG(duration_ms) as avg_ms,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) as p50_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_ms,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_ms
            FROM spans
            WHERE duration_ms IS NOT NULL AND start_epoch IS NOT NULL
              AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
            GROUP BY operation ORDER BY count DESC
        """
        return self._execute_to_dicts(query, [str(self._traces_path), hours])

    def get_token_metrics(self, hours: int = 24) -> dict[str, Any]:
        if not self._file_exists(self._traces_path):
            return {
                "total_input": 0,
                "total_output": 0,
                "total": 0,
                "input_per_response_avg": 0,
                "output_per_response_avg": 0,
                "burn_rate_per_hour": {"input": 0, "output": 0},
            }
        query = f"""
            WITH traces AS (
                SELECT {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."gen_ai.usage.input_tokens"')::BIGINT as input_tokens,
                    json_extract_string(attributes, '$."gen_ai.usage.output_tokens"')::BIGINT as output_tokens
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'attributes': 'JSON'}})
                WHERE start_time IS NOT NULL AND name LIKE 'generate_content%'
            )
            SELECT COALESCE(SUM(input_tokens), 0) as total_input, COALESCE(SUM(output_tokens), 0) as total_output, COUNT(*) as response_count
            FROM traces WHERE start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
        """
        result = self._execute_to_dict(query, [str(self._traces_path), hours])
        total_input = result.get("total_input") or 0
        total_output = result.get("total_output") or 0
        response_count = max(result.get("response_count") or 0, 1)
        return {
            "total_input": total_input,
            "total_output": total_output,
            "total": total_input + total_output,
            "input_per_response_avg": round(total_input / response_count, 1),
            "output_per_response_avg": round(total_output / response_count, 1),
            "burn_rate_per_hour": {"input": total_input, "output": total_output},
        }

    def get_token_usage_by_user(self, hours: int = 24) -> list[dict[str, Any]]:
        if not self._file_exists(self._traces_path):
            return []
        query = f"""
            WITH traces AS (
                SELECT {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."user.id"') as user_id,
                    COALESCE(json_extract_string(attributes, '$."gen_ai.usage.input_tokens"')::BIGINT, 0) as input_tokens,
                    COALESCE(json_extract_string(attributes, '$."gen_ai.usage.output_tokens"')::BIGINT, 0) as output_tokens
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'attributes': 'JSON'}})
                WHERE start_time IS NOT NULL AND name LIKE 'generate_content%'
            )
            SELECT user_id, SUM(input_tokens) as input_tokens, SUM(output_tokens) as output_tokens,
                SUM(input_tokens + output_tokens) as total_tokens, COUNT(*) as request_count
            FROM traces WHERE start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?) AND user_id IS NOT NULL
            GROUP BY user_id ORDER BY total_tokens DESC LIMIT 20
        """
        return self._execute_to_dicts(query, [str(self._traces_path), hours])

    def get_tool_inspector(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self._file_exists(self._traces_path):
            return []
        query = f"""
            SELECT name, {self.DURATION_EXPR} as duration_ms,
                json_extract_string(attributes, '$."gen_ai.operation.name"') as operation,
                CASE WHEN json_type(status) = 'VARCHAR' THEN json_extract_string(status, '$.status_code') ELSE NULL END as status_code,
                {self.EPOCH_EXPR} as start_epoch
            FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'status': 'JSON', 'attributes': 'JSON'}})
            WHERE name LIKE 'execute_tool%' ORDER BY start_epoch DESC LIMIT ?
        """
        results = self._execute_to_dicts(query, [str(self._traces_path), limit])
        formatted = []
        for r in results:
            tool_name = r["name"].replace("execute_tool ", "") if r["name"] else None
            formatted.append(
                {
                    "name": tool_name,
                    "duration_ms": round(r["duration_ms"], 2)
                    if r["duration_ms"]
                    else None,
                    "status": r["status_code"] or "UNKNOWN",
                    "timestamp": r.get("start_epoch"),
                }
            )
        return formatted

    def get_tool_statistics(self, hours: int = 24) -> dict[str, Any]:
        if not self._file_exists(self._traces_path):
            return {
                "total_calls": 0,
                "success_rate": 0,
                "failure_rate": 0,
                "by_tool": [],
            }
        query = f"""
            WITH tools AS (
                SELECT name, {self.DURATION_EXPR} as duration_ms, {self.EPOCH_EXPR} as start_epoch,
                    CASE WHEN json_type(status) = 'VARCHAR' THEN json_extract_string(status, '$.status_code') ELSE NULL END as status_code
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'status': 'JSON', 'attributes': 'JSON'}})
                WHERE name LIKE 'execute_tool%'
            )
            SELECT REPLACE(name, 'execute_tool ', '') as tool_name, COUNT(*) as calls,
                SUM(CASE WHEN status_code = 'OK' THEN 1 ELSE 0 END) as successes,
                SUM(CASE WHEN status_code = 'ERROR' THEN 1 ELSE 0 END) as failures,
                AVG(duration_ms) as avg_ms, PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_ms
            FROM tools WHERE duration_ms IS NOT NULL AND start_epoch IS NOT NULL
              AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
            GROUP BY name ORDER BY calls DESC
        """
        by_tool = self._execute_to_dicts(query, [str(self._traces_path), hours])
        total_calls = sum(t.get("calls", 0) for t in by_tool)
        total_successes = sum(t.get("successes", 0) for t in by_tool)
        total_failures = sum(t.get("failures", 0) for t in by_tool)
        success_rate = round(total_successes / total_calls, 4) if total_calls > 0 else 0
        failure_rate = round(total_failures / total_calls, 4) if total_calls > 0 else 0
        formatted_tools = []
        for t in by_tool:
            calls = t.get("calls", 0) or 0
            successes = t.get("successes", 0) or 0
            formatted_tools.append(
                {
                    "name": t["tool_name"],
                    "calls": calls,
                    "avg_ms": round(t["avg_ms"], 2) if t["avg_ms"] else None,
                    "p95_ms": round(t["p95_ms"], 2) if t["p95_ms"] else None,
                    "success_rate": round(successes / calls, 4) if calls > 0 else 0,
                }
            )
        return {
            "total_calls": total_calls,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "by_tool": formatted_tools,
        }

    def get_agent_metrics(self) -> dict[str, Any]:
        if not self._file_exists(self._traces_path):
            return {
                "name": None,
                "models": [],
                "total_invocations": 0,
                "total_llm_calls": 0,
            }
        query = """
            SELECT json_extract_string(attributes, '$."agent.name"') as agent_name,
                json_extract_string(attributes, '$."gen_ai.request.model"') as model, COUNT(*) as count
            FROM read_json_auto(?, ignore_errors=true, columns={'name': 'VARCHAR', 'attributes': 'JSON'})
            WHERE name = 'agent_run' OR name = 'call_llm' OR name LIKE 'generate_content%'
            GROUP BY agent_name, model
        """
        results = self._execute_to_dicts(query, [str(self._traces_path)])
        agent_name = None
        models = set()
        for r in results:
            if r.get("agent_name"):
                agent_name = r["agent_name"]
            if r.get("model"):
                models.add(r["model"])
        inv_query = "SELECT COUNT(*) as count FROM read_json_auto(?, ignore_errors=true) WHERE name LIKE 'invocation%'"
        inv_result = self._execute_to_dict(inv_query, [str(self._traces_path)])
        total_invocations = inv_result.get("count", 0) or 0
        llm_query = "SELECT COUNT(*) as count FROM read_json_auto(?, ignore_errors=true) WHERE name = 'call_llm'"
        llm_result = self._execute_to_dict(llm_query, [str(self._traces_path)])
        total_llm_calls = llm_result.get("count", 0) or 0
        return {
            "name": agent_name,
            "models": sorted(models),
            "total_invocations": total_invocations,
            "total_llm_calls": total_llm_calls,
        }

    def get_sre_metrics(self, hours: int = 24) -> dict[str, Any]:
        if not self._file_exists(self._traces_path):
            return {
                "availability": 0,
                "throughput_per_min": 0,
                "active_users_24h": 0,
                "total_spans": 0,
            }
        avail_query = """
            WITH spans AS (
                SELECT CASE WHEN json_type(status) = 'VARCHAR' THEN json_extract_string(status, '$.status_code') ELSE NULL END as status_code
                FROM read_json_auto(?, ignore_errors=true, columns={'name': 'VARCHAR', 'start_time': 'JSON', 'status': 'JSON'}) WHERE start_time IS NOT NULL
            )
            SELECT COUNT(*) as total, SUM(CASE WHEN status_code = 'ERROR' THEN 1 ELSE 0 END) as errors FROM spans
        """
        avail_result = self._execute_to_dict(avail_query, [str(self._traces_path)])
        total = avail_result.get("total", 0) or 0
        errors = avail_result.get("errors", 0) or 0
        availability = round((total - errors) / total, 4) if total > 0 else 0
        throughput_query = f"""
            WITH spans AS (SELECT {self.EPOCH_EXPR} as start_epoch FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON'}}) WHERE start_time IS NOT NULL)
            SELECT COUNT(*) as count FROM spans WHERE start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
        """
        throughput_result = self._execute_to_dict(
            throughput_query, [str(self._traces_path), hours]
        )
        span_count = throughput_result.get("count", 0) or 0
        throughput_per_min = round(span_count / (hours * 60), 2) if hours > 0 else 0
        users_query = f"""
            WITH traces AS (
                SELECT {self.EPOCH_EXPR} as start_epoch, json_extract_string(attributes, '$."user.id"') as user_id
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'attributes': 'JSON'}}) WHERE start_time IS NOT NULL
            )
            SELECT COUNT(DISTINCT user_id) as count FROM traces WHERE start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?) AND user_id IS NOT NULL
        """
        users_result = self._execute_to_dict(
            users_query, [str(self._traces_path), hours]
        )
        active_users = users_result.get("count", 0) or 0
        return {
            "availability": availability,
            "throughput_per_min": throughput_per_min,
            "active_users_24h": active_users,
            "total_spans": total,
        }

    def get_span_latency_stats(self, hours: int = 24) -> dict[str, Any]:
        if not self._file_exists(self._traces_path):
            return {
                "count": 0,
                "min_ms": None,
                "max_ms": None,
                "avg_ms": None,
                "p95_ms": None,
            }
        query = f"""
            WITH spans AS (
                SELECT {self.DURATION_EXPR} as duration_ms, {self.EPOCH_EXPR} as start_epoch
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON'}})
                WHERE end_time IS NOT NULL AND start_time IS NOT NULL
            )
            SELECT COUNT(*) as count, MIN(duration_ms) as min_ms, MAX(duration_ms) as max_ms, AVG(duration_ms) as avg_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_ms
            FROM spans WHERE duration_ms IS NOT NULL AND start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
        """
        result = self._execute_to_dict(query, [str(self._traces_path), hours])
        return (
            result
            if result
            else {
                "count": 0,
                "min_ms": None,
                "max_ms": None,
                "avg_ms": None,
                "p95_ms": None,
            }
        )

    def get_token_usage_over_time(self, hours: int = 24) -> list[dict[str, Any]]:
        if not self._file_exists(self._traces_path):
            return []
        query = f"""
            WITH traces AS (
                SELECT {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."gen_ai.usage.input_tokens"')::BIGINT as input_tokens,
                    json_extract_string(attributes, '$."gen_ai.usage.output_tokens"')::BIGINT as output_tokens
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'attributes': 'JSON'}})
                WHERE start_time IS NOT NULL AND name LIKE 'generate_content%'
            )
            SELECT DATE_TRUNC('hour', to_timestamp(start_epoch)) as hour, SUM(input_tokens) as input_tokens, SUM(output_tokens) as output_tokens
            FROM traces WHERE start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
            GROUP BY hour ORDER BY hour
        """
        results = self._execute_to_dicts(query, [str(self._traces_path), hours])
        return [
            {
                "hour": r["hour"].isoformat() if r["hour"] else None,
                "input_tokens": r["input_tokens"] or 0,
                "output_tokens": r["output_tokens"] or 0,
            }
            for r in results
        ]

    def get_span_names_summary(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self._file_exists(self._traces_path):
            return []
        query = "SELECT name, COUNT(*) as count FROM read_json_auto(?, ignore_errors=true) GROUP BY name ORDER BY count DESC LIMIT ?"
        return self._execute_to_dicts(query, [str(self._traces_path), limit])

    def get_summary(self, hours: int = 24) -> dict[str, Any]:
        telemetry_exists = self._file_exists(self._telemetry_path)
        traces_exists = self._file_exists(self._traces_path)
        summary: dict[str, Any] = {
            "time_range_hours": hours,
            "latency": {
                "overall": {"count": 0, "avg_ms": None, "p95_ms": None},
                "by_operation": [],
            },
            "tokens": {
                "total_input": 0,
                "total_output": 0,
                "total": 0,
                "input_per_response_avg": 0,
                "output_per_response_avg": 0,
                "burn_rate_per_hour": {"input": 0, "output": 0},
                "over_time": [],
                "by_user": [],
            },
            "errors": {"over_time": [], "recent": [], "rate": 0},
            "tools": {
                "inspector": [],
                "statistics": {
                    "total_calls": 0,
                    "success_rate": 0,
                    "failure_rate": 0,
                    "by_tool": [],
                },
            },
            "agent": {
                "name": None,
                "models": [],
                "total_invocations": 0,
                "total_llm_calls": 0,
            },
            "sre": {
                "availability": 0,
                "throughput_per_min": 0,
                "active_users_24h": 0,
                "total_spans": 0,
            },
            "log_levels": [],
            "span_names": [],
            "files": {
                "telemetry_log": {
                    "exists": telemetry_exists,
                    "path": str(self._telemetry_path),
                },
                "traces_log": {"exists": traces_exists, "path": str(self._traces_path)},
            },
        }
        if telemetry_exists:
            summary["log_levels"] = self.get_log_level_counts()
            summary["errors"]["over_time"] = self.get_errors_over_time(hours)
            summary["errors"]["recent"] = self._get_recent_errors_simple(limit=5)
            total_logs = sum(l.get("count", 0) for l in summary["log_levels"])
            error_count = next(
                (
                    l.get("count", 0)
                    for l in summary["log_levels"]
                    if l.get("level") == "ERROR"
                ),
                0,
            )
            summary["errors"]["rate"] = (
                round(error_count / total_logs, 4) if total_logs > 0 else 0
            )
        if traces_exists:
            summary["latency"]["overall"] = self.get_span_latency_stats(hours)
            summary["latency"]["by_operation"] = self.get_latency_by_operation(hours)
            summary["tokens"]["over_time"] = self.get_token_usage_over_time(hours)
            summary["tokens"]["by_user"] = self.get_token_usage_by_user(hours)
            token_metrics = self.get_token_metrics(hours)
            summary["tokens"]["total_input"] = token_metrics["total_input"]
            summary["tokens"]["total_output"] = token_metrics["total_output"]
            summary["tokens"]["total"] = token_metrics["total"]
            summary["tokens"]["input_per_response_avg"] = token_metrics[
                "input_per_response_avg"
            ]
            summary["tokens"]["output_per_response_avg"] = token_metrics[
                "output_per_response_avg"
            ]
            summary["tokens"]["burn_rate_per_hour"] = token_metrics[
                "burn_rate_per_hour"
            ]
            summary["tools"]["inspector"] = self.get_tool_inspector(limit=20)
            summary["tools"]["statistics"] = self.get_tool_statistics(hours)
            summary["agent"] = self.get_agent_metrics()
            summary["sre"] = self.get_sre_metrics(hours)
            summary["span_names"] = self.get_span_names_summary()
        return summary

    def _get_recent_errors_simple(self, limit: int = 5) -> list[dict[str, Any]]:
        if not self._file_exists(self._telemetry_path):
            return []
        query = "SELECT timestamp, level, name, message FROM read_json_auto(?, ignore_errors=true) WHERE level = 'ERROR' ORDER BY timestamp DESC LIMIT ?"
        return self._execute_to_dicts(query, [str(self._telemetry_path), limit])


_telemetry_queries: TelemetryQueries | None = None


def get_telemetry_queries() -> TelemetryQueries:
    global _telemetry_queries
    if _telemetry_queries is None:
        _telemetry_queries = TelemetryQueries()
    return _telemetry_queries
