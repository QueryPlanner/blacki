"""DuckDB-based telemetry query module.

Provides a query interface for local JSON log files using DuckDB.
Designed for both FastAPI endpoints and potential ADK tool integration.
"""

import json
from pathlib import Path
from typing import Any

import duckdb

from blacki.utils.observability import get_log_dir


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost using LiteLLM's pricing data.

    Args:
        model: The model name (e.g., 'openrouter/deepseek/deepseek-v4-pro')
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens

    Returns:
        Estimated cost in USD
    """
    try:
        import litellm

        cost = litellm.completion_cost(
            model=model,
            prompt=prompt_tokens,
            completion=completion_tokens,
        )
        return cost or 0.0
    except Exception:
        return 0.0


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
                "total_cached": 0,
                "cache_hit_rate": 0,
                "input_per_response_avg": 0,
                "output_per_response_avg": 0,
                "burn_rate_per_hour": {"input": 0, "output": 0},
            }
        query = f"""
            WITH traces AS (
                SELECT {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."gen_ai.usage.input_tokens"')::BIGINT as input_tokens,
                    json_extract_string(attributes, '$."gen_ai.usage.output_tokens"')::BIGINT as output_tokens,
                    COALESCE(
                        json_extract_string(attributes, '$."llm.usage.cached_tokens"')::BIGINT,
                        TRY_CAST(json_extract_string(
                            json_extract_string(attributes, '$."gcp.vertex.agent.llm_response"'),
                            '$.usage_metadata.cached_content_token_count'
                        ) AS BIGINT),
                        0
                    ) as cached_tokens
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'attributes': 'JSON'}})
                WHERE start_time IS NOT NULL AND name LIKE 'call_llm%'
            )
            SELECT COALESCE(SUM(input_tokens), 0) as total_input,
                   COALESCE(SUM(output_tokens), 0) as total_output,
                   COALESCE(SUM(cached_tokens), 0) as total_cached,
                   COUNT(*) as response_count
            FROM traces WHERE start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
        """
        result = self._execute_to_dict(query, [str(self._traces_path), hours])
        total_input = result.get("total_input") or 0
        total_output = result.get("total_output") or 0
        total_cached = result.get("total_cached") or 0
        response_count = max(result.get("response_count") or 0, 1)
        cache_hit_rate = round(total_cached / total_input, 3) if total_input > 0 else 0
        return {
            "total_input": total_input,
            "total_output": total_output,
            "total": total_input + total_output,
            "total_cached": total_cached,
            "cache_hit_rate": cache_hit_rate,
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
                WHERE start_time IS NOT NULL AND name LIKE 'call_llm%'
            )
            SELECT user_id, SUM(input_tokens) as input_tokens, SUM(output_tokens) as output_tokens,
                SUM(input_tokens + output_tokens) as total_tokens, COUNT(*) as request_count
            FROM traces WHERE start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?) AND user_id IS NOT NULL
            GROUP BY user_id ORDER BY total_tokens DESC LIMIT 20
        """
        return self._execute_to_dicts(query, [str(self._traces_path), hours])

    def get_cost_metrics(self, hours: int = 24) -> dict[str, Any]:
        """Calculate cost metrics using LiteLLM's pricing data.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with total cost, cost by model, and cached savings
        """
        if not self._file_exists(self._traces_path):
            return {
                "total_cost": 0.0,
                "by_model": [],
                "cached_savings": 0.0,
            }
        query = f"""
            WITH traces AS (
                SELECT {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."gen_ai.request.model"') as model,
                    COALESCE(json_extract_string(attributes, '$."gen_ai.usage.input_tokens"')::BIGINT, 0) as input_tokens,
                    COALESCE(json_extract_string(attributes, '$."gen_ai.usage.output_tokens"')::BIGINT, 0) as output_tokens,
                    COALESCE(
                        json_extract_string(attributes, '$."llm.usage.cached_tokens"')::BIGINT,
                        TRY_CAST(json_extract_string(
                            json_extract_string(attributes, '$."gcp.vertex.agent.llm_response"'),
                            '$.usage_metadata.cached_content_token_count'
                        ) AS BIGINT),
                        0
                    ) as cached_tokens
                FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'attributes': 'JSON'}})
                WHERE start_time IS NOT NULL AND name LIKE 'call_llm%'
            )
            SELECT model, SUM(input_tokens) as input_tokens, SUM(output_tokens) as output_tokens, SUM(cached_tokens) as cached_tokens
            FROM traces
            WHERE start_epoch IS NOT NULL AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
            GROUP BY model
        """
        results = self._execute_to_dicts(query, [str(self._traces_path), hours])

        total_cost = 0.0
        cached_savings = 0.0
        by_model = []

        for row in results:
            model = row.get("model") or "unknown"
            input_tokens = row.get("input_tokens") or 0
            output_tokens = row.get("output_tokens") or 0
            cached_tokens = row.get("cached_tokens") or 0

            cost = _calculate_cost(model, input_tokens, output_tokens)
            cached_cost = _calculate_cost(model, cached_tokens, 0)

            total_cost += cost
            cached_savings += cached_cost

            by_model.append(
                {
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "cost": round(cost, 6),
                }
            )

        by_model.sort(key=lambda x: x["cost"], reverse=True)

        return {
            "total_cost": round(total_cost, 6),
            "by_model": by_model[:10],
            "cached_savings": round(cached_savings, 6),
        }

    def get_tokens_per_second(self, hours: int = 24) -> dict[str, Any]:
        if not self._file_exists(self._traces_path):
            return {
                "avg_tokens_per_sec": 0.0,
                "by_model": [],
                "by_request": [],
            }
        query = f"""
            WITH traces AS (
                SELECT
                    {self.DURATION_EXPR} as duration_ms,
                    {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."gen_ai.request.model"') as model,
                    COALESCE(json_extract_string(attributes, '$."gen_ai.usage.output_tokens"')::BIGINT, 0) as output_tokens,
                    context->>'trace_id' as trace_id
                FROM read_json_auto(?, ignore_errors=true, columns={{
                    'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON',
                    'attributes': 'JSON', 'context': 'JSON'
                }})
                WHERE start_time IS NOT NULL AND name LIKE 'call_llm%'
            )
            SELECT
                model,
                trace_id,
                output_tokens,
                duration_ms,
                CASE
                    WHEN duration_ms > 0 THEN ROUND(output_tokens / (duration_ms / 1000.0), 2)
                    ELSE 0
                END as tokens_per_sec
            FROM traces
            WHERE start_epoch IS NOT NULL
              AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
              AND duration_ms > 0
              AND output_tokens > 0
        """
        results = self._execute_to_dicts(query, [str(self._traces_path), hours])

        if not results:
            return {
                "avg_tokens_per_sec": 0.0,
                "by_model": [],
                "by_request": [],
            }

        total_tokens = sum(r["output_tokens"] or 0 for r in results)
        total_duration_sec = sum((r["duration_ms"] or 0) / 1000.0 for r in results)
        avg_tokens_per_sec = (
            round(total_tokens / total_duration_sec, 2)
            if total_duration_sec > 0
            else 0.0
        )

        model_stats: dict[str, dict[str, float]] = {}
        for r in results:
            model = r.get("model") or "unknown"
            if model not in model_stats:
                model_stats[model] = {"tokens": 0, "duration_sec": 0, "count": 0}
            model_stats[model]["tokens"] += r.get("output_tokens") or 0
            model_stats[model]["duration_sec"] += (r.get("duration_ms") or 0) / 1000.0
            model_stats[model]["count"] += 1

        by_model: list[dict[str, Any]] = []
        for model, stats in model_stats.items():
            avg = (
                round(stats["tokens"] / stats["duration_sec"], 2)
                if stats["duration_sec"] > 0
                else 0.0
            )
            by_model.append(
                {
                    "model": model,
                    "avg_tokens_per_sec": avg,
                    "total_tokens": int(stats["tokens"]),
                    "request_count": int(stats["count"]),
                }
            )
        by_model.sort(key=lambda x: x["avg_tokens_per_sec"], reverse=True)

        by_request = [
            {
                "trace_id": r.get("trace_id"),
                "model": r.get("model") or "unknown",
                "output_tokens": r.get("output_tokens"),
                "duration_ms": round(r.get("duration_ms") or 0, 2),
                "tokens_per_sec": r.get("tokens_per_sec"),
            }
            for r in sorted(
                results,
                key=lambda x: float(x.get("tokens_per_sec") or 0),
                reverse=True,
            )[:50]
        ]

        return {
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "by_model": by_model,
            "by_request": by_request,
        }

    def get_tool_inspector(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self._file_exists(self._traces_path):
            return []
        query = f"""
            SELECT name, {self.DURATION_EXPR} as duration_ms,
                json_extract_string(attributes, '$."gen_ai.operation.name"') as operation,
                CASE WHEN json_type(status) = 'OBJECT' THEN json_extract_string(status, '$.status_code') ELSE NULL END as status_code,
                {self.EPOCH_EXPR} as start_epoch,
                json_extract_string(attributes, '$."input.value"') as input_value,
                json_extract_string(attributes, '$."output.value"') as output_value,
                json_extract_string(attributes, '$."gen_ai.tool.description"') as tool_description,
                context->>'trace_id' as trace_id,
                parent_id
            FROM read_json_auto(?, ignore_errors=true, columns={{'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'status': 'JSON', 'attributes': 'JSON', 'context': 'JSON', 'parent_id': 'VARCHAR'}})
            WHERE name LIKE 'execute_tool%' ORDER BY start_epoch DESC LIMIT ?
        """
        results = self._execute_to_dicts(query, [str(self._traces_path), limit])
        formatted = []
        for r in results:
            tool_name = r["name"].replace("execute_tool ", "") if r["name"] else None
            input_data = r.get("input_value")
            output_data = r.get("output_value")
            try:
                if input_data:
                    input_data = json.loads(input_data)
                if output_data:
                    output_data = json.loads(output_data)
            except (json.JSONDecodeError, TypeError):
                pass
            formatted.append(
                {
                    "name": tool_name,
                    "duration_ms": round(r["duration_ms"], 2)
                    if r["duration_ms"]
                    else None,
                    "status": r["status_code"] or "UNKNOWN",
                    "timestamp": r.get("start_epoch"),
                    "input": input_data,
                    "output": output_data,
                    "description": (r.get("tool_description") or "")[:200]
                    if r.get("tool_description")
                    else None,
                    "trace_id": r.get("trace_id"),
                    "parent_id": r.get("parent_id"),
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
                    CASE WHEN json_type(status) = 'OBJECT' THEN json_extract_string(status, '$.status_code') ELSE NULL END as status_code
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
            WHERE name = 'agent_run' OR name = 'call_llm'
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
                SELECT CASE WHEN json_type(status) = 'OBJECT' THEN json_extract_string(status, '$.status_code') ELSE NULL END as status_code
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
                WHERE start_time IS NOT NULL AND name LIKE 'call_llm%'
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
                "total_cached": 0,
                "cache_hit_rate": 0,
                "input_per_response_avg": 0,
                "output_per_response_avg": 0,
                "burn_rate_per_hour": {"input": 0, "output": 0},
                "over_time": [],
                "by_user": [],
                "by_conversation": [],
                "tokens_per_second": {
                    "avg_tokens_per_sec": 0.0,
                    "by_model": [],
                    "by_request": [],
                },
            },
            "cost": {
                "total": 0.0,
                "by_model": [],
                "cached_savings": 0.0,
            },
            "errors": {
                "over_time": [],
                "recent": [],
                "rate": 0,
                "cognitive": {
                    "cognitive_errors": [],
                    "infra_errors": [],
                    "summary": {
                        "total_cognitive": 0,
                        "total_infra": 0,
                        "tool_failures": 0,
                        "validation_errors": 0,
                    },
                },
            },
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
            "waterfall_traces": [],
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
            summary["tokens"]["total_cached"] = token_metrics["total_cached"]
            summary["tokens"]["cache_hit_rate"] = token_metrics["cache_hit_rate"]
            summary["tokens"]["input_per_response_avg"] = token_metrics[
                "input_per_response_avg"
            ]
            summary["tokens"]["output_per_response_avg"] = token_metrics[
                "output_per_response_avg"
            ]
            summary["tokens"]["burn_rate_per_hour"] = token_metrics[
                "burn_rate_per_hour"
            ]
            cost_metrics = self.get_cost_metrics(hours)
            summary["cost"]["total"] = cost_metrics["total_cost"]
            summary["cost"]["by_model"] = cost_metrics["by_model"]
            summary["cost"]["cached_savings"] = cost_metrics["cached_savings"]
            summary["tools"]["inspector"] = self.get_tool_inspector(limit=20)
            summary["tools"]["statistics"] = self.get_tool_statistics(hours)
            summary["agent"] = self.get_agent_metrics()
            summary["sre"] = self.get_sre_metrics(hours)
            summary["span_names"] = self.get_span_names_summary()
            summary["waterfall_traces"] = self.get_waterfall_traces(limit=10)
            summary["tokens"]["by_conversation"] = self.get_token_accumulation_by_turn(
                hours
            )
            summary["errors"]["cognitive"] = self.get_cognitive_errors(hours)
            summary["tokens"]["tokens_per_second"] = self.get_tokens_per_second(hours)
        return summary

    def _get_recent_errors_simple(self, limit: int = 5) -> list[dict[str, Any]]:
        if not self._file_exists(self._telemetry_path):
            return []
        query = "SELECT timestamp, level, name, message FROM read_json_auto(?, ignore_errors=true) WHERE level = 'ERROR' ORDER BY timestamp DESC LIMIT ?"
        return self._execute_to_dicts(query, [str(self._telemetry_path), limit])

    def get_waterfall_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get hierarchical execution traces for waterfall visualization.

        Returns traces with parent-child relationships to visualize agent
        execution flow: agent_run -> call_llm -> execute_tool -> call_llm.

        Args:
            limit: Maximum number of root traces to return

        Returns:
            List of trace trees with nested spans
        """
        if not self._file_exists(self._traces_path):
            return []

        query = f"""
            WITH spans AS (
                SELECT
                    name,
                    context->>'trace_id' as trace_id,
                    context->>'span_id' as span_id,
                    parent_id,
                    {self.DURATION_EXPR} as duration_ms,
                    {self.EPOCH_EXPR} as start_epoch,
                    CASE WHEN json_type(status) = 'OBJECT'
                        THEN json_extract_string(status, '$.status_code')
                        ELSE NULL END as status_code,
                    json_extract_string(attributes, '$."gen_ai.operation.name"') as operation,
                    json_extract_string(attributes, '$."gen_ai.request.model"') as model,
                    json_extract_string(attributes, '$."gen_ai.usage.input_tokens"')::BIGINT as input_tokens,
                    json_extract_string(attributes, '$."gen_ai.usage.output_tokens"')::BIGINT as output_tokens,
                    json_extract_string(attributes, '$."tool.name"') as tool_name
                FROM read_json_auto(?, ignore_errors=true, columns={{
                    'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON',
                    'status': 'JSON', 'attributes': 'JSON', 'context': 'JSON', 'parent_id': 'VARCHAR'
                }})
                WHERE start_time IS NOT NULL AND end_time IS NOT NULL
            )
            SELECT * FROM spans
            WHERE start_epoch IS NOT NULL
            ORDER BY start_epoch DESC
            LIMIT 500
        """
        results = self._execute_to_dicts(query, [str(self._traces_path)])

        traces: dict[str, dict[str, Any]] = {}
        spans_by_id: dict[str, dict[str, Any]] = {}

        for r in results:
            trace_id = r.get("trace_id")
            span_id = r.get("span_id")
            if not trace_id or not span_id:
                continue

            span_data = {
                "name": r.get("name"),
                "span_id": span_id,
                "trace_id": trace_id,
                "parent_id": r.get("parent_id"),
                "duration_ms": round(r["duration_ms"], 2) if r["duration_ms"] else None,
                "start_epoch": r.get("start_epoch"),
                "status": r.get("status_code") or "UNSET",
                "operation": r.get("operation"),
                "model": r.get("model"),
                "input_tokens": r.get("input_tokens"),
                "output_tokens": r.get("output_tokens"),
                "tool_name": r.get("tool_name"),
                "children": [],
            }
            spans_by_id[span_id] = span_data

            if trace_id not in traces:
                traces[trace_id] = {
                    "trace_id": trace_id,
                    "spans": [],
                    "root": None,
                }

        for _span_id, span_data in spans_by_id.items():
            parent_id = span_data.get("parent_id")
            if parent_id and parent_id in spans_by_id:
                spans_by_id[parent_id]["children"].append(span_data)
            else:
                trace_id = span_data["trace_id"]
                traces[trace_id]["root"] = span_data

        def sort_children(span: dict[str, Any]) -> dict[str, Any]:
            span["children"] = sorted(
                [sort_children(c) for c in span.get("children", [])],
                key=lambda x: x.get("start_epoch") or 0,
            )
            return span

        def get_latest_epoch(trace_id: str) -> float:
            epochs = [
                s.get("start_epoch") or 0
                for s in spans_by_id.values()
                if s.get("trace_id") == trace_id
            ]
            return max(epochs) if epochs else 0

        result = []
        for trace_id, trace in sorted(
            traces.items(),
            key=lambda x: get_latest_epoch(x[0]),
            reverse=True,
        )[:limit]:
            if trace.get("root"):
                result.append(
                    {
                        "trace_id": trace_id,
                        "root": sort_children(trace["root"]),
                    }
                )
        return result

    def get_token_accumulation_by_turn(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get token usage per conversation turn to diagnose context bloat.

        Tracks how input token count grows across turns within a conversation,
        helping identify if the agent is dragging too much context.

        Args:
            hours: Number of hours to look back

        Returns:
            List of turn-by-turn token metrics grouped by conversation
        """
        if not self._file_exists(self._traces_path):
            return []

        query = f"""
            WITH traces AS (
                SELECT
                    {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."gen_ai.conversation.id"') as conversation_id,
                    json_extract_string(attributes, '$."gen_ai.usage.input_tokens"')::BIGINT as input_tokens,
                    json_extract_string(attributes, '$."gen_ai.usage.output_tokens"')::BIGINT as output_tokens,
                    json_extract_string(attributes, '$."gen_ai.request.model"') as model
                FROM read_json_auto(?, ignore_errors=true, columns={{
                    'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON', 'attributes': 'JSON'
                }})
                WHERE start_time IS NOT NULL AND name LIKE 'call_llm%'
            )
            SELECT
                conversation_id,
                COUNT(*) as turn_count,
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                AVG(input_tokens) as avg_input_per_turn,
                MAX(input_tokens) as max_input,
                MIN(input_tokens) as min_input,
                array_agg(input_tokens ORDER BY start_epoch) as input_tokens_by_turn
            FROM traces
            WHERE start_epoch IS NOT NULL
              AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
              AND conversation_id IS NOT NULL
            GROUP BY conversation_id
            ORDER BY total_input DESC
            LIMIT 20
        """
        results = self._execute_to_dicts(query, [str(self._traces_path), hours])

        formatted = []
        for r in results:
            turns = r.get("input_tokens_by_turn") or []
            growth_rate = 0.0
            if len(turns) > 1:
                first_half = turns[: len(turns) // 2]
                second_half = turns[len(turns) // 2 :]
                avg_first = sum(first_half) / len(first_half) if first_half else 0
                avg_second = sum(second_half) / len(second_half) if second_half else 0
                growth_rate = (
                    round((avg_second - avg_first) / avg_first, 3)
                    if avg_first > 0
                    else 0
                )

            formatted.append(
                {
                    "conversation_id": r.get("conversation_id"),
                    "turn_count": r.get("turn_count") or 0,
                    "total_input": r.get("total_input") or 0,
                    "total_output": r.get("total_output") or 0,
                    "avg_input_per_turn": round(r.get("avg_input_per_turn") or 0, 1),
                    "max_input": r.get("max_input") or 0,
                    "min_input": r.get("min_input") or 0,
                    "input_tokens_by_turn": turns,
                    "context_growth_rate": growth_rate,
                }
            )
        return formatted

    def get_cognitive_errors(self, hours: int = 24) -> dict[str, Any]:
        """Get cognitive errors separate from infrastructure errors.

        Cognitive errors include:
        - Tool execution failures (tool returned error status)
        - Schema validation failures (malformed JSON input)
        - Context overflows (exceeding max context length)
        - Hallucinations (calling non-existent tools)

        Infrastructure errors include:
        - Network errors
        - Polling loop failures
        - API rate limits

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with cognitive_errors, infra_errors, and summary
        """
        if not self._file_exists(self._traces_path):
            return {
                "cognitive_errors": [],
                "infra_errors": [],
                "summary": {
                    "total_cognitive": 0,
                    "total_infra": 0,
                    "tool_failures": 0,
                    "validation_errors": 0,
                },
            }

        tool_errors_query = f"""
            WITH tool_spans AS (
                SELECT
                    name,
                    {self.DURATION_EXPR} as duration_ms,
                    {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."tool.name"') as tool_name,
                    json_extract_string(attributes, '$."input.value"') as input_value,
                    json_extract_string(attributes, '$."output.value"') as output_value,
                    CASE WHEN json_type(status) = 'OBJECT'
                        THEN json_extract_string(status, '$.status_code')
                        ELSE NULL END as status_code
                FROM read_json_auto(?, ignore_errors=true, columns={{
                    'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON',
                    'status': 'JSON', 'attributes': 'JSON'
                }})
                WHERE name LIKE 'execute_tool%'
                  AND start_epoch IS NOT NULL
                  AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
            )
            SELECT * FROM tool_spans WHERE status_code = 'ERROR'
        """
        tool_errors = self._execute_to_dicts(
            tool_errors_query, [str(self._traces_path), hours]
        )

        infra_errors_query = f"""
            WITH infra_spans AS (
                SELECT
                    name,
                    {self.EPOCH_EXPR} as start_epoch,
                    json_extract_string(attributes, '$."error.type"') as error_type,
                    json_extract_string(attributes, '$."error.message"') as error_message,
                    CASE WHEN json_type(status) = 'OBJECT'
                        THEN json_extract_string(status, '$.status_code')
                        ELSE NULL END as status_code
                FROM read_json_auto(?, ignore_errors=true, columns={{
                    'name': 'VARCHAR', 'start_time': 'JSON', 'end_time': 'JSON',
                    'status': 'JSON', 'attributes': 'JSON'
                }})
                WHERE name IN ('agent_run', 'invocation')
                  AND start_epoch IS NOT NULL
                  AND start_epoch > epoch(now() - INTERVAL '1 hour' * ?)
            )
            SELECT * FROM infra_spans WHERE status_code = 'ERROR'
        """
        infra_errors = self._execute_to_dicts(
            infra_errors_query, [str(self._traces_path), hours]
        )

        cognitive_formatted = []
        for e in tool_errors:
            output_data = e.get("output_value")
            error_message = None
            try:
                if output_data:
                    parsed = json.loads(output_data)
                    if isinstance(parsed, dict):
                        error_message = parsed.get("response", {}).get(
                            "message"
                        ) or parsed.get("error")
            except (json.JSONDecodeError, TypeError):
                error_message = output_data

            cognitive_formatted.append(
                {
                    "type": "tool_failure",
                    "tool_name": e.get("tool_name")
                    or (e.get("name", "").replace("execute_tool ", "")),
                    "timestamp": e.get("start_epoch"),
                    "duration_ms": round(e["duration_ms"], 2)
                    if e.get("duration_ms")
                    else None,
                    "error_message": error_message,
                    "input": e.get("input_value"),
                }
            )

        infra_formatted = []
        for e in infra_errors:
            infra_formatted.append(
                {
                    "type": "infrastructure",
                    "operation": e.get("name"),
                    "timestamp": e.get("start_epoch"),
                    "error_type": e.get("error_type"),
                    "error_message": e.get("error_message"),
                }
            )

        return {
            "cognitive_errors": cognitive_formatted[:20],
            "infra_errors": infra_formatted[:20],
            "summary": {
                "total_cognitive": len(cognitive_formatted),
                "total_infra": len(infra_formatted),
                "tool_failures": len(cognitive_formatted),
                "validation_errors": 0,
            },
        }


_telemetry_queries: TelemetryQueries | None = None


def get_telemetry_queries() -> TelemetryQueries:
    global _telemetry_queries
    if _telemetry_queries is None:
        _telemetry_queries = TelemetryQueries()
    return _telemetry_queries
