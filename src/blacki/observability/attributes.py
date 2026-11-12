"""OpenTelemetry attribute names safe to keep in the slim trace log."""

from __future__ import annotations

# Only these OpenTelemetry attributes are useful for the operator dashboard.
# In particular, no prompt/content/tool argument attributes are included.
TRACE_ATTRIBUTE_ALLOWLIST = frozenset(
    {
        "service.name",
        "service.namespace",
        "service.version",
        "service.instance.id",
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.operation.name",
        "gen_ai.agent.name",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.total_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.prompt_token_count",
        "gen_ai.usage.candidates_token_count",
        "gen_ai.usage.cost",
        "gen_ai.usage.cost_estimate",
        "gen_ai.cost.upstream_inference_cost",
        "gen_ai.cost.source",
        "gen_ai.cost.kind",
        "llm.token_count.input",
        "llm.token_count.output",
        "llm.token_count.total",
        "llm.token_count.prompt",
        "llm.token_count.completion",
        "http.method",
        "http.route",
        "http.status_code",
        "rpc.system",
        "error.type",
        "error.code",
        "blacki.user_id",
        "blacki.session_id",
        "blacki.invocation_id",
        "gcp.vertex.agent.session_id",
        "gcp.vertex.agent.invocation_id",
        "gen_ai.conversation.id",
        "gen_ai.tool.name",
        "agent.name",
        "llm.model_name",
        "session.id",
        "user.id",
    }
)
