"""Tests for provider cost capture at the LiteLLM boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from blacki.observability.costs import (
    CostAwareLiteLLMClient,
    _observe_response,
    _remember_observation,
    _set_span_cost_attributes,
    extract_cost_observation,
)
from blacki.observability.ledger import read_usage_ledger
from blacki.observability.lifecycle import attach_cost_metadata, begin_cost_capture


def test_extract_cost_observation_preserves_provider_and_upstream_costs() -> None:
    observation = extract_cost_observation(
        {
            "id": "generation-1",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "cost": 0.0025,
                "cost_details": {"upstream_inference_cost": 0.0022},
            },
        },
        "openrouter/google/gemini-2.5-flash",
    )

    assert observation is not None
    assert observation["provider_response_id"] == "generation-1"
    assert observation["input_tokens"] == 12
    assert observation["output_tokens"] == 8
    assert observation["cost_usd"] == 0.0025
    assert observation["upstream_cost_usd"] == 0.0022
    assert observation["estimated_cost_usd"] is None
    assert observation["cost_kind"] == "reported"


def test_extract_cost_observation_distinguishes_estimated_unknown_and_invalid() -> None:
    estimated = extract_cost_observation(
        {
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "_hidden_params": {"response_cost": 0.0004},
        },
        "openai/test",
    )
    assert estimated is not None
    assert estimated["estimated_cost_usd"] == 0.0004
    assert estimated["cost_kind"] == "estimated"

    unknown = extract_cost_observation(
        {"id": "unknown", "usage": {"total_tokens": 5}},
        "openai/test",
    )
    assert unknown is not None
    assert unknown["cost_usd"] is None
    assert unknown["cost_kind"] == "unknown"

    invalid = extract_cost_observation(
        {"usage": {"cost": "not-a-number", "total_tokens": 5}},
        "openai/test",
    )
    assert invalid is not None
    assert invalid["cost_usd"] is None

    assert extract_cost_observation({"usage": {"cost": -1}}, "openai/test") is None
    assert extract_cost_observation({}, "openai/test") is None


def test_span_cost_attributes_are_recorded_when_span_is_active() -> None:
    span = MagicMock()
    span.is_recording.return_value = True
    with patch("blacki.observability.costs.trace.get_current_span", return_value=span):
        _set_span_cost_attributes(
            {
                "cost_usd": 0.01,
                "estimated_cost_usd": 0.02,
                "upstream_cost_usd": 0.009,
                "cost_source": "provider_usage",
                "cost_kind": "reported",
            }
        )

    assert span.set_attribute.call_count == 5
    span.set_attribute.assert_any_call("gen_ai.usage.cost", 0.01)
    span.set_attribute.assert_any_call("gen_ai.usage.cost_estimate", 0.02)
    span.set_attribute.assert_any_call("gen_ai.cost.upstream_inference_cost", 0.009)
    with patch("blacki.observability.costs.trace.get_current_span", return_value=span):
        _set_span_cost_attributes({"cost_source": "unknown", "cost_kind": "unknown"})


def test_remember_observation_preserves_reported_cost_and_handles_no_capture() -> None:
    assert _remember_observation({"cost_usd": 0.01}) is None

    begin_cost_capture(user_id="user-1", session_id="session-1", invocation_id="inv-1")
    reported = _remember_observation(
        {
            "model": "openrouter/test",
            "provider_response_id": "generation-1",
            "cost_usd": 0.01,
            "cost_source": "provider_usage",
            "cost_kind": "reported",
            "currency": "USD",
        }
    )
    assert reported is not None
    preserved = _remember_observation({"cost_usd": None})
    assert preserved is reported
    assert attach_cost_metadata(LlmResponse()) is not None


@pytest.mark.asyncio
async def test_observe_response_handles_missing_capture_and_write_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    await _observe_response({}, "openrouter/test", tmp_path / "costs.db")
    await _observe_response(
        {"usage": {"total_tokens": 1}}, "openrouter/test", tmp_path / "costs.db"
    )

    begin_cost_capture(user_id="user-1", session_id="session-1", invocation_id="inv-1")
    caplog.set_level("WARNING")
    with patch(
        "blacki.observability.costs.write_usage_record",
        side_effect=OSError("read-only"),
    ):
        await _observe_response(
            {"usage": {"total_tokens": 1, "cost": 0.001}},
            "openrouter/test",
            tmp_path / "costs.db",
        )
    assert "Unable to persist model usage" in caplog.text
    assert attach_cost_metadata(LlmResponse()) is not None


def test_partial_response_keeps_capture_until_final_response() -> None:
    begin_cost_capture(user_id="user-1", session_id="session-1", invocation_id="inv-1")
    _remember_observation(
        {
            "model": "openrouter/test",
            "provider_response_id": "generation-1",
            "cost_usd": 0.01,
            "cost_source": "provider_usage",
            "cost_kind": "reported",
            "currency": "USD",
        }
    )
    partial = LlmResponse(partial=True)
    assert attach_cost_metadata(partial) is not None
    final = LlmResponse(partial=False)
    final.custom_metadata = {"existing": True}
    assert attach_cost_metadata(final) is not None


@pytest.mark.asyncio
async def test_non_stream_response_is_ledgered_and_attached_to_adk_event(
    tmp_path: Path,
) -> None:
    raw_response = {
        "id": "generation-1",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.01,
            "cost_details": {"upstream_inference_cost": 0.009},
        },
    }
    begin_cost_capture(
        user_id="user-1",
        session_id="session-1",
        invocation_id="inv-1",
    )
    client = CostAwareLiteLLMClient(tmp_path / "costs.db")
    with patch.object(
        LiteLLMClient,
        "acompletion",
        new=AsyncMock(return_value=raw_response),
    ):
        returned = await client.acompletion(
            model="openrouter/test",
            messages=[],
            tools=[],
            stream=False,
        )

    assert returned is raw_response
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text="ok")])
    )
    metadata = attach_cost_metadata(response)
    assert metadata is not None
    assert metadata["cost_usd"] == 0.01
    assert response.custom_metadata is not None
    assert response.custom_metadata["blacki.cost"]["dedupe_key"].startswith("provider:")

    ledger = read_usage_ledger(
        tmp_path / "costs.db",
        selected_since=None,
        selected_until=10**12,
        month_start=0,
        now=10**12,
    )
    assert ledger.cumulative.records == 1
    assert ledger.cumulative.reported_records == 1
    assert ledger.users["user-1"].cost_nano_usd is not None


@pytest.mark.asyncio
async def test_streaming_final_usage_updates_one_idempotent_ledger_row(
    tmp_path: Path,
) -> None:
    async def response_stream() -> AsyncIterator[dict[str, object]]:
        yield {"id": "generation-stream", "usage": {"total_tokens": 4}}
        yield {
            "id": "generation-stream",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
                "cost": 0.004,
            },
        }

    begin_cost_capture(
        user_id="user-stream",
        session_id="session-stream",
        invocation_id="inv-stream",
    )
    client = CostAwareLiteLLMClient(tmp_path / "costs.db")
    with patch.object(
        LiteLLMClient,
        "acompletion",
        new=AsyncMock(return_value=response_stream()),
    ):
        returned = await client.acompletion(
            model="openrouter/test",
            messages=[],
            tools=[],
            stream=True,
        )
        chunks = [chunk async for chunk in returned]

    assert len(chunks) == 2
    response = LlmResponse()
    assert attach_cost_metadata(response)["cost_usd"] == 0.004  # type: ignore[index]
    ledger = read_usage_ledger(
        tmp_path / "costs.db",
        selected_since=None,
        selected_until=10**12,
        month_start=0,
        now=10**12,
    )
    assert ledger.cumulative.records == 1
    assert ledger.cumulative.reported_records == 1


def test_attach_cost_metadata_without_active_capture_is_noop() -> None:
    assert attach_cost_metadata(LlmResponse()) is None
