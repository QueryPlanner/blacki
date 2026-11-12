"""Tests for model usage and cost capture from OpenAI catalog prices."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.adk.models.llm_response import LlmResponse

from blacki.models import catalog
from blacki.models.catalog import ModelEntry, get_model
from blacki.observability.costs import (
    _ACTIVE_CAPTURE,
    _remember_observation,
    _reset_unpriced_model_warnings,
    _set_span_cost_attributes,
    observe_openai_usage,
)
from blacki.observability.ledger import LEDGER_TABLE, read_usage_ledger
from blacki.observability.lifecycle import attach_cost_metadata, begin_cost_capture

SPEC_USAGE: dict[str, Any] = {
    "input_tokens": 10_000,
    "input_tokens_details": {"cached_tokens": 4_000},
    "output_tokens": 1_000,
    "total_tokens": 11_000,
}


@pytest.fixture(autouse=True)
def _isolated_capture_and_warnings() -> Iterator[None]:
    _ACTIVE_CAPTURE.set(None)
    _reset_unpriced_model_warnings()
    yield
    _ACTIVE_CAPTURE.set(None)
    _reset_unpriced_model_warnings()


def _begin(invocation_id: str = "inv-1") -> None:
    begin_cost_capture(
        user_id="user-1",
        session_id="session-1",
        invocation_id=invocation_id,
    )


def _ledger_rows(ledger_path: Path) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(ledger_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT * FROM {LEDGER_TABLE} ORDER BY observed_at"  # noqa: S608
        ).fetchall()
        return [dict(row) for row in rows]


def _model_entry(**overrides: Any) -> ModelEntry:
    fields: dict[str, Any] = {
        "id": "test-model",
        "display_name": "Test model",
        "efforts": ("low",),
        "default_effort": "low",
        "accepts_sampling_at_none": False,
        "input": 1.00,
        "cached_input": 0.10,
        "cache_write": None,
        "output": 10.00,
    } | overrides
    return ModelEntry(**fields)


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
            "model": "test-model",
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
async def test_usage_handles_missing_capture_and_write_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ledger_path = tmp_path / "costs.db"
    assert await observe_openai_usage({}, "gpt-6.1-terra", ledger_path) is None
    await observe_openai_usage({"input_tokens": 1}, "gpt-6.1-terra", ledger_path)

    _begin()
    caplog.set_level("WARNING")
    with patch(
        "blacki.observability.costs.write_usage_record",
        side_effect=OSError("read-only"),
    ):
        await observe_openai_usage(
            {"input_tokens": 1, "output_tokens": 1}, "gpt-6.1-terra", ledger_path
        )
    assert "Unable to persist model usage" in caplog.text
    assert attach_cost_metadata(LlmResponse()) is not None


def test_partial_response_keeps_capture_until_final_response() -> None:
    begin_cost_capture(user_id="user-1", session_id="session-1", invocation_id="inv-1")
    _remember_observation(
        {
            "model": "test-model",
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
async def test_repeated_usage_updates_one_idempotent_ledger_row(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "costs.db"
    _begin("inv-stream")
    await observe_openai_usage({"total_tokens": 4}, "gpt-6.1-terra", ledger_path)
    await observe_openai_usage(
        {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        "gpt-6.1-terra",
        ledger_path,
    )

    metadata = attach_cost_metadata(LlmResponse())
    assert metadata is not None
    assert metadata["cost_kind"] == "estimate"
    assert metadata["estimated_cost_usd"] == pytest.approx(0.000008)
    ledger = read_usage_ledger(
        ledger_path,
        selected_since=None,
        selected_until=10**12,
        month_start=0,
        now=10**12,
    )
    assert ledger.cumulative.records == 1
    assert ledger.cumulative.estimated_records == 1


def test_attach_cost_metadata_without_active_capture_is_noop() -> None:
    assert attach_cost_metadata(LlmResponse()) is None


@pytest.mark.asyncio
async def test_openai_estimate_prices_cached_input_and_output_from_catalog(
    tmp_path: Path,
) -> None:
    entry = get_model("gpt-6.1-terra")
    assert entry is not None
    assert entry.output is not None
    expected = (
        6_000 * entry.input + 4_000 * entry.cached_input + 1_000 * entry.output
    ) / 1_000_000
    _begin()

    observation = await observe_openai_usage(
        SPEC_USAGE, "gpt-6.1-terra", tmp_path / "costs.db"
    )

    assert observation is not None
    assert observation["estimated_cost_usd"] == pytest.approx(expected)
    assert observation["estimated_cost_usd"] == pytest.approx(0.0112)
    assert observation["cost_kind"] == "estimate"
    assert observation["cost_source"] == "catalog_price"
    assert observation["cost_usd"] is None
    assert observation["upstream_cost_usd"] is None
    assert observation["provider"] == "openai"
    assert observation["cached_tokens"] == 4_000
    assert observation["cache_write_tokens"] is None
    metadata = attach_cost_metadata(LlmResponse())
    assert metadata is not None
    assert metadata["estimated_cost_usd"] == pytest.approx(0.0112)

    [row] = _ledger_rows(tmp_path / "costs.db")
    assert row["model"] == "gpt-6.1-terra"
    assert row["provider"] == "openai"
    assert row["input_tokens"] == 10_000
    assert row["cached_tokens"] == 4_000
    assert row["output_tokens"] == 1_000
    assert row["total_tokens"] == 11_000
    assert row["cost_nano_usd"] is None
    assert row["estimated_cost_nano_usd"] == 11_200_000
    assert row["cost_kind"] == "estimate"
    assert row["cost_source"] == "catalog_price"
    assert row["currency"] == "USD"


@pytest.mark.asyncio
async def test_reasoning_tokens_are_part_of_output_and_not_priced_again(
    tmp_path: Path,
) -> None:
    _begin()

    observation = await observe_openai_usage(
        {
            "input_tokens": 100,
            "output_tokens": 1_000,
            "output_tokens_details": {"reasoning_tokens": 600},
            "total_tokens": 1_100,
        },
        "gpt-6.1-terra",
        tmp_path / "costs.db",
    )

    # 100 input at $1.00/M plus 1,000 output at $5.00/M. Adding the 600
    # reasoning tokens again would give 0.0081.
    assert observation is not None
    assert observation["estimated_cost_usd"] == pytest.approx(0.0051)


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(
            {
                "input_tokens": 10_000,
                "input_tokens_details": {
                    "cached_tokens": 4_000,
                    "cache_write_tokens": 2_000,
                },
                "output_tokens": 1_000,
            },
            id="input-details",
        ),
        pytest.param(
            {
                "input_tokens": 10_000,
                "input_tokens_details": {"cached_tokens": 4_000},
                "cache_write_tokens": 2_000,
                "output_tokens": 1_000,
            },
            id="top-level",
        ),
    ],
)
@pytest.mark.asyncio
async def test_reported_cache_writes_are_priced_at_cache_write_rate(
    tmp_path: Path, usage: dict[str, Any]
) -> None:
    _begin()

    observation = await observe_openai_usage(usage, "gpt-6.1-terra", tmp_path / "c.db")

    # 4,000 uncached input at $1.00/M, 4,000 cached at $0.05/M,
    # 2,000 cache writes at $1.25/M, and 1,000 output at $5.00/M.
    assert observation is not None
    assert observation["cache_write_tokens"] == 2_000
    assert observation["estimated_cost_usd"] == pytest.approx(0.0117)
    [row] = _ledger_rows(tmp_path / "c.db")
    assert row["cache_write_tokens"] == 2_000
    assert row["estimated_cost_nano_usd"] == 11_700_000


@pytest.mark.asyncio
async def test_uncached_input_uses_input_rate_when_cache_writes_are_not_reported(
    tmp_path: Path,
) -> None:
    _begin()

    observation = await observe_openai_usage(
        {"input_tokens": 10_000, "output_tokens": 1_000, "total_tokens": 11_000},
        "gpt-6.1-terra",
        tmp_path / "costs.db",
    )

    # 10,000 input at $1.00/M plus 1,000 output at $5.00/M.
    assert observation is not None
    assert observation["cache_write_tokens"] is None
    assert observation["estimated_cost_usd"] == pytest.approx(0.015)


@pytest.mark.asyncio
async def test_reported_cache_writes_are_input_when_model_has_no_write_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The public catalog prices writes for every entry, so inject a test entry
    # without a write price into the catalog's lookup table.
    monkeypatch.setitem(
        catalog._ENTRIES_BY_ID,
        "test-no-write-price",
        _model_entry(id="test-no-write-price", cache_write=None),
    )
    _begin()

    observation = await observe_openai_usage(
        {
            "input_tokens": 10_000,
            "input_tokens_details": {
                "cached_tokens": 4_000,
                "cache_write_tokens": 2_000,
            },
            "output_tokens": 1_000,
        },
        "test-no-write-price",
        tmp_path / "costs.db",
    )

    # 6,000 input (writes included) at $1.00/M, 4,000 cached at $0.10/M,
    # and 1,000 output at $10.00/M.
    assert observation is not None
    assert observation["estimated_cost_usd"] == pytest.approx(0.0164)


@pytest.mark.asyncio
async def test_unpriced_model_records_tokens_without_cost_and_warns_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="blacki.llm_costs")
    ledger_path = tmp_path / "costs.db"
    usage = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}

    for invocation_id in ("inv-1", "inv-2"):
        _begin(invocation_id)
        observation = await observe_openai_usage(usage, "gpt-unlisted", ledger_path)
        assert observation is not None
        assert observation["estimated_cost_usd"] is None
        assert observation["cost_kind"] == "unknown"
        assert observation["cost_source"] == "usage_only"

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "No catalog price" in record.message
    ]
    assert len(warnings) == 1
    rows = _ledger_rows(ledger_path)
    assert [row["input_tokens"] for row in rows] == [12, 12]
    assert {row["cost_kind"] for row in rows} == {"unknown"}
    assert all(
        row["cost_nano_usd"] is None and row["estimated_cost_nano_usd"] is None
        for row in rows
    )


@pytest.mark.asyncio
async def test_model_without_output_price_records_tokens_without_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(
        catalog._ENTRIES_BY_ID,
        "test-no-output-price",
        _model_entry(id="test-no-output-price", output=None),
    )
    _begin()

    observation = await observe_openai_usage(
        {"input_tokens": 500, "output_tokens": 50},
        "test-no-output-price",
        tmp_path / "costs.db",
    )

    assert observation is not None
    assert observation["estimated_cost_usd"] is None
    assert observation["cost_kind"] == "unknown"
    assert "No catalog price for model test-no-output-price" in caplog.text


@pytest.mark.asyncio
async def test_usage_breakdown_larger_than_input_is_not_estimated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="blacki.llm_costs")
    _begin()

    observation = await observe_openai_usage(
        {
            "input_tokens": 1_000,
            "input_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 400},
            "output_tokens": 10,
        },
        "gpt-6.1-terra",
        tmp_path / "costs.db",
    )

    assert observation is not None
    assert observation["estimated_cost_usd"] is None
    assert observation["cost_kind"] == "unknown"
    assert "more cached and cache-write tokens than input" in caplog.text
    [row] = _ledger_rows(tmp_path / "costs.db")
    assert row["input_tokens"] == 1_000
    assert row["estimated_cost_nano_usd"] is None


@pytest.mark.asyncio
async def test_usage_without_input_or_output_tokens_records_no_estimate(
    tmp_path: Path,
) -> None:
    _begin()

    observation = await observe_openai_usage(
        {"total_tokens": 5}, "gpt-6.1-terra", tmp_path / "costs.db"
    )

    assert observation is not None
    assert observation["input_tokens"] is None
    assert observation["estimated_cost_usd"] is None
    assert observation["cost_kind"] == "unknown"


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param({}, id="empty"),
        pytest.param(
            {
                "input_tokens": "many",
                "output_tokens": -1,
                "input_tokens_details": "unparsed",
            },
            id="invalid-values",
        ),
        pytest.param({"input_tokens": True}, id="boolean"),
    ],
)
@pytest.mark.asyncio
async def test_usage_without_token_counts_is_not_recorded(
    tmp_path: Path, usage: dict[str, Any]
) -> None:
    _begin()

    observation = await observe_openai_usage(usage, "gpt-6.1-terra", tmp_path / "c.db")

    assert observation is None
    assert not (tmp_path / "c.db").exists()


@pytest.mark.asyncio
async def test_openai_usage_without_capture_returns_observation_and_writes_nothing(
    tmp_path: Path,
) -> None:
    observation = await observe_openai_usage(
        SPEC_USAGE, "gpt-6.1-terra", tmp_path / "costs.db"
    )

    assert observation is not None
    assert observation["estimated_cost_usd"] == pytest.approx(0.0112)
    assert not (tmp_path / "costs.db").exists()


@pytest.mark.asyncio
async def test_streamed_openai_call_writes_exactly_one_ledger_row(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "costs.db"
    _begin()
    assert attach_cost_metadata(LlmResponse(partial=True)) is None
    assert attach_cost_metadata(LlmResponse(partial=True)) is None

    # The adapter observes usage once, from the final stream event.
    await observe_openai_usage(SPEC_USAGE, "gpt-6.1-terra", ledger_path)

    assert attach_cost_metadata(LlmResponse(partial=True)) is not None
    final = LlmResponse(partial=False)
    assert attach_cost_metadata(final) is not None
    assert final.custom_metadata is not None
    assert final.custom_metadata["blacki.cost"]["cost_kind"] == "estimate"
    assert len(_ledger_rows(ledger_path)) == 1


@pytest.mark.asyncio
async def test_each_model_call_writes_one_row_and_repeated_observation_upserts(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "costs.db"
    small_usage = {"input_tokens": 10, "output_tokens": 2}

    _begin("inv-1")
    await observe_openai_usage(small_usage, "gpt-6.1-terra", ledger_path)
    await observe_openai_usage(small_usage, "gpt-6.1-terra", ledger_path)
    _begin("inv-2")
    await observe_openai_usage(SPEC_USAGE, "gpt-6.1-terra", ledger_path)

    rows = _ledger_rows(ledger_path)
    assert sorted(row["invocation_id"] for row in rows) == ["inv-1", "inv-2"]
    ledger = read_usage_ledger(
        ledger_path,
        selected_since=None,
        selected_until=10**12,
        month_start=0,
        now=10**12,
    )
    assert ledger.cumulative.records == 2
    assert ledger.cumulative.estimated_records == 2


@pytest.mark.asyncio
async def test_openai_estimate_sets_span_cost_attributes(tmp_path: Path) -> None:
    span = MagicMock()
    span.is_recording.return_value = True
    _begin()

    with patch("blacki.observability.costs.trace.get_current_span", return_value=span):
        await observe_openai_usage(SPEC_USAGE, "gpt-6.1-terra", tmp_path / "c.db")

    attributes = {
        call.args[0]: call.args[1] for call in span.set_attribute.call_args_list
    }
    assert attributes == {
        "gen_ai.usage.cost_estimate": pytest.approx(0.0112),
        "gen_ai.cost.source": "catalog_price",
        "gen_ai.cost.kind": "estimate",
    }

    span.reset_mock()
    _begin("inv-unpriced")
    with patch("blacki.observability.costs.trace.get_current_span", return_value=span):
        await observe_openai_usage(
            {"input_tokens": 1}, "gpt-unlisted", tmp_path / "c.db"
        )

    attributes = {
        call.args[0]: call.args[1] for call in span.set_attribute.call_args_list
    }
    assert attributes == {
        "gen_ai.cost.source": "usage_only",
        "gen_ai.cost.kind": "unknown",
    }
