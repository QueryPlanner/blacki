"""Capture model usage and cost before ADK reduces the response."""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import time
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from google.adk.models.llm_response import LlmResponse
from opentelemetry import trace

from ..models.catalog import get_model, normalize_model_id
from .ledger import UsageRecord, write_usage_record

logger = logging.getLogger("blacki.llm_costs")

COST_METADATA_KEY = "blacki.cost"
_OPENAI_PROVIDER = "openai"
_TOKENS_PER_PRICE_UNIT = Decimal(1_000_000)
_warned_unpriced_models: set[str] = set()


CostObservation = dict[str, Any]


@dataclass(slots=True)
class _CostCapture:
    user_id: str
    session_id: str
    invocation_id: str
    request_key: str
    observation: CostObservation | None = None


_ACTIVE_CAPTURE: ContextVar[_CostCapture | None] = ContextVar(
    "blacki_active_cost_capture", default=None
)


def _read_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _nonnegative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _nonnegative_integer(value: Any) -> int | None:
    number = _nonnegative_number(value)
    return int(number) if number is not None else None


def _begin_cost_capture(
    *,
    user_id: str | None,
    session_id: str | None,
    invocation_id: str | None,
) -> None:
    """Start a request-local capture associated with ADK identity fields."""
    _ACTIVE_CAPTURE.set(
        _CostCapture(
            user_id=str(user_id or "")[:512],
            session_id=str(session_id or "")[:512],
            invocation_id=str(invocation_id or "")[:512],
            request_key=uuid.uuid4().hex,
        )
    )


def _dedupe_key(capture: _CostCapture, observation: CostObservation) -> str:
    response_id = observation.get("provider_response_id")
    model = str(observation.get("model") or "")
    if response_id:
        return f"provider:{model}:{response_id}"[:1024]
    return f"request:{capture.request_key}"


def _set_span_cost_attributes(observation: CostObservation) -> None:
    span = trace.get_current_span()
    if not span.is_recording():
        return
    cost = observation.get("cost_usd")
    if cost is not None:
        span.set_attribute("gen_ai.usage.cost", cost)
    estimated = observation.get("estimated_cost_usd")
    if estimated is not None:
        span.set_attribute("gen_ai.usage.cost_estimate", estimated)
    upstream = observation.get("upstream_cost_usd")
    if upstream is not None:
        span.set_attribute("gen_ai.cost.upstream_inference_cost", upstream)
    span.set_attribute("gen_ai.cost.source", str(observation["cost_source"]))
    span.set_attribute("gen_ai.cost.kind", str(observation["cost_kind"]))


def _remember_observation(observation: CostObservation) -> _CostCapture | None:
    capture = _ACTIVE_CAPTURE.get()
    if capture is None:
        return None
    current = capture.observation
    remembered = dict(observation)
    if current is not None:
        remembered = current | {
            field: value for field, value in observation.items() if value is not None
        }
        if current.get("cost_usd") is not None and observation.get("cost_usd") is None:
            remembered["cost_kind"] = current["cost_kind"]
            remembered["cost_source"] = current["cost_source"]
    remembered["dedupe_key"] = _dedupe_key(capture, remembered)
    capture.observation = remembered
    return capture


async def _record_observation(observation: CostObservation, ledger_path: Path) -> None:
    """Set span attributes, merge into the active capture, and write its ledger row."""
    _set_span_cost_attributes(observation)
    capture = _remember_observation(observation)
    if capture is None:
        return
    remembered = capture.observation
    if remembered is None:  # pragma: no cover - defensive capture invariant
        return
    record = UsageRecord(
        dedupe_key=str(remembered["dedupe_key"]),
        observed_at=time.time(),
        user_id=capture.user_id,
        session_id=capture.session_id,
        invocation_id=capture.invocation_id,
        model=str(remembered["model"]),
        provider_response_id=remembered.get("provider_response_id"),
        input_tokens=remembered.get("input_tokens"),
        output_tokens=remembered.get("output_tokens"),
        total_tokens=remembered.get("total_tokens"),
        cost_usd=remembered.get("cost_usd"),
        upstream_cost_usd=remembered.get("upstream_cost_usd"),
        estimated_cost_usd=remembered.get("estimated_cost_usd"),
        cost_kind=str(remembered["cost_kind"]),
        cost_source=str(remembered["cost_source"]),
        currency=str(remembered["currency"]),
        cached_tokens=remembered.get("cached_tokens"),
        cache_write_tokens=remembered.get("cache_write_tokens"),
        resolved_model=remembered.get("resolved_model"),
        provider=remembered.get("provider"),
    )
    try:
        await asyncio.to_thread(write_usage_record, ledger_path, record)
    except (OSError, sqlite3.Error, ValueError):
        logger.warning("Unable to persist model usage to the local cost ledger")


def _reset_unpriced_model_warnings() -> None:
    """Forget which models already logged a missing-price warning (tests use this)."""
    _warned_unpriced_models.clear()


def _warn_unpriced_model_once(model: str) -> None:
    key = normalize_model_id(model)[:256]
    if key in _warned_unpriced_models:
        return
    _warned_unpriced_models.add(key)
    logger.warning(
        "No catalog price for model %s; recording tokens without a cost estimate.",
        key,
    )


def _catalog_estimate_usd(
    model: str,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_tokens: int | None,
    cache_write_tokens: int | None,
) -> float | None:
    """Estimate USD from catalog rates, or return None when that is not safe.

    Reasoning tokens are not an input: OpenAI counts them inside output tokens.
    Cache writes get their own rate only when the usage reports them and the
    catalog prices them; otherwise they are priced as ordinary input.
    """
    entry = get_model(model)
    if entry is None or entry.output is None:
        _warn_unpriced_model_once(model)
        return None
    if input_tokens is None or output_tokens is None:
        return None
    write_rate = entry.cache_write
    priced_writes = 0
    if write_rate is not None and cache_write_tokens:
        priced_writes = cache_write_tokens
    cached = cached_tokens or 0
    uncached = input_tokens - cached - priced_writes
    if uncached < 0:
        logger.warning(
            "OpenAI usage counts more cached and cache-write tokens than input "
            "tokens; recording tokens without a cost estimate."
        )
        return None
    cost = (
        Decimal(uncached) * Decimal(str(entry.input))
        + Decimal(cached) * Decimal(str(entry.cached_input))
        + Decimal(priced_writes) * Decimal(str(write_rate or 0))
        + Decimal(output_tokens) * Decimal(str(entry.output))
    )
    return float(cost / _TOKENS_PER_PRICE_UNIT)


def _openai_usage_observation(
    usage: Mapping[str, Any],
    model: str,
) -> CostObservation | None:
    """Build one observation from OpenAI Responses usage, or None without counts."""
    input_details = _read_value(usage, "input_tokens_details")
    input_tokens = _nonnegative_integer(_read_value(usage, "input_tokens"))
    output_tokens = _nonnegative_integer(_read_value(usage, "output_tokens"))
    total_tokens = _nonnegative_integer(_read_value(usage, "total_tokens"))
    cached_tokens = _nonnegative_integer(_read_value(input_details, "cached_tokens"))
    cache_write_tokens = _nonnegative_integer(
        _read_value(input_details, "cache_write_tokens")
    )
    if cache_write_tokens is None:
        cache_write_tokens = _nonnegative_integer(
            _read_value(usage, "cache_write_tokens")
        )
    if all(
        count is None
        for count in (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_tokens,
            cache_write_tokens,
        )
    ):
        return None
    estimated = _catalog_estimate_usd(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    return {
        "model": str(model)[:256],
        "resolved_model": None,
        "provider": _OPENAI_PROVIDER,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "provider_response_id": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": None,
        "upstream_cost_usd": None,
        "estimated_cost_usd": estimated,
        "cost_kind": "estimate" if estimated is not None else "unknown",
        "cost_source": "catalog_price" if estimated is not None else "usage_only",
        "currency": "USD",
    }


async def observe_openai_usage(
    usage: Mapping[str, Any],
    model: str,
    ledger_path: Path,
) -> CostObservation | None:
    """Record one OpenAI Responses call's usage, priced from the model catalog.

    Uses the request-scoped capture, dedupe, span and ledger path, so each model
    call writes one ledger row. Returns the call's observation, or None when the
    usage holds no token counts.
    """
    observation = _openai_usage_observation(usage, model)
    if observation is None:
        return None
    await _record_observation(observation, ledger_path)
    return observation


def _attach_cost_metadata(response: LlmResponse) -> CostObservation | None:
    """Attach the latest safe cost observation to the ADK response event."""
    capture = _ACTIVE_CAPTURE.get()
    if capture is None:
        return None
    observation = capture.observation
    if observation is not None:
        metadata = getattr(response, "custom_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        metadata[COST_METADATA_KEY] = dict(observation)
        response.custom_metadata = metadata
    if not bool(getattr(response, "partial", False)):
        _ACTIVE_CAPTURE.set(None)
    return dict(observation) if observation is not None else None
