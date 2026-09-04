"""Capture provider-reported LiteLLM usage before ADK reduces the response."""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.llm_response import LlmResponse
from opentelemetry import trace

from .ledger import UsageRecord, default_usage_ledger_path, write_usage_record

logger = logging.getLogger("blacki.llm_costs")

COST_METADATA_KEY = "blacki.cost"


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


def _first_value(value: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        candidate = _read_value(value, key)
        if candidate is not None:
            return candidate
    return None


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


def extract_cost_observation(response: Any, model: str) -> CostObservation | None:
    """Extract raw provider cost and token fields from a LiteLLM response."""
    usage = _read_value(response, "usage")

    reported_cost = _nonnegative_number(_first_value(usage, ("cost", "total_cost")))
    cost_details = _read_value(usage, "cost_details")
    upstream_cost = _nonnegative_number(
        _first_value(
            cost_details,
            ("upstream_inference_cost", "upstream_cost"),
        )
    )
    hidden_params = _read_value(response, "_hidden_params")
    estimated_cost = _nonnegative_number(
        _first_value(hidden_params, ("response_cost", "cost"))
    )
    if reported_cost is not None:
        estimated_cost = None
    input_tokens = _nonnegative_integer(
        _first_value(usage, ("prompt_tokens", "input_tokens"))
    )
    output_tokens = _nonnegative_integer(
        _first_value(usage, ("completion_tokens", "output_tokens"))
    )
    total_tokens = _nonnegative_integer(_first_value(usage, ("total_tokens",)))
    if (
        reported_cost is None
        and upstream_cost is None
        and estimated_cost is None
        and input_tokens is None
        and output_tokens is None
        and total_tokens is None
    ):
        return None

    response_id = _first_value(response, ("id", "response_id", "generation_id"))
    if response_id is not None:
        response_id = str(response_id)[:256]
    cost_kind = (
        "reported"
        if reported_cost is not None
        else "estimated"
        if estimated_cost is not None
        else "upstream_only"
        if upstream_cost is not None
        else "unknown"
    )
    cost_source = (
        "provider_usage"
        if reported_cost is not None or upstream_cost is not None
        else "litellm_response_cost"
    )
    return {
        "model": str(model)[:256],
        "provider_response_id": response_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": reported_cost,
        "upstream_cost_usd": upstream_cost,
        "estimated_cost_usd": estimated_cost,
        "cost_kind": cost_kind,
        "cost_source": cost_source,
        "currency": "USD",
    }


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
    if (
        current is not None
        and current.get("cost_usd") is not None
        and observation.get("cost_usd") is None
    ):
        return capture
    remembered = dict(observation)
    remembered["dedupe_key"] = _dedupe_key(capture, remembered)
    capture.observation = remembered
    return capture


async def _observe_response(
    response: Any,
    model: str,
    ledger_path: Path,
) -> None:
    observation = extract_cost_observation(response, model)
    if observation is None:
        return
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
    )
    try:
        await asyncio.to_thread(write_usage_record, ledger_path, record)
    except (OSError, sqlite3.Error, ValueError):
        logger.warning("Unable to persist model usage to the local cost ledger")


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


class CostAwareLiteLLMClient(LiteLLMClient):
    """Observe LiteLLM responses while preserving ADK's client contract."""

    def __init__(self, ledger_path: Path | str | None = None) -> None:
        super().__init__()
        self.ledger_path = (
            Path(ledger_path) if ledger_path else default_usage_ledger_path()
        )

    async def _observed_stream(
        self,
        response: Any,
        model: str,
    ) -> AsyncIterator[Any]:
        async for chunk in response:
            await _observe_response(chunk, model, self.ledger_path)
            yield chunk

    async def acompletion(
        self,
        model: Any,
        messages: Any,
        tools: Any,
        **kwargs: Any,
    ) -> Any:
        response = await super().acompletion(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        )
        model_name = str(model)
        if kwargs.get("stream"):
            return self._observed_stream(response, model_name)
        await _observe_response(response, model_name, self.ledger_path)
        return response
