"""Tests for provider-neutral model capability discovery."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import create_autospec

import httpx
import pytest

from blacki.model_capabilities import (
    OPENROUTER_MODELS_BASE_URL,
    OpenRouterModelCapabilitiesResolver,
    normalize_openrouter_model_id,
)


class FakeClock:
    """Deterministic monotonic clock for cache-boundary tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _response(status_code: int, payload: object) -> httpx.Response:
    request = httpx.Request(
        "GET",
        f"{OPENROUTER_MODELS_BASE_URL}/openai/gpt-5.6-luna",
    )
    return httpx.Response(status_code, json=payload, request=request)


def _client(*responses: httpx.Response | BaseException) -> Any:
    client = create_autospec(httpx.AsyncClient, instance=True, spec_set=True)
    client.get.side_effect = list(responses)
    return client


def _payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "openai/gpt-5.6-luna",
        "name": "GPT-5.6 Luna",
        "supported_parameters": ["tools", "reasoning"],
        "reasoning": {
            "supported_efforts": ["max", "high", "high", "medium"],
            "default_effort": "high",
            "default_enabled": True,
            "supports_max_tokens": True,
            "mandatory": False,
        },
    }
    data.update(overrides)
    return {"data": data}


@pytest.mark.parametrize(
    ("model_id", "routed", "expected"),
    [
        ("openrouter/openai/gpt-5.6-luna", False, "openai/gpt-5.6-luna"),
        (" OPENROUTER/openai/gpt-5.6-luna:free ", False, "openai/gpt-5.6-luna:free"),
        ("openai/gpt-5.6-luna", True, "openai/gpt-5.6-luna"),
        ("gpt-5.6-luna", True, None),
        ("openrouter/openrouter/openai/gpt-5.6-luna", False, None),
        ("google/gemini-2.5-flash", False, None),
        ("", True, None),
        (123, True, None),
        ("openrouter/../gpt-5.6-luna", False, None),
    ],
)
def test_normalize_openrouter_model_id(
    model_id: object,
    routed: bool,
    expected: str | None,
) -> None:
    assert (
        normalize_openrouter_model_id(
            model_id,
            openrouter_routed=routed,
        )
        == expected
    )


@pytest.mark.asyncio
async def test_resolve_parses_reasoning_metadata() -> None:
    client = _client(_response(200, _payload()))
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.model_id == "openai/gpt-5.6-luna"
    assert result.name == "GPT-5.6 Luna"
    assert result.reasoning is not None
    assert result.reasoning.supports_effort is True
    assert result.reasoning.supported_efforts == ("max", "high", "medium")
    assert result.reasoning.default_effort == "high"
    assert result.reasoning.default_enabled is True
    assert result.reasoning.supports_max_tokens is True
    assert result.reasoning.mandatory is False

    client.get.assert_awaited_once_with(
        f"{OPENROUTER_MODELS_BASE_URL}/openai/gpt-5.6-luna",
        timeout=3.0,
    )


@pytest.mark.asyncio
async def test_explicit_api_key_is_sent_to_injected_client() -> None:
    client = _client(_response(200, _payload()))
    resolver = OpenRouterModelCapabilitiesResolver(
        client=client,
        api_key="test-key",
    )

    await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    client.get.assert_awaited_once_with(
        f"{OPENROUTER_MODELS_BASE_URL}/openai/gpt-5.6-luna",
        timeout=3.0,
        headers={"Authorization": "Bearer test-key"},
    )


@pytest.mark.asyncio
async def test_get_capabilities_alias_resolves_model() -> None:
    client = _client(_response(200, _payload()))
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.get_capabilities("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.model_id == "openai/gpt-5.6-luna"


@pytest.mark.asyncio
async def test_resolve_accepts_unprefixed_model_when_explicitly_routed() -> None:
    client = _client(_response(200, _payload()))
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openai/gpt-5.6-luna", openrouter_routed=True)

    assert result is not None
    client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_unavailable_model_does_not_make_http_request() -> None:
    client = _client(_response(200, _payload()))
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    assert await resolver.resolve("google/gemini-2.5-flash") is None
    assert await resolver.resolve("gpt-5.6-luna", openrouter_routed=True) is None
    client.get.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"id": "openai/gpt-5.6-luna", "supported_parameters": []}},
        {
            "data": {
                "id": "openai/gpt-5.6-luna",
                "supported_parameters": ["include_reasoning"],
            }
        },
    ],
)
async def test_missing_effort_metadata_is_reasoning_unavailable(
    payload: dict[str, Any],
) -> None:
    client = _client(_response(200, payload))
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.reasoning is None


@pytest.mark.asyncio
async def test_supported_parameters_fallback_does_not_invent_effort_controls() -> None:
    client = _client(
        _response(
            200,
            {
                "data": {
                    "id": "openai/gpt-5.6-luna",
                    "supported_parameters": ["reasoning"],
                }
            },
        )
    )
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.reasoning is not None
    assert result.reasoning.supports_effort is False
    assert result.reasoning.supported_efforts is None
    assert result.reasoning.default_effort is None


@pytest.mark.asyncio
async def test_reasoning_object_without_efforts_hides_effort_selector() -> None:
    client = _client(
        _response(
            200,
            {
                "data": {
                    "id": "openai/gpt-5.6-luna",
                    "reasoning": {
                        "default_enabled": True,
                        "mandatory": True,
                    },
                }
            },
        )
    )
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.reasoning is not None
    assert result.reasoning.supports_effort is False
    assert result.reasoning.supported_efforts is None
    assert result.reasoning.default_enabled is True
    assert result.reasoning.mandatory is True


@pytest.mark.asyncio
async def test_reasoning_metadata_is_tolerant_of_malformed_optional_values() -> None:
    client = _client(
        _response(
            200,
            {
                "data": {
                    "id": "openai/gpt-5.6-luna",
                    "reasoning": {
                        "supported_efforts": [" HIGH ", 4, ""],
                        "default_effort": 4,
                        "default_enabled": "true",
                        "supports_max_tokens": 1,
                        "mandatory": True,
                    },
                }
            },
        )
    )
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.reasoning is not None
    assert result.reasoning.supported_efforts == ("high",)
    assert result.reasoning.default_effort is None
    assert result.reasoning.default_enabled is None
    assert result.reasoning.supports_max_tokens is False
    assert result.reasoning.mandatory is True


@pytest.mark.asyncio
async def test_scalar_supported_efforts_is_ignored() -> None:
    client = _client(
        _response(
            200,
            {
                "data": {
                    "id": "openai/gpt-5.6-luna",
                    "reasoning": {"supported_efforts": 3},
                }
            },
        )
    )
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.reasoning is not None
    assert result.reasoning.supports_effort is False
    assert result.reasoning.supported_efforts is None


@pytest.mark.asyncio
async def test_null_supported_efforts_means_gateway_values() -> None:
    client = _client(
        _response(
            200,
            {
                "data": {
                    "id": "openai/gpt-5.6-luna",
                    "reasoning": {
                        "supported_efforts": None,
                        "default_effort": "none",
                        "mandatory": True,
                    },
                }
            },
        )
    )
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.reasoning is not None
    assert result.reasoning.supports_effort is True
    assert result.reasoning.supported_efforts is None
    assert result.reasoning.default_effort == "none"
    assert result.reasoning.mandatory is True


@pytest.mark.asyncio
async def test_success_is_cached_until_ttl_expires() -> None:
    clock = FakeClock()
    client = _client(_response(200, _payload()), _response(200, _payload(name="New")))
    resolver = OpenRouterModelCapabilitiesResolver(
        client=client,
        cache_ttl_seconds=10,
        clock=clock,
    )

    first = await resolver.resolve("openrouter/openai/gpt-5.6-luna")
    clock.value = 9.999
    second = await resolver.resolve("openrouter/openai/gpt-5.6-luna")
    assert first is second
    client.get.assert_awaited_once()

    clock.value = 10
    refreshed = await resolver.resolve("openrouter/openai/gpt-5.6-luna")
    assert refreshed is not None
    assert refreshed.name == "New"
    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_expired_cache_is_used_when_refresh_times_out() -> None:
    clock = FakeClock()
    client = _client(
        _response(200, _payload()),
        httpx.TimeoutException("timed out"),
    )
    resolver = OpenRouterModelCapabilitiesResolver(
        client=client,
        cache_ttl_seconds=1,
        clock=clock,
    )

    first = await resolver.resolve("openrouter/openai/gpt-5.6-luna")
    clock.value = 2
    stale = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert stale is first
    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_refreshes_share_one_in_flight_request() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    response = _response(200, _payload())

    async def delayed_get(*_args: Any, **_kwargs: Any) -> httpx.Response:
        started.set()
        await release.wait()
        return response

    client = _client()
    client.get.side_effect = delayed_get
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    first_task = asyncio.create_task(resolver.resolve("openrouter/openai/gpt-5.6-luna"))
    await started.wait()
    second_task = asyncio.create_task(
        resolver.resolve("openrouter/openai/gpt-5.6-luna")
    )
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first is second
    client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_failure_without_cache_returns_unavailable() -> None:
    client = _client(httpx.TimeoutException("timed out"))
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    assert await resolver.resolve("openrouter/openai/gpt-5.6-luna") is None


@pytest.mark.asyncio
async def test_malformed_success_response_uses_stale_cache() -> None:
    clock = FakeClock()
    client = _client(_response(200, _payload()), _response(200, {"data": []}))
    resolver = OpenRouterModelCapabilitiesResolver(
        client=client,
        cache_ttl_seconds=1,
        clock=clock,
    )

    first = await resolver.resolve("openrouter/openai/gpt-5.6-luna")
    clock.value = 2
    stale = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert stale is first


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[], {"data": None}, {"data": []}])
async def test_invalid_model_payload_returns_unavailable(payload: object) -> None:
    client = _client(_response(200, payload))
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    assert await resolver.resolve("openrouter/openai/gpt-5.6-luna") is None


@pytest.mark.asyncio
async def test_blank_response_model_id_uses_requested_id() -> None:
    client = _client(
        _response(
            200,
            {"data": {"id": "   ", "name": "Model"}},
        )
    )
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    result = await resolver.resolve("openrouter/openai/gpt-5.6-luna")

    assert result is not None
    assert result.model_id == "openai/gpt-5.6-luna"


@pytest.mark.asyncio
async def test_injected_client_is_not_closed() -> None:
    client = _client()
    resolver = OpenRouterModelCapabilitiesResolver(client=client)

    await resolver.aclose()

    client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_owned_client_is_closed() -> None:
    resolver = OpenRouterModelCapabilitiesResolver()
    client = resolver._client

    await resolver.aclose()

    assert client.is_closed is True


@pytest.mark.parametrize(
    ("cache_ttl", "timeout"),
    [(-1, 3), (1, 0)],
)
def test_invalid_cache_configuration_raises(
    cache_ttl: float,
    timeout: float,
) -> None:
    with pytest.raises(ValueError):
        OpenRouterModelCapabilitiesResolver(
            cache_ttl_seconds=cache_ttl,
            timeout_seconds=timeout,
        )


def test_protocol_shape_is_async_callable() -> None:
    assert callable(OpenRouterModelCapabilitiesResolver.resolve)
