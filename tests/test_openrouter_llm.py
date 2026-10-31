"""Integration tests for Blacki's OpenRouter ADK adapter."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from openai import AsyncOpenAI

from blacki.models.openrouter import OpenRouterLlm
from blacki.models.openrouter_request import OpenRouterRequestError
from blacki.observability.ledger import read_usage_ledger
from blacki.observability.lifecycle import attach_cost_metadata, begin_cost_capture

RequestRecord = dict[str, Any]


def _llm(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    tmp_path: Path,
    capture: list[dict[str, Any]] | None = None,
    clients: list[AsyncOpenAI] | None = None,
    model: str = "openrouter/google/gemini-2.5-flash",
) -> OpenRouterLlm:
    transport = httpx.MockTransport(handler)

    def factory(timeout: float, max_retries: int) -> AsyncOpenAI:
        if capture is not None:
            capture.append({"timeout": timeout, "max_retries": max_retries})
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            http_client=httpx.AsyncClient(transport=transport),
            timeout=timeout,
            max_retries=max_retries,
        )
        if clients is not None:
            clients.append(client)
        return client

    return OpenRouterLlm(
        model=model,
        api_key="test-key",
        ledger_path=tmp_path / "costs.db",
        client_factory=factory,
    )


def _handler(
    records: list[RequestRecord],
    response_body: dict[str, Any] | None = None,
    *,
    status_code: int = 200,
    sse: str | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        records.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        if sse is not None:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=sse,
            )
        return httpx.Response(status_code, json=response_body or {})

    return handle


def _request(*, model: str | None = None, config: Any = None) -> LlmRequest:
    return LlmRequest(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="hello")],
            )
        ],
        config=config or types.GenerateContentConfig(),
    )


@pytest.mark.asyncio
async def test_non_streaming_request_and_response_mapping(tmp_path: Path) -> None:
    records: list[RequestRecord] = []
    response_body = {
        "id": "gen-1",
        "model": "google/gemini-2.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    llm = _llm(_handler(records, response_body), tmp_path=tmp_path)
    request = _request(
        config=types.GenerateContentConfig(
            http_options=types.HttpOptions(extra_body={"reasoning": {"effort": "low"}})
        )
    )

    responses = [response async for response in llm.generate_content_async(request)]

    assert len(responses) == 1
    assert responses[0].content is not None
    assert responses[0].content.parts is not None
    assert responses[0].content.parts[0].text == "hi"
    assert responses[0].usage_metadata is not None
    assert responses[0].usage_metadata.total_token_count == 4

    assert len(records) == 1
    assert records[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert records[0]["headers"]["authorization"] == "Bearer test-key"
    assert records[0]["body"] == {
        "model": "google/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "reasoning": {"effort": "low"},
    }


@pytest.mark.asyncio
async def test_per_turn_model_override_and_prefix_are_stripped(tmp_path: Path) -> None:
    records: list[RequestRecord] = []
    llm = _llm(
        _handler(
            records,
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
        tmp_path=tmp_path,
    )

    request = _request(model="openrouter/openai/gpt-4o")
    responses = [response async for response in llm.generate_content_async(request)]

    assert records[0]["body"]["model"] == "openai/gpt-4o"
    assert responses[0].content is not None


@pytest.mark.asyncio
async def test_streaming_yields_partials_and_final_aggregate(tmp_path: Path) -> None:
    records: list[RequestRecord] = []
    sse = (
        ": OPENROUTER PROCESSING\n\n"
        'data: {"id":"gen-1","model":"google/gemini-2.5-flash",'
        '"choices":[{"index":0,"delta":{"reasoning":"think"},'
        '"finish_reason":null}]}\n\n'
        'data: {"id":"gen-1","choices":[{"index":0,'
        '"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
        'data: {"id":"gen-1","choices":[{"index":0,'
        '"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"gen-1","choices":[{"index":0,'
        '"delta":{"content":""},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}\n\n'
        "data: [DONE]\n\n"
    )
    clients: list[AsyncOpenAI] = []
    llm = _llm(
        _handler(records, sse=sse),
        tmp_path=tmp_path,
        clients=clients,
    )

    responses = [
        response
        async for response in llm.generate_content_async(_request(), stream=True)
    ]

    assert len(responses) == 4
    assert responses[0].partial is True
    assert responses[0].content is not None
    assert responses[0].content.parts is not None
    assert responses[0].content.parts[0].thought is True
    assert responses[1].partial is True
    assert responses[1].content is not None
    assert responses[1].content.parts is not None
    assert responses[1].content.parts[0].text == "Hel"
    assert responses[2].partial is True
    assert responses[2].content is not None
    assert responses[2].content.parts is not None
    assert responses[2].content.parts[0].text == "lo"
    assert responses[3].partial is False
    assert responses[3].content is not None
    assert responses[3].content.parts is not None
    assert responses[3].content.parts[0].text == "think"
    assert responses[3].content.parts[1].text == "Hello"
    assert responses[3].usage_metadata is not None
    assert responses[3].usage_metadata.total_token_count == 4

    assert records[0]["body"]["stream"] is True
    assert records[0]["body"]["stream_options"] == {"include_usage": True}
    assert clients[0].is_closed() is True


@pytest.mark.asyncio
async def test_http_error_propagates_and_client_is_closed(tmp_path: Path) -> None:
    records: list[RequestRecord] = []
    clients: list[AsyncOpenAI] = []
    llm = _llm(
        _handler(records, {"error": {"message": "bad key"}}, status_code=401),
        tmp_path=tmp_path,
        clients=clients,
    )

    with pytest.raises(openai.AuthenticationError):
        [response async for response in llm.generate_content_async(_request())]

    assert clients[0].is_closed() is True


@pytest.mark.asyncio
async def test_unsupported_config_fails_before_network(tmp_path: Path) -> None:
    records: list[RequestRecord] = []
    llm = _llm(_handler(records, {}), tmp_path=tmp_path)

    with pytest.raises(OpenRouterRequestError, match="response_mime_type"):
        [
            response
            async for response in llm.generate_content_async(
                _request(
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            )
        ]

    assert records == []


@pytest.mark.asyncio
async def test_cost_observation_writes_prefixed_model_to_ledger(
    tmp_path: Path,
) -> None:
    llm = _llm(
        _handler(
            [],
            {
                "id": "gen-cost",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.01,
                    "cost_details": {"upstream_inference_cost": 0.009},
                },
            },
        ),
        tmp_path=tmp_path,
    )
    begin_cost_capture(
        user_id="user-1",
        session_id="session-1",
        invocation_id="inv-1",
    )

    [response async for response in llm.generate_content_async(_request())]
    metadata = attach_cost_metadata(LlmResponse())

    assert metadata is not None
    assert metadata["cost_usd"] == 0.01
    assert metadata["cost_source"] == "provider_usage"

    ledger = read_usage_ledger(
        tmp_path / "costs.db",
        selected_since=None,
        selected_until=10**12,
        month_start=0,
        now=10**12,
    )
    assert ledger.cumulative.records == 1
    assert ledger.users["user-1"].cost_nano_usd is not None


@pytest.mark.asyncio
async def test_streaming_cost_observation_writes_one_ledger_row(
    tmp_path: Path,
) -> None:
    sse = (
        'data: {"id":"gen-stream","choices":[{"index":0,'
        '"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"gen-stream","choices":[{"index":0,'
        '"delta":{"content":""},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,'
        '"cost":0.002}}\n\n'
        "data: [DONE]\n\n"
    )
    llm = _llm(_handler([], sse=sse), tmp_path=tmp_path)
    begin_cost_capture(
        user_id="user-stream",
        session_id="session-stream",
        invocation_id="inv-stream",
    )

    [response async for response in llm.generate_content_async(_request(), stream=True)]
    metadata = attach_cost_metadata(LlmResponse())

    assert metadata is not None
    assert metadata["cost_usd"] == 0.002

    ledger = read_usage_ledger(
        tmp_path / "costs.db",
        selected_since=None,
        selected_until=10**12,
        month_start=0,
        now=10**12,
    )
    assert ledger.cumulative.records == 1


def test_model_copy_repr_and_dump_do_not_expose_credentials(tmp_path: Path) -> None:
    llm = _llm(lambda request: httpx.Response(200, json={}), tmp_path=tmp_path)

    copied = copy.deepcopy(llm)
    model_copied = llm.model_copy(deep=True)

    for candidate in (llm, copied, model_copied):
        assert "test-key" not in repr(candidate)
        assert "test-key" not in str(candidate.model_dump())
    assert copied.model == llm.model
    assert model_copied.model == llm.model


def test_constructor_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenRouterLlm(model="openai/gpt-4o", api_key="  ")
    with pytest.raises(ValueError, match="timeout"):
        OpenRouterLlm(
            model="openai/gpt-4o",
            api_key="key",
            timeout_seconds=0,
            ledger_path=tmp_path / "costs.db",
        )
    with pytest.raises(ValueError, match="max_retries"):
        OpenRouterLlm(
            model="openai/gpt-4o",
            api_key="key",
            max_retries=-1,
            ledger_path=tmp_path / "costs.db",
        )


@pytest.mark.asyncio
async def test_default_client_construction(tmp_path: Path) -> None:
    llm = OpenRouterLlm(
        model="openai/gpt-4o",
        api_key="key",
        ledger_path=str(tmp_path / "costs.db"),
        base_url="https://openrouter.ai/api/v1/",
    )

    client = llm._create_client(5.0, 0)
    try:
        assert isinstance(client, AsyncOpenAI)
        assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_per_request_timeout_and_retry_options_are_forwarded(
    tmp_path: Path,
) -> None:
    captures: list[dict[str, Any]] = []
    llm = _llm(
        _handler(
            [],
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
        tmp_path=tmp_path,
        capture=captures,
    )

    request = _request(
        config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                timeout=5_000,
                retry_options=types.HttpRetryOptions(attempts=4),
            )
        )
    )
    [response async for response in llm.generate_content_async(request)]

    assert captures == [{"timeout": 5.0, "max_retries": 3}]


@pytest.mark.asyncio
async def test_default_timeout_and_retries_are_used_without_http_options(
    tmp_path: Path,
) -> None:
    captures: list[dict[str, Any]] = []
    llm = _llm(
        _handler(
            [],
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
        tmp_path=tmp_path,
        capture=captures,
    )

    [response async for response in llm.generate_content_async(_request())]

    assert captures == [{"timeout": 120.0, "max_retries": 2}]
