"""Tests for the OpenAI Responses ADK adapter.

Only the HTTP boundary is replaced. An ``httpx.MockTransport`` answers the
SDK's requests, while request building, response translation, streaming, the
reasoning retry and error mapping all run for real.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from openai import AsyncOpenAI

from blacki.models.openai_items import StoredItem, item_metadata, stored_item
from blacki.models.openai_request import OpenAIRequestError
from blacki.models.openai_responses import (
    DEFAULT_BASE_URL,
    SESSION_LABEL_KEY,
    OpenAIResponsesLlm,
)
from blacki.observability.lifecycle import attach_cost_metadata, begin_cost_capture
from blacki.utils.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderPermissionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
)

MODEL = "gpt-6.1-terra"
DATED_MODEL = "gpt-6.1-terra-2026-06-01"
KEY = "test-key"
REQUEST_ID = "req_test_0001"
CALL_ID = "call_AbC123xyz"
PROMPT_MARKER = "PROMPT-SECRET-TEXT"
ENCRYPTED = "ENC_fixture_reasoning_blob"
REASONING_SUMMARY = "Checking the meal log first."
MESSAGE_TEXT = "You logged two meals."
TOOL_OUTPUT = '{"meals": []}'
ADAPTER_PROMPT = "Handle the requests as specified in the System Instruction."
LOGGER_NAME = "blacki.models.openai_responses"

REASONING_ITEM: dict[str, Any] = {
    "id": "rs_0a1b2c3d",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": REASONING_SUMMARY}],
    "encrypted_content": ENCRYPTED,
}

FUNCTION_CALL_ITEM: dict[str, Any] = {
    "id": "fc_7g8h9i0j",
    "type": "function_call",
    "call_id": CALL_ID,
    "name": "list_meals",
    "arguments": '{"date": "2026-11-12"}',
    "status": "completed",
}

MESSAGE_ITEM: dict[str, Any] = {
    "id": "msg_4e5f6a7b",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": MESSAGE_TEXT, "annotations": []}],
}

USAGE: dict[str, Any] = {
    "input_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 1024},
    "output_tokens": 340,
    "output_tokens_details": {"reasoning_tokens": 256},
    "total_tokens": 1540,
}


def _response(
    output: list[dict[str, Any]],
    *,
    status: str = "completed",
    incomplete_reason: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp_0123",
        "object": "response",
        "created_at": 1_790_000_000,
        "model": DATED_MODEL,
        "status": status,
        "output": output,
        "usage": USAGE,
        "error": error,
        "incomplete_details": (
            {"reason": incomplete_reason} if incomplete_reason else None
        ),
        "store": False,
    }


def _event(event_type: str, sequence: int, **fields: Any) -> dict[str, Any]:
    return {"type": event_type, "sequence_number": sequence, **fields}


def _stream_turn(final: dict[str, Any]) -> list[dict[str, Any]]:
    """A turn that streams a reasoning summary, answer text and a tool call."""

    return [
        _event("response.created", 0, response=_response([], status="in_progress")),
        _event(
            "response.reasoning_summary_text.delta",
            1,
            item_id=REASONING_ITEM["id"],
            output_index=0,
            summary_index=0,
            delta="Checking ",
        ),
        _event(
            "response.reasoning_summary_text.delta",
            2,
            item_id=REASONING_ITEM["id"],
            output_index=0,
            summary_index=0,
            delta="the meal log first.",
        ),
        _event("response.output_item.done", 3, output_index=0, item=REASONING_ITEM),
        _event(
            "response.output_text.delta",
            4,
            item_id=MESSAGE_ITEM["id"],
            output_index=1,
            content_index=0,
            delta="You logged ",
        ),
        _event(
            "response.output_text.delta",
            5,
            item_id=MESSAGE_ITEM["id"],
            output_index=1,
            content_index=0,
            delta="two meals.",
        ),
        _event("response.output_item.done", 6, output_index=1, item=MESSAGE_ITEM),
        _event("response.output_item.done", 7, output_index=2, item=FUNCTION_CALL_ITEM),
        _event("response.completed", 8, response=final),
    ]


def _json_reply(body: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body, headers={"x-request-id": REQUEST_ID})


def _error_reply(
    status: int,
    *,
    code: str | None,
    param: str | None = None,
    message: str = "Request failed.",
) -> httpx.Response:
    body = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code,
        }
    }
    return _json_reply(body, status)


def _sse_reply(events: list[dict[str, Any]]) -> httpx.Response:
    text = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)


class _TrackedBody(httpx.AsyncByteStream):
    """A response body that records whether the SDK closed it."""

    def __init__(self, text: str) -> None:
        self._data = text.encode()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._data

    async def aclose(self) -> None:
        self.closed = True


class _Server:
    """Answers HTTP requests in order and records each one sent.

    A reply that is an exception is raised instead, as a transport failure.
    """

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self._replies = list(replies)
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _Transport(httpx.AsyncBaseTransport):
    """Passes requests to a ``_Server`` and counts how often the client closes."""

    def __init__(self, server: _Server) -> None:
        self._mock = httpx.MockTransport(server)
        self.closes = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._mock.handle_async_request(request)

    async def aclose(self) -> None:
        self.closes += 1
        await self._mock.aclose()


def _llm(
    server: _Server,
    *,
    transports: list[_Transport] | None = None,
    captures: list[tuple[float, int]] | None = None,
    max_retries: int = 0,
    model: str = MODEL,
    ledger_path: Path | None = None,
) -> OpenAIResponsesLlm:
    def factory(timeout: float, retries: int) -> AsyncOpenAI:
        if captures is not None:
            captures.append((timeout, retries))
        transport = _Transport(server)
        if transports is not None:
            transports.append(transport)
        return AsyncOpenAI(
            api_key=KEY,
            base_url=DEFAULT_BASE_URL,
            http_client=httpx.AsyncClient(transport=transport),
            timeout=timeout,
            max_retries=retries,
        )

    return OpenAIResponsesLlm(
        model,
        api_key=KEY,
        max_retries=max_retries,
        client_factory=factory,
        ledger_path=ledger_path,
    )


def _user_text(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def _user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _tool_result() -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=CALL_ID,
                    name="list_meals",
                    response={"meals": []},
                )
            )
        ],
    )


def _history() -> list[types.Content]:
    """A turn where the model reasoned, called a tool, and the tool answered.

    The model parts carry the items a real response stored, so the request
    builder replays them.
    """

    return [
        _user_text("Log my lunch"),
        types.Content(
            role="model",
            parts=[
                types.Part(
                    text=REASONING_SUMMARY,
                    thought=True,
                    part_metadata=item_metadata(MODEL, REASONING_ITEM),
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id=CALL_ID,
                        name="list_meals",
                        args={"date": "2026-11-12"},
                    ),
                    part_metadata=item_metadata(MODEL, FUNCTION_CALL_ITEM),
                ),
            ],
        ),
        _tool_result(),
    ]


def _request(
    contents: list[types.Content] | None = None,
    *,
    model: str | None = None,
    config: types.GenerateContentConfig | None = None,
) -> LlmRequest:
    return LlmRequest(
        model=model,
        contents=[_user_text("hello")] if contents is None else contents,
        config=types.GenerateContentConfig() if config is None else config,
    )


async def _collect(
    llm: OpenAIResponsesLlm,
    request: LlmRequest,
    *,
    stream: bool = False,
) -> list[LlmResponse]:
    return [
        response
        async for response in llm.generate_content_async(request, stream=stream)
    ]


def _reasoning_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in body["input"] if item.get("type") == "reasoning"]


def test_empty_api_key_is_rejected() -> None:
    for key in ("", "   "):
        with pytest.raises(ValueError, match=re.escape("OpenAI API key is required.")):
            OpenAIResponsesLlm(MODEL, api_key=key)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "OpenAI timeout must be positive."),
        ({"max_retries": -1}, "OpenAI max_retries must be non-negative."),
    ],
)
def test_invalid_timeout_or_retries_are_rejected(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        OpenAIResponsesLlm(MODEL, api_key=KEY, **kwargs)


def test_credentials_stay_out_of_repr_dump_and_copies() -> None:
    llm = OpenAIResponsesLlm(MODEL, api_key=KEY)

    for candidate in (llm, copy.deepcopy(llm), llm.model_copy(deep=True)):
        assert KEY not in repr(candidate)
        assert KEY not in str(candidate.model_dump())
        assert candidate.model == MODEL


@pytest.mark.asyncio
async def test_default_client_targets_openai_and_strips_a_trailing_slash() -> None:
    default = OpenAIResponsesLlm(MODEL, api_key=KEY)
    custom = OpenAIResponsesLlm(
        MODEL, api_key=KEY, base_url="https://proxy.example.test/v1/"
    )

    for llm, expected in (
        (default, DEFAULT_BASE_URL),
        (custom, "https://proxy.example.test/v1"),
    ):
        client = llm._create_client(5.0, 0)
        try:
            assert isinstance(client, AsyncOpenAI)
            assert str(client.base_url).rstrip("/") == expected
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_non_streamed_turn_is_sent_statelessly_and_mapped_to_parts() -> None:
    server = _Server(_json_reply(_response([REASONING_ITEM, FUNCTION_CALL_ITEM])))
    config = types.GenerateContentConfig(
        system_instruction="Be brief.",
        http_options=types.HttpOptions(extra_body={"reasoning": {"effort": "high"}}),
    )

    [response] = await _collect(
        _llm(server), _request([_user_text("Log my lunch")], config=config)
    )

    assert str(server.requests[0].url) == f"{DEFAULT_BASE_URL}/responses"
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert server.bodies[0] == {
        "model": MODEL,
        "input": [_user_message("Log my lunch")],
        "instructions": "Be brief.",
        "reasoning": {"effort": "high", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "stream": False,
    }

    assert response.partial is False
    assert response.model_version == DATED_MODEL
    assert response.content is not None
    assert response.content.parts is not None
    thought, call = response.content.parts
    assert thought.thought is True
    assert thought.text == REASONING_SUMMARY
    assert call.function_call == types.FunctionCall(
        id=CALL_ID, name="list_meals", args={"date": "2026-11-12"}
    )
    assert stored_item(thought) == StoredItem(model=MODEL, item=REASONING_ITEM)
    assert stored_item(call) == StoredItem(model=MODEL, item=FUNCTION_CALL_ITEM)

    usage = response.usage_metadata
    assert usage is not None
    assert usage.prompt_token_count == 1200
    assert usage.cached_content_token_count == 1024
    assert usage.candidates_token_count == 340
    assert usage.thoughts_token_count == 256


@pytest.mark.asyncio
async def test_streamed_turn_yields_partials_then_one_final_equal_to_non_streamed() -> (
    None
):
    final = _response([REASONING_ITEM, MESSAGE_ITEM, FUNCTION_CALL_ITEM])
    streamed = await _collect(
        _llm(_Server(_sse_reply(_stream_turn(final)))), _request(), stream=True
    )
    plain = await _collect(_llm(_Server(_json_reply(final))), _request())

    *partials, last = streamed
    assert [response.partial for response in partials] == [True] * 4
    texts = [
        (bool(response.content.parts[0].thought), response.content.parts[0].text)
        for response in partials
        if response.content is not None and response.content.parts is not None
    ]
    assert texts == [
        (True, "Checking "),
        (True, "the meal log first."),
        (False, "You logged "),
        (False, "two meals."),
    ]
    assert last.partial is False
    assert last.content == plain[0].content
    assert last.usage_metadata == plain[0].usage_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "finish"),
    [
        pytest.param("max_output_tokens", types.FinishReason.MAX_TOKENS, id="tokens"),
        pytest.param("content_filter", types.FinishReason.SAFETY, id="filter"),
    ],
)
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_incomplete_turn_yields_exactly_one_final_with_its_finish_reason(
    reason: str, finish: types.FinishReason, stream: bool
) -> None:
    body = _response([MESSAGE_ITEM], status="incomplete", incomplete_reason=reason)
    reply = (
        _sse_reply(
            [
                _event(
                    "response.output_text.delta",
                    1,
                    item_id=MESSAGE_ITEM["id"],
                    output_index=0,
                    content_index=0,
                    delta="Partial",
                ),
                _event("response.incomplete", 2, response=body),
            ]
        )
        if stream
        else _json_reply(body)
    )

    responses = await _collect(_llm(_Server(reply)), _request(), stream=stream)

    assert [response.partial for response in responses] == (
        [True, False] if stream else [False]
    )
    final = responses[-1]
    assert final.finish_reason == finish
    assert final.error_code == finish


@pytest.mark.asyncio
async def test_refusal_is_returned_as_visible_text_without_a_retry() -> None:
    refusal = {
        "id": "msg_refusal",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
    }
    server = _Server(_json_reply(_response([refusal])))

    [response] = await _collect(_llm(server), _request())

    assert response.content is not None
    assert response.content.parts is not None
    assert response.content.parts[0].text == "I can't help with that."
    assert len(server.bodies) == 1


@pytest.mark.asyncio
async def test_failed_response_status_raises_a_code_only_provider_error() -> None:
    body = _response(
        [],
        status="failed",
        error={"code": "server_error", "message": f"Internal {PROMPT_MARKER}"},
    )

    with pytest.raises(ProviderError) as info:
        await _collect(_llm(_Server(_json_reply(body))), _request())

    assert info.value.code == "server_error"
    assert info.value.status is None
    assert PROMPT_MARKER not in str(info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "partials", "code"),
    [
        pytest.param(
            [
                _event(
                    "response.output_text.delta",
                    1,
                    item_id=MESSAGE_ITEM["id"],
                    output_index=0,
                    content_index=0,
                    delta="Hel",
                ),
                _event(
                    "response.failed",
                    2,
                    response=_response(
                        [],
                        status="failed",
                        error={
                            "code": "server_error",
                            "message": f"Internal {PROMPT_MARKER}",
                        },
                    ),
                ),
            ],
            ["Hel"],
            "server_error",
            id="failed-event",
        ),
        pytest.param(
            [
                _event(
                    "error",
                    1,
                    code="rate_limit_exceeded",
                    message=f"Slow down {PROMPT_MARKER}",
                    param=None,
                )
            ],
            [],
            "rate_limit_exceeded",
            id="error-event",
        ),
        pytest.param(
            [
                _event(
                    "response.output_text.delta",
                    1,
                    item_id=MESSAGE_ITEM["id"],
                    output_index=0,
                    content_index=0,
                    delta="Hi",
                )
            ],
            ["Hi"],
            None,
            id="no-terminal-event",
        ),
    ],
)
async def test_stream_failures_raise_after_the_partials_already_yielded(
    events: list[dict[str, Any]], partials: list[str], code: str | None
) -> None:
    llm = _llm(_Server(_sse_reply(events)))
    received: list[str | None] = []

    with pytest.raises(ProviderError) as info:
        async for response in llm.generate_content_async(_request(), stream=True):
            assert response.content is not None
            assert response.content.parts is not None
            received.append(response.content.parts[0].text)

    assert received == partials
    assert info.value.code == code
    assert info.value.status is None
    assert PROMPT_MARKER not in str(info.value)


@pytest.mark.asyncio
async def test_sdk_error_payload_in_a_stream_maps_to_a_code_only_error() -> None:
    payload = {"error": {"message": PROMPT_MARKER, "code": "server_error"}}
    reply = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"data: {json.dumps(payload)}\n\n",
    )

    with pytest.raises(ProviderError) as info:
        await _collect(_llm(_Server(reply)), _request(), stream=True)

    assert type(info.value) is ProviderError
    assert info.value.code == "server_error"
    assert PROMPT_MARKER not in str(info.value)


@pytest.mark.asyncio
async def test_stopping_a_stream_early_closes_the_response_and_client() -> None:
    transports: list[_Transport] = []
    body = _TrackedBody(
        "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
            for event in _stream_turn(_response([MESSAGE_ITEM]))
        )
    )
    server = _Server(
        httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    )
    llm = _llm(server, transports=transports)

    responses = llm.generate_content_async(_request(), stream=True)
    first = await anext(responses)
    await responses.aclose()

    assert first.partial is True
    assert body.closed is True
    assert [transport.closes for transport in transports] == [1]


@pytest.mark.asyncio
async def test_stored_reasoning_is_replayed_unchanged_on_the_next_call() -> None:
    server = _Server(
        _json_reply(_response([REASONING_ITEM, FUNCTION_CALL_ITEM])),
        _json_reply(_response([MESSAGE_ITEM])),
    )
    llm = _llm(server)

    [first] = await _collect(llm, _request([_user_text("Log my lunch")]))
    assert first.content is not None
    await _collect(
        llm, _request([_user_text("Log my lunch"), first.content, _tool_result()])
    )

    assert server.bodies[1]["input"] == [
        _user_message("Log my lunch"),
        REASONING_ITEM,
        FUNCTION_CALL_ITEM,
        {"type": "function_call_output", "call_id": CALL_ID, "output": TOOL_OUTPUT},
    ]


@pytest.mark.asyncio
async def test_rejected_reasoning_is_retried_once_then_skipped_by_later_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    server = _Server(
        _error_reply(
            400,
            code="invalid_encrypted_content",
            message=f"Encrypted content is invalid. {PROMPT_MARKER}",
        ),
        _json_reply(_response([MESSAGE_ITEM])),
        _json_reply(_response([MESSAGE_ITEM])),
    )
    llm = _llm(server)

    [first] = await _collect(llm, _request(_history()))
    [second] = await _collect(llm, _request(_history()))

    assert first.content is not None
    assert first.content.parts is not None
    assert first.content.parts[0].text == MESSAGE_TEXT
    assert second.content is not None
    assert len(server.bodies) == 3
    assert _reasoning_items(server.bodies[0]) == [REASONING_ITEM]
    assert _reasoning_items(server.bodies[1]) == []
    assert _reasoning_items(server.bodies[2]) == []
    assert [item.get("type") for item in server.bodies[2]["input"]] == [
        None,
        "function_call",
        "function_call_output",
    ]
    assert "code=invalid_encrypted_content" in caplog.text
    assert PROMPT_MARKER not in caplog.text
    assert ENCRYPTED not in caplog.text
    assert REASONING_SUMMARY not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "param"),
    [
        pytest.param("invalid_encrypted_content", None, id="code"),
        pytest.param("invalid_prompt", "input[1]", id="reasoning-item"),
        pytest.param(
            "invalid_prompt", "input[1].encrypted_content", id="reasoning-field"
        ),
    ],
)
async def test_reasoning_rejection_is_retried_when_named_by_code_or_input_pointer(
    code: str, param: str | None
) -> None:
    server = _Server(
        _error_reply(400, code=code, param=param),
        _json_reply(_response([MESSAGE_ITEM])),
    )

    [response] = await _collect(_llm(server), _request(_history()))

    assert response.content is not None
    assert response.content.parts is not None
    assert response.content.parts[0].text == MESSAGE_TEXT
    assert len(server.bodies) == 2
    assert _reasoning_items(server.bodies[1]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "param"),
    [
        pytest.param("invalid_prompt", "input[0]", id="user-message"),
        pytest.param("invalid_prompt", "input[2]", id="function-call"),
        pytest.param("invalid_prompt", "input[99]", id="out-of-range"),
        pytest.param("invalid_prompt", "instructions", id="not-an-input-item"),
        pytest.param("invalid_prompt", None, id="no-param"),
    ],
)
async def test_bad_request_that_does_not_name_a_reasoning_item_is_raised_once(
    code: str, param: str | None
) -> None:
    server = _Server(_error_reply(400, code=code, param=param))

    with pytest.raises(ProviderRequestError) as info:
        await _collect(_llm(server), _request(_history()))

    assert len(server.bodies) == 1
    assert info.value.status == 400
    assert info.value.code == code


@pytest.mark.asyncio
async def test_invalid_encrypted_content_without_reasoning_items_is_not_retried() -> (
    None
):
    server = _Server(_error_reply(400, code="invalid_encrypted_content"))

    with pytest.raises(ProviderRequestError) as info:
        await _collect(_llm(server), _request())

    assert len(server.bodies) == 1
    assert info.value.code == "invalid_encrypted_content"


@pytest.mark.asyncio
async def test_streamed_call_retries_without_reasoning_when_creation_is_rejected() -> (
    None
):
    server = _Server(
        _error_reply(400, code="invalid_encrypted_content"),
        _sse_reply(_stream_turn(_response([MESSAGE_ITEM]))),
    )

    responses = await _collect(_llm(server), _request(_history()), stream=True)

    assert responses[-1].partial is False
    assert responses[-1].content is not None
    assert responses[-1].content.parts is not None
    assert responses[-1].content.parts[0].text == MESSAGE_TEXT
    assert _reasoning_items(server.bodies[0]) == [REASONING_ITEM]
    assert _reasoning_items(server.bodies[1]) == []
    assert server.bodies[1]["stream"] is True


@pytest.mark.asyncio
async def test_failed_retry_raises_and_leaves_reasoning_enabled_for_later_calls() -> (
    None
):
    server = _Server(
        _error_reply(400, code="invalid_encrypted_content"),
        _error_reply(400, code="invalid_prompt"),
        _json_reply(_response([MESSAGE_ITEM])),
    )
    llm = _llm(server)

    with pytest.raises(ProviderRequestError) as info:
        await _collect(llm, _request(_history()))
    assert info.value.code == "invalid_prompt"
    assert len(server.bodies) == 2

    await _collect(llm, _request(_history()))
    assert len(server.bodies) == 3
    assert _reasoning_items(server.bodies[2]) == [REASONING_ITEM]


@pytest.mark.asyncio
async def test_a_new_adapter_instance_replays_reasoning_again() -> None:
    server = _Server(
        _error_reply(400, code="invalid_encrypted_content"),
        _json_reply(_response([MESSAGE_ITEM])),
        _json_reply(_response([MESSAGE_ITEM])),
    )
    await _collect(_llm(server), _request(_history()))

    # A new adapter stands in for a restarted process, which clears the fallback.
    await _collect(_llm(server), _request(_history()))

    assert _reasoning_items(server.bodies[2]) == [REASONING_ITEM]


def _labelled_request(
    label: str, contents: list[types.Content] | None = None
) -> LlmRequest:
    """A request carrying the session label that the override plugin sets."""

    return _request(
        contents,
        config=types.GenerateContentConfig(labels={SESSION_LABEL_KEY: label}),
    )


@pytest.mark.asyncio
async def test_rejection_downgrades_only_the_session_that_hit_it() -> None:
    server = _Server(
        _error_reply(400, code="invalid_encrypted_content"),
        _json_reply(_response([MESSAGE_ITEM])),
        _json_reply(_response([MESSAGE_ITEM])),
        _json_reply(_response([MESSAGE_ITEM])),
    )
    llm = _llm(server)

    await _collect(llm, _labelled_request("session-a", _history()))
    await _collect(llm, _labelled_request("session-a", _history()))
    await _collect(llm, _labelled_request("session-b", _history()))

    assert len(server.bodies) == 4
    assert _reasoning_items(server.bodies[0]) == [REASONING_ITEM]
    assert _reasoning_items(server.bodies[1]) == []
    assert _reasoning_items(server.bodies[2]) == []
    assert _reasoning_items(server.bodies[3]) == [REASONING_ITEM]


@pytest.mark.asyncio
async def test_requests_without_a_label_share_one_downgrade_per_model() -> None:
    server = _Server(
        _error_reply(400, code="invalid_encrypted_content"),
        _json_reply(_response([MESSAGE_ITEM])),
        _json_reply(_response([MESSAGE_ITEM])),
        _json_reply(_response([MESSAGE_ITEM])),
    )
    llm = _llm(server)

    await _collect(llm, _request(_history()))
    await _collect(llm, _request(_history()))
    await _collect(llm, _labelled_request("session-a", _history()))

    assert len(server.bodies) == 4
    assert _reasoning_items(server.bodies[0]) == [REASONING_ITEM]
    assert _reasoning_items(server.bodies[1]) == []
    assert _reasoning_items(server.bodies[2]) == []
    assert _reasoning_items(server.bodies[3]) == [REASONING_ITEM]


@pytest.mark.asyncio
async def test_session_label_is_never_sent_to_openai() -> None:
    server = _Server(_json_reply(_response([MESSAGE_ITEM])))

    await _collect(_llm(server), _labelled_request("label-value-7f3a", _history()))

    assert "labels" not in server.bodies[0]
    assert "label-value-7f3a" not in json.dumps(server.bodies[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        pytest.param(401, ProviderAuthenticationError, id="401"),
        pytest.param(403, ProviderPermissionError, id="403"),
        pytest.param(404, ProviderRequestError, id="404"),
        pytest.param(429, ProviderRateLimitError, id="429"),
        pytest.param(500, ProviderServerError, id="500"),
    ],
)
async def test_http_errors_map_to_typed_errors_without_key_or_prompt_text(
    status: int, error_type: type[ProviderError]
) -> None:
    code = "invalid_api_key" if status == 401 else f"status_{status}"
    message = f"Incorrect API key provided: {KEY}. Prompt: {PROMPT_MARKER}"
    server = _Server(_error_reply(status, code=code, message=message))

    with pytest.raises(error_type) as info:
        await _collect(_llm(server), _request([_user_text(PROMPT_MARKER)]))

    error = info.value
    assert error.provider == "openai"
    assert error.status == status
    assert error.code == code
    assert error.request_id == REQUEST_ID
    assert str(error).startswith("openai ")
    assert f"HTTP {status}" in str(error)
    assert REQUEST_ID in str(error)
    assert KEY not in str(error)
    assert PROMPT_MARKER not in str(error)


@pytest.mark.asyncio
async def test_unmapped_status_is_a_base_provider_error() -> None:
    server = _Server(httpx.Response(302, headers={"x-request-id": REQUEST_ID}))

    with pytest.raises(ProviderError) as info:
        await _collect(_llm(server), _request())

    assert type(info.value) is ProviderError
    assert info.value.status == 302


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(httpx.ConnectError("connection refused"), id="connect"),
        pytest.param(httpx.ReadTimeout("no response in time"), id="timeout"),
    ],
)
async def test_connection_failures_map_to_server_error_without_a_status(
    failure: httpx.HTTPError,
) -> None:
    with pytest.raises(ProviderServerError) as info:
        await _collect(_llm(_Server(failure)), _request())

    assert info.value.status is None
    assert info.value.code is None
    assert info.value.request_id is None
    assert str(info.value) == "openai server or network error."


@pytest.mark.asyncio
async def test_unsupported_configuration_fails_before_any_request() -> None:
    server = _Server()
    config = types.GenerateContentConfig(response_mime_type="application/json")

    with pytest.raises(OpenAIRequestError, match="response_mime_type"):
        await _collect(_llm(server), _request(config=config))

    assert server.bodies == []


@pytest.mark.asyncio
async def test_empty_history_sends_the_adapter_prompt_turn() -> None:
    server = _Server(_json_reply(_response([MESSAGE_ITEM])))

    await _collect(_llm(server), _request([]))

    assert server.bodies[0]["input"] == [_user_message(ADAPTER_PROMPT)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_model", "expected"),
    [
        pytest.param(None, "gpt-6-luna", id="adapter-model"),
        pytest.param("openrouter/openai/GPT-6.1-Terra", MODEL, id="legacy-prefix"),
    ],
)
async def test_model_id_is_normalized_and_falls_back_to_the_adapter_model(
    request_model: str | None, expected: str
) -> None:
    server = _Server(_json_reply(_response([MESSAGE_ITEM])))

    await _collect(_llm(server, model="gpt-6-luna"), _request(model=request_model))

    assert server.bodies[0]["model"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_final_response_hook_runs_once_per_final_response(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    hooked: list[tuple[LlmResponse, str]] = []

    async def record(
        self: OpenAIResponsesLlm, response: LlmResponse, *, model: str
    ) -> None:
        hooked.append((response, model))

    monkeypatch.setattr(OpenAIResponsesLlm, "_after_final_response", record)
    reply = (
        _sse_reply(_stream_turn(_response([MESSAGE_ITEM])))
        if stream
        else _json_reply(_response([MESSAGE_ITEM]))
    )

    responses = await _collect(
        _llm(_Server(reply)),
        _request(model="openrouter/openai/GPT-6.1-Terra"),
        stream=stream,
    )

    assert len(hooked) == 1
    hooked_response, hooked_model = hooked[0]
    assert hooked_response is responses[-1]
    assert hooked_response.partial is False
    assert hooked_model == MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replies", "stream"),
    [
        pytest.param([_json_reply(_response([MESSAGE_ITEM]))], False, id="json"),
        pytest.param(
            [_sse_reply(_stream_turn(_response([MESSAGE_ITEM])))], True, id="sse"
        ),
        pytest.param(
            [
                _error_reply(400, code="invalid_encrypted_content"),
                _json_reply(_response([MESSAGE_ITEM])),
            ],
            False,
            id="retried",
        ),
    ],
)
async def test_each_successful_call_closes_its_single_client(
    replies: list[httpx.Response], stream: bool
) -> None:
    transports: list[_Transport] = []
    llm = _llm(_Server(*replies), transports=transports)

    await _collect(llm, _request(_history()), stream=stream)

    assert [transport.closes for transport in transports] == [1]


@pytest.mark.asyncio
async def test_failed_call_still_closes_its_client() -> None:
    transports: list[_Transport] = []
    llm = _llm(
        _Server(_error_reply(401, code="invalid_api_key")), transports=transports
    )

    with pytest.raises(ProviderAuthenticationError):
        await _collect(llm, _request())

    assert [transport.closes for transport in transports] == [1]


@pytest.mark.asyncio
async def test_timeout_and_retries_come_from_http_options_then_adapter_defaults() -> (
    None
):
    captures: list[tuple[float, int]] = []
    server = _Server(
        _json_reply(_response([MESSAGE_ITEM])),
        _json_reply(_response([MESSAGE_ITEM])),
    )
    tuned = types.GenerateContentConfig(
        http_options=types.HttpOptions(
            timeout=5_000,
            retry_options=types.HttpRetryOptions(attempts=4),
        )
    )

    await _collect(_llm(server, captures=captures), _request(config=tuned))
    await _collect(_llm(server, captures=captures, max_retries=2), _request())

    assert captures == [(5.0, 3), (120.0, 2)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_final_response_records_one_estimated_ledger_row(
    tmp_path: Path, stream: bool
) -> None:
    ledger_path = tmp_path / "costs.db"
    reply = (
        _sse_reply(_stream_turn(_response([MESSAGE_ITEM])))
        if stream
        else _json_reply(_response([MESSAGE_ITEM]))
    )
    begin_cost_capture(user_id="u", session_id="s", invocation_id="inv")

    responses = await _collect(
        _llm(_Server(reply), ledger_path=ledger_path), _request(), stream=stream
    )

    assert attach_cost_metadata(responses[-1]) is not None
    with closing(sqlite3.connect(ledger_path)) as connection:
        rows = connection.execute(
            "SELECT cost_kind, estimated_cost_nano_usd FROM llm_usage_ledger"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "estimate"
    assert rows[0][1] > 0


@pytest.mark.asyncio
async def test_final_response_without_usage_records_nothing(tmp_path: Path) -> None:
    llm = _llm(
        _Server(_json_reply(_response([MESSAGE_ITEM]))), ledger_path=tmp_path / "c.db"
    )

    await llm._after_final_response(LlmResponse(partial=False), model=MODEL)

    assert not (tmp_path / "c.db").exists()
