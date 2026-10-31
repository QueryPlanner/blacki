"""Tests for converting OpenRouter responses into ADK responses."""

from __future__ import annotations

from typing import Any

import pytest
from google.genai import types

from blacki.models.openrouter_response import (
    OpenRouterResponseError,
    OpenRouterStreamAccumulator,
    _map_finish_reason,
    _usage_metadata,
    response_to_llm_response,
)


def _completion(**message: Any) -> dict[str, Any]:
    return {
        "id": "gen-1",
        "model": "google/gemini-2.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", **message},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.01,
            "cost_details": {"upstream_inference_cost": 0.009},
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


def test_non_streaming_response_maps_text_reasoning_tools_and_usage() -> None:
    payload = _completion(
        content="hello",
        reasoning="internal",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "x"}'},
            }
        ],
    )
    payload["choices"][0]["finish_reason"] = "tool_calls"

    response = response_to_llm_response(payload)

    assert response.content is not None
    assert response.content.parts is not None
    assert [part.thought for part in response.content.parts] == [True, None, None]
    assert response.content.parts[0].text == "internal"
    assert response.content.parts[1].text == "hello"
    call = response.content.parts[2].function_call
    assert call is not None
    assert call.id == "call-1"
    assert call.name == "lookup"
    assert call.args == {"query": "x"}
    assert response.finish_reason == types.FinishReason.STOP
    assert response.error_code is None
    assert response.model_version == "google/gemini-2.5-flash"
    assert response.usage_metadata is not None
    assert response.usage_metadata.prompt_token_count == 10
    assert response.usage_metadata.candidates_token_count == 5
    assert response.usage_metadata.total_token_count == 15
    assert response.usage_metadata.cached_content_token_count == 2
    assert response.usage_metadata.thoughts_token_count == 3


def test_non_streaming_content_list_and_reasoning_details() -> None:
    payload = _completion(
        content=[
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": "ignored"}},
            "not-a-mapping",
            {"type": "text", "text": 5},
            {"type": "text", "text": "b"},
        ],
        reasoning_details=[
            {"type": "reasoning.text", "text": "one "},
            {"type": "reasoning.summary", "summary": "two"},
            {"type": "reasoning.encrypted", "data": "secret"},
            {"type": "reasoning.text", "text": ""},
            "not-a-mapping",
        ],
    )

    response = response_to_llm_response(payload)

    assert response.content is not None
    assert response.content.parts is not None
    assert response.content.parts[0].text == "one two"
    assert response.content.parts[0].thought is True
    assert response.content.parts[1].text == "ab"


def test_non_streaming_missing_call_id_gets_blacki_id() -> None:
    payload = _completion(
        tool_calls=[
            {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]
    )

    response = response_to_llm_response(payload)

    assert response.content is not None
    assert response.content.parts is not None
    call = response.content.parts[0].function_call
    assert call is not None
    assert call.id is not None
    assert call.id.startswith("blacki_call_")


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("length", types.FinishReason.MAX_TOKENS),
        ("content_filter", types.FinishReason.SAFETY),
        ("mystery", types.FinishReason.OTHER),
    ],
)
def test_non_streaming_non_stop_finish_reasons_set_errors(
    finish_reason: str,
    expected: types.FinishReason,
) -> None:
    payload = _completion(content="text")
    payload["choices"][0]["finish_reason"] = finish_reason

    response = response_to_llm_response(payload)

    assert response.finish_reason == expected
    assert response.error_code == expected
    assert response.error_message


def test_non_streaming_invalid_tool_json_raises() -> None:
    payload = _completion(
        tool_calls=[
            {"id": "1", "type": "function", "function": {"name": "x", "arguments": "{"}}
        ]
    )

    with pytest.raises(OpenRouterResponseError, match="invalid JSON"):
        response_to_llm_response(payload)


def test_non_streaming_truncated_tool_json_returns_max_tokens_response() -> None:
    payload = _completion(
        tool_calls=[
            {"id": "1", "type": "function", "function": {"name": "x", "arguments": "{"}}
        ]
    )
    payload["choices"][0]["finish_reason"] = "length"

    response = response_to_llm_response(payload)

    assert response.finish_reason == types.FinishReason.MAX_TOKENS
    assert response.error_message is not None
    assert "truncated" in response.error_message
    assert response.content is not None
    assert response.content.parts == []


def test_non_streaming_empty_choices_returns_empty_content() -> None:
    response = response_to_llm_response({"model": "m", "choices": []})

    assert response.content is not None
    assert response.content.parts == []
    assert response.model_version == "m"


@pytest.mark.parametrize(
    "error",
    [
        {"code": 402, "message": "Insufficient credits"},
        "provider disconnected",
    ],
)
def test_non_streaming_error_payload_raises(error: object) -> None:
    with pytest.raises(OpenRouterResponseError):
        response_to_llm_response({"error": error})


def test_non_streaming_finish_reason_error_raises() -> None:
    payload = _completion(content="x")
    payload["choices"][0]["finish_reason"] = "error"

    with pytest.raises(OpenRouterResponseError, match="finish_reason=error"):
        response_to_llm_response(payload)


def test_non_streaming_invalid_choice_raises() -> None:
    with pytest.raises(OpenRouterResponseError, match="invalid choice"):
        response_to_llm_response({"choices": ["bad"]})


def test_non_streaming_tool_call_shape_errors_raise() -> None:
    with pytest.raises(OpenRouterResponseError, match="without a function"):
        response_to_llm_response(
            _completion(tool_calls=[{"id": "1", "type": "function"}])
        )

    with pytest.raises(OpenRouterResponseError, match="without a name"):
        response_to_llm_response(
            _completion(
                tool_calls=[{"id": "1", "type": "function", "function": {"name": None}}]
            )
        )


def test_non_streaming_non_mapping_tool_call_is_skipped() -> None:
    response = response_to_llm_response(_completion(tool_calls=["bad"]))

    assert response.content is not None
    assert response.content.parts == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"prompt_tokens": "not-a-number"},
        {"prompt_tokens": -1},
    ],
)
def test_usage_metadata_handles_missing_or_invalid_values(value: object) -> None:
    assert _usage_metadata(value) is None


def test_usage_metadata_handles_missing_details() -> None:
    metadata = _usage_metadata(
        {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": "bad"},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    assert metadata is not None
    assert metadata.cached_content_token_count is None
    assert metadata.thoughts_token_count is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        (types.FinishReason.SAFETY, types.FinishReason.SAFETY),
        ("stop", types.FinishReason.STOP),
        ("unknown", types.FinishReason.OTHER),
    ],
)
def test_map_finish_reason(value: object, expected: types.FinishReason | None) -> None:
    assert _map_finish_reason(value) == expected


def test_stream_accumulator_yields_partials_and_one_final_response() -> None:
    accumulator = OpenRouterStreamAccumulator()

    partials = accumulator.add(
        {
            "model": "google/gemini-2.5-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": "thinking"},
                    "finish_reason": None,
                }
            ],
        }
    )
    assert len(partials) == 1
    assert partials[0].partial is True
    assert partials[0].content is not None
    assert partials[0].content.parts is not None
    assert partials[0].content.parts[0].thought is True

    partials = accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hello"},
                    "finish_reason": None,
                }
            ]
        }
    )
    assert len(partials) == 1
    assert partials[0].content is not None
    assert partials[0].content.parts is not None
    assert partials[0].content.parts[0].text == "hello"

    partials = accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"q":',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    assert partials == []

    partials = accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"x"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
                "cost": 0.001,
            },
        }
    )
    assert partials == []

    final = accumulator.finish()
    assert final.partial is False
    assert final.finish_reason == types.FinishReason.STOP
    assert final.usage_metadata is not None
    assert final.usage_metadata.total_token_count == 6
    assert final.content is not None
    assert final.content.parts is not None
    assert final.content.parts[0].text == "thinking"
    assert final.content.parts[0].thought is True
    assert final.content.parts[1].text == "hello"
    call = final.content.parts[2].function_call
    assert call is not None
    assert call.id == "call-1"
    assert call.args == {"q": "x"}


def test_stream_accumulator_usage_chunk_with_empty_choices() -> None:
    accumulator = OpenRouterStreamAccumulator()
    accumulator.add(
        {
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    final = accumulator.finish()

    assert final.content is not None
    assert final.content.parts == []
    assert final.usage_metadata is not None
    assert final.finish_reason is None


def test_stream_accumulator_top_level_error_raises() -> None:
    accumulator = OpenRouterStreamAccumulator()

    with pytest.raises(OpenRouterResponseError, match="Insufficient"):
        accumulator.add({"error": {"code": 402, "message": "Insufficient credits"}})


def test_stream_accumulator_finish_reason_error_raises() -> None:
    accumulator = OpenRouterStreamAccumulator()
    accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": "error",
                }
            ]
        }
    )

    with pytest.raises(OpenRouterResponseError, match="finish_reason=error"):
        accumulator.finish()


def test_stream_accumulator_invalid_and_truncated_tool_json() -> None:
    accumulator = OpenRouterStreamAccumulator()
    accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "1",
                                "function": {"name": "x", "arguments": "{"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    with pytest.raises(OpenRouterResponseError, match="invalid JSON"):
        accumulator.finish()

    truncated = OpenRouterStreamAccumulator()
    truncated.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "1",
                                "function": {"name": "x", "arguments": "{"},
                            }
                        ]
                    },
                    "finish_reason": "length",
                }
            ]
        }
    )
    final = truncated.finish()
    assert final.finish_reason == types.FinishReason.MAX_TOKENS
    assert final.content is not None
    assert final.content.parts == []


def test_stream_accumulator_tool_arguments_without_name_raise() -> None:
    accumulator = OpenRouterStreamAccumulator()
    accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"q":"x"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    with pytest.raises(OpenRouterResponseError, match="without a function name"):
        accumulator.finish()


def test_stream_accumulator_skips_invalid_shapes() -> None:
    accumulator = OpenRouterStreamAccumulator()

    assert accumulator.add({"choices": "not-a-list"}) == []
    assert accumulator.add({"choices": ["bad"]}) == []
    assert (
        accumulator.add(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": "not-a-mapping",
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        == []
    )
    assert (
        accumulator.add(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": ["bad", {"function": "bad"}]},
                        "finish_reason": None,
                    }
                ]
            }
        )
        == []
    )

    final = accumulator.finish()
    assert final.content is not None
    assert final.content.parts == []


def test_stream_accumulator_defaults_to_stop_and_deduplicates_repeated_names() -> None:
    accumulator = OpenRouterStreamAccumulator()
    accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "answer",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "lookup", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )

    final = accumulator.finish()

    assert final.finish_reason == types.FinishReason.STOP
    assert final.content is not None
    assert final.content.parts is not None
    call = final.content.parts[1].function_call
    assert call is not None
    assert call.name == "lookup"
    assert call.id is not None
    assert call.id.startswith("blacki_call_")


def test_stream_accumulator_ignores_empty_tool_call_entries() -> None:
    accumulator = OpenRouterStreamAccumulator()
    accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    final = accumulator.finish()

    assert final.content is not None
    assert final.content.parts == []


def test_non_streaming_missing_message_becomes_empty_content() -> None:
    response = response_to_llm_response(
        {"choices": [{"message": "bad", "finish_reason": "stop"}]}
    )

    assert response.content is not None
    assert response.content.parts == []


@pytest.mark.parametrize(
    "error",
    [
        {"message": "oops"},
        {"code": 402},
        {},
    ],
)
def test_error_payload_variants_raise(error: object) -> None:
    with pytest.raises(OpenRouterResponseError):
        response_to_llm_response({"error": error})


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (None, {}),
        ({"q": "x"}, {"q": "x"}),
        ("", {}),
    ],
)
def test_tool_argument_shapes(arguments: object, expected: dict[str, Any]) -> None:
    response = response_to_llm_response(
        _completion(
            tool_calls=[
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "x", "arguments": arguments},
                }
            ]
        )
    )

    assert response.content is not None
    assert response.content.parts is not None
    call = response.content.parts[0].function_call
    assert call is not None
    assert call.args == expected


@pytest.mark.parametrize("arguments", [5, "[1,2]"])
def test_invalid_tool_argument_shapes_raise(arguments: object) -> None:
    with pytest.raises(OpenRouterResponseError):
        response_to_llm_response(
            _completion(
                tool_calls=[
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "x", "arguments": arguments},
                    }
                ]
            )
        )


def test_usage_metadata_accepts_float_and_string_tokens() -> None:
    metadata = _usage_metadata(
        {"prompt_tokens": 2.5, "completion_tokens": "3", "total_tokens": 5}
    )

    assert metadata is not None
    assert metadata.prompt_token_count == 2
    assert metadata.candidates_token_count == 3


def test_usage_metadata_rejects_unsupported_types() -> None:
    assert _usage_metadata({"prompt_tokens": []}) is None


def test_stream_accumulator_concatenates_name_fragments() -> None:
    accumulator = OpenRouterStreamAccumulator()
    accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "look", "arguments": "{"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    accumulator.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "up",
                                    "arguments": '"q":"x"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    final = accumulator.finish()

    assert final.content is not None
    assert final.content.parts is not None
    call = final.content.parts[0].function_call
    assert call is not None
    assert call.name == "lookup"
    assert call.args == {"q": "x"}
