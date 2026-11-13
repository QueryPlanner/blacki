"""Tests for converting OpenAI Responses payloads into ADK responses."""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from blacki.models.openai_items import StoredItem, is_commentary, stored_item
from blacki.models.openai_response import (
    OpenAIResponseError,
    OpenAIStreamAccumulator,
    response_to_llm_response,
)

LOGGER_NAME = "blacki.models.openai_response"
REQUESTED_MODEL = "gpt-6.1-terra"
DATED_MODEL = "gpt-6.1-terra-2026-06-01"
ENCRYPTED = "ENC_fixture_reasoning_blob_must_not_be_logged"
INVALID_ARGUMENTS_MESSAGE = "OpenAI returned invalid JSON in tool arguments."
NON_OBJECT_ARGUMENTS_MESSAGE = (
    "OpenAI returned tool arguments that are not a JSON object."
)
MESSAGE_TEXT = "You logged two meals today."

REASONING_ITEM: dict[str, Any] = {
    "id": "rs_0a1b2c3d",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "Checking the meal log first."}],
    "encrypted_content": ENCRYPTED,
}

MESSAGE_ITEM: dict[str, Any] = {
    "id": "msg_4e5f6a7b",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "phase": "final_answer",
    "content": [
        {"type": "output_text", "text": MESSAGE_TEXT, "annotations": []},
    ],
}

COMMENTARY_ITEM: dict[str, Any] = {
    "id": "msg_9c8d7e6f",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "phase": "commentary",
    "content": [
        {
            "type": "output_text",
            "text": "Let me pull today's entries.",
            "annotations": [],
        },
    ],
}

FUNCTION_CALL_ITEM: dict[str, Any] = {
    "id": "fc_7g8h9i0j",
    "type": "function_call",
    "call_id": "call_AbC123xyz",
    "name": "list_meals",
    "arguments": '{"date": "2026-11-12"}',
    "status": "completed",
}

USAGE: dict[str, Any] = {
    "input_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 1024},
    "output_tokens": 340,
    "output_tokens_details": {"reasoning_tokens": 256},
    "total_tokens": 1540,
}

BASE_RESPONSE: dict[str, Any] = {
    "id": "resp_0f1e2d3c",
    "object": "response",
    "created_at": 1789200000,
    "model": DATED_MODEL,
    "status": "completed",
    "output": [],
    "usage": USAGE,
    "store": False,
}

INCOMPLETE_MAX_TOKENS: dict[str, Any] = {
    "status": "incomplete",
    "incomplete_details": {"reason": "max_output_tokens"},
}


def _response(output: Any, **overrides: Any) -> dict[str, Any]:
    return {**BASE_RESPONSE, "output": output, **overrides}


def _parts(response: LlmResponse) -> list[types.Part]:
    assert response.content is not None
    assert response.content.parts is not None
    return response.content.parts


def _stored(part: types.Part) -> StoredItem:
    stored = stored_item(part)
    assert stored is not None
    return stored


def test_one_part_per_item_in_order_with_requested_model_metadata() -> None:
    raw = _response([REASONING_ITEM, MESSAGE_ITEM, FUNCTION_CALL_ITEM])
    before = copy.deepcopy(raw)

    response = response_to_llm_response(raw, model=REQUESTED_MODEL)

    assert raw == before
    assert response.content is not None
    assert response.content.role == "model"
    parts = _parts(response)
    assert len(parts) == 3
    reasoning, message, call = parts
    assert reasoning.thought is True
    assert reasoning.text == "Checking the meal log first."
    assert message.text == MESSAGE_TEXT
    assert message.thought is None
    assert call.function_call is not None
    assert call.function_call.id == "call_AbC123xyz"
    assert call.function_call.name == "list_meals"
    assert call.function_call.args == {"date": "2026-11-12"}
    for part, item in zip(
        parts, [REASONING_ITEM, MESSAGE_ITEM, FUNCTION_CALL_ITEM], strict=True
    ):
        assert _stored(part) == StoredItem(model=REQUESTED_MODEL, item=item)
    assert response.model_version == DATED_MODEL
    assert response.partial is False
    assert response.finish_reason == types.FinishReason.STOP
    assert response.error_code is None


@pytest.mark.parametrize(
    "summary",
    [
        [],
        [
            {"type": "summary_text", "text": "First. "},
            {"type": "other", "text": "ignored"},
            {"type": "summary_text", "text": None},
            "not a mapping",
            {"type": "summary_text", "text": "Second."},
        ],
    ],
)
def test_reasoning_summary_texts_are_joined_in_order(summary: list[Any]) -> None:
    item = {**REASONING_ITEM, "summary": summary}

    parts = _parts(response_to_llm_response(_response([item]), model=REQUESTED_MODEL))

    assert parts[0].thought is True
    assert parts[0].text == ("" if not summary else "First. Second.")
    assert _stored(parts[0]) == StoredItem(model=REQUESTED_MODEL, item=item)


def test_reasoning_with_missing_summary_is_kept_as_empty_thought() -> None:
    item = {k: v for k, v in REASONING_ITEM.items() if k != "summary"}

    parts = _parts(
        response_to_llm_response(
            _response([item, FUNCTION_CALL_ITEM]), model=REQUESTED_MODEL
        )
    )

    assert parts[0].thought is True
    assert parts[0].text == ""
    assert _stored(parts[0]).item == item
    assert parts[1].function_call is not None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            [{"type": "refusal", "refusal": "I can't help with that."}],
            "I can't help with that.",
        ),
        (
            [
                {"type": "output_text", "text": "Hi. "},
                {"type": "refusal", "refusal": "No."},
            ],
            "Hi. No.",
        ),
        (
            [
                {"type": "output_text", "text": "Kept. "},
                {"type": "image", "url": "ignored"},
                "not a mapping",
                {"type": "output_text", "text": None},
            ],
            "Kept. ",
        ),
        (None, ""),
    ],
)
def test_message_text_is_output_text_and_refusal(
    content: Any,
    expected: str,
) -> None:
    item = {**MESSAGE_ITEM, "content": content}

    parts = _parts(response_to_llm_response(_response([item]), model=REQUESTED_MODEL))

    assert [part.text for part in parts] == [expected]
    assert parts[0].thought is None
    assert _stored(parts[0]).item == item


def test_commentary_phase_is_kept_in_stored_item() -> None:
    parts = _parts(
        response_to_llm_response(
            _response([COMMENTARY_ITEM, FUNCTION_CALL_ITEM]),
            model=REQUESTED_MODEL,
        )
    )

    assert parts[0].text == "Let me pull today's entries."
    assert is_commentary(parts[0]) is True
    assert _stored(parts[0]).item["phase"] == "commentary"
    assert is_commentary(parts[1]) is False


@pytest.mark.parametrize(
    "arguments",
    ['{"date": "2026-11-12", "note": "private lunch', "{not json", 5],
)
def test_invalid_json_arguments_raise_without_echoing_them(arguments: Any) -> None:
    item = {**FUNCTION_CALL_ITEM, "arguments": arguments}

    with pytest.raises(OpenAIResponseError) as excinfo:
        response_to_llm_response(_response([item]), model=REQUESTED_MODEL)

    assert excinfo.value.message == INVALID_ARGUMENTS_MESSAGE
    assert excinfo.value.code is None
    assert excinfo.value.__cause__ is None
    assert "private lunch" not in str(excinfo.value)


@pytest.mark.parametrize("arguments", ["[1, 2]", "null", '"private lunch"', "5"])
def test_valid_json_arguments_that_are_not_an_object_raise(arguments: str) -> None:
    item = {**FUNCTION_CALL_ITEM, "arguments": arguments}

    with pytest.raises(OpenAIResponseError) as excinfo:
        response_to_llm_response(_response([item]), model=REQUESTED_MODEL)

    assert excinfo.value.message == NON_OBJECT_ARGUMENTS_MESSAGE
    assert "private lunch" not in str(excinfo.value)


def test_non_object_arguments_still_raise_on_max_tokens() -> None:
    item = {**FUNCTION_CALL_ITEM, "arguments": "[1, 2]"}

    with pytest.raises(OpenAIResponseError, match="not a JSON object"):
        response_to_llm_response(
            _response([item], **INCOMPLETE_MAX_TOKENS), model=REQUESTED_MODEL
        )


@pytest.mark.parametrize(
    "item",
    [
        {**FUNCTION_CALL_ITEM, "arguments": ""},
        {**FUNCTION_CALL_ITEM, "arguments": "  \n"},
        {k: v for k, v in FUNCTION_CALL_ITEM.items() if k != "arguments"},
    ],
)
def test_empty_or_missing_arguments_are_empty_object_without_warning(
    item: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        response = response_to_llm_response(_response([item]), model=REQUESTED_MODEL)

    call = _parts(response)[0].function_call
    assert call is not None
    assert call.args == {}
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_truncated_arguments_return_max_tokens_response_without_parts() -> None:
    item = {**FUNCTION_CALL_ITEM, "arguments": '{"date": "2026-11-'}
    raw = _response([REASONING_ITEM, item], **INCOMPLETE_MAX_TOKENS)

    response = response_to_llm_response(raw, model=REQUESTED_MODEL)

    assert _parts(response) == []
    assert response.finish_reason == types.FinishReason.MAX_TOKENS
    assert response.error_code == types.FinishReason.MAX_TOKENS
    assert response.error_message is not None
    assert "truncated" in response.error_message
    assert response.usage_metadata is not None
    assert response.usage_metadata.prompt_token_count == 1200
    assert response.custom_metadata == {"openai_usage": USAGE}


def test_valid_arguments_on_max_tokens_keep_the_call_and_report_max_tokens() -> None:
    raw = _response([FUNCTION_CALL_ITEM], **INCOMPLETE_MAX_TOKENS)

    response = response_to_llm_response(raw, model=REQUESTED_MODEL)

    parts = _parts(response)
    assert len(parts) == 1
    assert parts[0].function_call is not None
    assert parts[0].function_call.args == {"date": "2026-11-12"}
    assert response.finish_reason == types.FinishReason.MAX_TOKENS
    assert response.error_message == "Maximum tokens reached"


def test_unknown_item_types_are_skipped_and_logged_once_per_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    web_search = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": "private meal query"},
    }
    file_search = {
        "id": "fs_1",
        "type": "file_search_call",
        "status": "completed",
        "queries": ["private file query"],
    }
    raw = _response(
        [web_search, MESSAGE_ITEM, web_search, file_search, {"id": "no_type"}]
    )

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        response = response_to_llm_response(raw, model=REQUESTED_MODEL)

    assert [part.text for part in _parts(response)] == [MESSAGE_TEXT]
    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
    assert messages == [
        "Skipping OpenAI output item of type web_search_call.",
        "Skipping OpenAI output item of type file_search_call.",
        "Skipping OpenAI output item of type unknown.",
    ]
    assert "private meal query" not in caplog.text
    assert "private file query" not in caplog.text


def test_non_object_output_item_raises() -> None:
    with pytest.raises(OpenAIResponseError, match="invalid output item"):
        response_to_llm_response(_response(["not an item"]), model=REQUESTED_MODEL)


@pytest.mark.parametrize("output", [None, "text", {"type": "message"}])
def test_output_that_is_not_a_list_yields_no_parts(output: Any) -> None:
    response = response_to_llm_response(_response(output), model=REQUESTED_MODEL)

    assert _parts(response) == []


def test_missing_output_yields_no_parts() -> None:
    raw = {k: v for k, v in BASE_RESPONSE.items() if k != "output"}

    response = response_to_llm_response(raw, model=REQUESTED_MODEL)

    assert _parts(response) == []


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (
            {k: v for k, v in FUNCTION_CALL_ITEM.items() if k != "name"},
            "without a name",
        ),
        ({**FUNCTION_CALL_ITEM, "name": ""}, "without a name"),
        (
            {k: v for k, v in FUNCTION_CALL_ITEM.items() if k != "call_id"},
            "without a call_id",
        ),
        ({**FUNCTION_CALL_ITEM, "call_id": None}, "without a call_id"),
    ],
)
def test_function_call_without_name_or_call_id_raises(
    item: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(OpenAIResponseError, match=message):
        response_to_llm_response(_response([item]), model=REQUESTED_MODEL)


def test_usage_maps_cache_and_reasoning_and_keeps_raw_usage() -> None:
    response = response_to_llm_response(
        _response([MESSAGE_ITEM]), model=REQUESTED_MODEL
    )

    usage = response.usage_metadata
    assert usage is not None
    assert usage.prompt_token_count == 1200
    assert usage.candidates_token_count == 340
    assert usage.total_token_count == 1540
    assert usage.cached_content_token_count == 1024
    assert usage.thoughts_token_count == 256
    assert response.custom_metadata == {"openai_usage": USAGE}


def test_total_tokens_falls_back_to_input_plus_output() -> None:
    usage = {k: v for k, v in USAGE.items() if k != "total_tokens"}

    response = response_to_llm_response(
        _response([MESSAGE_ITEM], usage=usage), model=REQUESTED_MODEL
    )

    assert response.usage_metadata is not None
    assert response.usage_metadata.total_token_count == 1540


def test_zero_reasoning_tokens_are_not_reported_as_thoughts() -> None:
    usage = {**USAGE, "output_tokens_details": {"reasoning_tokens": 0}}

    response = response_to_llm_response(
        _response([MESSAGE_ITEM], usage=usage), model=REQUESTED_MODEL
    )

    assert response.usage_metadata is not None
    assert response.usage_metadata.thoughts_token_count is None


@pytest.mark.parametrize(
    ("usage", "expected_metadata", "expected_custom"),
    [
        (None, None, None),
        ({}, None, None),
        ({"unknown": 1}, None, {"openai_usage": {"unknown": 1}}),
    ],
)
def test_usage_without_known_counts_maps_to_none(
    usage: Any,
    expected_metadata: None,
    expected_custom: dict[str, Any] | None,
) -> None:
    response = response_to_llm_response(
        _response([MESSAGE_ITEM], usage=usage), model=REQUESTED_MODEL
    )

    assert response.usage_metadata is expected_metadata
    assert response.custom_metadata == expected_custom


def test_usage_missing_key_maps_to_none() -> None:
    raw = {k: v for k, v in BASE_RESPONSE.items() if k != "usage"}

    response = response_to_llm_response(raw, model=REQUESTED_MODEL)

    assert response.usage_metadata is None
    assert response.custom_metadata is None


def test_invalid_usage_counts_are_ignored_and_valid_ones_kept() -> None:
    usage = {
        "input_tokens": -5,
        "output_tokens": 7,
        "total_tokens": None,
        "input_tokens_details": {"cached_tokens": -1},
        "output_tokens_details": "not a mapping",
    }

    response = response_to_llm_response(
        _response([MESSAGE_ITEM], usage=usage), model=REQUESTED_MODEL
    )

    assert response.usage_metadata is not None
    assert response.usage_metadata.prompt_token_count == 0
    assert response.usage_metadata.candidates_token_count == 7
    assert response.usage_metadata.total_token_count == 7
    assert response.usage_metadata.cached_content_token_count is None
    assert response.usage_metadata.thoughts_token_count is None


def test_usage_with_only_invalid_counts_maps_to_no_usage_metadata() -> None:
    usage = {"input_tokens": -5, "output_tokens": True}

    response = response_to_llm_response(
        _response([MESSAGE_ITEM], usage=usage), model=REQUESTED_MODEL
    )

    assert response.usage_metadata is None
    assert response.custom_metadata == {"openai_usage": usage}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"status": "completed"}, types.FinishReason.STOP),
        (INCOMPLETE_MAX_TOKENS, types.FinishReason.MAX_TOKENS),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            types.FinishReason.SAFETY,
        ),
        (
            {"status": "incomplete", "incomplete_details": {"reason": "new_reason"}},
            types.FinishReason.OTHER,
        ),
        ({"status": "incomplete"}, types.FinishReason.OTHER),
        ({"status": "in_progress"}, types.FinishReason.OTHER),
        ({"status": None}, None),
    ],
)
def test_finish_reason_maps_status_and_incomplete_reason(
    overrides: dict[str, Any],
    expected: types.FinishReason | None,
) -> None:
    response = response_to_llm_response(
        _response([MESSAGE_ITEM], **overrides), model=REQUESTED_MODEL
    )

    assert response.finish_reason == expected


@pytest.mark.parametrize(
    ("overrides", "error_code", "error_message"),
    [
        ({"status": "completed"}, None, None),
        (INCOMPLETE_MAX_TOKENS, "MAX_TOKENS", "Maximum tokens reached"),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            "SAFETY",
            "Finished with SAFETY",
        ),
    ],
)
def test_non_stop_finish_sets_error_fields(
    overrides: dict[str, Any],
    error_code: str | None,
    error_message: str | None,
) -> None:
    response = response_to_llm_response(
        _response([MESSAGE_ITEM], **overrides), model=REQUESTED_MODEL
    )

    assert response.error_code == error_code
    assert response.error_message == error_message


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (
            {"code": "server_error", "message": "The model failed."},
            "server_error",
            "The model failed.",
        ),
        (None, None, "OpenAI returned an error."),
        ("boom", None, "OpenAI returned an error."),
        ({"code": "", "message": ""}, None, "OpenAI returned an error."),
        ({"code": 500}, None, "OpenAI returned an error."),
        ({"message": "x" * 5000}, None, "x" * 1000),
    ],
)
def test_failed_response_raises_with_code_and_message(
    error: Any,
    code: str | None,
    message: str,
) -> None:
    raw = _response([], status="failed", error=error)

    with pytest.raises(OpenAIResponseError) as excinfo:
        response_to_llm_response(raw, model=REQUESTED_MODEL)

    assert excinfo.value.code == code
    assert excinfo.value.message == message
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (DATED_MODEL, DATED_MODEL),
        ("", None),
        (7, None),
    ],
)
def test_model_version_reports_the_dated_id_or_none(model: Any, expected: str) -> None:
    raw = _response([MESSAGE_ITEM], model=model)

    response = response_to_llm_response(raw, model=REQUESTED_MODEL)

    assert response.model_version == expected
    assert _stored(_parts(response)[0]).model == REQUESTED_MODEL


def test_missing_model_key_leaves_model_version_unset() -> None:
    raw = {k: v for k, v in BASE_RESPONSE.items() if k != "model"}

    response = response_to_llm_response(raw, model=REQUESTED_MODEL)

    assert response.model_version is None


def _event(event_type: str, **fields: Any) -> dict[str, Any]:
    return {"type": event_type, **fields}


def _streamed_turn(final: dict[str, Any]) -> list[dict[str, Any]]:
    in_progress = {**BASE_RESPONSE, "status": "in_progress", "output": []}
    return [
        _event("response.created", response=in_progress),
        _event("response.in_progress", response=in_progress),
        _event(
            "response.output_item.added",
            output_index=0,
            item={"id": "rs_0a1b2c3d", "type": "reasoning", "summary": []},
        ),
        _event(
            "response.reasoning_summary_part.added",
            output_index=0,
            summary_index=0,
            part={"type": "summary_text", "text": ""},
        ),
        _event(
            "response.reasoning_summary_text.delta",
            output_index=0,
            summary_index=0,
            delta="Checking ",
        ),
        _event(
            "response.reasoning_summary_text.delta",
            output_index=0,
            summary_index=0,
            delta="the meal log.",
        ),
        _event(
            "response.reasoning_summary_text.done",
            output_index=0,
            text="Checking the meal log.",
        ),
        _event("response.output_item.done", output_index=0, item=REASONING_ITEM),
        _event(
            "response.output_item.added",
            output_index=1,
            item={**COMMENTARY_ITEM, "content": []},
        ),
        _event(
            "response.output_text.delta",
            output_index=1,
            content_index=0,
            delta="Let me pull ",
        ),
        _event(
            "response.output_text.delta",
            output_index=1,
            content_index=0,
            delta="today's entries.",
        ),
        _event("response.output_text.done", output_index=1, text="today's entries."),
        _event("response.output_item.done", output_index=1, item=COMMENTARY_ITEM),
        _event(
            "response.output_item.added",
            output_index=2,
            item={**FUNCTION_CALL_ITEM, "arguments": ""},
        ),
        _event(
            "response.function_call_arguments.delta",
            output_index=2,
            delta='{"date": ',
        ),
        _event(
            "response.function_call_arguments.delta",
            output_index=2,
            delta='"2026-11-12"}',
        ),
        _event(
            "response.function_call_arguments.done",
            output_index=2,
            arguments='{"date": "2026-11-12"}',
        ),
        _event("response.output_item.done", output_index=2, item=FUNCTION_CALL_ITEM),
        _event("response.completed", response=final),
    ]


def test_stream_yields_partials_then_one_final_equal_to_non_streamed() -> None:
    final_raw = _response([REASONING_ITEM, COMMENTARY_ITEM, FUNCTION_CALL_ITEM])
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)
    partials: list[LlmResponse] = []

    for event in _streamed_turn(final_raw):
        partials.extend(accumulator.add(event))

    assert len(partials) == 4
    assert all(partial.partial is True for partial in partials)
    assert [_parts(partial)[0].text for partial in partials] == [
        "Checking ",
        "the meal log.",
        "Let me pull ",
        "today's entries.",
    ]
    assert [_parts(partial)[0].thought for partial in partials] == [
        True,
        True,
        None,
        None,
    ]
    assert all(_parts(partial)[0].part_metadata is None for partial in partials)
    final = accumulator.finish()
    assert final.partial is False
    assert final == response_to_llm_response(final_raw, model=REQUESTED_MODEL)
    assert [_stored(part).item for part in _parts(final)] == [
        REASONING_ITEM,
        COMMENTARY_ITEM,
        FUNCTION_CALL_ITEM,
    ]


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("max_output_tokens", types.FinishReason.MAX_TOKENS),
        ("content_filter", types.FinishReason.SAFETY),
    ],
)
def test_incomplete_terminal_event_builds_the_final_response(
    reason: str,
    expected: types.FinishReason,
) -> None:
    final_raw = _response(
        [MESSAGE_ITEM],
        status="incomplete",
        incomplete_details={"reason": reason},
    )
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)

    partial = accumulator.add(_event("response.output_text.delta", delta="You logged"))
    terminal = accumulator.add(_event("response.incomplete", response=final_raw))

    assert len(partial) == 1
    assert terminal == []
    final = accumulator.finish()
    assert final.finish_reason == expected
    assert final == response_to_llm_response(final_raw, model=REQUESTED_MODEL)


@pytest.mark.parametrize(
    ("event", "code", "message"),
    [
        (
            _event(
                "response.failed",
                response=_response(
                    [],
                    status="failed",
                    error={"code": "server_error", "message": "The model failed."},
                ),
            ),
            "server_error",
            "The model failed.",
        ),
        (
            _event(
                "response.failed",
                response=_response([], status="failed", error=None),
            ),
            None,
            "OpenAI returned an error.",
        ),
        (_event("response.failed"), None, "OpenAI returned an error."),
        (
            _event(
                "error",
                code="rate_limit_exceeded",
                message="Rate limit reached.",
            ),
            "rate_limit_exceeded",
            "Rate limit reached.",
        ),
        (_event("error"), None, "OpenAI returned an error."),
    ],
)
def test_failed_and_error_events_raise(
    event: dict[str, Any],
    code: str | None,
    message: str,
) -> None:
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)

    with pytest.raises(OpenAIResponseError) as excinfo:
        accumulator.add(event)

    assert excinfo.value.code == code
    assert excinfo.value.message == message


def test_stream_invalid_arguments_raise_on_the_terminal_event() -> None:
    bad_call = {**FUNCTION_CALL_ITEM, "arguments": '{"note": "private lunch'}
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)

    with pytest.raises(OpenAIResponseError) as excinfo:
        accumulator.add(_event("response.completed", response=_response([bad_call])))

    assert excinfo.value.message == INVALID_ARGUMENTS_MESSAGE
    assert "private lunch" not in str(excinfo.value)
    with pytest.raises(OpenAIResponseError, match="ended without a final response"):
        accumulator.finish()


def test_stream_non_object_arguments_raise_on_the_terminal_event() -> None:
    bad_call = {**FUNCTION_CALL_ITEM, "arguments": "[1, 2]"}
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)

    with pytest.raises(OpenAIResponseError, match="not a JSON object"):
        accumulator.add(_event("response.completed", response=_response([bad_call])))


def test_stream_truncated_arguments_build_the_max_tokens_final() -> None:
    truncated_call = {**FUNCTION_CALL_ITEM, "arguments": '{"note": "private'}
    final_raw = _response([REASONING_ITEM, truncated_call], **INCOMPLETE_MAX_TOKENS)
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)

    assert accumulator.add(_event("response.incomplete", response=final_raw)) == []

    final = accumulator.finish()
    assert _parts(final) == []
    assert final.finish_reason == types.FinishReason.MAX_TOKENS
    assert final == response_to_llm_response(final_raw, model=REQUESTED_MODEL)


def test_terminal_event_without_response_raises() -> None:
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)

    with pytest.raises(OpenAIResponseError, match="without a response"):
        accumulator.add(_event("response.completed"))


def test_finish_without_terminal_event_raises() -> None:
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)
    accumulator.add(_event("response.output_text.delta", delta="partial"))

    with pytest.raises(OpenAIResponseError, match="ended without a final response"):
        accumulator.finish()


@pytest.mark.parametrize(
    "event",
    [
        _event("response.output_text.delta", delta=""),
        _event("response.reasoning_summary_text.delta", delta=None),
    ],
)
def test_empty_or_missing_delta_yields_no_partial(event: dict[str, Any]) -> None:
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)

    assert accumulator.add(event) == []


def test_events_after_the_terminal_event_are_ignored() -> None:
    final_raw = _response([MESSAGE_ITEM])
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)
    accumulator.add(_event("response.completed", response=final_raw))

    assert accumulator.add(_event("response.output_text.delta", delta="late")) == []
    assert accumulator.add(_event("error", code="late", message="late")) == []
    assert (
        accumulator.add(
            _event(
                "response.failed",
                response=_response([], status="failed", error=None),
            )
        )
        == []
    )
    assert (
        accumulator.add(
            _event("response.completed", response=_response([FUNCTION_CALL_ITEM]))
        )
        == []
    )
    final = accumulator.finish()
    assert final == response_to_llm_response(final_raw, model=REQUESTED_MODEL)
    assert accumulator.finish() is final


def test_unrelated_stream_events_are_ignored() -> None:
    accumulator = OpenAIStreamAccumulator(REQUESTED_MODEL)

    assert accumulator.add(_event("response.created", response=BASE_RESPONSE)) == []
    assert (
        accumulator.add(_event("response.function_call_arguments.delta", delta="{"))
        == []
    )
    assert accumulator.add({}) == []
