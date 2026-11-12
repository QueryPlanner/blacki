"""Tests for translating ADK requests into OpenAI Responses requests."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from blacki.models.openai_items import item_metadata
from blacki.models.openai_request import (
    OpenAIRequestError,
    ResponsesRequest,
    _system_instruction_text,
    _tool_declarations,
    build_responses_request,
)

LUNA = "gpt-6-luna"
TERRA = "gpt-6.1-terra"
SOL = "gpt-6.1-sol"
ASTRA = "gpt-6-astra"
LOGGER = "blacki.models.openai_request"
MISSING_OUTPUT = '{"error":"Missing tool response."}'

REASONING_ITEM: dict[str, Any] = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [],
    "encrypted_content": "enc",
}


def _user(*parts: types.Part) -> types.Content:
    return types.Content(role="user", parts=list(parts))


def _model(*parts: types.Part) -> types.Content:
    return types.Content(role="model", parts=list(parts))


def _call(
    name: str = "lookup",
    *,
    call_id: str | None = "call-1",
) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(
            id=call_id,
            name=name,
            args={"query": "weather"},
        )
    )


def _response(
    name: str = "lookup",
    *,
    call_id: str | None = "call-1",
    payload: dict[str, Any] | None = None,
) -> types.Part:
    return types.Part(
        function_response=types.FunctionResponse(
            id=call_id,
            name=name,
            response=payload or {"result": "sunny"},
        )
    )


def _call_item(call_id: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "lookup",
        "arguments": json.dumps({"day": "monday"}),
        "status": "completed",
    }


def _stored_call(call_id: str, *, model: str = LUNA) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(
            id=call_id,
            name="lookup",
            args={"day": "monday"},
        ),
        part_metadata=item_metadata(model, _call_item(call_id)),
    )


def _message_item(text: str, *, phase: str = "final_answer") -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _stored_message(
    text: str,
    *,
    model: str = LUNA,
    phase: str = "final_answer",
) -> types.Part:
    return types.Part(
        text=text,
        part_metadata=item_metadata(model, _message_item(text, phase=phase)),
    )


def _reasoning_part(
    *,
    model: str = LUNA,
    item: dict[str, Any] | None = None,
) -> types.Part:
    return types.Part(
        text="",
        thought=True,
        part_metadata=item_metadata(model, item or REASONING_ITEM),
    )


def _output(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(payload),
    }


def _lookup_tool() -> types.Tool:
    return types.Tool(
        function_declarations=[types.FunctionDeclaration(name="lookup")],
    )


def _config(**kwargs: Any) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(**kwargs)


def _http_options(**fields: Any) -> types.HttpOptions:
    options = types.HttpOptions()
    for name, value in fields.items():
        setattr(options, name, value)
    return options


def _build(
    *contents: types.Content,
    model: str = LUNA,
    config: types.GenerateContentConfig | None = None,
    replay_reasoning: bool = True,
) -> ResponsesRequest:
    return build_responses_request(
        LlmRequest(
            contents=list(contents),
            config=config or types.GenerateContentConfig(),
        ),
        model=model,
        replay_reasoning=replay_reasoning,
    )


# User content


def test_user_text_parts_become_one_input_message() -> None:
    request = _build(
        _user(types.Part.from_text(text="hello"), types.Part.from_text(text=" world"))
    )

    assert request.input == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": " world"},
            ],
        }
    ]


def test_thought_and_empty_user_parts_produce_no_message() -> None:
    request = _build(
        _user(types.Part(text="hidden", thought=True), types.Part(text=""))
    )

    assert request.input == []


@pytest.mark.parametrize(
    ("mime_type", "expected_mime"),
    [
        ("image/png", "image/png"),
        ("image/jpg", "image/jpeg"),
        ("IMAGE/JPG", "image/jpeg"),
        ("image/webp", "image/webp"),
    ],
)
def test_inline_images_become_data_urls(mime_type: str, expected_mime: str) -> None:
    request = _build(
        _user(types.Part.from_bytes(data=b"img-bytes", mime_type=mime_type))
    )

    encoded = base64.b64encode(b"img-bytes").decode("ascii")
    assert request.input == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:{expected_mime};base64,{encoded}",
                    "detail": "auto",
                }
            ],
        }
    ]


def test_text_and_image_keep_their_order() -> None:
    request = _build(
        _user(
            types.Part.from_text(text="what is this?"),
            types.Part.from_bytes(data=b"png", mime_type="image/png"),
        )
    )

    content = request.input[0]["content"]
    assert [item["type"] for item in content] == ["input_text", "input_image"]


@pytest.mark.parametrize(
    ("part", "message"),
    [
        pytest.param(
            types.Part.from_bytes(data=b"%PDF", mime_type="application/pdf"),
            "Unsupported inline media type: application/pdf",
            id="pdf",
        ),
        pytest.param(
            types.Part(inline_data=types.Blob(data=b"x", mime_type=None)),
            "Unsupported inline media type: unknown",
            id="no-mime-type",
        ),
        pytest.param(
            types.Part(inline_data=types.Blob(data=None, mime_type="image/png")),
            "Inline image data is missing",
            id="image-without-data",
        ),
        pytest.param(
            types.Part(
                file_data=types.FileData(
                    file_uri="gs://bucket/file.png",
                    mime_type="image/png",
                )
            ),
            "File URI content",
            id="file-uri",
        ),
    ],
)
def test_unsupported_user_media_is_rejected(part: types.Part, message: str) -> None:
    with pytest.raises(OpenAIRequestError, match=message):
        _build(_user(part))


def test_function_responses_become_outputs_before_user_text() -> None:
    request = _build(
        _model(_call(call_id="call-1")),
        _user(_response(call_id="call-1"), types.Part.from_text(text="continue")),
    )

    assert request.input == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query": "weather"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"result": "sunny"}',
        },
        {"role": "user", "content": [{"type": "input_text", "text": "continue"}]},
    ]


def test_empty_tool_response_is_sent_as_an_empty_object() -> None:
    empty_response = types.Part(
        function_response=types.FunctionResponse(
            id="call-1",
            name="lookup",
            response=None,
        )
    )

    request = _build(_model(_call(call_id="call-1")), _user(empty_response))

    assert request.input[1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "{}",
    }


def test_unserializable_tool_response_is_rejected() -> None:
    circular: dict[str, Any] = {}
    circular["self"] = circular

    with pytest.raises(OpenAIRequestError, match="could not be serialized"):
        _build(_model(_call()), _user(_response(payload=circular)))


# Model content without stored items


def test_unstored_model_text_and_calls_are_sent_portably() -> None:
    request = _build(
        _model(
            types.Part(text="private", thought=True),
            types.Part.from_text(text="answer"),
            _call(call_id="call-1"),
        ),
        _user(_response(call_id="call-1")),
    )

    assert request.input == [
        {"role": "assistant", "content": "answer"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query": "weather"}',
        },
        _output("call-1", {"result": "sunny"}),
    ]


def test_empty_model_text_and_empty_contents_are_skipped() -> None:
    request = _build(types.Content(role="model", parts=[]), _model(types.Part(text="")))

    assert request.input == []


def test_call_without_arguments_sends_an_empty_object() -> None:
    part = types.Part(
        function_call=types.FunctionCall(id="call-1", name="ping", args=None)
    )

    request = _build(_model(part), _user(_response(call_id="call-1")))

    assert request.input[0] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "ping",
        "arguments": "{}",
    }


def test_call_without_name_is_rejected() -> None:
    part = types.Part(function_call=types.FunctionCall(name=None, args={}))

    with pytest.raises(OpenAIRequestError, match="missing its name"):
        _build(_model(part))


def test_call_arguments_that_cannot_be_json_are_rejected() -> None:
    circular: dict[str, Any] = {}
    circular["self"] = circular
    part = types.Part(
        function_call=types.FunctionCall(id="call-1", name="lookup", args=circular)
    )

    with pytest.raises(OpenAIRequestError, match="argument could not be serialized"):
        _build(_model(part))


@pytest.mark.parametrize(
    "part",
    [
        pytest.param(
            types.Part.from_bytes(data=b"png", mime_type="image/png"),
            id="inline-image",
        ),
        pytest.param(
            types.Part(
                file_data=types.FileData(
                    file_uri="gs://bucket/file.png",
                    mime_type="image/png",
                )
            ),
            id="file-uri",
        ),
    ],
)
def test_assistant_media_is_rejected(part: types.Part) -> None:
    with pytest.raises(OpenAIRequestError, match="Assistant media"):
        _build(_model(part))


# Replay of stored items


def test_same_model_replays_reasoning_directly_before_parallel_calls() -> None:
    request = _build(
        _model(_reasoning_part(), _stored_call("call_a"), _stored_call("call_b")),
        _user(
            _response(call_id="call_a", payload={"meal": "oats"}),
            _response(call_id="call_b", payload={"meal": "eggs"}),
        ),
    )

    assert request.input == [
        REASONING_ITEM,
        _call_item("call_a"),
        _call_item("call_b"),
        _output("call_a", {"meal": "oats"}),
        _output("call_b", {"meal": "eggs"}),
    ]


def test_empty_summary_reasoning_is_replayed_before_its_message() -> None:
    request = _build(_model(_reasoning_part(), _stored_message("answer")))

    assert request.input == [REASONING_ITEM, _message_item("answer")]


def test_commentary_message_is_replayed_unchanged() -> None:
    request = _build(
        _model(
            _stored_message("let me check", phase="commentary"), _stored_call("call_a")
        )
    )

    assert request.input[0] == _message_item("let me check", phase="commentary")


def test_replayed_items_are_copies_of_the_stored_items() -> None:
    part = _stored_message("answer")

    request = _build(_model(part))
    request.input[0]["id"] = "changed"

    stored = part.part_metadata
    assert stored is not None
    assert stored["blacki_openai"]["item"]["id"] == "msg_1"


def test_model_id_is_normalized_before_matching_stored_items() -> None:
    request = _build(
        _model(_reasoning_part(model=LUNA), _stored_message("answer", model=LUNA)),
        model="openrouter/openai/GPT-6-Luna",
    )

    assert request.model == LUNA
    assert request.input == [REASONING_ITEM, _message_item("answer")]


def test_stored_result_for_a_different_model_is_sent_portably() -> None:
    request = _build(
        _model(
            _reasoning_part(),
            _stored_message("answer"),
            _stored_call("call_a"),
        ),
        _user(_response(call_id="call_a", payload={"meal": "oats"})),
        model=TERRA,
    )

    assert request.input == [
        {"role": "assistant", "content": "answer"},
        {
            "type": "function_call",
            "call_id": "call_a",
            "name": "lookup",
            "arguments": '{"day": "monday"}',
        },
        _output("call_a", {"meal": "oats"}),
    ]


def test_replay_reasoning_false_drops_reasoning_and_keeps_stored_messages() -> None:
    request = _build(
        _model(_reasoning_part(), _stored_message("answer")),
        replay_reasoning=False,
    )

    assert request.input == [_message_item("answer")]


def test_stored_item_without_type_is_sent_portably() -> None:
    part = types.Part(text="answer", part_metadata=item_metadata(LUNA, {"id": "x"}))

    request = _build(_model(part))

    assert request.input == [{"role": "assistant", "content": "answer"}]


def test_stored_call_without_call_id_is_sent_portably_with_its_own_id() -> None:
    part = types.Part(
        function_call=types.FunctionCall(
            id="call-1",
            name="lookup",
            args={"query": "weather"},
        ),
        part_metadata=item_metadata(
            LUNA,
            {"type": "function_call", "name": "lookup", "arguments": "{}"},
        ),
    )

    request = _build(
        _model(part), _user(_response(call_id="call-1", payload={"ok": 1}))
    )

    assert request.input == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query": "weather"}',
        },
        _output("call-1", {"ok": 1}),
    ]


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        pytest.param(
            (_model(_reasoning_part()),),
            [],
            id="last-in-its-content",
        ),
        pytest.param(
            (_model(_reasoning_part(), types.Part.from_text(text="answer")),),
            [{"role": "assistant", "content": "answer"}],
            id="followed-by-unstored-text",
        ),
        pytest.param(
            (_model(_reasoning_part(), _call(call_id="call-x")),),
            [
                {
                    "type": "function_call",
                    "call_id": "call-x",
                    "name": "lookup",
                    "arguments": '{"query": "weather"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-x",
                    "output": MISSING_OUTPUT,
                },
            ],
            id="followed-by-unstored-call",
        ),
        pytest.param(
            (_model(_reasoning_part()), _model(_stored_message("answer"))),
            [_message_item("answer")],
            id="followed-by-message-in-another-content",
        ),
        pytest.param(
            (
                _model(
                    _reasoning_part(item=REASONING_ITEM),
                    _reasoning_part(
                        item={**REASONING_ITEM, "id": "rs_2", "encrypted_content": "e2"}
                    ),
                    _stored_message("answer"),
                ),
            ),
            [
                REASONING_ITEM,
                {**REASONING_ITEM, "id": "rs_2", "encrypted_content": "e2"},
                _message_item("answer"),
            ],
            id="chained-reasoning-kept",
        ),
    ],
)
def test_orphaned_reasoning_is_dropped(
    contents: tuple[types.Content, ...],
    expected: list[dict[str, Any]],
) -> None:
    request = _build(*contents)

    assert request.input == expected


# Call and result pairing


def test_matching_ids_pair_calls_and_results() -> None:
    request = _build(
        _model(_call(call_id="call-1")),
        _user(_response(call_id="call-1", payload={"ok": True})),
    )

    assert request.input[1] == _output("call-1", {"ok": True})


def test_none_ids_pair_by_order_with_one_shared_synthetic_id_per_call() -> None:
    request = _build(
        _model(_call(call_id=None), _call(call_id=None)),
        _user(
            _response(call_id=None, payload={"n": 1}),
            _response(call_id=None, payload={"n": 2}),
        ),
    )

    first_id, second_id = (item["call_id"] for item in request.input[:2])
    assert first_id.startswith("call_blacki_")
    assert second_id.startswith("call_blacki_")
    assert first_id != second_id
    assert request.input[2:] == [
        _output(first_id, {"n": 1}),
        _output(second_id, {"n": 2}),
    ]


def test_result_without_id_pairs_with_the_call_of_the_same_name() -> None:
    request = _build(
        _model(_call("lookup", call_id="call-1"), _call("search", call_id="call-2")),
        _user(_response("search", call_id=None), _response("lookup", call_id=None)),
    )

    call_ids = [item["call_id"] for item in request.input]
    assert call_ids == ["call-1", "call-2", "call-2", "call-1"]


@pytest.mark.parametrize(
    ("result_id", "result_name"),
    [("adk-response-1", "lookup"), ("other-id", "weather")],
)
def test_unmatched_result_id_falls_back_to_name_then_first_pending_call(
    result_id: str,
    result_name: str,
) -> None:
    request = _build(
        _model(_call("lookup", call_id="call-1")),
        _user(_response(result_name, call_id=result_id)),
    )

    assert request.input[-1]["call_id"] == "call-1"


def test_result_without_pending_call_keeps_its_own_id() -> None:
    request = _build(_user(_response(call_id="call-9")))

    assert request.input == [_output("call-9", {"result": "sunny"})]


def test_result_without_id_or_pending_call_gets_a_fresh_id() -> None:
    request = _build(_user(_response(call_id=None)))

    assert request.input[0]["call_id"].startswith("call_blacki_")


def test_unanswered_call_gets_placeholder_before_next_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        request = _build(
            _model(_call(call_id="call-1")),
            _user(types.Part.from_text(text="continue")),
        )

    assert request.input == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query": "weather"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": MISSING_OUTPUT,
        },
        {"role": "user", "content": [{"type": "input_text", "text": "continue"}]},
    ]
    assert "Missing tool results" in caplog.text


def test_unanswered_call_at_the_end_gets_placeholder() -> None:
    request = _build(_model(_call(call_id="call-1")))

    assert request.input[-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": MISSING_OUTPUT,
    }


def test_only_unanswered_parallel_call_gets_placeholder() -> None:
    request = _build(
        _model(_call(call_id="call-1"), _call(call_id="call-2")),
        _user(
            _response(call_id="call-1"),
            types.Part.from_text(text="next"),
        ),
    )

    assert [item.get("call_id") for item in request.input[:4]] == [
        "call-1",
        "call-2",
        "call-1",
        "call-2",
    ]
    assert request.input[3]["output"] == MISSING_OUTPUT
    assert request.input[4] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "next"}],
    }


# Tools


def test_function_declarations_become_responses_tools() -> None:
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="lookup",
                description="Look something up",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"query": types.Schema(type=types.Type.STRING)},
                    required=["query"],
                ),
            )
        ]
    )

    request = _build(
        _user(types.Part.from_text(text="hi")), config=_config(tools=[tool])
    )

    assert request.tools == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look something up",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "strict": False,
        }
    ]


def test_property_named_type_keeps_its_schema_normalized() -> None:
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="lookup",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "type": types.Schema(
                            type=types.Type.STRING,
                            enum=["meal", "snack"],
                        )
                    },
                ),
            )
        ]
    )

    request = _build(
        _user(types.Part.from_text(text="hi")), config=_config(tools=[tool])
    )

    assert request.tools[0]["parameters"]["properties"] == {
        "type": {"type": "string", "enum": ["meal", "snack"]}
    }


def test_parameters_json_schema_is_used_unchanged() -> None:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(name="lookup", parameters_json_schema=schema)
        ]
    )

    request = _build(
        _user(types.Part.from_text(text="hi")), config=_config(tools=[tool])
    )

    assert request.tools == [
        {
            "type": "function",
            "name": "lookup",
            "description": "",
            "parameters": schema,
            "strict": False,
        }
    ]


def test_declaration_without_parameters_gets_an_empty_object_schema() -> None:
    tool = types.Tool(function_declarations=[types.FunctionDeclaration(name="ping")])

    request = _build(
        _user(types.Part.from_text(text="hi")), config=_config(tools=[tool])
    )

    assert request.tools == [
        {
            "type": "function",
            "name": "ping",
            "description": "",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]


def test_declaration_without_name_is_rejected() -> None:
    tool = types.Tool(function_declarations=[types.FunctionDeclaration(name=None)])

    with pytest.raises(OpenAIRequestError, match="missing its name"):
        _build(_user(types.Part.from_text(text="hi")), config=_config(tools=[tool]))


def test_non_function_tool_is_rejected() -> None:
    tool = types.Tool(google_search=types.GoogleSearch())

    with pytest.raises(OpenAIRequestError, match="function tools only"):
        _build(_user(types.Part.from_text(text="hi")), config=_config(tools=[tool]))


def test_empty_tool_is_skipped() -> None:
    request = _build(
        _user(types.Part.from_text(text="hi")),
        config=_config(tools=[types.Tool()]),
    )

    assert request.tools == []
    assert request.tool_choice is None


def test_unsupported_tool_object_is_rejected() -> None:
    with pytest.raises(OpenAIRequestError, match="Unsupported tool type: object"):
        _tool_declarations([object()])


@pytest.mark.parametrize(
    ("mode", "allowed_names", "expected"),
    [
        pytest.param(types.FunctionCallingConfigMode.AUTO, None, "auto", id="auto"),
        pytest.param(types.FunctionCallingConfigMode.NONE, None, "none", id="none"),
        pytest.param(types.FunctionCallingConfigMode.ANY, None, "required", id="any"),
        pytest.param(
            types.FunctionCallingConfigMode.ANY,
            ["lookup"],
            {"type": "function", "name": "lookup"},
            id="any-one-name",
        ),
        pytest.param(
            types.FunctionCallingConfigMode.ANY,
            ["lookup", "search"],
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "function", "name": "lookup"},
                    {"type": "function", "name": "search"},
                ],
            },
            id="any-several-names",
        ),
        pytest.param(
            types.FunctionCallingConfigMode.AUTO,
            ["lookup"],
            "auto",
            id="auto-ignores-names",
        ),
        pytest.param(
            types.FunctionCallingConfigMode.MODE_UNSPECIFIED,
            None,
            None,
            id="unspecified-mode",
        ),
    ],
)
def test_tool_config_maps_to_tool_choice(
    mode: types.FunctionCallingConfigMode,
    allowed_names: list[str] | None,
    expected: object,
) -> None:
    config = _config(
        tools=[_lookup_tool()],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=mode,
                allowed_function_names=allowed_names,
            )
        ),
    )

    request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.tool_choice == expected


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            _config(
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                    )
                )
            ),
            id="no-tools",
        ),
        pytest.param(_config(tools=[_lookup_tool()]), id="no-tool-config"),
        pytest.param(
            _config(tools=[_lookup_tool()], tool_config=types.ToolConfig()),
            id="no-calling-config",
        ),
    ],
)
def test_tool_choice_is_omitted_without_tools_or_calling_config(
    config: types.GenerateContentConfig,
) -> None:
    request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.tool_choice is None


# Request kwargs and the request object


def test_create_kwargs_never_stores_or_chains_responses() -> None:
    request = _build(_user(types.Part.from_text(text="hi")))

    for stream in (False, True):
        kwargs = request.create_kwargs(stream=stream)

        assert kwargs == {
            "model": LUNA,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "stream": stream,
            "reasoning": {"summary": "auto"},
        }
        assert "previous_response_id" not in kwargs
        assert "conversation" not in kwargs


def test_create_kwargs_omits_optional_values_that_are_unset() -> None:
    request = ResponsesRequest(model=LUNA, input=[])

    assert request.create_kwargs(stream=False) == {
        "model": LUNA,
        "input": [],
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "stream": False,
    }


def test_create_kwargs_includes_every_configured_field() -> None:
    config = _config(
        system_instruction="Be concise",
        tools=[_lookup_tool()],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.AUTO,
            )
        ),
        max_output_tokens=321,
        temperature=0.2,
        top_p=0.9,
        http_options=_http_options(
            extra_body={"reasoning": {"effort": "none"}, "custom": 1},
            headers={"X-Test": "yes"},
        ),
    )

    request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.create_kwargs(stream=True) == {
        "model": LUNA,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "stream": True,
        "instructions": "Be concise",
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        "tool_choice": "auto",
        "reasoning": {"effort": "none", "summary": "auto"},
        "max_output_tokens": 321,
        "temperature": 0.2,
        "top_p": 0.9,
        "extra_body": {"custom": 1},
        "extra_headers": {"X-Test": "yes"},
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"store": True}, "never asks OpenAI to store", id="store"),
        pytest.param(
            {"include": ["message.output_text.logprobs"]},
            "include must contain reasoning.encrypted_content",
            id="include",
        ),
    ],
)
def test_responses_request_refuses_storage_and_missing_encrypted_reasoning(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(OpenAIRequestError, match=message):
        ResponsesRequest(model=LUNA, input=[], **overrides)


# System instruction


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("  hello  ", "hello"),
        ("   ", None),
        (types.Content(role="system", parts=[types.Part.from_text(text="hi")]), "hi"),
        (types.Content(role="system", parts=[]), None),
        (
            types.Content(
                role="system",
                parts=[
                    types.Part(text="secret", thought=True),
                    types.Part.from_text(text="shown"),
                ],
            ),
            "shown",
        ),
        ([types.Part.from_text(text="a"), types.Part.from_text(text="b")], "ab"),
        ((types.Part.from_text(text="t"),), "t"),
    ],
)
def test_system_instruction_variants(value: object, expected: str | None) -> None:
    assert _system_instruction_text(value) == expected


def test_system_instruction_rejects_media_and_unknown_types() -> None:
    media = types.Content(
        role="system",
        parts=[types.Part.from_bytes(data=b"png", mime_type="image/png")],
    )

    with pytest.raises(OpenAIRequestError, match="Media parts"):
        _system_instruction_text(media)
    with pytest.raises(OpenAIRequestError, match="Unsupported system_instruction"):
        _system_instruction_text(object())


# Reasoning settings


def test_reasoning_effort_is_sent_and_removed_from_extra_body() -> None:
    config = _config(
        http_options=_http_options(
            extra_body={"reasoning": {"effort": "high"}, "custom_flag": True},
        )
    )

    request = _build(_user(types.Part.from_text(text="hi")), model=TERRA, config=config)

    assert request.reasoning == {"effort": "high", "summary": "auto"}
    assert request.extra_body == {"custom_flag": True}


@pytest.mark.parametrize(
    "extra_body",
    [
        pytest.param(None, id="no-extra-body"),
        pytest.param({}, id="empty-extra-body"),
        pytest.param({"reasoning": None}, id="null-reasoning"),
        pytest.param({"reasoning": {}}, id="empty-reasoning"),
    ],
)
def test_reasoning_without_effort_requests_summaries_only(
    extra_body: dict[str, Any] | None,
) -> None:
    config = _config(http_options=_http_options(extra_body=extra_body))

    request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.reasoning == {"summary": "auto"}


@pytest.mark.parametrize(
    ("reasoning", "message"),
    [
        pytest.param("high", "reasoning must be a mapping", id="not-a-mapping"),
        pytest.param({"effort": ""}, "non-empty string", id="empty-effort"),
        pytest.param({"effort": 3}, "non-empty string", id="numeric-effort"),
    ],
)
def test_invalid_reasoning_settings_are_rejected(
    reasoning: object, message: str
) -> None:
    config = _config(http_options=_http_options(extra_body={"reasoning": reasoning}))

    with pytest.raises(OpenAIRequestError, match=message):
        _build(_user(types.Part.from_text(text="hi")), config=config)


def test_unused_reasoning_keys_are_ignored_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(
        http_options=_http_options(
            extra_body={"reasoning": {"effort": "low", "max_tokens": 512}},
        )
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.reasoning == {"effort": "low", "summary": "auto"}
    assert "max_tokens" in caplog.text


# Sampling parameters


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        pytest.param(LUNA, "none", id="luna-none"),
        pytest.param(TERRA, "none", id="terra-none"),
    ],
)
def test_sampling_is_sent_at_effort_none_on_models_that_accept_it(
    model: str,
    effort: str,
) -> None:
    config = _config(
        temperature=0.2,
        top_p=0.9,
        http_options=_http_options(extra_body={"reasoning": {"effort": effort}}),
    )

    request = _build(_user(types.Part.from_text(text="hi")), model=model, config=config)

    kwargs = request.create_kwargs(stream=False)
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        pytest.param(SOL, "none", id="sol-rejects-sampling-at-none"),
        pytest.param(ASTRA, "none", id="astra-rejects-sampling-at-none"),
        pytest.param(LUNA, "high", id="luna-with-effort-high"),
        pytest.param(LUNA, None, id="luna-effort-unset"),
        pytest.param("gpt-5", "none", id="model-outside-catalog"),
    ],
)
def test_sampling_is_dropped_with_a_warning_when_not_accepted(
    model: str,
    effort: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    http_options = (
        _http_options(extra_body={"reasoning": {"effort": effort}}) if effort else None
    )
    config = _config(temperature=0.2, top_p=0.9, http_options=http_options)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        request = _build(
            _user(types.Part.from_text(text="hi")),
            model=model,
            config=config,
        )

    kwargs = request.create_kwargs(stream=False)
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "Dropping temperature, top_p" in caplog.text


def test_no_sampling_warning_when_none_is_configured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(
        http_options=_http_options(extra_body={"reasoning": {"effort": "high"}}),
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _build(_user(types.Part.from_text(text="hi")), model=TERRA, config=config)

    assert "Dropping" not in caplog.text


# Reserved keys, headers and HTTP options


@pytest.mark.parametrize(
    "key",
    [
        "model",
        "input",
        "instructions",
        "stream",
        "tools",
        "tool_choice",
        "store",
        "previous_response_id",
        "conversation",
        "include",
        "text",
        "max_output_tokens",
        "temperature",
        "top_p",
        "STORE",
    ],
)
def test_reserved_extra_body_keys_are_filtered_with_a_warning(
    key: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(
        http_options=_http_options(extra_body={key: "override", "custom": 1}),
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.extra_body == {"custom": 1}
    assert f"reserved extra_body key {key!r}" in caplog.text


def test_caller_headers_drop_authorization_and_content_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(
        http_options=_http_options(
            headers={
                "X-Test": "yes",
                "Authorization": "Bearer secret",
                "content-type": "text/plain",
            }
        )
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.extra_headers == {"X-Test": "yes"}
    assert "Authorization" in caplog.text
    assert "content-type" in caplog.text
    assert "Bearer secret" not in caplog.text


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"extra_body": ["not", "a", "mapping"]}, id="extra-body-list"),
        pytest.param({"headers": ["not", "a", "mapping"]}, id="headers-list"),
        pytest.param({"timeout": 0}, id="zero-timeout"),
        pytest.param(
            {"retry_options": types.HttpRetryOptions(attempts=0)},
            id="zero-attempts",
        ),
    ],
)
def test_invalid_http_options_are_rejected(options: dict[str, Any]) -> None:
    config = _config(http_options=_http_options(**options))

    with pytest.raises(OpenAIRequestError):
        _build(_user(types.Part.from_text(text="hi")), config=config)


def test_timeout_and_retry_attempts_map_to_seconds_and_retries() -> None:
    config = _config(
        http_options=_http_options(
            timeout=60_000,
            retry_options=types.HttpRetryOptions(attempts=3),
        )
    )

    request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.timeout_seconds == 60.0
    assert request.max_retries == 2


def test_retry_options_without_attempts_leave_retries_unset() -> None:
    config = _config(
        http_options=_http_options(retry_options=types.HttpRetryOptions()),
    )

    request = _build(_user(types.Part.from_text(text="hi")), config=config)

    assert request.timeout_seconds is None
    assert request.max_retries is None


# Unsupported and ignored configuration


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            types.GenerateContentConfig(response_schema={"type": "object"}),
            id="response-schema",
        ),
        pytest.param(
            types.GenerateContentConfig(response_json_schema={"type": "object"}),
            id="response-json-schema",
        ),
        pytest.param(
            types.GenerateContentConfig(response_mime_type="application/json"),
            id="response-mime-type",
        ),
        pytest.param(
            types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    )
                ]
            ),
            id="safety-settings",
        ),
        pytest.param(
            types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
            id="thinking-config",
        ),
        pytest.param(
            types.GenerateContentConfig(candidate_count=2),
            id="candidate-count",
        ),
    ],
)
def test_unsupported_configuration_is_rejected(
    config: types.GenerateContentConfig,
) -> None:
    with pytest.raises(OpenAIRequestError, match="does not support|only supports"):
        _build(config=config)


@pytest.mark.parametrize(
    ("config", "name"),
    [
        pytest.param(types.GenerateContentConfig(top_k=5), "top_k", id="top-k"),
        pytest.param(
            types.GenerateContentConfig(stop_sequences=["END"]),
            "stop_sequences",
            id="stop-sequences",
        ),
        pytest.param(
            types.GenerateContentConfig(presence_penalty=0.1),
            "presence_penalty",
            id="presence-penalty",
        ),
        pytest.param(
            types.GenerateContentConfig(frequency_penalty=0.2),
            "frequency_penalty",
            id="frequency-penalty",
        ),
        pytest.param(types.GenerateContentConfig(seed=7), "seed", id="seed"),
    ],
)
def test_generation_settings_the_responses_api_lacks_are_ignored(
    config: types.GenerateContentConfig,
    name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        request = _build(_user(types.Part.from_text(text="hi")), config=config)

    kwargs = request.create_kwargs(stream=False)
    assert kwargs.keys().isdisjoint(
        {"top_k", "stop", "presence_penalty", "frequency_penalty", "seed"}
    )
    assert name in caplog.text
