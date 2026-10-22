"""Tests for translating ADK requests into OpenRouter payloads."""

from __future__ import annotations

from typing import Any, cast

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from blacki.models.openrouter_request import (
    ChatRequest,
    OpenRouterRequestError,
    _system_instruction_text,
    _tool_declarations,
    build_chat_request,
    contents_to_messages,
)


def _user(*parts: types.Part) -> types.Content:
    return types.Content(role="user", parts=list(parts))


def _model(*parts: types.Part) -> types.Content:
    return types.Content(role="model", parts=list(parts))


def _call(
    name: str = "lookup",
    *,
    call_id: str | None = "call-1",
    args: dict[str, Any] | None = None,
) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(
            id=call_id,
            name=name,
            args=args or {"query": "weather"},
        )
    )


def _response(
    name: str = "lookup",
    *,
    call_id: str | None = "call-1",
    response: dict[str, Any] | None = None,
) -> types.Part:
    return types.Part(
        function_response=types.FunctionResponse(
            id=call_id,
            name=name,
            response=response or {"result": "sunny"},
        )
    )


def test_text_only_user_content_uses_string() -> None:
    messages = contents_to_messages(
        [_user(types.Part.from_text(text="hello"), types.Part.from_text(text=" world"))]
    )

    assert messages == [{"role": "user", "content": "hello world"}]


def test_image_content_uses_data_url_and_normalizes_jpg() -> None:
    messages = contents_to_messages(
        [
            _user(
                types.Part.from_text(text="what is this?"),
                types.Part.from_bytes(data=b"image-bytes", mime_type="image/jpg"),
            )
        ]
    )

    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_assistant_tool_call_and_tool_response_keep_matching_ids() -> None:
    messages = contents_to_messages(
        [
            _model(_call(call_id="call-1")),
            _user(_response(call_id="call-1")),
        ]
    )

    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["id"] == "call-1"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"result": "sunny"}',
    }


def test_missing_ids_are_synthesized_and_paired_by_name() -> None:
    messages = contents_to_messages(
        [
            _model(_call(call_id=None)),
            _user(_response(call_id=None)),
        ]
    )

    call_id = messages[0]["tool_calls"][0]["id"]
    assert call_id.startswith("blacki_call_")
    assert messages[1]["tool_call_id"] == call_id


def test_mismatched_response_id_falls_back_to_name_pairing() -> None:
    messages = contents_to_messages(
        [
            _model(_call(call_id="blacki-call-1")),
            _user(_response(call_id="adk-response-1")),
        ]
    )

    assert messages[0]["tool_calls"][0]["id"] == "blacki-call-1"
    assert messages[1]["tool_call_id"] == "blacki-call-1"


def test_missing_tool_response_gets_placeholder_before_next_message() -> None:
    messages = contents_to_messages(
        [
            _model(_call(call_id="call-1")),
            _user(types.Part.from_text(text="continue")),
        ]
    )

    assert [message["role"] for message in messages] == ["assistant", "tool", "user"]
    assert messages[1]["tool_call_id"] == "call-1"


def test_mixed_function_response_and_image_splits_into_tool_and_user_messages() -> None:
    messages = contents_to_messages(
        [
            _model(_call(call_id="call-1")),
            _user(
                _response(call_id="call-1"),
                types.Part.from_bytes(data=b"png", mime_type="image/png"),
            ),
        ]
    )

    assert [message["role"] for message in messages] == ["assistant", "tool", "user"]
    user_content = messages[2]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "image_url"


def test_reasoning_parts_are_preserved_on_assistant_messages() -> None:
    messages = contents_to_messages(
        [
            _model(
                types.Part(text="private reasoning", thought=True),
                types.Part.from_text(text="public answer"),
            )
        ]
    )

    assert messages[0] == {
        "role": "assistant",
        "content": "public answer",
        "reasoning": "private reasoning",
    }


def test_empty_contents_are_skipped() -> None:
    assert contents_to_messages([types.Content(role="user", parts=[])]) == []


def test_assistant_media_is_rejected() -> None:
    with pytest.raises(OpenRouterRequestError, match="Assistant media"):
        contents_to_messages(
            [
                _model(
                    types.Part.from_bytes(data=b"png", mime_type="image/png"),
                )
            ]
        )


def test_function_call_without_name_is_rejected() -> None:
    with pytest.raises(OpenRouterRequestError, match="missing its name"):
        contents_to_messages(
            [
                _model(
                    types.Part(
                        function_call=types.FunctionCall(name=None, args={}),
                    )
                )
            ]
        )


def test_non_image_inline_data_is_rejected() -> None:
    with pytest.raises(OpenRouterRequestError, match="Unsupported inline media"):
        contents_to_messages(
            [
                _user(
                    types.Part.from_bytes(
                        data=b"%PDF",
                        mime_type="application/pdf",
                    )
                )
            ]
        )


def test_inline_image_without_data_is_rejected() -> None:
    with pytest.raises(OpenRouterRequestError, match="Inline image data"):
        contents_to_messages(
            [
                _user(
                    types.Part(
                        inline_data=types.Blob(
                            data=None,
                            mime_type="image/png",
                        )
                    )
                )
            ]
        )


def test_file_uri_content_is_rejected() -> None:
    with pytest.raises(OpenRouterRequestError, match="File URI"):
        contents_to_messages(
            [
                _user(
                    types.Part(
                        file_data=types.FileData(
                            file_uri="gs://bucket/file.png",
                            mime_type="image/png",
                        )
                    )
                )
            ]
        )


def test_unserializable_tool_response_is_rejected() -> None:
    circular: dict[str, Any] = {}
    circular["self"] = circular

    with pytest.raises(OpenRouterRequestError, match="could not be serialized"):
        contents_to_messages(
            [
                _model(_call()),
                _user(_response(response=circular)),
            ]
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("  hello  ", "hello"),
        (types.Content(role="system", parts=[types.Part.from_text(text="hi")]), "hi"),
        ([types.Part.from_text(text="a"), types.Part.from_text(text="b")], "ab"),
    ],
)
def test_system_instruction_variants(value: object, expected: str | None) -> None:
    assert _system_instruction_text(value) == expected


def test_system_instruction_rejects_media_and_unknown_types() -> None:
    with pytest.raises(OpenRouterRequestError, match="Media parts"):
        _system_instruction_text(
            types.Content(
                role="system",
                parts=[types.Part.from_bytes(data=b"png", mime_type="image/png")],
            )
        )
    with pytest.raises(OpenRouterRequestError, match="Unsupported system_instruction"):
        _system_instruction_text(object())


def test_build_chat_request_maps_supported_configuration() -> None:
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="lookup",
                description="Look something up",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]
    )
    config = types.GenerateContentConfig(
        system_instruction=types.Content(
            role="system",
            parts=[types.Part.from_text(text="Be concise")],
        ),
        temperature=0.2,
        max_output_tokens=123,
        top_p=0.9,
        top_k=5,
        stop_sequences=["END"],
        presence_penalty=0.1,
        frequency_penalty=0.2,
        seed=7,
        tools=[tool],
        http_options=types.HttpOptions(
            extra_body={
                "reasoning": {"effort": "high"},
                "model": "evil",
                "top_k": 9,
            },
            headers={"X-Test": "yes", "Authorization": "Bearer evil"},
            timeout=60_000,
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
    )

    request = build_chat_request(
        LlmRequest(
            contents=[_user(types.Part.from_text(text="hello"))],
            config=config,
        ),
        model="openrouter/google/gemini-2.5-flash",
    )

    assert request.model == "google/gemini-2.5-flash"
    assert request.messages[0] == {"role": "system", "content": "Be concise"}
    assert request.generation == {
        "temperature": 0.2,
        "max_tokens": 123,
        "top_p": 0.9,
        "stop": ["END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 7,
    }
    assert request.tools is not None
    assert request.tools[0]["function"]["name"] == "lookup"
    assert request.extra_body == {"reasoning": {"effort": "high"}, "top_k": 5}
    assert request.extra_headers == {"X-Test": "yes"}
    assert request.timeout_seconds == 60.0
    assert request.max_retries == 2

    kwargs = request.create_kwargs(stream=True)
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["extra_body"] == {"reasoning": {"effort": "high"}, "top_k": 5}


@pytest.mark.parametrize(
    "config",
    [
        types.GenerateContentConfig(response_schema={"type": "object"}),
        types.GenerateContentConfig(response_mime_type="application/json"),
        types.GenerateContentConfig(
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO
                )
            )
        ),
        types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                )
            ]
        ),
        types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
        types.GenerateContentConfig(candidate_count=2),
    ],
)
def test_build_chat_request_rejects_unsupported_configuration(
    config: types.GenerateContentConfig,
) -> None:
    with pytest.raises(OpenRouterRequestError, match="does not support|only supports"):
        build_chat_request(
            LlmRequest(contents=[], config=config),
            model="google/gemini-2.5-flash",
        )


def test_build_chat_request_rejects_invalid_http_options() -> None:
    bad_extra_body = types.HttpOptions()
    bad_extra_body.extra_body = cast(Any, ["not", "a", "mapping"])
    with pytest.raises(OpenRouterRequestError, match="extra_body must be a mapping"):
        build_chat_request(
            LlmRequest(
                contents=[],
                config=types.GenerateContentConfig(http_options=bad_extra_body),
            ),
            model="google/gemini-2.5-flash",
        )

    bad_headers = types.HttpOptions()
    bad_headers.headers = cast(Any, ["not", "a", "mapping"])
    with pytest.raises(OpenRouterRequestError, match="headers must be a mapping"):
        build_chat_request(
            LlmRequest(
                contents=[],
                config=types.GenerateContentConfig(http_options=bad_headers),
            ),
            model="google/gemini-2.5-flash",
        )

    for http_options in (
        types.HttpOptions(timeout=0),
        types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=0)),
    ):
        with pytest.raises(OpenRouterRequestError, match="must be"):
            build_chat_request(
                LlmRequest(
                    contents=[],
                    config=types.GenerateContentConfig(http_options=http_options),
                ),
                model="google/gemini-2.5-flash",
            )


def test_tool_declarations_from_json_schema() -> None:
    tools = _tool_declarations(
        [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="lookup",
                        description="d",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    )
                ]
            )
        ]
    )

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]


def test_tool_declarations_lowercase_legacy_schema_types() -> None:
    tools = _tool_declarations(
        [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="lookup",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={"query": types.Schema(type=types.Type.STRING)},
                        ),
                    )
                ]
            )
        ]
    )

    assert tools is not None
    parameters = tools[0]["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["query"]["type"] == "string"


def test_tool_declarations_rejects_builtin_and_non_tool_values() -> None:
    with pytest.raises(OpenRouterRequestError, match="function tools only"):
        _tool_declarations([types.Tool(google_search=types.GoogleSearch())])

    with pytest.raises(OpenRouterRequestError, match="Unsupported tool type"):
        _tool_declarations([object()])


def test_tool_declarations_skips_empty_tools() -> None:
    assert _tool_declarations([types.Tool()]) is None
    assert _tool_declarations(None) is None


def test_chat_request_create_kwargs_omits_optional_fields() -> None:
    request = ChatRequest(model="model", messages=[{"role": "user", "content": "hi"}])

    kwargs = request.create_kwargs(stream=False)

    assert kwargs == {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def test_chat_request_create_kwargs_includes_tools_and_headers() -> None:
    request = ChatRequest(
        model="model",
        messages=[],
        tools=[{"type": "function", "function": {"name": "x"}}],
        extra_headers={"X-Test": "yes"},
    )

    kwargs = request.create_kwargs(stream=False)

    assert kwargs["tools"] == [{"type": "function", "function": {"name": "x"}}]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["extra_headers"] == {"X-Test": "yes"}
    assert "stream_options" not in kwargs


def test_empty_model_part_is_skipped() -> None:
    assert contents_to_messages([_model(types.Part())]) == []


def test_user_thought_part_is_skipped() -> None:
    assert contents_to_messages([_user(types.Part(text="thought", thought=True))]) == []


def test_empty_user_part_is_skipped() -> None:
    assert contents_to_messages([_user(types.Part())]) == []


def test_system_instruction_thought_part_is_ignored() -> None:
    assert (
        _system_instruction_text(
            types.Content(
                role="system",
                parts=[types.Part(text="thought", thought=True)],
            )
        )
        is None
    )


def test_string_tool_response_is_passed_through() -> None:
    response = types.FunctionResponse(
        name="lookup",
        id="call-1",
        response={"result": "ok"},
    )
    response.response = cast(Any, "plain text")

    messages = contents_to_messages(
        [_model(_call()), _user(types.Part(function_response=response))]
    )

    assert messages[1]["content"] == "plain text"


def test_multiple_pending_calls_pair_by_name() -> None:
    messages = contents_to_messages(
        [
            _model(
                _call(name="first", call_id="c1"),
                _call(name="lookup", call_id="c2"),
            ),
            _user(_response(name="lookup", call_id=None)),
        ]
    )

    assert messages[1]["tool_call_id"] == "c2"


def test_unmatched_tool_response_uses_first_pending_call() -> None:
    messages = contents_to_messages(
        [
            _model(_call(name="first", call_id="c1")),
            _user(_response(name="other", call_id="unknown")),
        ]
    )

    assert messages[1]["tool_call_id"] == "c1"


def test_orphan_tool_response_gets_fallback_id() -> None:
    messages = contents_to_messages([_user(_response(name="lookup", call_id=None))])

    assert messages[0]["tool_call_id"] == "blacki_tool_0"


def test_trailing_tool_call_gets_placeholder() -> None:
    messages = contents_to_messages([_model(_call(call_id="call-1"))])

    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call-1"


def test_ensure_tool_results_ignores_orphan_tool_ids() -> None:
    from blacki.models.openrouter_request import _ensure_tool_results

    messages = _ensure_tool_results(
        [{"role": "tool", "tool_call_id": "orphan", "content": "{}"}]
    )

    assert messages == [{"role": "tool", "tool_call_id": "orphan", "content": "{}"}]


def test_tool_declaration_without_name_is_rejected() -> None:
    declaration = types.FunctionDeclaration(name="placeholder")
    declaration.name = None

    with pytest.raises(OpenRouterRequestError, match="missing its name"):
        _tool_declarations([types.Tool(function_declarations=[declaration])])


def test_tool_declaration_without_parameters_gets_empty_schema() -> None:
    tools = _tool_declarations(
        [types.Tool(function_declarations=[types.FunctionDeclaration(name="x")])]
    )

    assert tools is not None
    assert tools[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_lowercase_schema_types_handles_lists_and_scalars() -> None:
    from blacki.models.openrouter_request import (
        _lowercase_schema_types,
        _normalize_schema_type,
    )

    assert _lowercase_schema_types(
        {"type": "OBJECT", "enum": ["A", "B"], "description": "d"}
    ) == {"type": "object", "enum": ["A", "B"], "description": "d"}
    assert _normalize_schema_type("STRING") == "string"
    assert _normalize_schema_type(5) == 5


def test_circular_function_call_args_are_rejected() -> None:
    circular: dict[str, Any] = {}
    circular["self"] = circular

    with pytest.raises(OpenRouterRequestError, match="function-call argument"):
        contents_to_messages([_model(_call(args=circular))])
