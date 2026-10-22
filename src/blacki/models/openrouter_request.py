"""Translate ADK model requests into OpenRouter chat-completion payloads.

The conversion intentionally supports the features Blacki uses: system
instructions, text, inline images, function tools, function calls, function
responses, reasoning settings carried in ``http_options.extra_body``, and the
standard sampling parameters.  Provider-specific or structured-output
configuration is rejected explicitly instead of being silently ignored.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from .openrouter_ids import canonical_openrouter_model_id

logger = logging.getLogger(__name__)

_MISSING_TOOL_RESULT = '{"error":"Missing tool response."}'

# The SDK merges ``extra_body`` over explicit keyword arguments.  These keys
# define the conversation itself, so a stray extra_body value must never be
# able to replace them.
_RESERVED_EXTRA_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "response_format",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    }
)
_FILTERED_HEADER_NAMES = frozenset({"authorization", "content-type"})


class OpenRouterRequestError(ValueError):
    """Raised when an ADK request cannot be represented as an OpenRouter call."""


@dataclass(slots=True)
class ChatRequest:
    """One prepared OpenRouter chat-completions call."""

    model: str
    messages: list[dict[str, Any]]
    generation: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    max_retries: int | None = None

    def create_kwargs(self, *, stream: bool) -> dict[str, Any]:
        """Return keyword arguments for ``chat.completions.create``."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            **self.generation,
            "stream": stream,
        }
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        if self.extra_headers:
            kwargs["extra_headers"] = dict(self.extra_headers)
        return kwargs


def build_chat_request(
    llm_request: LlmRequest,
    *,
    model: str,
) -> ChatRequest:
    """Build one OpenRouter request from an ADK ``LlmRequest``."""

    config = llm_request.config
    _reject_unsupported_config(config)

    messages = contents_to_messages(llm_request.contents)
    system_text = _system_instruction_text(config.system_instruction)
    if system_text:
        messages.insert(0, {"role": "system", "content": system_text})

    extra_body, extra_headers, timeout_seconds, max_retries = _http_options(
        config.http_options
    )
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k

    return ChatRequest(
        model=canonical_openrouter_model_id(model),
        messages=messages,
        generation=_generation_params(config),
        tools=_tool_declarations(config.tools),
        extra_body=extra_body,
        extra_headers=extra_headers,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def contents_to_messages(
    contents: Sequence[types.Content],
) -> list[dict[str, Any]]:
    """Convert ADK conversation contents into OpenAI chat messages.

    Function responses become ``role=tool`` messages.  When a content block
    mixes function responses with text or images (ADK's multimodal tool-result
    plugin does this), the tool messages are emitted first and the remaining
    parts become a separate user message.
    """

    messages: list[dict[str, Any]] = []
    pending_calls: list[tuple[str | None, str]] = []
    fallback_counter = 0

    for content in contents:
        parts = content.parts or []
        if not parts:
            continue

        if content.role == "model":
            message, calls = _assistant_message(parts)
            if message is not None:
                messages.append(message)
            pending_calls.extend(calls)
            continue

        for part in parts:
            response = part.function_response
            if response is None:
                continue
            call_id = _resolve_tool_response_id(
                response,
                pending_calls,
                fallback_counter,
            )
            fallback_counter += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _tool_response_text(response),
                }
            )

        remaining = [part for part in parts if part.function_response is None]
        if remaining:
            user_message = _user_message(remaining)
            if user_message is not None:
                messages.append(user_message)

    return _ensure_tool_results(messages)


def _reject_unsupported_config(config: types.GenerateContentConfig) -> None:
    unsupported = {
        "response_schema": config.response_schema,
        "response_mime_type": config.response_mime_type,
        "tool_config": config.tool_config,
        "safety_settings": config.safety_settings,
        "thinking_config": config.thinking_config,
    }
    for name, value in unsupported.items():
        if value is not None:
            raise OpenRouterRequestError(
                f"Blacki's OpenRouter adapter does not support {name}."
            )
    candidate_count = config.candidate_count
    if candidate_count is not None and candidate_count != 1:
        raise OpenRouterRequestError(
            "Blacki's OpenRouter adapter only supports candidate_count=1."
        )


def _generation_params(config: types.GenerateContentConfig) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.temperature is not None:
        params["temperature"] = config.temperature
    if config.max_output_tokens is not None:
        params["max_tokens"] = config.max_output_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = list(config.stop_sequences)
    if config.presence_penalty is not None:
        params["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty is not None:
        params["frequency_penalty"] = config.frequency_penalty
    if config.seed is not None:
        params["seed"] = config.seed
    return params


def _http_options(
    http_options: types.HttpOptions | None,
) -> tuple[dict[str, Any], dict[str, str], float | None, int | None]:
    if http_options is None:
        return {}, {}, None, None

    extra_body: dict[str, Any] = {}
    if http_options.extra_body is not None:
        if not isinstance(http_options.extra_body, Mapping):
            raise OpenRouterRequestError("http_options.extra_body must be a mapping.")
        for key, value in http_options.extra_body.items():
            key_text = str(key)
            if key_text.lower() in _RESERVED_EXTRA_BODY_KEYS:
                logger.warning(
                    "Ignoring reserved extra_body key %r for OpenRouter",
                    key_text,
                )
                continue
            extra_body[key_text] = value

    headers: dict[str, str] = {}
    if http_options.headers is not None:
        if not isinstance(http_options.headers, Mapping):
            raise OpenRouterRequestError("http_options.headers must be a mapping.")
        for key, value in http_options.headers.items():
            key_text = str(key)
            if key_text.lower() in _FILTERED_HEADER_NAMES:
                logger.warning(
                    "Ignoring caller-supplied %r header for OpenRouter",
                    key_text,
                )
                continue
            headers[key_text] = str(value)

    timeout_seconds: float | None = None
    if http_options.timeout is not None:
        if http_options.timeout <= 0:
            raise OpenRouterRequestError("http_options.timeout must be positive.")
        timeout_seconds = http_options.timeout / 1000.0

    max_retries: int | None = None
    retry_options = http_options.retry_options
    if retry_options is not None and retry_options.attempts is not None:
        if retry_options.attempts < 1:
            raise OpenRouterRequestError(
                "http_options.retry_options.attempts must be at least 1."
            )
        max_retries = retry_options.attempts - 1

    return extra_body, headers, timeout_seconds, max_retries


def _system_instruction_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, types.Content):
        text = _join_text_parts(value.parts or [])
        return text or None
    if isinstance(value, (list, tuple)):
        text = _join_text_parts(value)
        return text or None
    raise OpenRouterRequestError(
        f"Unsupported system_instruction type: {type(value).__name__}"
    )


def _join_text_parts(parts: Sequence[types.Part]) -> str:
    texts: list[str] = []
    for part in parts:
        if part.inline_data is not None or part.file_data is not None:
            raise OpenRouterRequestError(
                "Media parts in a system instruction are not supported."
            )
        if part.text and not part.thought:
            texts.append(part.text)
    return "".join(texts).strip()


def _assistant_message(
    parts: Sequence[types.Part],
) -> tuple[dict[str, Any] | None, list[tuple[str | None, str]]]:
    texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    calls: list[tuple[str | None, str]] = []

    for part in parts:
        function_call = part.function_call
        if function_call is not None:
            name = function_call.name
            if not name:
                raise OpenRouterRequestError(
                    "An assistant function call is missing its name."
                )
            call_id = function_call.id or _new_call_id()
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _json_dumps(function_call.args or {}),
                    },
                }
            )
            calls.append((name, call_id))
            continue

        if part.inline_data is not None or part.file_data is not None:
            raise OpenRouterRequestError(
                "Assistant media content is not supported by OpenRouter."
            )

        if part.text:
            if part.thought:
                thoughts.append(part.text)
            else:
                texts.append(part.text)

    if not texts and not thoughts and not tool_calls:
        return None, calls

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(texts) or None,
    }
    if thoughts:
        message["reasoning"] = "".join(thoughts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, calls


def _user_message(parts: Sequence[types.Part]) -> dict[str, Any] | None:
    items: list[dict[str, Any]] = []
    texts: list[str] = []

    for part in parts:
        if part.text and not part.thought:
            items.append({"type": "text", "text": part.text})
            texts.append(part.text)
            continue

        inline_data = part.inline_data
        if inline_data is not None:
            mime_type = (inline_data.mime_type or "").strip()
            if mime_type.lower() == "image/jpg":
                mime_type = "image/jpeg"
            if not mime_type.lower().startswith("image/"):
                raise OpenRouterRequestError(
                    f"Unsupported inline media type: {mime_type or 'unknown'}"
                )
            data = inline_data.data
            if not isinstance(data, (bytes, bytearray)) or not data:
                raise OpenRouterRequestError("Inline image data is missing.")
            encoded = base64.b64encode(bytes(data)).decode("ascii")
            items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
            continue

        if part.file_data is not None:
            raise OpenRouterRequestError(
                "File URI content is not supported by OpenRouter."
            )

    if not items:
        return None
    if len(texts) == len(items):
        return {"role": "user", "content": "".join(texts)}
    return {"role": "user", "content": items}


def _tool_response_text(response: types.FunctionResponse) -> str:
    payload: object = response.response
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(
            payload if payload is not None else {},
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise OpenRouterRequestError(
            "A tool response could not be serialized as JSON."
        ) from exc


def _resolve_tool_response_id(
    response: types.FunctionResponse,
    pending_calls: list[tuple[str | None, str]],
    fallback_counter: int,
) -> str:
    response_id = response.id
    if response_id:
        for index, (_, call_id) in enumerate(pending_calls):
            if call_id == response_id:
                pending_calls.pop(index)
                return response_id

    name = response.name
    for index, (call_name, call_id) in enumerate(pending_calls):
        if call_name == name:
            pending_calls.pop(index)
            return call_id

    if pending_calls:
        return pending_calls.pop(0)[1]
    return response_id or f"blacki_tool_{fallback_counter}"


def _ensure_tool_results(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    healed: list[dict[str, Any]] = []
    pending_ids: list[str] = []

    for message in messages:
        role = message.get("role")
        if pending_ids and role != "tool":
            _insert_missing_tool_results(healed, pending_ids)
            pending_ids = []

        if role == "assistant":
            pending_ids = [
                str(call["id"])
                for call in message.get("tool_calls", [])
                if call.get("id")
            ]
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id in pending_ids:
                pending_ids.remove(call_id)

        healed.append(message)

    if pending_ids:
        _insert_missing_tool_results(healed, pending_ids)
    return healed


def _insert_missing_tool_results(
    messages: list[dict[str, Any]],
    pending_ids: list[str],
) -> None:
    logger.warning("Missing tool results for tool_call_id(s): %s", pending_ids)
    for call_id in pending_ids:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _MISSING_TOOL_RESULT,
            }
        )


def _tool_declarations(
    tools: Sequence[object] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    declarations: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, types.Tool):
            raise OpenRouterRequestError(
                f"Unsupported tool type: {type(tool).__name__}"
            )
        data = tool.model_dump(exclude_none=True)
        if not data:
            continue
        extra_fields = set(data) - {"function_declarations"}
        if extra_fields:
            fields = ", ".join(sorted(extra_fields))
            raise OpenRouterRequestError(
                f"OpenRouter supports function tools only; received: {fields}"
            )
        function_declarations = tool.function_declarations or []
        for declaration in function_declarations:
            declarations.append(_tool_for(declaration))
    return declarations or None


def _tool_for(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    name = declaration.name
    if not name:
        raise OpenRouterRequestError("A function declaration is missing its name.")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": declaration.description or "",
            "parameters": _declaration_schema(declaration),
        },
    }


def _declaration_schema(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    if declaration.parameters_json_schema:
        return cast(dict[str, Any], declaration.parameters_json_schema)
    if declaration.parameters is not None:
        normalized = _lowercase_schema_types(
            declaration.parameters.model_dump(by_alias=True, exclude_none=True)
        )
        return normalized if isinstance(normalized, dict) else {}
    return {"type": "object", "properties": {}}


def _lowercase_schema_types(value: object) -> object:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "type":
                normalized[key] = _normalize_schema_type(item)
            else:
                normalized[key] = _lowercase_schema_types(item)
        return normalized
    if isinstance(value, list):
        return [_lowercase_schema_types(item) for item in value]
    return value


def _normalize_schema_type(value: object) -> object:
    if isinstance(value, types.Type):
        return value.value.lower()
    if isinstance(value, str):
        return value.lower()
    return value


def _json_dumps(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise OpenRouterRequestError(
            "A function-call argument could not be serialized as JSON."
        ) from exc


def _new_call_id() -> str:
    return f"blacki_call_{uuid.uuid4().hex}"
