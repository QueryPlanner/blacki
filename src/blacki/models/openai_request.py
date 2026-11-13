"""Translate ADK model requests into OpenAI Responses API requests.

The conversion supports system instructions, text, inline images, function
tools and their results, and reasoning settings carried in
``http_options.extra_body``. Output items stored on earlier model parts are
replayed unchanged when the request goes to the same model; other parts are
converted to portable input items. Storage on OpenAI's side is always disabled.

Design decisions D2, D4, D5 and D7 are described in
``openspec/changes/replace-openrouter-with-openai-responses/design.md``.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from .catalog import get_model, normalize_model_id
from .openai_items import StoredItem, stored_item

logger = logging.getLogger(__name__)

ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"

_MISSING_TOOL_RESULT = '{"error":"Missing tool response."}'
_FUNCTION_CALL = "function_call"
_FUNCTION_CALL_OUTPUT = "function_call_output"
_REASONING = "reasoning"
_CALL_ID_PREFIX = "call_blacki_"

# The SDK merges ``extra_body`` over explicit keyword arguments. These keys
# define the request itself, so a stray extra_body value must never replace
# them. ``reasoning`` is read separately and never forwarded as-is.
_RESERVED_EXTRA_BODY_KEYS = frozenset(
    {
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
        "reasoning",
        "text",
        "max_output_tokens",
        "temperature",
        "top_p",
    }
)
_FILTERED_HEADER_NAMES = frozenset({"authorization", "content-type"})


class OpenAIRequestError(ValueError):
    """Raised when an ADK request cannot be represented as a Responses call."""


@dataclass(slots=True)
class ResponsesRequest:
    """One prepared OpenAI Responses call.

    ``store`` must stay ``False`` and ``include`` must request encrypted
    reasoning, so that no request can ask OpenAI to keep a response or drop
    the encrypted reasoning that later turns replay.
    """

    model: str
    input: list[dict[str, Any]]
    instructions: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    include: list[str] = field(default_factory=lambda: [ENCRYPTED_REASONING_INCLUDE])
    store: bool = False
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    max_retries: int | None = None

    def __post_init__(self) -> None:
        if self.store:
            raise OpenAIRequestError("Blacki never asks OpenAI to store responses.")
        if ENCRYPTED_REASONING_INCLUDE not in self.include:
            raise OpenAIRequestError(
                f"include must contain {ENCRYPTED_REASONING_INCLUDE}."
            )

    def create_kwargs(self, *, stream: bool) -> dict[str, Any]:
        """Return keyword arguments for ``responses.create``."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self.input,
            "include": list(self.include),
            "store": False,
            "stream": stream,
        }
        if self.instructions is not None:
            kwargs["instructions"] = self.instructions
        if self.tools:
            kwargs["tools"] = self.tools
        if self.tool_choice is not None:
            kwargs["tool_choice"] = self.tool_choice
        if self.reasoning is not None:
            kwargs["reasoning"] = dict(self.reasoning)
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        if self.extra_headers:
            kwargs["extra_headers"] = dict(self.extra_headers)
        return kwargs


@dataclass(frozen=True, slots=True)
class _HttpSettings:
    extra_body: dict[str, Any]
    reasoning: object
    headers: dict[str, str]
    timeout_seconds: float | None
    max_retries: int | None


class _Emitted(NamedTuple):
    """One input item built from a model part.

    ``replayed`` is true when the item is a raw stored item from the same model.
    """

    payload: dict[str, Any]
    replayed: bool


def build_responses_request(
    llm_request: LlmRequest,
    *,
    model: str,
    replay_reasoning: bool = True,
) -> ResponsesRequest:
    """Build one Responses request from an ADK ``LlmRequest``.

    ``model`` is the normalized requested model id. Stored items are replayed
    only when they were produced by that same id. ``replay_reasoning=False``
    drops every reasoning item from the request.
    """

    config = llm_request.config
    _reject_unsupported_config(config)

    model_id = normalize_model_id(model)
    http = _http_options(config.http_options)
    reasoning = _reasoning_settings(http.reasoning)
    tools = _tool_declarations(config.tools)
    temperature, top_p = _sampling_params(config, model_id, reasoning.get("effort"))
    _warn_ignored_generation(config)

    return ResponsesRequest(
        model=model_id,
        input=_input_items(llm_request.contents, model_id, replay_reasoning),
        instructions=_system_instruction_text(config.system_instruction),
        tools=tools,
        tool_choice=_tool_choice(config) if tools else None,
        reasoning=reasoning,
        max_output_tokens=config.max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=http.extra_body,
        extra_headers=http.headers,
        timeout_seconds=http.timeout_seconds,
        max_retries=http.max_retries,
    )


def _reject_unsupported_config(config: types.GenerateContentConfig) -> None:
    unsupported = {
        "response_schema": config.response_schema,
        "response_json_schema": config.response_json_schema,
        "response_mime_type": config.response_mime_type,
        "safety_settings": config.safety_settings,
        "thinking_config": config.thinking_config,
    }
    for name, value in unsupported.items():
        if value is not None:
            raise OpenAIRequestError(
                f"Blacki's OpenAI adapter does not support {name}."
            )
    candidate_count = config.candidate_count
    if candidate_count is not None and candidate_count != 1:
        raise OpenAIRequestError(
            "Blacki's OpenAI adapter only supports candidate_count=1."
        )


def _warn_ignored_generation(config: types.GenerateContentConfig) -> None:
    ignored = [
        name
        for name, value in (
            ("top_k", config.top_k),
            ("stop_sequences", config.stop_sequences or None),
            ("presence_penalty", config.presence_penalty),
            ("frequency_penalty", config.frequency_penalty),
            ("seed", config.seed),
        )
        if value is not None
    ]
    if ignored:
        logger.warning(
            "Ignoring generation settings the OpenAI Responses API does not accept: %s",
            ", ".join(ignored),
        )


def _http_options(http_options: types.HttpOptions | None) -> _HttpSettings:
    if http_options is None:
        return _HttpSettings({}, None, {}, None, None)

    extra_body: dict[str, Any] = {}
    reasoning: object = None
    if http_options.extra_body is not None:
        if not isinstance(http_options.extra_body, Mapping):
            raise OpenAIRequestError("http_options.extra_body must be a mapping.")
        for key, value in http_options.extra_body.items():
            key_text = str(key)
            lowered = key_text.lower()
            if lowered == "reasoning":
                reasoning = value
                continue
            if lowered in _RESERVED_EXTRA_BODY_KEYS:
                logger.warning(
                    "Ignoring reserved extra_body key %r for OpenAI", key_text
                )
                continue
            extra_body[key_text] = value

    headers: dict[str, str] = {}
    if http_options.headers is not None:
        if not isinstance(http_options.headers, Mapping):
            raise OpenAIRequestError("http_options.headers must be a mapping.")
        for key, value in http_options.headers.items():
            key_text = str(key)
            if key_text.lower() in _FILTERED_HEADER_NAMES:
                logger.warning(
                    "Ignoring caller-supplied %r header for OpenAI", key_text
                )
                continue
            headers[key_text] = str(value)

    timeout_seconds: float | None = None
    if http_options.timeout is not None:
        if http_options.timeout <= 0:
            raise OpenAIRequestError("http_options.timeout must be positive.")
        timeout_seconds = http_options.timeout / 1000.0

    max_retries: int | None = None
    retry_options = http_options.retry_options
    if retry_options is not None and retry_options.attempts is not None:
        if retry_options.attempts < 1:
            raise OpenAIRequestError(
                "http_options.retry_options.attempts must be at least 1."
            )
        max_retries = retry_options.attempts - 1

    return _HttpSettings(extra_body, reasoning, headers, timeout_seconds, max_retries)


def _reasoning_settings(value: object) -> dict[str, Any]:
    """Return the Responses ``reasoning`` object for a caller's settings.

    The summary is always requested, so reasoning summaries come back for
    display. Only ``effort`` is read from the caller's mapping.
    """

    if value is None:
        return {"summary": "auto"}
    if not isinstance(value, Mapping):
        raise OpenAIRequestError("http_options.extra_body reasoning must be a mapping.")

    settings: dict[str, Any] = {"summary": "auto"}
    effort = value.get("effort")
    if effort is not None:
        if not isinstance(effort, str) or not effort:
            raise OpenAIRequestError("reasoning effort must be a non-empty string.")
        settings = {"effort": effort, "summary": "auto"}

    ignored = sorted(str(key) for key in value if key != "effort")
    if ignored:
        logger.warning(
            "Ignoring reasoning settings the OpenAI adapter does not use: %s",
            ", ".join(ignored),
        )
    return settings


def _sampling_params(
    config: types.GenerateContentConfig,
    model: str,
    effort: object,
) -> tuple[float | None, float | None]:
    """Return ``(temperature, top_p)`` when OpenAI accepts them for this request.

    Reasoning models reject sampling parameters unless reasoning is explicitly
    set to ``none`` on a model whose catalog entry allows it.
    """

    temperature = config.temperature
    top_p = config.top_p
    if temperature is None and top_p is None:
        return None, None

    entry = get_model(model)
    if effort == "none" and entry is not None and entry.accepts_sampling_at_none:
        return temperature, top_p

    dropped = [
        name
        for name, value in (("temperature", temperature), ("top_p", top_p))
        if value is not None
    ]
    logger.warning(
        "Dropping %s for model %s: sampling is sent only with reasoning effort "
        "none on models that accept it.",
        ", ".join(dropped),
        model,
    )
    return None, None


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
    raise OpenAIRequestError(
        f"Unsupported system_instruction type: {type(value).__name__}"
    )


def _join_text_parts(parts: Sequence[types.Part]) -> str:
    texts: list[str] = []
    for part in parts:
        if part.inline_data is not None or part.file_data is not None:
            raise OpenAIRequestError(
                "Media parts in a system instruction are not supported."
            )
        if part.text and not part.thought:
            texts.append(part.text)
    return "".join(texts).strip()


def _input_items(
    contents: Sequence[types.Content],
    model: str,
    replay_reasoning: bool,
) -> list[dict[str, Any]]:
    """Convert ADK contents into Responses input items.

    Calls and results are paired across the whole history, because a result
    may refer to a call in an earlier content. Reasoning items are pruned one
    content at a time.
    """

    pending: list[tuple[str | None, str]] = []
    items: list[dict[str, Any]] = []
    for content in contents:
        parts = content.parts or []
        if not parts:
            continue
        if content.role == "model":
            emitted = _model_items(parts, model, replay_reasoning, pending)
            items.extend(_prune_orphan_reasoning(emitted))
        else:
            items.extend(_user_items(parts, pending))
    return _ensure_tool_results(items)


def _model_items(
    parts: Sequence[types.Part],
    model: str,
    replay_reasoning: bool,
    pending: list[tuple[str | None, str]],
) -> list[_Emitted]:
    emitted: list[_Emitted] = []
    for part in parts:
        stored = stored_item(part)
        if stored is not None and _replayable(stored, model, replay_reasoning):
            payload = copy.deepcopy(dict(stored.item))
            if payload.get("type") == _FUNCTION_CALL:
                pending.append((payload.get("name"), str(payload["call_id"])))
            emitted.append(_Emitted(payload, replayed=True))
            continue

        function_call = part.function_call
        if function_call is not None:
            name = function_call.name
            if not name:
                raise OpenAIRequestError(
                    "An assistant function call is missing its name."
                )
            call_id = function_call.id or _new_call_id()
            pending.append((name, call_id))
            payload = {
                "type": _FUNCTION_CALL,
                "call_id": call_id,
                "name": name,
                "arguments": _json_dumps(function_call.args or {}),
            }
            emitted.append(_Emitted(payload, replayed=False))
            continue

        if part.inline_data is not None or part.file_data is not None:
            raise OpenAIRequestError(
                "Assistant media content is not supported by the OpenAI adapter."
            )
        if part.text and not part.thought:
            emitted.append(
                _Emitted({"role": "assistant", "content": part.text}, replayed=False)
            )
    return emitted


def _replayable(stored: StoredItem, model: str, replay_reasoning: bool) -> bool:
    """Return whether a stored item can be sent unchanged to ``model``."""

    item_type = stored.item_type
    if stored.model != model or item_type is None:
        return False
    if item_type == _REASONING and not replay_reasoning:
        return False
    if item_type == _FUNCTION_CALL:
        call_id = stored.item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return False
    return True


def _prune_orphan_reasoning(emitted: list[_Emitted]) -> list[dict[str, Any]]:
    """Drop reasoning items that are not followed by a replayed item.

    A reasoning item is kept only when the next item in the same content is
    a replayed stored item, or another reasoning item that is itself kept.
    Anything else would be an orphan that OpenAI rejects.
    """

    keep = [True] * len(emitted)
    follower_kept = False
    for index in reversed(range(len(emitted))):
        item = emitted[index]
        if item.payload.get("type") == _REASONING:
            keep[index] = follower_kept
            follower_kept = keep[index]
        else:
            follower_kept = item.replayed
    return [item.payload for item, kept in zip(emitted, keep, strict=True) if kept]


def _user_items(
    parts: Sequence[types.Part],
    pending: list[tuple[str | None, str]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in parts:
        response = part.function_response
        if response is None:
            continue
        items.append(
            {
                "type": _FUNCTION_CALL_OUTPUT,
                "call_id": _resolve_output_call_id(response, pending),
                "output": _tool_response_text(response),
            }
        )

    remaining = [part for part in parts if part.function_response is None]
    message = _user_message(remaining)
    if message is not None:
        items.append(message)
    return items


def _user_message(parts: Sequence[types.Part]) -> dict[str, Any] | None:
    content: list[dict[str, Any]] = []
    for part in parts:
        if part.thought:
            continue
        if part.text:
            content.append({"type": "input_text", "text": part.text})
            continue
        if part.inline_data is not None:
            content.append(_inline_image_item(part.inline_data))
            continue
        if part.file_data is not None:
            raise OpenAIRequestError(
                "File URI content is not supported by the OpenAI adapter."
            )

    if not content:
        return None
    return {"role": "user", "content": content}


def _inline_image_item(inline_data: types.Blob) -> dict[str, Any]:
    mime_type = (inline_data.mime_type or "").strip()
    if mime_type.lower() == "image/jpg":
        mime_type = "image/jpeg"
    if not mime_type.lower().startswith("image/"):
        raise OpenAIRequestError(
            f"Unsupported inline media type: {mime_type or 'unknown'}"
        )
    data = inline_data.data
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise OpenAIRequestError("Inline image data is missing.")
    encoded = base64.b64encode(bytes(data)).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type};base64,{encoded}",
        "detail": "auto",
    }


def _tool_response_text(response: types.FunctionResponse) -> str:
    """Return a function result as the JSON text the Responses API expects."""

    try:
        return json.dumps(response.response or {}, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise OpenAIRequestError(
            "A tool response could not be serialized as JSON."
        ) from exc


def _resolve_output_call_id(
    response: types.FunctionResponse,
    pending: list[tuple[str | None, str]],
) -> str:
    """Return the call id a function result answers.

    A real id that matches a pending call is used as-is. Otherwise the result
    pairs with the first pending call of the same name, then with the first
    pending call. A result with no pending call gets a fresh id.
    """

    response_id = response.id or None
    if response_id is not None:
        for index, (_, call_id) in enumerate(pending):
            if call_id == response_id:
                pending.pop(index)
                return response_id

    for index, (call_name, call_id) in enumerate(pending):
        if call_name == response.name:
            pending.pop(index)
            return call_id

    if pending:
        return pending.pop(0)[1]
    return response_id or _new_call_id()


def _ensure_tool_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert a placeholder output for each call that has no output.

    The placeholder goes right after the call group, before the next item that
    is neither a call nor an output. OpenAI rejects a call with no output.
    """

    healed: list[dict[str, Any]] = []
    pending: list[str] = []
    for item in items:
        kind = item.get("type")
        if pending and kind not in (_FUNCTION_CALL, _FUNCTION_CALL_OUTPUT):
            _insert_missing_tool_results(healed, pending)
            pending = []
        if kind == _FUNCTION_CALL:
            pending.append(str(item["call_id"]))
        elif kind == _FUNCTION_CALL_OUTPUT and item["call_id"] in pending:
            pending.remove(item["call_id"])
        healed.append(item)

    if pending:
        _insert_missing_tool_results(healed, pending)
    return healed


def _insert_missing_tool_results(
    items: list[dict[str, Any]],
    pending: list[str],
) -> None:
    logger.warning("Missing tool results for call_id(s): %s", ", ".join(pending))
    for call_id in pending:
        items.append(
            {
                "type": _FUNCTION_CALL_OUTPUT,
                "call_id": call_id,
                "output": _MISSING_TOOL_RESULT,
            }
        )


def _tool_choice(config: types.GenerateContentConfig) -> str | dict[str, Any] | None:
    """Map ADK's function-calling mode to a Responses ``tool_choice``.

    An allow-list is honoured only with mode ``ANY``, as Gemini does. Several
    allowed names become an ``allowed_tools`` choice, so the model cannot call
    a tool outside the list.
    """

    tool_config = config.tool_config
    if tool_config is None or tool_config.function_calling_config is None:
        return None
    calling = tool_config.function_calling_config
    mode = calling.mode
    if mode == types.FunctionCallingConfigMode.AUTO:
        return "auto"
    if mode == types.FunctionCallingConfigMode.NONE:
        return "none"
    if mode == types.FunctionCallingConfigMode.ANY:
        names = list(calling.allowed_function_names or [])
        if not names:
            return "required"
        if len(names) == 1:
            return {"type": "function", "name": names[0]}
        return {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": name} for name in names],
        }
    return None


def _tool_declarations(tools: Sequence[object] | None) -> list[dict[str, Any]]:
    if not tools:
        return []

    declarations: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, types.Tool):
            raise OpenAIRequestError(f"Unsupported tool type: {type(tool).__name__}")
        data = tool.model_dump(exclude_none=True)
        if not data:
            continue
        extra_fields = set(data) - {"function_declarations"}
        if extra_fields:
            fields = ", ".join(sorted(extra_fields))
            raise OpenAIRequestError(
                f"The OpenAI adapter supports function tools only; received: {fields}"
            )
        for declaration in tool.function_declarations or []:
            declarations.append(_tool_for(declaration))
    return declarations


def _tool_for(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    name = declaration.name
    if not name:
        raise OpenAIRequestError("A function declaration is missing its name.")
    return {
        "type": "function",
        "name": name,
        "description": declaration.description or "",
        "parameters": _declaration_schema(declaration),
        "strict": False,
    }


def _declaration_schema(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    if declaration.parameters_json_schema:
        return dict(declaration.parameters_json_schema)
    if declaration.parameters is not None:
        normalized = _lowercase_schema_types(
            declaration.parameters.model_dump(by_alias=True, exclude_none=True)
        )
        return normalized if isinstance(normalized, dict) else {}
    return {"type": "object", "properties": {}}


def _lowercase_schema_types(value: object) -> object:
    """Lowercase JSON-schema ``type`` values, as the Responses API expects.

    Only a ``type`` keyword is lowercased. A property named ``type`` keeps its
    schema, and the schema's own keywords are normalized.
    """

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            # ``types.Type`` is a ``str`` enum, so one check covers both forms.
            if key == "type" and isinstance(item, str):
                normalized[key] = item.lower()
            else:
                normalized[key] = _lowercase_schema_types(item)
        return normalized
    if isinstance(value, list):
        return [_lowercase_schema_types(item) for item in value]
    return value


def _json_dumps(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise OpenAIRequestError(
            "A function-call argument could not be serialized as JSON."
        ) from exc


def _new_call_id() -> str:
    return f"{_CALL_ID_PREFIX}{uuid.uuid4().hex}"
