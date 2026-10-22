"""Convert OpenRouter chat-completion payloads into ADK responses."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

_FINISH_REASONS = {
    "stop": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "tool_calls": types.FinishReason.STOP,
    "function_call": types.FinishReason.STOP,
    "content_filter": types.FinishReason.SAFETY,
}
_TRUNCATED_TOOL_MESSAGE = (
    "Tool call arguments were truncated and could not be parsed as JSON. "
    "Increase max_output_tokens and retry."
)


class OpenRouterResponseError(RuntimeError):
    """Raised when OpenRouter returns a payload Blacki cannot interpret."""


class _TruncatedToolArgumentsError(OpenRouterResponseError):
    """Raised when a tool call was cut off by the token limit."""


def response_to_llm_response(payload: Mapping[str, Any]) -> LlmResponse:
    """Convert one non-streaming OpenRouter response into an ``LlmResponse``."""

    _raise_for_error(payload)
    model_version = _model_version(payload)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return LlmResponse(
            content=types.Content(role="model", parts=[]),
            partial=False,
            model_version=model_version,
        )

    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise OpenRouterResponseError("OpenRouter returned an invalid choice.")
    finish_reason = _finish_reason_value(choice.get("finish_reason"))
    if finish_reason == "error":
        raise OpenRouterResponseError(
            "OpenRouter returned finish_reason=error without an error payload."
        )

    message = choice.get("message")
    if not isinstance(message, Mapping):
        message = {}
    usage = payload.get("usage")
    try:
        return _response_from_message(
            message,
            finish_reason=finish_reason,
            usage=usage,
            model_version=model_version,
        )
    except _TruncatedToolArgumentsError:
        return _truncated_response(model_version, usage)


def _response_from_message(
    message: Mapping[str, Any],
    *,
    finish_reason: str | None,
    usage: object,
    model_version: str | None,
) -> LlmResponse:
    parts: list[types.Part] = []
    reasoning = _reasoning_text(message)
    if reasoning:
        parts.append(types.Part(text=reasoning, thought=True))

    text = _content_text(message.get("content"))
    if text:
        parts.append(types.Part.from_text(text=text))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                raise OpenRouterResponseError(
                    "OpenRouter returned a tool call without a function."
                )
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise OpenRouterResponseError(
                    "OpenRouter returned a tool call without a name."
                )
            args = _parse_arguments(function.get("arguments"), finish_reason)
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = _new_call_id()
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call_id,
                        name=name,
                        args=args,
                    )
                )
            )

    response = LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=False,
        model_version=model_version,
    )
    _apply_finish_reason(response, finish_reason)
    response.usage_metadata = _usage_metadata(usage)
    return response


def _apply_finish_reason(
    response: LlmResponse,
    finish_reason: str | None,
) -> None:
    mapped = _map_finish_reason(finish_reason)
    if mapped is None:
        return
    response.finish_reason = mapped
    if mapped != types.FinishReason.STOP:
        response.error_code = mapped
        response.error_message = _finish_error_message(mapped)


def _map_finish_reason(value: object) -> types.FinishReason | None:
    if value is None or value == "":
        return None
    if isinstance(value, types.FinishReason):
        return value
    return _FINISH_REASONS.get(str(value).lower(), types.FinishReason.OTHER)


def _finish_error_message(finish_reason: types.FinishReason) -> str:
    if finish_reason == types.FinishReason.MAX_TOKENS:
        return "Maximum tokens reached"
    return f"Finished with {finish_reason.name}"


def _finish_reason_value(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _model_version(payload: Mapping[str, Any]) -> str | None:
    model = payload.get("model")
    return model if isinstance(model, str) and model else None


def _raise_for_error(payload: Mapping[str, Any]) -> None:
    error = payload.get("error")
    if error is not None:
        raise OpenRouterResponseError(_error_message(error))


def _error_message(error: object) -> str:
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")
        parts: list[str] = []
        if code is not None:
            parts.append(str(code))
        if message is not None:
            parts.append(str(message))
        text = ": ".join(parts) or "Unknown OpenRouter error"
    else:
        text = str(error)
    return text[:1000]


def _truncated_response(
    model_version: str | None,
    usage: object,
) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[]),
        error_code=types.FinishReason.MAX_TOKENS,
        error_message=_TRUNCATED_TOOL_MESSAGE,
        finish_reason=types.FinishReason.MAX_TOKENS,
        partial=False,
        model_version=model_version,
        usage_metadata=_usage_metadata(usage),
    )


def _reasoning_text(value: Mapping[str, Any]) -> str:
    for key in ("reasoning", "reasoning_content"):
        raw = value.get(key)
        if isinstance(raw, str) and raw:
            return raw

    details = value.get("reasoning_details")
    if not isinstance(details, list):
        return ""

    texts: list[str] = []
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        detail_type = detail.get("type")
        if detail_type == "reasoning.text":
            text = detail.get("text")
        elif detail_type == "reasoning.summary":
            text = detail.get("summary")
        else:
            continue
        if isinstance(text, str) and text:
            texts.append(text)
    return "".join(texts)


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    texts: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            texts.append(text)
    return "".join(texts)


def _parse_arguments(
    value: object,
    finish_reason: str | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise OpenRouterResponseError(
            "OpenRouter returned tool arguments that are not JSON text."
        )
    stripped = value.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        if finish_reason == "length":
            raise _TruncatedToolArgumentsError(_TRUNCATED_TOOL_MESSAGE) from exc
        raise OpenRouterResponseError(
            "OpenRouter returned invalid JSON in tool arguments."
        ) from exc
    if not isinstance(parsed, dict):
        raise OpenRouterResponseError(
            "OpenRouter returned tool arguments that are not a JSON object."
        )
    return parsed


def _usage_metadata(
    usage: object,
) -> types.GenerateContentResponseUsageMetadata | None:
    if not isinstance(usage, Mapping) or not usage:
        return None

    prompt_tokens = _nonnegative_int(usage.get("prompt_tokens"))
    completion_tokens = _nonnegative_int(usage.get("completion_tokens"))
    total_tokens = _nonnegative_int(usage.get("total_tokens"))

    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = (
        _nonnegative_int(prompt_details.get("cached_tokens"))
        if isinstance(prompt_details, Mapping)
        else None
    )

    completion_details = usage.get("completion_tokens_details")
    reasoning_tokens = (
        _nonnegative_int(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, Mapping)
        else None
    )

    if (
        prompt_tokens is None
        and completion_tokens is None
        and total_tokens is None
        and cached_tokens is None
        and reasoning_tokens is None
    ):
        return None

    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_tokens or 0,
        candidates_token_count=completion_tokens or 0,
        total_token_count=total_tokens or 0,
        cached_content_token_count=cached_tokens,
        thoughts_token_count=reasoning_tokens or None,
    )


def _nonnegative_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value)
        except ValueError:
            return None
    else:
        return None
    return number if number >= 0 else None


def _new_call_id() -> str:
    return f"blacki_call_{uuid.uuid4().hex}"


@dataclass(slots=True)
class OpenRouterStreamAccumulator:
    """Accumulate streamed deltas into partial and final ADK responses."""

    _model_version: str | None = None
    _finish_reason: str | None = None
    _usage: Mapping[str, Any] | None = None
    _text: list[str] = field(default_factory=list)
    _reasoning: list[str] = field(default_factory=list)
    _tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)

    def add(self, payload: Mapping[str, Any]) -> list[LlmResponse]:
        """Consume one SSE payload and return any partial responses."""

        _raise_for_error(payload)
        model_version = _model_version(payload)
        if model_version:
            self._model_version = model_version
        usage = payload.get("usage")
        if isinstance(usage, Mapping) and usage:
            self._usage = usage

        choices = payload.get("choices")
        if not isinstance(choices, list):
            return []

        responses: list[LlmResponse] = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            finish_reason = _finish_reason_value(choice.get("finish_reason"))
            if finish_reason:
                self._finish_reason = finish_reason
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue

            reasoning = _reasoning_text(delta)
            if reasoning:
                self._reasoning.append(reasoning)
                responses.append(
                    LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[types.Part(text=reasoning, thought=True)],
                        ),
                        partial=True,
                        model_version=self._model_version,
                    )
                )

            text = _content_text(delta.get("content"))
            if text:
                self._text.append(text)
                responses.append(
                    LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=text)],
                        ),
                        partial=True,
                        model_version=self._model_version,
                    )
                )

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                self._merge_tool_calls(tool_calls)

        return responses

    def finish(self) -> LlmResponse:
        """Return the single aggregated response for the completed stream."""

        if self._finish_reason == "error":
            raise OpenRouterResponseError(
                "OpenRouter stream ended with finish_reason=error."
            )

        parts: list[types.Part] = []
        reasoning = "".join(self._reasoning)
        if reasoning:
            parts.append(types.Part(text=reasoning, thought=True))
        text = "".join(self._text)
        if text:
            parts.append(types.Part.from_text(text=text))

        try:
            for index in sorted(self._tool_calls):
                part = self._tool_call_part(self._tool_calls[index])
                if part is not None:
                    parts.append(part)
        except _TruncatedToolArgumentsError:
            return _truncated_response(self._model_version, self._usage)

        response = LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            model_version=self._model_version,
            usage_metadata=_usage_metadata(self._usage),
        )
        finish_reason = self._finish_reason
        if finish_reason is None and parts:
            finish_reason = "stop"
        _apply_finish_reason(response, finish_reason)
        return response

    def _tool_call_part(
        self,
        entry: Mapping[str, Any],
    ) -> types.Part | None:
        name = entry.get("name")
        arguments = entry.get("arguments")
        if not isinstance(name, str) or not name:
            if isinstance(arguments, str) and arguments:
                raise OpenRouterResponseError(
                    "OpenRouter streamed tool arguments without a function name."
                )
            return None
        args = _parse_arguments(arguments, self._finish_reason)
        call_id = entry.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = _new_call_id()
        return types.Part(
            function_call=types.FunctionCall(
                id=call_id,
                name=name,
                args=args,
            )
        )

    def _merge_tool_calls(self, tool_calls: list[object]) -> None:
        for position, call in enumerate(tool_calls):
            if not isinstance(call, Mapping):
                continue
            index = call.get("index")
            if not isinstance(index, int):
                index = position
            entry = self._tool_calls.setdefault(
                index,
                {"id": None, "name": "", "arguments": ""},
            )

            call_id = call.get("id")
            if isinstance(call_id, str) and call_id and not entry["id"]:
                entry["id"] = call_id

            function = call.get("function")
            if not isinstance(function, Mapping):
                continue

            name = function.get("name")
            if isinstance(name, str) and name:
                if not entry["name"]:
                    entry["name"] = name
                elif name != entry["name"]:
                    entry["name"] += name

            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                entry["arguments"] += arguments
