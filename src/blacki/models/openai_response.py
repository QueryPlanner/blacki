"""Convert OpenAI Responses payloads into ADK responses.

Each output item becomes one ADK part. The raw item is stored unchanged in the
part's metadata (see ``openai_items``) together with the model id that was
requested, so the next request to the same model can replay it. The streaming
accumulator builds its final response from the terminal stream event with the
same code, so streamed and non-streamed turns store identical items.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from blacki.models.openai_items import item_metadata

logger = logging.getLogger(__name__)

_INCOMPLETE_FINISH_REASONS = {
    "max_output_tokens": types.FinishReason.MAX_TOKENS,
    "content_filter": types.FinishReason.SAFETY,
}
_MESSAGE_TEXT_KEYS = {"output_text": "text", "refusal": "refusal"}
_TRUNCATED_TOOL_MESSAGE = (
    "Tool call arguments were truncated and could not be parsed as JSON. "
    "Increase max_output_tokens and retry."
)
_INVALID_ARGUMENTS_MESSAGE = "OpenAI returned invalid JSON in tool arguments."
_NON_OBJECT_ARGUMENTS_MESSAGE = (
    "OpenAI returned tool arguments that are not a JSON object."
)
_GENERIC_ERROR_MESSAGE = "OpenAI returned an error."
_MAX_ERROR_CHARS = 1000
_TEXT_DELTA = "response.output_text.delta"
_SUMMARY_DELTA = "response.reasoning_summary_text.delta"
_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete"})


class OpenAIResponseError(RuntimeError):
    """Raised when OpenAI reports a failed response or a malformed payload.

    Messages never contain prompt content or encrypted reasoning.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class _TruncatedToolArgumentsError(Exception):
    """Internal signal: a tool call was cut off by the output token limit."""


def response_to_llm_response(raw: Mapping[str, Any], *, model: str) -> LlmResponse:
    """Convert one non-streamed OpenAI response into an ``LlmResponse``.

    ``model`` is the normalized id that was requested. It is stored on every
    part; the dated id OpenAI reports in ``raw["model"]`` is never stored.
    """

    _raise_for_failure(raw)
    finish_reason = _finish_reason(raw)
    model_version = _model_version(raw)
    usage = raw.get("usage")
    truncated = finish_reason == types.FinishReason.MAX_TOKENS
    try:
        parts = _output_parts(raw.get("output"), model=model, truncated=truncated)
    except _TruncatedToolArgumentsError:
        response = LlmResponse(
            content=types.Content(role="model", parts=[]),
            error_code=types.FinishReason.MAX_TOKENS,
            error_message=_TRUNCATED_TOOL_MESSAGE,
            finish_reason=types.FinishReason.MAX_TOKENS,
            partial=False,
            model_version=model_version,
        )
    else:
        response = LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            model_version=model_version,
        )
        _apply_finish_reason(response, finish_reason)
    _attach_usage(response, usage)
    return response


class OpenAIStreamAccumulator:
    """Turn Responses stream events into partial responses and one final response."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._final: LlmResponse | None = None

    def add(self, event: Mapping[str, Any]) -> list[LlmResponse]:
        """Consume one stream event and return the partial responses it yields.

        Events after the terminal event are ignored.
        """

        if self._final is not None:
            return []
        event_type = event.get("type")
        if event_type == _TEXT_DELTA:
            return _partial_response(event, thought=False)
        if event_type == _SUMMARY_DELTA:
            return _partial_response(event, thought=True)
        if event_type in _TERMINAL_EVENTS:
            self._final = _final_from_event(event, model=self._model)
            return []
        if event_type == "response.failed":
            response = event.get("response")
            error = response.get("error") if isinstance(response, Mapping) else None
            raise _error_from_payload(error)
        if event_type == "error":
            raise _error_from_payload(event)
        return []

    def finish(self) -> LlmResponse:
        """Return the single final response once the stream has terminated."""

        if self._final is None:
            raise OpenAIResponseError("OpenAI stream ended without a final response.")
        return self._final


def _output_parts(
    output: object,
    *,
    model: str,
    truncated: bool,
) -> list[types.Part]:
    if not isinstance(output, list):
        return []
    parts: list[types.Part] = []
    skipped_types: set[str] = set()
    for item in output:
        if not isinstance(item, Mapping):
            raise OpenAIResponseError("OpenAI returned an invalid output item.")
        item_type = item.get("type")
        if item_type == "reasoning":
            parts.append(
                types.Part(
                    text=_reasoning_text(item),
                    thought=True,
                    part_metadata=item_metadata(model, item),
                )
            )
        elif item_type == "message":
            parts.append(
                types.Part(
                    text=_message_text(item.get("content")),
                    part_metadata=item_metadata(model, item),
                )
            )
        elif item_type == "function_call":
            parts.append(_function_call_part(item, model=model, truncated=truncated))
        else:
            label = item_type if isinstance(item_type, str) else "unknown"
            if label not in skipped_types:
                skipped_types.add(label)
                logger.debug("Skipping OpenAI output item of type %s.", label)
    return parts


def _reasoning_text(item: Mapping[str, Any]) -> str:
    summary = item.get("summary")
    if not isinstance(summary, list):
        return ""
    texts: list[str] = []
    for entry in summary:
        if isinstance(entry, Mapping) and entry.get("type") == "summary_text":
            text = entry.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "".join(texts)


def _message_text(content: object) -> str:
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        key = _MESSAGE_TEXT_KEYS.get(part_type) if isinstance(part_type, str) else None
        value = part.get(key) if key else None
        if isinstance(value, str):
            texts.append(value)
    return "".join(texts)


def _function_call_part(
    item: Mapping[str, Any],
    *,
    model: str,
    truncated: bool,
) -> types.Part:
    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise OpenAIResponseError("OpenAI returned a function call without a name.")
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise OpenAIResponseError("OpenAI returned a function call without a call_id.")
    args = _parse_arguments(item.get("arguments"), truncated=truncated)
    return types.Part(
        function_call=types.FunctionCall(id=call_id, name=name, args=args),
        part_metadata=item_metadata(model, item),
    )


def _parse_arguments(value: object, *, truncated: bool) -> dict[str, Any]:
    """Decode a tool call's JSON arguments object.

    Empty or missing arguments mean no arguments. Invalid JSON raises, except
    that a cut-off call on a max-tokens response becomes the truncated error
    response. Error messages never include the arguments text.
    """

    if value is None:
        return {}
    if not isinstance(value, str):
        raise OpenAIResponseError(_INVALID_ARGUMENTS_MESSAGE)
    stripped = value.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        if truncated:
            raise _TruncatedToolArgumentsError() from None
        raise OpenAIResponseError(_INVALID_ARGUMENTS_MESSAGE) from None
    if not isinstance(parsed, dict):
        raise OpenAIResponseError(_NON_OBJECT_ARGUMENTS_MESSAGE)
    return parsed


def _finish_reason(raw: Mapping[str, Any]) -> types.FinishReason | None:
    status = raw.get("status")
    if status is None:
        return None
    if status == "completed":
        return types.FinishReason.STOP
    if status == "incomplete":
        return _incomplete_finish_reason(raw)
    return types.FinishReason.OTHER


def _incomplete_finish_reason(raw: Mapping[str, Any]) -> types.FinishReason:
    details = raw.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, Mapping) else None
    if isinstance(reason, str):
        return _INCOMPLETE_FINISH_REASONS.get(reason, types.FinishReason.OTHER)
    return types.FinishReason.OTHER


def _apply_finish_reason(
    response: LlmResponse,
    finish_reason: types.FinishReason | None,
) -> None:
    if finish_reason is None:
        return
    response.finish_reason = finish_reason
    if finish_reason != types.FinishReason.STOP:
        response.error_code = finish_reason
        response.error_message = _finish_error_message(finish_reason)


def _finish_error_message(finish_reason: types.FinishReason) -> str:
    if finish_reason == types.FinishReason.MAX_TOKENS:
        return "Maximum tokens reached"
    return f"Finished with {finish_reason.name}"


def _model_version(raw: Mapping[str, Any]) -> str | None:
    model = raw.get("model")
    return model if isinstance(model, str) and model else None


def _raise_for_failure(raw: Mapping[str, Any]) -> None:
    if raw.get("status") == "failed":
        raise _error_from_payload(raw.get("error"))


def _error_from_payload(payload: object) -> OpenAIResponseError:
    """Build the error for an OpenAI error object or an event that carries one."""

    if not isinstance(payload, Mapping):
        return OpenAIResponseError(_GENERIC_ERROR_MESSAGE)
    code = payload.get("code")
    message = payload.get("message")
    return OpenAIResponseError(
        message[:_MAX_ERROR_CHARS]
        if isinstance(message, str) and message
        else _GENERIC_ERROR_MESSAGE,
        code=code if isinstance(code, str) and code else None,
    )


def _attach_usage(response: LlmResponse, usage: object) -> None:
    response.usage_metadata = _usage_metadata(usage)
    if isinstance(usage, Mapping) and usage:
        response.custom_metadata = {"openai_usage": dict(usage)}


def _usage_metadata(
    usage: object,
) -> types.GenerateContentResponseUsageMetadata | None:
    if not isinstance(usage, Mapping) or not usage:
        return None

    input_tokens = _nonnegative_int(usage.get("input_tokens"))
    output_tokens = _nonnegative_int(usage.get("output_tokens"))
    total_tokens = _nonnegative_int(usage.get("total_tokens"))
    cached_tokens = _detail_int(usage, "input_tokens_details", "cached_tokens")
    reasoning_tokens = _detail_int(usage, "output_tokens_details", "reasoning_tokens")
    counts = (
        input_tokens,
        output_tokens,
        total_tokens,
        cached_tokens,
        reasoning_tokens,
    )
    if all(count is None for count in counts):
        return None

    prompt_tokens = input_tokens or 0
    completion_tokens = output_tokens or 0
    # Reasoning tokens are already counted inside output_tokens, so they are
    # reported as a breakdown only and never added to the candidate count.
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=(
            total_tokens
            if total_tokens is not None
            else prompt_tokens + completion_tokens
        ),
        cached_content_token_count=cached_tokens,
        thoughts_token_count=reasoning_tokens or None,
    )


def _detail_int(usage: Mapping[str, Any], group: str, key: str) -> int | None:
    details = usage.get(group)
    if not isinstance(details, Mapping):
        return None
    return _nonnegative_int(details.get(key))


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _partial_response(event: Mapping[str, Any], *, thought: bool) -> list[LlmResponse]:
    delta = event.get("delta")
    if not isinstance(delta, str) or not delta:
        return []
    part = types.Part(text=delta, thought=True) if thought else types.Part(text=delta)
    return [
        LlmResponse(
            content=types.Content(role="model", parts=[part]),
            partial=True,
        )
    ]


def _final_from_event(event: Mapping[str, Any], *, model: str) -> LlmResponse:
    response = event.get("response")
    if not isinstance(response, Mapping):
        raise OpenAIResponseError(
            "OpenAI returned a final stream event without a response."
        )
    return response_to_llm_response(response, model=model)
