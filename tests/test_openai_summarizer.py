"""Tests for the OpenAI compaction summarizer (design D12).

The model boundary is a fake ``BaseLlm`` that records each request and returns a
fixed response. ADK's event formatting and the summarizer's filtering run for
real.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from blacki.models.openai_items import item_metadata
from blacki.models.openai_summarizer import OpenAIEventsSummarizer

LOGGER_NAME = "blacki.models.openai_summarizer"
REASONING_SUMMARY = "Thinking about the lunch log."
SUMMARY_TEXT = "The user logged lunch."
OPEN_QUESTION = "No open questions."

REASONING_ITEM: dict[str, Any] = {
    "id": "rs_0a1b2c3d",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": REASONING_SUMMARY}],
    "encrypted_content": "ENC_summary_reasoning",
}
MESSAGE_ITEM: dict[str, Any] = {
    "id": "msg_4e5f6a7b",
    "type": "message",
    "role": "assistant",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": SUMMARY_TEXT, "annotations": []}],
}
USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=900, candidates_token_count=40
)


class FakeSummaryLlm(BaseLlm):
    """Answer every request with ``reply`` and record the requests it received."""

    reply: LlmResponse | None = None
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests.append(llm_request)
        if self.reply is not None:
            yield self.reply


def _model_reply(*parts: types.Part) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=list(parts)),
        usage_metadata=USAGE,
    )


def _history() -> list[Event]:
    return [
        Event(
            author="user",
            invocation_id="invocation-1",
            timestamp=1.0,
            content=types.Content(role="user", parts=[types.Part(text="Log my lunch")]),
        ),
        Event(
            author="blacki",
            invocation_id="invocation-1",
            timestamp=2.0,
            content=types.Content(
                role="model", parts=[types.Part(text="Logged a sandwich.")]
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_summary_keeps_only_message_text_without_metadata() -> None:
    llm = FakeSummaryLlm(
        model="gpt-6-luna",
        reply=_model_reply(
            types.Part(
                text=REASONING_SUMMARY,
                thought=True,
                part_metadata=item_metadata("gpt-6-luna", REASONING_ITEM),
            ),
            types.Part(
                text=SUMMARY_TEXT,
                part_metadata=item_metadata("gpt-6-luna", MESSAGE_ITEM),
            ),
            types.Part(text=""),
            types.Part(text=OPEN_QUESTION),
        ),
    )
    events = _history()

    event = await OpenAIEventsSummarizer(llm=llm).maybe_summarize_events(events=events)

    assert event is not None
    assert event.author == "user"
    assert event.usage_metadata == USAGE
    compaction = event.actions.compaction
    assert compaction is not None
    assert compaction.start_timestamp == 1.0
    assert compaction.end_timestamp == 2.0
    assert compaction.compacted_content == types.Content(
        role="model",
        parts=[types.Part(text=SUMMARY_TEXT), types.Part(text=OPEN_QUESTION)],
    )


@pytest.mark.asyncio
async def test_summary_request_uses_adk_prompt_and_lowest_effort_above_none() -> None:
    llm = FakeSummaryLlm(
        model="gpt-6-luna",
        reply=_model_reply(types.Part(text=SUMMARY_TEXT)),
    )

    await OpenAIEventsSummarizer(llm=llm).maybe_summarize_events(events=_history())

    [request] = llm.requests
    assert request.model == "gpt-6-luna"
    assert request.config.http_options is not None
    assert request.config.http_options.extra_body == {"reasoning": {"effort": "low"}}
    [prompt] = request.contents
    assert prompt.role == "user"
    prompt_text = "".join(part.text or "" for part in prompt.parts or [])
    assert "Log my lunch" in prompt_text
    assert "Logged a sandwich." in prompt_text
    assert "conversation history" in prompt_text


@pytest.mark.asyncio
async def test_non_catalog_model_gets_no_reasoning_override() -> None:
    llm = FakeSummaryLlm(
        model="some-other-model",
        reply=_model_reply(types.Part(text=SUMMARY_TEXT)),
    )

    event = await OpenAIEventsSummarizer(llm=llm).maybe_summarize_events(
        events=_history()
    )

    [request] = llm.requests
    assert request.config.http_options is None
    assert event is not None


@pytest.mark.asyncio
async def test_reasoning_only_summary_is_not_compacted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    llm = FakeSummaryLlm(
        model="gpt-6-luna",
        reply=_model_reply(types.Part(text=REASONING_SUMMARY, thought=True)),
    )

    event = await OpenAIEventsSummarizer(llm=llm).maybe_summarize_events(
        events=_history()
    )

    assert event is None
    assert "no message text" in caplog.text
    assert REASONING_SUMMARY not in caplog.text


@pytest.mark.asyncio
async def test_response_without_content_is_not_compacted() -> None:
    llm = FakeSummaryLlm(model="gpt-6-luna", reply=LlmResponse())

    event = await OpenAIEventsSummarizer(llm=llm).maybe_summarize_events(
        events=_history()
    )

    assert event is None


@pytest.mark.asyncio
async def test_model_yielding_nothing_is_not_compacted() -> None:
    llm = FakeSummaryLlm(model="gpt-6-luna", reply=None)

    event = await OpenAIEventsSummarizer(llm=llm).maybe_summarize_events(
        events=_history()
    )

    assert event is None
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_no_events_means_no_summary_call() -> None:
    llm = FakeSummaryLlm(
        model="gpt-6-luna", reply=_model_reply(types.Part(text=SUMMARY_TEXT))
    )

    event = await OpenAIEventsSummarizer(llm=llm).maybe_summarize_events(events=[])

    assert event is None
    assert llm.requests == []
