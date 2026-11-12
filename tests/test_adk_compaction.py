"""Tests for Blacki's native Google ADK context compaction configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from google.adk.agents import LlmAgent
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.compaction import _run_compaction_for_token_threshold_config
from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types
from openai import AsyncOpenAI

from blacki import app
from blacki.agent import create_app
from blacki.models.openai_responses import DEFAULT_BASE_URL, OpenAIResponsesLlm
from blacki.models.openai_summarizer import OpenAIEventsSummarizer

SUMMARY_TEXT = "Summary: the user logged lunch."


class RecordingSummarizer(BaseEventsSummarizer):
    """Return a deterministic summary while exercising ADK compaction."""

    def __init__(self) -> None:
        self.batches: list[list[Event]] = []

    async def maybe_summarize_events(self, *, events: list[Event]) -> Event | None:
        self.batches.append(events)
        return Event(
            author="user",
            invocation_id="compaction-invocation",
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=events[0].timestamp,
                    end_timestamp=events[-1].timestamp,
                    compacted_content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Compacted history")],
                    ),
                )
            ),
        )


def _conversation_event(index: int) -> Event:
    return Event(
        author="blacki",
        invocation_id=f"invocation-{index}",
        timestamp=float(index + 1),
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=f"Conversation event {index}")],
        ),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=200_000
        ),
    )


@pytest.mark.asyncio
async def test_token_compaction_is_persisted_in_sqlite(tmp_path: Path) -> None:
    """Compact older events at the threshold without creating a new session."""
    config = app.events_compaction_config
    assert config is not None
    retention_size = config.event_retention_size
    assert retention_size is not None
    summarizer = RecordingSummarizer()
    config = config.model_copy(update={"summarizer": summarizer})

    service = DatabaseSessionService(f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    try:
        session = await service.create_session(
            app_name=app.name,
            user_id="compaction-test-user",
            session_id="compaction-session-v1",
        )
        for index in range(retention_size + 1):
            await service.append_event(
                session=session,
                event=_conversation_event(index),
            )

        hydrated_session = await service.get_session(
            app_name=app.name,
            user_id="compaction-test-user",
            session_id="compaction-session-v1",
        )
        assert hydrated_session is not None
        assert app.root_agent is not None

        compacted = await _run_compaction_for_token_threshold_config(
            config=config,
            session=hydrated_session,
            session_service=service,
            agent=app.root_agent,
            agent_name="blacki",
        )

        assert compacted is True
        assert len(summarizer.batches) == 1
        assert len(summarizer.batches[0]) == 1

        persisted_session = await service.get_session(
            app_name=app.name,
            user_id="compaction-test-user",
            session_id="compaction-session-v1",
        )
        assert persisted_session is not None
        assert persisted_session.id == "compaction-session-v1"
        assert persisted_session.events[-1].actions.compaction is not None
        summary_parts = persisted_session.events[
            -1
        ].actions.compaction.compacted_content.parts
        assert summary_parts is not None
        assert summary_parts[0].text == "Compacted history"
    finally:
        await service.close()


REASONING_ITEM: dict[str, Any] = {
    "id": "rs_compaction",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "Summarizing the lunch log."}],
    "encrypted_content": "ENC_compaction_fixture",
}
MESSAGE_ITEM: dict[str, Any] = {
    "id": "msg_compaction",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": SUMMARY_TEXT, "annotations": []}],
}


def _responses_payload(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_compaction_1",
        "object": "response",
        "created_at": 1_790_000_000,
        "model": "gpt-6-luna-2026-06-01",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 900,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 60,
            "output_tokens_details": {"reasoning_tokens": 20},
            "total_tokens": 960,
        },
        "error": None,
        "incomplete_details": None,
        "store": False,
    }


@pytest.mark.asyncio
async def test_openai_compaction_stores_message_text_and_asks_for_low_effort(
    tmp_path: Path,
) -> None:
    """ADK's real compaction runs with OpenAI answering over a mocked HTTP boundary."""
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json=_responses_payload([REASONING_ITEM, MESSAGE_ITEM])
        )

    def client_factory(timeout: float, max_retries: int) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="test-key",
            base_url=DEFAULT_BASE_URL,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
            timeout=timeout,
            max_retries=max_retries,
        )

    root = OpenAIResponsesLlm(
        model="gpt-6-luna", api_key="test-key", client_factory=client_factory
    )
    agent = LlmAgent(name="blacki", model=root)
    composed = create_app(agent)
    config = composed.events_compaction_config
    assert config is not None
    assert isinstance(config.summarizer, OpenAIEventsSummarizer)
    retention_size = config.event_retention_size
    assert retention_size is not None

    user_id = "compaction-openai-user"
    session_id = "compaction-openai-session"
    service = DatabaseSessionService(f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    try:
        session = await service.create_session(
            app_name=composed.name, user_id=user_id, session_id=session_id
        )
        for index in range(retention_size + 1):
            await service.append_event(
                session=session,
                event=_conversation_event(index),
            )
        hydrated_session = await service.get_session(
            app_name=composed.name, user_id=user_id, session_id=session_id
        )
        assert hydrated_session is not None

        compacted = await _run_compaction_for_token_threshold_config(
            config=config,
            session=hydrated_session,
            session_service=service,
            agent=agent,
            agent_name="blacki",
        )

        assert compacted is True
        [body] = bodies
        assert body["reasoning"] == {"effort": "low", "summary": "auto"}
        assert body["store"] is False

        persisted = await service.get_session(
            app_name=composed.name, user_id=user_id, session_id=session_id
        )
        assert persisted is not None
        compaction = persisted.events[-1].actions.compaction
        assert compaction is not None
        parts = compaction.compacted_content.parts or []
        assert [part.text for part in parts] == [SUMMARY_TEXT]
        assert all(not part.thought for part in parts)
        assert all(part.part_metadata is None for part in parts)
    finally:
        await service.close()
