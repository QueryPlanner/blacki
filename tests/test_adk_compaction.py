"""Tests for Blacki's native Google ADK context compaction configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.compaction import _run_compaction_for_token_threshold_config
from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types

from blacki import app


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
