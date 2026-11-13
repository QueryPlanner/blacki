"""Compaction summarizer for sessions whose root model is OpenAI (design D12).

ADK's default summarizer stores the first response's whole content as the
compacted turn, so reasoning parts would be replayed into later requests. This
subclass keeps ADK's prompt and event formatting, asks the model for its lowest
reasoning effort above ``none``, and stores only readable message text.
"""

from __future__ import annotations

import logging

from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from .catalog import get_model, lowest_effort_above_none

logger = logging.getLogger(__name__)


class OpenAIEventsSummarizer(LlmEventSummarizer):
    """Summarize compacted events with an OpenAI model, keeping message text only.

    The summary call is an ordinary model call, so the adapter records its usage
    like any other call.
    """

    async def maybe_summarize_events(self, *, events: list[Event]) -> Event | None:
        """Return a compaction event whose content is the summary's message text.

        Returns ``None`` when there is nothing to compact or the model returned no
        message text. ADK then keeps the events uncompacted.
        """

        if not events:
            return None

        conversation_history = self._format_events_for_prompt(events)
        prompt = self._prompt_template.format(conversation_history=conversation_history)
        request = LlmRequest(
            model=self._llm.model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=_summary_config(self._llm.model),
        )

        summary_content: types.Content | None = None
        usage_metadata: types.GenerateContentResponseUsageMetadata | None = None
        async for response in self._llm.generate_content_async(request, stream=False):
            if response.content:
                summary_content = response.content
                usage_metadata = response.usage_metadata
                break

        if summary_content is None:
            return None

        text_parts = [
            types.Part(text=part.text)
            for part in summary_content.parts or []
            if part.text and not part.thought
        ]
        if not text_parts:
            logger.warning("Compaction summary had no message text; not compacting.")
            return None

        return Event(
            author="user",
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=events[0].timestamp,
                    end_timestamp=events[-1].timestamp,
                    compacted_content=types.Content(role="model", parts=text_parts),
                )
            ),
            invocation_id=Event.new_id(),
            usage_metadata=usage_metadata,
        )


def _summary_config(model: str) -> types.GenerateContentConfig:
    """Request the lowest reasoning effort above ``none`` for catalog models."""

    entry = get_model(model)
    if entry is None:
        return types.GenerateContentConfig()
    effort = lowest_effort_above_none(entry)
    return types.GenerateContentConfig(
        http_options=types.HttpOptions(extra_body={"reasoning": {"effort": effort}})
    )
