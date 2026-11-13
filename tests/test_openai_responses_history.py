"""Pin the ADK 2.5.0 history behaviors the OpenAI Responses adapter depends on.

Design decisions (D2, D3, D5) are defined in
openspec/changes/replace-openrouter-with-openai-responses/design.md.

These tests run ADK's real contents request processor, real Event and Session
objects, and a real SQLite-backed DatabaseSessionService in tmp_path. The only
stand-in is a BaseLlm subclass that is never called, so ADK takes the same
contents path it takes for a custom model.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events import Event
from google.adk.flows.llm_flows.contents import request_processor
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types

APP_NAME = "blacki"
USER_ID = "telegram-user"
AGENT_NAME = "blacki"
MODEL_ID = "gpt-6.1-terra"

# Raw Responses reasoning item, as the adapter stores it (design D2).
REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [],
    "encrypted_content": "enc",
}


class ResponsesLikeLlm(BaseLlm):
    """Stand-in for the OpenAI Responses adapter. The model is never called.

    It is a custom BaseLlm, not one of the types ADK's contents processor
    special-cases for keeping adk-* function call ids (AnthropicLlm, LiteLlm,
    labs OpenAIResponsesLlm). That gives these tests the id handling the real
    adapter will see.
    """

    def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        raise AssertionError("request-building tests must not call the model")


def _agent() -> LlmAgent:
    return LlmAgent(name=AGENT_NAME, model=ResponsesLikeLlm(model=MODEL_ID))


def _session(events: list[Event], session_id: str = "session-1") -> Session:
    return Session(id=session_id, app_name=APP_NAME, user_id=USER_ID, events=events)


def _thought_part() -> types.Part:
    """The empty-summary reasoning part the adapter emits (design D2, D3)."""
    return types.Part(
        text="",
        thought=True,
        part_metadata={"blacki_openai": {"model": MODEL_ID, "item": REASONING_ITEM}},
    )


def _tool_turn_events(first_call_id: str, second_call_id: str) -> list[Event]:
    """One user turn, then a model turn with reasoning and two parallel calls,
    then the results. Tool results are authored by the agent, as ADK authors
    its own tool-response events, and carry the user role.
    """
    return [
        Event(
            author="user",
            invocation_id="invocation-1",
            timestamp=1.0,
            content=types.Content(role="user", parts=[types.Part(text="Plan my week")]),
        ),
        Event(
            author=AGENT_NAME,
            invocation_id="invocation-1",
            timestamp=2.0,
            content=types.Content(
                role="model",
                parts=[
                    _thought_part(),
                    types.Part(
                        function_call=types.FunctionCall(
                            id=first_call_id,
                            name="lookup_meal",
                            args={"day": "monday"},
                        )
                    ),
                    types.Part(
                        function_call=types.FunctionCall(
                            id=second_call_id,
                            name="lookup_meal",
                            args={"day": "tuesday"},
                        )
                    ),
                ],
            ),
        ),
        Event(
            author=AGENT_NAME,
            invocation_id="invocation-1",
            timestamp=3.0,
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=first_call_id,
                            name="lookup_meal",
                            response={"meal": "oats"},
                        )
                    ),
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=second_call_id,
                            name="lookup_meal",
                            response={"meal": "eggs"},
                        )
                    ),
                ],
            ),
        ),
    ]


async def _request_contents(
    agent: LlmAgent,
    session: Session,
    session_service: InMemorySessionService | DatabaseSessionService,
) -> list[types.Content]:
    """Build the LLM request contents with ADK's real contents processor."""
    invocation_context = InvocationContext(
        session_service=session_service,
        invocation_id="invocation-2",
        agent=agent,
        session=session,
    )
    llm_request = LlmRequest()
    # The processor yields no events. It only fills llm_request.contents.
    async for _ in request_processor.run_async(invocation_context, llm_request):
        pass
    return llm_request.contents


async def test_model_turn_keeps_empty_thought_and_parallel_calls_in_order() -> None:
    """Protects D2 and D3: ADK keeps an empty-summary reasoning part, followed by
    the parallel function calls, in order, in the model turn it replays.
    Losing the reasoning part would break the same-model replay.
    """
    session = _session(_tool_turn_events("call_a", "call_b"))

    contents = await _request_contents(_agent(), session, InMemorySessionService())

    assert [content.role for content in contents] == ["user", "model", "user"]

    model_parts = contents[1].parts
    assert model_parts is not None
    assert len(model_parts) == 3
    thought, first_call, second_call = model_parts
    assert thought.thought is True
    assert thought.text == ""
    assert thought.part_metadata == _thought_part().part_metadata
    assert first_call.function_call is not None
    assert first_call.function_call.id == "call_a"
    assert second_call.function_call is not None
    assert second_call.function_call.id == "call_b"

    result_parts = contents[2].parts
    assert result_parts is not None
    result_ids = [
        part.function_response.id
        for part in result_parts
        if part.function_response is not None
    ]
    assert result_ids == ["call_a", "call_b"]


async def test_part_metadata_and_empty_thought_part_survive_database_reload(
    tmp_path: Path,
) -> None:
    """Protects D2 and the restart claim: DatabaseSessionService stores each
    event as JSON, so part_metadata and the empty thought part come back
    unchanged after a reload. A restart then replays the same request contents.
    """
    events = _tool_turn_events("call_a", "call_b")
    expected_contents = await _request_contents(
        _agent(), _session(events), InMemorySessionService()
    )

    service = DatabaseSessionService(f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    try:
        session = await service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id="session-1"
        )
        for event in events:
            await service.append_event(session=session, event=event)

        reloaded = await service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id="session-1"
        )
        assert reloaded is not None
        assert len(reloaded.events) == 3

        reloaded_content = reloaded.events[1].content
        assert reloaded_content is not None
        reloaded_model_parts = reloaded_content.parts
        assert reloaded_model_parts is not None
        reloaded_thought = reloaded_model_parts[0]
        assert reloaded_thought.thought is True
        assert reloaded_thought.text == ""
        assert reloaded_thought.part_metadata == _thought_part().part_metadata

        actual_contents = await _request_contents(_agent(), reloaded, service)
    finally:
        await service.close()

    assert actual_contents == expected_contents


async def test_adk_generated_call_ids_reach_custom_model_as_none() -> None:
    """Protects D5: for a custom BaseLlm, ADK sets adk-* call and result ids to
    None in the request. The adapter then pairs calls with results by order and
    creates its own call_blacki_* ids. If ADK kept the ids, the adapter would
    pair them by id instead, and the order-based path would never run.
    """
    events = [
        Event(
            author="user",
            invocation_id="invocation-1",
            timestamp=1.0,
            content=types.Content(role="user", parts=[types.Part(text="Plan my week")]),
        ),
        Event(
            author=AGENT_NAME,
            invocation_id="invocation-1",
            timestamp=2.0,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="adk-call-1",
                            name="lookup_meal",
                            args={"day": "monday"},
                        )
                    )
                ],
            ),
        ),
        Event(
            author=AGENT_NAME,
            invocation_id="invocation-1",
            timestamp=3.0,
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="adk-call-1",
                            name="lookup_meal",
                            response={"meal": "oats"},
                        )
                    )
                ],
            ),
        ),
    ]
    session = _session(events)

    contents = await _request_contents(_agent(), session, InMemorySessionService())

    model_parts = contents[1].parts
    result_parts = contents[2].parts
    assert model_parts is not None
    assert result_parts is not None
    call = model_parts[0].function_call
    result = result_parts[0].function_response
    assert call is not None
    assert result is not None
    assert call.id is None
    assert result.id is None

    # The stored session event keeps its id. Only the request copy is cleared.
    stored_content = session.events[1].content
    assert stored_content is not None
    assert stored_content.parts is not None
    stored_call = stored_content.parts[0].function_call
    assert stored_call is not None
    assert stored_call.id == "adk-call-1"
