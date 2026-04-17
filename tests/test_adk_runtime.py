"""Tests for shared ADK runtime helpers."""

from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types

from blacki.adk_runtime import (
    DEFAULT_EMPTY_RESPONSE,
    AdkRuntime,
    SessionLocator,
    StreamChunk,
    TurnResponse,
    _extract_event_text,
    _extract_session_version,
    _extract_turn_parts,
    _merge_stream_fragment,
    build_session_db_kwargs,
    build_session_service_uri,
    create_adk_runtime,
    create_session_service,
)
from blacki.utils.config import ServerEnv


def _build_server_env(**overrides: str) -> ServerEnv:
    environment = {
        "AGENT_NAME": "test-agent",
    }
    environment.update(overrides)
    return ServerEnv.model_validate(environment)


def test_build_session_service_uri_converts_postgres_scheme() -> None:
    """Test that postgres URIs are normalized for asyncpg."""
    env = _build_server_env(DATABASE_URL="postgresql://user:pass@localhost/db")

    assert (
        build_session_service_uri(env) == "postgresql+asyncpg://user:pass@localhost/db"
    )


def test_build_session_service_uri_returns_none_without_config() -> None:
    """Test that missing session config returns None."""
    env = _build_server_env()

    assert build_session_service_uri(env) is None


def test_build_session_service_uri_keeps_agentengine_scheme() -> None:
    """Test that non-Postgres session URIs are returned unchanged."""
    env = _build_server_env(AGENT_ENGINE="test-engine-id")

    assert build_session_service_uri(env) == "agentengine://test-engine-id"


def test_build_session_db_kwargs_uses_env_values() -> None:
    """Test that session DB kwargs are derived from ServerEnv."""
    env = _build_server_env(
        DB_POOL_PRE_PING="false",
        DB_POOL_RECYCLE="99",
        DB_POOL_SIZE="7",
        DB_MAX_OVERFLOW="8",
        DB_POOL_TIMEOUT="9",
    )

    assert build_session_db_kwargs(env) == {
        "pool_pre_ping": False,
        "pool_recycle": 99,
        "pool_size": 7,
        "max_overflow": 8,
        "pool_timeout": 9,
    }


def test_create_session_service_without_uri_uses_in_memory() -> None:
    """Test that missing session URI falls back to ADK's in-memory service."""
    session_service = create_session_service(None, {})

    assert isinstance(session_service, InMemorySessionService)


def test_create_session_service_with_postgres_uri() -> None:
    """Test that Postgres session services use DatabaseSessionService."""
    session_service = create_session_service(
        "postgresql+asyncpg://user:pass@localhost/db",
        {},
    )

    assert session_service.__class__.__name__ == "DatabaseSessionService"


def test_create_session_service_rejects_unsupported_uri() -> None:
    """Test that unsupported session URIs fail fast."""
    with pytest.raises(ValueError, match="does not support"):
        create_session_service("agentengine://test-engine-id", {})


async def test_create_next_session_increments_version() -> None:
    """Test that reset-style session creation increments version numbers."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    first_session = await runtime.create_next_session(locator=locator)
    second_session = await runtime.create_next_session(locator=locator)

    assert first_session.id == "telegram-chat-123-v1"
    assert second_session.id == "telegram-chat-123-v2"


async def test_get_or_create_session_reuses_latest_version() -> None:
    """Test that the active session resolves to the latest version."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    initial_session = await runtime.get_or_create_session(locator=locator)
    await runtime.create_next_session(locator=locator)
    resolved_session = await runtime.get_or_create_session(locator=locator)

    assert initial_session.id == "telegram-chat-123-v1"
    assert resolved_session.id == "telegram-chat-123-v2"


async def test_run_user_turn_uses_final_non_partial_text() -> None:
    """Test that runtime returns the final non-partial assistant text."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Partial answer")],
            ),
        )
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Final answer")],
            ),
        )

    with patch.object(runtime.runner, "run_async", fake_run_async):
        response = await runtime.run_user_turn(locator=locator, message_text="Hello")

    assert response == "Final answer"


async def test_run_user_turn_returns_default_when_events_have_no_text() -> None:
    """Test that empty ADK events fall back to the default empty response."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(author="root_agent")

    with patch.object(runtime.runner, "run_async", fake_run_async):
        response = await runtime.run_user_turn(locator=locator, message_text="Hello")

    assert response == DEFAULT_EMPTY_RESPONSE


async def test_run_user_turn_raises_on_event_error() -> None:
    """Test that ADK event errors bubble up as RuntimeError."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            error_code="boom",
            error_message="runner exploded",
        )

    with (
        patch.object(runtime.runner, "run_async", fake_run_async),
        pytest.raises(RuntimeError, match="runner exploded"),
    ):
        await runtime.run_user_turn(locator=locator, message_text="Hello")


async def test_close_awaits_async_close_method() -> None:
    """Test that runtime close awaits async session service close methods."""

    class ClosableSessionService(InMemorySessionService):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    session_service = ClosableSessionService()
    runtime = AdkRuntime(session_service)

    await runtime.close()

    assert session_service.closed is True


async def test_close_returns_when_session_service_has_no_close() -> None:
    """Test that runtime close is a no-op without a close method."""
    runtime = AdkRuntime(InMemorySessionService())

    await runtime.close()


async def test_close_handles_sync_close_method() -> None:
    """Test that runtime close supports synchronous close methods."""

    class SyncClosableSessionService(InMemorySessionService):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session_service = SyncClosableSessionService()
    runtime = AdkRuntime(session_service)

    await runtime.close()

    assert session_service.closed is True


async def test_get_or_create_session_ignores_malformed_session_ids() -> None:
    """Test that non-versioned session IDs are filtered out during lookup."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    await runtime.session_service.create_session(
        app_name=runtime.app_name,
        user_id=locator.user_id,
        session_id="telegram-chat-123-malformed",
    )
    await runtime.session_service.create_session(
        app_name=runtime.app_name,
        user_id=locator.user_id,
        session_id="telegram-chat-123-v1",
    )

    session = await runtime.get_or_create_session(locator=locator)

    assert session.id == "telegram-chat-123-v1"


async def test_get_or_create_session_creates_v1_when_no_valid_sessions() -> None:
    """Test that session v1 is created when only malformed sessions exist."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-456",
        session_id_prefix="telegram-chat-456",
    )

    await runtime.session_service.create_session(
        app_name=runtime.app_name,
        user_id=locator.user_id,
        session_id="telegram-chat-456-invalid",
    )

    session = await runtime.get_or_create_session(locator=locator)

    assert session.id == "telegram-chat-456-v1"


def test_create_adk_runtime_uses_env_configuration() -> None:
    """Test shared runtime construction from environment config."""
    runtime = create_adk_runtime(_build_server_env())

    assert isinstance(runtime.session_service, InMemorySessionService)


def test_extract_session_version_rejects_invalid_format() -> None:
    """Test that malformed versioned session IDs fail fast."""
    with pytest.raises(ValueError, match="Unexpected session id format"):
        _extract_session_version(
            session_id="telegram-chat-123-vnot-a-number",
            session_id_prefix="telegram-chat-123",
        )


def test_extract_event_text_concatenates_parts_directly() -> None:
    """Test that event text extraction concatenates parts without breaking."""
    event = Event(
        author="root_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="Hello "),
                types.Part.from_text(text="world"),
            ],
        ),
    )

    assert _extract_event_text(event) == "Hello  world"


def test_extract_event_text_returns_empty_without_content() -> None:
    """Test that missing content returns an empty string."""
    assert _extract_event_text(Event(author="root_agent")) == ""


def test_extract_event_text_skips_parts_without_text() -> None:
    """Test that non-text parts are ignored during extraction."""
    event = Event(
        author="root_agent",
        content=types.Content(
            role="model",
            parts=[types.Part()],
        ),
    )

    assert _extract_event_text(event) == ""


def test_extract_event_text_handles_leading_trailing_whitespace() -> None:
    """Test that whitespace is stripped from the final concatenated result."""
    event = Event(
        author="root_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="  Hello "),
                types.Part.from_text(text="world  "),
            ],
        ),
    )

    assert _extract_event_text(event) == "Hello  world"


def test_extract_turn_parts_separates_thoughts_from_content() -> None:
    """Test that thoughts are separated from regular content."""
    event = Event(
        author="root_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="Thinking hard...", thought=True),
                types.Part(text="Final answer", thought=False),
            ],
        ),
    )

    thoughts, content = _extract_turn_parts(event)

    assert thoughts == "Thinking hard..."
    assert content == "Final answer"


def test_extract_turn_parts_handles_only_thoughts() -> None:
    """Test event with only thought parts."""
    event = Event(
        author="root_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="First thought", thought=True),
                types.Part(text="Second thought", thought=True),
            ],
        ),
    )

    thoughts, content = _extract_turn_parts(event)

    assert thoughts == "First thought Second thought"
    assert content == ""


def test_extract_turn_parts_handles_only_content() -> None:
    """Test event with only content parts (no thoughts)."""
    event = Event(
        author="root_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="Hello "),
                types.Part(text="world"),
            ],
        ),
    )

    thoughts, content = _extract_turn_parts(event)

    assert thoughts == ""
    assert content == "Hello  world"


def test_extract_turn_parts_handles_empty_event() -> None:
    """Test event with no content returns empty strings."""
    event = Event(author="root_agent")

    thoughts, content = _extract_turn_parts(event)

    assert thoughts == ""
    assert content == ""


def test_extract_turn_parts_skips_empty_parts() -> None:
    """Test that parts without text are skipped."""
    event = Event(
        author="root_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="Content", thought=False),
                types.Part(),
                types.Part(text="More", thought=False),
            ],
        ),
    )

    thoughts, content = _extract_turn_parts(event)

    assert thoughts == ""
    assert content == "Content More"


async def test_run_user_turn_with_thoughts_returns_structured_response() -> None:
    """Test that run_user_turn_with_thoughts separates thoughts and content."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Analyzing the question...", thought=True),
                    types.Part(text="Here is my answer."),
                ],
            ),
        )

    with patch.object(runtime.runner, "run_async", fake_run_async):
        response = await runtime.run_user_turn_with_thoughts(
            locator=locator, message_text="Hello"
        )

    assert isinstance(response, TurnResponse)
    assert response.thoughts == "Analyzing the question..."
    assert response.content == "Here is my answer."


async def test_run_user_turn_with_thoughts_handles_partial_thoughts() -> None:
    """Test that partial thoughts are used when no final thoughts are available."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Partial thinking...", thought=True),
                ],
            ),
        )
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Final answer."),
                ],
            ),
        )

    with patch.object(runtime.runner, "run_async", fake_run_async):
        response = await runtime.run_user_turn_with_thoughts(
            locator=locator, message_text="Hello"
        )

    assert response.thoughts == "Partial thinking..."
    assert response.content == "Final answer."


async def test_run_user_turn_streaming_yields_chunks() -> None:
    """Test that streaming yields chunks as events arrive."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Thinking...", thought=True),
                ],
            ),
        )
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Partial answer"),
                ],
            ),
        )
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Analyzing...", thought=True),
                    types.Part(text="Final answer."),
                ],
            ),
        )

    chunks: list[StreamChunk] = []
    with patch.object(runtime.runner, "run_async", fake_run_async):
        async for chunk in runtime.run_user_turn_streaming(
            locator=locator, message_text="Hello"
        ):
            chunks.append(chunk)

    assert len(chunks) >= 3
    assert chunks[-1].is_partial is False
    assert chunks[-1].thoughts == "Thinking...Analyzing..."
    assert chunks[-1].content == "Partial answerFinal answer."


async def test_run_user_turn_streaming_handles_partial_content() -> None:
    """Test that partial content is accumulated correctly."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Building..."),
                ],
            ),
        )
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Complete."),
                ],
            ),
        )

    chunks: list[StreamChunk] = []
    with patch.object(runtime.runner, "run_async", fake_run_async):
        async for chunk in runtime.run_user_turn_streaming(
            locator=locator, message_text="Hello"
        ):
            chunks.append(chunk)

    assert len(chunks) >= 2
    assert "Building..." in chunks[0].content
    assert chunks[-1].content == "Building...Complete."
    assert chunks[-1].is_partial is False


async def test_run_user_turn_streaming_preserves_partial_whitespace() -> None:
    """Test that streaming keeps token spacing when partial chunks include spaces."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[types.Part(text="Hi")],
            ),
        )
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[types.Part(text=" Chirag")],
            ),
        )
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[types.Part(text="Hi Chirag!")],
            ),
        )

    chunks: list[StreamChunk] = []
    with patch.object(runtime.runner, "run_async", fake_run_async):
        async for chunk in runtime.run_user_turn_streaming(
            locator=locator, message_text="Hello"
        ):
            chunks.append(chunk)

    assert len(chunks) >= 3
    assert chunks[1].content == "Hi Chirag"
    assert chunks[-1].content == "Hi Chirag!"


async def test_run_user_turn_streaming_does_not_duplicate_final_snapshot() -> None:
    """Test that a final full snapshot replaces partial chunks instead of appending."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[types.Part(text="Hi Chir")],
            ),
        )
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(
                role="model",
                parts=[types.Part(text="ag!")],
            ),
        )
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[types.Part(text="Hi Chirag!")],
            ),
        )

    chunks: list[StreamChunk] = []
    with patch.object(runtime.runner, "run_async", fake_run_async):
        async for chunk in runtime.run_user_turn_streaming(
            locator=locator, message_text="Hello"
        ):
            chunks.append(chunk)

    assert len(chunks) >= 3
    assert chunks[-1].content == "Hi Chirag!"


class TestMergeStreamFragment:
    """Tests for _merge_stream_fragment function."""

    def test_returns_incoming_when_existing_is_empty(self) -> None:
        """Test that empty existing text returns incoming."""
        result = _merge_stream_fragment("", "Hello")
        assert result == "Hello"

    def test_returns_existing_when_incoming_is_empty(self) -> None:
        """Test that empty incoming text returns existing."""
        result = _merge_stream_fragment("Hello", "")
        assert result == "Hello"

    def test_returns_incoming_when_it_starts_with_existing(self) -> None:
        """Test snapshot-style merge where incoming contains existing."""
        result = _merge_stream_fragment("Hello", "Hello world")
        assert result == "Hello world"

    def test_returns_existing_when_it_starts_with_incoming(self) -> None:
        """Test case where existing is longer than incoming."""
        result = _merge_stream_fragment("Hello world", "Hello")
        assert result == "Hello world"

    def test_merges_with_overlap(self) -> None:
        """Test delta-style merge with overlapping suffix/prefix."""
        result = _merge_stream_fragment("Hello wor", "world!")
        assert result == "Hello world!"

    def test_concatenates_when_no_overlap(self) -> None:
        """Test concatenation when no overlap exists."""
        result = _merge_stream_fragment("Hello ", "world")
        assert result == "Hello world"

    def test_handles_unicode_characters(self) -> None:
        """Test that unicode characters are merged correctly."""
        result = _merge_stream_fragment("Hello ", "世界!")
        assert result == "Hello 世界!"

    def test_handles_emoji(self) -> None:
        """Test that emoji characters are merged correctly."""
        result = _merge_stream_fragment("Hello ", "👋")
        assert result == "Hello 👋"

    def test_handles_single_character(self) -> None:
        """Test single character merges."""
        result = _merge_stream_fragment("a", "ab")
        assert result == "ab"

    def test_handles_empty_strings(self) -> None:
        """Test both empty strings."""
        result = _merge_stream_fragment("", "")
        assert result == ""

    def test_full_overlap_returns_existing(self) -> None:
        """Test when incoming is a prefix of existing."""
        result = _merge_stream_fragment("Hello world", "Hello")
        assert result == "Hello world"

    def test_exact_match_returns_either(self) -> None:
        """Test when strings are identical."""
        result = _merge_stream_fragment("Hello", "Hello")
        assert result == "Hello"
