# mypy: disable-error-code="no-untyped-def"
"""Tests for shared ADK runtime helpers."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService, Session
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types

from blacki.adk_runtime import (
    MODEL_RETURNED_NO_CONTENT,
    AdkRuntime,
    EmptyModelResponseError,
    SessionLocator,
    StreamChunk,
    TurnResponse,
    _confirmation_response,
    _extract_session_version,
    _format_confirmation,
    _pending_confirmations,
    _provider_from_model,
    build_session_db_kwargs,
    build_session_service_uri,
    create_adk_runtime,
    create_session_service,
)
from blacki.inference import (
    InferenceProfile,
    ReasoningConfig,
    ReasoningEffort,
    get_active_inference_profile,
)
from blacki.utils.config import ServerEnv


def _build_server_env(**overrides: str) -> ServerEnv:
    environment = {
        "AGENT_NAME": "test-agent",
    }
    environment.update(overrides)
    return ServerEnv.model_validate(environment)


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        (None, None),
        ("gemini-2.5-flash", "google"),
        ("openrouter/test-model", "openrouter"),
        ("custom-model", None),
    ],
)
def test_provider_from_model(model: str | None, provider: str | None) -> None:
    assert _provider_from_model(model) == provider


def test_build_session_service_uri_ignores_database_url() -> None:
    """Test that DATABASE_URL is ignored for session URIs (reserved for Reminders)."""
    env = _build_server_env(DATABASE_URL="postgresql://user:pass@localhost/db")

    assert build_session_service_uri(env) is None


def test_build_session_service_uri_returns_none_without_config() -> None:
    """Test that missing session config returns None."""
    env = _build_server_env()

    assert build_session_service_uri(env) is None


def test_build_session_service_uri_keeps_agentengine_scheme() -> None:
    """Test that non-Postgres session URIs are returned unchanged."""
    env = _build_server_env(AGENT_ENGINE="test-engine-id")

    assert build_session_service_uri(env) == "agentengine://test-engine-id"


def test_build_session_service_uri_converts_postgresql_to_asyncpg() -> None:
    """Test that postgresql:// URIs are converted to postgresql+asyncpg://."""
    env = _build_server_env()

    with patch.object(
        type(env),
        "session_uri",
        property(lambda self: "postgresql://user:pass@localhost/db"),
    ):
        assert (
            build_session_service_uri(env)
            == "postgresql+asyncpg://user:pass@localhost/db"
        )


def test_build_session_db_kwargs_returns_timeout_for_sqlite() -> None:
    """Test that session DB kwargs returns timeout config for SQLite."""
    env = _build_server_env()

    assert build_session_db_kwargs(env) == {"connect_args": {"timeout": 15}}


def test_build_session_db_kwargs_returns_empty_for_postgres() -> None:
    """Test that session DB kwargs returns empty dict for PostgreSQL."""
    env = _build_server_env(agent_engine="postgresql://localhost:5432/db")

    assert build_session_db_kwargs(env) == {}


def test_create_session_service_without_uri_uses_sqlite(tmp_path: Path) -> None:
    """Test that missing session URI falls back to SQLite service."""
    session_service = create_session_service(None, {}, agent_dir=str(tmp_path))

    assert isinstance(session_service, DatabaseSessionService)


def test_create_session_service_with_sqlite_uri(tmp_path: Path) -> None:
    """Test that SQLite session services use DatabaseSessionService."""
    db_path = tmp_path / "sessions.db"
    session_service = create_session_service(
        f"sqlite+aiosqlite:///{db_path}",
        {},
    )

    assert isinstance(session_service, DatabaseSessionService)


def test_create_session_service_sqlite_event_listener(tmp_path: Path) -> None:
    """Test that SQLite session services execute PRAGMAs on connect."""
    db_path = tmp_path / "sessions.db"

    with patch("sqlalchemy.event.listens_for") as mock_listens_for:
        create_session_service(
            f"sqlite+aiosqlite:///{db_path}",
            {},
        )

        # Get the decorator returned by event.listens_for and the function it wrapped
        decorator = mock_listens_for.return_value
        set_sqlite_pragma = decorator.call_args[0][0]

        # Call the wrapped function with a mock connection
        mock_conn = MagicMock()
        set_sqlite_pragma(mock_conn, None)

        # Verify it executed the pragmas
        cursor = mock_conn.cursor.return_value
        assert cursor.execute.call_count == 2
        cursor.execute.assert_any_call("PRAGMA journal_mode=WAL")
        cursor.execute.assert_any_call("PRAGMA synchronous=NORMAL")
        cursor.close.assert_called_once()


@patch("blacki.adk_runtime.DatabaseSessionService")
def test_create_session_service_with_postgres_uri(mock_db_service: AsyncMock) -> None:
    """Test that PostgreSQL session services skip SQLite PRAGMAs."""
    session_service = create_session_service(
        "postgresql+asyncpg://user:pass@localhost/db",
        {},
    )

    # Since we mocked it, it returns the mock object
    assert session_service == mock_db_service.return_value


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


@pytest.mark.asyncio
async def test_run_user_turn_scopes_inference_profile_across_runner_stream() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-profile",
        session_id_prefix="telegram-chat-profile",
    )
    profile = InferenceProfile(
        model="openrouter/openai/gpt-5.6-luna",
        reasoning=ReasoningConfig(effort=ReasoningEffort.MAX),
    )
    observed: list[InferenceProfile | None] = []

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        observed.append(get_active_inference_profile())
        yield Event(
            author="root_agent",
            content=types.Content(
                role="model", parts=[types.Part.from_text(text="Scoped")]
            ),
        )

    with patch.object(runtime.runner, "run_async", fake_run_async):
        response = await runtime.run_user_turn(
            locator=locator,
            message_text="Hello",
            inference_profile=profile,
        )

    assert response == "Scoped"
    assert observed == [profile]
    assert get_active_inference_profile() is None


@pytest.mark.asyncio
async def test_run_user_turn_resets_inference_profile_after_runner_error() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-error",
        session_id_prefix="telegram-chat-error",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        assert get_active_inference_profile() is not None
        yield Event(
            author="root_agent",
            error_code="model_failed",
            error_message="model failed",
        )

    with (
        patch.object(runtime.runner, "run_async", fake_run_async),
        pytest.raises(RuntimeError, match="model failed"),
    ):
        await runtime.run_user_turn(
            locator=locator,
            message_text="Hello",
            inference_profile=InferenceProfile(model="error-model"),
        )

    assert get_active_inference_profile() is None


@pytest.mark.asyncio
async def test_run_user_turn_consumes_empty_response_event_before_raising() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-empty",
        session_id_prefix="telegram-chat-empty",
    )
    runner_finished = False

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            invocation_id="empty-invocation",
            model_version="openrouter/test-model",
            error_code=MODEL_RETURNED_NO_CONTENT,
            error_message=(
                "The model returned no content (finish_reason=STOP with empty parts)."
            ),
        )
        yield Event(author="root_agent")
        nonlocal runner_finished
        runner_finished = True

    with (
        patch.object(runtime.runner, "run_async", fake_run_async),
        pytest.raises(EmptyModelResponseError) as raised,
    ):
        await runtime.run_user_turn(
            locator=locator,
            message_text="Hello",
            inference_profile=InferenceProfile(model="openrouter/test-model"),
        )

    error = raised.value
    assert runner_finished is True
    assert error.model == "openrouter/test-model"
    assert error.provider == "openrouter"
    assert error.invocation_id == "empty-invocation"
    assert error.retryable is True
    assert get_active_inference_profile() is None


@pytest.mark.asyncio
async def test_run_user_turn_marks_empty_response_after_unusable_events() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-empty-final",
        session_id_prefix="telegram-chat-empty-final",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            invocation_id="empty-final-invocation",
            model_version="openrouter/test-model",
            content=types.Content(role="model", parts=[]),
        )

    with (
        patch.object(runtime.runner, "run_async", fake_run_async),
        pytest.raises(EmptyModelResponseError) as raised,
    ):
        await runtime.run_user_turn(locator=locator, message_text="Hello")

    error = raised.value
    assert error.model == "openrouter/test-model"
    assert error.provider == "openrouter"
    assert error.invocation_id == "empty-final-invocation"
    assert error.retryable is True


@pytest.mark.asyncio
async def test_run_user_turn_resets_inference_profile_after_cancellation() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-cancel",
        session_id_prefix="telegram-chat-cancel",
    )
    started = asyncio.Event()

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        assert get_active_inference_profile() is not None
        started.set()
        await asyncio.Event().wait()
        yield Event(author="root_agent")

    with patch.object(runtime.runner, "run_async", fake_run_async):
        task = asyncio.create_task(
            runtime.run_user_turn(
                locator=locator,
                message_text="Hello",
                inference_profile=InferenceProfile(model="cancel-model"),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert get_active_inference_profile() is None


@pytest.mark.asyncio
async def test_concurrent_turns_keep_inference_profiles_isolated() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    first_locator = SessionLocator(
        user_id="telegram-chat-first",
        session_id_prefix="telegram-chat-first",
    )
    second_locator = SessionLocator(
        user_id="telegram-chat-second",
        session_id_prefix="telegram-chat-second",
    )
    profiles = {
        first_locator.session_id_prefix: InferenceProfile(model="first-model"),
        second_locator.session_id_prefix: InferenceProfile(model="second-model"),
    }
    ready = set[str]()
    all_ready = asyncio.Event()
    observed: dict[str, InferenceProfile | None] = {}

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        session_id = str(kwargs["session_id"])
        observed[session_id] = get_active_inference_profile()
        ready.add(session_id)
        if len(ready) == 2:
            all_ready.set()
        await all_ready.wait()
        yield Event(
            author="root_agent",
            content=types.Content(
                role="model", parts=[types.Part.from_text(text=session_id)]
            ),
        )

    with patch.object(runtime.runner, "run_async", fake_run_async):
        first, second = await asyncio.gather(
            runtime.run_user_turn(
                locator=first_locator,
                message_text="first",
                inference_profile=profiles[first_locator.session_id_prefix],
            ),
            runtime.run_user_turn(
                locator=second_locator,
                message_text="second",
                inference_profile=profiles[second_locator.session_id_prefix],
            ),
        )

    assert first == f"{first_locator.session_id_prefix}-v1"
    assert second == f"{second_locator.session_id_prefix}-v1"
    assert observed == {f"{key}-v1": value for key, value in profiles.items()}
    assert get_active_inference_profile() is None


async def test_run_user_turn_sends_multimodal_parts_to_runner() -> None:
    """Runtime should preserve text and inline image data in the ADK message."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )
    image_bytes = b"\xff\xd8\xffimage"
    user_parts = (
        types.Part.from_text(text="Describe this image."),
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        new_message = kwargs["new_message"]
        assert isinstance(new_message, types.Content)
        assert new_message.role == "user"
        assert new_message.parts is not None
        assert new_message.parts[0].text == "Describe this image."
        assert new_message.parts[1].inline_data is not None
        assert new_message.parts[1].inline_data.mime_type == "image/jpeg"
        assert new_message.parts[1].inline_data.data == image_bytes
        yield Event(
            author="root_agent",
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="An image")],
            ),
        )

    with patch.object(runtime.runner, "run_async", fake_run_async):
        response = await runtime.run_user_turn(
            locator=locator,
            message_text="Describe this image.",
            user_parts=user_parts,
        )

    assert response == "An image"


async def test_run_user_turn_raises_for_events_with_no_text() -> None:
    """Test that empty ADK events become a recoverable model error."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(author="root_agent")

    with (
        patch.object(runtime.runner, "run_async", fake_run_async),
        pytest.raises(EmptyModelResponseError),
    ):
        await runtime.run_user_turn(locator=locator, message_text="Hello")


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

    with patch.object(runtime.runner, "close", new=AsyncMock()) as close_runner:
        await runtime.close()

    close_runner.assert_awaited_once()
    assert session_service.closed is True


async def test_close_returns_when_session_service_has_no_close() -> None:
    """Test that runtime close is a no-op without a close method."""
    runtime = AdkRuntime(InMemorySessionService())

    with patch.object(runtime.runner, "close", new=AsyncMock()) as close_runner:
        await runtime.close()

    close_runner.assert_awaited_once()


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

    with patch.object(runtime.runner, "close", new=AsyncMock()) as close_runner:
        await runtime.close()

    close_runner.assert_awaited_once()
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


def test_create_adk_runtime_uses_env_configuration(tmp_path: Path) -> None:
    """Test shared runtime construction from environment config."""
    env = _build_server_env()
    env.agent_dir = str(tmp_path)
    runtime = create_adk_runtime(env)

    assert isinstance(runtime.session_service, DatabaseSessionService)


def test_runtime_accepts_transport_specific_app() -> None:
    """A Telegram runtime can use an app isolated from the HTTP runner."""
    from blacki.agent import app as transport_app

    runtime = AdkRuntime(
        InMemorySessionService(),
        agent_app=transport_app,
    )

    assert runtime.app is transport_app
    assert runtime.runner.app is transport_app


def test_create_adk_runtime_uses_mem0_when_client_available(
    tmp_path: Path,
) -> None:
    """Test runtime uses Mem0MemoryService when client is available."""
    from unittest.mock import MagicMock

    from blacki.memory.mem0_memory_service import Mem0MemoryService

    env = _build_server_env()
    env.agent_dir = str(tmp_path)

    mock_client = MagicMock()
    with patch("blacki.memory.config.get_memory_client", return_value=mock_client):
        runtime = create_adk_runtime(env)

    assert isinstance(runtime.runner.memory_service, Mem0MemoryService)


def test_create_adk_runtime_falls_back_to_in_memory_when_mem0_unavailable(
    tmp_path: Path,
) -> None:
    """Test runtime falls back to InMemoryMemoryService when Mem0 unavailable."""
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

    env = _build_server_env()
    env.agent_dir = str(tmp_path)

    with patch("blacki.memory.config.get_memory_client", return_value=None):
        runtime = create_adk_runtime(env)

    assert isinstance(runtime.runner.memory_service, InMemoryMemoryService)


def test_extract_session_version_rejects_invalid_format() -> None:
    """Test that malformed versioned session IDs fail fast."""
    with pytest.raises(ValueError, match="Unexpected session id format"):
        _extract_session_version(
            session_id="telegram-chat-123-vnot-a-number",
            session_id_prefix="telegram-chat-123",
        )


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


async def test_get_or_create_session_merges_state() -> None:
    """Test that get_or_create_session merges state into existing sessions."""
    service = InMemorySessionService()
    runtime = AdkRuntime(service)

    locator = SessionLocator(
        user_id="telegram-chat-789",
        session_id_prefix="telegram-chat-789",
    )

    initial_session = await runtime.get_or_create_session(
        locator=locator, state={"key_a": "val_a"}
    )
    assert initial_session.state.get("key_a") == "val_a"

    session_again = await runtime.get_or_create_session(
        locator=locator, state={"key_b": "val_b"}
    )
    assert session_again.state.get("key_a") == "val_a"
    assert session_again.state.get("key_b") == "val_b"


async def test_get_or_create_session_no_update_session() -> None:
    service = InMemorySessionService()
    runtime = AdkRuntime(service)

    # Ensure no update_session
    if hasattr(service, "update_session"):
        delattr(service, "update_session")

    locator = SessionLocator(
        user_id="telegram-chat-999",
        session_id_prefix="telegram-chat-999",
    )

    await runtime.get_or_create_session(locator=locator, state={"key": "val"})
    await runtime.get_or_create_session(locator=locator, state={"key2": "val2"})


async def test_get_or_create_session_no_session_service() -> None:
    service = InMemorySessionService()
    runtime = AdkRuntime(service)

    # Remove session_service
    runtime.runner.session_service = None  # type: ignore[assignment]

    locator = SessionLocator(
        user_id="telegram-chat-888",
        session_id_prefix="telegram-chat-888",
    )

    # Initial get_or_create_session will use _get_latest_session which
    # depends on runtime's logic.
    # Wait, if session_service is None, _get_latest_session might crash.
    # Let's just mock _get_latest_session directly for this test.
    with patch.object(runtime, "_get_latest_session", AsyncMock()) as mock_latest:
        mock_latest.return_value = type("MockSession", (), {"state": {}})()
        await runtime.get_or_create_session(locator=locator, state={"key": "val"})


async def test_run_user_turn_with_thoughts_skips_function_calls() -> None:
    """Test that function_call events are excluded from final content."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-456",
        session_id_prefix="telegram-chat-456",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        part_with_fc = types.Part.from_text(text="fc_text")
        part_with_fc.function_call = types.FunctionCall(name="test_function")
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(role="model", parts=[part_with_fc]),
        )
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Final answer after fc")],
            ),
        )

    with patch.object(runtime.runner, "run_async", fake_run_async):
        response = await runtime.run_user_turn_with_thoughts(
            locator=locator, message_text="Hello"
        )

    assert response.content == "Final answer after fc"


def _confirmation_event(
    *,
    interrupt_id: str = "confirm-1",
    original_id: str = "tool-1",
    tool_name: str = "zepto_update_cart",
    tool_args: dict[str, object] | None = None,
) -> Event:
    original = types.FunctionCall(
        id=original_id,
        name=tool_name,
        args=tool_args or {"sku": "milk", "quantity": 1},
    )
    confirmation = types.FunctionCall(
        id=interrupt_id,
        name="adk_request_confirmation",
        args={
            "originalFunctionCall": original.model_dump(
                exclude_none=True,
                by_alias=True,
            ),
            "toolConfirmation": {"confirmed": False},
        },
    )
    return Event(
        author="blacki",
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=confirmation)],
        ),
        long_running_tool_ids={interrupt_id},
    )


def _session_with_events(events: list[Event]) -> Session:
    return Session(
        id="session-v1",
        app_name="blacki",
        user_id="telegram-chat-123",
        events=events,
    )


@pytest.mark.asyncio
async def test_database_confirmation_survives_runtime_restart(
    tmp_path: Path,
) -> None:
    """Persisted confirmation events must be hydrated after a process restart."""
    database_uri = f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )
    first_runtime = AdkRuntime(DatabaseSessionService(database_uri))
    session = await first_runtime.get_or_create_session(locator=locator)
    await first_runtime.session_service.append_event(
        session,
        _confirmation_event(),
    )
    await first_runtime.close()

    second_runtime = AdkRuntime(DatabaseSessionService(database_uri))
    try:
        reloaded = await second_runtime.get_or_create_session(locator=locator)
        pending = _pending_confirmations(reloaded)
        response = _confirmation_response(pending, "Approve")
    finally:
        await second_runtime.close()

    assert len(pending) == 1
    assert pending[0].interrupt_id == "confirm-1"
    assert response is not None
    assert response.parts is not None
    function_response = response.parts[0].function_response
    assert function_response is not None
    assert function_response.id == "confirm-1"
    assert function_response.response == {"confirmed": True}


@pytest.mark.asyncio
async def test_latest_session_disappearing_fails_without_creating_v1() -> None:
    """A list/get race must fail instead of colliding with an existing v1."""

    class VanishingSessionService(InMemorySessionService):
        async def get_session(self, **kwargs: object) -> Session | None:
            del kwargs
            return None

    service = VanishingSessionService()
    runtime = AdkRuntime(service)
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )
    await service.create_session(
        app_name=runtime.app_name,
        user_id=locator.user_id,
        session_id="telegram-chat-123-v1",
    )

    with pytest.raises(RuntimeError, match="disappeared"):
        await runtime.get_or_create_session(locator=locator)


def test_pending_confirmation_ignores_answered_interrupts() -> None:
    request = _confirmation_event()
    response = Event(
        author="user",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="confirm-1",
                        name="adk_request_confirmation",
                        response={"confirmed": False},
                    )
                )
            ],
        ),
    )

    assert len(_pending_confirmations(_session_with_events([request]))) == 1
    assert _pending_confirmations(_session_with_events([request, response])) == []


def test_pending_confirmation_ignores_malformed_calls() -> None:
    malformed_calls = [
        types.FunctionCall(id="other", name="other_tool", args={}),
        types.FunctionCall(id="", name="adk_request_confirmation", args={}),
        types.FunctionCall(
            id="missing-original",
            name="adk_request_confirmation",
            args={"originalFunctionCall": "not-a-dict"},
        ),
        types.FunctionCall(
            id="bad-name",
            name="adk_request_confirmation",
            args={"originalFunctionCall": {"name": 42, "args": {}}},
        ),
        types.FunctionCall(
            id="bad-args",
            name="adk_request_confirmation",
            args={"originalFunctionCall": {"name": "zepto_tool", "args": "bad"}},
        ),
    ]
    event = Event(
        author="blacki",
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=call) for call in malformed_calls],
        ),
    )

    assert _pending_confirmations(_session_with_events([event])) == []


@pytest.mark.parametrize(
    ("message", "confirmed"),
    [
        ("yes", True),
        (" APPROVE ", True),
        ("no", False),
        ("cancel", False),
    ],
)
def test_confirmation_response_accepts_only_unambiguous_answers(
    message: str,
    confirmed: bool,
) -> None:
    pending = _pending_confirmations(_session_with_events([_confirmation_event()]))

    content = _confirmation_response(pending, message)

    assert content is not None
    assert content.parts is not None
    function_response = content.parts[0].function_response
    assert function_response is not None
    assert function_response.id == "confirm-1"
    assert function_response.response == {"confirmed": confirmed}
    assert _confirmation_response(pending, "maybe") is None
    assert _confirmation_response([], "yes") is None


def test_confirmation_prompt_shows_exact_first_call_and_multiple_warning() -> None:
    pending = _pending_confirmations(
        _session_with_events(
            [
                _confirmation_event(),
                _confirmation_event(
                    interrupt_id="confirm-2",
                    original_id="tool-2",
                    tool_name="zepto_place_order",
                    tool_args={"total": 999},
                ),
            ]
        )
    )

    prompt = _format_confirmation(pending[0], pending_count=len(pending))

    assert "zepto_update_cart" in prompt
    assert '"quantity": 1' in prompt
    assert "2 pending calls" in prompt
    assert "zepto_place_order" not in prompt


@pytest.mark.asyncio
async def test_run_user_turn_surfaces_confirmation_instead_of_empty_response() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield _confirmation_event()

    with patch.object(runtime.runner, "run_async", fake_run_async):
        response = await runtime.run_user_turn_with_thoughts(
            locator=locator,
            message_text="Add milk",
        )

    assert "zepto_update_cart" in response.content
    assert "Reply exactly `yes` or `no`" in response.content


@pytest.mark.asyncio
async def test_run_user_turn_ignores_malformed_confirmation_event() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )
    malformed = types.FunctionCall(
        id="confirm-bad",
        name="adk_request_confirmation",
        args={"originalFunctionCall": "bad"},
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="blacki",
            content=types.Content(
                role="model",
                parts=[types.Part(function_call=malformed)],
            ),
        )

    with (
        patch.object(runtime.runner, "run_async", fake_run_async),
        pytest.raises(EmptyModelResponseError) as raised,
    ):
        await runtime.run_user_turn_with_thoughts(
            locator=locator,
            message_text="Hello",
        )

    assert raised.value.retryable is False


@pytest.mark.asyncio
async def test_pending_confirmation_reprompts_or_resumes_same_call() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )
    session = _session_with_events([_confirmation_event()])

    with (
        patch.object(
            runtime,
            "get_or_create_session",
            new=AsyncMock(return_value=session),
        ),
        patch.object(runtime.runner, "run_async", new=MagicMock()) as run_async,
    ):
        response = await runtime.run_user_turn_with_thoughts(
            locator=locator,
            message_text="maybe",
            user_parts=(
                types.Part.from_text(text="maybe"),
                types.Part.from_bytes(data=b"image", mime_type="image/jpeg"),
            ),
        )

    assert "zepto_update_cart" in response.content
    run_async.assert_not_called()

    async def resumed_run(**kwargs: object) -> AsyncIterator[Event]:
        new_message = kwargs["new_message"]
        assert isinstance(new_message, types.Content)
        assert new_message.parts is not None
        function_response = new_message.parts[0].function_response
        assert function_response is not None
        assert function_response.id == "confirm-1"
        assert function_response.response == {"confirmed": True}
        yield Event(
            author="blacki",
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Cart updated")],
            ),
        )

    with (
        patch.object(
            runtime,
            "get_or_create_session",
            new=AsyncMock(return_value=session),
        ),
        patch.object(runtime.runner, "run_async", resumed_run),
    ):
        response = await runtime.run_user_turn_with_thoughts(
            locator=locator,
            message_text="yes",
        )

    assert response.content == "Cart updated"


@pytest.mark.asyncio
async def test_close_still_closes_session_when_runner_close_fails() -> None:
    class ClosableSessionService(InMemorySessionService):
        closed = False

        async def close(self) -> None:
            self.closed = True

    service = ClosableSessionService()
    runtime = AdkRuntime(service)
    with (
        patch.object(
            runtime.runner,
            "close",
            new=AsyncMock(side_effect=RuntimeError("runner close failed")),
        ),
        pytest.raises(RuntimeError, match="runner close failed"),
    ):
        await runtime.close()

    assert service.closed is True


async def test_run_user_turn_streaming_basic() -> None:
    """Basic test for simplified run_user_turn_streaming."""
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
                parts=[types.Part(text="Thinking...", thought=True)],
            ),
        )
        yield Event(
            author="root_agent",
            partial=False,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Thinking...", thought=True),
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

    assert len(chunks) == 3
    assert chunks[0].is_partial is True
    assert chunks[0].thoughts == "Thinking..."
    assert chunks[0].content == ""
    assert chunks[1].is_partial is True
    assert chunks[1].thoughts == "Thinking..."
    assert chunks[1].content == "Final answer."
    assert chunks[2].is_partial is False
    assert chunks[2].thoughts == ""
    assert chunks[2].content == ""


async def test_run_user_turn_streaming_consumes_error_event_before_raising() -> None:
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-stream-error",
        session_id_prefix="telegram-chat-stream-error",
    )
    runner_finished = False

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        yield Event(
            author="root_agent",
            invocation_id="stream-error-invocation",
            error_code=MODEL_RETURNED_NO_CONTENT,
            error_message="empty response",
        )
        yield Event(author="root_agent")
        nonlocal runner_finished
        runner_finished = True

    with (
        patch.object(runtime.runner, "run_async", fake_run_async),
        pytest.raises(EmptyModelResponseError),
    ):
        async for _ in runtime.run_user_turn_streaming(
            locator=locator,
            message_text="Hello",
        ):
            pass

    assert runner_finished is True


async def test_run_user_turn_streaming_empty_content_and_partial_skips() -> None:
    """Test that streaming skips events with no parts and partial without content."""
    runtime = AdkRuntime(InMemorySessionService())
    locator = SessionLocator(
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )

    async def fake_run_async(**kwargs: object) -> AsyncIterator[Event]:
        del kwargs
        # 1. No content
        yield Event(author="root_agent", partial=True, content=None)
        # 2. Content with empty parts
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(role="model", parts=[]),
        )
        # 3. Content with parts but text is empty, and partial is True
        yield Event(
            author="root_agent",
            partial=True,
            content=types.Content(role="model", parts=[types.Part()]),
        )

    chunks: list[StreamChunk] = []
    with patch.object(runtime.runner, "run_async", fake_run_async):
        async for chunk in runtime.run_user_turn_streaming(
            locator=locator, message_text="Hello"
        ):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].is_partial is False
    assert chunks[0].thoughts == ""
    assert chunks[0].content == ""
