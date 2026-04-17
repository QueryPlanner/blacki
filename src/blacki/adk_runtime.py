"""Shared ADK runtime helpers for FastAPI and Telegram."""

import inspect
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types

from .utils import ServerEnv

logger = logging.getLogger(__name__)

DEFAULT_EMPTY_RESPONSE = "I apologize, but I couldn't generate a response."
SESSION_VERSION_SEPARATOR = "-v"


@dataclass(slots=True, frozen=True)
class TurnResponse:
    """Structured response from an ADK turn with separated thoughts and content."""

    thoughts: str
    content: str


@dataclass(slots=True, frozen=True)
class StreamChunk:
    """A streaming chunk from ADK with partial or complete thoughts/content."""

    thoughts: str
    content: str
    is_partial: bool = True


def build_session_service_uri(env: ServerEnv) -> str | None:
    """Build the canonical session service URI for shared ADK runtimes."""
    session_uri = env.session_uri
    if session_uri is None:
        return None

    if session_uri.startswith("postgresql://"):
        return session_uri.replace("postgresql://", "postgresql+asyncpg://", 1)

    return session_uri


def build_session_db_kwargs(env: ServerEnv) -> dict[str, Any]:
    """Build shared SQLAlchemy kwargs for database-backed ADK sessions."""
    return {
        "pool_pre_ping": env.db_pool_pre_ping,
        "pool_recycle": env.db_pool_recycle,
        "pool_size": env.db_pool_size,
        "max_overflow": env.db_max_overflow,
        "pool_timeout": env.db_pool_timeout,
    }


def create_session_service(
    session_service_uri: str | None,
    session_db_kwargs: dict[str, Any],
) -> BaseSessionService:
    """Create a session service for programmatic ADK runner usage."""
    if session_service_uri is None:
        logger.warning(
            "No shared session service configured; using in-memory ADK sessions."
        )
        return InMemorySessionService()

    if session_service_uri.startswith("postgresql+asyncpg://"):
        return DatabaseSessionService(session_service_uri, **session_db_kwargs)

    msg = (
        "Shared ADK runtime does not support the configured session URI: "
        f"{session_service_uri}"
    )
    raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class SessionLocator:
    """Stable identifiers used to resolve an ADK session."""

    user_id: str
    session_id_prefix: str


class AdkRuntime:
    """Small helper around ADK Runner and SessionService."""

    def __init__(self, session_service: BaseSessionService) -> None:
        from .agent import app as agent_app

        self.app = agent_app
        self.app_name = agent_app.name
        self.session_service = session_service
        self.runner = Runner(
            app=self.app,
            app_name=self.app_name,
            session_service=self.session_service,
            auto_create_session=False,
        )

    async def get_or_create_session(
        self,
        *,
        locator: SessionLocator,
        state: dict[str, Any] | None = None,
    ) -> Session:
        """Return the latest session for a locator, or create version 1.

        Note: The `state` parameter is only used when creating a new session.
        If a session already exists, the provided state is ignored and the
        existing session's state is preserved.
        """
        existing_session = await self._get_latest_session(locator=locator)
        if existing_session is not None:
            return existing_session

        return await self._create_versioned_session(
            locator=locator,
            version=1,
            state=state,
        )

    async def create_next_session(
        self,
        *,
        locator: SessionLocator,
        state: dict[str, Any] | None = None,
    ) -> Session:
        """Create the next versioned session for a locator."""
        existing_session = await self._get_latest_session(locator=locator)
        next_version = 1

        if existing_session is not None:
            current_version = _extract_session_version(
                session_id=existing_session.id,
                session_id_prefix=locator.session_id_prefix,
            )
            next_version = current_version + 1

        return await self._create_versioned_session(
            locator=locator,
            version=next_version,
            state=state,
        )

    async def run_user_turn(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
    ) -> str:
        """Run one user turn through ADK and return the final assistant text."""
        response = await self.run_user_turn_with_thoughts(
            locator=locator,
            message_text=message_text,
            state=state,
        )
        return response.content or DEFAULT_EMPTY_RESPONSE

    async def run_user_turn_with_thoughts(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
    ) -> TurnResponse:
        """Run one user turn through ADK and return structured response."""
        session = await self.get_or_create_session(locator=locator, state=state)
        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=message_text)],
        )

        thoughts_parts: list[str] = []
        content_parts: list[str] = []
        partial_thoughts = ""
        partial_content = ""

        async for event in self.runner.run_async(
            user_id=locator.user_id,
            session_id=session.id,
            new_message=new_message,
        ):
            self._raise_on_event_error(event)

            event_thoughts, event_content = _extract_turn_parts(event)
            if event_thoughts:
                if event.partial:
                    partial_thoughts = event_thoughts
                else:
                    thoughts_parts.append(event_thoughts)
            if event_content:
                if event.partial:
                    partial_content = event_content
                else:
                    content_parts.append(event_content)

        final_thoughts = "".join(thoughts_parts) or partial_thoughts
        final_content = "".join(content_parts) or partial_content

        return TurnResponse(thoughts=final_thoughts, content=final_content)

    async def run_user_turn_streaming(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield streaming chunks as ADK events arrive.

        The final chunk has is_partial=False, indicating the stream is complete.
        """
        session = await self.get_or_create_session(locator=locator, state=state)
        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=message_text)],
        )

        accumulated_thoughts: list[str] = []
        accumulated_content: list[str] = []
        partial_thoughts = ""
        partial_content = ""

        streaming_config = RunConfig(streaming_mode=StreamingMode.SSE)

        async for event in self.runner.run_async(
            user_id=locator.user_id,
            session_id=session.id,
            new_message=new_message,
            run_config=streaming_config,
        ):
            self._raise_on_event_error(event)

            event_thoughts, event_content = _extract_stream_turn_parts(event)
            if event_thoughts:
                if event.partial:
                    partial_thoughts = _merge_stream_fragment(
                        partial_thoughts,
                        event_thoughts,
                    )
                else:
                    accumulated_thoughts.append(
                        _merge_stream_fragment(partial_thoughts, event_thoughts)
                    )
                    partial_thoughts = ""
            if event_content:
                if event.partial:
                    partial_content = _merge_stream_fragment(
                        partial_content,
                        event_content,
                    )
                else:
                    accumulated_content.append(
                        _merge_stream_fragment(partial_content, event_content)
                    )
                    partial_content = ""

            current_thoughts = "".join(accumulated_thoughts) + partial_thoughts
            current_content = "".join(accumulated_content) + partial_content

            if current_thoughts or current_content:
                yield StreamChunk(
                    thoughts=current_thoughts,
                    content=current_content,
                    is_partial=True,
                )

        final_thoughts = "".join(accumulated_thoughts) or partial_thoughts
        final_content = "".join(accumulated_content) or partial_content

        yield StreamChunk(
            thoughts=final_thoughts,
            content=final_content,
            is_partial=False,
        )

    async def close(self) -> None:
        """Close the underlying session service when supported."""
        close_method = getattr(self.session_service, "close", None)
        if close_method is None:
            return

        close_result = close_method()
        if inspect.isawaitable(close_result):
            await close_result

    async def _get_latest_session(self, *, locator: SessionLocator) -> Session | None:
        response = await self.session_service.list_sessions(
            app_name=self.app_name,
            user_id=locator.user_id,
        )
        version_prefix = f"{locator.session_id_prefix}{SESSION_VERSION_SEPARATOR}"
        matching_sessions = [
            session
            for session in response.sessions
            if _matches_session_prefix(
                session_id=session.id,
                session_id_prefix=locator.session_id_prefix,
            )
            and session.id.removeprefix(version_prefix).isdigit()
        ]
        if not matching_sessions:
            return None

        return max(
            matching_sessions,
            key=lambda session: (
                _extract_session_version(
                    session_id=session.id,
                    session_id_prefix=locator.session_id_prefix,
                ),
                session.last_update_time,
            ),
        )

    async def _create_versioned_session(
        self,
        *,
        locator: SessionLocator,
        version: int,
        state: dict[str, Any] | None = None,
    ) -> Session:
        session_state = _build_session_state(user_id=locator.user_id, state=state)
        session_id = _build_versioned_session_id(
            session_id_prefix=locator.session_id_prefix,
            version=version,
        )
        return await self.session_service.create_session(
            app_name=self.app_name,
            user_id=locator.user_id,
            session_id=session_id,
            state=session_state,
        )

    def _raise_on_event_error(self, event: Event) -> None:
        if not event.error_message:
            return

        error_code = event.error_code or "unknown_error"
        msg = f"ADK runner error ({error_code}): {event.error_message}"
        raise RuntimeError(msg)


def create_adk_runtime(env: ServerEnv) -> AdkRuntime:
    """Create a shared ADK runtime using the current environment config."""
    session_service_uri = build_session_service_uri(env)
    session_db_kwargs = build_session_db_kwargs(env)
    session_service = create_session_service(
        session_service_uri=session_service_uri,
        session_db_kwargs=session_db_kwargs,
    )
    return AdkRuntime(session_service=session_service)


def _build_session_state(
    *,
    user_id: str,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    session_state = dict(state or {})
    session_state.setdefault("user_id", user_id)
    return session_state


def _build_versioned_session_id(*, session_id_prefix: str, version: int) -> str:
    return f"{session_id_prefix}{SESSION_VERSION_SEPARATOR}{version}"


def _matches_session_prefix(*, session_id: str, session_id_prefix: str) -> bool:
    return session_id.startswith(f"{session_id_prefix}{SESSION_VERSION_SEPARATOR}")


def _extract_session_version(*, session_id: str, session_id_prefix: str) -> int:
    version_prefix = f"{session_id_prefix}{SESSION_VERSION_SEPARATOR}"
    version_text = session_id.removeprefix(version_prefix)
    if not version_text.isdigit():
        msg = f"Unexpected session id format: {session_id}"
        raise ValueError(msg)
    return int(version_text)


_PUNCTUATION_START = frozenset(".,!?;:'\"-–—…)]}©®™")


def _join_token(accumulated: str, token: str) -> str:
    """Join a token to accumulated text with smart spacing.

    Adds a space before the token unless it starts with punctuation.
    """
    if not token:
        return accumulated
    if not accumulated:
        return token
    if token[0] in _PUNCTUATION_START:
        return accumulated + token
    return accumulated + " " + token


def _join_text_parts(parts: list[str]) -> str:
    """Join text parts with smart spacing."""
    result = ""
    for part in parts:
        result = _join_token(result, part)
    return result


def _extract_event_text(event: Event) -> str:
    """Extract all text from an event (backward compatibility helper)."""
    thoughts, content = _extract_turn_parts(event)
    return f"{thoughts}\n{content}".strip()


def _extract_turn_parts(event: Event) -> tuple[str, str]:
    """Extract thoughts and content from an event.

    Returns:
        A tuple of (thoughts, content) where thoughts are from parts marked
        with thought=True and content is from all other text parts.
    """
    if event.content is None or not event.content.parts:
        return "", ""

    thoughts: list[str] = []
    content: list[str] = []

    for part in event.content.parts:
        if not part.text:
            continue
        if part.thought:
            thoughts.append(part.text)
        else:
            content.append(part.text)

    return _join_text_parts(thoughts).strip(), _join_text_parts(content).strip()


def _extract_stream_turn_parts(event: Event) -> tuple[str, str]:
    """Extract streaming thoughts/content while preserving exact token spacing."""
    if event.content is None or not event.content.parts:
        return "", ""

    thoughts: list[str] = []
    content: list[str] = []

    for part in event.content.parts:
        if not part.text:
            continue
        if part.thought:
            thoughts.append(part.text)
        else:
            content.append(part.text)

    return "".join(thoughts), "".join(content)


def _merge_stream_fragment(existing_text: str, incoming_text: str) -> str:
    """Merge streaming fragments that may be deltas or full snapshots."""
    if not existing_text:
        return incoming_text

    if not incoming_text:
        return existing_text

    if incoming_text.startswith(existing_text):
        return incoming_text

    if existing_text.startswith(incoming_text):
        return existing_text

    max_overlap = min(len(existing_text), len(incoming_text))
    for overlap_size in range(max_overlap, 0, -1):
        if existing_text.endswith(incoming_text[:overlap_size]):
            return existing_text + incoming_text[overlap_size:]

    return existing_text + incoming_text
