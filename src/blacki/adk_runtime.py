"""Shared ADK runtime helpers for FastAPI and Telegram."""

import inspect
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.cli.service_registry import get_service_registry
from google.adk.events import Event
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.runners import Runner
from google.adk.sessions import Session
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types

from .utils.config import ServerEnv

logger = logging.getLogger(__name__)


def _create_mem0_memory_service(uri: str, **kwargs: Any) -> BaseMemoryService:
    """Factory for mem0:// URI scheme.

    Returns Mem0MemoryService if client is available, InMemoryMemoryService otherwise.
    """
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

    from blacki.memory.config import get_memory_client

    client = get_memory_client()
    if client is None:
        logger.info("Mem0 client not available, using in-memory memory service")
        return InMemoryMemoryService()

    from blacki.memory.mem0_memory_service import Mem0MemoryService

    logger.info("Mem0 memory service initialized")
    return Mem0MemoryService(client)


get_service_registry().register_memory_service("mem0", _create_mem0_memory_service)

DEFAULT_EMPTY_RESPONSE = "I apologize, but I couldn't generate a response."
SESSION_VERSION_SEPARATOR = "-v"
_CONFIRMATION_FUNCTION = "adk_request_confirmation"
_CONFIRMATION_APPROVE = frozenset({"y", "yes", "confirm", "approve"})
_CONFIRMATION_REJECT = frozenset({"n", "no", "reject", "cancel"})


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


@dataclass(slots=True, frozen=True)
class PendingConfirmation:
    """One unresolved ADK tool-confirmation interrupt."""

    interrupt_id: str
    tool_name: str
    tool_args: dict[str, Any]


def _pending_confirmations(session: Session) -> list[PendingConfirmation]:
    """Return confirmation calls that do not yet have a matching response."""
    answered_ids = {
        response.id
        for event in session.events
        for response in event.get_function_responses()
        if response.id
    }
    pending: list[PendingConfirmation] = []
    for event in session.events:
        for function_call in event.get_function_calls():
            if (
                function_call.name != _CONFIRMATION_FUNCTION
                or not function_call.id
                or function_call.id in answered_ids
            ):
                continue
            args = function_call.args or {}
            original = args.get("originalFunctionCall")
            if not isinstance(original, dict):
                continue
            tool_name = original.get("name")
            tool_args = original.get("args", {})
            if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
                continue
            pending.append(
                PendingConfirmation(
                    interrupt_id=function_call.id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
            )
    return pending


def _format_confirmation(
    confirmation: PendingConfirmation, *, pending_count: int
) -> str:
    args = json.dumps(
        confirmation.tool_args,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    suffix = ""
    if pending_count > 1:
        suffix = (
            f"\n\nThere are {pending_count} pending calls. This reply applies "
            "only to the first; each call must be confirmed separately."
        )
    return (
        f"Confirm Zepto tool `{confirmation.tool_name}` with these arguments?\n\n"
        f"```json\n{args}\n```\n\nReply exactly `yes` or `no`."
        f"{suffix}"
    )


def _confirmation_response(
    pending: list[PendingConfirmation], message_text: str
) -> types.Content | None:
    if not pending:
        return None
    normalized = message_text.strip().lower()
    if normalized in _CONFIRMATION_APPROVE:
        confirmed = True
    elif normalized in _CONFIRMATION_REJECT:
        confirmed = False
    else:
        return None
    first = pending[0]
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=first.interrupt_id,
                    name=_CONFIRMATION_FUNCTION,
                    response={"confirmed": confirmed},
                )
            )
        ],
    )


def build_session_service_uri(env: ServerEnv) -> str | None:
    """Build the canonical session service URI for shared ADK runtimes."""
    session_uri = env.session_uri
    if session_uri is None:
        return None

    if session_uri.startswith("postgresql://"):
        return session_uri.replace("postgresql://", "postgresql+asyncpg://", 1)

    return session_uri


def build_session_db_kwargs(env: ServerEnv) -> dict[str, Any]:
    """Build shared SQLAlchemy kwargs for database-backed ADK sessions.

    Note: Pool settings are only relevant for PostgreSQL. SQLite uses
    a single connection and ignores pool settings.
    """
    session_uri = build_session_service_uri(env)
    is_sqlite = session_uri is None or session_uri.startswith("sqlite")

    if is_sqlite:
        return {"connect_args": {"timeout": 15}}
    return {}


def create_session_service(
    session_service_uri: str | None,
    session_db_kwargs: dict[str, Any],
    agent_dir: str = "src",
) -> BaseSessionService:
    """Create a session service for programmatic ADK runner usage."""
    if session_service_uri is None:
        default_db_path = Path(agent_dir).resolve() / ".adk" / "sessions.db"
        # Create the directory if it doesn't exist to prevent sqlite errors
        default_db_path.parent.mkdir(parents=True, exist_ok=True)
        session_service_uri = f"sqlite+aiosqlite:///{default_db_path}"
        logger.info(
            f"No shared session service configured; using SQLite at {default_db_path}."
        )

    if session_service_uri.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
        service = DatabaseSessionService(session_service_uri, **session_db_kwargs)

        if session_service_uri.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(service.db_engine.sync_engine, "connect")
            def set_sqlite_pragma(
                dbapi_connection: Any, connection_record: Any
            ) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

        return service

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

    def __init__(
        self,
        session_service: BaseSessionService,
        memory_service: BaseMemoryService | None = None,
        *,
        agent_app: App | None = None,
    ) -> None:
        if agent_app is None:
            from .agent import app as default_agent_app

            agent_app = default_agent_app

        self.app = agent_app
        self.app_name = agent_app.name
        self.session_service = session_service
        self.runner = Runner(
            app=self.app,
            app_name=self.app_name,
            session_service=self.session_service,
            memory_service=memory_service,
            auto_create_session=False,
        )

    async def get_or_create_session(
        self,
        *,
        locator: SessionLocator,
        state: dict[str, Any] | None = None,
    ) -> Session:
        """Return the latest session for a locator, or create version 1.

        When an existing session is found, any new keys from the ``state``
        parameter are merged in (existing values take precedence). This
        ensures callbacks relying on session state (e.g. Telegram tool
        notifications) receive the expected keys even when their transport
        was not the one that originally created the session.
        """
        existing_session = await self._get_latest_session(locator=locator)
        if existing_session is not None:
            if state:
                for key, value in state.items():
                    existing_session.state[key] = value

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
        pending_before_turn = _pending_confirmations(session)
        new_message = _confirmation_response(pending_before_turn, message_text)
        if pending_before_turn and new_message is None:
            return TurnResponse(
                thoughts="",
                content=_format_confirmation(
                    pending_before_turn[0],
                    pending_count=len(pending_before_turn),
                ),
            )
        if new_message is None:
            new_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=message_text)],
            )

        thoughts_parts: list[str] = []
        content_parts: list[str] = []
        partial_thoughts = ""
        partial_content = ""
        pending_after_turn: list[PendingConfirmation] = []

        async for event in self.runner.run_async(
            user_id=locator.user_id,
            session_id=session.id,
            new_message=new_message,
            state_delta=state,
        ):
            self._raise_on_event_error(event)
            for function_call in event.get_function_calls():
                if function_call.name != _CONFIRMATION_FUNCTION:
                    continue
                args = function_call.args or {}
                original = args.get("originalFunctionCall")
                if (
                    not function_call.id
                    or not isinstance(original, dict)
                    or not isinstance(original.get("name"), str)
                    or not isinstance(original.get("args", {}), dict)
                ):
                    continue
                pending_after_turn.append(
                    PendingConfirmation(
                        interrupt_id=function_call.id,
                        tool_name=original["name"],
                        tool_args=original.get("args", {}),
                    )
                )

            has_function_call = (
                event.content is not None
                and event.content.parts
                and any(
                    getattr(p, "function_call", None) is not None
                    for p in event.content.parts
                )
            )

            if event.content and event.content.parts:
                event_thoughts = " ".join(
                    p.text
                    for p in event.content.parts
                    if getattr(p, "thought", False) and p.text
                )
                event_content = " ".join(
                    p.text
                    for p in event.content.parts
                    if not getattr(p, "thought", False) and p.text
                )

                if event_thoughts:
                    if event.partial:
                        partial_thoughts = event_thoughts
                    else:
                        thoughts_parts.append(event_thoughts)
                if event_content and not has_function_call:
                    if event.partial:
                        partial_content = event_content
                    else:
                        content_parts.append(event_content)

        final_thoughts = " ".join(thoughts_parts).strip() or partial_thoughts
        final_content = " ".join(content_parts).strip() or partial_content
        if pending_after_turn:
            final_content = _format_confirmation(
                pending_after_turn[0],
                pending_count=len(pending_after_turn),
            )

        return TurnResponse(
            thoughts=final_thoughts,
            content=final_content,
        )

    async def run_user_turn_streaming(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield streaming chunks as ADK events arrive.

        The final chunk has is_partial=False, indicating the stream is complete.

        Note: This method is implemented and tested but not yet integrated into
        the Telegram bot. It is available for future streaming support.
        """
        session = await self.get_or_create_session(locator=locator, state=state)
        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=message_text)],
        )

        streaming_config = RunConfig(streaming_mode=StreamingMode.SSE)

        async for event in self.runner.run_async(
            user_id=locator.user_id,
            session_id=session.id,
            new_message=new_message,
            run_config=streaming_config,
            state_delta=state,
        ):
            self._raise_on_event_error(event)

            if event.content and event.content.parts:
                event_thoughts = " ".join(
                    p.text
                    for p in event.content.parts
                    if getattr(p, "thought", False) and p.text
                )
                event_content = " ".join(
                    p.text
                    for p in event.content.parts
                    if not getattr(p, "thought", False) and p.text
                )

                if event_thoughts or event_content:
                    yield StreamChunk(
                        thoughts=event_thoughts,
                        content=event_content,
                        is_partial=True,
                    )

        yield StreamChunk(
            thoughts="",
            content="",
            is_partial=False,
        )

    async def close(self) -> None:
        """Close runner toolsets, then the underlying session service."""
        try:
            await self.runner.close()
        finally:
            close_method = getattr(self.session_service, "close", None)
            if close_method is not None:
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

        latest_session = max(
            matching_sessions,
            key=lambda session: (
                _extract_session_version(
                    session_id=session.id,
                    session_id_prefix=locator.session_id_prefix,
                ),
                session.last_update_time,
            ),
        )
        hydrated_session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=locator.user_id,
            session_id=latest_session.id,
        )
        if hydrated_session is None:
            msg = (
                "Latest ADK session disappeared while loading event history: "
                f"{latest_session.id}"
            )
            raise RuntimeError(msg)
        return hydrated_session

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


def create_adk_runtime(
    env: ServerEnv,
    *,
    agent_app: App | None = None,
) -> AdkRuntime:
    """Create a shared ADK runtime using the current environment config."""
    session_service_uri = build_session_service_uri(env)
    session_db_kwargs = build_session_db_kwargs(env)
    session_service = create_session_service(
        session_service_uri=session_service_uri,
        session_db_kwargs=session_db_kwargs,
        agent_dir=env.agent_dir,
    )
    memory_service = get_service_registry().create_memory_service(
        "mem0://", agents_dir=str(Path(env.agent_dir).resolve())
    )
    return AdkRuntime(
        session_service=session_service,
        memory_service=memory_service,
        agent_app=agent_app,
    )


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
