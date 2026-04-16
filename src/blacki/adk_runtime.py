"""Shared ADK runtime helpers for FastAPI and Telegram."""

import inspect
import logging
from dataclasses import dataclass
from typing import Any

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
            "No shared session service configured; Telegram will use in-memory "
            "ADK sessions."
        )
        return InMemorySessionService()

    if session_service_uri.startswith("postgresql+asyncpg://"):
        return DatabaseSessionService(session_service_uri, **session_db_kwargs)

    msg = (
        "Telegram ADK runtime does not support the configured session URI: "
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
        """Return the latest session for a locator, or create version 1."""
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
        session = await self.get_or_create_session(locator=locator, state=state)
        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=message_text)],
        )

        final_response = ""
        partial_response = ""

        async for event in self.runner.run_async(
            user_id=locator.user_id,
            session_id=session.id,
            new_message=new_message,
        ):
            self._raise_on_event_error(event)

            event_text = _extract_event_text(event)
            if not event_text:
                continue

            if event.partial:
                partial_response = event_text
                continue

            final_response = event_text

        return final_response or partial_response or DEFAULT_EMPTY_RESPONSE

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
        matching_sessions = [
            session
            for session in response.sessions
            if _matches_session_prefix(
                session_id=session.id,
                session_id_prefix=locator.session_id_prefix,
            )
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


def _extract_event_text(event: Event) -> str:
    if event.content is None or not event.content.parts:
        return ""

    text_parts = []
    for part in event.content.parts:
        if not part.text:
            continue

        stripped_text = part.text.strip()
        if stripped_text:
            text_parts.append(stripped_text)

    return "\n".join(text_parts)
