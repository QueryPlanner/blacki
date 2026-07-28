"""Memory service that bridges Mem0 to ADK's BaseMemoryService interface."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from google.adk.events.event import Event
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types

if TYPE_CHECKING:
    from mem0 import Memory

logger = logging.getLogger(__name__)


class Mem0MemoryService(BaseMemoryService):
    """Memory service backed by Mem0 OSS.

    Wraps the existing Mem0 client to provide ADK-compatible memory operations.
    Memories are managed manually via save_memory tool (no automatic session ingestion).
    """

    def __init__(self, client: Memory):
        self._client = client

    async def add_session_to_memory(self, session: Session) -> None:
        """Not used - user chose manual memory management via save_memory tool."""
        pass

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Not used - user chose manual memory management."""
        pass

    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Search memories via Mem0 and convert to ADK format.

        Args:
            app_name: The application name (unused, user_id is passed directly).
            user_id: The user identifier.
            query: The search query.

        Returns:
            SearchMemoryResponse with matching MemoryEntry objects.
        """
        from .config import get_search_limit

        limit = get_search_limit()

        try:
            result = await asyncio.to_thread(
                self._client.search, query=query, user_id=user_id, limit=limit
            )

            raw_results = (
                result.get("results", []) if isinstance(result, dict) else result
            ) or []

            memories: list[MemoryEntry] = []
            for m in raw_results:
                if not isinstance(m, dict):
                    continue
                memory_text = m.get("memory", "")
                if not memory_text:
                    continue

                memories.append(
                    MemoryEntry(
                        content=types.Content(
                            role="user",
                            parts=[types.Part(text=memory_text)],
                        ),
                        id=m.get("id"),
                    )
                )

            logger.debug(
                "Found %d memories for user %s",
                len(memories),
                user_id,
            )
            return SearchMemoryResponse(memories=memories)

        except Exception:
            logger.exception("Failed to search memories for user %s", user_id)
            return SearchMemoryResponse(memories=[])
