"""Memory service that bridges Mem0 to ADK's BaseMemoryService interface."""

from __future__ import annotations

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
            app_name: The application name (used as part of composite user_id).
            user_id: The user identifier.
            query: The search query.

        Returns:
            SearchMemoryResponse with matching MemoryEntry objects.
        """
        from .config import get_search_limit

        mem0_user_id = f"{app_name}/{user_id}"
        limit = get_search_limit()

        try:
            result = self._client.search(query=query, user_id=mem0_user_id, limit=limit)
        except Exception:
            logger.exception("Failed to search memories for user %s", mem0_user_id)
            return SearchMemoryResponse(memories=[])

        memories: list[MemoryEntry] = []
        for m in result.get("results", []):
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
            "Found %d memories for query '%s' (user: %s)",
            len(memories),
            query[:30],
            mem0_user_id,
        )

        return SearchMemoryResponse(memories=memories)
