"""Tests for Mem0MemoryService."""

from unittest.mock import MagicMock

import pytest
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry

from blacki.memory.config import reset_memory_client
from blacki.memory.mem0_memory_service import Mem0MemoryService


class TestMem0MemoryService:
    """Tests for Mem0MemoryService class."""

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    def test_init(self) -> None:
        """Should initialize with Mem0 client."""
        mock_client = MagicMock()
        service = Mem0MemoryService(mock_client)
        assert service._client is mock_client

    @pytest.mark.asyncio
    async def test_search_memory_success(self) -> None:
        """Should search memories and convert to MemoryEntry objects."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"id": "mem_1", "memory": "User likes pizza", "score": 0.95},
                {"id": "mem_2", "memory": "User prefers tea", "score": 0.85},
            ]
        }

        service = Mem0MemoryService(mock_client)
        response = await service.search_memory(
            app_name="test_app", user_id="test_user", query="food preferences"
        )

        assert isinstance(response, SearchMemoryResponse)
        assert len(response.memories) == 2
        assert all(isinstance(m, MemoryEntry) for m in response.memories)
        assert response.memories[0].content.parts is not None
        assert response.memories[1].content.parts is not None
        assert response.memories[0].content.parts[0].text == "User likes pizza"
        assert response.memories[1].content.parts[0].text == "User prefers tea"

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["query"] == "food preferences"
        assert call_kwargs["user_id"] == "test_app/test_user"

    @pytest.mark.asyncio
    async def test_search_memory_empty_results(self) -> None:
        """Should return empty list when no memories found."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        service = Mem0MemoryService(mock_client)
        response = await service.search_memory(
            app_name="test_app", user_id="test_user", query="nonexistent"
        )

        assert isinstance(response, SearchMemoryResponse)
        assert len(response.memories) == 0

    @pytest.mark.asyncio
    async def test_search_memory_skips_empty_text(self) -> None:
        """Should skip results with empty memory text."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"id": "mem_1", "memory": "Valid memory", "score": 0.95},
                {"id": "mem_2", "memory": "", "score": 0.85},
                {"id": "mem_3", "memory": "Another valid", "score": 0.75},
            ]
        }

        service = Mem0MemoryService(mock_client)
        response = await service.search_memory(
            app_name="test_app", user_id="test_user", query="test"
        )

        assert len(response.memories) == 2
        assert response.memories[0].content.parts is not None
        assert response.memories[1].content.parts is not None
        assert response.memories[0].content.parts[0].text == "Valid memory"
        assert response.memories[1].content.parts[0].text == "Another valid"

    @pytest.mark.asyncio
    async def test_search_memory_handles_exception(self) -> None:
        """Should return empty list on search failure."""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Connection failed")

        service = Mem0MemoryService(mock_client)
        response = await service.search_memory(
            app_name="test_app", user_id="test_user", query="test"
        )

        assert isinstance(response, SearchMemoryResponse)
        assert len(response.memories) == 0

    @pytest.mark.asyncio
    async def test_add_session_to_memory_noop(self) -> None:
        """Should do nothing for add_session_to_memory."""
        from google.adk.sessions.session import Session

        mock_client = MagicMock()
        mock_session = MagicMock(spec=Session)
        service = Mem0MemoryService(mock_client)

        await service.add_session_to_memory(mock_session)

        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_events_to_memory_noop(self) -> None:
        """Should do nothing for add_events_to_memory."""
        mock_client = MagicMock()
        service = Mem0MemoryService(mock_client)

        await service.add_events_to_memory(
            app_name="test_app",
            user_id="test_user",
            events=[],
        )

        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_memory_uses_custom_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use configured search limit."""
        monkeypatch.setenv("MEM0_SEARCH_LIMIT", "10")

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        service = Mem0MemoryService(mock_client)
        await service.search_memory(
            app_name="test_app", user_id="test_user", query="test"
        )

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["limit"] == 10
