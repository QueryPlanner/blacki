"""Tests for Memory tools."""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from conftest import MockState, MockToolContext
from google.adk.tools import ToolContext

from blacki.memory.config import reset_memory_client
from blacki.memory.tools import (
    delete_all_memories,
    delete_memory,
    get_all_memories,
    get_memory,
    save_memory,
    search_memory,
    update_memory,
)


class TestSaveMemory:
    """Tests for save_memory function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    @pytest.mark.asyncio
    async def test_save_memory_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should save memory successfully."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.add.return_value = {"id": "mem_123", "event": "ADD"}

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await save_memory("I love pizza", tool_context)

        assert result["status"] == "success"
        mock_client.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_memory_empty_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return error for empty text."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await save_memory("   ", tool_context)

        assert result["status"] == "error"
        assert "non-empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_save_memory_no_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return error when memory client is not configured."""
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        tool_context = self._tool_context()

        with patch("blacki.memory.tools.get_memory_client", return_value=None):
            result = await save_memory("test memory", tool_context)

        assert result["status"] == "error"
        assert "not configured" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_save_memory_custom_user_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use provided user_id."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.add.return_value = {"id": "mem_123"}

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await save_memory("test", tool_context, user_id="custom_user")

        assert result["status"] == "success"
        mock_client.add.assert_called_once()
        call_kwargs = mock_client.add.call_args
        assert call_kwargs[1]["user_id"] == "custom_user"


class TestSearchMemory:
    """Tests for search_memory function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    @pytest.mark.asyncio
    async def test_search_memory_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should search and return memories."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "id": "mem_123",
                    "memory": "User likes pizza",
                    "score": 0.95,
                }
            ]
        }

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await search_memory("food preferences", tool_context)

        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "mem_123"
        assert result["results"][0]["memory"] == "User likes pizza"

    @pytest.mark.asyncio
    async def test_search_memory_empty_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return error for empty query."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await search_memory("   ", tool_context)

        assert result["status"] == "error"
        assert "non-empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_memory_no_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return empty results list when no matches."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await search_memory("nonexistent", tool_context)

        assert result["status"] == "success"
        assert result["results"] == []


class TestGetAllMemories:
    """Tests for get_all_memories function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    @pytest.mark.asyncio
    async def test_get_all_memories_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should list all memories with pagination."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.get_all.return_value = {
            "results": [
                {"id": "mem_1", "memory": "Memory 1", "created_at": "2024-01-01"},
                {"id": "mem_2", "memory": "Memory 2", "created_at": "2024-01-02"},
            ]
        }

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await get_all_memories(tool_context, page=1, page_size=10)

        assert result["status"] == "success"
        assert len(result["results"]) == 2
        assert result["page"] == 1


class TestGetMemory:
    """Tests for get_memory function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    @pytest.mark.asyncio
    async def test_get_memory_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should retrieve a single memory by ID."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.get.return_value = {
            "id": "mem_123",
            "memory": "User likes pizza",
            "created_at": "2024-01-01",
            "metadata": {"source": "chat"},
        }

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await get_memory("mem_123", tool_context)

        assert result["status"] == "success"
        assert result["memory"]["id"] == "mem_123"
        assert result["memory"]["memory"] == "User likes pizza"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return error when memory not found."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.get.return_value = None

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await get_memory("nonexistent", tool_context)

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()


class TestUpdateMemory:
    """Tests for update_memory function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    @pytest.mark.asyncio
    async def test_update_memory_cloud_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should update memory using cloud client (options param)."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()

        with (
            patch("blacki.memory.tools.get_memory_client", return_value=mock_client),
            patch("blacki.memory.tools.is_cloud_client", return_value=True),
        ):
            result = await update_memory(
                "mem_123", "Updated text", tool_context, metadata={"key": "value"}
            )

        assert result["status"] == "success"
        mock_client.update.assert_called_once()
        call_args = mock_client.update.call_args
        assert call_args[0][0] == "mem_123"
        assert call_args[1]["options"]["text"] == "Updated text"

    @pytest.mark.asyncio
    async def test_update_memory_oss_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should update memory using OSS client (data param)."""
        monkeypatch.setenv("MEM0_API_KEY", "")
        tool_context = self._tool_context()

        mock_client = MagicMock()

        with (
            patch("blacki.memory.tools.get_memory_client", return_value=mock_client),
            patch("blacki.memory.tools.is_cloud_client", return_value=False),
        ):
            result = await update_memory("mem_123", "Updated text", tool_context)

        assert result["status"] == "success"
        mock_client.update.assert_called_once()
        call_args = mock_client.update.call_args
        assert call_args[0][0] == "mem_123"
        assert call_args[1]["data"] == "Updated text"


class TestDeleteMemory:
    """Tests for delete_memory function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should delete memory by ID."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await delete_memory("mem_123", tool_context)

        assert result["status"] == "success"
        mock_client.delete.assert_called_once_with(memory_id="mem_123")

    @pytest.mark.asyncio
    async def test_delete_memory_empty_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return error for empty memory ID."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await delete_memory("   ", tool_context)

        assert result["status"] == "error"
        assert "non-empty" in result["error"].lower()


class TestDeleteAllMemories:
    """Tests for delete_all_memories function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    @pytest.mark.asyncio
    async def test_delete_all_memories_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should delete all memories for a user."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_client = MagicMock()

        with patch("blacki.memory.tools.get_memory_client", return_value=mock_client):
            result = await delete_all_memories(tool_context, user_id="test_user")

        assert result["status"] == "success"
        mock_client.delete_all.assert_called_once_with(user_id="test_user")
