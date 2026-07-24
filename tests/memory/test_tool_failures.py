"""Failure-path coverage for the Mem0-backed memory tools."""

from unittest.mock import MagicMock, patch

import pytest
from google.adk.tools import ToolContext

from blacki.memory import tools


@pytest.fixture
def tool_context() -> ToolContext:
    """Return an authenticated ADK tool context."""
    context = MagicMock(spec=ToolContext)
    context.user_id = "test_user"
    return context


@pytest.mark.asyncio
async def test_unavailable_client_returns_safe_responses(
    tool_context: ToolContext,
) -> None:
    """Every tool should fail safely when Mem0 is unavailable."""
    with (
        patch.object(tools, "get_memory_client", return_value=None),
        patch.object(
            tools,
            "get_memory_client_error",
            return_value="backend unavailable",
        ),
    ):
        search_result = await tools.search_memory("query", tool_context)
        list_result = await tools.get_all_memories(tool_context)
        get_result = await tools.get_memory("memory-1", tool_context)
        update_result = await tools.update_memory(
            "memory-1",
            "updated",
            tool_context,
        )
        delete_result = await tools.delete_memory("memory-1", tool_context)

    assert search_result == {
        "status": "error",
        "error": "backend unavailable",
        "results": [],
    }
    assert list_result["results"] == []
    assert get_result["status"] == "error"
    assert update_result["status"] == "error"
    assert delete_result["status"] == "error"


@pytest.mark.asyncio
async def test_blank_identity_is_rejected() -> None:
    """Whitespace-only identities should be treated as missing."""
    context = MagicMock(spec=ToolContext)
    context.user_id = "  "

    result = await tools.save_memory("private", context)

    assert result == {
        "status": "error",
        "error": "Missing user_id in tool_context.",
    }


@pytest.mark.asyncio
async def test_external_memory_errors_are_reported(
    tool_context: ToolContext,
) -> None:
    """Mem0 operation failures should return structured error responses."""
    client = MagicMock()
    client.add.side_effect = RuntimeError("save failed")
    client.search.side_effect = RuntimeError("search failed")
    client.get_all.side_effect = RuntimeError("list failed")

    with patch.object(tools, "get_memory_client", return_value=client):
        save_result = await tools.save_memory("text", tool_context)
        search_result = await tools.search_memory("query", tool_context)
        list_result = await tools.get_all_memories(tool_context)

    assert save_result == {
        "status": "error",
        "error": "Failed to save memory: save failed",
    }
    assert search_result == {
        "status": "error",
        "error": "Failed to search memories: search failed",
        "results": [],
    }
    assert list_result == {
        "status": "error",
        "error": "Failed to get all memories: list failed",
        "results": [],
    }


@pytest.mark.asyncio
async def test_get_all_handles_direct_lists_and_late_pages(
    tool_context: ToolContext,
) -> None:
    """Direct Mem0 lists should tolerate malformed items while paginating."""
    client = MagicMock()
    client.get_all.return_value = [
        {"id": "one"},
        {"id": "two"},
        {"id": "three"},
        "malformed",
    ]

    with patch.object(tools, "get_memory_client", return_value=client):
        result = await tools.get_all_memories(
            tool_context,
            page=4,
            page_size=1,
        )

    assert result["page"] == 4
    assert result["results"] == []
    assert result["page_size"] == 1


@pytest.mark.asyncio
async def test_get_rejects_blank_id_and_handles_operation_errors(
    tool_context: ToolContext,
) -> None:
    """Get should validate IDs and safely handle Mem0 failures."""
    client = MagicMock()
    client.get.side_effect = RuntimeError("lookup failed")

    with patch.object(tools, "get_memory_client", return_value=client):
        blank_result = await tools.get_memory(" ", tool_context)
        error_result = await tools.get_memory("memory-1", tool_context)

    assert blank_result == {
        "status": "error",
        "error": "Memory ID must be a non-empty string.",
    }
    assert error_result == {
        "status": "error",
        "error": tools.INACCESSIBLE_MEMORY_ERROR,
    }


@pytest.mark.asyncio
async def test_get_handles_malformed_owned_record(
    tool_context: ToolContext,
) -> None:
    """A malformed record should not escape the tool's error handling."""

    class MalformedOwnedRecord(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            if key == "user_id":
                return "test_user"
            raise RuntimeError("malformed record")

    client = MagicMock()
    client.get.return_value = MalformedOwnedRecord()

    with patch.object(tools, "get_memory_client", return_value=client):
        result = await tools.get_memory("memory-1", tool_context)

    assert result == {
        "status": "error",
        "error": "Failed to get memory: malformed record",
    }


@pytest.mark.asyncio
async def test_update_validates_inputs_and_handles_operation_errors(
    tool_context: ToolContext,
) -> None:
    """Update should reject empty input and report Mem0 failures."""
    client = MagicMock()
    client.get.return_value = {"id": "memory-1", "user_id": "test_user"}
    client.update.side_effect = RuntimeError("update failed")

    with patch.object(tools, "get_memory_client", return_value=client):
        blank_id_result = await tools.update_memory(
            " ",
            "updated",
            tool_context,
        )
        blank_text_result = await tools.update_memory(
            "memory-1",
            " ",
            tool_context,
        )
        error_result = await tools.update_memory(
            "memory-1",
            "updated",
            tool_context,
        )

    assert blank_id_result["error"] == "Memory ID must be a non-empty string."
    assert blank_text_result["error"] == "Memory text must be a non-empty string."
    assert error_result == {
        "status": "error",
        "error": "Failed to update memory: update failed",
    }


@pytest.mark.asyncio
async def test_delete_handles_operation_errors(
    tool_context: ToolContext,
) -> None:
    """Delete should report Mem0 failures without claiming success."""
    client = MagicMock()
    client.get.return_value = {"id": "memory-1", "user_id": "test_user"}
    client.delete.side_effect = RuntimeError("delete failed")

    with patch.object(tools, "get_memory_client", return_value=client):
        result = await tools.delete_memory("memory-1", tool_context)

    assert result == {
        "status": "error",
        "error": "Failed to delete memory: delete failed",
    }
