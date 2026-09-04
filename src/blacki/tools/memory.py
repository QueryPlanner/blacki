"""Memory tools for ADK agent using Mem0."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.tools import ToolContext

from blacki.memory.config import (
    get_memory_client,
    get_memory_client_error,
    get_search_limit,
)

logger = logging.getLogger(__name__)

INACCESSIBLE_MEMORY_ERROR = "Memory not found or inaccessible."


def _memory_service_unavailable_response(
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured response when the Mem0 backend is unavailable."""
    response: dict[str, Any] = {
        "status": "error",
        "error": get_memory_client_error() or "Memory service is not configured.",
    }
    if extra_fields:
        response.update(extra_fields)
    return response


def _context_user_id(tool_context: ToolContext) -> str | None:
    """Return the authenticated ADK user ID, rejecting missing identities."""
    user_id = tool_context.user_id
    if not isinstance(user_id, str) or not user_id.strip():
        return None
    return user_id


def _missing_user_response(
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common response for an unauthenticated tool invocation."""
    response: dict[str, Any] = {
        "status": "error",
        "error": "Missing user_id in tool_context.",
    }
    if extra_fields:
        response.update(extra_fields)
    return response


def _inaccessible_memory_response() -> dict[str, str]:
    """Return a non-enumerating response for missing or foreign memory IDs."""
    return {
        "status": "error",
        "error": INACCESSIBLE_MEMORY_ERROR,
    }


def _get_owned_memory(
    client: Any,
    memory_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    """Fetch a memory and fail closed unless its stored owner matches exactly."""
    try:
        result = client.get(memory_id=memory_id)
    except Exception:
        logger.exception("Failed to verify memory ownership")
        return None

    if not isinstance(result, dict) or result.get("user_id") != user_id:
        return None
    return result


async def save_memory(
    text: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Save a memory for a user.

    Use this tool to store important information about the user that should
    persist across conversations, such as preferences, facts, or context.

    Args:
        text: The memory text to save.
        tool_context: ADK tool context.

    Returns:
        Dictionary with status and result message.
    """
    user_id = _context_user_id(tool_context)
    if user_id is None:
        return _missing_user_response()

    client = get_memory_client()
    if client is None:
        return _memory_service_unavailable_response()

    if not text.strip():
        return {
            "status": "error",
            "error": "Memory text must be a non-empty string.",
        }

    try:
        result = client.add(text, user_id=user_id)
        logger.info("Saved memory for user %s", user_id)
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        logger.exception("Failed to save memory")
        return {
            "status": "error",
            "error": f"Failed to save memory: {e}",
        }


async def search_memory(
    query: str,
    tool_context: ToolContext,
    limit: int | None = None,
) -> dict[str, Any]:
    """Search memories semantically for a user.

    Use this tool to retrieve relevant memories based on meaning, not just
    exact keyword matches. Returns memories with their IDs for further operations.

    Args:
        query: The search query string.
        tool_context: ADK tool context.
        limit: Maximum number of results to return. Defaults to MEM0_SEARCH_LIMIT.

    Returns:
        Dictionary with status and list of matching memories
        (each with id, memory, score).
    """
    user_id = _context_user_id(tool_context)
    if user_id is None:
        return _missing_user_response({"results": []})

    client = get_memory_client()
    if client is None:
        return _memory_service_unavailable_response({"results": []})

    if not query.strip():
        return {
            "status": "error",
            "error": "Search query must be a non-empty string.",
            "results": [],
        }

    limit = limit or get_search_limit()

    try:
        result = client.search(query=query, user_id=user_id, limit=limit)
        memories = result.get("results", [])

        formatted_results: list[dict[str, Any]] = []
        for m in memories:
            formatted_results.append(
                {
                    "id": m.get("id", ""),
                    "memory": m.get("memory", ""),
                    "score": m.get("score", 0),
                }
            )

        logger.info(
            "Found %d memories for user %s",
            len(formatted_results),
            user_id,
        )

        return {
            "status": "success",
            "query": query,
            "results": formatted_results,
        }
    except Exception as e:
        logger.exception("Failed to search memories")
        return {
            "status": "error",
            "error": f"Failed to search memories: {e}",
            "results": [],
        }


async def get_all_memories(
    tool_context: ToolContext,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    """List all memories for a user with pagination.

    Use this tool to retrieve all stored memories for a user, not just
    those matching a semantic query. Useful for browsing or auditing.

    Note: Mem0 OSS does not support offset-based pagination. This function
    fetches `page * page_size` records and slices client-side. Deep pagination
    (high page numbers) may be inefficient for users with many memories.

    Args:
        tool_context: ADK tool context.
        page: Page number for pagination (default 1).
        page_size: Number of results per page (default 50).

    Returns:
        Dictionary with status and list of memories (each with id, memory, created_at).
    """
    user_id = _context_user_id(tool_context)
    if user_id is None:
        return _missing_user_response({"results": []})

    client = get_memory_client()
    if client is None:
        return _memory_service_unavailable_response({"results": []})

    if page > 3:
        logger.warning(
            "Deep pagination requested (page %d). Mem0 OSS does not support "
            "offset-based pagination, so this fetches %d records client-side. "
            "Consider using search_memory for targeted queries instead.",
            page,
            page * page_size,
        )

    try:
        result_limit = page * page_size
        result = client.get_all(user_id=user_id, limit=result_limit)

        memories = (
            result.get("results", []) if isinstance(result, dict) else result
        ) or []
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paged_memories = memories[start_index:end_index]

        formatted_results: list[dict[str, Any]] = []
        for m in paged_memories:
            if isinstance(m, dict):
                formatted_results.append(
                    {
                        "id": m.get("id", ""),
                        "memory": m.get("memory", ""),
                        "created_at": m.get("created_at", ""),
                    }
                )

        logger.info(
            "Retrieved %d memories for user %s (page %d)",
            len(formatted_results),
            user_id,
            page,
        )

        return {
            "status": "success",
            "results": formatted_results,
            "page": page,
            "page_size": page_size,
        }
    except Exception as e:
        logger.exception("Failed to get all memories")
        return {
            "status": "error",
            "error": f"Failed to get all memories: {e}",
            "results": [],
        }


async def get_memory(
    memory_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Retrieve a single memory by its ID.

    Use this tool to fetch a specific memory when you know its ID
    (e.g., from a previous search or list operation).

    Args:
        memory_id: The unique identifier of the memory.
        tool_context: ADK tool context.

    Returns:
        Dictionary with status and memory details (id, memory, metadata, etc.).
    """
    user_id = _context_user_id(tool_context)
    if user_id is None:
        return _missing_user_response()

    client = get_memory_client()
    if client is None:
        return _memory_service_unavailable_response()

    if not memory_id.strip():
        return {
            "status": "error",
            "error": "Memory ID must be a non-empty string.",
        }

    try:
        result = _get_owned_memory(client, memory_id, user_id)
        if result is None:
            return _inaccessible_memory_response()

        return {
            "status": "success",
            "memory": {
                "id": result.get("id", memory_id),
                "memory": result.get("memory", ""),
                "created_at": result.get("created_at", ""),
                "updated_at": result.get("updated_at", ""),
                "metadata": result.get("metadata", {}),
            },
        }
    except Exception as e:
        logger.exception("Failed to format an owned memory")
        return {
            "status": "error",
            "error": f"Failed to get memory: {e}",
        }


async def update_memory(
    memory_id: str,
    text: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Update an existing memory by its ID.

    Use this tool to modify a memory's content when a user's information
    has changed. You must know the memory_id (from search or list operations).

    Args:
        memory_id: The unique identifier of the memory to update.
        text: The new memory text.
        tool_context: ADK tool context.

    Returns:
        Dictionary with status and result message.
    """
    user_id = _context_user_id(tool_context)
    if user_id is None:
        return _missing_user_response()

    client = get_memory_client()
    if client is None:
        return _memory_service_unavailable_response()

    if not memory_id.strip():
        return {
            "status": "error",
            "error": "Memory ID must be a non-empty string.",
        }

    if not text.strip():
        return {
            "status": "error",
            "error": "Memory text must be a non-empty string.",
        }

    try:
        if _get_owned_memory(client, memory_id, user_id) is None:
            return _inaccessible_memory_response()

        client.update(memory_id, data=text)

        logger.info("Updated memory %s", memory_id)
        return {
            "status": "success",
            "message": f"Memory {memory_id} updated successfully.",
        }
    except Exception as e:
        logger.exception("Failed to update memory %s", memory_id)
        return {
            "status": "error",
            "error": f"Failed to update memory: {e}",
        }


async def delete_memory(
    memory_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Delete a single memory by its ID.

    Use this tool when a user wants to forget specific information.
    You must know the memory_id (from search or list operations).

    Args:
        memory_id: The unique identifier of the memory to delete.
        tool_context: ADK tool context.

    Returns:
        Dictionary with status and result message.
    """
    user_id = _context_user_id(tool_context)
    if user_id is None:
        return _missing_user_response()

    client = get_memory_client()
    if client is None:
        return _memory_service_unavailable_response()

    if not memory_id.strip():
        return {
            "status": "error",
            "error": "Memory ID must be a non-empty string.",
        }

    try:
        if _get_owned_memory(client, memory_id, user_id) is None:
            return _inaccessible_memory_response()

        client.delete(memory_id=memory_id)
        logger.info("Deleted memory %s", memory_id)
        return {
            "status": "success",
            "message": f"Memory {memory_id} deleted successfully.",
        }
    except Exception as e:
        logger.exception("Failed to delete memory %s", memory_id)
        return {
            "status": "error",
            "error": f"Failed to delete memory: {e}",
        }
