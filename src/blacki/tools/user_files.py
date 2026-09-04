"""Agent-callable, Telegram-sender-scoped durable file tools."""

from __future__ import annotations

from typing import Any

from google.adk.tools import FunctionTool, ToolContext

from blacki.sandbox.manager import get_sandbox_manager
from blacki.user_files.config import SENDER_STATE_KEY
from blacki.user_files.service import get_user_file_service, sanitize_display_name


def _sender_id(tool_context: ToolContext) -> str:
    value = tool_context.state.get(SENDER_STATE_KEY)
    if value is None or not str(value).strip():
        raise ValueError("Durable files require an authenticated Telegram sender")
    return str(value).strip()


async def list_user_files(
    query: str, limit: int, tool_context: ToolContext
) -> dict[str, Any]:
    """List this Telegram sender's stored files, optionally matching a filename.

    Args:
        query: Case-insensitive filename fragment, or an empty string for recent files.
        limit: Number of results to return, from 1 through 50.

    Returns:
        A status and safe metadata for matching files.
    """
    sender_id = _sender_id(tool_context)
    files = await get_user_file_service().list_files(sender_id, query, limit)
    return {
        "status": "success",
        "files": [
            {
                "object_id": item.object_id,
                "filename": item.display_name,
                "kind": item.media_kind,
                "mime_type": item.mime_type,
                "size_bytes": item.size_bytes,
                "uploaded_at": item.uploaded_at,
                "expires_at": item.expires_at,
            }
            for item in files
        ],
    }


async def restore_user_file(
    object_id: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Restore one stored file by opaque object ID into the current sandbox.

    Args:
        object_id: Opaque ID returned by the file list or recent-file context.

    Returns:
        A status, verified sandbox path, and safe file metadata.
    """
    sender_id = _sender_id(tool_context)
    try:
        item, data = await get_user_file_service().restore(sender_id, object_id)
    except FileNotFoundError:
        return {
            "status": "not_found",
            "message": "No available file matches that object ID.",
        }
    manager = get_sandbox_manager()
    result = await manager.get_or_create_sandbox(tool_context.state)
    sandbox = result.get("sandbox")
    if sandbox is None:
        raise RuntimeError(str(result.get("error") or "Sandbox is unavailable"))
    safe_name = sanitize_display_name(item.display_name)
    path = f"/workspace/uploads/{item.object_id}-{safe_name}"
    await sandbox.files.write_file(path, data)
    return {
        "status": "success",
        "object_id": item.object_id,
        "filename": item.display_name,
        "sandbox_path": path,
        "size_bytes": item.size_bytes,
    }


async def delete_user_file(object_id: str, tool_context: ToolContext) -> dict[str, Any]:
    """Permanently delete one stored file after explicit user confirmation.

    Args:
        object_id: Opaque ID returned by the file list or recent-file context.

    Returns:
        Whether the owner-scoped object was deleted.
    """
    sender_id = _sender_id(tool_context)
    deleted = await get_user_file_service().delete(sender_id, object_id)
    return {"status": "success", "deleted": deleted, "object_id": object_id}


def create_user_file_tools() -> list[Any]:
    """Return Telegram-root-only file tools with deletion confirmation."""
    return [
        list_user_files,
        restore_user_file,
        FunctionTool(delete_user_file, require_confirmation=True),
    ]
