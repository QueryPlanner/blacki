"""ADK tools for OpenSandbox operations."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from google.adk.tools import ToolContext
from opensandbox.exceptions import SandboxException
from opensandbox.models.execd import RunCommandOpts
from opensandbox.models.filesystem import SearchEntry

from .manager import get_sandbox_manager

logger = logging.getLogger(__name__)


def _format_command_output(execution: Any) -> str:
    """Format command execution output for tool response."""
    stdout = "\n".join(msg.text for msg in execution.logs.stdout)
    stderr = "\n".join(msg.text for msg in execution.logs.stderr)

    output = stdout.strip()
    if stderr:
        output = f"{output}\n[stderr]\n{stderr}".strip()

    if execution.error:
        output = (
            f"{output}\n[error] {execution.error.name}: {execution.error.value}".strip()
        )

    return output or "(no output)"


async def sandbox_run_command(
    command: str,
    tool_context: ToolContext,
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    """Execute a shell command in an isolated sandbox.

    The sandbox is created lazily and reused within the session.
    Commands run in a secure container with resource limits.

    Args:
        command: Shell command to execute (e.g., "ls -la", "pip install pandas").
        tool_context: ADK tool context for session state.
        timeout: Maximum execution time in seconds (default 60).

    Returns:
        Dictionary with status, output, and optional error.
    """
    manager = get_sandbox_manager()

    result = await manager.get_or_create_sandbox(tool_context)
    sandbox = result.get("sandbox")
    error = result.get("error")

    if error or sandbox is None:
        return {"status": "error", "error": error, "output": None}

    try:
        opts = RunCommandOpts(timeout=timedelta(seconds=timeout))
        execution = await sandbox.commands.run(command, opts=opts)
        output = _format_command_output(execution)

        if execution.error:
            return {
                "status": "error",
                "error": f"Command failed: {execution.error.value}",
                "output": output,
            }

        return {"status": "success", "output": output}
    except SandboxException as e:
        error_msg = f"Sandbox error: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg, "output": None}
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg, "output": None}


async def sandbox_write_file(
    path: str,
    content: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Write content to a file in the sandbox.

    Creates the file if it doesn't exist, overwrites if it does.
    Parent directories are not created automatically.

    Args:
        path: Absolute or relative file path in the sandbox.
        content: Text content to write to the file.
        tool_context: ADK tool context for session state.

    Returns:
        Dictionary with status and optional error.
    """
    manager = get_sandbox_manager()

    result = await manager.get_or_create_sandbox(tool_context)
    sandbox = result.get("sandbox")
    error = result.get("error")

    if error or sandbox is None:
        return {"status": "error", "error": error}

    try:
        await sandbox.files.write_file(path, content)
        return {"status": "success", "path": path, "bytes_written": len(content)}
    except SandboxException as e:
        error_msg = f"Failed to write file: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg}


async def sandbox_read_file(
    path: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Read file contents from the sandbox.

    Args:
        path: Absolute or relative file path in the sandbox.
        tool_context: ADK tool context for session state.

    Returns:
        Dictionary with status, content, and optional error.
    """
    manager = get_sandbox_manager()

    result = await manager.get_or_create_sandbox(tool_context)
    sandbox = result.get("sandbox")
    error = result.get("error")

    if error or sandbox is None:
        return {"status": "error", "error": error, "content": None}

    try:
        content = await sandbox.files.read_file(path)
        return {"status": "success", "content": content, "path": path}
    except SandboxException as e:
        error_msg = f"Failed to read file: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg, "content": None}
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg, "content": None}


async def sandbox_list_files(
    directory: str,
    pattern: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Search for files in the sandbox matching a glob pattern.

    Args:
        directory: Directory to search in (e.g., "/workspace").
        pattern: Glob pattern to match (e.g., "*.py", "**/*.txt").
        tool_context: ADK tool context for session state.

    Returns:
        Dictionary with status, files list, and optional error.
        Each file has path, size, and modified_at fields.
    """
    manager = get_sandbox_manager()

    result = await manager.get_or_create_sandbox(tool_context)
    sandbox = result.get("sandbox")
    error = result.get("error")

    if error or sandbox is None:
        return {"status": "error", "error": error, "files": []}

    try:
        entries = await sandbox.files.search(
            SearchEntry(path=directory, pattern=pattern)
        )
        files = [
            {
                "path": entry.path,
                "size": entry.size,
                "modified_at": entry.modified_at.isoformat(),
            }
            for entry in entries
        ]
        return {
            "status": "success",
            "files": files,
            "directory": directory,
            "pattern": pattern,
        }
    except SandboxException as e:
        error_msg = f"Failed to list files: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg, "files": []}
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg, "files": []}


def sandbox_enabled() -> bool:
    """Check if sandbox tools are enabled."""
    from .config import load_sandbox_config

    return load_sandbox_config().enabled
