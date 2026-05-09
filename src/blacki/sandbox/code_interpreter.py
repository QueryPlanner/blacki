"""ADK tool for OpenSandbox Python code execution."""

from __future__ import annotations

import logging
from typing import Any

from code_interpreter import CodeInterpreter, SupportedLanguage
from google.adk.tools import ToolContext
from opensandbox.exceptions import SandboxException

from .manager import get_sandbox_manager

logger = logging.getLogger(__name__)


async def sandbox_execute_code(
    code: str,
    tool_context: ToolContext,
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    """Execute Python code in an isolated sandbox with state persistence.

    Variables and imports persist across calls within the same session.

    Args:
        code: Python code to execute.
        tool_context: ADK tool context for session state.
        timeout: Maximum execution time in seconds (default 60).

    Returns:
        Dictionary with status, output, and optional error.
    """
    manager = get_sandbox_manager()

    result = await manager.get_or_create_sandbox(tool_context.state)
    sandbox = result.get("sandbox")
    error = result.get("error")

    if error or sandbox is None:
        return {"status": "error", "error": error, "output": None}

    try:
        interpreter = await CodeInterpreter.create(sandbox=sandbox)

        # Use default context for Python to maintain state
        execution = await interpreter.codes.run(
            code,
            language=SupportedLanguage.PYTHON,
        )

        stdout = "\n".join(msg.text for msg in execution.logs.stdout if msg.text)
        stderr = "\n".join(msg.text for msg in execution.logs.stderr if msg.text)

        output = stdout.strip()
        if stderr:
            output = f"{output}\n[stderr]\n{stderr}".strip()

        if execution.result:
            result_text = "\n".join(msg.text for msg in execution.result if msg.text)
            output = f"{output}\n[result]\n{result_text}".strip()

        if execution.error:
            error_msg = f"{execution.error.name}: {execution.error.value}"
            return {
                "status": "error",
                "error": error_msg,
                "output": output or "(no output)",
            }

        return {"status": "success", "output": output or "(no output)"}
    except SandboxException as e:
        error_msg = f"Sandbox code execution error: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg, "output": None}
    except Exception as e:
        error_msg = f"Unexpected error in code execution: {e}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg, "output": None}
