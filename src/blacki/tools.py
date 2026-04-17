"""Custom tools for the LLM agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from browser_use_sdk import AsyncBrowserUse  # type: ignore[import-untyped]
from browser_use_sdk.v2.helpers import (  # type: ignore[import-untyped]
    _async_poll_output,
)
from google.adk.tools import ToolContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_BROWSER_USE_DEFAULT_LLM = "browser-use-llm"

_browser_use_lock = asyncio.Lock()
_browser_use_client: AsyncBrowserUse | None = None
_browser_use_api_key: str | None = None


async def reset_browser_use_client_cache() -> None:
    """Close and clear the shared Browser Use client.

    Used between tests and if the API key changes, so connection pools are not
    leaked and mocked clients are not reused across pytest cases.
    """
    global _browser_use_client, _browser_use_api_key
    async with _browser_use_lock:
        if _browser_use_client is not None:
            try:
                await _browser_use_client.close()
            except Exception:
                logger.exception("Error while closing shared Browser Use client")
        _browser_use_client = None
        _browser_use_api_key = None


async def _get_shared_browser_use_client(api_key: str) -> AsyncBrowserUse:
    """Return a process-wide ``AsyncBrowserUse`` for the given API key."""
    global _browser_use_client, _browser_use_api_key
    async with _browser_use_lock:
        if _browser_use_client is not None and _browser_use_api_key == api_key:
            return _browser_use_client
        if _browser_use_client is not None:
            try:
                await _browser_use_client.close()
            except Exception:
                logger.exception(
                    "Error closing Browser Use client before API key rotation"
                )
        _browser_use_client = AsyncBrowserUse(api_key=api_key)
        _browser_use_api_key = api_key
        return _browser_use_client


def _serialize_browser_output(output: Any) -> Any:
    """Normalize Browser Use task output for JSON-friendly tool responses."""
    if output is None:
        return None
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    if isinstance(output, (list, tuple)):
        return [_serialize_browser_output(item) for item in output]
    if isinstance(output, dict):
        return {key: _serialize_browser_output(value) for key, value in output.items()}
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output
    return output


async def browser_task(
    task: str,
    tool_context: ToolContext,
    output_schema: object | None = None,
) -> dict[str, Any]:
    """Run a Browser Use Cloud agent task (navigation, forms, scraping, research).

    Uses the managed Browser Use Cloud browser with stealth features and optional
    structured JSON output. Set ``BROWSER_USE_API_KEY`` (see ``.env.example``).

    Args:
        task: Natural language description of what to do on the web.
        tool_context: ADK tool context (session state available if needed).
        output_schema: Optional JSON Schema as a ``dict``, or a Pydantic
            ``BaseModel`` subclass, for structured output (``structuredOutput``).

    Returns:
        Dictionary with ``status``, ``output``, ``live_preview_url``, ``task_id``,
        ``session_id``, and optional ``is_success`` / ``error`` fields.
    """
    _ = tool_context
    api_key = os.environ.get("BROWSER_USE_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "error",
            "error": (
                "BROWSER_USE_API_KEY is not set. Add it to the environment to enable "
                "browser automation."
            ),
            "output": None,
            "live_preview_url": None,
            "task_id": None,
            "session_id": None,
        }

    stripped_task = task.strip()
    if not stripped_task:
        return {
            "status": "error",
            "error": "Task description must be a non-empty string.",
            "output": None,
            "live_preview_url": None,
            "task_id": None,
            "session_id": None,
        }

    create_kwargs: dict[str, Any] = {
        "llm": _BROWSER_USE_DEFAULT_LLM,
        "keepAlive": False,
    }
    pydantic_schema: type[BaseModel] | None = None
    if output_schema is not None:
        if isinstance(output_schema, dict):
            create_kwargs["structured_output"] = json.dumps(output_schema)
        elif isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            pydantic_schema = output_schema
            create_kwargs["structured_output"] = json.dumps(
                output_schema.model_json_schema()
            )
        else:
            return {
                "status": "error",
                "error": (
                    "output_schema must be a JSON Schema dict or a Pydantic BaseModel "
                    "subclass."
                ),
                "output": None,
                "live_preview_url": None,
                "task_id": None,
                "session_id": None,
            }

    try:
        client = await _get_shared_browser_use_client(api_key)
        created = await client.tasks.create(stripped_task, **create_kwargs)
        session_id_str = str(created.session_id)
        task_id_str = str(created.id)

        live_preview_url: str | None = None
        try:
            session_view = await client.sessions.get(session_id_str)
            live_preview_url = session_view.live_url
        except Exception:
            logger.exception(
                "Failed to load Browser Use session for live URL (session_id=%s)",
                session_id_str,
            )

        task_result = await _async_poll_output(
            client.tasks,
            task_id_str,
            pydantic_schema,
        )
        terminal_status = task_result.task.status.value

        return {
            "status": terminal_status,
            "output": _serialize_browser_output(task_result.output),
            "live_preview_url": live_preview_url,
            "task_id": task_id_str,
            "session_id": session_id_str,
            "is_success": task_result.task.is_success,
        }
    except Exception as exc:
        logger.exception("Browser Use Cloud task failed")
        return {
            "status": "error",
            "error": str(exc),
            "output": None,
            "live_preview_url": None,
            "task_id": None,
            "session_id": None,
        }


def example_tool(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Example tool that logs a success message.

    This is a placeholder example tool. Replace with actual implementation.

    Args:
        tool_context: ADK ToolContext with access to session state

    Returns:
        A dictionary with status and message about the logging operation.
    """
    # TODO: add tool logic

    # Log the session state keys
    logger.info(f"Session state keys: {tool_context.state.to_dict().keys()}")

    message = "Successfully used example_tool."
    logger.info(message)
    return {"status": "success", "message": message}
