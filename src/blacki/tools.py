"""Custom tools for the LLM agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Literal

import httpx
from browser_use_sdk import AsyncBrowserUse  # type: ignore[import-untyped]
from browser_use_sdk.v2.client import SessionSettings  # type: ignore[import-untyped]
from google.adk.tools import ToolContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_BROWSER_USE_DEFAULT_MODEL: Literal["browser-use-llm"] = "browser-use-llm"

_browser_use_lock = asyncio.Lock()
_browser_use_client: AsyncBrowserUse | None = None
_browser_use_api_key: str | None = None

_brave_search_lock = asyncio.Lock()
_brave_search_client: httpx.AsyncClient | None = None

_pending_tasks: dict[str, dict[str, Any]] = {}
_pending_schemas: dict[str, type[BaseModel] | None] = {}


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


async def reset_brave_search_client_cache() -> None:
    """Close and clear the shared Brave Search httpx client.

    Used between tests to ensure connection pools are not leaked.
    """
    global _brave_search_client
    async with _brave_search_lock:
        if _brave_search_client is not None:
            try:
                await _brave_search_client.aclose()
            except Exception:
                logger.exception("Error while closing shared Brave Search client")
        _brave_search_client = None


async def _get_shared_brave_search_client() -> httpx.AsyncClient:
    """Return a process-wide ``httpx.AsyncClient`` for Brave Search API."""
    global _brave_search_client
    async with _brave_search_lock:
        if _brave_search_client is not None:
            return _brave_search_client
        _brave_search_client = httpx.AsyncClient(timeout=30.0)
        return _brave_search_client


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


async def _poll_task_output(
    tasks_client: Any,
    task_id: str,
    output_schema: type[BaseModel] | None,
    timeout: float,
    interval: float,
) -> Any:
    """Poll a Browser Use task until completion or timeout.

    Args:
        tasks_client: The AsyncTasks client from AsyncBrowserUse.
        task_id: The task ID to poll.
        output_schema: Optional Pydantic model for structured output parsing.
        timeout: Maximum time to wait in seconds.
        interval: Time between polls in seconds.

    Returns:
        An object with ``task`` (TaskView) and ``output`` attributes.

    Raises:
        TimeoutError: If the task doesn't complete within the timeout.
    """
    import time

    start = time.monotonic()
    while True:
        task_view = await tasks_client.get(task_id)
        status_value = task_view.status.value

        if status_value in ("finished", "failed", "stopped"):
            output: Any = task_view.output
            if output is not None and output_schema is not None:
                try:
                    parsed = json.loads(output)
                    output = output_schema.model_validate(parsed)
                except (json.JSONDecodeError, Exception):  # noqa: S110
                    pass

            class _TaskResult:
                task: Any
                output: Any

            result = _TaskResult()
            result.task = task_view
            result.output = output
            return result

        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise TimeoutError(
                f"Task {task_id} did not complete within {timeout}s "
                f"(status: {status_value})"
            )

        await asyncio.sleep(interval)


async def browser_task(
    task: str,
    tool_context: ToolContext,
    output_schema: object | None = None,
    *,
    keep_alive: bool = False,
    session_id: str | None = None,
    profile_id: str | None = None,
    model: str | None = None,
    start_url: str | None = None,
    max_steps: int | None = None,
    proxy_country: str | None = None,
) -> dict[str, Any]:
    """Start a Browser Use Cloud agent task and return immediately.

    This is a long-running tool. It creates the browser task and returns
    immediately with a task_id. Use ``browser_get_task_status`` to poll for
    completion and retrieve results.

    The browser runs in the cloud with stealth features. Set
    ``BROWSER_USE_API_KEY`` (see ``.env.example``).

    Args:
        task: Natural language description of what to do on the web.
        tool_context: ADK tool context (session state available if needed).
        output_schema: Optional JSON Schema as a ``dict``, or a Pydantic
            ``BaseModel`` subclass, for structured output.
        keep_alive: If True, keep the browser session alive for follow-up tasks.
        session_id: Resume an existing session from a previous ``keep_alive=True`` call.
        profile_id: Browser profile ID for persistent authentication.
        model: LLM model: "browser-use-llm" (default), "claude-sonnet-4.6",
            "claude-opus-4.6", "gpt-5.4-mini".
        start_url: Navigate to this URL before starting the task.
        max_steps: Maximum number of browser actions to take.
        proxy_country: Two-letter country code for geo-targeted proxy (e.g., "us").

    Returns:
        Dictionary with ``status`` ("pending"), ``task_id``, ``session_id``,
        ``live_preview_url``, and optional ``error`` field.
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
        "llm": model or _BROWSER_USE_DEFAULT_MODEL,
        "keepAlive": keep_alive,
    }

    if session_id:
        create_kwargs["session_id"] = session_id
    if start_url:
        create_kwargs["start_url"] = start_url
    if max_steps is not None:
        create_kwargs["max_steps"] = max_steps

    session_settings_data: dict[str, Any] = {}
    if profile_id:
        session_settings_data["profileId"] = profile_id
    if proxy_country:
        session_settings_data["proxyCountryCode"] = proxy_country
    if session_settings_data:
        create_kwargs["session_settings"] = SessionSettings(**session_settings_data)

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

        _pending_tasks[task_id_str] = {
            "api_key": api_key,
            "session_id": session_id_str,
            "live_preview_url": live_preview_url,
            "status": "pending",
        }
        _pending_schemas[task_id_str] = pydantic_schema

        return {
            "status": "pending",
            "task_id": task_id_str,
            "session_id": session_id_str,
            "live_preview_url": live_preview_url,
            "message": (
                "Task started. Call browser_get_task_status with task_id to check "
                "completion and get results."
            ),
        }
    except Exception as exc:
        logger.exception("Browser Use Cloud task creation failed")
        return {
            "status": "error",
            "error": str(exc),
            "output": None,
            "live_preview_url": None,
            "task_id": None,
            "session_id": None,
        }


async def browser_get_task_status(
    task_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Check status of a browser task and get results if complete.

    Call this after ``browser_task`` returns a task_id. Polls the Browser Use
    Cloud API for task completion. Returns immediately with current status.

    Args:
        task_id: The task ID returned from ``browser_task``.
        tool_context: ADK tool context.

    Returns:
        Dictionary with ``status`` ("pending", "running", "finished", "error"),
        ``output`` (when finished), ``live_preview_url``, ``is_success``, and
        optional ``error`` field.
    """
    _ = tool_context
    api_key = os.environ.get("BROWSER_USE_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "error",
            "error": "BROWSER_USE_API_KEY is not set.",
            "task_id": task_id,
        }

    task_info = _pending_tasks.get(task_id)
    if not task_info:
        return {
            "status": "error",
            "error": (
                f"Unknown task_id: {task_id}. Start a task with browser_task first."
            ),
            "task_id": task_id,
        }

    pydantic_schema = _pending_schemas.get(task_id)

    try:
        client = await _get_shared_browser_use_client(api_key)
        task_result = await _poll_task_output(
            client.tasks,
            task_id,
            pydantic_schema,
            timeout=2.0,
            interval=1.0,
        )
        terminal_status = task_result.task.status.value

        result: dict[str, Any] = {
            "status": terminal_status,
            "task_id": task_id,
            "session_id": task_info.get("session_id"),
            "live_preview_url": task_info.get("live_preview_url"),
            "is_success": task_result.task.is_success,
        }

        if terminal_status in ("finished", "error", "stopped"):
            result["output"] = _serialize_browser_output(task_result.output)
            _pending_tasks.pop(task_id, None)
            _pending_schemas.pop(task_id, None)
        else:
            result["output"] = None
            result["message"] = (
                f"Task is {terminal_status}. Call browser_get_task_status again to "
                "check for completion."
            )

        return result
    except TimeoutError:
        return {
            "status": "running",
            "task_id": task_id,
            "session_id": task_info.get("session_id"),
            "live_preview_url": task_info.get("live_preview_url"),
            "output": None,
            "message": (
                "Task still running. Call browser_get_task_status again to check "
                "for completion."
            ),
        }
    except Exception as exc:
        logger.exception("Failed to get browser task status")
        return {
            "status": "error",
            "error": str(exc),
            "task_id": task_id,
        }


async def browser_stop_session(
    session_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Stop a keep-alive Browser Use session and release resources.

    Call this when you're done with a session that was created with
    ``keep_alive=True``. Stopping frees the browser sandbox and prevents
    unnecessary charges.

    Args:
        session_id: The session ID returned from a previous ``browser_task`` call.
        tool_context: ADK tool context.

    Returns:
        Dictionary with ``status`` and optional ``error`` field.
    """
    _ = tool_context
    api_key = os.environ.get("BROWSER_USE_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "error",
            "error": "BROWSER_USE_API_KEY is not set.",
        }

    try:
        client = await _get_shared_browser_use_client(api_key)
        await client.sessions.delete(session_id)
        return {"status": "success", "session_id": session_id}
    except Exception as exc:
        logger.exception("Failed to stop Browser Use session")
        return {"status": "error", "error": str(exc), "session_id": session_id}


async def browser_list_profiles(
    tool_context: ToolContext,
    *,
    query: str | None = None,
) -> dict[str, Any]:
    """List available Browser Use profiles for authenticated browsing.

    Profiles store persistent browser state (cookies, localStorage, saved
    passwords). Use a ``profile_id`` in ``browser_task`` to log in once and
    reuse the session across multiple tasks.

    Args:
        tool_context: ADK tool context.
        query: Optional search query to filter profiles by name.

    Returns:
        Dictionary with ``status``, ``profiles`` (list of profile objects with
        ``id`` and ``name``), and optional ``error`` field.
    """
    _ = tool_context
    api_key = os.environ.get("BROWSER_USE_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "error",
            "error": "BROWSER_USE_API_KEY is not set.",
            "profiles": [],
        }

    try:
        client = await _get_shared_browser_use_client(api_key)
        response = await client.profiles.list(query=query)
        profiles = [{"id": str(p.id), "name": p.name} for p in response.profiles]
        return {"status": "success", "profiles": profiles}
    except Exception as exc:
        logger.exception("Failed to list Browser Use profiles")
        return {"status": "error", "error": str(exc), "profiles": []}


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


BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"


async def brave_search(
    query: str,
    tool_context: ToolContext,
    count: int = 10,
) -> dict[str, Any]:
    """Search the web using Brave Search API.

    This is a model-agnostic alternative to Google Search that works with
    any LLM provider via LiteLLM/OpenRouter. Set BRAVE_SEARCH_API_KEY
    in your environment.

    Args:
        query: The search query string.
        tool_context: ADK tool context.
        count: Maximum number of results to return (default 10, max 20).

    Returns:
        Dictionary with status, query, and results list. Each result has
        title, url, and description fields.
    """
    _ = tool_context

    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "error",
            "error": (
                "BRAVE_SEARCH_API_KEY is not set. Get a free API key at "
                "https://brave.com/search/api/ and add it to your environment."
            ),
            "query": query,
            "results": [],
        }

    if not query.strip():
        return {
            "status": "error",
            "error": "Search query must be a non-empty string.",
            "query": query,
            "results": [],
        }

    count = min(max(1, count), 20)

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    params: dict[str, str | int] = {
        "q": query.strip(),
        "count": count,
        "search_lang": "en",
        "country": "us",
    }

    try:
        client = await _get_shared_brave_search_client()
        response = await client.get(
            BRAVE_SEARCH_API_URL,
            headers=headers,
            params=params,
        )

        if response.status_code == 401:
            return {
                "status": "error",
                "error": "Invalid BRAVE_SEARCH_API_KEY. Check your API key.",
                "query": query,
                "results": [],
            }

        if response.status_code == 429:
            return {
                "status": "error",
                "error": "Brave Search API rate limit exceeded. Try again later.",
                "query": query,
                "results": [],
            }

        response.raise_for_status()

        data = response.json()
        web_results = data.get("web", {}).get("results", [])

        results: list[dict[str, str]] = []
        for item in web_results[:count]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                }
            )

        return {
            "status": "success",
            "query": query,
            "results": results,
        }

    except Exception:
        logger.exception("Brave Search API error")
        return {
            "status": "error",
            "error": "Brave Search API request failed.",
            "query": query,
            "results": [],
        }


def brave_search_api_key_available() -> bool:
    """Check if BRAVE_SEARCH_API_KEY is set in environment."""
    return bool(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip())
