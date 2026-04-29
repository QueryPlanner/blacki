"""Custom tools for the LLM agent."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

_brave_search_lock = asyncio.Lock()
_brave_search_client: httpx.AsyncClient | None = None


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
