"""Exa-backed web search tool."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

EXA_SEARCH_API_URL = "https://api.exa.ai/search"

_exa_search_lock = asyncio.Lock()
_exa_search_client: httpx.AsyncClient | None = None


async def reset_exa_search_client_cache() -> None:
    """Close and clear the shared Exa Search client between tests."""
    await close_shared_exa_search_client()


async def close_shared_exa_search_client() -> None:
    """Close the process-wide Exa Search client during shutdown."""
    global _exa_search_client
    async with _exa_search_lock:
        if _exa_search_client is not None:
            try:
                await _exa_search_client.aclose()
            except Exception:
                logger.exception("Error while closing shared Exa Search client")
        _exa_search_client = None


async def _get_shared_exa_search_client() -> httpx.AsyncClient:
    """Return a process-wide async client for the Exa Search API."""
    global _exa_search_client
    async with _exa_search_lock:
        if _exa_search_client is not None:
            return _exa_search_client
        _exa_search_client = httpx.AsyncClient(timeout=30.0)
        return _exa_search_client


def _error_result(query: str, message: str) -> dict[str, Any]:
    """Build the stable error contract returned to the agent."""
    return {
        "status": "error",
        "error": message,
        "query": query,
        "results": [],
    }


async def exa_search(
    query: str,
    num_results: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Search the public web with Exa and return relevant page highlights.

    Use this as the primary web search tool for current information, factual
    lookups, and source discovery. Results contain original URLs and concise
    excerpts suitable for agent workflows.

    Args:
        query: Natural-language search query.
        num_results: Number of results requested, clamped to Exa's 1-100 range.

    Returns:
        A dictionary containing status, query, search type, and normalized results.
    """
    _ = tool_context

    api_key = os.environ.get("EXA_API_KEY", "").strip()
    if not api_key:
        return _error_result(
            query,
            "EXA_API_KEY is not set. Add an Exa API key to the environment.",
        )

    normalized_query = query.strip()
    if not normalized_query:
        return _error_result(query, "Search query must be a non-empty string.")

    result_limit = min(max(num_results, 1), 100)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    payload: dict[str, Any] = {
        "query": normalized_query,
        "type": "auto",
        "numResults": result_limit,
        "contents": {"highlights": True},
    }

    try:
        client = await _get_shared_exa_search_client()
        response = await client.post(
            EXA_SEARCH_API_URL,
            headers=headers,
            json=payload,
        )

        if response.status_code in (401, 403):
            return _error_result(
                normalized_query,
                "Invalid EXA_API_KEY. Check the configured API key.",
            )
        if response.status_code == 429:
            return _error_result(
                normalized_query,
                "Exa Search API rate limit exceeded. Try again later.",
            )
        if response.status_code in (400, 422):
            return _error_result(
                normalized_query,
                "Exa Search rejected the query parameters.",
            )

        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        logger.exception("Exa Search API request failed")
        return _error_result(
            normalized_query,
            "Exa Search API request failed.",
        )

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return _error_result(
            normalized_query,
            "Exa Search API returned an invalid response.",
        )

    results: list[dict[str, Any]] = []
    for item in data["results"][:result_limit]:
        if not isinstance(item, dict):
            continue
        raw_highlights = item.get("highlights", [])
        highlights = (
            [value for value in raw_highlights if isinstance(value, str)]
            if isinstance(raw_highlights, list)
            else []
        )
        title = item.get("title")
        url = item.get("url")
        published_date = item.get("publishedDate")
        author = item.get("author")
        results.append(
            {
                "title": title if isinstance(title, str) else "",
                "url": url if isinstance(url, str) else "",
                "published_date": (
                    published_date if isinstance(published_date, str) else None
                ),
                "author": author if isinstance(author, str) else None,
                "highlights": highlights,
            }
        )

    search_type = data.get("searchType")
    return {
        "status": "success",
        "query": normalized_query,
        "search_type": search_type if isinstance(search_type, str) else "auto",
        "results": results,
    }


def exa_search_api_key_available() -> bool:
    """Return whether EXA_API_KEY is configured in the environment."""
    return bool(os.environ.get("EXA_API_KEY", "").strip())
