"""Exercise provider selection and the per-turn search budget."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import patch

import httpx
import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext

from blacki.tools.search import create_web_search_tool


@pytest.fixture
async def invocation() -> InvocationContext:
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="blacki", user_id="reader")
    return InvocationContext(
        session_service=sessions, session=session, invocation_id="search-turn"
    )


@pytest.fixture
async def search_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.MockTransport]:
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-brave")
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport) as client:
        with (
            patch("blacki.tools.search._exa_search_client", client),
            patch("blacki.tools.brave_search._brave_search_client", client),
        ):
            yield transport


@pytest.mark.parametrize(
    ("exa_enabled", "brave_enabled", "exa_status", "exa_has_results", "expected_hosts"),
    [
        (True, True, 200, True, ["api.exa.ai"]),
        (True, True, 200, False, ["api.exa.ai", "api.search.brave.com"]),
        (True, True, 500, False, ["api.exa.ai", "api.search.brave.com"]),
        (True, False, 500, False, ["api.exa.ai"]),
        (False, True, 500, False, ["api.search.brave.com"]),
    ],
)
async def test_search_provider_order_and_fallback(
    invocation: InvocationContext,
    search_transport: httpx.MockTransport,
    exa_enabled: bool,
    brave_enabled: bool,
    exa_status: int,
    exa_has_results: bool,
    expected_hosts: list[str],
) -> None:
    hosts: list[str] = []
    page = {"title": "Source", "url": "https://example.test/source"}

    def respond(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "api.exa.ai":
            assert b'"numResults":5' in request.content
            return httpx.Response(
                exa_status, json={"results": [page] if exa_has_results else []}
            )
        assert request.url.params["q"] == "release information"
        assert request.url.params["count"] == "5"
        return httpx.Response(200, json={"web": {"results": [page]}})

    search_transport.handler = respond
    tool = create_web_search_tool(exa_enabled=exa_enabled, brave_enabled=brave_enabled)
    args = {"query": " release information "}
    result = await tool.run_async(args=args, tool_context=ToolContext(invocation))
    duplicate = await tool.run_async(args=args, tool_context=ToolContext(invocation))
    different = await tool.run_async(
        args={"query": "another query"}, tool_context=ToolContext(invocation)
    )

    assert hosts == expected_hosts
    assert duplicate == result
    assert different["status"] == "error"
    if exa_has_results or brave_enabled:
        assert result["results"][0]["url"] == page["url"]
    else:
        assert result["status"] == "error"


async def test_blank_query_and_parallel_calls_respect_search_budget(
    invocation: InvocationContext, search_transport: httpx.MockTransport
) -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        assert not started.is_set()
        started.set()
        await finish.wait()
        return httpx.Response(200, json={"results": []})

    search_transport.handler = respond
    tool = create_web_search_tool(exa_enabled=True, brave_enabled=False)
    context = ToolContext(invocation)
    blank = await tool.run_async(args={"query": "  "}, tool_context=context)
    assert blank["status"] == "error"
    assert not started.is_set()

    first = asyncio.create_task(
        tool.run_async(args={"query": "first"}, tool_context=context)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        second = await tool.run_async(
            args={"query": "second"}, tool_context=ToolContext(invocation)
        )
        assert second["status"] == "error"
        assert "already requested" in second["error"]
    finally:
        finish.set()
        await first
