"""Tests for the Exa-backed web search tool."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, create_autospec, patch

import httpx
import pytest
from conftest import MockState, MockToolContext
from google.adk.tools import FunctionTool, ToolContext

from blacki.tools.search import (
    EXA_SEARCH_API_URL,
    _get_shared_exa_search_client,
    exa_search,
    exa_search_api_key_available,
    reset_exa_search_client_cache,
)


def _response(status_code: int, data: Any) -> httpx.Response:
    request = httpx.Request("POST", EXA_SEARCH_API_URL)
    return httpx.Response(status_code, json=data, request=request)


def _client(response: httpx.Response) -> AsyncMock:
    client = cast(
        AsyncMock,
        create_autospec(httpx.AsyncClient, instance=True, spec_set=True),
    )
    client.post.return_value = response
    return client


def _tool_context() -> ToolContext:
    return cast(ToolContext, MockToolContext(state=MockState({})))


@pytest.fixture(autouse=True)
async def reset_client() -> AsyncGenerator[None, None]:
    """Reset the shared Exa client around every test."""
    await reset_exa_search_client_cache()
    yield
    await reset_exa_search_client_cache()


class TestSharedExaSearchClient:
    """Verify the shared client lifecycle."""

    @pytest.mark.asyncio
    async def test_reuses_shared_client(self) -> None:
        first = await _get_shared_exa_search_client()
        second = await _get_shared_exa_search_client()

        assert first is second

    @pytest.mark.asyncio
    async def test_reset_closes_shared_client(self) -> None:
        client = await _get_shared_exa_search_client()

        await reset_exa_search_client_cache()

        assert client.is_closed

    @pytest.mark.asyncio
    async def test_reset_handles_missing_client(self) -> None:
        import blacki.tools.search as exa

        exa._exa_search_client = None

        await reset_exa_search_client_cache()

        assert exa._exa_search_client is None

    @pytest.mark.asyncio
    async def test_reset_handles_close_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import blacki.tools.search as exa

        client = create_autospec(httpx.AsyncClient, instance=True, spec_set=True)
        client.aclose.side_effect = RuntimeError("close failed")
        exa._exa_search_client = client

        with caplog.at_level(logging.ERROR):
            await reset_exa_search_client_cache()

        assert exa._exa_search_client is None
        assert "Error while closing shared Exa Search client" in caplog.text


class TestExaSearch:
    """Verify Exa requests and the stable response contract."""

    def test_adk_declaration_exposes_only_model_arguments(self) -> None:
        """Verify ADK omits ToolContext from the model-visible schema."""
        declaration = FunctionTool(exa_search)._get_declaration()

        assert declaration is not None
        parameters = declaration.parameters_json_schema
        assert parameters is not None
        assert set(parameters["properties"]) == {"query", "num_results"}
        assert parameters["required"] == ["query", "num_results"]

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)

        result = await exa_search("test query", 5, _tool_context())

        assert result == {
            "status": "error",
            "error": "EXA_API_KEY is not set. Add an Exa API key to the environment.",
            "query": "test query",
            "results": [],
        }

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")

        result = await exa_search("   ", 5, _tool_context())

        assert result["status"] == "error"
        assert "non-empty" in result["error"]
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_success_sends_expected_request_and_normalizes_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        response = _response(
            200,
            {
                "searchType": "auto",
                "results": [
                    {
                        "title": "First result",
                        "url": "https://example.com/first",
                        "publishedDate": "2026-07-19T00:00:00.000Z",
                        "author": "Example Author",
                        "highlights": ["Useful excerpt", 42],
                        "text": "Raw page text must not be returned",
                    },
                    {
                        "title": None,
                        "url": 42,
                        "highlights": "invalid shape",
                    },
                    "invalid result",
                ],
                "costDollars": {"total": 0.01},
            },
        )
        client = _client(response)

        with patch(
            "blacki.tools.search._get_shared_exa_search_client",
            new=AsyncMock(return_value=client),
        ):
            result = await exa_search("  current AI news  ", 3, _tool_context())

        client.post.assert_awaited_once_with(
            EXA_SEARCH_API_URL,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": "test-key",
            },
            json={
                "query": "current AI news",
                "type": "auto",
                "numResults": 3,
                "contents": {"highlights": True},
            },
        )
        assert result == {
            "status": "success",
            "query": "current AI news",
            "search_type": "auto",
            "results": [
                {
                    "title": "First result",
                    "url": "https://example.com/first",
                    "published_date": "2026-07-19T00:00:00.000Z",
                    "author": "Example Author",
                    "highlights": ["Useful excerpt"],
                },
                {
                    "title": "",
                    "url": "",
                    "published_date": None,
                    "author": None,
                    "highlights": [],
                },
            ],
        }
        assert "text" not in result["results"][0]
        assert "costDollars" not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("requested", "expected"),
        [(-5, 1), (101, 100)],
    )
    async def test_clamps_result_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        requested: int,
        expected: int,
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        client = _client(_response(200, {"results": []}))

        with patch(
            "blacki.tools.search._get_shared_exa_search_client",
            new=AsyncMock(return_value=client),
        ):
            await exa_search("query", requested, _tool_context())

        request_payload = client.post.await_args.kwargs["json"]
        assert request_payload["numResults"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status_code", "expected_error"),
        [
            (401, "Invalid EXA_API_KEY"),
            (403, "Invalid EXA_API_KEY"),
            (429, "rate limit"),
            (400, "rejected"),
            (422, "rejected"),
        ],
    )
    async def test_known_http_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status_code: int,
        expected_error: str,
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        client = _client(_response(status_code, {"error": "provider detail"}))

        with patch(
            "blacki.tools.search._get_shared_exa_search_client",
            new=AsyncMock(return_value=client),
        ):
            result = await exa_search("query", 5, _tool_context())

        assert result["status"] == "error"
        assert expected_error in result["error"]
        assert "provider detail" not in result["error"]

    @pytest.mark.asyncio
    async def test_server_error_does_not_log_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sensitive_value = "sensitive-test-value"
        monkeypatch.setenv("EXA_API_KEY", sensitive_value)
        client = _client(_response(500, {"error": sensitive_value}))

        with (
            patch(
                "blacki.tools.search._get_shared_exa_search_client",
                new=AsyncMock(return_value=client),
            ),
            caplog.at_level(logging.ERROR),
        ):
            result = await exa_search("query", 5, _tool_context())

        assert result["status"] == "error"
        assert "request failed" in result["error"]
        assert sensitive_value not in caplog.text

    @pytest.mark.asyncio
    async def test_request_error_returns_stable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        client = create_autospec(httpx.AsyncClient, instance=True, spec_set=True)
        client.post.side_effect = httpx.TimeoutException("timed out")

        with patch(
            "blacki.tools.search._get_shared_exa_search_client",
            new=AsyncMock(return_value=client),
        ):
            result = await exa_search("query", 5, _tool_context())

        assert result["status"] == "error"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_malformed_json_returns_stable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        response = httpx.Response(
            200,
            content=b"not-json",
            request=httpx.Request("POST", EXA_SEARCH_API_URL),
        )
        client = _client(response)

        with patch(
            "blacki.tools.search._get_shared_exa_search_client",
            new=AsyncMock(return_value=client),
        ):
            result = await exa_search("query", 5, _tool_context())

        assert result["status"] == "error"
        assert "request failed" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [[], {}, {"results": None}],
    )
    async def test_invalid_response_shape_returns_stable_error(
        self, monkeypatch: pytest.MonkeyPatch, payload: Any
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        client = _client(_response(200, payload))

        with patch(
            "blacki.tools.search._get_shared_exa_search_client",
            new=AsyncMock(return_value=client),
        ):
            result = await exa_search("query", 5, _tool_context())

        assert result["status"] == "error"
        assert "invalid response" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_search_type_defaults_to_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        client = _client(_response(200, {"searchType": 7, "results": []}))

        with patch(
            "blacki.tools.search._get_shared_exa_search_client",
            new=AsyncMock(return_value=client),
        ):
            result = await exa_search("query", 5, _tool_context())

        assert result["search_type"] == "auto"


class TestExaSearchApiKeyAvailable:
    """Verify feature detection without exposing the key."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("test-key", True), ("", False), ("   ", False)],
    )
    def test_detects_configured_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
        expected: bool,
    ) -> None:
        monkeypatch.setenv("EXA_API_KEY", value)

        assert exa_search_api_key_available() is expected
