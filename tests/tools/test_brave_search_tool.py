"""Tests for Brave Search tool."""

import logging
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import MockState, MockToolContext
from google.adk.tools import ToolContext

from blacki.tools import (
    brave_search,
    brave_search_api_key_available,
)


class TestBraveSearch:
    """Tests for brave_search function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_brave_search_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return a clear error payload."""
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await brave_search("test query", tool_context)

        assert result["status"] == "error"
        assert "BRAVE_SEARCH_API_KEY" in result["error"]
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_brave_search_empty_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty query should return error."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_key")
        tool_context = self._tool_context()

        result = await brave_search("   ", tool_context)

        assert result["status"] == "error"
        assert "non-empty" in result["error"].lower()
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_brave_search_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful search should return results."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "description": "Description 1",
                    },
                    {
                        "title": "Test Result 2",
                        "url": "https://example.com/2",
                        "description": "Description 2",
                    },
                ]
            }
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await brave_search("test query", tool_context)

        assert result["status"] == "success"
        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Test Result 1"
        assert result["results"][0]["url"] == "https://example.com/1"
        assert result["results"][1]["title"] == "Test Result 2"

    @pytest.mark.asyncio
    async def test_brave_search_limits_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should limit results to count parameter."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": f"Result {i}",
                        "url": f"https://example.com/{i}",
                        "description": f"Desc {i}",
                    }
                    for i in range(20)
                ]
            }
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await brave_search("test query", tool_context, count=5)

        assert result["status"] == "success"
        assert len(result["results"]) == 5

    @pytest.mark.asyncio
    async def test_brave_search_invalid_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid API key should return error."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "invalid_key")
        tool_context = self._tool_context()

        mock_response = MagicMock()
        mock_response.status_code = 401

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await brave_search("test query", tool_context)

        assert result["status"] == "error"
        assert "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_brave_search_rate_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rate limit should return error."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await brave_search("test query", tool_context)

        assert result["status"] == "error"
        assert "rate limit" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_brave_search_timeout(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Timeout should be handled gracefully."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_key")
        caplog.set_level(logging.DEBUG)
        tool_context = self._tool_context()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await brave_search("test query", tool_context)

        assert result["status"] == "error"
        assert "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_brave_search_http_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """HTTP errors should be handled gracefully."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_key")
        caplog.set_level(logging.DEBUG)
        tool_context = self._tool_context()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await brave_search("test query", tool_context)

        assert result["status"] == "error"
        assert "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_brave_search_empty_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty results should return empty list."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_key")
        tool_context = self._tool_context()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"web": {"results": []}}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await brave_search("obscure query", tool_context)

        assert result["status"] == "success"
        assert result["results"] == []


class TestBraveSearchApiKeyAvailable:
    """Tests for brave_search_api_key_available function."""

    def test_returns_true_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return True when API key is set."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test_key")
        assert brave_search_api_key_available() is True

    def test_returns_false_when_key_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return False when API key is not set."""
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        assert brave_search_api_key_available() is False

    def test_returns_false_when_key_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return False when API key is empty string."""
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "   ")
        assert brave_search_api_key_available() is False
