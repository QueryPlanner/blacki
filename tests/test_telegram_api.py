"""Tests for Telegram API client."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blacki.telegram.api import TelegramApiClient, TelegramApiError
from blacki.telegram.types import Message, ParseMode


class TestTelegramApiClient:
    """Tests for TelegramApiClient."""

    def test_build_file_url(self) -> None:
        """Test _build_file_url constructs correct URL."""
        client = TelegramApiClient("my-token")
        url = client._build_file_url("documents/file.txt")
        assert url == "https://api.telegram.org/file/botmy-token/documents/file.txt"

    @pytest.mark.asyncio
    async def test_get_file(self) -> None:
        """Test get_file returns file info as dict."""
        client = TelegramApiClient("token")
        mock_result: dict[str, Any] = {
            "file_id": "abc",
            "file_path": "documents/file.txt",
        }

        with patch.object(client, "_request", AsyncMock(return_value=mock_result)):
            result = await client.get_file("abc")

        assert result == mock_result

    @pytest.mark.asyncio
    async def test_get_file_non_dict_result(self) -> None:
        """Test get_file returns empty dict when result is not a dict."""
        client = TelegramApiClient("token")

        with patch.object(client, "_request", AsyncMock(return_value="not_a_dict")):
            result = await client.get_file("abc")

        assert result == {}

    @pytest.mark.asyncio
    async def test_download_file(self) -> None:
        """Test download_file returns bytes content."""
        client = TelegramApiClient("token")
        mock_response = MagicMock()
        mock_response.content = b"file bytes"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_ensure_client", AsyncMock(return_value=mock_client)
        ):
            result = await client.download_file("documents/file.txt")

        assert result == b"file bytes"
        mock_client.get.assert_called_once_with(
            "https://api.telegram.org/file/bottoken/documents/file.txt",
            timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_send_document_success(self) -> None:
        """Test send_document returns Message on success."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "result": {
                "message_id": 42,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "document": {
                    "file_id": "doc123",
                    "file_unique_id": "uniq123",
                },
            },
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_ensure_client", AsyncMock(return_value=mock_http_client)
        ):
            result = await client.send_document(
                chat_id=123,
                document_bytes=b"content",
                filename="test.txt",
                caption="A test file",
                message_thread_id=456,
            )

        assert isinstance(result, Message)
        assert result.message_id == 42
        assert result.chat.id == 123

    @pytest.mark.asyncio
    async def test_send_document_minimal_params(self) -> None:
        """Test send_document with only required params."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "result": {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
            },
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_ensure_client", AsyncMock(return_value=mock_http_client)
        ):
            result = await client.send_document(
                chat_id=123, document_bytes=b"content", filename="test.txt"
            )

        assert isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_send_document_with_parse_mode(self) -> None:
        """Test send_document with parse_mode."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "result": {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
            },
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_ensure_client", AsyncMock(return_value=mock_http_client)
        ):
            await client.send_document(
                chat_id=123,
                document_bytes=b"content",
                filename="test.txt",
                parse_mode=ParseMode.HTML,
            )

        call_args = mock_http_client.post.call_args
        assert call_args.kwargs["data"]["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_send_document_http_error_with_json_description(
        self,
    ) -> None:
        """Test send_document raises on HTTP error with JSON description."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"description": "Bad Request"}
        mock_response.text = "raw error"

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                client, "_ensure_client", AsyncMock(return_value=mock_http_client)
            ),
            pytest.raises(TelegramApiError, match="Bad Request"),
        ):
            await client.send_document(
                chat_id=123, document_bytes=b"content", filename="test.txt"
            )

    @pytest.mark.asyncio
    async def test_send_document_http_error_no_json(self) -> None:
        """Test send_document raises on HTTP error without JSON."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("not json")
        mock_response.text = "raw error text"

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                client, "_ensure_client", AsyncMock(return_value=mock_http_client)
            ),
            pytest.raises(TelegramApiError, match="raw error text"),
        ):
            await client.send_document(
                chat_id=123, document_bytes=b"content", filename="test.txt"
            )

    @pytest.mark.asyncio
    async def test_send_document_api_not_ok(self) -> None:
        """Test send_document raises when API returns ok=False."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": False,
            "description": "Forbidden",
            "error_code": 403,
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                client, "_ensure_client", AsyncMock(return_value=mock_http_client)
            ),
            pytest.raises(TelegramApiError, match="Forbidden"),
        ):
            await client.send_document(
                chat_id=123, document_bytes=b"content", filename="test.txt"
            )
