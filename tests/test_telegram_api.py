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

    @pytest.mark.asyncio
    async def test_send_document_api_not_ok_with_retry_after(self) -> None:
        """Test send_document raises with retry_after when rate-limited."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": False,
            "description": "Too Many Requests",
            "error_code": 429,
            "parameters": {"retry_after": 30},
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                client, "_ensure_client", AsyncMock(return_value=mock_http_client)
            ),
            pytest.raises(TelegramApiError, match="Too Many Requests") as exc_info,
        ):
            await client.send_document(
                chat_id=123, document_bytes=b"content", filename="test.txt"
            )

        assert exc_info.value.retry_after == 30

    @pytest.mark.asyncio
    async def test_send_audio_success_with_optional_fields(self) -> None:
        """send_audio uploads an MP3 to Telegram's native audio endpoint."""
        client = TelegramApiClient("token")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "result": {
                "message_id": 42,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": -123, "type": "supergroup"},
                "audio": {
                    "file_id": "audio123",
                    "file_unique_id": "unique123",
                    "duration": 1,
                    "mime_type": "audio/mpeg",
                },
            },
        }
        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_ensure_client", AsyncMock(return_value=mock_http_client)
        ):
            result = await client.send_audio(
                chat_id=-123,
                audio_bytes=b"ID3-audio",
                filename="speech.mp3",
                caption="Spoken response",
                message_thread_id=77,
                parse_mode=ParseMode.HTML,
            )

        assert isinstance(result, Message)
        assert result.audio is not None
        call = mock_http_client.post.await_args
        assert call.args == ("https://api.telegram.org/bottoken/sendAudio",)
        assert call.kwargs["data"] == {
            "chat_id": -123,
            "caption": "Spoken response",
            "message_thread_id": 77,
            "parse_mode": "HTML",
        }
        assert call.kwargs["files"] == {
            "audio": ("speech.mp3", b"ID3-audio", "audio/mpeg")
        }
        assert call.kwargs["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_send_audio_success_with_required_fields_only(self) -> None:
        """send_audio omits unset optional multipart fields."""
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
            await client.send_audio(123, b"audio", "speech.mp3")

        assert mock_http_client.post.await_args.kwargs["data"] == {"chat_id": 123}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("json_error", [False, True])
    async def test_send_audio_http_error(self, json_error: bool) -> None:
        """send_audio sanitizes either structured or plain HTTP failures."""
        client = TelegramApiClient("token")
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "plain failure"
        if json_error:
            mock_response.json.side_effect = ValueError("not json")
            expected = "plain failure"
        else:
            mock_response.json.return_value = {"description": "API failure"}
            expected = "API failure"
        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                client, "_ensure_client", AsyncMock(return_value=mock_http_client)
            ),
            pytest.raises(TelegramApiError, match=expected),
        ):
            await client.send_audio(123, b"audio", "speech.mp3")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("description", "parameters", "expected", "retry_after"),
        [
            ("Forbidden", None, "Forbidden", None),
            ("Rate limited", {"retry_after": 9}, "Rate limited", 9),
            (None, None, "Unknown Telegram API error", None),
        ],
    )
    async def test_send_audio_api_error(
        self,
        description: str | None,
        parameters: dict[str, int] | None,
        expected: str,
        retry_after: int | None,
    ) -> None:
        """send_audio preserves Telegram API error metadata."""
        client = TelegramApiClient("token")
        body: dict[str, Any] = {
            "ok": False,
            "description": description,
            "error_code": 429 if parameters else 403,
        }
        if parameters is not None:
            body["parameters"] = parameters
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = body
        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                client, "_ensure_client", AsyncMock(return_value=mock_http_client)
            ),
            pytest.raises(TelegramApiError, match=expected) as exc_info,
        ):
            await client.send_audio(123, b"audio", "speech.mp3")

        assert exc_info.value.retry_after == retry_after

    @pytest.mark.asyncio
    async def test_answer_callback_query(self) -> None:
        """Test answer_callback_query sends correct params."""
        client = TelegramApiClient("token")

        with patch.object(client, "_request", AsyncMock(return_value=True)) as mock_req:
            result = await client.answer_callback_query(
                "query123",
                text="Hello",
                show_alert=True,
                url="https://example.com",
                cache_time=60,
            )

            assert result is True
            mock_req.assert_called_once_with(
                "answerCallbackQuery",
                {
                    "callback_query_id": "query123",
                    "text": "Hello",
                    "show_alert": True,
                    "url": "https://example.com",
                    "cache_time": 60,
                },
            )

    @pytest.mark.asyncio
    async def test_answer_callback_query_minimal(self) -> None:
        """Test answer_callback_query sends correct params minimal."""
        client = TelegramApiClient("token")

        with patch.object(client, "_request", AsyncMock(return_value=True)) as mock_req:
            result = await client.answer_callback_query("query123")

            assert result is True
            mock_req.assert_called_once_with(
                "answerCallbackQuery",
                {
                    "callback_query_id": "query123",
                },
            )
        """Test answer_callback_query sends correct params."""
        client = TelegramApiClient("token")

        with patch.object(client, "_request", AsyncMock(return_value=True)) as mock_req:
            result = await client.answer_callback_query(
                "query123",
                text="Hello",
                show_alert=True,
                url="https://example.com",
                cache_time=60,
            )

            assert result is True
            mock_req.assert_called_once_with(
                "answerCallbackQuery",
                {
                    "callback_query_id": "query123",
                    "text": "Hello",
                    "show_alert": True,
                    "url": "https://example.com",
                    "cache_time": 60,
                },
            )

    @pytest.mark.asyncio
    async def test_send_message_with_reply_markup(self) -> None:
        """Test send_message handles reply_markup."""
        client = TelegramApiClient("token")

        from blacki.telegram.types import InlineKeyboardButton, InlineKeyboardMarkup

        markup = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Btn", callback_data="cb")]]
        )

        mock_result = {
            "message_id": 1,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": 123, "type": "private"},
        }
        with patch.object(
            client, "_request", AsyncMock(return_value=mock_result)
        ) as mock_req:
            await client.send_message(
                chat_id=123,
                text="text",
                reply_markup=markup,
            )

            call_args = mock_req.call_args
            assert call_args[0][0] == "sendMessage"
            assert "reply_markup" in call_args[0][1]
            assert (
                call_args[0][1]["reply_markup"]["inline_keyboard"][0][0]["text"]
                == "Btn"
            )

    @pytest.mark.asyncio
    async def test_edit_message_text_with_reply_markup(self) -> None:
        """Test edit_message_text handles reply_markup."""
        client = TelegramApiClient("token")

        from blacki.telegram.types import InlineKeyboardButton, InlineKeyboardMarkup

        markup = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Btn", callback_data="cb")]]
        )

        mock_result = {
            "message_id": 1,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": 123, "type": "private"},
        }
        with patch.object(
            client, "_request", AsyncMock(return_value=mock_result)
        ) as mock_req:
            await client.edit_message_text(
                chat_id=123,
                message_id=1,
                text="text",
                reply_markup=markup,
            )

            call_args = mock_req.call_args
            assert call_args[0][0] == "editMessageText"
            assert "reply_markup" in call_args[0][1]
            assert (
                call_args[0][1]["reply_markup"]["inline_keyboard"][0][0]["text"]
                == "Btn"
            )
        """Test send_photo returns Message on success."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "result": {
                "message_id": 42,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "photo": [
                    {
                        "file_id": "photo123",
                        "file_unique_id": "uniq123",
                        "width": 100,
                        "height": 100,
                    }
                ],
            },
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_ensure_client", AsyncMock(return_value=mock_http_client)
        ):
            result = await client.send_photo(
                chat_id=123,
                photo_bytes=b"image content",
                filename="test.png",
                caption="A test photo",
                message_thread_id=456,
            )

        assert isinstance(result, Message)
        assert result.message_id == 42
        assert result.chat.id == 123

    @pytest.mark.asyncio
    async def test_send_photo_minimal_params(self) -> None:
        """Test send_photo with only required params."""
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
            result = await client.send_photo(
                chat_id=123, photo_bytes=b"image", filename="test.jpg"
            )

        assert isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_send_photo_with_parse_mode(self) -> None:
        """Test send_photo with parse_mode."""
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
            await client.send_photo(
                chat_id=123,
                photo_bytes=b"image",
                filename="test.png",
                parse_mode=ParseMode.HTML,
            )

        call_args = mock_http_client.post.call_args
        assert call_args.kwargs["data"]["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_send_photo_http_error(self) -> None:
        """Test send_photo raises on HTTP error."""
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
            await client.send_photo(
                chat_id=123, photo_bytes=b"image", filename="test.png"
            )

    @pytest.mark.asyncio
    async def test_send_photo_http_error_no_json(self) -> None:
        """Test send_photo raises on HTTP error without JSON."""
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
            await client.send_photo(
                chat_id=123, photo_bytes=b"image", filename="test.png"
            )

    @pytest.mark.asyncio
    async def test_send_photo_api_not_ok(self) -> None:
        """Test send_photo raises when API returns ok=False."""
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
            await client.send_photo(
                chat_id=123, photo_bytes=b"image", filename="test.png"
            )

    @pytest.mark.asyncio
    async def test_send_photo_api_not_ok_with_retry_after(self) -> None:
        """Test send_photo raises with retry_after when rate-limited."""
        client = TelegramApiClient("token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": False,
            "description": "Too Many Requests",
            "error_code": 429,
            "parameters": {"retry_after": 30},
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(
                client, "_ensure_client", AsyncMock(return_value=mock_http_client)
            ),
            pytest.raises(TelegramApiError, match="Too Many Requests") as exc_info,
        ):
            await client.send_photo(
                chat_id=123, photo_bytes=b"image", filename="test.png"
            )

        assert exc_info.value.retry_after == 30
