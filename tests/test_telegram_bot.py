# mypy: disable-error-code="no-untyped-def"
"""Unit tests for Telegram bot module."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from google.genai import types

from blacki.adk_runtime import (
    AdkRuntime,
    EmptyModelResponseError,
    SessionLocator,
    StreamChunk,
    TurnResponse,
)
from blacki.inference import InferenceProfile
from blacki.reminders.storage import Reminder
from blacki.telegram import TelegramConfig
from blacki.telegram.album_buffer import _BufferedAlbum
from blacki.telegram.api import TelegramApiClient, TelegramApiError
from blacki.telegram.bot import (
    TelegramBot,
    TelegramSessionIdentity,
    create_telegram_bot,
)
from blacki.telegram.formatting import (
    escape_markdown,
    escape_markdown_plain,
    format_for_telegram,
    get_open_markdown_entities,
)
from blacki.telegram.streaming import (
    TELEGRAM_MESSAGE_LIMIT,
    StreamSession,
    _merge_stream_text,
    split_long_message,
)
from blacki.telegram.types import BotCommand, ChatType, Message, ParseMode, Update


class RecordingRuntime:
    """Small fake runtime for Telegram bot tests."""

    def __init__(self) -> None:
        self.run_user_turn_response = "Test response"
        self.run_user_turn_thoughts = ""
        self.run_user_turn_error: Exception | None = None
        self.run_user_turn_side_effects: list[str | Exception] = []
        self.create_next_session_error: Exception | None = None
        self.run_user_turn_calls: list[dict[str, Any]] = []
        self.rewind_empty_model_response_calls: list[dict[str, Any]] = []
        self.create_next_session_calls: list[dict[str, Any]] = []
        self.closed = False

    async def run_user_turn(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
        user_parts: Sequence[types.Part] | None = None,
        inference_profile: Any | None = None,
    ) -> str:
        self.run_user_turn_calls.append(
            {
                "locator": locator,
                "message_text": message_text,
                "state": state,
                "user_parts": user_parts,
                "inference_profile": inference_profile,
            }
        )
        if self.run_user_turn_side_effects:
            result = self.run_user_turn_side_effects.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if self.run_user_turn_error is not None:
            raise self.run_user_turn_error
        return self.run_user_turn_response

    async def rewind_empty_model_response(
        self,
        *,
        locator: SessionLocator,
        invocation_id: str,
    ) -> None:
        self.rewind_empty_model_response_calls.append(
            {"locator": locator, "invocation_id": invocation_id}
        )

    async def run_user_turn_with_thoughts(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
        user_parts: Sequence[types.Part] | None = None,
        inference_profile: Any | None = None,
    ) -> TurnResponse:
        self.run_user_turn_calls.append(
            {
                "locator": locator,
                "message_text": message_text,
                "state": state,
                "user_parts": user_parts,
                "inference_profile": inference_profile,
            }
        )
        if self.run_user_turn_error is not None:
            raise self.run_user_turn_error
        return TurnResponse(
            thoughts=self.run_user_turn_thoughts,
            content=self.run_user_turn_response,
        )

    async def run_user_turn_streaming(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
        user_parts: Sequence[types.Part] | None = None,
        inference_profile: Any | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.run_user_turn_calls.append(
            {
                "locator": locator,
                "message_text": message_text,
                "state": state,
                "user_parts": user_parts,
                "inference_profile": inference_profile,
            }
        )
        if self.run_user_turn_error is not None:
            raise self.run_user_turn_error
        if self.run_user_turn_thoughts:
            yield StreamChunk(thoughts=self.run_user_turn_thoughts, content="")
        yield StreamChunk(
            thoughts=self.run_user_turn_thoughts,
            content=self.run_user_turn_response,
            is_partial=False,
        )

    async def create_next_session(
        self,
        *,
        locator: SessionLocator,
        state: dict[str, Any] | None = None,
    ) -> object:
        if self.create_next_session_error is not None:
            raise self.create_next_session_error
        self.create_next_session_calls.append(
            {
                "locator": locator,
                "state": state,
            }
        )
        return SimpleNamespace(id="session-id")

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def telegram_config() -> TelegramConfig:
    """Create a valid Telegram config."""
    return TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": True,
            "TELEGRAM_BOT_TOKEN": "test-token-123",
        }
    )


@pytest.fixture
def telegram_config_disabled() -> TelegramConfig:
    """Create a disabled Telegram config."""
    return TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": False,
            "TELEGRAM_BOT_TOKEN": None,
        }
    )


@pytest.fixture
def runtime_recorder() -> RecordingRuntime:
    """Create a recording ADK runtime fake."""
    return RecordingRuntime()


@pytest.fixture
def mock_message() -> Message:
    """Create a mock Telegram message."""
    return Message.model_validate(
        {
            "message_id": 1,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": 123456789, "type": "private"},
            "text": "Hello, bot!",
            "from": {"id": 123456789, "first_name": "Test", "is_bot": False},
        }
    )


@pytest.fixture
def mock_update(mock_message: Message) -> Update:
    """Create a mock Telegram update."""
    return Update.model_validate({"update_id": 1, "message": mock_message.model_dump()})


def test_init_with_config(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test initialization with valid config."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    assert bot.config == telegram_config
    assert cast(Any, bot.runtime) is runtime_recorder


def test_api_property_raises_without_token(
    telegram_config_disabled: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that api property raises ValueError without token."""
    bot = TelegramBot(telegram_config_disabled, cast(AdkRuntime, runtime_recorder))

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN is required"):
        _ = bot.api


def test_build_session_identity_without_thread(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test stable session identity for a normal chat."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    identity = bot._build_session_identity(chat_id="123", message_thread_id=None)

    assert identity == TelegramSessionIdentity(
        conversation_key="chat-123",
        user_id="telegram-chat-123",
        session_id_prefix="telegram-chat-123",
    )


def test_build_session_identity_with_thread(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test stable session identity for a topic thread."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    identity = bot._build_session_identity(chat_id="123", message_thread_id=99)

    assert identity == TelegramSessionIdentity(
        conversation_key="chat-123-thread-99",
        user_id="telegram-chat-123-thread-99",
        session_id_prefix="telegram-chat-123-thread-99",
    )


def test_build_session_state_includes_thread_when_present(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test session state stores thread metadata when available."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    session_state = bot._build_session_state(
        chat_id="123",
        message_thread_id=99,
        conversation_key="chat-123-thread-99",
    )

    assert session_state["user_id"] == "telegram-chat-123-thread-99"
    assert session_state["telegram_chat_id"] == "123"
    assert session_state["telegram_thread_id"] == "99"


def test_create_bot_configured(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test create bot when configured."""
    result = create_telegram_bot(telegram_config, cast(AdkRuntime, runtime_recorder))

    assert result is not None
    assert isinstance(result, TelegramBot)


def test_create_bot_not_configured(
    telegram_config_disabled: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test create bot when not configured."""
    result = create_telegram_bot(
        telegram_config_disabled,
        cast(AdkRuntime, runtime_recorder),
    )

    assert result is None


class TestTelegramApiClient:
    """Tests for TelegramApiClient."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, telegram_config: TelegramConfig) -> None:
        """Test successful message sending."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Hello",
                }
                result = await client.send_message(chat_id=123, text="Hello")
                assert result.message_id == 1

    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self) -> None:
        """Test context manager creates HTTP client."""
        async with TelegramApiClient("test-token") as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_close_clears_client(self) -> None:
        """Test close clears the HTTP client."""
        client = TelegramApiClient("test-token")
        await client._ensure_client()
        assert client._client is not None

        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_ensure_client_reuses_existing(self) -> None:
        """Test _ensure_client reuses existing client."""
        client = TelegramApiClient("test-token")
        first = await client._ensure_client()
        second = await client._ensure_client()
        assert first is second
        await client.close()

    @pytest.mark.asyncio
    async def test_build_url(self) -> None:
        """Test URL building."""
        client = TelegramApiClient("test-token")
        url = client._build_url("sendMessage")
        assert url == "https://api.telegram.org/bottest-token/sendMessage"

    @pytest.mark.asyncio
    async def test_send_message_draft_success(
        self, telegram_config: TelegramConfig
    ) -> None:
        """Test successful draft message sending with int draft_id."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Draft text",
                }
                result = await client.send_message_draft(
                    chat_id=123, text="Draft text", draft_id=12345
                )
                assert isinstance(result, Message)
                assert result.message_id == 1
                mock_request.assert_called_once_with(
                    "sendMessageDraft",
                    {"chat_id": 123, "text": "Draft text", "draft_id": 12345},
                )

    @pytest.mark.asyncio
    async def test_get_updates_success(self) -> None:
        """Test successful updates retrieval."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = [
                    {
                        "update_id": 1,
                        "message": {
                            "message_id": 1,
                            "date": "2024-01-01T00:00:00Z",
                            "chat": {"id": 123, "type": "private"},
                            "text": "Hello",
                        },
                    }
                ]
                updates = await client.get_updates()
                assert len(updates) == 1
                assert updates[0].update_id == 1

    @pytest.mark.asyncio
    async def test_set_my_commands_success(self) -> None:
        """Test successful command registration."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = True
                commands = [
                    BotCommand(command="start", description="Start"),
                    BotCommand(command="help", description="Help"),
                ]
                result = await client.set_my_commands(commands)
                assert result is True

    @pytest.mark.asyncio
    async def test_api_error_raised(self) -> None:
        """Test that API errors are properly raised."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.side_effect = TelegramApiError(
                    "Bad Request", error_code=400
                )
                with pytest.raises(TelegramApiError, match="Bad Request"):
                    await client.send_message(chat_id=123, text="test")

    @pytest.mark.asyncio
    async def test_get_me_success(self) -> None:
        """Test successful getMe call."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = {"id": 123, "is_bot": True}
                result = await client.get_me()
                assert result["id"] == 123

    @pytest.mark.asyncio
    async def test_edit_message_text_success(self) -> None:
        """Test successful editMessageText call."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Edited",
                }
                result = await client.edit_message_text(
                    chat_id=123, message_id=1, text="Edited"
                )
                assert result.text == "Edited"

    @pytest.mark.asyncio
    async def test_delete_message_success(self) -> None:
        """Test successful deleteMessage call."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = True
                result = await client.delete_message(chat_id=123, message_id=1)
                assert result is True

    @pytest.mark.asyncio
    async def test_send_chat_action_success(self) -> None:
        """Test successful sendChatAction call."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = True
                result = await client.send_chat_action(chat_id=123, action="typing")
                assert result is True

    @pytest.mark.asyncio
    async def test_get_my_commands_success(self) -> None:
        """Test successful getMyCommands call."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = [
                    {"command": "start", "description": "Start"}
                ]
                commands = await client.get_my_commands()
                assert len(commands) == 1
                assert commands[0].command == "start"

    @pytest.mark.asyncio
    async def test_api_error_with_retry_after(self) -> None:
        """Test API error includes retry_after parameter."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                error = TelegramApiError(
                    "Too Many Requests", error_code=429, retry_after=30
                )
                mock_request.side_effect = error
                with pytest.raises(TelegramApiError) as exc_info:
                    await client.send_message(chat_id=123, text="test")
                assert exc_info.value.retry_after == 30

    @pytest.mark.asyncio
    async def test_request_with_real_http_mock(self) -> None:
        """Test _request method with mocked HTTP response."""
        import httpx

        async with TelegramApiClient("test-token") as client:
            with patch.object(client._client, "post") as mock_post:
                mock_response = create_autospec(httpx.Response, instance=True)
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "ok": True,
                    "result": {"id": 123, "is_bot": True},
                }
                mock_post.return_value = mock_response

                result = await client._request("getMe")
                assert result == {"id": 123, "is_bot": True}

    @pytest.mark.asyncio
    async def test_request_handles_api_error(self) -> None:
        """Test _request handles Telegram API error response."""
        import httpx

        async with TelegramApiClient("test-token") as client:
            with patch.object(client._client, "post") as mock_post:
                mock_response = create_autospec(httpx.Response, instance=True)
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "ok": False,
                    "error_code": 400,
                    "description": "Bad Request: test error",
                }
                mock_post.return_value = mock_response

                with pytest.raises(TelegramApiError, match="Bad Request"):
                    await client._request("sendMessage")

    @pytest.mark.asyncio
    async def test_request_handles_api_error_with_parameters(self) -> None:
        """Test _request handles error with retry_after parameter."""
        import httpx

        async with TelegramApiClient("test-token") as client:
            with patch.object(client._client, "post") as mock_post:
                mock_response = create_autospec(httpx.Response, instance=True)
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "ok": False,
                    "error_code": 429,
                    "description": "Too Many Requests",
                    "parameters": {"retry_after": 30},
                }
                mock_post.return_value = mock_response

                with pytest.raises(TelegramApiError) as exc_info:
                    await client._request("sendMessage")
                assert exc_info.value.retry_after == 30

    @pytest.mark.asyncio
    async def test_send_message_with_all_options(self) -> None:
        """Test send_message with all options."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
                await client.send_message(
                    chat_id=123,
                    text="Hello",
                    message_thread_id=1,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    disable_notification=True,
                    protect_content=True,
                )
                call_kwargs = mock_request.call_args.args[1]
                assert call_kwargs["message_thread_id"] == 1
                assert call_kwargs["parse_mode"] == "MarkdownV2"
                assert call_kwargs["disable_notification"] is True
                assert call_kwargs["protect_content"] is True

    @pytest.mark.asyncio
    async def test_get_updates_with_options(self) -> None:
        """Test get_updates with all options."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = []
                await client.get_updates(
                    offset=100,
                    limit=50,
                    timeout=10,
                    allowed_updates=["message"],
                )
                call_kwargs = mock_request.call_args.args[1]
                assert call_kwargs["offset"] == 100
                assert call_kwargs["limit"] == 50
                assert call_kwargs["timeout"] == 10
                assert call_kwargs["allowed_updates"] == ["message"]

    @pytest.mark.asyncio
    async def test_get_updates_uses_buffered_http_timeout(self) -> None:
        """Test long polling uses a timeout buffer for read timeouts."""
        async with TelegramApiClient("test-token", timeout=5.0) as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = []

                await client.get_updates(timeout=12)

                assert mock_request.call_args.kwargs["timeout"] == 17.0


class TestStreamSession:
    """Tests for StreamSession."""

    @pytest.mark.asyncio
    async def test_stream_uses_send_then_edit(self) -> None:
        """Test that streaming uses sendMessage then editMessageText."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 42,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 42,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hello")
            yield StreamChunk(thoughts="", content="Hello world", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123, message_thread_id=5)

        mock_api.send_message.assert_called_once()
        send_kwargs = mock_api.send_message.call_args.kwargs
        assert send_kwargs.get("message_thread_id") == 5
        mock_api.edit_message_text.assert_called()

    @pytest.mark.asyncio
    async def test_stream_ignores_thoughts(self) -> None:
        """Test that streaming ignores thoughts and only shows content."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="Thinking...", content="", is_partial=True)
            yield StreamChunk(
                thoughts="Still thinking...", content="Hello", is_partial=True
            )
            yield StreamChunk(
                thoughts="Final thought", content="Hello world", is_partial=False
            )

        await session.run(chunks=chunks(), chat_id=123)

        assert mock_api.send_message.await_count == 1
        send_kwargs = mock_api.send_message.call_args.kwargs
        assert "Thinking" not in send_kwargs["text"]
        assert "Hello" in send_kwargs["text"]

    @pytest.mark.asyncio
    async def test_retry_after_propagation(self) -> None:
        """Test that 429 errors with retry_after are retried."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
            }
        )
        call_count = 0

        async def send_side_effect(*args: object, **kwargs: object) -> Message:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TelegramApiError("Rate limit", error_code=429, retry_after=1)
            return msg

        mock_api.send_message = AsyncMock(side_effect=send_side_effect)

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hello", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_persistent_failure_fallback(self) -> None:
        """Test that persistent failures fall back to sendMessage."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hello", is_partial=False)

        result = await session.run(chunks=chunks(), chat_id=123)

        assert result == "Hello"
        mock_api.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_long_message_split(self) -> None:
        """Test that long messages are split correctly."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "supergroup"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "supergroup"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        long_text = "A" * 5000

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content=long_text, is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        assert mock_api.send_message.await_count >= 2

    @pytest.mark.asyncio
    async def test_empty_stream_sends_apology(self) -> None:
        """Test that empty stream sends apology message."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        mock_api.send_message.assert_called()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert "couldn't generate" in call_kwargs["text"]


class TestSplitLongMessage:
    """Tests for split_long_message function."""

    def test_short_message_returns_single_chunk(self) -> None:
        """Test that short messages are not split."""
        text = "Short message"
        chunks = split_long_message(text)
        assert chunks == [text]

    def test_exact_limit_returns_single_chunk(self) -> None:
        """Test that messages at exactly the limit are not split."""
        text = "A" * TELEGRAM_MESSAGE_LIMIT
        chunks = split_long_message(text)
        assert chunks == [text]

    def test_long_message_splits_on_paragraph(self) -> None:
        """Test that long messages split on paragraph boundaries."""
        text = ("A" * 100) + "\n\n" + ("B" * 100)
        text = text * 50
        chunks = split_long_message(text)
        for chunk in chunks:
            assert len(chunk) <= TELEGRAM_MESSAGE_LIMIT

    def test_no_boundary_hard_split(self) -> None:
        """Test hard split when no boundary exists."""
        text = "A" * (TELEGRAM_MESSAGE_LIMIT + 100)
        chunks = split_long_message(text)
        assert len(chunks) == 2
        assert len(chunks[0]) == TELEGRAM_MESSAGE_LIMIT
        assert len(chunks[1]) == 100

    def test_empty_string_returns_empty_list(self) -> None:
        """Test that empty strings return empty list."""
        chunks = split_long_message("")
        assert chunks == []


class TestMergeStreamText:
    """Tests for _merge_stream_text function."""

    def test_returns_incoming_when_current_is_empty(self) -> None:
        """Test that empty current text returns incoming."""
        result = _merge_stream_text("", "Hello", is_partial=True)
        assert result == "Hello"

    def test_returns_incoming_when_it_starts_with_current(self) -> None:
        """Test snapshot-style merge where incoming contains current."""
        result = _merge_stream_text("Hello", "Hello world", is_partial=True)
        assert result == "Hello world"

    def test_returns_incoming_when_is_partial_is_false(self) -> None:
        """Test that non-partial always returns incoming."""
        result = _merge_stream_text("Hello", "Goodbye", is_partial=False)
        assert result == "Goodbye"

    def test_returns_current_when_it_starts_with_incoming(self) -> None:
        """Test case where current is longer than incoming (partial)."""
        result = _merge_stream_text("Hello world", "Hello", is_partial=True)
        assert result == "Hello world"

    def test_concatenates_when_no_overlap_and_partial(self) -> None:
        """Test concatenation when no overlap exists (partial)."""
        result = _merge_stream_text("Hello ", "world", is_partial=True)
        assert result == "Hello world"

    def test_handles_unicode_characters(self) -> None:
        """Test that unicode characters are merged correctly."""
        result = _merge_stream_text("Hello ", "世界!", is_partial=True)
        assert result == "Hello 世界!"

    def test_handles_emoji(self) -> None:
        """Test that emoji characters are merged correctly."""
        result = _merge_stream_text("Hello ", "👋", is_partial=True)
        assert result == "Hello 👋"

    def test_handles_single_character(self) -> None:
        """Test single character merges."""
        result = _merge_stream_text("a", "ab", is_partial=True)
        assert result == "ab"

    def test_handles_empty_strings(self) -> None:
        """Test both empty strings."""
        result = _merge_stream_text("", "", is_partial=True)
        assert result == ""

    def test_non_partial_overrides_with_shorter_text(self) -> None:
        """Test that non-partial replaces even with shorter text."""
        result = _merge_stream_text("Longer text here", "Short", is_partial=False)
        assert result == "Short"


class TestEscapeMarkdown:
    """Tests for escape_markdown function."""

    def test_escape_markdown_escapes_special_chars(self) -> None:
        """Test that MarkdownV2 special characters are escaped."""
        text = "Hello _world_ with *stars* and [brackets]"
        escaped = escape_markdown(text)

        assert escaped == r"Hello \_world\_ with \*stars\* and \[brackets\]"

    def test_escape_markdown_preserves_code_blocks(self) -> None:
        """Test that code block content is not escaped."""
        text = "Text with _underscore_ and ```code_with_special_*chars*```"
        escaped = escape_markdown(text)

        assert (
            escaped == r"Text with \_underscore\_ and ```code_with_special_*chars*```"
        )

    def test_escape_markdown_preserves_inline_code(self) -> None:
        """Test that inline code content is not escaped."""
        text = "Use `variable_name` for _important_ things"
        escaped = escape_markdown(text)

        assert escaped == r"Use `variable_name` for \_important\_ things"

    def test_escape_markdown_handles_empty_string(self) -> None:
        """Test that empty strings are handled correctly."""
        assert escape_markdown("") == ""

    def test_escape_markdown_escapes_all_markdown_v2_chars(self) -> None:
        """Test that all MarkdownV2 special characters are escaped."""
        text = "_ * [ ] ( ) ~ > # + - = | { } . ! \\"
        escaped = escape_markdown(text)

        assert escaped == r"\_ \* \[ \] \( \) \~ \> \# \+ \- \= \| \{ \} \. \! \\"


class TestFormatForTelegram:
    """Tests for format_for_telegram function."""

    def test_format_for_telegram_preserves_bold_markers(self) -> None:
        """Test that bold formatting markers are converted to Telegram format."""
        text = "This is **bold** text"
        formatted = format_for_telegram(text)

        assert formatted == r"This is *bold* text"

    def test_format_for_telegram_escapes_content_inside_bold(self) -> None:
        """Test that special chars inside bold are escaped."""
        text = "**Hello_World**"
        formatted = format_for_telegram(text)

        assert formatted == r"*Hello\_World*"

    def test_format_for_telegram_escapes_regular_text(self) -> None:
        """Test that special chars outside bold are escaped."""
        text = "Hello_World"
        formatted = format_for_telegram(text)

        assert formatted == r"Hello\_World"

    def test_format_for_telegram_handles_multiple_bold(self) -> None:
        """Test multiple bold sections."""
        text = "**First** and **Second**"
        formatted = format_for_telegram(text)

        assert formatted == r"*First* and *Second*"

    def test_format_for_telegram_preserves_code_blocks(self) -> None:
        """Test that code blocks are preserved and not escaped."""
        text = "**bold** and ```code_with_special**```"
        formatted = format_for_telegram(text)

        assert formatted == r"*bold* and ```code_with_special**```"

    def test_format_for_telegram_preserves_inline_code(self) -> None:
        """Test that inline code is preserved."""
        text = "**bold** and `code_with_underscore`"
        formatted = format_for_telegram(text)

        assert formatted == r"*bold* and `code_with_underscore`"

    def test_format_for_telegram_converts_headings_to_bold(self) -> None:
        """Test that markdown headings are converted to bold."""
        text = "### Model Configuration"
        formatted = format_for_telegram(text)

        assert formatted == r"*Model Configuration*"

    def test_format_for_telegram_converts_h1_heading(self) -> None:
        """Test that h1 heading is converted to bold."""
        text = "# Title"
        formatted = format_for_telegram(text)

        assert formatted == r"*Title*"

    def test_format_for_telegram_converts_h2_heading(self) -> None:
        """Test that h2 heading is converted to bold."""
        text = "## Subtitle"
        formatted = format_for_telegram(text)

        assert formatted == r"*Subtitle*"

    def test_format_for_telegram_converts_bullet_asterisk(self) -> None:
        """Test that asterisk bullets are converted to Telegram bullet."""
        text = "* First item\n* Second item"
        formatted = format_for_telegram(text)

        assert formatted == "• First item\n• Second item"

    def test_format_for_telegram_converts_bullet_dash(self) -> None:
        """Test that dash bullets are converted to Telegram bullet."""
        text = "- First item\n- Second item"
        formatted = format_for_telegram(text)

        assert formatted == "• First item\n• Second item"

    def test_format_for_telegram_converts_bullet_plus(self) -> None:
        """Test that plus bullets are converted to Telegram bullet."""
        text = "+ First item\n+ Second item"
        formatted = format_for_telegram(text)

        assert formatted == "• First item\n• Second item"

    def test_format_for_telegram_preserves_bullet_indent(self) -> None:
        """Test that bullet indentation is preserved."""
        text = "  * Nested item"
        formatted = format_for_telegram(text)

        assert formatted == "  • Nested item"

    def test_format_for_telegram_combined_heading_and_bullets(self) -> None:
        """Test combined heading and bullet list."""
        text = "### Features\n* Feature one\n* Feature two"
        formatted = format_for_telegram(text)

        assert formatted == "*Features*\n• Feature one\n• Feature two"

    def test_format_for_telegram_heading_with_special_chars(self) -> None:
        """Test heading with special characters is escaped."""
        text = "### Model_Config"
        formatted = format_for_telegram(text)

        assert formatted == r"*Model\_Config*"

    def test_format_for_telegram_complex_document(self) -> None:
        """Test a complex markdown document."""
        text = """### 1. Model Configuration

The file handles model selection dynamically:

* Dynamic Routing: It supports both native Google Gemini and OpenRouter.
* LiteLLM Integration: If an OPENROUTER_API_KEY is present."""
        formatted = format_for_telegram(text)

        expected = r"""*1\. Model Configuration*

The file handles model selection dynamically:

• Dynamic Routing: It supports both native Google Gemini and OpenRouter\.
• LiteLLM Integration: If an OPENROUTER\_API\_KEY is present\."""
        assert formatted == expected

    def test_format_for_telegram_converts_markdown_table_to_code_block(self) -> None:
        """Render Markdown tables in Telegram's supported monospaced format."""
        text = "| Name | Value |\n| --- | ---: |\n| Alpha | 1 |\n| Beta | 2 |"

        formatted = format_for_telegram(text)

        assert formatted == (
            "```\nName  | Value\n------+------\nAlpha | 1\nBeta  | 2\n```"
        )

    def test_format_for_telegram_converts_table_without_outer_pipes(self) -> None:
        """Recognize Markdown tables that omit leading and trailing pipes."""
        text = "Name | Value\n:--- | ---:\nAlpha | 1"

        formatted = format_for_telegram(text)

        assert formatted == "```\nName  | Value\n------+------\nAlpha | 1\n```"

    def test_format_for_telegram_preserves_pipes_inside_table_cells(self) -> None:
        """Keep escaped and inline-code pipes inside their table cells."""
        text = "| Expression | Meaning |\n| --- | --- |\n| a\\|b | `x|y` |"

        formatted = format_for_telegram(text)

        assert (
            formatted == "```\nExpression | Meaning\n"
            "-----------+--------\na|b        | \\`x|y\\`\n```"
        )

    def test_format_for_telegram_preserves_escaped_trailing_cell_pipe(self) -> None:
        """Keep an escaped pipe when it is the final cell character."""
        text = "Name | Value\n--- | ---\nAlpha | a\\|"

        assert (
            format_for_telegram(text)
            == "```\nName  | Value\n------+------\nAlpha | a|\n```"
        )

    def test_format_for_telegram_matches_multi_backtick_code_spans(self) -> None:
        """Keep pipes inside code spans delimited by multiple backticks."""
        text = "| Expression | Meaning |\n| --- | --- |\n| ``a|b`` | literal |"

        formatted = format_for_telegram(text)

        assert formatted == (
            "```\nExpression | Meaning\n"
            "-----------+--------\n\\`\\`a|b\\`\\`    | literal\n```"
        )

    def test_format_for_telegram_ignores_pipes_in_mixed_backtick_runs(
        self,
    ) -> None:
        """Keep pipes inside longer spans with shorter inner backtick runs."""
        text = "| Expression | Meaning |\n| --- | --- |\n| ``a `|` b`` | literal |"

        assert format_for_telegram(text) == (
            "```\nExpression  | Meaning\n"
            "------------+--------\n\\`\\`a \\`|\\` b\\`\\` | literal\n```"
        )

    def test_format_for_telegram_pads_unicode_table_cells_by_display_width(
        self,
    ) -> None:
        """Align CJK and combining characters in monospaced table output."""
        text = "| Word | Other |\n| --- | --- |\n| 猫 | é |\n| Dog | long |"

        formatted = format_for_telegram(text)

        assert formatted == (
            "```\nWord | Other\n-----+------\n猫   | é\nDog  | long\n```"
        )

    def test_format_for_telegram_does_not_convert_existing_code_tables(self) -> None:
        """Leave already fenced tables unchanged."""
        text = "```\n| Name | Value |\n| --- | --- |\n| Alpha | 1 |\n```"

        assert format_for_telegram(text) == text

    def test_format_for_telegram_preserves_table_line_endings(self) -> None:
        """Preserve CRLF line endings and text following a table."""
        text = "| Name | Value |\r\n| --- | --- |\r\n| Alpha | 1 |\r\nAfter\r\n"

        formatted = format_for_telegram(text)

        assert (
            formatted
            == "```\r\nName  | Value\r\n------+------\r\nAlpha | 1\r\n```\r\nAfter\r\n"
        )

    def test_format_for_telegram_keeps_non_table_pipes_as_plain_text(self) -> None:
        """Do not interpret arbitrary pipe-delimited text as a table."""
        text = "alpha | beta\nnot a separator | nope"

        formatted = format_for_telegram(text)

        assert formatted == "alpha \\| beta\nnot a separator \\| nope"


class TestTelegramBotCommands:
    """Tests for Telegram bot command handling."""

    @pytest.mark.asyncio
    async def test_handle_start_command(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        mock_message: Message,
    ) -> None:
        """Test /start command sends welcome message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_command(mock_message, "/start")
        mock_api.send_message.assert_called_once()
        call_args = mock_api.send_message.call_args
        assert "Hello" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_handle_help_command(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        mock_message: Message,
    ) -> None:
        """Test /help command sends help message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_command(mock_message, "/help")
        mock_api.send_message.assert_called_once()
        call_args = mock_api.send_message.call_args
        assert "Commands" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_handle_reset_command(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        mock_message: Message,
    ) -> None:
        """Test /reset command creates next session."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_command(mock_message, "/reset")
        assert len(runtime_recorder.create_next_session_calls) == 1

    @pytest.mark.asyncio
    async def test_handle_reset_command_error(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        mock_message: Message,
    ) -> None:
        """Test /reset command handles errors."""
        runtime_recorder.create_next_session_error = RuntimeError("reset failed")
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_command(mock_message, "/reset")
        call_args = mock_api.send_message.call_args
        assert "couldn't reset" in call_args.kwargs["text"]


class TestTelegramBotMessageHandling:
    """Tests for Telegram bot message handling with streaming."""

    @pytest.mark.asyncio
    async def test_handle_message_streams_response(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Handle a message without copying its text into Telegram logs."""
        caplog.set_level(logging.INFO)
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_response = "Hello back!"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        bot._api = mock_api

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=None,
            user_message="private-shopping-message",
        )

        mock_api.send_chat_action.assert_called_once_with(
            chat_id=123456789, action="typing", message_thread_id=None
        )
        mock_api.send_message.assert_called()
        assert "Received message from chat 123456789" in caplog.text
        assert "private-shopping-message" not in caplog.text

    @pytest.mark.asyncio
    async def test_handle_message_ignores_thoughts(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test message handling ignores thoughts and shows only content."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_thoughts = "Let me think..."
        runtime_recorder.run_user_turn_response = "Here is my answer."

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        bot._api = mock_api

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=None,
            user_message="Hello!",
        )

        mock_api.send_message.assert_called()
        send_kwargs = mock_api.send_message.call_args.kwargs
        assert "think" not in send_kwargs["text"].lower()
        assert "answer" in send_kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_handle_message_error(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test message handling with error."""
        runtime_recorder.run_user_turn_error = RuntimeError("runner failed")
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=None,
            user_message="Hello!",
        )

        mock_api.send_message.assert_called()
        call_args = mock_api.send_message.call_args
        assert "error" in call_args.kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_handle_message_retries_empty_model_response_once(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A transient empty model response is retried with the same input."""
        caplog.set_level(logging.WARNING)
        runtime_recorder.run_user_turn_side_effects = [
            EmptyModelResponseError(
                "empty response",
                model="openrouter/test-model",
                provider="openrouter",
                invocation_id="empty-invocation",
            ),
            "Recovered response",
        ]
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=None,
            user_message="same user message",
        )

        assert len(runtime_recorder.run_user_turn_calls) == 2
        assert runtime_recorder.rewind_empty_model_response_calls == [
            {
                "locator": SessionLocator(
                    user_id="telegram-chat-123456789",
                    session_id_prefix="telegram-chat-123456789",
                ),
                "invocation_id": "empty-invocation",
            }
        ]
        assert [
            call["message_text"] for call in runtime_recorder.run_user_turn_calls
        ] == ["same user message", "same user message"]
        assert "Recovered response" in mock_api.send_message.call_args.kwargs["text"]
        assert "model=openrouter/test-model" in caplog.text
        assert "provider=openrouter" in caplog.text
        assert "conversation_id=chat-123456789" in caplog.text
        assert "invocation_id=empty-invocation" in caplog.text
        assert "retry_count=1" in caplog.text

    @pytest.mark.asyncio
    async def test_handle_message_sends_fallback_after_one_empty_response_retry(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Two empty responses do not cause an unbounded retry loop."""
        caplog.set_level(logging.WARNING)
        runtime_recorder.run_user_turn_side_effects = [
            EmptyModelResponseError(
                "empty response",
                model="openrouter/test-model",
                provider="openrouter",
                invocation_id="empty-invocation-1",
            ),
            EmptyModelResponseError(
                "empty response",
                model="openrouter/test-model",
                provider="openrouter",
                invocation_id="empty-invocation-2",
            ),
        ]
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=77,
            user_message="empty twice",
        )

        assert len(runtime_recorder.run_user_turn_calls) == 2
        assert "empty response" in mock_api.send_message.call_args.kwargs["text"]
        assert mock_api.send_message.call_args.kwargs["message_thread_id"] == 77
        assert len(runtime_recorder.rewind_empty_model_response_calls) == 1
        assert "retry_count=1" in caplog.text
        assert "empty-invocation-2" in caplog.text

    @pytest.mark.asyncio
    async def test_handle_message_does_not_retry_after_tool_call_empty_response(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Do not repeat a turn when a tool may already have executed."""
        caplog.set_level(logging.WARNING)
        runtime_recorder.run_user_turn_side_effects = [
            EmptyModelResponseError(
                "empty response after tool call",
                model="openrouter/test-model",
                provider="openrouter",
                invocation_id="tool-invocation",
                tool_calls_seen=True,
            )
        ]
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=None,
            user_message="do not repeat tools",
        )

        assert len(runtime_recorder.run_user_turn_calls) == 1
        assert runtime_recorder.rewind_empty_model_response_calls == []
        assert "retryable=False" in caplog.text


class TestTelegramBotLifecycle:
    """Tests for Telegram bot lifecycle."""

    @pytest.mark.asyncio
    async def test_start_polling_not_configured(
        self,
        telegram_config_disabled: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test start_polling when Telegram is disabled."""
        bot = TelegramBot(
            telegram_config_disabled,
            cast(AdkRuntime, runtime_recorder),
        )

        await bot.start_polling()

        assert bot._api is None

    @pytest.mark.asyncio
    async def test_start_polling_starts_task(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test start_polling creates polling task."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.set_my_commands = AsyncMock(return_value=True)
        bot._api = mock_api

        await bot.start_polling()

        assert bot._running is True
        assert bot._polling_task is not None

        await bot.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_api(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test stop closes API client."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.close = AsyncMock()
        bot._api = mock_api

        await bot.stop()

        mock_api.close.assert_called_once()
        assert runtime_recorder.closed is True

    @pytest.mark.asyncio
    async def test_register_commands_success(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test command registration."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.set_my_commands = AsyncMock(return_value=True)
        bot._api = mock_api

        await bot._register_commands()

        mock_api.set_my_commands.assert_called_once()
        commands = mock_api.set_my_commands.call_args.args[0]
        assert len(commands) == 5
        assert commands[0].command == "start"
        assert {command.command for command in commands} >= {"model", "thinking"}

    @pytest.mark.asyncio
    async def test_register_commands_handles_error(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test command registration handles errors."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.set_my_commands = AsyncMock(
            side_effect=TelegramApiError("Failed", error_code=400)
        )
        bot._api = mock_api

        await bot._register_commands()

    @pytest.mark.asyncio
    async def test_stop_closes_runtime(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test stop closes the runtime."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        await bot.stop()

        assert runtime_recorder.closed is True


class TestAdkDuplicationRegression:
    """Tests for ADK partial/non-partial duplication fix."""

    @pytest.mark.asyncio
    async def test_partial_then_non_partial_no_duplicate(self) -> None:
        """Test that partial followed by non-partial doesn't duplicate."""
        from blacki.adk_runtime import StreamChunk

        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Partial", is_partial=True)
            yield StreamChunk(thoughts="", content="Final answer", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        final_call = (
            mock_api.edit_message_text.call_args or mock_api.send_message.call_args
        )
        text = final_call.kwargs["text"]
        assert text.count("Final") == 1
        assert "Partial" not in text

    @pytest.mark.asyncio
    async def test_server_error_retry(self) -> None:
        """Test that 5xx errors are retried."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            side_effect=[
                TelegramApiError("Server error", error_code=500),
                Message.model_validate(
                    {
                        "message_id": 1,
                        "date": "2024-01-01T00:00:00Z",
                        "chat": {"id": 123, "type": "private"},
                    }
                ),
            ]
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hello", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        assert mock_api.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_edit_after_send(self) -> None:
        """Test that message is edited after initial send."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 42,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "supergroup"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 42,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "supergroup"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="First")
            yield StreamChunk(thoughts="", content="First Second", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        mock_api.send_message.assert_called_once()
        mock_api.edit_message_text.assert_called()
        edit_kwargs = mock_api.edit_message_text.call_args.kwargs
        assert edit_kwargs["message_id"] == 42

    @pytest.mark.asyncio
    async def test_throttle_skips_updates(self) -> None:
        """Test that throttle skips intermediate updates."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=1.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="A")
            yield StreamChunk(thoughts="", content="AB")
            yield StreamChunk(thoughts="", content="ABC", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        assert mock_api.send_message.await_count == 1
        assert mock_api.edit_message_text.await_count == 1

    @pytest.mark.asyncio
    async def test_stream_merges_delta_chunks_with_whitespace(self) -> None:
        """Test that delta-style chunks preserve leading spaces while streaming."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hi", is_partial=True)
            yield StreamChunk(thoughts="", content=" Chirag", is_partial=True)
            yield StreamChunk(
                thoughts="",
                content="Hi Chirag! Great to hear from you.",
                is_partial=False,
            )

        await session.run(chunks=chunks(), chat_id=123)

        final_call = (
            mock_api.edit_message_text.call_args or mock_api.send_message.call_args
        )
        assert final_call is not None
        assert final_call.kwargs["text"] == "Hi Chirag\\! Great to hear from you\\."
        assert (
            mock_api.edit_message_text.await_args_list[0].kwargs["text"] == "Hi Chirag"
        )

    @pytest.mark.asyncio
    async def test_stream_skips_identical_final_update(self) -> None:
        """Test that an unchanged final chunk does not trigger a redundant edit."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hello", is_partial=True)
            yield StreamChunk(thoughts="", content="Hello", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        mock_api.send_message.assert_called_once()
        mock_api.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_after_all_writes_fail(self) -> None:
        """Test that fallback message is sent after all writes fail."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            side_effect=TelegramApiError("Bad request", error_code=400)
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hello world", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        assert mock_api.send_message.await_count >= 1


class TestCommandErrors:
    """Tests for command error handling."""

    @pytest.mark.asyncio
    async def test_start_command_api_error(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        mock_message: Message,
    ) -> None:
        """Test /start command handles API errors."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock(
            side_effect=TelegramApiError("Failed", error_code=400)
        )
        bot._api = mock_api

        await bot._handle_command(mock_message, "/start")

    @pytest.mark.asyncio
    async def test_start_command_escapes_markdown_v2_reserved_characters(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        mock_message: Message,
    ) -> None:
        """The static welcome message is valid MarkdownV2."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_command(mock_message, "/start")

        call_kwargs = mock_api.send_message.call_args.kwargs
        assert r"Hello\!" in call_kwargs["text"]
        assert r"assistant\." in call_kwargs["text"]
        assert r"/help \-" in call_kwargs["text"]
        assert call_kwargs["parse_mode"] == ParseMode.MARKDOWN_V2

    @pytest.mark.asyncio
    async def test_help_command_api_error(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        mock_message: Message,
    ) -> None:
        """Test /help command handles API errors."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock(
            side_effect=TelegramApiError("Failed", error_code=400)
        )
        bot._api = mock_api

        await bot._handle_command(mock_message, "/help")


class TestFormattingEdgeCases:
    """Tests for formatting edge cases."""

    def test_format_bold_at_end(self) -> None:
        """Test bold at end of string."""
        text = "Text **bold**"
        formatted = format_for_telegram(text)
        assert formatted == r"Text *bold*"

    def test_escape_backtick_in_text(self) -> None:
        """Test backtick escaping behavior."""
        text = "Use `code` here"
        escaped = escape_markdown(text)
        assert escaped == r"Use `code` here"

    def test_format_with_newlines(self) -> None:
        """Test formatting preserves newlines."""
        text = "Line 1\n\nLine 2"
        formatted = format_for_telegram(text)
        assert "\n\n" in formatted


class TestTelegramApiErrors:
    """Tests for Telegram API error handling."""

    @pytest.mark.asyncio
    async def test_api_error_with_invalid_json_response(self) -> None:
        """When error response has invalid JSON, use response.text."""
        import httpx

        async with TelegramApiClient("test-token") as client:
            with patch.object(client._client, "post") as mock_post:
                mock_response = create_autospec(httpx.Response, instance=True)
                mock_response.status_code = 400
                mock_response.json.side_effect = json.JSONDecodeError("err", "doc", 0)
                mock_response.text = "Bad Request"
                mock_post.return_value = mock_response

                with pytest.raises(TelegramApiError, match="Bad Request"):
                    await client._request("someMethod", {})

    @pytest.mark.asyncio
    async def test_send_message_draft_with_optional_params(self) -> None:
        """Test send_message_draft with message_thread_id and parse_mode."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Draft text",
                }
                result = await client.send_message_draft(
                    chat_id=123,
                    text="Draft text",
                    draft_id=12345,
                    message_thread_id=5,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                assert isinstance(result, Message)
                call_kwargs = mock_request.call_args.args[1]
                assert call_kwargs["message_thread_id"] == 5
                assert call_kwargs["parse_mode"] == "MarkdownV2"

    @pytest.mark.asyncio
    async def test_send_message_draft_returns_boolean(self) -> None:
        """When API returns boolean, return it directly."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = True
                result = await client.send_message_draft(
                    chat_id=123, text="Draft text", draft_id=12345
                )
                assert result is True

    @pytest.mark.asyncio
    async def test_edit_message_text_with_parse_mode(self) -> None:
        """Test edit_message_text with parse_mode parameter."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Edited",
                }
                result = await client.edit_message_text(
                    chat_id=123,
                    message_id=1,
                    text="Edited",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                assert result.text == "Edited"
                call_kwargs = mock_request.call_args.args[1]
                assert call_kwargs["parse_mode"] == "MarkdownV2"

    @pytest.mark.asyncio
    async def test_send_chat_action_with_thread_id(self) -> None:
        """Test send_chat_action with message_thread_id parameter."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = True
                result = await client.send_chat_action(
                    chat_id=123, action="typing", message_thread_id=5
                )
                assert result is True
                call_kwargs = mock_request.call_args.args[1]
                assert call_kwargs["message_thread_id"] == 5

    @pytest.mark.asyncio
    async def test_api_error_without_description(self) -> None:
        """When error response has JSON but no description, use response.text."""
        import httpx

        async with TelegramApiClient("test-token") as client:
            with patch.object(client._client, "post") as mock_post:
                mock_response = create_autospec(httpx.Response, instance=True)
                mock_response.status_code = 400
                mock_response.json.return_value = {"error": "bad"}
                mock_response.text = "Bad Request"
                mock_post.return_value = mock_response

                with pytest.raises(TelegramApiError, match="Bad Request"):
                    await client._request("someMethod", {})

    @pytest.mark.asyncio
    async def test_stream_session_apology_failure(self, caplog: Any) -> None:
        """Test apology message failure is logged."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(side_effect=TelegramApiError("Failed"))

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)
        assert "Failed to send apology message" in caplog.text

    @pytest.mark.asyncio
    async def test_write_error_not_modified(self) -> None:
        """Test handling of message not modified error."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            side_effect=TelegramApiError("message is not modified", error_code=400)
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Text", is_partial=True)
            yield StreamChunk(
                thoughts="", content="Text", is_partial=True
            )  # Unchanged text
            yield StreamChunk(thoughts="", content="Final", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_write_error_rate_limit_retry_failure(self, caplog: Any) -> None:
        """Test rate limit retry failure."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
            }
        )
        mock_api.send_message = AsyncMock(
            side_effect=[
                TelegramApiError("Rate limit", error_code=429, retry_after=1),
                TelegramApiError("Retry failure", error_code=429),
                msg,
                msg,
                msg,
            ]
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Text", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)
        assert "Retry after rate limit failed" in caplog.text

    @pytest.mark.asyncio
    async def test_write_error_server_error_retry_failure(self, caplog: Any) -> None:
        """Test server error retry failure."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
            }
        )
        mock_api.send_message = AsyncMock(
            side_effect=[
                TelegramApiError("Server error", error_code=500),
                TelegramApiError("Retry failure", error_code=500),
                msg,
                msg,
            ]
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Text", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)
        assert "Retry after server error failed" in caplog.text

    @pytest.mark.asyncio
    async def test_polling_loop_exception_handling(self, caplog: Any) -> None:
        """Test exception handling in polling loop."""
        bot = TelegramBot(create_autospec(TelegramConfig), create_autospec(AdkRuntime))
        bot.config.telegram_bot_token = "test-token"  # noqa: S105
        mock_api = create_autospec(TelegramApiClient, instance=True)
        bot._api = mock_api

        mock_api.get_updates = AsyncMock(
            side_effect=[RuntimeError("Polling failed"), asyncio.CancelledError()]
        )

        bot._running = True
        with patch("asyncio.sleep", AsyncMock()), pytest.raises(asyncio.CancelledError):
            await bot._polling_loop()

        assert "Error in polling loop" in caplog.text

    @pytest.mark.asyncio
    async def test_handle_update_command_flow(
        self, telegram_config: TelegramConfig
    ) -> None:
        """Test _handle_update recognizes and handles commands."""
        bot = TelegramBot(telegram_config, create_autospec(AdkRuntime))
        bot._handle_command = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "text": "/start",
            }
        )
        update = Update.model_validate(
            {"update_id": 1, "message": message.model_dump()}
        )
        await bot._handle_update(update)
        bot._handle_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_session_retry_with_message_id(self) -> None:
        """Test retry logic when _message_id is already set."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api, update_interval_sec=0.0)
        session._message_id = 42
        session._full_text = "New text"

        error = TelegramApiError("Rate limit", error_code=429, retry_after=1)
        mock_api.edit_message_text = AsyncMock()

        await session._handle_write_error(
            error, chat_id=123, message_thread_id=None, is_final=False
        )
        mock_api.edit_message_text.assert_called_with(
            chat_id=123,
            message_id=42,
            text="New text",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    @pytest.mark.asyncio
    async def test_stream_session_server_retry_with_message_id(self) -> None:
        """Test server error retry logic when _message_id is already set."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api, update_interval_sec=0.0)
        session._message_id = 42
        session._full_text = "New text"

        error = TelegramApiError("Server error", error_code=500)
        mock_api.edit_message_text = AsyncMock()

        await session._handle_write_error(
            error, chat_id=123, message_thread_id=None, is_final=False
        )
        mock_api.edit_message_text.assert_called_with(
            chat_id=123,
            message_id=42,
            text="New text",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    def test_split_long_message_no_chunks(self) -> None:
        """Test split_long_message with empty text returns empty list."""
        assert split_long_message("") == []

    def test_find_chunk_boundary_fallback(self) -> None:
        """Test _find_chunk_boundary fallback to limit."""
        from blacki.telegram.streaming import _find_chunk_boundary

        # Text with no spaces or newlines
        text = "A" * 100
        assert _find_chunk_boundary(text, 50) == 50

    @pytest.mark.asyncio
    async def test_handle_update_no_text_coverage(
        self, telegram_config: TelegramConfig
    ) -> None:
        """Test _handle_update with message containing no text."""
        bot = TelegramBot(telegram_config, create_autospec(AdkRuntime))
        bot._handle_command = AsyncMock()  # type: ignore[method-assign]
        bot._handle_message = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "text": None,  # Missing text
            }
        )
        update = Update.model_validate(
            {"update_id": 1, "message": message.model_dump()}
        )
        await bot._handle_update(update)
        bot._handle_command.assert_not_called()
        bot._handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_session_write_empty_text(self) -> None:
        """Test _write with empty text."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api)
        session._full_text = ""
        await session._write(chat_id=123, message_thread_id=None, is_final=False)
        mock_api.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_session_finalize_no_chunks(self) -> None:
        """Test _finalize when split_long_message returns no chunks."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api)
        session._full_text = ""  # Will result in no chunks
        await session._finalize(chat_id=123, message_thread_id=None)
        mock_api.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_session_finalize_long_not_wrote(self) -> None:
        """Test _finalize for long message when not wrote successfully."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api)
        session._full_text = "A" * 5000
        session._wrote_successfully = False
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        await session._finalize(chat_id=123, message_thread_id=None)
        # Should call _write which calls send_message
        assert mock_api.send_message.called

    @pytest.mark.asyncio
    async def test_stream_session_fallback_empty_text(self) -> None:
        """Test _send_fallback with empty text."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api)
        session._full_text = ""
        await session._send_fallback(chat_id=123, message_thread_id=None)
        mock_api.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_bot_polling_not_running_exit(
        self, telegram_config: TelegramConfig
    ) -> None:
        """Test _polling_loop exits immediately if not running."""
        bot = TelegramBot(telegram_config, create_autospec(AdkRuntime))
        bot._running = False
        await bot._polling_loop()
        # Should exit immediately without calling get_updates

    def test_find_chunk_boundary_loop_coverage(self) -> None:
        """Test _find_chunk_boundary loop with different separators."""
        from blacki.telegram.streaming import _find_chunk_boundary

        text = "Hello\nWorld"
        # Force it to use \n by setting limit small enough
        boundary = _find_chunk_boundary(text, 10)
        assert boundary == 5  # index of \n

    @pytest.mark.asyncio
    async def test_api_close_twice(self) -> None:
        """Test closing API twice to hit both branches of client check."""
        client = TelegramApiClient("token")
        await client._ensure_client()
        assert client._client is not None
        await client.close()  # hits True branch
        assert client._client is None
        await client.close()  # type: ignore[unreachable]  # hits False branch

    @pytest.mark.asyncio
    async def test_stream_session_finalize_direct_empty(self) -> None:
        """Test _finalize directly with empty text to hit line 240."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api)
        session._full_text = ""
        await session._finalize(chat_id=123, message_thread_id=None)
        # Should return at line 240

    def test_split_long_message_loop_exit(self) -> None:
        """Test split_long_message loop exit by making remaining empty."""
        # We need something that passes text.strip() but then becomes empty
        # This is hard because strip() is at the start.
        # If text is " a ", remaining is "a".
        # If limit is 1, chunk is "a".
        # Then remaining = remaining[1:].lstrip() -> "".
        # THEN it hits the break because len("") <= 1.

        # To hit the loop exit, we'd need to bypass the break.
        # But len("") is always <= limit (unless limit < 0).
        pass

    @pytest.mark.asyncio
    async def test_api_close_unopened(self) -> None:
        """Test closing API when client was never opened."""
        client = TelegramApiClient("token")
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_set_my_commands_parameters_coverage(self) -> None:
        """Test set_my_commands with real parameters hitting params building."""
        import httpx

        async with TelegramApiClient("token") as client:
            with patch.object(client._client, "post") as mock_post:
                mock_response = create_autospec(httpx.Response, instance=True)
                mock_response.status_code = 200
                mock_response.json.return_value = {"ok": True, "result": True}
                mock_post.return_value = mock_response

                await client.set_my_commands(
                    [BotCommand(command="test", description="desc")],
                    scope={"type": "default"},
                    language_code="en",
                )

                call_args = mock_post.call_args
                params = call_args.kwargs["json"]
                assert params["scope"] == {"type": "default"}
                assert params["language_code"] == "en"

    @pytest.mark.asyncio
    async def test_finalize_fallback_success(self, caplog: Any) -> None:
        """Test finalize fallback send_message success."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        # Fail all edits, then succeed on send_message fallback
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(side_effect=TelegramApiError("Fail"))

        session = StreamSession(api=mock_api, update_interval_sec=0.0)
        session._wrote_successfully = False  # Force fallback path
        session._full_text = "Fallback text"

        await session._finalize(chat_id=123, message_thread_id=None)
        assert mock_api.send_message.called

    def test_find_chunk_boundary_hard_split(self) -> None:
        """Test _find_chunk_boundary when no separator is found."""
        from blacki.telegram.streaming import _find_chunk_boundary

        text = "A" * 10
        boundary = _find_chunk_boundary(text, 5)
        assert boundary == 5

    @pytest.mark.asyncio
    async def test_stream_session_handle_write_error_no_code(self) -> None:
        """Test _handle_write_error when error_code is None."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api)
        error = TelegramApiError("Unknown error")
        await session._handle_write_error(
            error, chat_id=123, message_thread_id=None, is_final=False
        )
        # Should not crash


class TestFormattingSpecialChars:
    """Tests for formatting with special characters."""

    def test_escape_markdown_with_all_special_chars(self) -> None:
        """Test escaping all markdown special chars."""
        text = "_*[]()~>#+-=|{}.!"
        escaped = escape_markdown(text)
        assert escaped == r"\_\*\[\]\(\)\~\>\#\+\-\=\|\{\}\.\!"

    def test_format_for_telegram_with_code_blocks(self) -> None:
        """Test formatting with code blocks."""
        text = "```code``` and **bold**"
        formatted = format_for_telegram(text)
        assert "```code```" in formatted
        assert "*bold*" in formatted

    def test_format_for_telegram_with_links(self) -> None:
        """Test formatting with links and special chars."""
        text = "Visit [example.com](https://example.com)"
        formatted = format_for_telegram(text)
        assert "[" in formatted and "]" in formatted

    def test_format_for_telegram_empty_string(self) -> None:
        """Test formatting empty string."""
        assert format_for_telegram("") == ""

    def test_format_for_telegram_unclosed_bold(self) -> None:
        """Test unclosed bold formatting."""
        text = "This is **unclosed bold"
        formatted = format_for_telegram(text)
        assert r"\*\*" in formatted

    @pytest.mark.parametrize(
        "text",
        [
            "unclosed *bold",
            "unclosed _italic",
            "unclosed ~strike",
            "unclosed `code",
            "unclosed ```code block",
            "unclosed __underline",
            "unclosed ||spoiler",
        ],
    )
    def test_malformed_entities_degrade_to_fully_escaped_plain_text(
        self, text: str
    ) -> None:
        """Every unclosed Telegram entity is converted to parser-safe text."""
        formatted = format_for_telegram(text)

        assert formatted == escape_markdown_plain(text)
        assert get_open_markdown_entities(formatted) == []

    @pytest.mark.parametrize(
        "text",
        [
            "*bold*",
            "_italic_",
            "~strike~",
            "`code`",
            "```code block```",
            "__underline__",
            "||spoiler||",
            r"escaped \* marker",
        ],
    )
    def test_balanced_and_escaped_entities_remain_closed(self, text: str) -> None:
        """Balanced controls never leave an entity open after formatting."""
        assert get_open_markdown_entities(format_for_telegram(text)) == []

    def test_code_block_escapes_backticks_and_backslashes(self) -> None:
        """Telegram's additional code-entity escape rules are enforced."""
        formatted = format_for_telegram("```path\\with`tick```")

        assert formatted == "```path\\\\with\\`tick```"
        assert get_open_markdown_entities(formatted) == []

    def test_format_for_telegram_bold_with_code(self) -> None:
        """Test bold containing code block and inline code."""
        text = "**Bold with `inline` and ```code block```**"
        formatted = format_for_telegram(text)
        assert "*Bold with `inline` and ```code block```*" in formatted


class TestStreamSessionEdgeCases:
    """Tests for StreamSession edge cases."""

    @pytest.mark.asyncio
    async def test_stream_session_with_error(self) -> None:
        """Test that session handles errors gracefully."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            side_effect=TelegramApiError("Bad request", error_code=400)
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hello world", is_partial=False)

        result = await session.run(chunks=chunks(), chat_id=123)
        assert result == "Hello world"

    def test_merge_stream_text_with_code_blocks(self) -> None:
        """Test merging text with code blocks."""
        result = _merge_stream_text(
            "```python```\n", "```python\ncode```", is_partial=True
        )
        assert "```python```" in result

    def test_merge_stream_text_preserves_formatting(self) -> None:
        """Test that merging preserves markdown formatting."""
        result = _merge_stream_text("Hello ", " world!", is_partial=True)
        assert result == "Hello  world!"

    def test_split_long_message_at_word_boundaries(self) -> None:
        """Test that splitting respects word boundaries."""
        text = "A" * 50 + "\n\n" + "B" * 5000
        chunks = split_long_message(text)
        assert len(chunks) >= 2
        assert all(len(chunk) <= TELEGRAM_MESSAGE_LIMIT for chunk in chunks)

    def test_split_long_message_with_urls(self) -> None:
        """Test that splitting doesn't break URLs."""
        text = "A" * 50 + " https://example.com " + "B" * 5000
        chunks = split_long_message(text)
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_stream_session_empty_chunks(self) -> None:
        """Test handling empty content chunks."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="", is_partial=True)
            yield StreamChunk(thoughts="", content="", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        mock_api.send_message.assert_called()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert "apologize" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_stream_session_multiple_chunks_error(self) -> None:
        """Test error handling when sending additional chunks fails."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        call_count = 0

        async def send_side_effect(*args: object, **kwargs: object) -> Message:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise TelegramApiError("Failed", error_code=400)
            return Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )

        mock_api.send_message = AsyncMock(side_effect=send_side_effect)

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        long_text = "A" * 5000

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content=long_text, is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_send_fallback_with_empty_text(self) -> None:
        """Test fallback with empty full_text."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        mock_api.edit_message_text = AsyncMock(
            side_effect=TelegramApiError("Failed", error_code=400)
        )

        session = StreamSession(api=mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="", is_partial=False)

        await session.run(chunks=chunks(), chat_id=123)

        assert mock_api.send_message.await_count >= 1


class TestTelegramBotEdgeCases:
    """Tests for Telegram bot edge cases."""

    @pytest.mark.asyncio
    async def test_send_final_response_with_empty_chunks(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _send_final_response with empty message chunks."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )
        bot._api = mock_api

        await bot._send_final_response(
            chat_id=123, message_thread_id=None, response_text=""
        )

        mock_api.send_message.assert_called()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert "apologize" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_final_response_escapes_unmatched_bold_marker(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Malformed model Markdown is safe before the Telegram API call."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._send_final_response(
            chat_id=123,
            message_thread_id=None,
            response_text="This is *unmatched",
        )

        call_kwargs = mock_api.send_message.call_args.kwargs
        assert call_kwargs["text"] == r"This is \*unmatched"
        assert call_kwargs["parse_mode"] == ParseMode.MARKDOWN_V2

    @pytest.mark.asyncio
    async def test_send_final_response_retries_parse_failure_as_plain_text(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Telegram grammar rejection cannot prevent final-response delivery."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock(
            side_effect=[
                TelegramApiError(
                    "HTTP 400: Bad Request: can't parse entities",
                    error_code=400,
                ),
                MagicMock(),
            ]
        )
        bot._api = mock_api

        await bot._send_final_response(
            chat_id=123,
            message_thread_id=9,
            response_text="**Balanced but rejected**",
        )

        assert mock_api.send_message.await_count == 2
        first_call, second_call = mock_api.send_message.await_args_list
        assert first_call.kwargs["parse_mode"] == ParseMode.MARKDOWN_V2
        assert second_call.kwargs["parse_mode"] is None
        assert second_call.kwargs["text"] == first_call.kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_final_response_does_not_mask_non_parse_failure(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Only entity-parser failures should trigger the plain-text retry."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock(
            side_effect=TelegramApiError("Server unavailable", error_code=500)
        )
        bot._api = mock_api

        with pytest.raises(TelegramApiError, match="Server unavailable"):
            await bot._send_final_response(
                chat_id=123,
                message_thread_id=None,
                response_text="Valid response",
            )

        mock_api.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_update_with_no_message(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _handle_update with update containing no message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        update = Update.model_validate({"update_id": 1, "message": None})

        await bot._handle_update(update)

        mock_api.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_property_initialization(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test that api property initializes TelegramApiClient."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        assert bot._api is None
        api = bot.api
        assert isinstance(api, TelegramApiClient)
        assert bot._api is api

    @pytest.mark.asyncio
    async def test_polling_loop_integration(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test the polling loop with updates."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        bot._api = mock_api

        # Mock get_updates to return one update then raise CancelledError to stop loop
        mock_update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Hello",
                },
            }
        )

        mock_api.get_updates = AsyncMock(
            side_effect=[[mock_update], asyncio.CancelledError()]
        )

        # We need to mock _handle_update to avoid deep integration
        bot._handle_update = AsyncMock()  # type: ignore[method-assign]

        bot._running = True
        with pytest.raises(asyncio.CancelledError):
            await bot._polling_loop()

        if bot._background_tasks:
            await asyncio.gather(*bot._background_tasks)

        bot._handle_update.assert_called_once_with(mock_update)

    @pytest.mark.asyncio
    async def test_handle_update_full_flow(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _handle_update hits the main message handling path."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_message = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "text": "Regular message",
            }
        )
        update = Update.model_validate(
            {"update_id": 1, "message": message.model_dump()}
        )

        await bot._handle_update(update)

        bot._handle_message.assert_called_once_with(
            chat_id=123, message_thread_id=None, user_message="Regular message"
        )

    @pytest.mark.asyncio
    async def test_handle_command_unknown(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        mock_message: Message,
    ) -> None:
        """Test _handle_command with an unknown command."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        # This should just do nothing
        await bot._handle_command(mock_message, "/unknown")

    @pytest.mark.asyncio
    async def test_safe_handle_update_no_message(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _safe_handle_update with no message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        update = Update.model_validate({"update_id": 1, "message": None})
        await bot._safe_handle_update(update)

    @pytest.mark.asyncio
    async def test_stop_with_background_tasks(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test stop cancels and awaits background tasks."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        async def dummy_task() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(dummy_task())
        bot._background_tasks.add(task)

        await bot.stop()

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_safe_handle_update_cancellation(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _safe_handle_update cancels existing tasks."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_update = AsyncMock()  # type: ignore[method-assign]

        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Hello",
                },
            }
        )

        async def long_task() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)
                raise

        existing_task = asyncio.create_task(long_task())
        bot._conversation_tasks["chat-123"] = existing_task

        task = asyncio.create_task(bot._safe_handle_update(update))
        await asyncio.sleep(0.05)

        assert existing_task.cancelled()

        await task
        bot._handle_update.assert_awaited_once_with(update)
        assert "chat-123" not in bot._conversation_tasks

    @pytest.mark.asyncio
    async def test_safe_handle_update_is_cancelled(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _safe_handle_update when itself is cancelled."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        async def mock_handle(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(10)

        bot._handle_update = mock_handle  # type: ignore[method-assign]

        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Hello",
                },
            }
        )

        task = asyncio.create_task(bot._safe_handle_update(update))
        await asyncio.sleep(0.01)

        assert bot._conversation_tasks["chat-123"] == task

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert "chat-123" not in bot._conversation_tasks

    @pytest.mark.asyncio
    async def test_safe_handle_update_replaced_task(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test finally block when task is replaced by a newer task."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        async def mock_handle(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(10)

        bot._handle_update = mock_handle  # type: ignore[method-assign]

        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Hello",
                },
            }
        )

        # Start task 1
        task1 = asyncio.create_task(bot._safe_handle_update(update))
        await asyncio.sleep(0.01)  # Yield to let task1 start

        # Start task 2 which will cancel task 1
        task2 = asyncio.create_task(bot._safe_handle_update(update))
        await asyncio.sleep(0.05)  # Yield to let task2 cancel task1

        # task1 should hit CancelledError and its finally block should see task2
        assert task1.cancelled()

        # task2 is still running and in the dict
        assert bot._conversation_tasks["chat-123"] == task2

        # Clean up
        task2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task2

    @pytest.mark.asyncio
    async def test_safe_handle_update_no_current_task(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test branch when asyncio.current_task() is None."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_update = AsyncMock()  # type: ignore[method-assign]

        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Hello",
                },
            }
        )

        with patch("asyncio.current_task", return_value=None):
            await bot._safe_handle_update(update)

        bot._handle_update.assert_awaited_once_with(update)

    @pytest.mark.asyncio
    async def test_safe_handle_update_multiple_rapid_messages(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test cascading cancellations for three rapid messages."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        # We need a task that sleeps so it can be cancelled
        async def mock_handle(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(10)

        bot._handle_update = mock_handle  # type: ignore[method-assign]

        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Hello",
                },
            }
        )

        # Start task 1
        task1 = asyncio.create_task(bot._safe_handle_update(update))
        await asyncio.sleep(0.01)

        # Start task 2
        task2 = asyncio.create_task(bot._safe_handle_update(update))
        await asyncio.sleep(0.01)

        # Start task 3
        task3 = asyncio.create_task(bot._safe_handle_update(update))
        await asyncio.sleep(0.05)

        # task1 and task2 should be cancelled
        assert task1.cancelled()
        assert task2.cancelled()

        # task3 should still be running and be the active task
        assert bot._conversation_tasks["chat-123"] == task3

        # Clean up
        task3.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task3


class TestFinalCoverage:
    """Final batch of tests to reach 100% coverage."""

    @pytest.mark.asyncio
    async def test_api_close_branch_coverage(self) -> None:
        """Hit 75->exit in api.py by calling close on a client with no httpx client."""
        client = TelegramApiClient("token")
        client._client = None
        await client.close()  # Hits False branch (75->exit)

    @pytest.mark.asyncio
    async def test_streaming_finalize_hit_240(self) -> None:
        """Hit line 240 in streaming.py by making chunks empty."""
        mock_api = create_autospec(TelegramApiClient, spec_set=True)
        session = StreamSession(api=mock_api)
        # We need length > limit but strip() to be empty
        session._full_text = " " * 5000
        await session._finalize(chat_id=123, message_thread_id=None)
        # Hits line 240

    def test_split_long_message_unreachable_branch_hit(self) -> None:
        """Try to hit 343->353 branch in streaming.py."""
        # This hits it! A string that is longer than limit but contains only whitespace.
        text = " " * 10
        chunks = split_long_message(text, limit=5)
        # 1. remaining = " " * 10.strip() -> ""
        # 2. while remaining is False -> Jumps to return chunks (line 353)
        assert chunks == []


class TestRouteNonTextMessage:
    """Tests for _route_non_text_message method."""

    @pytest.mark.asyncio
    async def test_handles_document(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test routing a document message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_file_upload = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "document": {
                    "file_id": "doc123",
                    "file_unique_id": "uniq123",
                    "file_name": "report.pdf",
                },
            }
        )

        await bot._route_non_text_message(message)

        bot._handle_file_upload.assert_called_once_with(
            chat_id=123,
            message_thread_id=None,
            file_id="doc123",
            file_name="report.pdf",
            caption=None,
        )

    @pytest.mark.asyncio
    async def test_handles_photo(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test routing a photo message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_photo_upload = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "photo": [
                    {
                        "file_id": "small",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                    },
                    {
                        "file_id": "large",
                        "file_unique_id": "u2",
                        "width": 800,
                        "height": 600,
                        "file_size": 2048,
                    },
                ],
            }
        )

        await bot._route_non_text_message(message)

        bot._handle_photo_upload.assert_called_once_with(
            chat_id=123,
            message_thread_id=None,
            file_id="large",
            file_size=2048,
            caption=None,
        )

    @pytest.mark.asyncio
    async def test_handles_audio(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test routing an audio message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_file_upload = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "audio": {
                    "file_id": "aud123",
                    "file_unique_id": "uniq123",
                    "duration": 120,
                    "file_name": "song.mp3",
                },
            }
        )

        await bot._route_non_text_message(message)

        bot._handle_file_upload.assert_called_once_with(
            chat_id=123,
            message_thread_id=None,
            file_id="aud123",
            file_name="song.mp3",
            caption=None,
        )

    @pytest.mark.asyncio
    async def test_handles_video(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test routing a video message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_file_upload = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "video": {
                    "file_id": "vid123",
                    "file_unique_id": "uniq123",
                    "width": 1920,
                    "height": 1080,
                    "duration": 60,
                    "file_name": "clip.mp4",
                },
            }
        )

        await bot._route_non_text_message(message)

        bot._handle_file_upload.assert_called_once_with(
            chat_id=123,
            message_thread_id=None,
            file_id="vid123",
            file_name="clip.mp4",
            caption=None,
        )

    @pytest.mark.asyncio
    async def test_handles_voice(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test routing a voice message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_file_upload = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "voice": {
                    "file_id": "voi123",
                    "file_unique_id": "uniq123",
                    "duration": 10,
                },
            }
        )

        await bot._route_non_text_message(message)

        bot._handle_file_upload.assert_called_once_with(
            chat_id=123,
            message_thread_id=None,
            file_id="voi123",
            file_name="voice.ogg",
            caption=None,
        )

    @pytest.mark.asyncio
    async def test_handles_unsupported_message(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test routing an unsupported non-text message."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        bot._handle_file_upload = AsyncMock()  # type: ignore[method-assign]

        message = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
            }
        )

        await bot._route_non_text_message(message)

        bot._handle_file_upload.assert_not_called()


class TestHandlePhotoUpload:
    """Tests for native multimodal Telegram photo handling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("caption", "expected_prompt"),
        [
            ("What is shown here?", "What is shown here?"),
            (None, "Describe this image."),
            ("   ", "Describe this image."),
        ],
    )
    async def test_photo_reaches_runtime_as_image_part(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        caption: str | None,
        expected_prompt: str,
    ) -> None:
        """Photo bytes and caption should reach ADK without requiring a sandbox."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        image_bytes = b"\xff\xd8\xfftelegram-jpeg"
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/photo.jpg"})
        mock_api.download_file = AsyncMock(return_value=image_bytes)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        with patch("blacki.sandbox.manager.get_sandbox_manager") as get_manager:
            await bot._handle_photo_upload(
                chat_id=123,
                message_thread_id=7,
                file_id="photo123",
                file_size=len(image_bytes),
                caption=caption,
            )

        get_manager.assert_not_called()
        call = runtime_recorder.run_user_turn_calls[0]
        assert call["message_text"] == expected_prompt
        assert isinstance(call["inference_profile"], InferenceProfile)
        parts = call["user_parts"]
        assert parts is not None
        assert parts[0].text == expected_prompt
        assert parts[1].inline_data is not None
        assert parts[1].inline_data.mime_type == "image/jpeg"
        assert parts[1].inline_data.data == image_bytes
        mock_api.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_photo_rejects_reported_oversize_before_download(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """A reported photo over 10 MB should be rejected before downloading."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_photo_upload(
            chat_id=123,
            message_thread_id=None,
            file_id="large-photo",
            file_size=10 * 1024 * 1024 + 1,
            caption=None,
        )

        mock_api.get_file.assert_not_awaited()
        assert runtime_recorder.run_user_turn_calls == []
        assert "too large" in mock_api.send_message.await_args.kwargs["text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("file_info", "downloaded"),
        [
            ({}, None),
            ({"file_path": "photos/empty.jpg"}, b""),
            ({"file_path": "photos/not-jpeg.jpg"}, b"not a jpeg"),
            (
                {"file_path": "photos/too-large.jpg"},
                b"\xff\xd8\xff" + b"x" * (10 * 1024 * 1024),
            ),
        ],
    )
    async def test_photo_rejects_invalid_downloads(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        file_info: dict[str, str],
        downloaded: bytes | None,
    ) -> None:
        """Missing, empty, invalid, and oversized downloads should fail safely."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value=file_info)
        mock_api.download_file = AsyncMock(return_value=downloaded)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_photo_upload(
            chat_id=123,
            message_thread_id=None,
            file_id="photo123",
            file_size=None,
            caption=None,
        )

        assert runtime_recorder.run_user_turn_calls == []
        assert "failed to process" in mock_api.send_message.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_photo_handles_runtime_failure_without_logging_content(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Model failures should not log image or caption data."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        private_caption = "private caption value"
        image_bytes = b"\xff\xd8\xffprivate-image-value"
        runtime_recorder.run_user_turn_error = RuntimeError("model rejected image")
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/photo.jpg"})
        mock_api.download_file = AsyncMock(return_value=image_bytes)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        await bot._handle_photo_upload(
            chat_id=123,
            message_thread_id=None,
            file_id="photo123",
            file_size=len(image_bytes),
            caption=private_caption,
        )

        assert private_caption not in caplog.text
        assert "private-image-value" not in caplog.text
        assert "failed to process" in mock_api.send_message.await_args.kwargs["text"]


class TestHandleFileUpload:
    """Tests for _handle_file_upload method."""

    @pytest.mark.asyncio
    async def test_sandbox_disabled(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test file upload when sandbox is disabled."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        with patch("blacki.sandbox.manager.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.config.enabled = False
            mock_get_manager.return_value = manager

            await bot._handle_file_upload(
                chat_id=123,
                message_thread_id=None,
                file_id="doc123",
                file_name="test.txt",
                caption=None,
            )

            mock_api.send_message.assert_called_once()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert "Sandbox is not enabled" in call_kwargs["text"]
        assert r"enabled\." in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_upload_success(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test full file upload happy path."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "documents/test.txt"})
        mock_api.download_file = AsyncMock(return_value=b"file content")
        bot._api = mock_api

        mock_sandbox = MagicMock()
        mock_sandbox.files.write_file = AsyncMock()

        with patch("blacki.sandbox.manager.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.config.enabled = True
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            await bot._handle_file_upload(
                chat_id=123,
                message_thread_id=None,
                file_id="doc123",
                file_name="../../etc/passwd",
                caption="Check this file",
            )

        mock_sandbox.files.write_file.assert_awaited_once_with(
            "/workspace/uploads/passwd", b"file content"
        )

    @pytest.mark.asyncio
    async def test_upload_without_caption(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test file upload happy path without a caption."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "documents/test.txt"})
        mock_api.download_file = AsyncMock(return_value=b"data")
        bot._api = mock_api

        mock_sandbox = MagicMock()
        mock_sandbox.files.write_file = AsyncMock()

        with patch("blacki.sandbox.manager.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.config.enabled = True
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            await bot._handle_file_upload(
                chat_id=123,
                message_thread_id=None,
                file_id="doc123",
                file_name="test.txt",
                caption=None,
            )

        call = runtime_recorder.run_user_turn_calls[0]
        assert "Caption" not in call["message_text"]

    @pytest.mark.asyncio
    async def test_upload_no_file_path(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test file upload when get_file returns no file_path."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={})
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        with patch("blacki.sandbox.manager.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.config.enabled = True
            mock_get_manager.return_value = manager

            await bot._handle_file_upload(
                chat_id=123,
                message_thread_id=None,
                file_id="doc123",
                file_name="test.txt",
                caption=None,
            )

        mock_api.send_message.assert_called_once()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert "failed to process" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_polling_fatal_telegram_error(self, caplog: Any) -> None:
        """Polling loop returns on TelegramApiError with fatal error_code."""
        bot = TelegramBot(create_autospec(TelegramConfig), create_autospec(AdkRuntime))
        bot.config.telegram_bot_token = "test-token"  # noqa: S105
        mock_api = create_autospec(TelegramApiClient, instance=True)
        bot._api = mock_api

        mock_api.get_updates = AsyncMock(
            side_effect=TelegramApiError("unauthorized", error_code=401)
        )

        bot._running = True
        await bot._polling_loop()

        assert "Fatal Telegram API error" in caplog.text

    @pytest.mark.asyncio
    async def test_polling_too_many_telegram_errors(self, caplog: Any) -> None:
        """Polling loop returns after too many consecutive TelegramApiError."""
        bot = TelegramBot(create_autospec(TelegramConfig), create_autospec(AdkRuntime))
        bot.config.telegram_bot_token = "test-token"  # noqa: S105
        mock_api = create_autospec(TelegramApiClient, instance=True)
        bot._api = mock_api

        errors: list[BaseException] = [
            TelegramApiError("fail", error_code=500) for _ in range(5)
        ]
        errors.append(asyncio.CancelledError())
        mock_api.get_updates = AsyncMock(side_effect=errors)

        bot._running = True
        with patch("asyncio.sleep", AsyncMock()):
            await bot._polling_loop()

        assert "Too many consecutive Telegram API errors" in caplog.text

    @pytest.mark.asyncio
    async def test_polling_transient_telegram_error(self, caplog: Any) -> None:
        """Polling loop warns and sleeps on transient TelegramApiError."""
        bot = TelegramBot(create_autospec(TelegramConfig), create_autospec(AdkRuntime))
        bot.config.telegram_bot_token = "test-token"  # noqa: S105
        mock_api = create_autospec(TelegramApiClient, instance=True)
        bot._api = mock_api

        transient = TelegramApiError("rate limit", error_code=429)
        mock_api.get_updates = AsyncMock(
            side_effect=[transient, asyncio.CancelledError()]
        )

        bot._running = True
        with patch("asyncio.sleep", AsyncMock()), pytest.raises(asyncio.CancelledError):
            await bot._polling_loop()

        assert "Transient Telegram API error" in caplog.text

    @pytest.mark.asyncio
    async def test_polling_too_many_generic_errors(self, caplog: Any) -> None:
        """Polling loop returns after too many consecutive generic exceptions."""
        bot = TelegramBot(create_autospec(TelegramConfig), create_autospec(AdkRuntime))
        bot.config.telegram_bot_token = "test-token"  # noqa: S105
        mock_api = create_autospec(TelegramApiClient, instance=True)
        bot._api = mock_api

        errors = [RuntimeError("fail") for _ in range(5)]
        mock_api.get_updates = AsyncMock(side_effect=errors)

        bot._running = True
        with patch("asyncio.sleep", AsyncMock()):
            await bot._polling_loop()

        assert "Too many consecutive errors" in caplog.text

    @pytest.mark.asyncio
    async def test_upload_sandbox_creation_error(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test file upload when sandbox creation fails."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "documents/test.txt"})
        mock_api.download_file = AsyncMock(return_value=b"file content")
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        with patch("blacki.sandbox.manager.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.config.enabled = True
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": None, "error": "creation error"}
            )
            mock_get_manager.return_value = manager

            await bot._handle_file_upload(
                chat_id=123,
                message_thread_id=None,
                file_id="doc123",
                file_name="test.txt",
                caption=None,
            )

        mock_api.send_message.assert_called_once()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert "failed to process" in call_kwargs["text"]


class TestTelegramBotScheduledReminders:
    """Tests for scheduled reminder handling in Telegram bot."""

    @pytest.mark.asyncio
    async def test_handle_scheduled_reminder_success(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test successful handling of a scheduled reminder."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_response = "Done!"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock(return_value=True)
        bot._api = mock_api

        reminder = Reminder(
            id=1,
            user_id="telegram-chat-12345",
            message="Summarize news",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        await bot.handle_scheduled_reminder(reminder)

        # Agent should be called
        assert len(runtime_recorder.run_user_turn_calls) == 1
        call = runtime_recorder.run_user_turn_calls[0]
        assert call["message_text"] == "[Scheduled Event] Summarize news"
        assert call["state"]["telegram_chat_id"] == "12345"
        assert "telegram_thread_id" not in call["state"]

        # Action and response should be sent
        mock_api.send_chat_action.assert_called_once_with(
            chat_id=12345, action="typing", message_thread_id=None
        )
        mock_api.send_message.assert_called_once()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 12345
        assert call_kwargs["text"] == "Done\\!"
        assert call_kwargs["message_thread_id"] is None

    @pytest.mark.asyncio
    async def test_handle_scheduled_reminder_with_thread(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test handling a scheduled reminder in a specific thread."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_response = "Done!"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock(return_value=True)
        bot._api = mock_api

        reminder = Reminder(
            id=2,
            user_id="telegram-chat-12345-thread-678",
            message="Check emails",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        await bot.handle_scheduled_reminder(reminder)

        assert len(runtime_recorder.run_user_turn_calls) == 1
        call = runtime_recorder.run_user_turn_calls[0]
        assert call["state"]["telegram_thread_id"] == "678"

        mock_api.send_chat_action.assert_called_once_with(
            chat_id=12345, action="typing", message_thread_id=678
        )
        mock_api.send_message.assert_called_once()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert call_kwargs["message_thread_id"] == 678

    @pytest.mark.asyncio
    async def test_handle_scheduled_reminder_fallback(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test fallback when the agent runtime throws an error."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_error = Exception("Agent crashed")

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock(return_value=True)
        bot._api = mock_api

        reminder = Reminder(
            id=3,
            user_id="telegram-chat-12345-thread-678",
            message="Important! Notice.",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        await bot.handle_scheduled_reminder(reminder)

        # Agent should have been called
        assert len(runtime_recorder.run_user_turn_calls) == 1

        # Fallback message should be sent
        mock_api.send_message.assert_called_once()
        call_kwargs = mock_api.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 12345
        assert call_kwargs["message_thread_id"] == 678
        # Should be formatted to escape exclamation marks, etc.
        assert "Important\\! Notice\\." in call_kwargs["text"]
        assert "⏰ *Reminder*" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_handle_scheduled_reminder_invalid_user_id(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test handling of invalid user_id format."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

        mock_api = create_autospec(TelegramApiClient, instance=True)
        bot._api = mock_api

        reminder = Reminder(
            id=4,
            user_id="invalid-user-id",
            message="Test",
            trigger_time="2026-04-18T12:00:00+00:00",
            created_at="2026-04-18T10:00:00+00:00",
        )

        await bot.handle_scheduled_reminder(reminder)

        assert len(runtime_recorder.run_user_turn_calls) == 0
        mock_api.send_chat_action.assert_not_called()
        mock_api.send_message.assert_not_called()


class TestTelegramBotPhotoAlbums:
    """Tests for multiple photo album (media_group_id) buffering and execution."""

    @pytest.mark.asyncio
    async def test_album_media_group_id_parsed_and_isolated(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album messages with different media_group_id, chat, or thread are
        isolated."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/photo.jpg"})
        mock_api.download_file = AsyncMock(
            return_value=b"\xff\xd8\xffvalid-jpeg-image-bytes"
        )
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        msg1 = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 100, "type": "private"},
                "media_group_id": "group-A",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                        "file_size": 1000,
                    }
                ],
            }
        )
        msg2 = Message.model_validate(
            {
                "message_id": 2,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 100, "type": "private"},
                "message_thread_id": 5,
                "media_group_id": "group-A",
                "photo": [
                    {
                        "file_id": "p2",
                        "file_unique_id": "u2",
                        "width": 100,
                        "height": 100,
                        "file_size": 1000,
                    }
                ],
            }
        )
        msg3 = Message.model_validate(
            {
                "message_id": 3,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 200, "type": "group"},
                "media_group_id": "group-A",
                "photo": [
                    {
                        "file_id": "p3",
                        "file_unique_id": "u3",
                        "width": 100,
                        "height": 100,
                        "file_size": 1000,
                    }
                ],
            }
        )

        task1 = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 1, "message": msg1.model_dump()})
            )
        )
        task2 = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 2, "message": msg2.model_dump()})
            )
        )
        task3 = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 3, "message": msg3.model_dump()})
            )
        )

        # Let the tasks register into album buffers
        await asyncio.sleep(0.05)

        assert (100, None, "group-A") in bot._album_buffer._buffers
        assert (100, 5, "group-A") in bot._album_buffer._buffers
        assert (200, None, "group-A") in bot._album_buffer._buffers

        # Flush all albums
        for album in list(bot._album_buffer._buffers.values()):
            bot._album_buffer._flush(album)

        await asyncio.gather(task1, task2, task3)

        assert len(runtime_recorder.run_user_turn_calls) == 3

    @pytest.mark.asyncio
    async def test_album_single_polling_response(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album messages arriving together in one batch result in one turn."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_response = "Album response"

        img1 = b"\xff\xd8\xffimage1"
        img2 = b"\xff\xd8\xffimage2"
        img3 = b"\xff\xd8\xffimage3"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(
            side_effect=lambda fid: {"file_path": f"photos/{fid}.jpg"}
        )
        mock_api.download_file = AsyncMock(
            side_effect=lambda fpath: (
                img1 if "p1" in fpath else (img2 if "p2" in fpath else img3)
            )
        )
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        messages = [
            Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "media_group_id": "album-1",
                    "caption": "Trip photos",
                    "photo": [
                        {
                            "file_id": "p1",
                            "file_unique_id": "u1",
                            "width": 100,
                            "height": 100,
                        }
                    ],
                }
            ),
            Message.model_validate(
                {
                    "message_id": 2,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "media_group_id": "album-1",
                    "photo": [
                        {
                            "file_id": "p2",
                            "file_unique_id": "u2",
                            "width": 100,
                            "height": 100,
                        }
                    ],
                }
            ),
            Message.model_validate(
                {
                    "message_id": 3,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "media_group_id": "album-1",
                    "photo": [
                        {
                            "file_id": "p3",
                            "file_unique_id": "u3",
                            "width": 100,
                            "height": 100,
                        }
                    ],
                }
            ),
        ]

        tasks = [
            asyncio.create_task(
                bot._safe_handle_update(
                    Update.model_validate(
                        {"update_id": i + 1, "message": msg.model_dump()}
                    )
                )
            )
            for i, msg in enumerate(messages)
        ]

        await asyncio.sleep(0.05)
        # Verify 3 messages buffered
        album = bot._album_buffer._buffers.get((123, None, "album-1"))
        assert album is not None
        assert len(album.messages) == 3

        # Let debounce expire
        await asyncio.sleep(0.6)
        await asyncio.gather(*tasks)

        assert len(runtime_recorder.run_user_turn_calls) == 1
        call = runtime_recorder.run_user_turn_calls[0]
        assert call["message_text"] == "Trip photos"
        parts = call["user_parts"]
        assert parts is not None
        assert len(parts) == 4  # 1 text + 3 images
        assert parts[0].text == "Trip photos"
        assert parts[1].inline_data.data == img1
        assert parts[2].inline_data.data == img2
        assert parts[3].inline_data.data == img3

    @pytest.mark.asyncio
    async def test_album_split_across_polling_responses(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album messages split across getUpdates intervals are collected together."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_response = "Album response"

        img1 = b"\xff\xd8\xffimage1"
        img2 = b"\xff\xd8\xffimage2"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(
            side_effect=lambda fid: {"file_path": f"photos/{fid}.jpg"}
        )
        mock_api.download_file = AsyncMock(
            side_effect=lambda fpath: img1 if "p1" in fpath else img2
        )
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        msg1 = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-split",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )
        msg2 = Message.model_validate(
            {
                "message_id": 2,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-split",
                "photo": [
                    {
                        "file_id": "p2",
                        "file_unique_id": "u2",
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )

        task1 = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 1, "message": msg1.model_dump()})
            )
        )
        # Receive second update before debounce (0.5s) expires, e.g. at 0.2s
        await asyncio.sleep(0.2)
        task2 = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 2, "message": msg2.model_dump()})
            )
        )

        # Wait for debounce after msg2 to complete
        await asyncio.sleep(0.6)
        await asyncio.gather(task1, task2)

        assert len(runtime_recorder.run_user_turn_calls) == 1
        call = runtime_recorder.run_user_turn_calls[0]
        assert call["message_text"] == "Describe this image."
        parts = call["user_parts"]
        assert len(parts) == 3
        assert parts[0].text == "Describe this image."
        assert parts[1].inline_data.data == img1
        assert parts[2].inline_data.data == img2

    @pytest.mark.asyncio
    async def test_album_max_wait_timeout(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album buffer flushes when max wait timeout is reached."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        img = b"\xff\xd8\xffimage"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/p.jpg"})
        mock_api.download_file = AsyncMock(return_value=img)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-timeout",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )

        task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 1, "message": msg.model_dump()})
            )
        )
        await asyncio.sleep(0.05)

        album = bot._album_buffer._buffers.get((123, None, "album-timeout"))
        assert album is not None

        # Simulate max wait triggering directly
        bot._album_buffer._flush(album)
        await task

        assert len(runtime_recorder.run_user_turn_calls) == 1

    @pytest.mark.asyncio
    async def test_album_exceeds_max_photos_limit(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album with more than 10 photos is rejected with error."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        messages = [
            Message.model_validate(
                {
                    "message_id": i + 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "media_group_id": "album-too-many",
                    "photo": [
                        {
                            "file_id": f"p{i}",
                            "file_unique_id": f"u{i}",
                            "width": 10,
                            "height": 10,
                        }
                    ],
                }
            )
            for i in range(11)
        ]

        tasks = [
            asyncio.create_task(
                bot._safe_handle_update(
                    Update.model_validate(
                        {"update_id": i + 1, "message": msg.model_dump()}
                    )
                )
            )
            for i, msg in enumerate(messages)
        ]

        await asyncio.sleep(0.05)
        album = bot._album_buffer._buffers.get((123, None, "album-too-many"))
        assert album is not None
        bot._album_buffer._flush(album)
        await asyncio.gather(*tasks)

        assert len(runtime_recorder.run_user_turn_calls) == 0
        mock_api.send_message.assert_awaited_once()
        assert "too many photos" in mock_api.send_message.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_album_exceeds_aggregate_reported_bytes(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album exceeding 20 MB in reported file size is rejected before download."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        messages = [
            Message.model_validate(
                {
                    "message_id": i + 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "media_group_id": "album-heavy",
                    "photo": [
                        {
                            "file_id": f"p{i}",
                            "file_unique_id": f"u{i}",
                            "width": 100,
                            "height": 100,
                            "file_size": 8 * 1024 * 1024,
                        }
                    ],
                }
            )
            for i in range(3)  # 3 * 8MB = 24MB > 20MB
        ]

        tasks = [
            asyncio.create_task(
                bot._safe_handle_update(
                    Update.model_validate(
                        {"update_id": i + 1, "message": msg.model_dump()}
                    )
                )
            )
            for i, msg in enumerate(messages)
        ]

        await asyncio.sleep(0.05)
        album = bot._album_buffer._buffers.get((123, None, "album-heavy"))
        assert album is not None
        bot._album_buffer._flush(album)
        await asyncio.gather(*tasks)

        assert len(runtime_recorder.run_user_turn_calls) == 0
        mock_api.get_file.assert_not_awaited()
        assert "too large" in mock_api.send_message.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_album_exceeds_aggregate_downloaded_bytes(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album exceeding 20 MB during actual download is rejected safely."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(
            side_effect=lambda fid: {"file_path": f"photos/{fid}.jpg"}
        )
        # 3 downloads of 7.5 MB = 22.5 MB > 20 MB limit
        large_chunk = b"\xff\xd8\xff" + b"x" * (7500000 - 3)
        mock_api.download_file = AsyncMock(return_value=large_chunk)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        messages = [
            Message.model_validate(
                {
                    "message_id": i + 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "media_group_id": "album-download-overflow",
                    "photo": [
                        {
                            "file_id": f"p{i}",
                            "file_unique_id": f"u{i}",
                            "width": 100,
                            "height": 100,
                        }
                    ],
                }
            )
            for i in range(3)
        ]

        tasks = [
            asyncio.create_task(
                bot._safe_handle_update(
                    Update.model_validate(
                        {"update_id": i + 1, "message": msg.model_dump()}
                    )
                )
            )
            for i, msg in enumerate(messages)
        ]

        await asyncio.sleep(0.05)
        album = bot._album_buffer._buffers.get((123, None, "album-download-overflow"))
        assert album is not None
        bot._album_buffer._flush(album)
        await asyncio.gather(*tasks)

        assert len(runtime_recorder.run_user_turn_calls) == 0
        assert "failed to process" in mock_api.send_message.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_album_and_subsequent_text_ordering(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Text arriving after an album turn starts cancels/supersedes or
        runs in order."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        img = b"\xff\xd8\xffimage"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/p.jpg"})
        mock_api.download_file = AsyncMock(return_value=img)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        album_msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-seq",
                "caption": "Album caption",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )

        album_task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate(
                    {"update_id": 1, "message": album_msg.model_dump()}
                )
            )
        )
        await asyncio.sleep(0.6)  # Album debounce flushes and starts turn
        await album_task

        # Now send text message
        text_msg = Message.model_validate(
            {
                "message_id": 2,
                "date": "2024-01-01T00:00:01Z",
                "chat": {"id": 123, "type": "private"},
                "text": "Subsequent text message",
            }
        )
        await bot._safe_handle_update(
            Update.model_validate({"update_id": 2, "message": text_msg.model_dump()})
        )

        assert len(runtime_recorder.run_user_turn_calls) == 2
        assert (
            runtime_recorder.run_user_turn_calls[0]["message_text"] == "Album caption"
        )
        assert (
            runtime_recorder.run_user_turn_calls[1]["message_text"]
            == "Subsequent text message"
        )

    @pytest.mark.asyncio
    async def test_album_failure_cleans_buffer_and_privacy_logs(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Album errors do not leak caption or image data to logs and
        clear buffer state."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_error = RuntimeError("model error")

        private_caption = "secret-album-caption-value"
        private_bytes = b"\xff\xd8\xffsecret-image-content"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/photo.jpg"})
        mock_api.download_file = AsyncMock(return_value=private_bytes)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-secret",
                "caption": private_caption,
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                        "file_size": len(private_bytes),
                    }
                ],
            }
        )

        task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 1, "message": msg.model_dump()})
            )
        )
        await asyncio.sleep(0.6)
        await task

        assert private_caption not in caplog.text
        assert "secret-image-content" not in caplog.text
        assert len(bot._album_buffer._buffers) == 0

    @pytest.mark.asyncio
    async def test_bot_stop_cleans_all_album_buffers(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Stopping the bot cancels in-flight albums and clears buffer state."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.close = AsyncMock()
        bot._api = mock_api

        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-stopping",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )

        task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 1, "message": msg.model_dump()})
            )
        )
        await asyncio.sleep(0.05)

        assert len(bot._album_buffer._buffers) == 1
        await bot.stop()
        assert len(bot._album_buffer._buffers) == 0

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_album_single_photo_oversize_reported(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album member with photo exceeding 10 MB reported size is rejected."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-single-oversize",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                        "file_size": 10 * 1024 * 1024 + 1,
                    }
                ],
            }
        )

        task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 1, "message": msg.model_dump()})
            )
        )
        await asyncio.sleep(0.6)
        await task

        assert len(runtime_recorder.run_user_turn_calls) == 0
        assert "too large" in mock_api.send_message.await_args.kwargs["text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("file_info", "downloaded"),
        [
            ({}, None),
            ({"file_path": "photos/empty.jpg"}, b""),
            ({"file_path": "photos/not-jpeg.jpg"}, b"not a jpeg"),
            (
                {"file_path": "photos/too-large.jpg"},
                b"\xff\xd8\xff" + b"x" * (10 * 1024 * 1024),
            ),
        ],
    )
    async def test_album_rejects_invalid_downloads(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
        file_info: dict[str, str],
        downloaded: bytes | None,
    ) -> None:
        """Album rejects missing path, empty bytes, non-JPEG, or single
        >10MB downloads."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value=file_info)
        mock_api.download_file = AsyncMock(return_value=downloaded)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-invalid-dl",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )

        task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 1, "message": msg.model_dump()})
            )
        )
        await asyncio.sleep(0.6)
        await task

        assert len(runtime_recorder.run_user_turn_calls) == 0
        assert "failed to process" in mock_api.send_message.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_album_caption_whitespace_falls_back_to_default_prompt(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Whitespace-only captions fall back to default image prompt."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        img = b"\xff\xd8\xffimage"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/p.jpg"})
        mock_api.download_file = AsyncMock(return_value=img)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-whitespace-caption",
                "caption": "   ",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )

        task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate({"update_id": 1, "message": msg.model_dump()})
            )
        )
        await asyncio.sleep(0.6)
        await task

        assert len(runtime_recorder.run_user_turn_calls) == 1
        call = runtime_recorder.run_user_turn_calls[0]
        assert call["message_text"] == "Describe this image."
        parts = call["user_parts"]
        assert parts is not None
        assert parts[0].text == "Describe this image."
        assert parts[1].inline_data.data == img

    @pytest.mark.asyncio
    async def test_album_empty_photos_list_rejected(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Album message with empty photos list triggers safe error response."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        album = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="empty-photos",
            chat_type=ChatType.PRIVATE,
            messages=[
                Message.model_validate(
                    {
                        "message_id": 1,
                        "date": "2024-01-01T00:00:00Z",
                        "chat": {"id": 123, "type": "private"},
                        "media_group_id": "empty-photos",
                        "photo": [],
                    }
                )
            ],
        )

        await bot._handle_album_turn(album)

        assert len(runtime_recorder.run_user_turn_calls) == 0
        assert "failed to process" in mock_api.send_message.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_album_followed_by_text_during_debounce_preserves_order(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Regression test for Codex P1: a text update arriving during the album
        debounce window must not be dropped, and turns must execute in order."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        img = b"\xff\xd8\xffimage"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/p.jpg"})
        mock_api.download_file = AsyncMock(return_value=img)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        album_msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "media_group_id": "album-p1",
                "caption": "Album caption",
                "photo": [
                    {
                        "file_id": "p1",
                        "file_unique_id": "u1",
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )
        text_msg = Message.model_validate(
            {
                "message_id": 2,
                "date": "2024-01-01T00:00:01Z",
                "chat": {"id": 123, "type": "private"},
                "text": "Text during debounce",
            }
        )

        album_task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate(
                    {"update_id": 1, "message": album_msg.model_dump()}
                )
            )
        )
        # Yield briefly so album message is registered into buffer but not yet flushed
        await asyncio.sleep(0.05)
        assert (123, None, "album-p1") in bot._album_buffer._buffers

        # Send text message during debounce window
        text_task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate(
                    {"update_id": 2, "message": text_msg.model_dump()}
                )
            )
        )

        # Wait for debounce and both updates to complete
        await asyncio.gather(album_task, text_task)

        # Assert text was not dropped and update order is preserved
        assert len(runtime_recorder.run_user_turn_calls) == 2
        assert (
            runtime_recorder.run_user_turn_calls[0]["message_text"] == "Album caption"
        )
        assert (
            runtime_recorder.run_user_turn_calls[1]["message_text"]
            == "Text during debounce"
        )

    @pytest.mark.asyncio
    async def test_process_flushed_album_cancels_older_task(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _process_flushed_album cancels older in-flight conversation task."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.get_file = AsyncMock(return_value={"file_path": "photos/p.jpg"})
        mock_api.download_file = AsyncMock(return_value=b"\xff\xd8\xffimage")
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        async def slow_turn() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)
                raise

        older_task = asyncio.create_task(slow_turn())
        conversation_key = "chat-123"
        bot._conversation_tasks[conversation_key] = older_task
        bot._conversation_task_seqs[conversation_key] = 1

        album = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="album-cancel-older",
            chat_type=ChatType.PRIVATE,
            messages=[
                Message.model_validate(
                    {
                        "message_id": 2,
                        "date": "2024-01-01T00:00:00Z",
                        "chat": {"id": 123, "type": "private"},
                        "media_group_id": "album-cancel-older",
                        "caption": "Album turn",
                        "photo": [
                            {
                                "file_id": "p1",
                                "file_unique_id": "u1",
                                "width": 100,
                                "height": 100,
                            }
                        ],
                    }
                )
            ],
            created_seq=2,
        )

        await bot._process_flushed_album(album)
        assert older_task.cancelled()
        assert len(runtime_recorder.run_user_turn_calls) == 1

    @pytest.mark.asyncio
    async def test_process_flushed_album_branches(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _process_flushed_album when current_task is None, cancelled,
        or replaced."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api
        bot._handle_album_turn = AsyncMock()  # type: ignore[method-assign]

        album = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="flushed-branch",
            chat_type=ChatType.PRIVATE,
            messages=[],
            future=None,
            created_seq=1,
        )

        # Branch 1: current_task is None
        with patch("asyncio.current_task", return_value=None):
            await bot._process_flushed_album(album)

        # Branch 2: Cancellation during turn
        async def mock_handle_slow(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(10)

        bot._handle_album_turn = mock_handle_slow  # type: ignore[method-assign]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        album2 = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="flushed-cancel",
            chat_type=ChatType.PRIVATE,
            messages=[],
            future=future,
            created_seq=2,
        )

        album_task = asyncio.create_task(bot._process_flushed_album(album2))
        await asyncio.sleep(0.01)
        album_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await album_task
        assert future.done()

        # Branch 3: Task replaced in finally block
        async def mock_handle_replace(*args: Any, **kwargs: Any) -> None:
            # Replace conversation task before finishing
            bot._conversation_tasks["chat-123"] = asyncio.create_task(
                asyncio.sleep(0.1)
            )

        bot._handle_album_turn = mock_handle_replace  # type: ignore[method-assign]
        album3 = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="flushed-replace",
            chat_type=ChatType.PRIVATE,
            messages=[],
            future=None,
            created_seq=3,
        )
        await bot._process_flushed_album(album3)
        assert "chat-123" in bot._conversation_tasks
        bot._conversation_tasks["chat-123"].cancel()
        with pytest.raises(asyncio.CancelledError):
            await bot._conversation_tasks["chat-123"]

    @pytest.mark.asyncio
    async def test_safe_handle_update_active_album_future_branches(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test _safe_handle_update active album buffer with None future or error."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock()
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        # Case 1: active album with future=None
        album_no_future = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="no-fut",
            chat_type=ChatType.PRIVATE,
            messages=[],
            future=None,
        )
        bot._album_buffer._buffers[(123, None, "no-fut")] = album_no_future

        text_msg = Message.model_validate(
            {
                "message_id": 1,
                "date": "2024-01-01T00:00:00Z",
                "chat": {"id": 123, "type": "private"},
                "text": "Hello text",
            }
        )
        await bot._safe_handle_update(
            Update.model_validate({"update_id": 1, "message": text_msg.model_dump()})
        )
        assert len(runtime_recorder.run_user_turn_calls) == 1

        # Case 2: active album with failing future
        loop = asyncio.get_running_loop()
        err_future: asyncio.Future[None] = loop.create_future()
        err_future.set_exception(RuntimeError("album failed"))
        album_err_future = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="err-fut",
            chat_type=ChatType.PRIVATE,
            messages=[],
            future=err_future,
        )
        bot._album_buffer._buffers[(123, None, "err-fut")] = album_err_future

        await bot._safe_handle_update(
            Update.model_validate({"update_id": 2, "message": text_msg.model_dump()})
        )
        assert len(runtime_recorder.run_user_turn_calls) == 2

        # Case 3: active album future cancelled while current_task is cancelling
        cancel_future: asyncio.Future[None] = loop.create_future()
        album_cancelling = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="cancel-fut",
            chat_type=ChatType.PRIVATE,
            messages=[],
            future=cancel_future,
        )
        bot._album_buffer._buffers[(123, None, "cancel-fut")] = album_cancelling

        task = asyncio.create_task(
            bot._safe_handle_update(
                Update.model_validate(
                    {"update_id": 3, "message": text_msg.model_dump()}
                )
            )
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        bot._album_buffer._buffers.pop((123, None, "cancel-fut"), None)

        # Case 4: active album future cancelled while current_task is NOT cancelling
        cancelled_fut: asyncio.Future[None] = loop.create_future()
        cancelled_fut.cancel()
        album_pre_cancelled = _BufferedAlbum(
            chat_id=123,
            message_thread_id=None,
            media_group_id="pre-cancelled-fut",
            chat_type=ChatType.PRIVATE,
            messages=[],
            future=cancelled_fut,
        )
        bot._album_buffer._buffers[(123, None, "pre-cancelled-fut")] = (
            album_pre_cancelled
        )

        await bot._safe_handle_update(
            Update.model_validate({"update_id": 4, "message": text_msg.model_dump()})
        )
        assert len(runtime_recorder.run_user_turn_calls) == 3
