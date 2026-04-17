"""Unit tests for Telegram bot module."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, create_autospec, patch

import pytest

from blacki.adk_runtime import AdkRuntime, SessionLocator, StreamChunk, TurnResponse
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient, TelegramApiError
from blacki.telegram.bot import (
    TELEGRAM_MESSAGE_LIMIT,
    TelegramBot,
    TelegramSessionIdentity,
    create_telegram_bot,
    escape_markdown,
    format_for_telegram,
)
from blacki.telegram.streaming import DraftManager, split_long_message
from blacki.telegram.types import BotCommand, Message, ParseMode, Update


class RecordingRuntime:
    """Small fake runtime for Telegram bot tests."""

    def __init__(self) -> None:
        self.run_user_turn_response = "Test response"
        self.run_user_turn_thoughts = ""
        self.run_user_turn_error: Exception | None = None
        self.create_next_session_error: Exception | None = None
        self.run_user_turn_calls: list[dict[str, Any]] = []
        self.create_next_session_calls: list[dict[str, Any]] = []
        self.closed = False

    async def run_user_turn(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
    ) -> str:
        self.run_user_turn_calls.append(
            {
                "locator": locator,
                "message_text": message_text,
                "state": state,
            }
        )
        if self.run_user_turn_error is not None:
            raise self.run_user_turn_error
        return self.run_user_turn_response

    async def run_user_turn_with_thoughts(
        self,
        *,
        locator: SessionLocator,
        message_text: str,
        state: dict[str, Any] | None = None,
    ) -> TurnResponse:
        self.run_user_turn_calls.append(
            {
                "locator": locator,
                "message_text": message_text,
                "state": state,
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
    ) -> AsyncIterator[StreamChunk]:
        self.run_user_turn_calls.append(
            {
                "locator": locator,
                "message_text": message_text,
                "state": state,
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


def test_draft_manager_creates_on_demand(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that draft_manager is created on first access."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    dm = bot.draft_manager
    assert dm is not None
    assert bot._draft_manager is dm


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
        """Test successful draft message sending."""
        async with TelegramApiClient("test-token") as client:
            with patch.object(client, "_request") as mock_request:
                mock_request.return_value = {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                    "text": "Draft text",
                }
                result = await client.send_message_draft(
                    chat_id=123, text="Draft text", draft_id="draft-123"
                )
                assert isinstance(result, Message)
                assert result.message_id == 1
                mock_request.assert_called_once_with(
                    "sendMessageDraft",
                    {"chat_id": 123, "text": "Draft text", "draft_id": "draft-123"},
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


class TestDraftManager:
    """Tests for DraftManager."""

    @pytest.mark.asyncio
    async def test_stream_response_updates_drafts(self) -> None:
        """Test that streaming updates drafts."""
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message_draft = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        manager = DraftManager(mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="Thinking...", content="Hello")
            yield StreamChunk(
                thoughts="Thinking...", content="Hello world", is_partial=False
            )

        final_thoughts, final_content = await manager.stream_response(
            chat_id=123, chunks=chunks()
        )

        assert final_thoughts == "Thinking..."
        assert final_content == "Hello world"
        assert mock_api.send_message_draft.await_count >= 1

    @pytest.mark.asyncio
    async def test_stream_response_handles_empty_chunks(self) -> None:
        """Test streaming with empty content."""
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message_draft = AsyncMock()

        manager = DraftManager(mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="", is_partial=False)

        final_thoughts, final_content = await manager.stream_response(
            chat_id=123, chunks=chunks()
        )

        assert final_thoughts == ""
        assert final_content == ""

    @pytest.mark.asyncio
    async def test_stream_response_with_api_error(self) -> None:
        """Test streaming handles API errors gracefully."""
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message_draft = AsyncMock(
            side_effect=TelegramApiError("Rate limit", error_code=429)
        )

        manager = DraftManager(mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="Thinking...", content="Hello", is_partial=False)

        final_thoughts, final_content = await manager.stream_response(
            chat_id=123, chunks=chunks()
        )

        assert final_thoughts == "Thinking..."
        assert final_content == "Hello"

    @pytest.mark.asyncio
    async def test_stream_response_with_throttling(self) -> None:
        """Test streaming respects throttle interval."""
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message_draft = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        manager = DraftManager(mock_api, update_interval_sec=1.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="A", content="B")
            yield StreamChunk(thoughts="A", content="B", is_partial=False)

        final_thoughts, final_content = await manager.stream_response(
            chat_id=123, chunks=chunks()
        )

        assert final_thoughts == "A"
        assert final_content == "B"

    @pytest.mark.asyncio
    async def test_stream_response_with_thread_id(self) -> None:
        """Test streaming with message thread ID."""
        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_message_draft = AsyncMock(
            return_value=Message.model_validate(
                {
                    "message_id": 1,
                    "date": "2024-01-01T00:00:00Z",
                    "chat": {"id": 123, "type": "private"},
                }
            )
        )

        manager = DraftManager(mock_api, update_interval_sec=0.0)

        async def chunks() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(thoughts="", content="Hello", is_partial=False)

        await manager.stream_response(
            chat_id=123, chunks=chunks(), message_thread_id=42
        )

        call_kwargs = mock_api.send_message_draft.call_args.kwargs
        assert call_kwargs.get("message_thread_id") == 42


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
        """Test that bold formatting markers are not escaped."""
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
    ) -> None:
        """Test message handling with streaming."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_response = "Hello back!"

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        mock_draft_manager = create_autospec(DraftManager, instance=True)
        mock_draft_manager.stream_response = AsyncMock(return_value=("", "Hello back!"))
        bot._draft_manager = mock_draft_manager

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=None,
            user_message="Hello, bot!",
        )

        mock_api.send_chat_action.assert_called_once_with(
            chat_id=123456789, action="typing"
        )
        mock_draft_manager.stream_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_sends_thoughts_and_content(
        self,
        telegram_config: TelegramConfig,
        runtime_recorder: RecordingRuntime,
    ) -> None:
        """Test message handling sends both thoughts and content."""
        bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
        runtime_recorder.run_user_turn_thoughts = "Let me think..."
        runtime_recorder.run_user_turn_response = "Here is my answer."

        mock_api = create_autospec(TelegramApiClient, instance=True)
        mock_api.send_chat_action = AsyncMock(return_value=True)
        mock_api.send_message = AsyncMock()
        bot._api = mock_api

        mock_draft_manager = create_autospec(DraftManager, instance=True)
        mock_draft_manager.stream_response = AsyncMock(
            return_value=("Let me think...", "Here is my answer.")
        )
        bot._draft_manager = mock_draft_manager

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=None,
            user_message="Hello!",
        )

        assert mock_api.send_message.await_count == 2

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

        mock_draft_manager = create_autospec(DraftManager, instance=True)
        mock_draft_manager.stream_response = AsyncMock(
            side_effect=RuntimeError("runner failed")
        )
        bot._draft_manager = mock_draft_manager

        await bot._handle_message(
            chat_id=123456789,
            message_thread_id=None,
            user_message="Hello!",
        )

        mock_api.send_message.assert_called()
        call_args = mock_api.send_message.call_args
        assert "error" in call_args.kwargs["text"].lower()


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
        assert len(commands) == 3
        assert commands[0].command == "start"

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
