"""Unit tests for Telegram bot module."""

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from telegram import Message, Update
from telegram.ext import Application

from blacki.adk_runtime import AdkRuntime, SessionLocator
from blacki.telegram import TelegramConfig
from blacki.telegram.bot import (
    TELEGRAM_MESSAGE_LIMIT,
    TelegramBot,
    TelegramSessionIdentity,
    create_telegram_bot,
)


class RecordingRuntime:
    """Small fake runtime for Telegram bot tests."""

    def __init__(self) -> None:
        self.run_user_turn_response = "Test response"
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
            "TELEGRAM_BOT_TOKEN": "test-token-123",  # noqa: S106
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
def mock_update() -> Any:
    """Create a mock Telegram update."""
    message = create_autospec(Message, instance=True, spec_set=True)
    message.text = "Hello, bot!"
    message.reply_text = AsyncMock()
    message.message_thread_id = None

    update = create_autospec(Update, instance=True, spec_set=True)
    update.effective_chat = SimpleNamespace(id=123456789)
    update.message = message
    return update


@pytest.fixture
def mock_update_no_chat() -> Any:
    """Create a mock Telegram update without effective_chat."""
    update = create_autospec(Update, instance=True, spec_set=True)
    update.effective_chat = None
    update.message = create_autospec(Message, instance=True, spec_set=True)
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_context() -> Any:
    """Create a placeholder Telegram context."""
    return object()


def test_init_with_config(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test initialization with valid config."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    assert bot.config == telegram_config
    assert cast(Any, bot.runtime) is runtime_recorder
    assert bot._app is None


def test_ensure_app_creates_app(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that _ensure_app creates the Application."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    with patch("blacki.telegram.bot.Application.builder") as mock_builder:
        mock_app = create_autospec(Application, instance=True, spec_set=True)
        mock_builder.return_value.token.return_value.build.return_value = mock_app

        app = bot._ensure_app()

        assert app == mock_app
        mock_builder.return_value.token.assert_called_once_with(
            telegram_config.telegram_bot_token
        )


def test_app_property_reuses_existing_app(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that app property returns the already-created Application."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
    mock_app = create_autospec(Application, instance=True, spec_set=True)
    bot._app = mock_app

    assert bot.app is mock_app


def test_ensure_app_raises_without_token(
    telegram_config_disabled: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that _ensure_app raises ValueError without token."""
    bot = TelegramBot(telegram_config_disabled, cast(AdkRuntime, runtime_recorder))

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN is required"):
        bot._ensure_app()


def test_setup_handlers_adds_handlers(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that _setup_handlers adds all required handlers."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
    mock_app = create_autospec(Application, instance=True, spec_set=True)

    bot._setup_handlers(mock_app)

    assert mock_app.add_handler.call_count == 4


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


@pytest.mark.asyncio
async def test_start_command_sends_welcome(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update: Any,
    mock_context: Any,
) -> None:
    """Test that /start sends welcome message."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._start_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once()
    welcome_message = mock_update.message.reply_text.call_args.args[0]
    assert "Hello" in welcome_message
    assert "ADK" in welcome_message


@pytest.mark.asyncio
async def test_start_command_no_chat(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update_no_chat: Any,
    mock_context: Any,
) -> None:
    """Test that /start exits early without a chat."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._start_command(mock_update_no_chat, mock_context)

    mock_update_no_chat.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_help_command_sends_help(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update: Any,
    mock_context: Any,
) -> None:
    """Test that /help sends help message."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._help_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once()
    help_message = mock_update.message.reply_text.call_args.args[0]
    assert "Commands" in help_message
    assert "/reset" in help_message


@pytest.mark.asyncio
async def test_help_command_no_chat(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update_no_chat: Any,
    mock_context: Any,
) -> None:
    """Test that /help exits early without a chat."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._help_command(mock_update_no_chat, mock_context)

    mock_update_no_chat.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_reset_command_creates_next_session(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update: Any,
    mock_context: Any,
) -> None:
    """Test that /reset creates the next ADK session version."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._reset_command(mock_update, mock_context)

    assert runtime_recorder.create_next_session_calls == [
        {
            "locator": SessionLocator(
                user_id="telegram-chat-123456789",
                session_id_prefix="telegram-chat-123456789",
            ),
            "state": {
                "user_id": "telegram-chat-123456789",
                "telegram_chat_id": "123456789",
                "telegram_conversation_key": "chat-123456789",
            },
        }
    ]
    mock_update.message.reply_text.assert_called_once()
    assert "fresh ADK conversation" in mock_update.message.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_reset_command_no_chat(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update_no_chat: Any,
    mock_context: Any,
) -> None:
    """Test that /reset exits early without a chat."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._reset_command(mock_update_no_chat, mock_context)

    assert runtime_recorder.create_next_session_calls == []
    mock_update_no_chat.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_reset_command_uses_thread_identity(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update: Any,
    mock_context: Any,
) -> None:
    """Test that /reset uses topic thread identity when present."""
    mock_update.message.message_thread_id = 22
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._reset_command(mock_update, mock_context)

    locator = runtime_recorder.create_next_session_calls[0]["locator"]
    assert locator.user_id == "telegram-chat-123456789-thread-22"
    assert locator.session_id_prefix == "telegram-chat-123456789-thread-22"


@pytest.mark.asyncio
async def test_reset_command_error(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update: Any,
    mock_context: Any,
) -> None:
    """Test that /reset returns a user-facing error when runtime reset fails."""
    runtime_recorder.create_next_session_error = RuntimeError("reset failed")
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._reset_command(mock_update, mock_context)

    error_message = mock_update.message.reply_text.call_args.args[0]
    assert "couldn't reset" in error_message


@pytest.mark.asyncio
async def test_handle_message_runs_adk_turn(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update: Any,
    mock_context: Any,
) -> None:
    """Test successful message handling through the ADK runtime."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._handle_message(mock_update, mock_context)

    assert runtime_recorder.run_user_turn_calls == [
        {
            "locator": SessionLocator(
                user_id="telegram-chat-123456789",
                session_id_prefix="telegram-chat-123456789",
            ),
            "message_text": "Hello, bot!",
            "state": {
                "user_id": "telegram-chat-123456789",
                "telegram_chat_id": "123456789",
                "telegram_conversation_key": "chat-123456789",
            },
        }
    ]
    mock_update.message.reply_text.assert_called_once_with("Test response")


@pytest.mark.asyncio
async def test_handle_message_splits_long_responses(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update: Any,
    mock_context: Any,
) -> None:
    """Test that long responses are split into Telegram-safe chunks."""
    runtime_recorder.run_user_turn_response = (
        ("A" * TELEGRAM_MESSAGE_LIMIT) + "\n\n" + ("B" * 100)
    )
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._handle_message(mock_update, mock_context)

    assert mock_update.message.reply_text.await_count == 2
    first_chunk = mock_update.message.reply_text.await_args_list[0].args[0]
    second_chunk = mock_update.message.reply_text.await_args_list[1].args[0]
    assert len(first_chunk) <= TELEGRAM_MESSAGE_LIMIT
    assert second_chunk == "B" * 100


@pytest.mark.asyncio
async def test_handle_message_error(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update: Any,
    mock_context: Any,
) -> None:
    """Test message handling with an ADK runtime error."""
    runtime_recorder.run_user_turn_error = RuntimeError("runner failed")
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once()
    error_message = mock_update.message.reply_text.call_args.args[0]
    assert "error" in error_message.lower()


@pytest.mark.asyncio
async def test_handle_message_no_chat(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
    mock_update_no_chat: Any,
    mock_context: Any,
) -> None:
    """Test message handling with missing effective_chat."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot._handle_message(mock_update_no_chat, mock_context)

    mock_update_no_chat.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_start_polling_success(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test successful start_polling."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    mock_app = create_autospec(Application, instance=True, spec_set=True)
    mock_app.initialize = AsyncMock()
    mock_app.start = AsyncMock()
    mock_app.bot = SimpleNamespace(set_my_commands=AsyncMock())
    mock_app.updater = SimpleNamespace(start_polling=AsyncMock())

    with patch.object(bot, "_ensure_app", return_value=mock_app):
        await bot.start_polling()

    mock_app.initialize.assert_awaited_once()
    mock_app.start.assert_awaited_once()
    mock_app.updater.start_polling.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_polling_not_configured(
    telegram_config_disabled: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test start_polling when Telegram is disabled."""
    bot = TelegramBot(
        telegram_config_disabled,
        cast(AdkRuntime, runtime_recorder),
    )

    await bot.start_polling()

    assert bot._app is None


@pytest.mark.asyncio
async def test_start_polling_without_updater(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test start_polling when the app has no updater."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    mock_app = create_autospec(Application, instance=True, spec_set=True)
    mock_app.initialize = AsyncMock()
    mock_app.start = AsyncMock()
    mock_app.bot = SimpleNamespace(set_my_commands=AsyncMock())
    mock_app.updater = None

    with patch.object(bot, "_ensure_app", return_value=mock_app):
        await bot.start_polling()

    mock_app.initialize.assert_awaited_once()
    mock_app.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_commands_error(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that command registration errors are swallowed."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
    mock_app = create_autospec(Application, instance=True, spec_set=True)
    mock_app.bot = SimpleNamespace(
        set_my_commands=AsyncMock(side_effect=RuntimeError("boom"))
    )

    await bot._register_commands(mock_app)


@pytest.mark.asyncio
async def test_stop_with_app(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test stop with initialized app."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    mock_app = create_autospec(Application, instance=True, spec_set=True)
    mock_app.stop = AsyncMock()
    mock_app.shutdown = AsyncMock()
    mock_app.updater = SimpleNamespace(stop=AsyncMock())
    bot._app = mock_app

    await bot.stop()

    mock_app.updater.stop.assert_awaited_once()
    mock_app.stop.assert_awaited_once()
    mock_app.shutdown.assert_awaited_once()
    assert runtime_recorder.closed is True


@pytest.mark.asyncio
async def test_stop_with_app_without_updater(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test stop when the app was initialized without an updater."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    mock_app = create_autospec(Application, instance=True, spec_set=True)
    mock_app.stop = AsyncMock()
    mock_app.shutdown = AsyncMock()
    mock_app.updater = None
    bot._app = mock_app

    await bot.stop()

    mock_app.stop.assert_awaited_once()
    mock_app.shutdown.assert_awaited_once()
    assert runtime_recorder.closed is True


@pytest.mark.asyncio
async def test_stop_without_app_still_closes_runtime(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test stop when the bot app was never initialized."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    await bot.stop()

    assert runtime_recorder.closed is True


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


@pytest.mark.asyncio
async def test_start_typing_indicator_creates_task(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that _start_typing_indicator creates a task."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    mock_app = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))
    bot._app = cast(Any, mock_app)

    task = bot._start_typing_indicator(12345)

    assert task is not None
    assert not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_typing_indicator_returns_on_send_error(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that typing indicator stops when Telegram rejects chat actions."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    mock_app = SimpleNamespace(
        bot=SimpleNamespace(
            send_chat_action=AsyncMock(side_effect=RuntimeError("network error"))
        )
    )
    bot._app = cast(Any, mock_app)

    task = bot._start_typing_indicator(12345)
    await task

    assert task.done()
    assert task.exception() is None


@pytest.mark.asyncio
async def test_typing_indicator_reraises_cancelled_error(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that typing indicator re-raises CancelledError during sleep."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    mock_app = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))
    bot._app = cast(Any, mock_app)

    task = bot._start_typing_indicator(12345)
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_stop_typing_indicator_cancels_task(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that _stop_typing_indicator cancels the task."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    mock_app = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))
    bot._app = cast(Any, mock_app)

    task = bot._start_typing_indicator(12345)
    await bot._stop_typing_indicator(task)

    assert task.cancelled()


def test_split_response_text_returns_empty_chunk_for_blank_response(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test that blank responses produce a single empty chunk."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    assert bot._split_response_text("   ") == [""]


def test_split_response_text_uses_hard_split_without_boundaries(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test fallback chunk splitting when no readable boundary exists."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
    response_text = "A" * (TELEGRAM_MESSAGE_LIMIT + 10)

    chunks = bot._split_response_text(response_text)

    assert len(chunks) == 2
    assert len(chunks[0]) == TELEGRAM_MESSAGE_LIMIT
    assert len(chunks[1]) == 10


def test_find_chunk_boundary_falls_back_to_limit(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test chunk boundary fallback when no separator exists."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))

    boundary = bot._find_chunk_boundary("A" * (TELEGRAM_MESSAGE_LIMIT + 10))

    assert boundary == TELEGRAM_MESSAGE_LIMIT


def test_split_response_text_exits_loop_when_remaining_text_is_consumed(
    telegram_config: TelegramConfig,
    runtime_recorder: RecordingRuntime,
) -> None:
    """Test the loop path where one chunk consumes the remaining long response."""
    bot = TelegramBot(telegram_config, cast(AdkRuntime, runtime_recorder))
    response_text = "A" * (TELEGRAM_MESSAGE_LIMIT + 10)

    with patch.object(bot, "_find_chunk_boundary", return_value=len(response_text)):
        chunks = bot._split_response_text(response_text)

    assert chunks == [response_text]
