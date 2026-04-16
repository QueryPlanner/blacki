"""Unit tests for Telegram bot module."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Update
from telegram.ext import Application

from blacki.telegram import TelegramConfig
from blacki.telegram.bot import TelegramBot, create_telegram_bot


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
def mock_model() -> str:
    """Create a mock model string."""
    return "openrouter/openai/gpt-4"


@pytest.fixture
def mock_model_object() -> MagicMock:
    """Create a mock model object with .model attribute."""
    mock = MagicMock()
    mock.model = "openrouter/openai/gpt-4"
    return mock


@pytest.fixture
def mock_update() -> MagicMock:
    """Create a mock Telegram Update."""
    update = MagicMock(spec=Update)
    update.effective_chat = MagicMock()
    update.effective_chat.id = 123456789
    update.message = MagicMock()
    update.message.text = "Hello, bot!"
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_update_no_chat() -> MagicMock:
    """Create a mock Telegram Update without effective_chat."""
    update = MagicMock(spec=Update)
    update.effective_chat = None
    update.message = MagicMock()
    return update


@pytest.fixture
def mock_context() -> MagicMock:
    """Create a mock Telegram context."""
    return MagicMock()


class TestTelegramBotInit:
    """Tests for TelegramBot initialization."""

    def test_init_with_config(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test initialization with valid config."""
        bot = TelegramBot(telegram_config, mock_model)

        assert bot.config == telegram_config
        assert bot.model == mock_model
        assert bot._app is None

    def test_init_with_model_object(
        self, telegram_config: TelegramConfig, mock_model_object: MagicMock
    ) -> None:
        """Test initialization with model object."""
        bot = TelegramBot(telegram_config, mock_model_object)

        assert bot.config == telegram_config
        assert bot.model == mock_model_object


class TestTelegramBotEnsureApp:
    """Tests for _ensure_app and app property."""

    def test_ensure_app_creates_app(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _ensure_app creates the Application."""
        bot = TelegramBot(telegram_config, mock_model)

        with patch("blacki.telegram.bot.Application.builder") as mock_builder:
            mock_app = MagicMock(spec=Application)
            mock_builder.return_value.token.return_value.build.return_value = mock_app

            app = bot._ensure_app()

            assert app == mock_app
            mock_builder.return_value.token.assert_called_once_with(
                telegram_config.telegram_bot_token
            )

    def test_ensure_app_raises_without_token(
        self, telegram_config_disabled: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _ensure_app raises ValueError without token."""
        bot = TelegramBot(telegram_config_disabled, mock_model)

        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN is required"):
            bot._ensure_app()

    def test_ensure_app_reuses_existing_app(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _ensure_app reuses existing app."""
        bot = TelegramBot(telegram_config, mock_model)
        mock_app = MagicMock(spec=Application)
        bot._app = mock_app

        app = bot._ensure_app()

        assert app == mock_app

    def test_app_property_calls_ensure_app(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that app property calls _ensure_app."""
        bot = TelegramBot(telegram_config, mock_model)

        with patch.object(bot, "_ensure_app") as mock_ensure:
            mock_ensure.return_value = MagicMock(spec=Application)
            _ = bot.app

            mock_ensure.assert_called_once()


class TestTelegramBotSetupHandlers:
    """Tests for _setup_handlers."""

    def test_setup_handlers_adds_handlers(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _setup_handlers adds all required handlers."""
        bot = TelegramBot(telegram_config, mock_model)
        mock_app = MagicMock(spec=Application)

        bot._setup_handlers(mock_app)

        # Should add 4 handlers: start, help, clear, message
        assert mock_app.add_handler.call_count == 4


class TestTelegramBotStartCommand:
    """Tests for _start_command."""

    @pytest.mark.asyncio
    async def test_start_command_sends_welcome(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test that /start sends welcome message."""
        bot = TelegramBot(telegram_config, mock_model)

        await bot._start_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Hello" in call_args
        assert "blacki" in call_args

    @pytest.mark.asyncio
    async def test_start_command_no_chat(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update_no_chat: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test /start handles missing effective_chat."""
        bot = TelegramBot(telegram_config, mock_model)

        await bot._start_command(mock_update_no_chat, mock_context)

        mock_update_no_chat.message.reply_text.assert_not_called()


class TestTelegramBotHelpCommand:
    """Tests for _help_command."""

    @pytest.mark.asyncio
    async def test_help_command_sends_help(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test that /help sends help message."""
        bot = TelegramBot(telegram_config, mock_model)

        await bot._help_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Commands" in call_args
        assert "/start" in call_args
        assert "/help" in call_args
        assert "/reset" in call_args

    @pytest.mark.asyncio
    async def test_help_command_no_chat(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update_no_chat: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test /help handles missing effective_chat."""
        bot = TelegramBot(telegram_config, mock_model)

        await bot._help_command(mock_update_no_chat, mock_context)

        mock_update_no_chat.message.reply_text.assert_not_called()


class TestTelegramBotResetCommand:
    """Tests for _reset_command."""

    @pytest.mark.asyncio
    async def test_reset_command_creates_new_session(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test /reset creates a new session."""
        bot = TelegramBot(telegram_config, mock_model)

        # Set initial session
        bot._session_ids["123456789"] = "oldsession"

        await bot._reset_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "reset" in call_args.lower()
        # Verify session was changed
        assert bot._session_ids["123456789"] != "oldsession"

    @pytest.mark.asyncio
    async def test_reset_command_no_chat(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update_no_chat: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test /reset handles missing effective_chat."""
        bot = TelegramBot(telegram_config, mock_model)

        await bot._reset_command(mock_update_no_chat, mock_context)

        mock_update_no_chat.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_command_multiple_resets(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test multiple /reset calls generate different session IDs."""
        bot = TelegramBot(telegram_config, mock_model)

        await bot._reset_command(mock_update, mock_context)
        first_session = bot._session_ids["123456789"]

        await bot._reset_command(mock_update, mock_context)
        second_session = bot._session_ids["123456789"]

        assert first_session != second_session


class TestTelegramBotHandleMessage:
    """Tests for _handle_message."""

    @pytest.mark.asyncio
    async def test_handle_message_success(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test successful message handling."""
        bot = TelegramBot(telegram_config, mock_model)

        with (
            patch.object(bot, "_get_memory_context", return_value=""),
            patch.object(bot, "_generate_response", return_value="Test response"),
            patch.object(bot, "_save_to_memory"),
        ):
            await bot._handle_message(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once_with("Test response")

    @pytest.mark.asyncio
    async def test_handle_message_with_memory_context(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test message handling with memory context."""
        bot = TelegramBot(telegram_config, mock_model)

        with (
            patch.object(bot, "_get_memory_context", return_value="Previous context"),
            patch.object(bot, "_generate_response", return_value="Response"),
            patch.object(bot, "_save_to_memory"),
        ):
            await bot._handle_message(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_error(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test message handling with error."""
        bot = TelegramBot(telegram_config, mock_model)

        with patch.object(
            bot, "_get_memory_context", side_effect=Exception("Test error")
        ):
            await bot._handle_message(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "error" in call_args.lower()

    @pytest.mark.asyncio
    async def test_handle_message_no_chat(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update_no_chat: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test message handling with missing effective_chat."""
        bot = TelegramBot(telegram_config, mock_model)

        await bot._handle_message(mock_update_no_chat, mock_context)

        mock_update_no_chat.message.reply_text.assert_not_called()


class TestTelegramBotGetMemoryContext:
    """Tests for _get_memory_context."""

    @pytest.mark.asyncio
    async def test_get_memory_context_disabled(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test memory context when mem0 is disabled."""
        bot = TelegramBot(telegram_config, mock_model)

        with patch("blacki.telegram.bot.is_mem0_enabled", return_value=False):
            result = await bot._get_memory_context("123", "test query")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_memory_context_with_memories(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test memory context with memories found."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {
            "memories": [
                {"memory": "User likes Python"},
                {"memory": "User is a developer"},
            ]
        }

        with (
            patch("blacki.telegram.bot.is_mem0_enabled", return_value=True),
            patch("blacki.telegram.bot.get_mem0_manager", return_value=mock_manager),
        ):
            result = await bot._get_memory_context("123", "test query")

        assert "Context from memory" in result
        assert "User likes Python" in result

    @pytest.mark.asyncio
    async def test_get_memory_context_no_memories(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test memory context with no memories found."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": []}

        with (
            patch("blacki.telegram.bot.is_mem0_enabled", return_value=True),
            patch("blacki.telegram.bot.get_mem0_manager", return_value=mock_manager),
        ):
            result = await bot._get_memory_context("123", "test query")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_memory_context_error(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test memory context handles errors gracefully."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_manager = MagicMock()
        mock_manager.search_memory.side_effect = Exception("DB error")

        with (
            patch("blacki.telegram.bot.is_mem0_enabled", return_value=True),
            patch("blacki.telegram.bot.get_mem0_manager", return_value=mock_manager),
        ):
            result = await bot._get_memory_context("123", "test query")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_memory_context_non_dict_memory(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test memory context with non-dict memory items."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {
            "memories": [
                "plain string memory",  # Not a dict
                {"text": "wrong key"},  # Dict without 'memory' key
            ]
        }

        with (
            patch("blacki.telegram.bot.is_mem0_enabled", return_value=True),
            patch("blacki.telegram.bot.get_mem0_manager", return_value=mock_manager),
        ):
            result = await bot._get_memory_context("123", "test query")

        # Should return empty since no valid memories
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_memory_context_empty_memory_texts(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test memory context when all memories are invalid."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {
            "memories": [
                {"text": "no memory key"},  # Dict without 'memory' key
            ]
        }

        with (
            patch("blacki.telegram.bot.is_mem0_enabled", return_value=True),
            patch("blacki.telegram.bot.get_mem0_manager", return_value=mock_manager),
        ):
            result = await bot._get_memory_context("123", "test query")

        # Should return empty since memory_texts is empty
        assert result == ""


class TestTelegramBotBuildPrompt:
    """Tests for _build_prompt."""

    def test_build_prompt_basic(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test basic prompt building."""
        bot = TelegramBot(telegram_config, mock_model)

        result = bot._build_prompt(
            instruction="Be helpful",
            memory_context="",
            user_message="Hello",
        )

        assert "Be helpful" in result
        assert "User: Hello" in result

    def test_build_prompt_with_memory(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test prompt building with memory context."""
        bot = TelegramBot(telegram_config, mock_model)

        result = bot._build_prompt(
            instruction="Be helpful",
            memory_context="Previous: User likes Python",
            user_message="What language should I learn?",
        )

        assert "Be helpful" in result
        assert "Previous: User likes Python" in result
        assert "User: What language should I learn?" in result


class TestTelegramBotGenerateResponse:
    """Tests for _generate_response."""

    @pytest.mark.asyncio
    async def test_generate_response_with_string_model(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test response generation with string model."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await bot._generate_response("Test prompt")

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_response_with_model_object(
        self, telegram_config: TelegramConfig, mock_model_object: MagicMock
    ) -> None:
        """Test response generation with model object."""
        bot = TelegramBot(telegram_config, mock_model_object)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response from object"

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await bot._generate_response("Test prompt")

        assert result == "Response from object"

    @pytest.mark.asyncio
    async def test_generate_response_no_choices(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test response generation with no choices."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_response = MagicMock()
        mock_response.choices = []

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await bot._generate_response("Test prompt")

        assert "couldn't generate" in result

    @pytest.mark.asyncio
    async def test_generate_response_error(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test response generation handles errors."""
        bot = TelegramBot(telegram_config, mock_model)

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await bot._generate_response("Test prompt")

        assert "error" in result.lower()


class TestTelegramBotSaveToMemory:
    """Tests for _save_to_memory."""

    @pytest.mark.asyncio
    async def test_save_to_memory_disabled(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test save to memory when mem0 is disabled."""
        bot = TelegramBot(telegram_config, mock_model)

        with patch("blacki.telegram.bot.is_mem0_enabled", return_value=False):
            await bot._save_to_memory("123", "Hello", "Hi there")

    @pytest.mark.asyncio
    async def test_save_to_memory_success(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test successful save to memory."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_manager = MagicMock()

        with (
            patch("blacki.telegram.bot.is_mem0_enabled", return_value=True),
            patch("blacki.telegram.bot.get_mem0_manager", return_value=mock_manager),
        ):
            await bot._save_to_memory("123", "Hello", "Hi there")

        mock_manager.save_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_to_memory_error(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test save to memory handles errors gracefully."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_manager = MagicMock()
        mock_manager.save_memory.side_effect = Exception("DB error")

        with (
            patch("blacki.telegram.bot.is_mem0_enabled", return_value=True),
            patch("blacki.telegram.bot.get_mem0_manager", return_value=mock_manager),
        ):
            # Should not raise
            await bot._save_to_memory("123", "Hello", "Hi there")


class TestTelegramBotStartPolling:
    """Tests for start_polling."""

    @pytest.mark.asyncio
    async def test_start_polling_not_configured(
        self, telegram_config_disabled: TelegramConfig, mock_model: str
    ) -> None:
        """Test start_polling when not configured."""
        bot = TelegramBot(telegram_config_disabled, mock_model)

        await bot.start_polling()

    @pytest.mark.asyncio
    async def test_start_polling_success(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test successful start_polling."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_app = MagicMock(spec=Application)
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.updater = MagicMock()
        mock_app.updater.start_polling = AsyncMock()

        with patch.object(bot, "_ensure_app", return_value=mock_app):
            await bot.start_polling()

        mock_app.initialize.assert_called_once()
        mock_app.start.assert_called_once()
        mock_app.updater.start_polling.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_polling_no_updater(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test start_polling when app has no updater."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_app = MagicMock(spec=Application)
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.updater = None

        with patch.object(bot, "_ensure_app", return_value=mock_app):
            await bot.start_polling()

        mock_app.initialize.assert_called_once()
        mock_app.start.assert_called_once()


class TestTelegramBotStop:
    """Tests for stop."""

    @pytest.mark.asyncio
    async def test_stop_no_app(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test stop when no app is initialized."""
        bot = TelegramBot(telegram_config, mock_model)

        await bot.stop()

    @pytest.mark.asyncio
    async def test_stop_with_app(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test stop with initialized app."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_app = MagicMock(spec=Application)
        mock_app.updater = MagicMock()
        mock_app.updater.stop = AsyncMock()
        mock_app.stop = AsyncMock()
        mock_app.shutdown = AsyncMock()

        bot._app = mock_app

        await bot.stop()

        mock_app.updater.stop.assert_called_once()
        mock_app.stop.assert_called_once()
        mock_app.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_no_updater(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test stop when app has no updater."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_app = MagicMock(spec=Application)
        mock_app.updater = None
        mock_app.stop = AsyncMock()
        mock_app.shutdown = AsyncMock()

        bot._app = mock_app

        await bot.stop()

        mock_app.stop.assert_called_once()
        mock_app.shutdown.assert_called_once()


class TestCreateTelegramBot:
    """Tests for create_telegram_bot factory function."""

    def test_create_bot_configured(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test create bot when configured."""
        result = create_telegram_bot(telegram_config, mock_model)

        assert result is not None
        assert isinstance(result, TelegramBot)

    def test_create_bot_not_configured(
        self, telegram_config_disabled: TelegramConfig, mock_model: str
    ) -> None:
        """Test create bot when not configured."""
        result = create_telegram_bot(telegram_config_disabled, mock_model)

        assert result is None


class TestTelegramBotTypingIndicator:
    """Tests for typing indicator functionality."""

    @pytest.mark.asyncio
    async def test_start_typing_indicator_creates_task(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _start_typing_indicator creates a task."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_app = MagicMock()
        mock_app.bot.send_chat_action = AsyncMock()
        bot._app = mock_app

        task = bot._start_typing_indicator(12345)

        assert task is not None
        assert not task.done()  # Task should be running
        # Cancel the task to clean up
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_stop_typing_indicator_cancels_task(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _stop_typing_indicator cancels the task."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_app = MagicMock()
        mock_app.bot.send_chat_action = AsyncMock()
        bot._app = mock_app

        task = bot._start_typing_indicator(12345)
        await bot._stop_typing_indicator(task)

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_typing_indicator_in_handle_message(
        self,
        telegram_config: TelegramConfig,
        mock_model: str,
        mock_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test that typing indicator is started and stopped during message handling."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_app = MagicMock()
        mock_app.bot.send_chat_action = AsyncMock()
        bot._app = mock_app

        with (
            patch.object(bot, "_get_memory_context", return_value=""),
            patch.object(bot, "_generate_response", return_value="Test response"),
            patch.object(bot, "_save_to_memory"),
            patch.object(bot, "_start_typing_indicator") as mock_start,
            patch.object(bot, "_stop_typing_indicator") as mock_stop,
        ):
            mock_start.return_value = asyncio.create_task(asyncio.sleep(10))

            await bot._handle_message(mock_update, mock_context)

            mock_start.assert_called_once_with(123456789)
            mock_stop.assert_called_once()


class TestTelegramBotSessionManagement:
    """Tests for session management functionality."""

    def test_get_session_user_id_creates_session(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _get_session_user_id creates a session if not exists."""
        bot = TelegramBot(telegram_config, mock_model)

        user_id = bot._get_session_user_id("123")

        assert "123" in bot._session_ids
        assert user_id.startswith("123_")
        assert len(user_id.split("_")[1]) == 8  # 8-char session ID

    def test_get_session_user_id_returns_same_session(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _get_session_user_id returns same session on repeated calls."""
        bot = TelegramBot(telegram_config, mock_model)

        user_id1 = bot._get_session_user_id("123")
        user_id2 = bot._get_session_user_id("123")

        assert user_id1 == user_id2

    def test_reset_session_creates_new_session(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that _reset_session creates a new session ID."""
        bot = TelegramBot(telegram_config, mock_model)

        old_user_id = bot._get_session_user_id("123")
        bot._reset_session("123")
        new_user_id = bot._get_session_user_id("123")

        assert old_user_id != new_user_id

    def test_session_isolation_per_chat(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that different chats have different sessions."""
        bot = TelegramBot(telegram_config, mock_model)

        user_id1 = bot._get_session_user_id("123")
        user_id2 = bot._get_session_user_id("456")

        assert user_id1 != user_id2
        assert user_id1.startswith("123_")
        assert user_id2.startswith("456_")

    @pytest.mark.asyncio
    async def test_memory_uses_session_user_id(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that memory operations use session-scoped user_id."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_manager = MagicMock()
        mock_manager.search_memory.return_value = {"memories": []}
        mock_manager.save_memory.return_value = {"status": "success"}

        with (
            patch("blacki.telegram.bot.is_mem0_enabled", return_value=True),
            patch("blacki.telegram.bot.get_mem0_manager", return_value=mock_manager),
        ):
            # Test search uses session user_id
            await bot._get_memory_context("123", "test query")
            call_args = mock_manager.search_memory.call_args
            assert call_args.kwargs["user_id"].startswith("123_")

            # Test save uses session user_id
            await bot._save_to_memory("123", "Hello", "Hi")
            call_args = mock_manager.save_memory.call_args
            assert call_args.kwargs["user_id"].startswith("123_")


class TestTelegramBotCommandRegistration:
    """Tests for command registration with Telegram."""

    @pytest.mark.asyncio
    async def test_register_commands_called_on_start(
        self, telegram_config: TelegramConfig, mock_model: str
    ) -> None:
        """Test that commands are registered when polling starts."""
        bot = TelegramBot(telegram_config, mock_model)

        mock_app = MagicMock(spec=Application)
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.updater = MagicMock()
        mock_app.updater.start_polling = AsyncMock()
        mock_app.bot = MagicMock()
        mock_app.bot.set_my_commands = AsyncMock()

        with patch.object(bot, "_ensure_app", return_value=mock_app):
            await bot.start_polling()

        mock_app.bot.set_my_commands.assert_called_once()
        # Verify commands are correct
        call_args = mock_app.bot.set_my_commands.call_args[0][0]
        command_names = [cmd.command for cmd in call_args]
        assert "start" in command_names
        assert "help" in command_names
        assert "reset" in command_names
