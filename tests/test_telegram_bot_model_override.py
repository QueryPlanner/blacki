# mypy: ignore-errors
"""Tests for Telegram bot model override and callback queries."""

from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest

from blacki.adk_runtime import AdkRuntime
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import CallbackQuery, Chat, Message, Update, User


@pytest.fixture
def telegram_config() -> TelegramConfig:
    return TelegramConfig(
        telegram_enabled=True,
        telegram_bot_token="test-token",
        telegram_tool_notifications=False,
    )


@pytest.fixture
def mock_runtime() -> MagicMock:
    runtime = create_autospec(AdkRuntime, instance=True)
    return runtime


@pytest.fixture
def bot(telegram_config: TelegramConfig, mock_runtime: MagicMock) -> TelegramBot:
    bot_inst = TelegramBot(telegram_config, mock_runtime)
    mock_api = create_autospec(TelegramApiClient, instance=True)
    bot_inst._api = mock_api
    return bot_inst


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
async def test_send_model_menu_success(mock_get_prefs, bot: TelegramBot) -> None:
    mock_storage = AsyncMock()
    mock_storage.get.return_value = "openrouter/deepseek/deepseek-v4-pro"
    mock_get_prefs.return_value = mock_storage

    await bot._send_model_menu(chat_id=123, message_thread_id=None)
    bot._api.send_message.assert_called_once()
    kwargs = bot._api.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 123
    assert "reply_markup" in kwargs


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
async def test_send_model_menu_exception(mock_get_prefs, bot: TelegramBot) -> None:
    mock_storage = AsyncMock()
    mock_storage.get.return_value = None
    mock_get_prefs.return_value = mock_storage

    bot._api.send_message.side_effect = Exception("failed")
    # Should not raise
    await bot._send_model_menu(chat_id=123, message_thread_id=None)


@pytest.mark.asyncio
async def test_safe_handle_update_callback_query(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(id="cq1", from_user=user, chat_instance="inst1", data="mod:m1")
    update = Update(update_id=1, callback_query=cq)

    with patch.object(bot, "_handle_callback_query", AsyncMock()) as mock_handle:
        await bot._safe_handle_update(update)
        mock_handle.assert_called_once_with(cq)


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
@patch("blacki.telegram.bot.MODEL_CHOICES", {"m1": ("m1", "M1"), "m2": ("m2", "M2")})
async def test_send_model_menu_even_choices(mock_get_prefs, bot: TelegramBot) -> None:
    mock_storage = AsyncMock()
    mock_storage.get.return_value = "m1"
    mock_get_prefs.return_value = mock_storage

    await bot._send_model_menu(chat_id=123, message_thread_id=None)
    bot._api.send_message.assert_called_once()


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
@patch("blacki.telegram.bot.MODEL_CHOICES", {"m1": ("m1", "M1")})
async def test_send_model_menu_unknown_current_model(
    mock_get_prefs, bot: TelegramBot
) -> None:
    mock_storage = AsyncMock()
    mock_storage.get.return_value = "unknown_model_id"
    mock_get_prefs.return_value = mock_storage

    await bot._send_model_menu(chat_id=123, message_thread_id=None)
    bot._api.send_message.assert_called_once()
    kwargs = bot._api.send_message.call_args.kwargs
    assert "System Default" in kwargs["text"]
    bot._api.send_message.assert_called_once()


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
async def test_handle_callback_query_no_message(
    mock_get_prefs, bot: TelegramBot
) -> None:
    mock_storage = AsyncMock()
    mock_get_prefs.return_value = mock_storage
    user = User(id=1, is_bot=False, first_name="Test")
    # Message is None
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="mod:m1", message=None
    )

    await bot._handle_callback_query(cq)
    bot._api.answer_callback_query.assert_called_once()
    bot._api.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_handle_command_model(bot: TelegramBot) -> None:
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)

    with patch.object(bot, "_send_model_menu", AsyncMock()) as mock_send_menu:
        await bot._handle_command(msg, "/model")
        mock_send_menu.assert_called_once_with(123, None)


@pytest.mark.asyncio
async def test_build_session_state_no_pref(bot: TelegramBot) -> None:
    state = bot._build_session_state(
        chat_id="123", message_thread_id=None, conversation_key="k"
    )
    assert "telegram_model_override" not in state
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(id="cq1", from_user=user, chat_instance="inst1", data="mod:m1")
    update = Update(update_id=1, callback_query=cq)

    # Try throwing a ValueError
    with patch.object(
        bot, "_handle_callback_query", AsyncMock(side_effect=ValueError("fail"))
    ):
        await bot._safe_handle_update(update)


@pytest.mark.asyncio
async def test_handle_callback_query_invalid_data(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(id="cq1", from_user=user, chat_instance="inst", data="invalid")
    await bot._handle_callback_query(cq)
    bot._api.answer_callback_query.assert_called_once_with("cq1", text="Unknown action")


@pytest.mark.asyncio
async def test_handle_callback_query_unknown_model(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="mod:unknown"
    )
    await bot._handle_callback_query(cq)
    bot._api.answer_callback_query.assert_called_once_with("cq1", text="Unknown model")


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
async def test_handle_callback_query_valid_model(
    mock_get_prefs, bot: TelegramBot
) -> None:
    mock_storage = AsyncMock()
    mock_get_prefs.return_value = mock_storage
    user = User(id=1, is_bot=False, first_name="Test")
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="mod:m1", message=msg
    )

    await bot._handle_callback_query(cq)
    mock_storage.set.assert_called_once_with(
        "123", "telegram_model_override", "openrouter/openai/gpt-oss-120b"
    )
    bot._api.answer_callback_query.assert_called_once()
    bot._api.edit_message_text.assert_called_once()


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
async def test_handle_callback_query_default_model(
    mock_get_prefs, bot: TelegramBot
) -> None:
    mock_storage = AsyncMock()
    mock_get_prefs.return_value = mock_storage
    user = User(id=1, is_bot=False, first_name="Test")
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)
    cq = CallbackQuery(
        id="cq1",
        from_user=user,
        chat_instance="inst",
        data="mod:m_default",
        message=msg,
    )

    await bot._handle_callback_query(cq)
    mock_storage.delete.assert_called_once_with("123", "telegram_model_override")


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
async def test_handle_callback_query_edit_msg_exception(
    mock_get_prefs, bot: TelegramBot
) -> None:
    mock_storage = AsyncMock()
    mock_get_prefs.return_value = mock_storage
    user = User(id=1, is_bot=False, first_name="Test")
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="mod:m1", message=msg
    )

    bot._api.edit_message_text.side_effect = Exception("fail")
    await bot._handle_callback_query(cq)
    bot._api.answer_callback_query.assert_called_once()
