# mypy: ignore-errors
"""Tests for TelegramBot's delegation to the settings menu and profile loading.

Settings-menu behavior itself (model/thinking panels, callback handling,
reasoning options, etc.) is tested directly against SettingsMenu in
test_telegram_settings_menu.py. This file only covers the thin TelegramBot
glue: command dispatch, callback routing, and the shared profile loader.
"""

from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest

from blacki.models.inference import InferenceProfile, ReasoningConfig, ReasoningEffort
from blacki.runtime.adk import AdkRuntime
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import CallbackQuery, Chat, Message, Update, User


@pytest.fixture
def telegram_config() -> TelegramConfig:
    return TelegramConfig(
        telegram_enabled=True,
        telegram_bot_token="test-token",
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
async def test_safe_handle_update_callback_query(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(id="cq1", from_user=user, chat_instance="inst1", data="mod:m1")
    update = Update(update_id=1, callback_query=cq)

    with patch.object(bot, "_handle_callback_query", AsyncMock()) as mock_handle:
        await bot._safe_handle_update(update)
        mock_handle.assert_called_once_with(cq)


@pytest.mark.asyncio
async def test_handle_command_model(bot: TelegramBot) -> None:
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)

    with patch.object(
        bot._settings_menu, "send_model_menu", AsyncMock()
    ) as mock_send_menu:
        await bot._handle_command(msg, "/model")
        mock_send_menu.assert_called_once_with(123, None)


@pytest.mark.asyncio
async def test_handle_command_thinking(bot: TelegramBot) -> None:
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)

    with patch.object(
        bot._settings_menu, "send_thinking_menu", AsyncMock()
    ) as mock_send_menu:
        await bot._handle_command(msg, "/thinking")
        mock_send_menu.assert_awaited_once_with(123, None)


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
async def test_handle_callback_query_routes_health_data(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="health:cancel"
    )

    with patch.object(bot, "_handle_health_callback", AsyncMock()) as mock_health:
        await bot._handle_callback_query(cq)
        mock_health.assert_awaited_once_with(cq)


@pytest.mark.asyncio
async def test_handle_callback_query_routes_settings_data(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(id="cq1", from_user=user, chat_instance="inst", data="s:b")

    with patch.object(bot._settings_menu, "handle_callback", AsyncMock()) as mock_menu:
        await bot._handle_callback_query(cq)
        mock_menu.assert_awaited_once_with(cq)


@pytest.mark.asyncio
async def test_load_chat_profile_uses_environment_after_storage_error(
    bot: TelegramBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")

    with patch(
        "blacki.telegram.bot.get_preferences_storage",
        side_effect=RuntimeError("storage unavailable"),
    ):
        profile = await bot._load_chat_profile(123)

    assert profile == InferenceProfile(
        reasoning=ReasoningConfig(effort=ReasoningEffort.MAX)
    )


@pytest.mark.asyncio
async def test_load_chat_profile_uses_environment_after_invalid_result(
    bot: TelegramBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")

    with (
        patch("blacki.telegram.bot.get_preferences_storage", return_value=AsyncMock()),
        patch(
            "blacki.telegram.bot.load_inference_profile",
            AsyncMock(return_value=None),
        ),
    ):
        profile = await bot._load_chat_profile(123)

    assert profile == InferenceProfile(
        reasoning=ReasoningConfig(effort=ReasoningEffort.MAX)
    )
