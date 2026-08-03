# mypy: ignore-errors
"""Tests for Telegram bot model override and callback queries."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest

from blacki.adk_runtime import AdkRuntime
from blacki.inference import InferenceProfile, ReasoningConfig, ReasoningEffort
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import (
    CallbackQuery,
    Chat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ParseMode,
    Update,
    User,
)


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

    with patch.dict(os.environ, {"ROOT_AGENT_MODEL": "default"}):
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

    with patch(
        "blacki.telegram.bot.update_inference_profile", new=AsyncMock()
    ) as mock_update:
        await bot._handle_callback_query(cq)
    mock_update.assert_awaited_once_with(
        mock_storage,
        "123",
        {"model": "openrouter/openai/gpt-oss-120b", "reasoning": None},
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

    with patch(
        "blacki.telegram.bot.update_inference_profile", new=AsyncMock()
    ) as mock_update:
        await bot._handle_callback_query(cq)
    mock_update.assert_awaited_once_with(
        mock_storage,
        "123",
        {"model": None, "reasoning": None},
    )


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
    with patch("blacki.telegram.bot.update_inference_profile", new=AsyncMock()):
        await bot._handle_callback_query(cq)
    bot._api.answer_callback_query.assert_called_once()


@pytest.mark.asyncio
async def test_handle_command_thinking(bot: TelegramBot) -> None:
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)

    with patch.object(bot, "_send_thinking_menu", AsyncMock()) as mock_send_menu:
        await bot._handle_command(msg, "/thinking")
        mock_send_menu.assert_awaited_once_with(123, None)


def test_settings_callback_data_is_within_telegram_limit(bot: TelegramBot) -> None:
    text, markup = bot._build_model_menu(InferenceProfile())
    assert text
    callback_data = [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data is not None
    ]
    assert callback_data
    assert all(len(value.encode("utf-8")) <= 64 for value in callback_data)


@pytest.mark.asyncio
async def test_message_less_settings_mutation_does_not_write(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:m:m1", message=None
    )

    with patch(
        "blacki.telegram.bot.update_inference_profile", new=AsyncMock()
    ) as update:
        await bot._handle_callback_query(cq)

    update.assert_not_awaited()
    bot._api.answer_callback_query.assert_called_once_with(
        "cq1", text="Settings expired"
    )


@pytest.mark.asyncio
@patch("blacki.telegram.bot.get_preferences_storage")
@patch(
    "blacki.telegram.bot.load_inference_profile",
    new_callable=AsyncMock,
    return_value=InferenceProfile(
        model="openrouter/openai/gpt-oss-120b",
        reasoning=ReasoningConfig(effort=ReasoningEffort.HIGH),
    ),
)
async def test_reasoning_callback_preserves_model(
    mock_load, mock_get_prefs, bot: TelegramBot
) -> None:
    mock_storage = AsyncMock()
    mock_get_prefs.return_value = mock_storage
    user = User(id=1, is_bot=False, first_name="Test")
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:r:max", message=msg
    )
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=("low", "high", "max"),
            mandatory=False,
        )
    )

    with (
        patch.object(bot, "_resolve_capabilities", AsyncMock(return_value=capability)),
        patch(
            "blacki.telegram.bot.update_inference_profile", new=AsyncMock()
        ) as update,
    ):
        await bot._handle_callback_query(cq)

    update.assert_awaited_once_with(
        mock_storage,
        "123",
        {"reasoning": ReasoningConfig(effort=ReasoningEffort.MAX)},
    )


@pytest.mark.asyncio
async def test_reasoning_menu_hides_off_for_mandatory_model(bot: TelegramBot) -> None:
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=("none", "high", "max"),
            mandatory=True,
        )
    )
    with patch.object(bot, "_resolve_capabilities", AsyncMock(return_value=capability)):
        _, markup = await bot._build_thinking_menu(InferenceProfile())

    labels = [button.text for row in markup.inline_keyboard for button in row]
    assert not any(label.startswith("Off") for label in labels)
    assert any(label.startswith("High") for label in labels)


@pytest.mark.asyncio
async def test_thinking_menu_falls_back_when_capability_client_fails(
    bot: TelegramBot,
) -> None:
    with patch(
        "blacki.telegram.bot.OpenRouterModelCapabilitiesResolver",
        side_effect=RuntimeError("capability client unavailable"),
    ):
        await bot._send_thinking_menu(chat_id=123, message_thread_id=None)

    bot._api.send_message.assert_awaited_once()
    markup = bot._api.send_message.call_args.kwargs["reply_markup"]
    callback_data = [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data is not None
    ]
    assert callback_data == ["s:r:inherit", "s:b"]


@pytest.mark.asyncio
async def test_stale_reasoning_callback_does_not_write(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:r:not-real", message=msg
    )

    with patch(
        "blacki.telegram.bot.update_inference_profile", new=AsyncMock()
    ) as update:
        await bot._handle_callback_query(cq)

    update.assert_not_awaited()
    bot._api.answer_callback_query.assert_called_once_with(
        "cq1", text="Unknown thinking option"
    )


@pytest.mark.asyncio
async def test_stop_closes_capability_resolver(bot: TelegramBot) -> None:
    resolver = AsyncMock()
    bot._capabilities_resolver = resolver
    bot.runtime.close = AsyncMock()
    bot._api.close = AsyncMock()

    await bot.stop()

    resolver.aclose.assert_awaited_once()
    assert bot._capabilities_resolver is None


@pytest.mark.asyncio
async def test_stop_suppresses_capability_resolver_close_error(
    bot: TelegramBot,
) -> None:
    resolver = AsyncMock()
    resolver.aclose.side_effect = RuntimeError("close failed")
    bot._capabilities_resolver = resolver
    bot.runtime.close = AsyncMock()
    bot._api.close = AsyncMock()

    await bot.stop()

    resolver.aclose.assert_awaited_once()
    assert bot._capabilities_resolver is None


@pytest.mark.asyncio
async def test_send_thinking_menu_handles_send_error(bot: TelegramBot) -> None:
    bot._api.send_message.side_effect = RuntimeError("send failed")

    await bot._send_thinking_menu(chat_id=123, message_thread_id=None)

    bot._api.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_thinking_menu_even_options_has_no_partial_row(bot: TelegramBot) -> None:
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=("max",),
            mandatory=False,
        )
    )
    with patch.object(bot, "_resolve_capabilities", AsyncMock(return_value=capability)):
        _, markup = await bot._build_thinking_menu(InferenceProfile())

    assert [button.callback_data for button in markup.inline_keyboard[0]] == [
        "s:r:inherit",
        "s:r:max",
    ]
    assert markup.inline_keyboard[-1][0].callback_data == "s:b"


@pytest.mark.asyncio
async def test_thinking_menu_notes_when_effort_is_unsupported(
    bot: TelegramBot,
) -> None:
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=False,
            supported_efforts=(),
            mandatory=False,
        )
    )
    with patch.object(bot, "_resolve_capabilities", AsyncMock(return_value=capability)):
        text, markup = await bot._build_thinking_menu(InferenceProfile())

    assert "does not expose effort controls" in text
    assert [
        button.callback_data for row in markup.inline_keyboard for button in row
    ] == [
        "s:r:inherit",
        "s:b",
    ]


@pytest.mark.asyncio
async def test_callback_model_none_value_is_rejected_without_write(
    bot: TelegramBot,
) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    message = Message(
        message_id=42,
        date="2024-01-01T00:00:00Z",
        chat=Chat(id=123, type="private"),
    )
    query = CallbackQuery(
        id="cq1",
        from_user=user,
        chat_instance="inst",
        data="s:m:m1",
        message=message,
    )

    with (
        patch.object(bot, "_parse_settings_callback", return_value=("model", None)),
        patch(
            "blacki.telegram.bot.update_inference_profile", new=AsyncMock()
        ) as update,
    ):
        await bot._handle_callback_query(query)

    update.assert_not_awaited()
    assert bot._api.answer_callback_query.await_args_list[-1].kwargs["text"] == (
        "Unknown model"
    )


@pytest.mark.asyncio
async def test_callback_rejects_effort_not_supported_by_model(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    message = Message(
        message_id=42,
        date="2024-01-01T00:00:00Z",
        chat=Chat(id=123, type="private"),
    )
    query = CallbackQuery(
        id="cq1",
        from_user=user,
        chat_instance="inst",
        data="s:r:max",
        message=message,
    )
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=("low",),
            mandatory=False,
        )
    )
    mock_storage = AsyncMock()

    with (
        patch("blacki.telegram.bot.get_preferences_storage", return_value=mock_storage),
        patch.object(bot, "_resolve_capabilities", AsyncMock(return_value=capability)),
        patch(
            "blacki.telegram.bot.update_inference_profile", new=AsyncMock()
        ) as update,
    ):
        await bot._handle_callback_query(query)

    update.assert_not_awaited()
    assert bot._api.edit_message_text.await_count == 1


@pytest.mark.asyncio
async def test_reset_callback_updates_both_profile_fields(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    message = Message(
        message_id=42,
        date="2024-01-01T00:00:00Z",
        chat=Chat(id=123, type="private"),
    )
    query = CallbackQuery(
        id="cq1",
        from_user=user,
        chat_instance="inst",
        data="s:x",
        message=message,
    )

    mock_storage = AsyncMock()
    with (
        patch("blacki.telegram.bot.get_preferences_storage", return_value=mock_storage),
        patch(
            "blacki.telegram.bot.update_inference_profile", new=AsyncMock()
        ) as update,
    ):
        await bot._handle_callback_query(query)

    update.assert_awaited_once_with(
        mock_storage,
        "123",
        {"model": None, "reasoning": None},
    )


@pytest.mark.asyncio
async def test_thinking_callback_edits_capability_menu(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    message = Message(
        message_id=42,
        date="2024-01-01T00:00:00Z",
        chat=Chat(id=123, type="private"),
    )
    query = CallbackQuery(
        id="cq1",
        from_user=user,
        chat_instance="inst",
        data="s:t",
        message=message,
    )
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Default", callback_data="s:r:inherit")]
        ]
    )
    with (
        patch("blacki.telegram.bot.get_preferences_storage", return_value=AsyncMock()),
        patch.object(
            bot,
            "_build_thinking_menu",
            AsyncMock(return_value=("thinking", markup)),
        ),
    ):
        await bot._handle_callback_query(query)

    bot._api.edit_message_text.assert_awaited_once_with(
        chat_id=123,
        message_id=42,
        text="thinking",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=markup,
    )


@pytest.mark.asyncio
async def test_back_callback_returns_to_model_menu(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    message = Message(
        message_id=42,
        date="2024-01-01T00:00:00Z",
        chat=Chat(id=123, type="private"),
    )
    query = CallbackQuery(
        id="cq1",
        from_user=user,
        chat_instance="inst",
        data="s:b",
        message=message,
    )

    with (
        patch("blacki.telegram.bot.get_preferences_storage", return_value=AsyncMock()),
        patch.object(bot, "_edit_model_menu", AsyncMock()) as edit_menu,
    ):
        await bot._handle_callback_query(query)

    edit_menu.assert_awaited_once_with(query, 123)


def test_settings_callback_parser_handles_navigation_actions() -> None:
    assert TelegramBot._parse_settings_callback("s:t") == ("thinking", None)
    assert TelegramBot._parse_settings_callback("s:b") == ("back", None)
    assert TelegramBot._parse_settings_callback("s:x") == ("reset", None)


@pytest.mark.asyncio
async def test_edit_model_menu_ignores_message_less_callback(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    query = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:b", message=None
    )

    await bot._edit_model_menu(query, 123)

    bot._api.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_error_ignores_message_less_callback(bot: TelegramBot) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    query = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:b", message=None
    )

    await bot._edit_error(query, 123, "failed")

    bot._api.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_capabilities_skips_missing_models(bot: TelegramBot) -> None:
    assert await bot._resolve_capabilities(None) is None
    assert await bot._resolve_capabilities("default") is None


@pytest.mark.asyncio
async def test_resolve_capabilities_uses_cached_resolver(bot: TelegramBot) -> None:
    resolver = AsyncMock()
    resolver.resolve.return_value = None
    bot._capabilities_resolver = resolver

    assert await bot._resolve_capabilities("openrouter/openai/gpt-5.6-luna") is None

    resolver.resolve.assert_awaited_once()
    assert resolver.resolve.await_args.args == ("openrouter/openai/gpt-5.6-luna",)


def test_reasoning_display_inherits_when_only_token_budget_is_set(
    bot: TelegramBot,
) -> None:
    profile = InferenceProfile(reasoning=ReasoningConfig(max_tokens=256))

    assert bot._reasoning_display(profile) == "Default"


def test_reasoning_options_include_gateway_values_when_unspecified() -> None:
    bot = TelegramBot.__new__(TelegramBot)
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=None,
            mandatory=False,
        )
    )

    options = bot._reasoning_options(capability)

    assert ("max", "Max") in options
    assert ("none", "Off") in options


def test_reasoning_options_skip_empty_and_inherit_values() -> None:
    bot = TelegramBot.__new__(TelegramBot)
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=(None, "inherit", "max"),
            mandatory=False,
        )
    )

    assert bot._reasoning_options(capability) == [
        ("inherit", "Default"),
        ("max", "Max"),
    ]


def test_reasoning_config_handles_inherit_and_invalid_values() -> None:
    assert TelegramBot._reasoning_config("inherit") is None
    assert TelegramBot._reasoning_config("not-an-effort") is None
