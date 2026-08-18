# mypy: ignore-errors
"""Tests for the standalone Telegram settings-menu UI (model/thinking panels)."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import aiosqlite
import pytest

from blacki.inference import (
    INFERENCE_PROFILE_PREFERENCE_KEY,
    LEGACY_MODEL_PREFERENCE_KEY,
    InferenceProfile,
    ReasoningConfig,
    ReasoningEffort,
    update_inference_profile,
)
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.settings_menu import SettingsMenu
from blacki.telegram.types import (
    CallbackQuery,
    Chat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ParseMode,
    User,
)
from blacki.utils.preferences import SqlitePreferencesStorage


@pytest.fixture
def mock_api() -> MagicMock:
    return create_autospec(TelegramApiClient, instance=True)


@pytest.fixture
def load_profile() -> AsyncMock:
    return AsyncMock(return_value=InferenceProfile())


@pytest.fixture
def menu(mock_api: MagicMock, load_profile: AsyncMock) -> SettingsMenu:
    return SettingsMenu(api_provider=lambda: mock_api, load_profile=load_profile)


@pytest.mark.asyncio
@patch("blacki.telegram.settings_menu.get_preferences_storage")
async def test_send_model_menu_success(
    mock_get_prefs, menu: SettingsMenu, mock_api: MagicMock
) -> None:
    mock_storage = AsyncMock()
    mock_storage.get.return_value = "openrouter/deepseek/deepseek-v4-pro"
    mock_get_prefs.return_value = mock_storage

    await menu.send_model_menu(chat_id=123, message_thread_id=None)
    mock_api.send_message.assert_called_once()
    kwargs = mock_api.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 123
    assert "reply_markup" in kwargs


@pytest.mark.asyncio
async def test_send_model_menu_exception(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    mock_api.send_message.side_effect = Exception("failed")
    # Should not raise
    await menu.send_model_menu(chat_id=123, message_thread_id=None)


@pytest.mark.asyncio
@patch(
    "blacki.telegram.settings_menu.MODEL_CHOICES",
    {"m1": ("m1", "M1"), "m2": ("m2", "M2")},
)
async def test_send_model_menu_even_choices(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    await menu.send_model_menu(chat_id=123, message_thread_id=None)
    mock_api.send_message.assert_called_once()


@pytest.mark.asyncio
@patch("blacki.telegram.settings_menu.MODEL_CHOICES", {"m1": ("m1", "M1")})
async def test_send_model_menu_unknown_current_model(
    menu: SettingsMenu, mock_api: MagicMock, load_profile: AsyncMock
) -> None:
    load_profile.return_value = InferenceProfile(model="unknown_model_id")

    with patch.dict(os.environ, {"ROOT_AGENT_MODEL": "default"}):
        await menu.send_model_menu(chat_id=123, message_thread_id=None)
    mock_api.send_message.assert_called_once()
    kwargs = mock_api.send_message.call_args.kwargs
    assert "unknown\\_model\\_id" in kwargs["text"]


@pytest.mark.asyncio
async def test_handle_callback_query_no_message(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    # Message is None
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="mod:m1", message=None
    )

    await menu.handle_callback(cq)
    mock_api.answer_callback_query.assert_called_once()
    mock_api.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_handle_callback_query_invalid_data(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(id="cq1", from_user=user, chat_instance="inst", data="invalid")
    await menu.handle_callback(cq)
    mock_api.answer_callback_query.assert_called_once_with("cq1", text="Unknown action")


@pytest.mark.asyncio
async def test_handle_callback_query_unknown_model(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="mod:unknown"
    )
    await menu.handle_callback(cq)
    mock_api.answer_callback_query.assert_called_once_with("cq1", text="Unknown model")


@pytest.mark.asyncio
@patch("blacki.telegram.settings_menu.get_preferences_storage")
async def test_handle_callback_query_valid_model(
    mock_get_prefs, menu: SettingsMenu, mock_api: MagicMock
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
        "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
    ) as mock_update:
        await menu.handle_callback(cq)
    mock_update.assert_awaited_once_with(
        mock_storage,
        "123",
        {"model": "openrouter/openai/gpt-oss-120b", "reasoning": None},
    )
    mock_api.answer_callback_query.assert_called_once()
    mock_api.edit_message_text.assert_called_once()


@pytest.mark.asyncio
@patch("blacki.telegram.settings_menu.get_preferences_storage")
async def test_handle_callback_query_default_model(
    mock_get_prefs, menu: SettingsMenu, mock_api: MagicMock
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
        "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
    ) as mock_update:
        await menu.handle_callback(cq)
    mock_update.assert_awaited_once_with(
        mock_storage,
        "123",
        {"model": None, "reasoning": None},
    )


@pytest.mark.asyncio
@patch("blacki.telegram.settings_menu.get_preferences_storage")
async def test_handle_callback_query_edit_msg_exception(
    mock_get_prefs, menu: SettingsMenu, mock_api: MagicMock
) -> None:
    mock_storage = AsyncMock()
    mock_get_prefs.return_value = mock_storage
    user = User(id=1, is_bot=False, first_name="Test")
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="mod:m1", message=msg
    )

    mock_api.edit_message_text.side_effect = Exception("fail")
    with patch(
        "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
    ):
        await menu.handle_callback(cq)
    mock_api.answer_callback_query.assert_called_once()


def test_settings_callback_data_is_within_telegram_limit(menu: SettingsMenu) -> None:
    text, markup = menu._build_model_menu(InferenceProfile())
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
async def test_message_less_settings_mutation_does_not_write(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:m:m1", message=None
    )

    with patch(
        "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
    ) as update:
        await menu.handle_callback(cq)

    update.assert_not_awaited()
    mock_api.answer_callback_query.assert_called_once_with(
        "cq1", text="Settings expired"
    )


@pytest.mark.asyncio
@patch("blacki.telegram.settings_menu.get_preferences_storage")
@patch(
    "blacki.telegram.settings_menu.OpenRouterModelCapabilitiesResolver",
)
async def test_reasoning_callback_preserves_model(
    mock_resolver_cls,
    mock_get_prefs,
    menu: SettingsMenu,
    mock_api: MagicMock,
    load_profile,
) -> None:
    load_profile.return_value = InferenceProfile(
        model="openrouter/openai/gpt-oss-120b",
        reasoning=ReasoningConfig(effort=ReasoningEffort.HIGH),
    )
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
        patch.object(menu, "_resolve_capabilities", AsyncMock(return_value=capability)),
        patch(
            "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
        ) as update,
    ):
        await menu.handle_callback(cq)

    update.assert_awaited_once_with(
        mock_storage,
        "123",
        {"reasoning": ReasoningConfig(effort=ReasoningEffort.MAX)},
        base_profile=InferenceProfile(
            model="openrouter/openai/gpt-oss-120b",
            reasoning=ReasoningConfig(effort=ReasoningEffort.HIGH),
        ),
    )


async def _initialized_preferences_storage() -> SqlitePreferencesStorage:
    connection = await aiosqlite.connect(":memory:", isolation_level=None)
    connection.row_factory = aiosqlite.Row
    storage = SqlitePreferencesStorage(connection, asyncio.Lock())
    await storage.initialize()
    return storage


def _reasoning_callback_query() -> CallbackQuery:
    return CallbackQuery(
        id="cq1",
        from_user=User(id=1, is_bot=False, first_name="Test"),
        chat_instance="inst",
        data="s:r:max",
        message=Message(
            message_id=42,
            date="2024-01-01T00:00:00Z",
            chat=Chat(id=123, type="private"),
        ),
    )


def _max_reasoning_capability() -> SimpleNamespace:
    return SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=("max",),
            mandatory=False,
        )
    )


@pytest.mark.asyncio
async def test_reasoning_callback_migrates_legacy_model(
    mock_api: MagicMock,
) -> None:
    storage = await _initialized_preferences_storage()
    await storage.set("123", LEGACY_MODEL_PREFERENCE_KEY, "legacy-model")

    from blacki.inference import load_inference_profile

    async def load_profile(chat_id: int | str) -> InferenceProfile:
        return await load_inference_profile(storage, str(chat_id))

    menu = SettingsMenu(api_provider=lambda: mock_api, load_profile=load_profile)

    try:
        with (
            patch(
                "blacki.telegram.settings_menu.get_preferences_storage",
                return_value=storage,
            ),
            patch.object(
                menu,
                "_resolve_capabilities",
                AsyncMock(return_value=_max_reasoning_capability()),
            ),
        ):
            await menu.handle_callback(_reasoning_callback_query())

        assert await storage.get("123", INFERENCE_PROFILE_PREFERENCE_KEY) == {
            "model": "legacy-model",
            "reasoning": {"effort": "max"},
        }
    finally:
        await storage.close()
        await storage.conn.close()


@pytest.mark.asyncio
async def test_stale_reasoning_callback_preserves_new_model(
    mock_api: MagicMock,
) -> None:
    storage = await _initialized_preferences_storage()
    await storage.set("123", LEGACY_MODEL_PREFERENCE_KEY, "legacy-model")

    from blacki.inference import load_inference_profile

    async def load_profile(chat_id: int | str) -> InferenceProfile:
        return await load_inference_profile(storage, str(chat_id))

    menu = SettingsMenu(api_provider=lambda: mock_api, load_profile=load_profile)

    async def select_new_model_during_capability_lookup(
        model_id: str,
    ) -> SimpleNamespace:
        assert model_id == "legacy-model"
        await update_inference_profile(
            storage,
            "123",
            {"model": "new-model", "reasoning": None},
        )
        return _max_reasoning_capability()

    try:
        with (
            patch(
                "blacki.telegram.settings_menu.get_preferences_storage",
                return_value=storage,
            ),
            patch.object(
                menu,
                "_resolve_capabilities",
                side_effect=select_new_model_during_capability_lookup,
            ),
        ):
            await menu.handle_callback(_reasoning_callback_query())

        assert await storage.get("123", INFERENCE_PROFILE_PREFERENCE_KEY) == {
            "model": "new-model",
            "reasoning": None,
        }
        assert (
            "Could not save settings"
            in mock_api.edit_message_text.await_args.kwargs["text"]
        )
    finally:
        await storage.close()
        await storage.conn.close()


@pytest.mark.asyncio
async def test_reasoning_menu_hides_off_for_mandatory_model(menu: SettingsMenu) -> None:
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=("none", "high", "max"),
            mandatory=True,
        )
    )
    with patch.object(
        menu, "_resolve_capabilities", AsyncMock(return_value=capability)
    ):
        _, markup = await menu._build_thinking_menu(InferenceProfile())

    labels = [button.text for row in markup.inline_keyboard for button in row]
    assert not any(label.startswith("Off") for label in labels)
    assert any(label.startswith("High") for label in labels)


@pytest.mark.asyncio
async def test_thinking_menu_falls_back_when_capability_client_fails(
    menu: SettingsMenu, mock_api: MagicMock, load_profile: AsyncMock
) -> None:
    load_profile.return_value = InferenceProfile(model="openrouter/openai/gpt-5.6-luna")

    with patch(
        "blacki.telegram.settings_menu.OpenRouterModelCapabilitiesResolver",
        side_effect=RuntimeError("capability client unavailable"),
    ):
        await menu.send_thinking_menu(chat_id=123, message_thread_id=None)

    mock_api.send_message.assert_awaited_once()
    markup = mock_api.send_message.call_args.kwargs["reply_markup"]
    callback_data = [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data is not None
    ]
    assert callback_data == ["s:r:inherit", "s:b"]


@pytest.mark.asyncio
async def test_stale_reasoning_callback_does_not_write(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    chat = Chat(id=123, type="private")
    msg = Message(message_id=42, date="2024-01-01T00:00:00Z", chat=chat)
    cq = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:r:not-real", message=msg
    )

    with patch(
        "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
    ) as update:
        await menu.handle_callback(cq)

    update.assert_not_awaited()
    mock_api.answer_callback_query.assert_called_once_with(
        "cq1", text="Unknown thinking option"
    )


@pytest.mark.asyncio
async def test_aclose_closes_capability_resolver(menu: SettingsMenu) -> None:
    resolver = AsyncMock()
    menu._capabilities_resolver = resolver

    await menu.aclose()

    resolver.aclose.assert_awaited_once()
    assert menu._capabilities_resolver is None


@pytest.mark.asyncio
async def test_aclose_suppresses_capability_resolver_close_error(
    menu: SettingsMenu,
) -> None:
    resolver = AsyncMock()
    resolver.aclose.side_effect = RuntimeError("close failed")
    menu._capabilities_resolver = resolver

    await menu.aclose()

    resolver.aclose.assert_awaited_once()
    assert menu._capabilities_resolver is None


@pytest.mark.asyncio
async def test_send_thinking_menu_handles_send_error(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    mock_api.send_message.side_effect = RuntimeError("send failed")

    await menu.send_thinking_menu(chat_id=123, message_thread_id=None)

    mock_api.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_thinking_menu_even_options_has_no_partial_row(
    menu: SettingsMenu,
) -> None:
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=("max",),
            mandatory=False,
        )
    )
    with patch.object(
        menu, "_resolve_capabilities", AsyncMock(return_value=capability)
    ):
        _, markup = await menu._build_thinking_menu(InferenceProfile())

    assert [button.callback_data for button in markup.inline_keyboard[0]] == [
        "s:r:inherit",
        "s:r:max",
    ]
    assert markup.inline_keyboard[-1][0].callback_data == "s:b"


@pytest.mark.asyncio
async def test_thinking_menu_notes_when_effort_is_unsupported(
    menu: SettingsMenu,
) -> None:
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=False,
            supported_efforts=(),
            mandatory=False,
        )
    )
    with patch.object(
        menu, "_resolve_capabilities", AsyncMock(return_value=capability)
    ):
        text, markup = await menu._build_thinking_menu(InferenceProfile())

    assert "does not expose effort controls" in text
    assert [
        button.callback_data for row in markup.inline_keyboard for button in row
    ] == [
        "s:r:inherit",
        "s:b",
    ]


@pytest.mark.asyncio
async def test_callback_model_none_value_is_rejected_without_write(
    menu: SettingsMenu, mock_api: MagicMock
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
        patch.object(menu, "_parse_settings_callback", return_value=("model", None)),
        patch(
            "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
        ) as update,
    ):
        await menu.handle_callback(query)

    update.assert_not_awaited()
    assert mock_api.answer_callback_query.await_args_list[-1].kwargs["text"] == (
        "Unknown model"
    )


@pytest.mark.asyncio
async def test_callback_rejects_effort_not_supported_by_model(
    menu: SettingsMenu, mock_api: MagicMock
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
        patch(
            "blacki.telegram.settings_menu.get_preferences_storage",
            return_value=mock_storage,
        ),
        patch.object(menu, "_resolve_capabilities", AsyncMock(return_value=capability)),
        patch(
            "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
        ) as update,
    ):
        await menu.handle_callback(query)

    update.assert_not_awaited()
    assert mock_api.edit_message_text.await_count == 1


@pytest.mark.asyncio
async def test_reset_callback_updates_both_profile_fields(
    menu: SettingsMenu, mock_api: MagicMock
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
        data="s:x",
        message=message,
    )

    mock_storage = AsyncMock()
    with (
        patch(
            "blacki.telegram.settings_menu.get_preferences_storage",
            return_value=mock_storage,
        ),
        patch(
            "blacki.telegram.settings_menu.update_inference_profile", new=AsyncMock()
        ) as update,
    ):
        await menu.handle_callback(query)

    update.assert_awaited_once_with(
        mock_storage,
        "123",
        {"model": None, "reasoning": None},
    )


@pytest.mark.asyncio
async def test_thinking_callback_edits_capability_menu(
    menu: SettingsMenu, mock_api: MagicMock
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
        data="s:t",
        message=message,
    )
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Default", callback_data="s:r:inherit")]
        ]
    )
    with (
        patch(
            "blacki.telegram.settings_menu.get_preferences_storage",
            return_value=AsyncMock(),
        ),
        patch.object(
            menu,
            "_build_thinking_menu",
            AsyncMock(return_value=("thinking", markup)),
        ),
    ):
        await menu.handle_callback(query)

    mock_api.edit_message_text.assert_awaited_once_with(
        chat_id=123,
        message_id=42,
        text="thinking",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=markup,
    )


@pytest.mark.asyncio
async def test_back_callback_returns_to_model_menu(
    menu: SettingsMenu, mock_api: MagicMock
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
        data="s:b",
        message=message,
    )

    with (
        patch(
            "blacki.telegram.settings_menu.get_preferences_storage",
            return_value=AsyncMock(),
        ),
        patch.object(menu, "_edit_model_menu", AsyncMock()) as edit_menu,
    ):
        await menu.handle_callback(query)

    edit_menu.assert_awaited_once_with(query, 123)


def test_settings_callback_parser_handles_navigation_actions() -> None:
    assert SettingsMenu._parse_settings_callback("s:t") == ("thinking", None)
    assert SettingsMenu._parse_settings_callback("s:b") == ("back", None)
    assert SettingsMenu._parse_settings_callback("s:x") == ("reset", None)


@pytest.mark.asyncio
async def test_edit_model_menu_ignores_message_less_callback(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    query = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:b", message=None
    )

    await menu._edit_model_menu(query, 123)

    mock_api.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_error_ignores_message_less_callback(
    menu: SettingsMenu, mock_api: MagicMock
) -> None:
    user = User(id=1, is_bot=False, first_name="Test")
    query = CallbackQuery(
        id="cq1", from_user=user, chat_instance="inst", data="s:b", message=None
    )

    await menu._edit_error(query, 123, "failed")

    mock_api.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_capabilities_skips_missing_models(menu: SettingsMenu) -> None:
    assert await menu._resolve_capabilities(None) is None
    assert await menu._resolve_capabilities("default") is None


@pytest.mark.asyncio
async def test_resolve_capabilities_uses_cached_resolver(menu: SettingsMenu) -> None:
    resolver = AsyncMock()
    resolver.resolve.return_value = None
    menu._capabilities_resolver = resolver

    assert await menu._resolve_capabilities("openrouter/openai/gpt-5.6-luna") is None

    resolver.resolve.assert_awaited_once()
    assert resolver.resolve.await_args.args == ("openrouter/openai/gpt-5.6-luna",)


def test_model_display_name_handles_unknown_future_model(menu: SettingsMenu) -> None:
    assert menu._model_display_name("openrouter/acme/future-model") == "future-model"


def test_model_display_name_handles_system_default(menu: SettingsMenu) -> None:
    with patch("blacki.telegram.settings_menu.MODEL_CHOICES", {}):
        assert menu._model_display_name("default") == "System Default"


def test_reasoning_display_inherits_when_only_token_budget_is_set(
    menu: SettingsMenu,
) -> None:
    profile = InferenceProfile(reasoning=ReasoningConfig(max_tokens=256))

    assert menu._reasoning_display(profile) == "Default"


def test_reasoning_options_include_gateway_values_when_unspecified() -> None:
    menu = SettingsMenu.__new__(SettingsMenu)
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=None,
            mandatory=False,
        )
    )

    options = menu._reasoning_options(capability)

    assert ("max", "Max") in options
    assert ("none", "Off") in options


def test_reasoning_options_skip_empty_and_inherit_values() -> None:
    menu = SettingsMenu.__new__(SettingsMenu)
    capability = SimpleNamespace(
        reasoning=SimpleNamespace(
            supports_effort=True,
            supported_efforts=(None, "inherit", "max"),
            mandatory=False,
        )
    )

    assert menu._reasoning_options(capability) == [
        ("inherit", "Default"),
        ("max", "Max"),
    ]


def test_reasoning_config_handles_inherit_and_invalid_values() -> None:
    assert SettingsMenu._reasoning_config("inherit") is None
    assert SettingsMenu._reasoning_config("not-an-effort") is None
