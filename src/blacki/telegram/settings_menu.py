"""Inline-keyboard settings UI for choosing model and reasoning effort."""

import contextlib
import logging
import os
import re
from collections.abc import Awaitable, Callable, Sequence

from blacki.models.capabilities import (
    ModelCapabilities,
    OpenRouterModelCapabilitiesResolver,
)
from blacki.models.inference import (
    InferenceProfile,
    ReasoningConfig,
    ReasoningEffort,
    update_inference_profile,
)
from blacki.models.openrouter_ids import canonical_openrouter_model_id
from blacki.utils.preferences import get_preferences_storage

from .api import TelegramApiClient
from .formatting import format_for_telegram
from .types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode

logger = logging.getLogger(__name__)

_OPENROUTER_MODEL_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:-]*"
)
_MODEL_USAGE = "Use /model <author/model-id> or /model default."

_SETTINGS_MODEL_PREFIX = "s:m:"
_SETTINGS_REASONING_PREFIX = "s:r:"
_SETTINGS_THINKING = "s:t"
_SETTINGS_BACK = "s:b"
_SETTINGS_RESET = "s:x"
_INHERIT_REASONING = "inherit"
_REASONING_LABELS = {
    "inherit": "Default",
    "none": "Off",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "XHigh",
    "max": "Max",
}

ProfileLoader = Callable[[int | str], Awaitable[InferenceProfile]]


class SettingsMenu:
    """Inline-keyboard settings UI for choosing model and reasoning effort.

    Owns no Telegram transport or session state of its own: it renders and
    reacts to the `/model` and `/thinking` settings panels via an API client
    obtained from ``api_provider`` and a chat's inference profile obtained
    from ``load_profile``.
    """

    def __init__(
        self,
        api_provider: Callable[[], TelegramApiClient],
        load_profile: ProfileLoader,
    ) -> None:
        self._api_provider = api_provider
        self._load_profile = load_profile
        self._capabilities_resolver: OpenRouterModelCapabilitiesResolver | None = None

    @property
    def _api(self) -> TelegramApiClient:
        return self._api_provider()

    async def aclose(self) -> None:
        """Release the cached model-capabilities resolver, if any."""
        if self._capabilities_resolver is not None:
            with contextlib.suppress(Exception):
                await self._capabilities_resolver.aclose()
            self._capabilities_resolver = None

    async def send_model_menu(
        self, chat_id: int, message_thread_id: int | None
    ) -> None:
        """Send the compact model-and-thinking settings panel."""
        profile = await self._load_profile(chat_id)
        text, reply_markup = self._build_model_menu(profile)
        try:
            await self._api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=message_thread_id,
                reply_markup=reply_markup,
            )
        except Exception:
            logger.exception("Failed to send model menu")

    async def send_thinking_menu(
        self, chat_id: int, message_thread_id: int | None
    ) -> None:
        """Send the reasoning-effort menu for the effective model."""
        profile = await self._load_profile(chat_id)
        text, reply_markup = await self._build_thinking_menu(profile)
        try:
            await self._api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=message_thread_id,
                reply_markup=reply_markup,
            )
        except Exception:
            logger.exception("Failed to send thinking menu")

    async def set_model(
        self, chat_id: int, message_thread_id: int | None, model_id: str
    ) -> None:
        """Save an exact OpenRouter ID for subsequent turns in this chat."""
        selected: str | None = None
        if model_id != "default":
            canonical = canonical_openrouter_model_id(model_id)
            if _OPENROUTER_MODEL_ID.fullmatch(canonical) is None:
                await self._send_model_result(chat_id, message_thread_id, _MODEL_USAGE)
                return
            if not os.getenv("OPENROUTER_API_KEY", "").strip():
                await self._send_model_result(
                    chat_id,
                    message_thread_id,
                    "OpenRouter is not configured. Set OPENROUTER_API_KEY "
                    "before selecting an OpenRouter model.",
                )
                return
            selected = f"openrouter/{canonical}"

        try:
            await update_inference_profile(
                get_preferences_storage(),
                str(chat_id),
                {"model": selected, "reasoning": None},
            )
        except Exception:
            logger.exception("Failed to save Telegram model selection")
            text = "Could not save settings. Please try again."
        else:
            effective = self._effective_model(InferenceProfile(model=selected))
            display_name = self._model_display_name(effective)
            text = (
                f"Model saved for the next turn: {display_name}\n"
                "Conversation history is preserved. Thinking uses the default setting."
            )
        await self._send_model_result(chat_id, message_thread_id, text)

    async def _send_model_result(
        self, chat_id: int, message_thread_id: int | None, text: str
    ) -> None:
        """Report command results without confusing delivery and save failures."""
        try:
            await self._api.send_message(
                chat_id=chat_id, text=text, message_thread_id=message_thread_id
            )
        except Exception:
            logger.exception("Failed to send model selection result")

    @staticmethod
    def _chunk_buttons(
        buttons: Sequence[InlineKeyboardButton], per_row: int = 2
    ) -> list[list[InlineKeyboardButton]]:
        """Group flat buttons into keyboard rows of a fixed width."""
        return [list(buttons[i : i + per_row]) for i in range(0, len(buttons), per_row)]

    def _build_model_menu(
        self, profile: InferenceProfile
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Build the model menu without performing network I/O."""
        effective_model = self._effective_model(profile)
        current_display_name = self._model_display_name(effective_model)
        current_thinking = self._reasoning_display(profile)

        buttons: list[list[InlineKeyboardButton]] = []

        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"🧠 Thinking: {current_thinking}",
                    callback_data=_SETTINGS_THINKING,
                )
            ]
        )
        buttons.append(
            [
                InlineKeyboardButton(
                    text="↩️ Reset settings", callback_data=_SETTINGS_RESET
                )
            ]
        )

        text = format_for_telegram(
            "⚙️ **Inference settings**\n\n"
            f"Model: **{current_display_name}**\n"
            f"Thinking: **{current_thinking}**\n\n"
            f"{_MODEL_USAGE}\n"
            "Copy the exact ID from OpenRouter. Changes apply to the next turn."
        )
        return text, InlineKeyboardMarkup(inline_keyboard=buttons)

    async def _build_thinking_menu(
        self, profile: InferenceProfile
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Build a capability-aware reasoning menu."""
        effective_model = self._effective_model(profile)
        capability = await self._resolve_capabilities(effective_model)
        options = self._reasoning_options(capability)
        current = self._reasoning_display(profile)

        reasoning_buttons = [
            InlineKeyboardButton(
                text=f"{label}{' ✓' if label == current else ''}",
                callback_data=f"{_SETTINGS_REASONING_PREFIX}{value}",
            )
            for value, label in options
        ]
        buttons = self._chunk_buttons(reasoning_buttons)
        buttons.append(
            [
                InlineKeyboardButton(
                    text="⬅️ Back to settings", callback_data=_SETTINGS_BACK
                )
            ]
        )

        if capability is None or capability.reasoning is None:
            note = (
                "Thinking controls are not published for this model. "
                "Only the provider default is available."
            )
        elif not capability.reasoning.supports_effort:
            note = "This model does not expose effort controls."
        else:
            note = "Only options supported by the selected model are shown."

        text = format_for_telegram(
            f"🧠 **Thinking for {self._model_display_name(effective_model)}**\n\n"
            f"Current: **{current}**\n"
            f"{note}"
        )
        return text, InlineKeyboardMarkup(inline_keyboard=buttons)

    async def handle_callback(self, query: CallbackQuery) -> None:
        """Handle a settings callback query (model, thinking, back, or reset)."""
        data = query.data or ""
        action, value = self._parse_settings_callback(data)
        if action is None:
            await self._api.answer_callback_query(query.id, text="Unknown action")
            return

        if action == "model" and value != "m_default":
            await self._api.answer_callback_query(
                query.id, text="Model preset expired. Use /model <author/model-id>."
            )
            return
        if action == "reasoning" and value not in {
            _INHERIT_REASONING,
            *tuple(_REASONING_LABELS),
        }:
            await self._api.answer_callback_query(
                query.id, text="Unknown thinking option"
            )
            return

        if query.message is None:
            await self._api.answer_callback_query(query.id, text="Settings expired")
            return

        chat_id = query.message.chat.id
        await self._api.answer_callback_query(query.id, text="Updating settings…")

        try:
            storage = get_preferences_storage()
            if action == "reasoning":
                profile = await self._load_profile(chat_id)
                capability = await self._resolve_capabilities(
                    self._effective_model(profile)
                )
                supported = {
                    option for option, _ in self._reasoning_options(capability)
                }
                if value not in supported:
                    await self._edit_error(
                        query,
                        chat_id,
                        "That thinking option is not available for this model.",
                    )
                    return
                reasoning = self._reasoning_config(value)
                await update_inference_profile(
                    storage,
                    str(chat_id),
                    {"reasoning": reasoning},
                    base_profile=profile,
                )
                await self._edit_model_menu(query, chat_id)
                return

            if action in {"reset", "model"}:
                await update_inference_profile(
                    storage,
                    str(chat_id),
                    {"model": None, "reasoning": None},
                )
                await self._edit_model_menu(query, chat_id)
                return

            if action == "thinking":
                profile = await self._load_profile(chat_id)
                text, markup = await self._build_thinking_menu(profile)
                await self._api.edit_message_text(
                    chat_id=chat_id,
                    message_id=query.message.message_id,
                    text=text,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=markup,
                )
                return

            # All other parsed actions return above, so the only remaining
            # valid action is Back.
            await self._edit_model_menu(query, chat_id)
        except Exception:
            logger.exception("Failed to update Telegram inference settings")
            await self._edit_error(
                query, chat_id, "Could not save settings. Please try again."
            )

    @staticmethod
    def _parse_settings_callback(data: str) -> tuple[str | None, str | None]:
        """Parse current and legacy callback payloads."""
        if data.startswith("mod:"):
            return "model", data.removeprefix("mod:")
        if data.startswith(_SETTINGS_MODEL_PREFIX):
            return "model", data.removeprefix(_SETTINGS_MODEL_PREFIX)
        if data.startswith(_SETTINGS_REASONING_PREFIX):
            return "reasoning", data.removeprefix(_SETTINGS_REASONING_PREFIX)
        if data == _SETTINGS_THINKING:
            return "thinking", None
        if data == _SETTINGS_BACK:
            return "back", None
        if data == _SETTINGS_RESET:
            return "reset", None
        return None, None

    async def _edit_model_menu(self, query: CallbackQuery, chat_id: int) -> None:
        """Render the settings panel into an existing callback message."""
        if query.message is None:
            return
        profile = await self._load_profile(chat_id)
        text, markup = self._build_model_menu(profile)
        await self._api.edit_message_text(
            chat_id=chat_id,
            message_id=query.message.message_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=markup,
        )

    async def _edit_error(
        self, query: CallbackQuery, chat_id: int, message: str
    ) -> None:
        """Show a recoverable settings error while retaining a back action."""
        if query.message is None:
            return
        try:
            await self._api.edit_message_text(
                chat_id=chat_id,
                message_id=query.message.message_id,
                text=format_for_telegram(f"⚠️ {message}"),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="⬅️ Back to settings", callback_data=_SETTINGS_BACK
                            )
                        ]
                    ]
                ),
            )
        except Exception:
            logger.exception("Failed to render Telegram settings error")

    async def _resolve_capabilities(
        self, model_id: str | None
    ) -> ModelCapabilities | None:
        """Resolve OpenRouter reasoning metadata without blocking turns."""
        if not model_id or model_id == "default":
            return None
        try:
            if self._capabilities_resolver is None:
                self._capabilities_resolver = OpenRouterModelCapabilitiesResolver()
            return await self._capabilities_resolver.resolve(
                model_id,
                openrouter_routed=bool(os.getenv("OPENROUTER_API_KEY")),
            )
        except Exception:
            logger.exception("Failed to resolve model capabilities for %s", model_id)
            return None

    @staticmethod
    def _effective_model(profile: InferenceProfile) -> str:
        """Resolve the profile model, then the process-wide model setting."""
        return profile.model or os.getenv("ROOT_AGENT_MODEL") or "default"

    @staticmethod
    def _model_display_name(model_id: str) -> str:
        """Show the exact API ID, including its provider and variant."""
        if model_id == "default":
            return "System Default"
        return canonical_openrouter_model_id(model_id)

    @staticmethod
    def _effort_value(value: object) -> str | None:
        """Normalize enum or string effort values for Telegram labels."""
        raw = getattr(value, "value", value)
        return raw.strip().lower() if isinstance(raw, str) and raw.strip() else None

    def _reasoning_display(self, profile: InferenceProfile) -> str:
        """Render the profile's current reasoning setting."""
        reasoning = profile.reasoning
        if reasoning is None:
            return _REASONING_LABELS[_INHERIT_REASONING]
        value = self._effort_value(reasoning.effort)
        if value is None:
            return _REASONING_LABELS[_INHERIT_REASONING]
        return _REASONING_LABELS.get(value, value.title())

    def _reasoning_options(
        self, capability: ModelCapabilities | None
    ) -> list[tuple[str, str]]:
        """Return default plus only the effort values the model supports."""
        options: list[tuple[str, str]] = [
            (_INHERIT_REASONING, _REASONING_LABELS[_INHERIT_REASONING])
        ]
        reasoning = getattr(capability, "reasoning", None)
        if reasoning is None or not reasoning.supports_effort:
            return options

        supported = reasoning.supported_efforts
        if supported is None:
            supported = tuple(_REASONING_LABELS)
        for effort in supported:
            value = self._effort_value(effort)
            if value is None or value == _INHERIT_REASONING:
                continue
            if value == "none" and reasoning.mandatory:
                continue
            label = _REASONING_LABELS.get(value, value.title())
            options.append((value, label))
        return options

    @staticmethod
    def _reasoning_config(value: str) -> ReasoningConfig | None:
        """Convert a Telegram value into the typed profile update."""
        if value == _INHERIT_REASONING:
            return None
        try:
            effort = ReasoningEffort(value)
        except ValueError:
            return None
        return ReasoningConfig(effort=effort)
