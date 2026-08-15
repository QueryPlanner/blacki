"""Telegram bot client backed by the shared ADK runtime."""

import asyncio
import contextlib
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from google.genai import types

from blacki.adk_runtime import AdkRuntime, EmptyModelResponseError, SessionLocator
from blacki.inference import (
    InferenceProfile,
    ReasoningConfig,
    ReasoningEffort,
    inference_profile_from_environment,
    load_inference_profile,
    update_inference_profile,
)
from blacki.model_capabilities import (
    ModelCapabilities,
    OpenRouterModelCapabilitiesResolver,
)
from blacki.reminders.storage import Reminder
from blacki.utils.preferences import get_preferences_storage

from . import TelegramConfig
from .api import TelegramApiClient, TelegramApiError
from .formatting import escape_markdown_plain, format_for_telegram
from .streaming import split_long_message
from .types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ParseMode,
    Update,
)

logger = logging.getLogger(__name__)

POLLING_TIMEOUT = 30
_MAX_CONSECUTIVE_ERRORS = 5
_FATAL_ERROR_CODES = {401, 403}
_MAX_EMPTY_RESPONSE_RETRIES = 1
_TELEGRAM_USER_ID_PATTERN = re.compile(r"^telegram-chat-(-?\d+)(?:-thread-(\d+))?$")
_MAX_NATIVE_IMAGE_BYTES = 10 * 1024 * 1024
_JPEG_MAGIC = b"\xff\xd8\xff"
_DEFAULT_IMAGE_PROMPT = "Describe this image."

MODEL_CHOICES = {
    "m1": ("openrouter/openai/gpt-oss-120b", "GPT-OSS 120B"),
    "m2": ("openrouter/x-ai/grok-4.3", "Grok 4.3"),
    "m3": ("google/gemini-flash-latest", "Gemini Flash"),
    "m4": ("openrouter/deepseek/deepseek-v4-pro", "DeepSeek v4 Pro"),
    "m5": ("openrouter/deepseek/deepseek-v4-flash", "DeepSeek v4 Flash"),
    "m6": ("google/gemini-pro-latest", "Gemini Pro"),
    "m7": ("moonshotai/kimi-latest", "Kimi Latest"),
    "m8": ("openrouter/minimax/minimax-m2.7", "MiniMax m2.7"),
    "m9": ("openrouter/nvidia/nemotron-3-super-120b-a12b", "Nemotron 3 Super"),
    "m10": ("openrouter/z-ai/glm-5", "GLM 5"),
    "m11": ("openrouter/openai/gpt-5.6-luna", "GPT-5.6 Luna"),
    "m_default": ("default", "System Default"),
}

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


@dataclass(slots=True, frozen=True)
class TelegramSessionIdentity:
    """Stable Telegram identifiers used to resolve ADK sessions."""

    conversation_key: str
    user_id: str
    session_id_prefix: str


class TelegramBot:
    """Telegram bot client that sends typing indicators and final replies."""

    def __init__(
        self,
        config: TelegramConfig,
        runtime: AdkRuntime,
    ) -> None:
        """Initialize the Telegram bot."""
        self.config = config
        self.runtime = runtime
        self._api: TelegramApiClient | None = None
        self._running = False
        self._polling_task: asyncio.Task[None] | None = None
        self._conversation_tasks: dict[str, asyncio.Task[None]] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._capabilities_resolver: OpenRouterModelCapabilitiesResolver | None = None

    @property
    def api(self) -> TelegramApiClient:
        """Get or create the Telegram API client."""
        if self._api is None:
            if not self.config.telegram_bot_token:
                msg = "TELEGRAM_BOT_TOKEN is required"
                raise ValueError(msg)
            self._api = TelegramApiClient(self.config.telegram_bot_token)
        return self._api

    async def start_polling(self) -> None:
        """Start the bot polling loop."""
        if not self.config.is_configured():
            logger.info("Telegram bot not configured, skipping start")
            return

        logger.info("Starting Telegram bot polling...")
        self._running = True

        await self._register_commands()

        self._polling_task = asyncio.create_task(self._polling_loop())
        logger.info("Telegram bot started successfully")

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("Stopping Telegram bot...")
        self._running = False

        if self._polling_task is not None:
            self._polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._polling_task

        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await self.runtime.close()

        if self._capabilities_resolver is not None:
            with contextlib.suppress(Exception):
                await self._capabilities_resolver.aclose()
            self._capabilities_resolver = None

        if self._api is not None:
            await self._api.close()

        logger.info("Telegram bot stopped")

    async def _register_commands(self) -> None:
        """Register bot commands with Telegram's command menu."""
        commands = [
            BotCommand(command="start", description="Start a conversation"),
            BotCommand(command="help", description="Show available commands"),
            BotCommand(
                command="reset", description="Start a fresh conversation session"
            ),
            BotCommand(command="model", description="Select AI model for this chat"),
            BotCommand(
                command="thinking",
                description="Set reasoning effort for this chat",
            ),
        ]
        try:
            await self.api.set_my_commands(commands)
            logger.info("Registered Telegram bot commands")
        except TelegramApiError:
            logger.exception("Failed to register bot commands")

    async def _polling_loop(self) -> None:
        """Long polling loop for updates."""
        offset = 0
        consecutive_errors = 0

        while self._running:
            try:
                updates = await self.api.get_updates(
                    offset=offset,
                    timeout=POLLING_TIMEOUT,
                    allowed_updates=["message", "callback_query"],
                )

                consecutive_errors = 0

                for update in updates:
                    offset = update.update_id + 1
                    task = asyncio.create_task(self._safe_handle_update(update))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

            except asyncio.CancelledError:
                raise
            except TelegramApiError as exc:
                consecutive_errors += 1
                status = exc.error_code
                if status in _FATAL_ERROR_CODES:
                    logger.critical(
                        "Fatal Telegram API error (status=%s), stopping polling: %s",
                        status,
                        exc,
                    )
                    return
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        "Too many consecutive Telegram API errors (%d), stopping: %s",
                        consecutive_errors,
                        exc,
                    )
                    return
                logger.warning(
                    "Transient Telegram API error in polling loop (%d/%d): %s",
                    consecutive_errors,
                    _MAX_CONSECUTIVE_ERRORS,
                    exc,
                )
                await asyncio.sleep(min(5 * consecutive_errors, 60))
            except Exception:
                consecutive_errors += 1
                logger.exception("Error in polling loop")
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        "Too many consecutive errors (%d), stopping polling",
                        consecutive_errors,
                    )
                    return
                await asyncio.sleep(min(5 * consecutive_errors, 60))

    async def _safe_handle_update(self, update: Update) -> None:
        """Handle update concurrently and allow cancellation."""
        if update.callback_query:
            # Handle callback queries immediately without cancelling conversation tasks
            try:
                await self._handle_callback_query(update.callback_query)
            except Exception:
                logger.exception("Error handling callback query")
            return

        if update.message is None:
            return

        chat_id = update.message.chat.id
        message_thread_id = update.message.message_thread_id
        conversation_key = self._build_conversation_key(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
        )

        existing_task = self._conversation_tasks.get(conversation_key)
        if existing_task is not None and not existing_task.done():
            logger.info(
                "Cancelling in-flight turn for conversation %s", conversation_key
            )
            existing_task.cancel()

        current_task = asyncio.current_task()
        if current_task is not None:
            self._conversation_tasks[conversation_key] = current_task

        try:
            # Wait for the superseded task to fully clean up before starting
            if existing_task is not None and not existing_task.done():
                await asyncio.wait([existing_task])

            await self._handle_update(update)
        except asyncio.CancelledError:
            logger.info("Message turn superseded for conversation %s", conversation_key)
            raise
        finally:
            if self._conversation_tasks.get(conversation_key) is current_task:
                self._conversation_tasks.pop(conversation_key, None)

    async def _handle_update(self, update: Update) -> None:
        """Handle an incoming update."""
        if update.message is None:
            return

        message = update.message

        if message.text is None:
            await self._route_non_text_message(message)
            return

        chat_id = message.chat.id
        message_thread_id = message.message_thread_id
        user_message = message.text

        if user_message.startswith("/"):
            await self._handle_command(message, user_message)
            return

        await self._handle_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            user_message=user_message,
        )

    async def _route_non_text_message(self, message: Message) -> None:
        """Route a non-text message to the appropriate handler."""
        chat_id = message.chat.id
        message_thread_id = message.message_thread_id

        if message.photo:
            photo = max(message.photo, key=lambda item: item.width * item.height)
            await self._handle_photo_upload(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                file_id=photo.file_id,
                file_size=photo.file_size,
                caption=message.caption,
            )
            return

        if message.document:
            file_id = message.document.file_id
            file_name = message.document.file_name or "document"
        elif message.audio:
            file_id = message.audio.file_id
            file_name = message.audio.file_name or "audio.mp3"
        elif message.video:
            file_id = message.video.file_id
            file_name = message.video.file_name or "video.mp4"
        elif message.voice:
            file_id = message.voice.file_id
            file_name = "voice.ogg"
        else:
            logger.debug("Unsupported non-text message from chat %s", chat_id)
            return

        await self._handle_file_upload(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            file_id=file_id,
            file_name=file_name,
            caption=message.caption,
        )

    async def _handle_photo_upload(
        self,
        *,
        chat_id: int,
        message_thread_id: int | None,
        file_id: str,
        file_size: int | None,
        caption: str | None,
    ) -> None:
        """Download a Telegram photo and send it to ADK as native image input."""
        if file_size is not None and file_size > _MAX_NATIVE_IMAGE_BYTES:
            await self._send_photo_error(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="❌ The photo is too large to process (10 MB maximum).",
            )
            return

        session_identity = self._build_session_identity(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
        )
        state = self._build_session_state(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
            conversation_key=session_identity.conversation_key,
        )

        try:
            await self.api.send_chat_action(
                chat_id=chat_id,
                action="typing",
                message_thread_id=message_thread_id,
            )
            file_info = await self.api.get_file(file_id)
            file_path_api = file_info.get("file_path")
            if not file_path_api:
                raise ValueError("Telegram did not return a photo file path")

            image_bytes = await self.api.download_file(file_path_api)
            if not image_bytes:
                raise ValueError("Telegram returned an empty photo")
            if len(image_bytes) > _MAX_NATIVE_IMAGE_BYTES:
                raise ValueError("Telegram photo exceeds the 10 MB limit")
            if not image_bytes.startswith(_JPEG_MAGIC):
                raise ValueError("Telegram photo is not a JPEG image")

            prompt = (
                caption.strip()
                if caption and caption.strip()
                else _DEFAULT_IMAGE_PROMPT
            )
            user_parts = (
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            )
            profile = await self._load_chat_profile(chat_id)
            final_response = await self._run_user_turn_with_retry(
                session_identity=session_identity,
                message_text=prompt,
                state=state,
                user_parts=user_parts,
                inference_profile=profile,
            )
            await self._send_final_response(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                response_text=final_response,
            )
        except Exception:
            logger.exception("Failed to handle Telegram photo")
            await self._send_photo_error(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="❌ Sorry, I failed to process the photo.",
            )

    async def _send_photo_error(
        self,
        *,
        chat_id: int,
        message_thread_id: int | None,
        text: str,
    ) -> None:
        """Send a plain-text photo processing error."""
        await self.api.send_message(
            chat_id=chat_id,
            text=text,
            message_thread_id=message_thread_id,
        )

    async def _handle_command(self, message: Message, command: str) -> None:
        """Handle a command message."""
        chat_id = message.chat.id

        if command == "/start":
            await self._send_start_message(chat_id)
        elif command == "/help":
            await self._send_help_message(chat_id)
        elif command == "/reset":
            await self._handle_reset(chat_id, message.message_thread_id)
        elif command == "/model":
            await self._send_model_menu(chat_id, message.message_thread_id)
        elif command == "/thinking":
            await self._send_thinking_menu(chat_id, message.message_thread_id)

    async def _send_model_menu(
        self, chat_id: int, message_thread_id: int | None
    ) -> None:
        """Send the compact model-and-thinking settings panel."""
        profile = await self._load_chat_profile(chat_id)
        text, reply_markup = self._build_model_menu(profile)
        try:
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=message_thread_id,
                reply_markup=reply_markup,
            )
        except Exception:
            logger.exception("Failed to send model menu")

    async def _send_thinking_menu(
        self, chat_id: int, message_thread_id: int | None
    ) -> None:
        """Send the reasoning-effort menu for the effective model."""
        profile = await self._load_chat_profile(chat_id)
        text, reply_markup = await self._build_thinking_menu(profile)
        try:
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=message_thread_id,
                reply_markup=reply_markup,
            )
        except Exception:
            logger.exception("Failed to send thinking menu")

    def _build_model_menu(
        self, profile: InferenceProfile
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Build the model menu without performing network I/O."""
        effective_model = self._effective_model(profile)
        current_display_name = self._model_display_name(effective_model)
        current_thinking = self._reasoning_display(profile)

        buttons: list[list[InlineKeyboardButton]] = []
        row: list[InlineKeyboardButton] = []
        for key, (_, display_name) in MODEL_CHOICES.items():
            row.append(
                InlineKeyboardButton(
                    text=display_name,
                    callback_data=f"{_SETTINGS_MODEL_PREFIX}{key}",
                )
            )
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

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
            "Choose a model or adjust Thinking. Changes apply to the next turn."
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

        buttons: list[list[InlineKeyboardButton]] = []
        row: list[InlineKeyboardButton] = []
        for value, label in options:
            row.append(
                InlineKeyboardButton(
                    text=f"{label}{' ✓' if label == current else ''}",
                    callback_data=f"{_SETTINGS_REASONING_PREFIX}{value}",
                )
            )
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
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

    async def _handle_callback_query(self, query: CallbackQuery) -> None:
        """Handle incoming callback query."""
        data = query.data or ""
        action, value = self._parse_settings_callback(data)
        if action is None:
            await self.api.answer_callback_query(query.id, text="Unknown action")
            return

        if action == "model" and (value is None or value not in MODEL_CHOICES):
            await self.api.answer_callback_query(query.id, text="Unknown model")
            return
        if action == "reasoning" and value not in {
            _INHERIT_REASONING,
            *tuple(_REASONING_LABELS),
        }:
            await self.api.answer_callback_query(
                query.id, text="Unknown thinking option"
            )
            return

        if query.message is None:
            await self.api.answer_callback_query(query.id, text="Settings expired")
            return

        chat_id = query.message.chat.id
        await self.api.answer_callback_query(query.id, text="Updating settings…")

        try:
            storage = get_preferences_storage()
            if action == "model":
                model_id, _ = MODEL_CHOICES[cast(str, value)]
                await update_inference_profile(
                    storage,
                    str(chat_id),
                    {
                        "model": None if model_id == "default" else model_id,
                        "reasoning": None,
                    },
                )
                await self._edit_model_menu(query, chat_id)
                return

            if action == "reasoning":
                profile = await self._load_chat_profile(chat_id)
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

            if action == "reset":
                await update_inference_profile(
                    storage,
                    str(chat_id),
                    {"model": None, "reasoning": None},
                )
                await self._edit_model_menu(query, chat_id)
                return

            if action == "thinking":
                profile = await self._load_chat_profile(chat_id)
                text, markup = await self._build_thinking_menu(profile)
                await self.api.edit_message_text(
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
        profile = await self._load_chat_profile(chat_id)
        text, markup = self._build_model_menu(profile)
        await self.api.edit_message_text(
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
            await self.api.edit_message_text(
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

    async def _load_chat_profile(self, chat_id: int | str) -> InferenceProfile:
        """Load a profile snapshot, retaining the process fallback on errors."""
        try:
            profile = await load_inference_profile(
                get_preferences_storage(), str(chat_id)
            )
        except Exception:
            logger.exception("Failed to load inference profile for chat %s", chat_id)
            return inference_profile_from_environment()
        return (
            profile
            if isinstance(profile, InferenceProfile)
            else inference_profile_from_environment()
        )

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
        """Return a friendly label while preserving unknown model IDs."""
        for configured_id, display_name in MODEL_CHOICES.values():
            if configured_id == model_id:
                return display_name
        if model_id == "default":
            return "System Default"
        return model_id.rsplit("/", 1)[-1]

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

    async def _send_start_message(self, chat_id: int) -> None:
        """Send the start/welcome message."""
        text = escape_markdown_plain(
            "👋 Hello! I'm blacki, your AI assistant.\n\n"
            "I run through the same ADK agent as the web interface, so our "
            "conversation history stays attached to this chat.\n\n"
            "Commands:\n"
            "/help - Show available commands\n"
            "/reset - Start a fresh conversation session\n"
            "/model - Choose the model and thinking settings\n"
            "/thinking - Choose supported reasoning effort"
        )
        try:
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except TelegramApiError:
            logger.exception("Failed to send start message")

    async def _send_help_message(self, chat_id: int) -> None:
        """Send the help message."""
        text = (
            "🤖 *blacki \\- AI Assistant*\n\n"
            "I'm powered by the same Google ADK runtime used by the HTTP app\\.\n\n"
            "*Commands:*\n"
            "• /start \\- Start a conversation\n"
            "• /help \\- Show this help message\n"
            "• /reset \\- Start a fresh conversation session\n"
            "• /model \\- Choose the model and thinking settings\n"
            "• /thinking \\- Choose supported reasoning effort\n\n"
            "*Features:*\n"
            "• Conversation history is tied to this chat\n"
            "• Topic threads can keep separate sessions\n"
            "• Ask me anything \\- questions, coding help, creative tasks"
        )
        try:
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except TelegramApiError:
            logger.exception("Failed to send help message")

    async def _handle_reset(self, chat_id: int, message_thread_id: int | None) -> None:
        """Handle /reset command."""
        session_identity = self._build_session_identity(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
        )

        try:
            state = self._build_session_state(
                chat_id=str(chat_id),
                message_thread_id=message_thread_id,
                conversation_key=session_identity.conversation_key,
            )
            await self.runtime.create_next_session(
                locator=SessionLocator(
                    user_id=session_identity.user_id,
                    session_id_prefix=session_identity.session_id_prefix,
                ),
                state=state,
            )
            text = "🔄 Session reset\\. Starting a fresh ADK conversation\\."
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except Exception:
            logger.exception("Failed to reset Telegram session for chat %s", chat_id)
            text = (
                "❌ Sorry, I couldn't reset the conversation right now\\. "
                "Please try again\\."
            )
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )

    async def _handle_file_upload(
        self,
        chat_id: int,
        message_thread_id: int | None,
        file_id: str,
        file_name: str,
        caption: str | None,
    ) -> None:
        """Handle incoming file uploads, save to sandbox, and message agent."""
        from blacki.sandbox.manager import get_sandbox_manager

        session_identity = self._build_session_identity(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
        )
        state = self._build_session_state(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
            conversation_key=session_identity.conversation_key,
        )

        manager = get_sandbox_manager()

        if not manager.config.enabled:
            await self.api.send_message(
                chat_id=chat_id,
                text=escape_markdown_plain(
                    "❌ Sandbox is not enabled. Cannot process file uploads."
                ),
                message_thread_id=message_thread_id,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            return

        try:
            await self.api.send_chat_action(
                chat_id=chat_id,
                action="upload_document",
                message_thread_id=message_thread_id,
            )

            file_info = await self.api.get_file(file_id)
            file_path_api = file_info.get("file_path")
            if not file_path_api:
                raise Exception("Failed to get file_path from Telegram API")

            file_bytes = await self.api.download_file(file_path_api)

            result = await manager.get_or_create_sandbox(state)
            sandbox = result.get("sandbox")
            error = result.get("error")

            if error or not sandbox:
                raise Exception(f"Failed to access sandbox: {error}")

            safe_name = Path(file_name).name
            sandbox_path = f"/workspace/uploads/{safe_name}"
            await sandbox.files.write_file(sandbox_path, file_bytes)

            user_message = (
                f"User uploaded a file which has been saved to "
                f"the sandbox at {sandbox_path}"
            )
            if caption:
                user_message += f"\nCaption provided by user: {caption}"

            logger.info("File %s saved to sandbox for chat %s", file_name, chat_id)

            await self.api.send_chat_action(
                chat_id=chat_id,
                action="typing",
                message_thread_id=message_thread_id,
            )

            final_response = await self._run_user_turn_with_retry(
                session_identity=session_identity,
                message_text=user_message,
                state=state,
                inference_profile=await self._load_chat_profile(chat_id),
            )

            await self._send_final_response(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                response_text=final_response,
            )

        except Exception:
            logger.exception("Failed to handle file upload")
            await self.api.send_message(
                chat_id=chat_id,
                text="❌ Sorry, I failed to process the uploaded file\\.",
                message_thread_id=message_thread_id,
                parse_mode=ParseMode.MARKDOWN_V2,
            )

    async def _run_user_turn_with_retry(
        self,
        *,
        session_identity: TelegramSessionIdentity,
        message_text: str,
        state: dict[str, str],
        inference_profile: InferenceProfile | None,
        user_parts: Sequence[types.Part] | None = None,
    ) -> str:
        """Run a Telegram turn with one safe empty-response retry."""
        retry_count = 0
        while True:
            try:
                return await self.runtime.run_user_turn(
                    locator=SessionLocator(
                        user_id=session_identity.user_id,
                        session_id_prefix=session_identity.session_id_prefix,
                    ),
                    message_text=message_text,
                    state=state,
                    user_parts=user_parts,
                    inference_profile=inference_profile,
                )
            except EmptyModelResponseError as error:
                model = error.model or (
                    inference_profile.model if inference_profile else None
                )
                model = model or "unknown"
                provider = error.provider or "unknown"
                invocation_id = error.invocation_id or "unknown"
                if retry_count >= _MAX_EMPTY_RESPONSE_RETRIES or not error.retryable:
                    logger.warning(
                        "Empty model response recovery stopped: "
                        "model=%s provider=%s conversation_id=%s "
                        "invocation_id=%s retry_count=%d retryable=%s",
                        model,
                        provider,
                        session_identity.conversation_key,
                        invocation_id,
                        retry_count,
                        error.retryable,
                    )
                    raise

                retry_count += 1
                logger.warning(
                    "Empty model response; retrying Telegram turn: "
                    "model=%s provider=%s conversation_id=%s "
                    "invocation_id=%s retry_count=%d",
                    model,
                    provider,
                    session_identity.conversation_key,
                    invocation_id,
                    retry_count,
                )

    async def _handle_message(
        self,
        chat_id: int,
        message_thread_id: int | None,
        user_message: str,
    ) -> None:
        """Handle a regular text message with typing + final response."""
        session_identity = self._build_session_identity(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
        )

        logger.info("Received message from chat %s", chat_id)

        try:
            await self.api.send_chat_action(
                chat_id=chat_id,
                action="typing",
                message_thread_id=message_thread_id,
            )

            state = self._build_session_state(
                chat_id=str(chat_id),
                message_thread_id=message_thread_id,
                conversation_key=session_identity.conversation_key,
            )
            profile = await self._load_chat_profile(chat_id)
            final_response = await self._run_user_turn_with_retry(
                session_identity=session_identity,
                message_text=user_message,
                state=state,
                inference_profile=profile,
            )
            await self._send_final_response(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                response_text=final_response,
            )

            logger.info("Sent ADK response to chat %s", chat_id)
        except EmptyModelResponseError:
            logger.exception("Empty model response after retry for chat %s", chat_id)
            text = (
                "❌ The model returned an empty response\\. "
                "Please try sending your message again\\."
            )
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except Exception:
            logger.exception("Error processing message for chat %s", chat_id)
            text = (
                "❌ Sorry, I encountered an error processing your message\\. "
                "Please try again\\."
            )
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )

    async def handle_scheduled_reminder(self, reminder: Reminder) -> None:
        """Handle an incoming scheduled reminder."""
        match = _TELEGRAM_USER_ID_PATTERN.match(reminder.user_id)
        if not match:
            logger.error("Could not extract chat_id from user_id: %s", reminder.user_id)
            return

        chat_id_str = match.group(1)
        chat_id = int(chat_id_str)
        message_thread_id = int(match.group(2)) if match.group(2) else None

        session_identity = self._build_session_identity(
            chat_id=chat_id_str,
            message_thread_id=message_thread_id,
        )

        logger.info("Handling scheduled reminder %s for chat %s", reminder.id, chat_id)

        try:
            await self.api.send_chat_action(
                chat_id=chat_id,
                action="typing",
                message_thread_id=message_thread_id,
            )

            state = self._build_session_state(
                chat_id=chat_id_str,
                message_thread_id=message_thread_id,
                conversation_key=session_identity.conversation_key,
            )
            profile = await self._load_chat_profile(chat_id_str)
            final_response = await self._run_user_turn_with_retry(
                session_identity=session_identity,
                message_text=f"[Scheduled Event] {reminder.message}",
                state=state,
                inference_profile=profile,
            )
            await self._send_final_response(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                response_text=final_response,
            )
        except Exception:
            logger.exception(
                "Error processing scheduled reminder %s for chat %s",
                reminder.id,
                chat_id,
            )
            text = format_for_telegram(f"⏰ *Reminder*\n\n{reminder.message}")
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=message_thread_id,
            )

    async def _send_final_response(
        self,
        *,
        chat_id: int,
        message_thread_id: int | None,
        response_text: str,
    ) -> None:
        """Send the final assistant response, splitting long messages if needed."""
        formatted_response = format_for_telegram(response_text)
        message_chunks = split_long_message(formatted_response)

        if not message_chunks:
            message_chunks = ["I apologize, but I couldn't generate a response\\."]

        for message_chunk in message_chunks:
            try:
                await self.api.send_message(
                    chat_id=chat_id,
                    text=message_chunk,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=message_thread_id,
                )
            except TelegramApiError as error:
                error_text = str(error).casefold()
                parse_failure = error.error_code == 400 and (
                    "can't parse entities" in error_text
                    or "can't find end of" in error_text
                )
                if not parse_failure:
                    raise

                logger.warning(
                    "Telegram rejected MarkdownV2 entities; retrying as plain text"
                )
                await self.api.send_message(
                    chat_id=chat_id,
                    text=message_chunk,
                    parse_mode=None,
                    message_thread_id=message_thread_id,
                )

    def _build_session_identity(
        self,
        *,
        chat_id: str,
        message_thread_id: int | None,
    ) -> TelegramSessionIdentity:
        """Build the stable ADK identity for a Telegram chat or topic thread."""
        conversation_key = self._build_conversation_key(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
        )
        stable_identity = f"telegram-{conversation_key}"
        return TelegramSessionIdentity(
            conversation_key=conversation_key,
            user_id=stable_identity,
            session_id_prefix=stable_identity,
        )

    def _build_conversation_key(
        self,
        *,
        chat_id: str,
        message_thread_id: int | None,
    ) -> str:
        """Build a stable Telegram conversation key."""
        if message_thread_id is None:
            return f"chat-{chat_id}"

        return f"chat-{chat_id}-thread-{message_thread_id}"

    def _build_session_state(
        self,
        *,
        chat_id: str,
        message_thread_id: int | None,
        conversation_key: str,
    ) -> dict[str, str]:
        """Build explicit session state for ADK callbacks and observability."""
        session_state: dict[str, str] = {
            "user_id": f"telegram-{conversation_key}",
            "telegram_chat_id": chat_id,
            "telegram_conversation_key": conversation_key,
        }
        if message_thread_id is not None:
            session_state["telegram_thread_id"] = str(message_thread_id)
        return session_state


def create_telegram_bot(
    config: TelegramConfig,
    runtime: AdkRuntime,
) -> TelegramBot | None:
    """Create a Telegram bot instance if configured."""
    if not config.is_configured():
        logger.info("Telegram bot not configured, skipping initialization")
        return None

    return TelegramBot(config, runtime)
