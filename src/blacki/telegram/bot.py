"""Telegram bot client backed by the shared ADK runtime."""

import asyncio
import contextlib
import logging
import re
from collections.abc import Coroutine, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from google.genai import types

from blacki.adk_runtime import AdkRuntime, EmptyModelResponseError, SessionLocator
from blacki.health.config import telegram_chat_id_for_health_user
from blacki.health.service import (
    GoogleHealthOAuthError,
    GoogleHealthService,
    SyncResult,
    format_health_summary,
)
from blacki.inference import (
    InferenceProfile,
    inference_profile_from_environment,
    load_inference_profile,
)
from blacki.reminders.storage import Reminder
from blacki.utils.preferences import get_preferences_storage

from . import TelegramConfig
from .api import TelegramApiClient, TelegramApiError
from .formatting import escape_markdown_plain, format_for_telegram
from .settings_menu import SettingsMenu
from .streaming import split_long_message
from .types import (
    BotCommand,
    CallbackQuery,
    ChatType,
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
_MAX_ALBUM_PHOTOS = 10
_MAX_ALBUM_BYTES = 20 * 1024 * 1024
_ALBUM_DEBOUNCE_SECONDS = 0.5
_ALBUM_MAX_WAIT_SECONDS = 2.0


@dataclass(slots=True)
class _BufferedAlbum:
    """In-memory buffer for a Telegram media group (photo album)."""

    chat_id: int
    message_thread_id: int | None
    media_group_id: str
    chat_type: ChatType | None
    messages: list[Message]
    debounce_handle: asyncio.TimerHandle | None = None
    max_wait_task: asyncio.Task[None] | None = None
    future: asyncio.Future[None] | None = None
    processed: bool = False
    created_seq: int = 0


def _format_health_sync_result(result: SyncResult) -> str:
    """Render a provider-sync result without exposing IDs or error payloads."""
    if result.status == "not_connected":
        return "Google Health is not connected. Use /connect_health first."
    if result.status == "reauthorization_required":
        return (
            "Google Health needs authorization again. Use /connect_health to reconnect."
        )
    if result.status == "rate_limited":
        return "A Google Health refresh was requested recently. Please try again later."
    if result.status == "success":
        return (
            f"Google Health refreshed {result.days_upserted} day(s) from "
            f"{result.records_fetched} record(s)."
        )
    return "Google Health could not be refreshed right now. Please try again later."


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
        google_health_service: GoogleHealthService | None = None,
    ) -> None:
        """Initialize the Telegram bot."""
        self.config = config
        self.runtime = runtime
        self.google_health_service = google_health_service
        self._api: TelegramApiClient | None = None
        self._running = False
        self._polling_task: asyncio.Task[None] | None = None
        self._conversation_tasks: dict[str, asyncio.Task[None]] = {}
        self._conversation_task_seqs: dict[str, int] = {}
        self._album_buffers: dict[tuple[int, int | None, str], _BufferedAlbum] = {}
        self._update_counter: int = 0
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._chat_type_context: ContextVar[ChatType | None] = ContextVar(
            "telegram_chat_type", default=None
        )
        self._settings_menu = SettingsMenu(
            api_provider=lambda: self.api,
            load_profile=self._load_chat_profile,
        )

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

        for album in list(self._album_buffers.values()):
            self._cleanup_album_buffer(album)

        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await self.runtime.close()

        await self._settings_menu.aclose()

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
        if self.google_health_service is not None:
            commands.extend(
                [
                    BotCommand(
                        command="connect_health",
                        description="Connect Google Health read-only data",
                    ),
                    BotCommand(
                        command="health_summary",
                        description="Show your Google Health summary",
                    ),
                    BotCommand(
                        command="health_refresh",
                        description="Refresh Google Health data",
                    ),
                    BotCommand(
                        command="disconnect_health",
                        description="Disconnect and delete Google Health data",
                    ),
                ]
            )
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

    def _cleanup_album_buffer(self, album: _BufferedAlbum) -> None:
        """Cancel and remove timers and tasks for a buffered album."""
        key = (album.chat_id, album.message_thread_id, album.media_group_id)
        self._album_buffers.pop(key, None)

        if album.debounce_handle is not None:
            album.debounce_handle.cancel()
            album.debounce_handle = None

        if album.max_wait_task is not None:
            album.max_wait_task.cancel()
            album.max_wait_task = None

        if album.future is not None and not album.future.done():
            album.future.cancel()

    def _get_active_album_buffers(
        self, chat_id: int, message_thread_id: int | None
    ) -> list[_BufferedAlbum]:
        """Return active album buffers for the given conversation."""
        return [
            album
            for (cid, tid, _), album in self._album_buffers.items()
            if cid == chat_id and tid == message_thread_id and not album.processed
        ]

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

        # Check if this update is part of a photo album
        if update.message.photo and update.message.media_group_id is not None:
            await self._buffer_album_message(update.message)
            return

        chat_id = update.message.chat.id
        message_thread_id = update.message.message_thread_id
        conversation_key = self._build_conversation_key(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
        )

        self._update_counter += 1
        current_seq = self._update_counter

        for active_album in self._get_active_album_buffers(chat_id, message_thread_id):
            if active_album.future is not None:
                try:
                    await asyncio.shield(active_album.future)
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling() > 0:
                        raise
                except Exception as exc:
                    logger.debug("Album buffer wait suppressed error: %s", exc)

        await self._run_sequenced_turn(
            conversation_key, current_seq, self._handle_update(update)
        )

    async def _run_sequenced_turn(
        self,
        conversation_key: str,
        seq: int,
        turn: Coroutine[Any, Any, None],
    ) -> None:
        """Run a turn, cancelling and waiting out any superseded turn first."""
        existing_task = self._conversation_tasks.get(conversation_key)
        existing_seq = self._conversation_task_seqs.get(conversation_key, 0)
        supersedes_existing = (
            existing_task is not None
            and not existing_task.done()
            and seq >= existing_seq
        )
        if supersedes_existing and existing_task is not None:
            logger.info(
                "Cancelling in-flight turn for conversation %s", conversation_key
            )
            existing_task.cancel()

        current_task = asyncio.current_task()
        if current_task is not None:
            self._conversation_tasks[conversation_key] = current_task
            self._conversation_task_seqs[conversation_key] = seq

        try:
            # Wait for the superseded task to fully clean up before starting
            if supersedes_existing and existing_task is not None:
                await asyncio.wait([existing_task])

            await turn
        except asyncio.CancelledError:
            logger.info("Message turn superseded for conversation %s", conversation_key)
            raise
        finally:
            if self._conversation_tasks.get(conversation_key) is current_task:
                self._conversation_tasks.pop(conversation_key, None)
                self._conversation_task_seqs.pop(conversation_key, None)

    async def _buffer_album_message(self, message: Message) -> None:
        """Buffer an incoming album message with debounce and max-wait."""
        chat_id = message.chat.id
        message_thread_id = message.message_thread_id
        media_group_id = cast(str, message.media_group_id)
        key = (chat_id, message_thread_id, media_group_id)

        self._update_counter += 1
        current_seq = self._update_counter

        album = self._album_buffers.get(key)
        if album is None:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[None] = loop.create_future()
            album = _BufferedAlbum(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                media_group_id=media_group_id,
                chat_type=message.chat.type,
                messages=[message],
                future=future,
                created_seq=current_seq,
            )
            self._album_buffers[key] = album

            # Schedule max wait timer
            max_wait_task = asyncio.create_task(self._album_max_wait(album))
            self._background_tasks.add(max_wait_task)
            max_wait_task.add_done_callback(self._background_tasks.discard)
            album.max_wait_task = max_wait_task
        else:
            album.messages.append(message)
            if album.debounce_handle is not None:
                album.debounce_handle.cancel()

        # Set or reset debounce timer
        loop = asyncio.get_running_loop()
        album.debounce_handle = loop.call_later(
            _ALBUM_DEBOUNCE_SECONDS,
            self._on_album_debounce_expired,
            album,
        )

        try:
            if album.future is not None:
                await asyncio.shield(album.future)
        except asyncio.CancelledError:
            raise

    async def _album_max_wait(self, album: _BufferedAlbum) -> None:
        """Flush album buffer after maximum wait timeout."""
        try:
            await asyncio.sleep(_ALBUM_MAX_WAIT_SECONDS)
            self._flush_album(album)
        except asyncio.CancelledError:
            pass

    def _on_album_debounce_expired(self, album: _BufferedAlbum) -> None:
        """Handle debounce timer expiration by flushing the album."""
        self._flush_album(album)

    def _flush_album(self, album: _BufferedAlbum) -> None:
        """Flush buffered album messages into a conversation turn."""
        if album.processed:
            return
        album.processed = True

        key = (album.chat_id, album.message_thread_id, album.media_group_id)
        self._album_buffers.pop(key, None)

        if album.debounce_handle is not None:
            album.debounce_handle.cancel()
            album.debounce_handle = None
        if album.max_wait_task is not None:
            album.max_wait_task.cancel()
            album.max_wait_task = None

        task = asyncio.create_task(self._process_flushed_album(album))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _process_flushed_album(self, album: _BufferedAlbum) -> None:
        """Enqueue flushed album as a conversation turn, respecting cancellation."""
        conversation_key = self._build_conversation_key(
            chat_id=str(album.chat_id),
            message_thread_id=album.message_thread_id,
        )
        try:
            await self._run_sequenced_turn(
                conversation_key,
                album.created_seq,
                self._handle_album_turn(album),
            )
        finally:
            if album.future is not None and not album.future.done():
                album.future.set_result(None)

    async def _handle_album_turn(self, album: _BufferedAlbum) -> None:
        """Download album images and run a single ADK turn."""
        chat_id = album.chat_id
        message_thread_id = album.message_thread_id
        messages = album.messages

        if len(messages) > _MAX_ALBUM_PHOTOS:
            await self._send_photo_error(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=(
                    f"❌ The album contains too many photos "
                    f"({_MAX_ALBUM_PHOTOS} maximum)."
                ),
            )
            return

        # Check total reported size before download
        total_reported_size = 0
        photo_items: list[tuple[str, int | None]] = []
        caption: str | None = None

        for msg in messages:
            if msg.caption and msg.caption.strip() and caption is None:
                caption = msg.caption.strip()
            if not msg.photo:
                await self._send_photo_error(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text="❌ Sorry, I failed to process the photo.",
                )
                return
            best_photo = max(msg.photo, key=lambda item: item.width * item.height)
            if best_photo.file_size is not None:
                if best_photo.file_size > _MAX_NATIVE_IMAGE_BYTES:
                    await self._send_photo_error(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text="❌ The photo is too large to process (10 MB maximum).",
                    )
                    return
                total_reported_size += best_photo.file_size
            photo_items.append((best_photo.file_id, best_photo.file_size))

        if total_reported_size > _MAX_ALBUM_BYTES:
            await self._send_photo_error(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="❌ The album is too large to process (20 MB maximum).",
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
            chat_type=album.chat_type,
        )

        try:
            await self.api.send_chat_action(
                chat_id=chat_id,
                action="typing",
                message_thread_id=message_thread_id,
            )

            downloaded_images: list[bytes] = []
            total_downloaded_bytes = 0

            for file_id, _ in photo_items:
                image_bytes = await self._download_and_validate_photo(file_id)

                total_downloaded_bytes += len(image_bytes)
                if total_downloaded_bytes > _MAX_ALBUM_BYTES:
                    raise ValueError("Album total downloaded size exceeds limit")

                downloaded_images.append(image_bytes)

            prompt = caption if caption is not None else _DEFAULT_IMAGE_PROMPT

            parts: list[types.Part] = [types.Part.from_text(text=prompt)]
            for img_bytes in downloaded_images:
                parts.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                )

            await self._run_turn_and_send_response(
                session_identity=session_identity,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                state=state,
                message_text=prompt,
                user_parts=parts,
            )
        except Exception:
            logger.exception("Failed to handle Telegram photo album")
            await self._send_photo_error(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="❌ Sorry, I failed to process the photo.",
            )

    async def _handle_update(self, update: Update) -> None:
        """Handle an incoming update."""
        if update.message is None:
            return

        message = update.message
        token = self._chat_type_context.set(message.chat.type)
        try:
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
        finally:
            self._chat_type_context.reset(token)

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
        chat_type: ChatType | None = None,
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
            chat_type=chat_type or self._chat_type_context.get(),
        )

        try:
            await self.api.send_chat_action(
                chat_id=chat_id,
                action="typing",
                message_thread_id=message_thread_id,
            )
            image_bytes = await self._download_and_validate_photo(file_id)

            prompt = (
                caption.strip()
                if caption and caption.strip()
                else _DEFAULT_IMAGE_PROMPT
            )
            user_parts = (
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            )
            await self._run_turn_and_send_response(
                session_identity=session_identity,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                state=state,
                message_text=prompt,
                user_parts=user_parts,
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

    async def _download_and_validate_photo(self, file_id: str) -> bytes:
        """Download a Telegram photo and validate its size and format."""
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
        return image_bytes

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
            await self._settings_menu.send_model_menu(
                chat_id, message.message_thread_id
            )
        elif command == "/thinking":
            await self._settings_menu.send_thinking_menu(
                chat_id, message.message_thread_id
            )
        elif command == "/connect_health":
            await self._connect_health(message)
        elif command == "/health_summary":
            await self._send_health_summary(message)
        elif command == "/health_refresh":
            await self._refresh_health(message)
        elif command == "/disconnect_health":
            await self._request_health_disconnect(message)

    async def _connect_health(self, message: Message) -> None:
        """Send a short-lived Google authorization link to a private chat."""
        if not await self._health_command_ready(message):
            return
        service = cast(GoogleHealthService, self.google_health_service)
        try:
            url = await service.begin_authorization(f"telegram-chat-{message.chat.id}")
            await self.api.send_message(
                chat_id=message.chat.id,
                text=(
                    "Google Health connection is read-only. It can summarize "
                    "data that reaches Google Health from Fitbit-compatible sources. "
                    "Blacki does not receive Apple ID credentials or raw Apple Health "
                    "records. Authorize only if you want this private chat to read "
                    "the selected health categories."
                ),
                message_thread_id=message.message_thread_id,
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="Connect Google Health",
                                url=url,
                                callback_data=None,
                            )
                        ]
                    ]
                ),
                protect_content=True,
            )
        except (GoogleHealthOAuthError, ValueError):
            logger.exception("Failed to create Google Health authorization link")
            await self._send_health_text(
                message,
                "Google Health is not available for this chat right now.",
            )

    async def _send_health_summary(self, message: Message) -> None:
        """Send the latest normalized health summary without provider access."""
        if not await self._health_command_ready(message):
            return
        service = cast(GoogleHealthService, self.google_health_service)
        try:
            summary = await service.summary(f"telegram-chat-{message.chat.id}")
            await self._send_health_text(message, format_health_summary(summary))
        except GoogleHealthOAuthError:
            await self._send_health_text(
                message,
                "Google Health summaries are available only in a private chat.",
            )
        except Exception:
            logger.exception("Failed to read Google Health summary")
            await self._send_health_text(
                message,
                "I couldn't read your Google Health summary right now.",
            )

    async def _refresh_health(self, message: Message) -> None:
        """Fetch a bounded window of provider data and show the result."""
        if not await self._health_command_ready(message):
            return
        service = cast(GoogleHealthService, self.google_health_service)
        try:
            result = await service.refresh_user(f"telegram-chat-{message.chat.id}")
            if result.status == "success":
                summary = await service.summary(f"telegram-chat-{message.chat.id}")
                text = format_health_summary(summary)
            else:
                text = _format_health_sync_result(result)
            await self._send_health_text(message, text)
        except GoogleHealthOAuthError:
            await self._send_health_text(
                message,
                "Google Health refreshes are available only in a private chat.",
            )
        except Exception:
            logger.exception("Failed to refresh Google Health data")
            await self._send_health_text(
                message,
                "I couldn't refresh your Google Health data right now.",
            )

    async def _request_health_disconnect(self, message: Message) -> None:
        """Ask for a final Telegram click before deleting local health data."""
        if not await self._health_command_ready(message):
            return
        await self.api.send_message(
            chat_id=message.chat.id,
            text=(
                "Disconnect Google Health and delete Blacki's stored health "
                "data for this chat? This cannot be undone locally."
            ),
            message_thread_id=message.message_thread_id,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="Disconnect and delete data",
                            callback_data="health:disconnect",
                        ),
                        InlineKeyboardButton(
                            text="Cancel",
                            callback_data="health:cancel",
                        ),
                    ]
                ]
            ),
            protect_content=True,
        )

    async def _health_command_ready(self, message: Message) -> bool:
        """Require a configured connector and a private Telegram chat."""
        if message.chat.type != ChatType.PRIVATE:
            await self._send_health_text(
                message,
                "Google Health is available only in a private Telegram chat.",
            )
            return False
        if self.google_health_service is None:
            await self._send_health_text(
                message,
                "Google Health is not configured on this Blacki server yet.",
            )
            return False
        return True

    async def _send_health_text(self, message: Message, text: str) -> None:
        """Send private health content with Telegram forwarding protection."""
        await self.api.send_message(
            chat_id=message.chat.id,
            text=text,
            message_thread_id=message.message_thread_id,
            protect_content=True,
        )

    async def _handle_health_callback(self, query: CallbackQuery) -> None:
        """Handle the explicit disconnect confirmation from a private chat."""
        if query.message is None:
            await self.api.answer_callback_query(query.id, text="Confirmation expired")
            return
        chat = query.message.chat
        if chat.type != ChatType.PRIVATE or query.from_user.id != chat.id:
            await self.api.answer_callback_query(query.id, text="Not authorized")
            return
        if self.google_health_service is None:
            await self.api.answer_callback_query(query.id, text="Not configured")
            return
        if query.data == "health:cancel":
            await self.api.answer_callback_query(query.id, text="Cancelled")
            return

        await self.api.answer_callback_query(query.id, text="Disconnecting…")
        try:
            deleted = await self.google_health_service.disconnect(
                f"telegram-chat-{chat.id}"
            )
            text = (
                "Google Health was disconnected and stored health data was deleted."
                if deleted
                else "Google Health was already disconnected."
            )
            await self._send_health_text(query.message, text)
        except Exception:
            logger.exception("Failed to disconnect Google Health")
            await self._send_health_text(
                query.message,
                "I couldn't finish disconnecting Google Health. Please try again.",
            )

    async def notify_health_connection(
        self, telegram_user_id: str, *, connected: bool
    ) -> None:
        """Notify the originating private chat after OAuth callback completion."""
        chat_id = telegram_chat_id_for_health_user(telegram_user_id)
        if chat_id is None:
            return
        text = (
            "Google Health is connected. Use /health_refresh for a fresh sync or "
            "/health_summary to read the latest stored records."
            if connected
            else (
                "Google Health authorization was cancelled. No credentials were stored."
            )
        )
        await self.api.send_message(
            chat_id=chat_id,
            text=text,
            protect_content=True,
        )

    async def _handle_callback_query(self, query: CallbackQuery) -> None:
        """Handle incoming callback query."""
        data = query.data or ""
        if data.startswith("health:"):
            await self._handle_health_callback(query)
            return
        await self._settings_menu.handle_callback(query)

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

    async def _send_start_message(self, chat_id: int) -> None:
        """Send the start/welcome message."""
        health_commands = ""
        if self.google_health_service is not None:
            health_commands = (
                "\n"
                "/connect_health - Connect Google Health read-only data\n"
                "/health_summary - Show the latest health summary\n"
                "/health_refresh - Refresh health data\n"
                "/disconnect_health - Disconnect and delete health data"
            )
        text = escape_markdown_plain(
            "👋 Hello! I'm blacki, your AI assistant.\n\n"
            "I run through the same ADK agent as the web interface, so our "
            "conversation history stays attached to this chat.\n\n"
            "Commands:\n"
            "/help - Show available commands\n"
            "/reset - Start a fresh conversation session\n"
            "/model - Choose the model and thinking settings\n"
            "/thinking - Choose supported reasoning effort"
            f"{health_commands}"
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
        health_commands = ""
        if self.google_health_service is not None:
            health_commands = (
                "• /connect_health \\- Connect Google Health read-only data\n"
                "• /health_summary \\- Show the latest health summary\n"
                "• /health_refresh \\- Refresh health data\n"
                "• /disconnect_health \\- Disconnect and delete health data\n"
            )
        text = (
            "🤖 *blacki \\- AI Assistant*\n\n"
            "I'm powered by the same Google ADK runtime used by the HTTP app\\.\n\n"
            "*Commands:*\n"
            "• /start \\- Start a conversation\n"
            "• /help \\- Show this help message\n"
            "• /reset \\- Start a fresh conversation session\n"
            "• /model \\- Choose the model and thinking settings\n"
            "• /thinking \\- Choose supported reasoning effort\n"
            f"{health_commands}\n"
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
        chat_type: ChatType | None = None,
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
            chat_type=chat_type or self._chat_type_context.get(),
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

            await self._run_turn_and_send_response(
                session_identity=session_identity,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                state=state,
                message_text=user_message,
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
                invocation_id = error.invocation_id
                if (
                    retry_count >= _MAX_EMPTY_RESPONSE_RETRIES
                    or not error.retryable
                    or invocation_id is None
                ):
                    logger.warning(
                        "Empty model response recovery stopped: "
                        "model=%s provider=%s conversation_id=%s "
                        "invocation_id=%s retry_count=%d retryable=%s",
                        model,
                        provider,
                        session_identity.conversation_key,
                        invocation_id or "unknown",
                        retry_count,
                        error.retryable,
                    )
                    raise

                await self.runtime.rewind_empty_model_response(
                    locator=SessionLocator(
                        user_id=session_identity.user_id,
                        session_id_prefix=session_identity.session_id_prefix,
                    ),
                    invocation_id=invocation_id,
                )
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

    async def _run_turn_and_send_response(
        self,
        *,
        session_identity: TelegramSessionIdentity,
        chat_id: int,
        message_thread_id: int | None,
        state: dict[str, str],
        message_text: str,
        user_parts: Sequence[types.Part] | None = None,
    ) -> None:
        """Load the chat profile, run a turn with retry, and send the response."""
        profile = await self._load_chat_profile(chat_id)
        final_response = await self._run_user_turn_with_retry(
            session_identity=session_identity,
            message_text=message_text,
            state=state,
            user_parts=user_parts,
            inference_profile=profile,
        )
        await self._send_final_response(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            response_text=final_response,
        )

    async def _handle_message(
        self,
        chat_id: int,
        message_thread_id: int | None,
        user_message: str,
        chat_type: ChatType | None = None,
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
                chat_type=chat_type or self._chat_type_context.get(),
            )
            await self._run_turn_and_send_response(
                session_identity=session_identity,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                state=state,
                message_text=user_message,
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
                message_thread_id=message_thread_id,
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
            await self._run_turn_and_send_response(
                session_identity=session_identity,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                state=state,
                message_text=f"[Scheduled Event] {reminder.message}",
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
        chat_type: ChatType | None = None,
    ) -> dict[str, str]:
        """Build explicit session state for ADK callbacks and observability."""
        session_state: dict[str, str] = {
            "user_id": f"telegram-{conversation_key}",
            "telegram_chat_id": chat_id,
            "telegram_conversation_key": conversation_key,
        }
        if message_thread_id is not None:
            session_state["telegram_thread_id"] = str(message_thread_id)
        if chat_type is not None:
            session_state["telegram_chat_type"] = chat_type.value
        return session_state


def create_telegram_bot(
    config: TelegramConfig,
    runtime: AdkRuntime,
    google_health_service: GoogleHealthService | None = None,
) -> TelegramBot | None:
    """Create a Telegram bot instance if configured."""
    if not config.is_configured():
        logger.info("Telegram bot not configured, skipping initialization")
        return None

    return TelegramBot(config, runtime, google_health_service)
