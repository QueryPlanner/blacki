"""Telegram bot client backed by the shared ADK runtime."""

import asyncio
import contextlib
import hashlib
import hmac
import logging
import re
from collections.abc import Coroutine, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

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
from .access import TelegramAccessStorage, TelegramIdentity, get_telegram_access_storage
from .album_buffer import AlbumBuffer, _BufferedAlbum
from .api import TelegramApiClient, TelegramApiError
from .formatting import escape_markdown_plain, format_for_telegram
from .settings_menu import SettingsMenu
from .streaming import split_long_message
from .transcription import (
    MAX_CLOUDFLARE_WHISPER_AUDIO_BYTES,
    MAX_CONCURRENT_CLOUDFLARE_TRANSCRIPTIONS,
    CloudflareWhisperError,
    CloudflareWhisperTranscriber,
)
from .types import (
    BotCommand,
    CallbackQuery,
    ChatType,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ParseMode,
    Update,
    User,
)

if TYPE_CHECKING:
    from blacki.user_files import IngestResult

logger = logging.getLogger(__name__)

POLLING_TIMEOUT = 30
_MAX_CONSECUTIVE_ERRORS = 5
_FATAL_ERROR_CODES = {401, 403}
_MAX_EMPTY_RESPONSE_RETRIES = 1
_TELEGRAM_USER_ID_PATTERN = re.compile(r"^telegram-chat-(-?\d+)(?:-thread-(\d+))?$")
_MAX_NATIVE_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_TELEGRAM_FILE_BYTES = 20 * 1024 * 1024
_JPEG_MAGIC = b"\xff\xd8\xff"
_DEFAULT_IMAGE_PROMPT = "Describe this image."
_MAX_ALBUM_PHOTOS = 10
_MAX_ALBUM_BYTES = 20 * 1024 * 1024


def _format_google_health_sync_counts(value: object) -> str:
    """Render safe durable meal-sync counts without provider details."""
    if not isinstance(value, Mapping):
        return ""

    labels = (
        ("pending", "pending"),
        ("synced", "synced"),
        ("failed", "failed"),
        ("authorization_required", "awaiting authorization"),
    )
    parts: list[str] = []
    for key, label in labels:
        count = value.get(key)
        if isinstance(count, int) and count >= 0:
            parts.append(f"{count} {label}")
    if not parts:
        return ""
    return "Meal sync status (pending includes deletions): " + ", ".join(parts)


def _format_health_sync_result(result: SyncResult) -> str:
    """Render a provider-sync result without exposing IDs or error payloads."""
    if result.status == "not_connected":
        text = "Google Health is not connected. Use /connect_health first."
        return _append_google_health_sync_counts(text, result)
    if result.status == "reauthorization_required":
        text = (
            "Google Health needs authorization again. Use /connect_health to reconnect."
        )
        return _append_google_health_sync_counts(text, result)
    if result.status == "rate_limited":
        text = "A Google Health refresh was requested recently. Please try again later."
        return _append_google_health_sync_counts(text, result)
    if result.status == "success":
        text = (
            f"Google Health refreshed {result.days_upserted} day(s) from "
            f"{result.records_fetched} record(s)."
        )
        return _append_google_health_sync_counts(text, result)
    text = "Google Health could not be refreshed right now. Please try again later."
    return _append_google_health_sync_counts(text, result)


def _append_google_health_sync_counts(text: str, result: object) -> str:
    """Append durable meal-sync counts when the health service provides them."""
    counts = _format_google_health_sync_counts(
        getattr(result, "google_health_sync", None)
    )
    return f"{text}\n{counts}" if counts else text


@dataclass(slots=True, frozen=True)
class TelegramSessionIdentity:
    """Stable Telegram identifiers used to resolve ADK sessions."""

    conversation_key: str
    user_id: str
    session_id_prefix: str


class VoiceTranscriber(Protocol):
    """Protocol implemented by the Telegram voice transcription service."""

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Return the transcript for one voice note."""

    async def close(self) -> None:
        """Release any provider resources."""


class TelegramBot:
    """Telegram bot client that sends typing indicators and final replies."""

    def __init__(
        self,
        config: TelegramConfig,
        runtime: AdkRuntime,
        google_health_service: GoogleHealthService | None = None,
        access_storage: TelegramAccessStorage | None = None,
        voice_transcriber: VoiceTranscriber | None = None,
    ) -> None:
        """Initialize the Telegram bot."""
        self.config = config
        self.runtime = runtime
        self.google_health_service = google_health_service
        self.access_storage = access_storage
        self._voice_transcriber = (
            voice_transcriber or CloudflareWhisperTranscriber.from_environment()
        )
        self._voice_transcription_semaphore = asyncio.Semaphore(
            MAX_CONCURRENT_CLOUDFLARE_TRANSCRIPTIONS
        )
        self._api: TelegramApiClient | None = None
        self._running = False
        self._polling_task: asyncio.Task[None] | None = None
        self._conversation_tasks: dict[str, asyncio.Task[None]] = {}
        self._conversation_task_seqs: dict[str, int] = {}
        self._album_buffer = AlbumBuffer(on_flush=self._on_album_flushed)
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

        await self._album_buffer.shutdown()

        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

        try:
            await self.runtime.close()
        finally:
            if self._voice_transcriber is not None:
                await self._voice_transcriber.close()

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
                        description="Connect Google Health and meal sync",
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
                        description="Disconnect Google Health and meal sync",
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

    async def _safe_handle_update(self, update: Update) -> None:
        """Handle update concurrently and allow cancellation."""
        if not await self._authorize_update(update):
            return

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
            self._update_counter += 1
            await self._album_buffer.add_message(update.message, self._update_counter)
            return

        chat_id = update.message.chat.id
        message_thread_id = update.message.message_thread_id
        conversation_key = self._build_conversation_key(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
        )

        self._update_counter += 1
        current_seq = self._update_counter

        for active_album in self._album_buffer.get_active(chat_id, message_thread_id):
            if active_album.future is not None:
                try:
                    await asyncio.shield(active_album.future)
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling() > 0:
                        raise
                except Exception as exc:
                    # A failed album turn already reports its own user-facing
                    # error via _send_photo_error inside _handle_album_turn.
                    # This wait only unblocks processing of the next message
                    # in this conversation, so it must not raise on the
                    # album's behalf, but it is logged at warning level (not
                    # debug) so an unexpected failure here stays visible.
                    logger.warning("Album buffer wait suppressed error: %s", exc)

        await self._run_sequenced_turn(
            conversation_key, current_seq, self._handle_update(update)
        )

    def _access_code_fingerprint(self) -> str:
        """Return a non-plaintext marker used to invalidate rotated access codes."""
        access_code = self.config.telegram_access_code
        if access_code is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Telegram access code is not configured")
        return hashlib.sha256(access_code.encode("utf-8")).hexdigest()

    def _get_access_storage(self) -> TelegramAccessStorage:
        """Get the injected or process-wide access storage."""
        return self.access_storage or get_telegram_access_storage()

    async def _authorize_update(self, update: Update) -> bool:
        """Allow only authenticated private Telegram traffic into the bot."""
        message = update.message
        callback = update.callback_query
        if message is None and callback is not None:
            message = callback.message

        sender = (
            callback.from_user
            if callback is not None
            else (message.from_user if message is not None else None)
        )
        if not self.config.access_control_enabled:
            await self._record_identity_if_private(message, sender)
            return True

        if message is None or message.chat.type != ChatType.PRIVATE:
            if callback is not None:
                await self.api.answer_callback_query(
                    callback.id, text="Access required"
                )
            return False

        if sender is None or sender.id != message.chat.id:
            return False

        storage = self._get_access_storage()
        fingerprint = self._access_code_fingerprint()
        try:
            authorized = await storage.is_authorized(sender.id, fingerprint)
            has_authorization_record = await storage.has_authorization_record(sender.id)
            if not authorized and not has_authorization_record:
                authorized = await self._grant_legacy_access_if_applicable(
                    message, sender, storage
                )
            if authorized:
                await storage.record_identity(self._identity_from_user(sender))
                return True
            if callback is not None:
                await self.api.answer_callback_query(
                    callback.id, text="Access required"
                )
                return False
            return await self._handle_new_user_start(message, sender, storage)
        except Exception:
            logger.exception("Telegram access control failed closed")
            return False

    async def _record_identity_if_private(
        self,
        message: Message | None,
        sender: User | None,
    ) -> None:
        """Persist a direct-chat sender label when local storage is available."""
        if (
            message is None
            or message.chat.type != ChatType.PRIVATE
            or sender is None
            or sender.id != message.chat.id
        ):
            return

        try:
            await self._get_access_storage().record_identity(
                self._identity_from_user(sender)
            )
        except RuntimeError:
            logger.debug("Telegram identity storage is unavailable; skipping label")
        except Exception:
            logger.exception("Telegram identity recording failed; continuing update")

    async def _grant_legacy_access_if_applicable(
        self,
        message: Message,
        sender: User,
        storage: TelegramAccessStorage,
    ) -> bool:
        """Grandfather a historical direct chat without changing its session key."""
        session_identity = self._build_session_identity(
            chat_id=str(message.chat.id),
            message_thread_id=None,
        )
        has_history = await self.runtime.has_existing_session(
            locator=SessionLocator(
                user_id=session_identity.user_id,
                session_id_prefix=session_identity.session_id_prefix,
            )
        )
        if not has_history:
            return False
        await storage.grant(sender.id, source="legacy")
        return True

    async def _handle_new_user_start(
        self,
        message: Message,
        sender: User,
        storage: TelegramAccessStorage,
    ) -> bool:
        """Authenticate a new private user through the locally consumed /start code."""
        command, separator, supplied_code = (message.text or "").partition(" ")
        if command != "/start" or not separator:
            await self.api.send_message(
                chat_id=message.chat.id,
                text="Access required. Send /start followed by your access code.",
            )
            return False

        configured_code = self.config.telegram_access_code
        if configured_code is None:  # pragma: no cover - guarded by caller
            return False
        is_valid = hmac.compare_digest(supplied_code.strip(), configured_code)
        await self._delete_access_code_message(message)
        if not is_valid:
            await self.api.send_message(
                chat_id=message.chat.id,
                text="That access code is not valid. Please try again.",
            )
            return False

        await storage.grant(
            sender.id,
            source="passphrase",
            access_code_fingerprint=self._access_code_fingerprint(),
        )
        await storage.record_identity(self._identity_from_user(sender))
        await self._send_start_message(message.chat.id)
        return False

    async def _delete_access_code_message(self, message: Message) -> None:
        """Best-effort removal of an access code from the visible private chat."""
        try:
            await self.api.delete_message(message.chat.id, message.message_id)
        except TelegramApiError:
            logger.warning("Could not delete Telegram access-code message")

    def _identity_from_user(self, user: User) -> TelegramIdentity:
        """Build a bounded local-only display label from Telegram profile fields."""
        display_name = " ".join(
            part.strip()
            for part in (user.first_name, user.last_name or "")
            if part.strip()
        )[:128]
        return TelegramIdentity(
            user_id=user.id,
            display_name=display_name or "Telegram user",
            username=user.username.strip()[:64] if user.username else None,
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

    def _on_album_flushed(self, album: _BufferedAlbum) -> None:
        """Schedule processing of a completed album as a background task."""
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
        """Download album images and run a single ADK turn.

        Resolves ``album.future`` before returning on every exit path, so
        this method is safe to call directly and does not rely on a caller
        (``_process_flushed_album``) to resolve the future on its behalf.
        """
        try:
            await self._run_album_turn(album)
        finally:
            if album.future is not None and not album.future.done():
                album.future.set_result(None)

    async def _run_album_turn(self, album: _BufferedAlbum) -> None:
        """Validate, download, and process a flushed album's photos."""
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
                sender_user_id=message.from_user.id if message.from_user else None,
            )
        finally:
            self._chat_type_context.reset(token)

    async def _route_non_text_message(self, message: Message) -> None:
        """Route a non-text message to the appropriate handler."""
        chat_id = message.chat.id
        message_thread_id = message.message_thread_id
        sender_user_id = message.from_user.id if message.from_user else None

        if message.photo:
            photo = max(message.photo, key=lambda item: item.width * item.height)
            await self._handle_photo_upload(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                file_id=photo.file_id,
                file_unique_id=photo.file_unique_id,
                file_size=photo.file_size,
                caption=message.caption,
                sender_user_id=sender_user_id,
            )
            return

        if message.document:
            file_id = message.document.file_id
            file_unique_id = message.document.file_unique_id
            file_name = message.document.file_name or "document"
            file_size = message.document.file_size
            mime_type = message.document.mime_type
            media_kind = "document"
        elif message.audio:
            file_id = message.audio.file_id
            file_unique_id = message.audio.file_unique_id
            file_name = message.audio.file_name or "audio.mp3"
            file_size = message.audio.file_size
            mime_type = message.audio.mime_type
            media_kind = "audio"
        elif message.video:
            file_id = message.video.file_id
            file_unique_id = message.video.file_unique_id
            file_name = message.video.file_name or "video.mp4"
            file_size = message.video.file_size
            mime_type = message.video.mime_type
            media_kind = "video"
        elif message.voice:
            await self._handle_voice_upload(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                file_id=message.voice.file_id,
                file_size=message.voice.file_size,
                mime_type=message.voice.mime_type,
                caption=message.caption,
                sender_user_id=sender_user_id,
            )
            return
        else:
            logger.debug("Unsupported non-text message from chat %s", chat_id)
            return

        await self._handle_file_upload(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            file_id=file_id,
            file_unique_id=file_unique_id,
            file_name=file_name,
            file_size=file_size,
            mime_type=mime_type,
            media_kind=media_kind,
            caption=message.caption,
            sender_user_id=sender_user_id,
        )

    async def _handle_voice_upload(
        self,
        *,
        chat_id: int,
        message_thread_id: int | None,
        file_id: str,
        file_size: int | None,
        mime_type: str | None,
        caption: str | None,
        sender_user_id: int | None = None,
    ) -> None:
        """Transcribe a Telegram voice note and process it as a text turn."""
        if self._voice_transcriber is None:
            await self.api.send_message(
                chat_id=chat_id,
                text=(
                    "Voice transcription is not configured. Add "
                    "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to Blacki's "
                    "environment."
                ),
                message_thread_id=message_thread_id,
            )
            return

        if file_size is not None and file_size > MAX_CLOUDFLARE_WHISPER_AUDIO_BYTES:
            await self.api.send_message(
                chat_id=chat_id,
                text="❌ Voice notes must be 8 MB or smaller to transcribe.",
                message_thread_id=message_thread_id,
            )
            return

        try:
            async with self._voice_transcription_semaphore:
                await self.api.send_chat_action(
                    chat_id=chat_id,
                    action="typing",
                    message_thread_id=message_thread_id,
                )
                file_info = await self.api.get_file(file_id)
                file_path_api = file_info.get("file_path")
                if not file_path_api:
                    raise ValueError("Failed to get voice file path from Telegram API")

                audio_bytes = await self.api.download_file(file_path_api)
                if not audio_bytes:
                    raise ValueError("Telegram returned an empty voice note")
                if len(audio_bytes) > MAX_CLOUDFLARE_WHISPER_AUDIO_BYTES:
                    raise ValueError("Telegram voice note exceeds the 8 MB limit")

                transcript = await self._voice_transcriber.transcribe(audio_bytes)
                del audio_bytes

            user_message = transcript
            if caption and caption.strip():
                user_message = f"{caption.strip()}\n\n{transcript}"

            await self._handle_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                user_message=user_message,
                sender_user_id=sender_user_id,
            )
        except CloudflareWhisperError as exc:
            logger.warning(
                "Failed to transcribe Telegram voice note (%s)",
                type(exc).__name__,
            )
            await self.api.send_message(
                chat_id=chat_id,
                text=(
                    "❌ Sorry, I couldn't transcribe that voice note. Please try again."
                ),
                message_thread_id=message_thread_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to handle Telegram voice note (%s)",
                type(exc).__name__,
            )
            await self.api.send_message(
                chat_id=chat_id,
                text="❌ Sorry, I failed to process the voice note.",
                message_thread_id=message_thread_id,
            )

    async def _handle_photo_upload(
        self,
        *,
        chat_id: int,
        message_thread_id: int | None,
        file_id: str,
        file_size: int | None,
        caption: str | None,
        file_unique_id: str | None = None,
        sender_user_id: int | None = None,
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
            sender_user_id=sender_user_id,
        )

        try:
            await self.api.send_chat_action(
                chat_id=chat_id,
                action="typing",
                message_thread_id=message_thread_id,
            )
            image_bytes = await self._download_and_validate_photo(file_id)

            ingest, sandbox_path, sandbox_error = await self._shield_attachment_ingest(
                state=state,
                owner_id=str(sender_user_id) if sender_user_id is not None else None,
                display_name=f"photo-{file_unique_id or file_id}.jpg",
                media_kind="photo",
                mime_type="image/jpeg",
                telegram_file_unique_id=file_unique_id,
                data=image_bytes,
            )
            if ingest.warning:
                await self._send_storage_warning(
                    chat_id, message_thread_id, ingest.warning
                )
            if sandbox_error and ingest.stored_file is not None:
                await self.api.send_message(
                    chat_id=chat_id,
                    text=(
                        "✅ The photo was saved in durable storage, but the sandbox is "
                        "unavailable. Ask me to restore it later."
                    ),
                    message_thread_id=message_thread_id,
                )
                return

            prompt = (
                caption.strip()
                if caption and caption.strip()
                else _DEFAULT_IMAGE_PROMPT
            )
            message_text = prompt
            if sandbox_path:
                message_text = f"{prompt}\nSandbox working copy: {sandbox_path}"
            user_parts = (
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            )
            await self._run_turn_and_send_response(
                session_identity=session_identity,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                state=state,
                message_text=message_text,
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

    async def _shield_attachment_ingest(
        self,
        *,
        state: dict[str, str],
        owner_id: str | None,
        display_name: str,
        media_kind: str,
        mime_type: str | None,
        telegram_file_unique_id: str | None,
        data: bytes,
    ) -> tuple["IngestResult", str | None, str | None]:
        """Finish durable storage and sandbox materialization before cancellation."""
        task = asyncio.create_task(
            self._store_and_materialize_attachment(
                state=state,
                owner_id=owner_id,
                display_name=display_name,
                media_kind=media_kind,
                mime_type=mime_type,
                telegram_file_unique_id=telegram_file_unique_id,
                data=data,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _store_and_materialize_attachment(
        self,
        *,
        state: dict[str, str],
        owner_id: str | None,
        display_name: str,
        media_kind: str,
        mime_type: str | None,
        telegram_file_unique_id: str | None,
        data: bytes,
    ) -> tuple["IngestResult", str | None, str | None]:
        """Persist an attachment when configured and create its sandbox copy."""
        from blacki.sandbox.manager import get_sandbox_manager
        from blacki.user_files import get_user_file_service, user_files_enabled
        from blacki.user_files.service import IngestResult, sanitize_display_name

        if user_files_enabled():
            try:
                ingest = await get_user_file_service().ingest(
                    owner_id=owner_id,
                    display_name=display_name,
                    media_kind=media_kind,
                    mime_type=mime_type,
                    telegram_file_unique_id=telegram_file_unique_id,
                    data=data,
                )
            except Exception:
                logger.exception("Durable file service initialization failed")
                ingest = IngestResult(
                    None,
                    "temporary",
                    "R2 storage is misconfigured; this attachment is available "
                    "only temporarily.",
                )
        else:
            ingest = IngestResult(None, "temporary")

        manager = get_sandbox_manager()
        if not manager.config.enabled:
            return ingest, None, "Sandbox is disabled"
        result = await manager.get_or_create_sandbox(state)
        sandbox = result.get("sandbox")
        if sandbox is None:
            return ingest, None, str(result.get("error") or "Sandbox is unavailable")
        safe_name = sanitize_display_name(display_name)
        if ingest.stored_file is not None:
            safe_name = f"{ingest.stored_file.object_id}-{safe_name}"
        sandbox_path = f"/workspace/uploads/{safe_name}"
        await sandbox.files.write_file(sandbox_path, data)
        return ingest, sandbox_path, None

    async def _send_storage_warning(
        self, chat_id: int, message_thread_id: int | None, warning: str
    ) -> None:
        """Send a plain-text warning without leaking attachment metadata."""
        await self.api.send_message(
            chat_id=chat_id,
            text=f"⚠️ {warning}",
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
                    "Google Health can provide read-only wellness summaries from "
                    "data that reaches Google Health from Fitbit-compatible sources. "
                    "If you grant both Google Health nutrition permissions, Blacki "
                    "will automatically export future meal logs, edits, and "
                    "deletions from this private chat. The nutrition read permission "
                    "also lets Blacki verify records it created; it does not import "
                    "unrelated food logs. Older meals are not backfilled. "
                    "Read-only summaries remain available without nutrition "
                    "permissions. Existing connections must reconnect to add them. "
                    "Blacki does not receive Apple ID credentials or raw Apple Health "
                    "records. Authorize only if you want these selected categories."
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
        """Ask for a final Telegram click before disconnecting health sync."""
        if not await self._health_command_ready(message):
            return
        await self.api.send_message(
            chat_id=message.chat.id,
            text=(
                "Disconnect Google Health and cancel future meal sync for this "
                "chat? Blacki will remove its stored health credentials and "
                "summaries, but keep local calorie logs. Blacki will not delete "
                "records already sent to Google Health; requests already submitted "
                "may still finish."
            ),
            message_thread_id=message.message_thread_id,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="Disconnect and cancel sync",
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
                "Google Health was disconnected. Pending meal sync was cancelled, "
                "local calorie logs remain, and Blacki did not delete records already "
                "sent to Google Health. Requests already submitted may still finish."
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
            "/health_summary to read the latest stored records. If both nutrition "
            "permissions were granted, future private meal logs, edits, and "
            "deletions will sync automatically; older meals are not backfilled."
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
                "/connect_health - Connect Google Health and optional meal sync\n"
                "/health_summary - Show the latest health summary\n"
                "/health_refresh - Refresh health data\n"
                "/disconnect_health - Disconnect and cancel meal sync"
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
                "• /connect_health \\- Connect Google Health and optional meal sync\n"
                "• /health_summary \\- Show the latest health summary\n"
                "• /health_refresh \\- Refresh health data\n"
                "• /disconnect_health \\- Disconnect and cancel meal sync\n"
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
        file_unique_id: str | None = None,
        file_size: int | None = None,
        mime_type: str | None = None,
        media_kind: str = "document",
        sender_user_id: int | None = None,
        chat_type: ChatType | None = None,
    ) -> None:
        """Handle incoming file uploads, save to sandbox, and message agent."""
        from blacki.sandbox.manager import get_sandbox_manager
        from blacki.user_files import user_files_enabled

        if not user_files_enabled() and not get_sandbox_manager().config.enabled:
            await self.api.send_message(
                chat_id=chat_id,
                text="❌ Sandbox is not enabled, so the file cannot be processed.",
                message_thread_id=message_thread_id,
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
            sender_user_id=sender_user_id,
        )

        try:
            if file_size is not None and file_size > _MAX_TELEGRAM_FILE_BYTES:
                raise ValueError("Telegram attachment exceeds the 20 MB limit")
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
            if not file_bytes:
                raise ValueError("Telegram returned an empty attachment")
            if len(file_bytes) > _MAX_TELEGRAM_FILE_BYTES:
                raise ValueError("Telegram attachment exceeds the 20 MB limit")

            ingest, sandbox_path, sandbox_error = await self._shield_attachment_ingest(
                state=state,
                owner_id=str(sender_user_id) if sender_user_id is not None else None,
                display_name=file_name,
                media_kind=media_kind,
                mime_type=mime_type,
                telegram_file_unique_id=file_unique_id,
                data=file_bytes,
            )
            if ingest.warning:
                await self._send_storage_warning(
                    chat_id, message_thread_id, ingest.warning
                )
            if sandbox_path is None:
                if ingest.stored_file is not None:
                    await self.api.send_message(
                        chat_id=chat_id,
                        text=(
                            "✅ The attachment was saved in durable storage, but the "
                            "sandbox is unavailable. Ask me to restore it later."
                        ),
                        message_thread_id=message_thread_id,
                    )
                    return
                raise RuntimeError(sandbox_error or "Sandbox is unavailable")

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
        sender_user_id: int | None = None,
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
                sender_user_id=sender_user_id,
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
        sender_user_id: int | None = None,
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
        if sender_user_id is not None:
            session_state["temp:telegram_sender_user_id"] = str(sender_user_id)
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
