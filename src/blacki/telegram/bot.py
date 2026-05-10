"""Telegram bot client backed by the shared ADK runtime."""

import asyncio
import contextlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from blacki.adk_runtime import AdkRuntime, SessionLocator
from blacki.reminders.storage import Reminder

from . import TelegramConfig
from .api import TelegramApiClient, TelegramApiError
from .formatting import format_for_telegram
from .streaming import split_long_message
from .types import BotCommand, Message, ParseMode, Update

logger = logging.getLogger(__name__)

POLLING_TIMEOUT = 30
_MAX_CONSECUTIVE_ERRORS = 5
_FATAL_ERROR_CODES = {401, 403}
_TELEGRAM_USER_ID_PATTERN = re.compile(r"^telegram-chat-(-?\d+)(?:-thread-(\d+))?$")


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
                    allowed_updates=["message"],
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

        if message.document:
            file_id = message.document.file_id
            file_name = message.document.file_name or "document"
        elif message.photo:
            file_id = message.photo[-1].file_id
            file_name = "photo.jpg"
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

    async def _handle_command(self, message: Message, command: str) -> None:
        """Handle a command message."""
        chat_id = message.chat.id

        if command == "/start":
            await self._send_start_message(chat_id)
        elif command == "/help":
            await self._send_help_message(chat_id)
        elif command == "/reset":
            await self._handle_reset(chat_id, message.message_thread_id)

    async def _send_start_message(self, chat_id: int) -> None:
        """Send the start/welcome message."""
        text = (
            "👋 Hello! I'm blacki, your AI assistant\\.\n\n"
            "I run through the same ADK agent as the web interface, so our "
            "conversation history stays attached to this chat\\.\n\n"
            "Commands:\n"
            "/help \\- Show available commands\n"
            "/reset \\- Start a fresh conversation session"
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
            "• /reset \\- Start a fresh conversation session\n\n"
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
            await self.runtime.create_next_session(
                locator=SessionLocator(
                    user_id=session_identity.user_id,
                    session_id_prefix=session_identity.session_id_prefix,
                ),
                state=self._build_session_state(
                    chat_id=str(chat_id),
                    message_thread_id=message_thread_id,
                    conversation_key=session_identity.conversation_key,
                ),
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
                text="❌ Sandbox is not enabled. Cannot process file uploads\\.",
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

            final_response = await self.runtime.run_user_turn(
                locator=SessionLocator(
                    user_id=session_identity.user_id,
                    session_id_prefix=session_identity.session_id_prefix,
                ),
                message_text=user_message,
                state=state,
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

        logger.info("Received message from chat %s: %s...", chat_id, user_message[:50])

        try:
            await self.api.send_chat_action(
                chat_id=chat_id,
                action="typing",
                message_thread_id=message_thread_id,
            )

            final_response = await self.runtime.run_user_turn(
                locator=SessionLocator(
                    user_id=session_identity.user_id,
                    session_id_prefix=session_identity.session_id_prefix,
                ),
                message_text=user_message,
                state=self._build_session_state(
                    chat_id=str(chat_id),
                    message_thread_id=message_thread_id,
                    conversation_key=session_identity.conversation_key,
                ),
            )
            await self._send_final_response(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                response_text=final_response,
            )

            logger.info("Sent ADK response to chat %s", chat_id)
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

            final_response = await self.runtime.run_user_turn(
                locator=SessionLocator(
                    user_id=session_identity.user_id,
                    session_id_prefix=session_identity.session_id_prefix,
                ),
                message_text=f"[Scheduled Event] {reminder.message}",
                state=self._build_session_state(
                    chat_id=chat_id_str,
                    message_thread_id=message_thread_id,
                    conversation_key=session_identity.conversation_key,
                ),
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
            await self.api.send_message(
                chat_id=chat_id,
                text=message_chunk,
                parse_mode=ParseMode.MARKDOWN_V2,
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
