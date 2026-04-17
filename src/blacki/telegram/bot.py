"""Telegram bot client backed by the shared ADK runtime."""

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from blacki.adk_runtime import AdkRuntime, SessionLocator

from . import TelegramConfig
from .api import TelegramApiClient, TelegramApiError
from .formatting import format_for_telegram
from .streaming import split_long_message
from .types import BotCommand, Message, ParseMode, Update

logger = logging.getLogger(__name__)

POLLING_TIMEOUT = 30


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

        while self._running:
            try:
                updates = await self.api.get_updates(
                    offset=offset,
                    timeout=POLLING_TIMEOUT,
                    allowed_updates=["message"],
                )

                for update in updates:
                    offset = update.update_id + 1
                    await self._handle_update(update)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in polling loop")
                await asyncio.sleep(5)

    async def _handle_update(self, update: Update) -> None:
        """Handle an incoming update."""
        if update.message is None:
            return

        message = update.message

        if message.text is None:
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
            await self.api.send_chat_action(chat_id=chat_id, action="typing")

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
