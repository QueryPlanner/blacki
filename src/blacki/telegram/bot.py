"""Telegram bot client backed by the shared ADK runtime."""

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from telegram import Message, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from blacki.adk_runtime import AdkRuntime, SessionLocator, TurnResponse

from . import TelegramConfig

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096


@dataclass(slots=True, frozen=True)
class TelegramSessionIdentity:
    """Stable Telegram identifiers used to resolve ADK sessions."""

    conversation_key: str
    user_id: str
    session_id_prefix: str


class TelegramBot:
    """Telegram bot client that sends each message through the ADK runner."""

    def __init__(
        self,
        config: TelegramConfig,
        runtime: AdkRuntime,
    ) -> None:
        """Initialize the Telegram bot."""
        self.config = config
        self.runtime = runtime
        self._app: Application | None = None

    def _ensure_app(self) -> Application:
        """Get or create the telegram Application."""
        if self._app is None:
            if not self.config.telegram_bot_token:
                msg = "TELEGRAM_BOT_TOKEN is required"
                raise ValueError(msg)
            self._app = (
                Application.builder().token(self.config.telegram_bot_token).build()
            )
            self._setup_handlers(self._app)
        return self._app

    @property
    def app(self) -> Application:
        """Get the telegram Application."""
        return self._ensure_app()

    async def _send_typing_periodically(self, chat_id: int) -> None:
        """Send typing action every 4 seconds until cancelled."""
        while True:
            try:
                await self.app.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed to send typing indicator", exc_info=True)
                return

    def _start_typing_indicator(self, chat_id: int) -> asyncio.Task[None]:
        """Start the typing indicator background task."""
        return asyncio.create_task(self._send_typing_periodically(chat_id))

    async def _stop_typing_indicator(self, task: asyncio.Task[None]) -> None:
        """Stop the typing indicator task."""
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _setup_handlers(self, app: Application) -> None:
        """Set up command and message handlers."""
        app.add_handler(CommandHandler("start", self._start_command))
        app.add_handler(CommandHandler("help", self._help_command))
        app.add_handler(CommandHandler("reset", self._reset_command))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

    async def _start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        if not update.effective_chat or not update.message:
            return

        welcome_message = (
            "👋 Hello! I'm blacki, your AI assistant.\n\n"
            "I run through the same ADK agent as the web interface, so our "
            "conversation history stays attached to this chat.\n\n"
            "Commands:\n"
            "/help - Show available commands\n"
            "/reset - Start a fresh conversation session"
        )
        await update.message.reply_text(welcome_message)

    async def _help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command."""
        if not update.effective_chat or not update.message:
            return

        help_message = (
            "🤖 **blacki - AI Assistant**\n\n"
            "I'm powered by the same Google ADK runtime used by the HTTP app.\n\n"
            "**Commands:**\n"
            "• /start - Start a conversation\n"
            "• /help - Show this help message\n"
            "• /reset - Start a fresh conversation session\n\n"
            "**Features:**\n"
            "• Conversation history is tied to this chat\n"
            "• Topic threads can keep separate sessions\n"
            "• Ask me anything - questions, coding help, creative tasks"
        )
        await update.message.reply_text(help_message)

    async def _reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /reset command by creating the next ADK session version."""
        if not update.effective_chat or not update.message:
            return

        chat_id = str(update.effective_chat.id)
        message_thread_id = update.message.message_thread_id
        session_identity = self._build_session_identity(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
        )

        try:
            await self.runtime.create_next_session(
                locator=SessionLocator(
                    user_id=session_identity.user_id,
                    session_id_prefix=session_identity.session_id_prefix,
                ),
                state=self._build_session_state(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    conversation_key=session_identity.conversation_key,
                ),
            )
        except Exception:
            logger.exception("Failed to reset Telegram session for chat %s", chat_id)
            await update.message.reply_text(
                "❌ Sorry, I couldn't reset the conversation right now. "
                "Please try again."
            )
            return

        await update.message.reply_text(
            "🔄 Session reset. Starting a fresh ADK conversation."
        )

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages by running one ADK turn."""
        if not update.effective_chat or not update.message or not update.message.text:
            return

        chat_id = str(update.effective_chat.id)
        message_thread_id = update.message.message_thread_id
        user_message = update.message.text
        session_identity = self._build_session_identity(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
        )

        logger.info("Received message from chat %s: %s...", chat_id, user_message[:50])

        typing_task = self._start_typing_indicator(update.effective_chat.id)

        try:
            response = await self.runtime.run_user_turn_with_thoughts(
                locator=SessionLocator(
                    user_id=session_identity.user_id,
                    session_id_prefix=session_identity.session_id_prefix,
                ),
                message_text=user_message,
                state=self._build_session_state(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    conversation_key=session_identity.conversation_key,
                ),
            )
            await self._send_response(update.message, response)
            logger.info("Sent ADK response to chat %s", chat_id)
        except Exception:
            logger.exception("Error processing message for chat %s", chat_id)
            await update.message.reply_text(
                "❌ Sorry, I encountered an error processing your message. "
                "Please try again."
            )
        finally:
            await self._stop_typing_indicator(typing_task)

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
        session_state = {
            "user_id": f"telegram-{conversation_key}",
            "telegram_chat_id": chat_id,
            "telegram_conversation_key": conversation_key,
        }
        if message_thread_id is not None:
            session_state["telegram_thread_id"] = str(message_thread_id)
        return session_state

    async def _send_response(self, message: Message, response: TurnResponse) -> None:
        """Send thoughts and content as separate Telegram messages."""
        if response.thoughts:
            await self._send_thoughts(message, response.thoughts)
        await self._send_content(message, response.content)

    async def _send_thoughts(self, message: Message, thoughts: str) -> None:
        """Send thinking content as italicized message."""
        formatted = f"_Thinking: {escape_markdown(thoughts)}_"
        for chunk in self._split_response_text(formatted):
            await message.reply_text(chunk, parse_mode="Markdown")

    async def _send_content(self, message: Message, content: str) -> None:
        """Send main response content with Markdown formatting."""
        if not content:
            await message.reply_text("I apologize, but I couldn't generate a response.")
            return
        formatted = escape_markdown(convert_bold_to_telegram(content))
        for chunk in self._split_response_text(formatted):
            await message.reply_text(chunk, parse_mode="Markdown")

    def _split_response_text(self, response_text: str) -> list[str]:
        """Split long responses into Telegram-safe chunks."""
        normalized_response = response_text.strip()
        if not normalized_response:
            return [""]

        if len(normalized_response) <= TELEGRAM_MESSAGE_LIMIT:
            return [normalized_response]

        chunks: list[str] = []
        remaining_text = normalized_response

        while remaining_text:
            if len(remaining_text) <= TELEGRAM_MESSAGE_LIMIT:
                chunks.append(remaining_text)
                break

            split_index = self._find_chunk_boundary(remaining_text)
            chunk = remaining_text[:split_index].rstrip()
            chunks.append(chunk)
            remaining_text = remaining_text[len(chunk) :].lstrip()

        return chunks

    def _find_chunk_boundary(self, response_text: str) -> int:
        """Find the most readable split point within Telegram's message limit.

        Note: Splitting on whitespace boundaries intentionally normalizes whitespace
        between chunks. The separator character is excluded from the chunk, which is
        acceptable for Telegram where whitespace between messages is rarely meaningful.
        """
        for separator in ("\n\n", "\n", " "):
            split_index = response_text.rfind(separator, 0, TELEGRAM_MESSAGE_LIMIT + 1)
            if split_index > 0:
                return split_index

        return TELEGRAM_MESSAGE_LIMIT

    async def start_polling(self) -> None:
        """Start the bot polling loop."""
        if not self.config.is_configured():
            logger.info("Telegram bot not configured, skipping start")
            return

        logger.info("Starting Telegram bot polling...")
        app = self._ensure_app()
        await app.initialize()
        await self._register_commands(app)
        await app.start()
        if app.updater:
            logger.info("Requesting Telegram polling start...")
            await app.updater.start_polling()
        logger.info("Telegram polling started successfully")

    async def _register_commands(self, app: Application) -> None:
        """Register bot commands with Telegram's command menu."""
        from telegram import BotCommand

        commands = [
            BotCommand("start", "Start a conversation"),
            BotCommand("help", "Show available commands"),
            BotCommand("reset", "Start a fresh conversation session"),
        ]
        try:
            await app.bot.set_my_commands(commands)
            logger.info("Registered Telegram bot commands")
        except Exception:
            logger.exception("Failed to register bot commands")

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        try:
            if self._app is not None:
                logger.info("Stopping Telegram bot...")
                if self._app.updater:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
                logger.info("Telegram bot stopped")
        finally:
            await self.runtime.close()


MARKDOWN_SPECIAL_CHARS = frozenset("_`")


def escape_markdown(text: str) -> str:
    """Escape special Markdown characters for Telegram Markdown (v1).

    Does NOT escape inside code blocks or inline code - those are preserved.
    """
    result: list[str] = []
    in_code_block = False
    in_inline_code = False
    i = 0

    while i < len(text):
        char = text[i]

        if i + 2 <= len(text) and text[i : i + 3] == "```":
            in_code_block = not in_code_block
            result.append("```")
            i += 3
            continue

        if char == "`" and not in_code_block:
            in_inline_code = not in_inline_code
            result.append(char)
            i += 1
            continue

        if not in_code_block and not in_inline_code:
            if char in MARKDOWN_SPECIAL_CHARS:
                result.append("\\")
                result.append(char)
            else:
                result.append(char)
        else:
            result.append(char)

        i += 1

    return "".join(result)


def convert_bold_to_telegram(text: str) -> str:
    """Convert **bold** markdown to *bold* for Telegram Markdown.

    Telegram Markdown uses single asterisks for bold, while standard Markdown
    uses double asterisks. This function converts standard Markdown bold syntax
    to Telegram's expected format.
    """
    result: list[str] = []
    i = 0

    while i < len(text):
        if i + 1 < len(text) and text[i : i + 2] == "**":
            result.append("*")
            i += 2
        else:
            result.append(text[i])
            i += 1

    return "".join(result)


def create_telegram_bot(
    config: TelegramConfig,
    runtime: AdkRuntime,
) -> TelegramBot | None:
    """Create a Telegram bot instance if configured."""
    if not config.is_configured():
        logger.info("Telegram bot not configured, skipping initialization")
        return None

    return TelegramBot(config, runtime)
