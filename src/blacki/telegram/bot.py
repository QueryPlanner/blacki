"""Telegram bot client backed by the shared ADK runtime.

This module implements a Telegram bot using direct HTTP calls to the
Telegram Bot API (version 9.5+) with streaming support via sendMessageDraft.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from blacki.adk_runtime import AdkRuntime, SessionLocator, StreamChunk

from . import TelegramConfig
from .api import TelegramApiClient, TelegramApiError
from .streaming import DraftManager, split_long_message
from .types import BotCommand, Message, ParseMode, Update

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
POLLING_TIMEOUT = 30


@dataclass(slots=True, frozen=True)
class TelegramSessionIdentity:
    """Stable Telegram identifiers used to resolve ADK sessions."""

    conversation_key: str
    user_id: str
    session_id_prefix: str


class TelegramBot:
    """Telegram bot client that streams ADK responses via sendMessageDraft."""

    def __init__(
        self,
        config: TelegramConfig,
        runtime: AdkRuntime,
    ) -> None:
        """Initialize the Telegram bot."""
        self.config = config
        self.runtime = runtime
        self._api: TelegramApiClient | None = None
        self._draft_manager: DraftManager | None = None
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

    @property
    def draft_manager(self) -> DraftManager:
        """Get or create the draft manager."""
        if self._draft_manager is None:
            self._draft_manager = DraftManager(self.api)
        return self._draft_manager

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
        """Handle a regular text message with streaming response."""
        session_identity = self._build_session_identity(
            chat_id=str(chat_id),
            message_thread_id=message_thread_id,
        )

        logger.info("Received message from chat %s: %s...", chat_id, user_message[:50])

        try:
            await self.api.send_chat_action(chat_id=chat_id, action="typing")

            async def get_chunks() -> AsyncIterator[StreamChunk]:
                async for chunk in self.runtime.run_user_turn_streaming(
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
                ):
                    yield chunk

            final_thoughts, final_content = await self.draft_manager.stream_response(
                chat_id=chat_id,
                chunks=get_chunks(),
                message_thread_id=message_thread_id,
            )

            await self._send_final_messages(
                chat_id=chat_id,
                thoughts=final_thoughts,
                content=final_content,
                message_thread_id=message_thread_id,
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

    async def _send_final_messages(
        self,
        chat_id: int,
        thoughts: str,
        content: str,
        *,
        message_thread_id: int | None = None,
    ) -> None:
        """Send final messages after streaming completes.

        Replaces the drafts with properly formatted final messages.
        """
        if thoughts:
            await self._send_thoughts(
                chat_id=chat_id,
                thoughts=thoughts,
                message_thread_id=message_thread_id,
            )

        if content:
            await self._send_content(
                chat_id=chat_id,
                content=content,
                message_thread_id=message_thread_id,
            )
        elif not thoughts:
            text = "I apologize, but I couldn't generate a response\\."
            await self.api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=message_thread_id,
            )

    async def _send_thoughts(
        self,
        chat_id: int,
        thoughts: str,
        *,
        message_thread_id: int | None = None,
    ) -> None:
        """Send thinking content as italicized message."""
        formatted = f"_Thinking: {escape_markdown(thoughts)}_"
        for chunk in split_long_message(formatted):
            try:
                await self.api.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=message_thread_id,
                )
            except TelegramApiError:
                logger.exception("Failed to send thoughts message")

    async def _send_content(
        self,
        chat_id: int,
        content: str,
        *,
        message_thread_id: int | None = None,
    ) -> None:
        """Send main response content with Markdown formatting."""
        formatted = format_for_telegram(content)
        for chunk in split_long_message(formatted):
            try:
                await self.api.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=message_thread_id,
                )
            except TelegramApiError:
                logger.exception("Failed to send content message")

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


MARKDOWN_SPECIAL_CHARS = frozenset("_*[]()~>#+-=|{}.!\\")


def escape_markdown(text: str) -> str:
    """Escape special Markdown characters for Telegram MarkdownV2.

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


def format_for_telegram(text: str) -> str:
    """Format text for Telegram MarkdownV2, preserving bold formatting.

    Handles **bold** markdown by:
    1. Identifying bold sections
    2. Escaping special chars in all content
    3. Converting ** to * for Telegram bold syntax
    """
    result: list[str] = []
    i = 0
    in_code_block = False
    in_inline_code = False

    while i < len(text):
        if i + 2 <= len(text) and text[i : i + 3] == "```":
            in_code_block = not in_code_block
            result.append("```")
            i += 3
            continue

        if text[i] == "`" and not in_code_block:
            in_inline_code = not in_inline_code
            result.append("`")
            i += 1
            continue

        if not in_code_block and not in_inline_code:
            if i + 1 < len(text) and text[i : i + 2] == "**":
                j = i + 2
                inner_in_code_block = False
                inner_in_inline_code = False
                while j + 1 < len(text):
                    if j + 2 <= len(text) and text[j : j + 3] == "```":
                        inner_in_code_block = not inner_in_code_block
                        j += 3
                        continue
                    if text[j] == "`" and not inner_in_code_block:
                        inner_in_inline_code = not inner_in_inline_code
                        j += 1
                        continue
                    if (
                        not inner_in_code_block
                        and not inner_in_inline_code
                        and j + 1 < len(text)
                        and text[j : j + 2] == "**"
                    ):
                        break
                    j += 1

                if j + 1 < len(text) and text[j : j + 2] == "**":
                    bold_content = text[i + 2 : j]
                    escaped_content = _escape_text_only(bold_content)
                    result.append(f"*{escaped_content}*")
                    i = j + 2
                    continue

            if text[i] in MARKDOWN_SPECIAL_CHARS:
                result.append("\\")
            result.append(text[i])
        else:
            result.append(text[i])

        i += 1

    return "".join(result)


def _escape_text_only(text: str) -> str:
    """Escape special chars without code block handling (for internal use)."""
    result: list[str] = []
    in_code_block = False
    in_inline_code = False
    i = 0

    while i < len(text):
        if i + 2 <= len(text) and text[i : i + 3] == "```":
            in_code_block = not in_code_block
            result.append("```")
            i += 3
            continue

        if text[i] == "`" and not in_code_block:
            in_inline_code = not in_inline_code
            result.append("`")
            i += 1
            continue

        if not in_code_block and not in_inline_code:
            if text[i] in MARKDOWN_SPECIAL_CHARS:
                result.append("\\")
            result.append(text[i])
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
