"""Telegram bot client with polling and memory integration.

This module provides the TelegramBot class that handles:
- Bot initialization with token
- Polling for updates
- Message routing through the ADK agent
- Memory context injection via mem0
"""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from blacki.mem0 import get_mem0_manager, is_mem0_enabled
from blacki.prompt import return_instruction_root

from . import TelegramConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot client with polling and memory integration.

    This class manages the Telegram bot lifecycle and integrates with
    the ADK agent for message processing with persistent memory.

    Attributes:
        config: Telegram configuration from environment.
        app: Python-telegram-bot Application instance.
        model: The LLM model to use for responses.
    """

    def __init__(
        self,
        config: TelegramConfig,
        model: Any,
    ) -> None:
        """Initialize the Telegram bot.

        Args:
            config: Telegram configuration with bot token.
            model: The LLM model (LiteLlm instance or string) for responses.
        """
        self.config = config
        self.model = model
        self._app: Application | None = None
        self._session_ids: dict[str, str] = {}

    def _ensure_app(self) -> Application:
        """Get or create the telegram Application.

        Returns:
            The telegram Application instance.
        """
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
        """Get the telegram Application (calls _ensure_app for initialization).

        Returns:
            The telegram Application instance.
        """
        return self._ensure_app()

    def _get_session_user_id(self, chat_id: str) -> str:
        """Get the mem0 user_id for this chat's current session.

        Uses a compound user_id (chat_id_session_id) to isolate sessions.
        A new session_id is created on first access or after /reset.

        Args:
            chat_id: The Telegram chat ID.

        Returns:
            The compound user_id for mem0 operations.
        """
        if chat_id not in self._session_ids:
            self._session_ids[chat_id] = uuid.uuid4().hex[:8]
        return f"{chat_id}_{self._session_ids[chat_id]}"

    def _reset_session(self, chat_id: str) -> str:
        """Reset the session for this chat.

        Args:
            chat_id: The Telegram chat ID.

        Returns:
            The new session_id.
        """
        new_session_id = uuid.uuid4().hex[:8]
        self._session_ids[chat_id] = new_session_id
        logger.info(f"Session reset for chat {chat_id}: new session_id={new_session_id}")
        return new_session_id

    async def _send_typing_periodically(self, chat_id: int) -> None:
        """Send typing action every 4 seconds until cancelled.

        Args:
            chat_id: The Telegram chat ID to send typing indicator to.
        """
        while True:
            try:
                await self.app.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to send typing indicator")
                raise

    def _start_typing_indicator(self, chat_id: int) -> asyncio.Task[None]:
        """Start the typing indicator background task.

        Args:
            chat_id: The Telegram chat ID to send typing indicator to.

        Returns:
            The asyncio Task running the typing indicator.
        """
        return asyncio.create_task(self._send_typing_periodically(chat_id))

    async def _stop_typing_indicator(self, task: asyncio.Task[None]) -> None:
        """Stop the typing indicator task.

        Args:
            task: The typing indicator task to stop.
        """
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _setup_handlers(self, app: Application) -> None:
        """Set up command and message handlers.

        Args:
            app: The Application instance to add handlers to.
        """
        # Command handlers
        app.add_handler(CommandHandler("start", self._start_command))
        app.add_handler(CommandHandler("help", self._help_command))
        app.add_handler(CommandHandler("reset", self._reset_command))

        # Message handler for text messages
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

    async def _start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command.

        Args:
            update: The Telegram update.
            context: The callback context.
        """
        if not update.effective_chat or not update.message:
            return

        welcome_message = (
            "👋 Hello! I'm blacki, your AI assistant.\n\n"
            "I can help you with questions and tasks. I remember our "
            "conversations, so feel free to continue where we left off.\n\n"
            "Commands:\n"
            "/help - Show available commands\n"
            "/reset - Start a fresh conversation session"
        )
        await update.message.reply_text(welcome_message)

    async def _help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command.

        Args:
            update: The Telegram update.
            context: The callback context.
        """
        if not update.effective_chat or not update.message:
            return

        help_message = (
            "🤖 **blacki - AI Assistant**\n\n"
            "I'm an AI assistant powered by the Google ADK framework.\n\n"
            "**Commands:**\n"
            "• /start - Start a conversation\n"
            "• /help - Show this help message\n"
            "• /reset - Start a fresh conversation session\n\n"
            "**Features:**\n"
            "• I remember our conversations (per chat)\n"
            "• Ask me anything - questions, coding help, creative tasks\n\n"
            "Just send me a message to get started!"
        )
        await update.message.reply_text(help_message)

    async def _reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /reset command to start a fresh session.

        Args:
            update: The Telegram update.
            context: The callback context.
        """
        if not update.effective_chat or not update.message:
            return

        chat_id = str(update.effective_chat.id)
        new_session_id = self._reset_session(chat_id)

        await update.message.reply_text(
            "🔄 Session reset. Starting fresh conversation."
        )
        logger.info(f"Session reset for chat {chat_id}: {new_session_id}")

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages.

        This is the main message processing pipeline:
        1. Start typing indicator
        2. Retrieve memories for context
        3. Process through LLM
        4. Save conversation to memory
        5. Send response
        6. Stop typing indicator

        Args:
            update: The Telegram update.
            context: The callback context.
        """
        if not update.effective_chat or not update.message or not update.message.text:
            return

        chat_id = str(update.effective_chat.id)
        user_message = update.message.text

        logger.info(f"Received message from chat {chat_id}: {user_message[:50]}...")

        # Start typing indicator
        typing_task = self._start_typing_indicator(update.effective_chat.id)

        try:
            # Build context with memories
            memory_context = await self._get_memory_context(chat_id, user_message)

            # Get instruction for the agent
            instruction = return_instruction_root()

            # Build the full prompt
            full_prompt = self._build_prompt(instruction, memory_context, user_message)

            # Generate response using the LLM
            response = await self._generate_response(full_prompt)

            # Save conversation to memory
            await self._save_to_memory(chat_id, user_message, response)

            # Send response to Telegram
            await update.message.reply_text(response)

            logger.info(f"Sent response to chat {chat_id}")

        except Exception:
            logger.exception(f"Error processing message for chat {chat_id}")
            await update.message.reply_text(
                "❌ Sorry, I encountered an error processing your message. "
                "Please try again."
            )
        finally:
            # Always stop typing indicator
            await self._stop_typing_indicator(typing_task)

    async def _get_memory_context(self, chat_id: str, query: str) -> str:
        """Retrieve memory context for the chat's current session.

        Args:
            chat_id: The Telegram chat ID.
            query: The user's message.

        Returns:
            A formatted string with relevant memories, or empty string.
        """
        if not is_mem0_enabled():
            return ""

        try:
            # Use session-scoped user_id for memory isolation
            session_user_id = self._get_session_user_id(chat_id)
            manager = get_mem0_manager()
            result = manager.search_memory(
                query=query,
                user_id=session_user_id,
                limit=5,
            )

            memories = result.get("memories", [])
            if not memories:
                return ""

            memory_texts = []
            for mem in memories:
                if isinstance(mem, dict) and "memory" in mem:
                    memory_texts.append(f"- {mem['memory']}")

            if memory_texts:
                return "Context from memory:\n" + "\n".join(memory_texts)

        except Exception:
            logger.exception(f"Failed to retrieve memories for chat {chat_id}")

        return ""

    def _build_prompt(
        self, instruction: str, memory_context: str, user_message: str
    ) -> str:
        """Build the full prompt for the LLM.

        Args:
            instruction: The agent's instruction.
            memory_context: Context from memory (may be empty).
            user_message: The user's message.

        Returns:
            The complete prompt string.
        """
        parts = [instruction]

        if memory_context:
            parts.append(f"\n\n{memory_context}")

        parts.append(f"\n\nUser: {user_message}")

        return "\n".join(parts)

    async def _generate_response(self, prompt: str) -> str:
        """Generate a response using the LLM.

        Args:
            prompt: The full prompt to send to the LLM.

        Returns:
            The generated response text.
        """
        try:
            # Import litellm for direct LLM calls
            import litellm

            # Determine model string
            model_str = self.model
            if hasattr(self.model, "model"):
                model_str = self.model.model

            # Make the completion call
            response = await litellm.acompletion(
                model=model_str,
                messages=[
                    {
                        "role": "system",
                        "content": "You are blacki, a helpful AI assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            # Extract the response text
            if response.choices and response.choices[0].message.content:
                return str(response.choices[0].message.content)

            return "I apologize, but I couldn't generate a response."

        except Exception:
            logger.exception("Failed to generate LLM response")
            return "I encountered an error while generating a response."

    async def _save_to_memory(
        self, chat_id: str, user_message: str, response: str
    ) -> None:
        """Save the conversation to memory for the current session.

        Args:
            chat_id: The Telegram chat ID.
            user_message: The user's message.
            response: The assistant's response.
        """
        if not is_mem0_enabled():
            return

        try:
            # Use session-scoped user_id for memory isolation
            session_user_id = self._get_session_user_id(chat_id)
            manager = get_mem0_manager()
            conversation = f"User: {user_message}\nAssistant: {response}"
            manager.save_memory(
                content=conversation,
                user_id=session_user_id,
            )
            logger.debug(f"Saved conversation to memory for chat {chat_id}")

        except Exception:
            logger.exception(f"Failed to save memory for chat {chat_id}")

    async def start_polling(self) -> None:
        """Start the bot polling loop.

        This runs in the background and processes incoming messages.
        """
        if not self.config.is_configured():
            logger.info("Telegram bot not configured, skipping start")
            return

        logger.info("Starting Telegram bot polling...")
        app = self._ensure_app()
        await app.initialize()

        # Register bot commands with Telegram
        await self._register_commands(app)

        await app.start()
        if app.updater:
            logger.info("Requesting Telegram polling start...")
            await app.updater.start_polling()
        logger.info("Telegram polling started successfully")

    async def _register_commands(self, app: Application) -> None:
        """Register bot commands with Telegram's command menu.

        Args:
            app: The Application instance.
        """
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
        if self._app is None:
            return

        logger.info("Stopping Telegram bot...")
        if self._app.updater:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram bot stopped")


def create_telegram_bot(config: TelegramConfig, model: Any) -> TelegramBot | None:
    """Create a Telegram bot instance if configured.

    Args:
        config: Telegram configuration from environment.
        model: The LLM model to use.

    Returns:
        A TelegramBot instance if configured, None otherwise.
    """
    if not config.is_configured():
        logger.info("Telegram bot not configured, skipping initialization")
        return None

    return TelegramBot(config, model)
