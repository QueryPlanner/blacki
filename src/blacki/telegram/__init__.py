"""Telegram bot configuration module.

This module provides Pydantic models for Telegram bot configuration
and environment variable validation.
"""

from pydantic import BaseModel, ConfigDict, Field


class TelegramConfig(BaseModel):
    """Configuration for Telegram bot integration.

    Attributes:
        telegram_enabled: Whether Telegram bot integration is enabled.
        telegram_bot_token: The bot token from @BotFather.
        telegram_tool_notifications: When True, send a chat message before each
            tool call for sessions that include Telegram chat identifiers in state.
    """

    telegram_enabled: bool = Field(
        default=False,
        alias="TELEGRAM_ENABLED",
        description="Whether Telegram bot integration is enabled",
    )

    telegram_bot_token: str | None = Field(
        default=None,
        alias="TELEGRAM_BOT_TOKEN",
        description="Telegram bot token obtained from @BotFather",
    )

    telegram_tool_notifications: bool = Field(
        default=False,
        alias="TELEGRAM_TOOL_NOTIFICATIONS",
        description=(
            "Send Markdown tool-use notices to Telegram for turns that carry "
            "telegram_chat_id in session state"
        ),
    )

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    def is_configured(self) -> bool:
        """Check if Telegram bot is properly configured.

        Returns:
            True if enabled and has a bot token, False otherwise.
        """
        return self.telegram_enabled and self.telegram_bot_token is not None

    def tool_notifications_active(self) -> bool:
        """Whether tool notifications should be sent to Telegram."""
        return self.is_configured() and self.telegram_tool_notifications
