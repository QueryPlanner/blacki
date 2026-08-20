"""Telegram bot configuration module.

This module provides Pydantic models for Telegram bot configuration
and environment variable validation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TelegramConfig(BaseModel):
    """Configuration for Telegram bot integration.

    Attributes:
        telegram_enabled: Whether Telegram bot integration is enabled.
        telegram_bot_token: The bot token from @BotFather.
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

    telegram_access_code: str | None = Field(
        default=None,
        alias="TELEGRAM_ACCESS_CODE",
        description="Shared access code required for new private Telegram users",
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

    @property
    def access_control_enabled(self) -> bool:
        """Return whether new-user access control is configured."""
        return bool(self.telegram_access_code)
