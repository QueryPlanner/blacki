"""Telegram bot configuration module.

This module provides Pydantic models for Telegram bot configuration
and environment variable validation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TelegramConfig(BaseModel):
    """Configuration for Telegram bot integration.

    Attributes:
        telegram_enabled: Whether Telegram bot integration is enabled.
        telegram_bot_token: The bot token from @BotFather.
        telegram_tool_notifications: Legacy flag: when True (and
            TELEGRAM_TOOL_PROGRESS_MODE is unset), enables 'messages' mode.
        telegram_tool_progress_mode: Tool progress notification mode:
            'off', 'messages', or 'live'.
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
            "Legacy flag: send Markdown tool-use notices for turns that carry "
            "telegram_chat_id in session state"
        ),
    )

    telegram_tool_progress_mode: str | None = Field(
        default=None,
        alias="TELEGRAM_TOOL_PROGRESS_MODE",
        description="Tool progress notification mode: 'off', 'messages', or 'live'",
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

    def tool_progress_mode(self) -> Literal["off", "messages", "live"]:
        """Return the resolved tool progress notification mode.

        Resolution order:
        1. If Telegram is not configured -> 'off'.
        2. If TELEGRAM_TOOL_PROGRESS_MODE is explicitly set -> that mode.
        3. If TELEGRAM_TOOL_NOTIFICATIONS is True -> 'messages' (back-compat).
        4. Otherwise -> 'off'.
        """
        if not self.is_configured():
            return "off"

        if self.telegram_tool_progress_mode is not None:
            mode = self.telegram_tool_progress_mode.strip().lower()
            if mode in ("off", "messages", "live"):
                return mode  # type: ignore[return-value]

        if self.telegram_tool_notifications:
            return "messages"

        return "off"

    def tool_notifications_active(self) -> bool:
        """Whether tool notifications should be sent to Telegram."""
        return self.tool_progress_mode() != "off"
