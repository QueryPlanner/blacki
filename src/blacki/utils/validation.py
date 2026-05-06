"""Startup validation for configuration.

This module provides validation functions that run before the server starts,
ensuring configuration is correct and providing actionable error messages.
"""

import os
import re

from .exceptions import (
    InvalidTokenFormatError,
    MissingAPIKeyError,
    TelegramAuthError,
)


def validate_telegram_token(token: str | None) -> None:
    """Validate Telegram bot token format.

    Args:
        token: The Telegram bot token to validate.

    Raises:
        TelegramAuthError: If token is missing or empty.
        InvalidTokenFormatError: If token format is invalid.
    """
    if not token:
        raise TelegramAuthError("TELEGRAM_BOT_TOKEN is not set")

    pattern = r"^\d+:[A-Za-z0-9_-]+$"
    if not re.match(pattern, token):
        raise InvalidTokenFormatError(
            "TELEGRAM_BOT_TOKEN", "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        )


def validate_api_keys() -> None:
    """Validate that at least one LLM API key is configured.

    Raises:
        MissingAPIKeyError: If no API key is configured.
    """
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))

    if not has_openrouter and not has_google:
        raise MissingAPIKeyError()


def validate_configuration(
    telegram_enabled: bool, telegram_token: str | None
) -> list[str]:
    """Validate all configuration and return warnings.

    This function validates the configuration and returns a list of warnings
    for non-critical issues. Critical issues raise exceptions.

    Args:
        telegram_enabled: Whether Telegram bot is enabled.
        telegram_token: The Telegram bot token (if configured).

    Returns:
        List of warning messages for non-critical issues.

    Raises:
        ConfigurationError: If critical configuration is invalid.
    """
    warnings: list[str] = []

    validate_api_keys()

    if telegram_enabled:
        if telegram_token:
            try:
                validate_telegram_token(telegram_token)
            except (TelegramAuthError, InvalidTokenFormatError):
                raise
        else:
            raise TelegramAuthError("TELEGRAM_BOT_TOKEN is not set")

    root_agent_model = os.getenv("ROOT_AGENT_MODEL")
    if not root_agent_model:
        warnings.append(
            "⚠️  ROOT_AGENT_MODEL is not set. Using default model.\n"
            "   Set ROOT_AGENT_MODEL in .env for explicit model selection.\n"
            "   Example: ROOT_AGENT_MODEL=openrouter/google/gemini-2.0-flash-001"
        )

    return warnings
