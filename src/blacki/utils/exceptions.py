"""Custom exceptions with actionable error messages.

This module provides exception classes that include actionable guidance
for developers, reducing the time to resolve configuration and runtime errors.
"""


class ConfigurationError(Exception):
    """Configuration error with actionable guidance.

    Attributes:
        message: Human-readable error message with actionable guidance.
        docs_url: Optional URL to documentation for more details.
    """

    def __init__(self, message: str, docs_url: str | None = None) -> None:
        self.docs_url = docs_url
        super().__init__(message)


class TelegramAuthError(ConfigurationError):
    """Telegram authentication failed.

    Raised when the Telegram bot token is invalid or missing.
    """

    def __init__(self, detail: str = "") -> None:
        message = (
            f"❌ Telegram authentication failed: {detail}\n"
            f"   Get a new token from @BotFather on Telegram.\n"
            f"   Steps:\n"
            f"   1. Open Telegram and search for @BotFather\n"
            f"   2. Send /newbot and follow the instructions\n"
            f"   3. Copy the token to TELEGRAM_BOT_TOKEN in your .env file"
        )
        super().__init__(
            message,
            docs_url="https://github.com/QueryPlanner/blacki/wiki/Errors#telegram",
        )


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing.

    Raised when neither OPENROUTER_API_KEY nor GOOGLE_API_KEY is set.
    """

    def __init__(self) -> None:
        message = (
            "❌ No LLM API key found.\n"
            "   You must set at least one of:\n"
            "   - OPENROUTER_API_KEY (recommended): Get one at https://openrouter.ai/keys\n"
            "   - GOOGLE_API_KEY: Get one at https://aistudio.google.com/apikey\n"
            "   Add your key to the .env file."
        )
        super().__init__(
            message,
            docs_url="https://github.com/QueryPlanner/blacki/wiki/Errors#api-keys",
        )


class InvalidTokenFormatError(ConfigurationError):
    """Token format is invalid.

    Raised when a token doesn't match the expected format.
    """

    def __init__(self, token_name: str, expected_format: str) -> None:
        message = (
            f"❌ Invalid {token_name} format.\n"
            f"   Expected format: {expected_format}\n"
            f"   Check your .env file and ensure the token is correct."
        )
        super().__init__(message)
