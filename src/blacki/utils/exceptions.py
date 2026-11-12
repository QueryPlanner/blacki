"""Custom exceptions with actionable error messages.

This module provides exception classes that include actionable guidance
for developers, reducing the time to resolve configuration and runtime errors.
"""

import re
from typing import ClassVar


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

    Raised when neither OPENAI_API_KEY nor GOOGLE_API_KEY is set.
    """

    def __init__(self) -> None:
        message = (
            "❌ No LLM API key found.\n"
            "   You must set at least one of:\n"
            "   - OPENAI_API_KEY (recommended): Get one at https://platform.openai.com/api-keys\n"
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


_SAFE_TOKEN = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def _safe_token(value: object) -> str | None:
    """Return a provider error code or request id that is safe to show.

    Only short tokens of letters, digits and ``._:-`` pass. Anything else is
    dropped, so free text from a provider response never reaches a message.
    """

    if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value):
        return value
    return None


class ProviderError(Exception):
    """A call to a model provider failed.

    Attributes:
        provider: Provider name, for example ``"openai"``.
        status: HTTP status code, or ``None`` when no response arrived.
        code: The provider's error code, when it sent a safe one.
        request_id: The provider's request id, when it sent a safe one.

    The message is built only from these fields. It never contains an API
    key, prompt text, or response text.
    """

    summary: ClassVar[str] = "request failed"

    def __init__(
        self,
        *,
        provider: str = "openai",
        status: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.status = status
        self.code = _safe_token(code)
        self.request_id = _safe_token(request_id)
        super().__init__(self._describe())

    def _describe(self) -> str:
        details: list[str] = []
        if self.status is not None:
            details.append(f"HTTP {self.status}")
        if self.code is not None:
            details.append(f"code {self.code}")
        if self.request_id is not None:
            details.append(f"request {self.request_id}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{self.provider} {self.summary}{suffix}."


class ProviderAuthenticationError(ProviderError):
    """The provider rejected the API key (HTTP 401)."""

    summary = "authentication failed"


class ProviderPermissionError(ProviderError):
    """The key lacks permission for the request (HTTP 403)."""

    summary = "permission denied"


class ProviderRateLimitError(ProviderError):
    """The provider is rate limiting the key (HTTP 429)."""

    summary = "rate limit exceeded"


class ProviderRequestError(ProviderError):
    """The provider rejected the request as invalid (other HTTP 4xx)."""

    summary = "rejected the request"


class ProviderServerError(ProviderError):
    """The provider failed or could not be reached (HTTP 5xx or network)."""

    summary = "server or network error"
