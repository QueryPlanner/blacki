"""Errors raised by the Gmail connector."""

from __future__ import annotations

from typing import Any


class GmailError(RuntimeError):
    """Base class for safe Gmail connector errors."""


class GmailConfigurationError(GmailError, ValueError):
    """Raised when the Gmail connector configuration is incomplete."""


class GmailCredentialError(GmailError):
    """Raised when a user has no usable Gmail credentials."""


class GmailAccessDeniedError(GmailCredentialError):
    """Raised when a non-private or non-Telegram caller requests Gmail."""


class GmailAlreadyConnectedError(GmailCredentialError):
    """Raised when a user must disconnect before connecting another account."""


class GmailAuthorizationRequiredError(GmailCredentialError):
    """Raised when Google requires the user to authorize Gmail again."""


class GmailMissingScopeError(GmailCredentialError):
    """Raised when a stored connection lacks the Gmail scope it needs."""


class GmailRevocationError(GmailCredentialError):
    """Raised when Google could not revoke a stored Gmail credential."""


class GmailInputError(GmailError, ValueError):
    """Raised when a Gmail tool argument is invalid or too large."""


class GmailDraftChangedError(GmailError):
    """Raised when a draft differs from the values the user confirmed."""


class GmailMalformedResponseError(GmailError):
    """Raised when Google returns a response outside the expected shape."""


class GmailApiError(GmailError):
    """Safe representation of a Gmail HTTP or provider error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after_seconds = retry_after_seconds


class GmailAuthError(GmailApiError):
    """Raised after an authenticated Gmail request still returns 401."""


class GmailRateLimitError(GmailApiError):
    """Raised when Google asks the caller to slow down."""


class GmailTransportError(GmailApiError):
    """Raised when the provider cannot be reached."""


def safe_provider_error_code(value: Any) -> str | None:
    """Keep provider error codes bounded and free of response payloads."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 80 or not cleaned.isascii():
        return None
    if not all(char.isalnum() or char in {"_", "-", "."} for char in cleaned):
        return None
    return cleaned
