"""Configuration and token protection for Google Health."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

from cryptography.fernet import Fernet, InvalidToken

GOOGLE_HEALTH_AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_HEALTH_TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
GOOGLE_HEALTH_REVOCATION_URL = "https://oauth2.googleapis.com/revoke"
GOOGLE_HEALTH_API_BASE_URL = "https://health.googleapis.com"
GOOGLE_HEALTH_DEFAULT_REDIRECT_URI = (
    "http://127.0.0.1:8080/integrations/google-health/callback"
)
GOOGLE_HEALTH_SCOPES = (
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
)

_TELEGRAM_HEALTH_USER_PATTERN = re.compile(
    r"^telegram-chat-(?P<chat_id>-?\d+)(?:-thread-\d+)?$"
)
_TELEGRAM_HEALTH_CHAT_PATTERN = re.compile(r"^telegram-chat-(?P<chat_id>-?\d+)$")


class GoogleHealthConfigurationError(ValueError):
    """Raised when an incomplete or invalid Google Health configuration exists."""


class TokenEncryptionError(ValueError):
    """Raised when an encrypted refresh token cannot be protected or opened."""


class TokenCipher:
    """Encrypt and decrypt refresh tokens with a Fernet key from a secret store."""

    def __init__(self, key: str) -> None:
        try:
            self._fernet = Fernet(key.encode("ascii"))
        except (ValueError, TypeError, UnicodeEncodeError) as exc:
            raise TokenEncryptionError(
                "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY is invalid"
            ) from exc

    def encrypt(self, value: str) -> str:
        """Return an encrypted representation of a refresh token."""
        if not value:
            raise TokenEncryptionError("Cannot encrypt an empty refresh token")
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str) -> str:
        """Return a refresh token from its encrypted representation."""
        try:
            return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError, UnicodeEncodeError) as exc:
            raise TokenEncryptionError("Stored Google Health token is invalid") from exc


@dataclass(frozen=True, slots=True)
class GoogleHealthConfig:
    """Validated server-side settings for the Google Health OAuth client."""

    client_id: str
    client_secret: str
    redirect_uri: str
    token_encryption_key: str
    sync_interval_hours: int = 12
    manual_refresh_cooldown_seconds: int = 3600
    oauth_state_ttl_seconds: int = 600

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> GoogleHealthConfig | None:
        """Build optional configuration from environment variables.

        Returns ``None`` when no Google Health settings are present. Partial
        configuration raises so a deployment cannot appear enabled while
        silently losing encryption or OAuth protection.
        """
        values = environ if environ is not None else os.environ
        client_id = values.get("GOOGLE_HEALTH_CLIENT_ID", "").strip()
        client_secret = values.get("GOOGLE_HEALTH_CLIENT_SECRET", "").strip()
        redirect_uri = (
            values.get("GOOGLE_HEALTH_REDIRECT_URI", "").strip()
            or GOOGLE_HEALTH_DEFAULT_REDIRECT_URI
        )
        encryption_key = values.get("GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY", "").strip()

        if not any((client_id, client_secret, encryption_key)):
            return None

        missing = [
            name
            for name, value in (
                ("GOOGLE_HEALTH_CLIENT_ID", client_id),
                ("GOOGLE_HEALTH_CLIENT_SECRET", client_secret),
                ("GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY", encryption_key),
            )
            if not value
        ]
        if missing:
            names = ", ".join(missing)
            raise GoogleHealthConfigurationError(
                f"Google Health configuration is incomplete: {names}"
            )

        _validate_redirect_uri(redirect_uri)
        try:
            TokenCipher(encryption_key)
        except TokenEncryptionError as exc:
            raise GoogleHealthConfigurationError(str(exc)) from exc
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_encryption_key=encryption_key,
            sync_interval_hours=_positive_int(
                values.get("GOOGLE_HEALTH_SYNC_INTERVAL_HOURS", "12"),
                "GOOGLE_HEALTH_SYNC_INTERVAL_HOURS",
            ),
            manual_refresh_cooldown_seconds=_positive_int(
                values.get("GOOGLE_HEALTH_MANUAL_REFRESH_COOLDOWN_SECONDS", "3600"),
                "GOOGLE_HEALTH_MANUAL_REFRESH_COOLDOWN_SECONDS",
            ),
            oauth_state_ttl_seconds=_positive_int(
                values.get("GOOGLE_HEALTH_OAUTH_STATE_TTL_SECONDS", "600"),
                "GOOGLE_HEALTH_OAUTH_STATE_TTL_SECONDS",
            ),
        )

    @property
    def cipher(self) -> TokenCipher:
        """Return the configured refresh-token cipher."""
        return TokenCipher(self.token_encryption_key)

    def authorization_url(self, state: str) -> str:
        """Build a Google OAuth authorization URL for one stored state value."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "access_type": "offline",
            "include_granted_scopes": "true",
            "prompt": "consent",
            "scope": " ".join(GOOGLE_HEALTH_SCOPES),
            "state": state,
        }
        return f"{GOOGLE_HEALTH_AUTHORIZATION_URL}?{urlencode(params)}"


def google_health_configured_from_environment(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether a complete Google Health configuration is present."""
    try:
        return GoogleHealthConfig.from_environment(environ) is not None
    except GoogleHealthConfigurationError:
        return False


def health_user_id_for_telegram_user(user_id: str) -> str | None:
    """Collapse a Telegram topic identity to its private-chat health identity."""
    match = _TELEGRAM_HEALTH_USER_PATTERN.fullmatch(user_id)
    if match is None:
        return None
    return f"telegram-chat-{match.group('chat_id')}"


def telegram_chat_id_for_health_user(user_id: str) -> int | None:
    """Extract a chat ID only from a topic-independent health identity."""
    match = _TELEGRAM_HEALTH_CHAT_PATTERN.fullmatch(user_id)
    if match is None:
        return None
    return int(match.group("chat_id"))


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise GoogleHealthConfigurationError(
            f"{name} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise GoogleHealthConfigurationError(f"{name} must be a positive integer")
    return parsed


def _validate_redirect_uri(redirect_uri: str) -> None:
    parsed = urlsplit(redirect_uri)
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}:
        return
    raise GoogleHealthConfigurationError(
        "GOOGLE_HEALTH_REDIRECT_URI must use HTTPS, except for localhost development"
    )
