"""Configuration and identity helpers for Gmail OAuth."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

from blacki.health.config import TokenCipher, TokenEncryptionError

from .errors import GmailConfigurationError

GMAIL_AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
GMAIL_REVOCATION_URL = "https://oauth2.googleapis.com/revoke"
GMAIL_API_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.modify"
GMAIL_SCOPES = (GMAIL_SCOPE,)
GMAIL_ENABLED_VALUES = frozenset({"1", "true", "yes"})
GMAIL_DEFAULT_REDIRECT_URI = "http://127.0.0.1:8080/integrations/gmail/callback"
GMAIL_STATE_TTL_SECONDS = 600
GMAIL_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
TOKEN_EXPIRY_GRACE_SECONDS = 60.0
_TELEGRAM_GMAIL_USER_PATTERN = re.compile(
    r"^telegram-chat-(?P<chat_id>[1-9][0-9]*)(?:-thread-[0-9]+)?$"
)


def resolve_gmail_redirect_uri(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the Gmail callback from Gmail-only configuration."""
    values = environ if environ is not None else os.environ
    explicit = values.get("GMAIL_REDIRECT_URI", "").strip()
    if explicit:
        _validate_redirect_uri(explicit)
        return explicit

    return GMAIL_DEFAULT_REDIRECT_URI


def canonical_gmail_user_id(user_id: object) -> str | None:
    """Return the topic-independent private Telegram identity, if valid."""
    if not isinstance(user_id, str):
        return None
    match = _TELEGRAM_GMAIL_USER_PATTERN.fullmatch(user_id)
    if match is None:
        return None
    return f"telegram-chat-{match.group('chat_id')}"


def gmail_user_id_for_chat(chat_id: int) -> str:
    """Build a canonical Gmail identity for one positive Telegram chat ID."""
    if not isinstance(chat_id, int) or isinstance(chat_id, bool) or chat_id <= 0:
        raise GmailConfigurationError("Gmail requires a private Telegram chat ID")
    return f"telegram-chat-{chat_id}"


@dataclass(frozen=True, slots=True)
class GmailConfig:
    """Validated server-side settings for the Gmail OAuth client."""

    client_id: str
    client_secret: str
    redirect_uri: str
    token_encryption_key: str
    oauth_state_ttl_seconds: int = GMAIL_STATE_TTL_SECONDS
    max_attachment_bytes: int = GMAIL_MAX_ATTACHMENT_BYTES

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attachment_bytes, bool)
            or not isinstance(self.max_attachment_bytes, int)
            or self.max_attachment_bytes <= 0
        ):
            raise GmailConfigurationError(
                "GMAIL_MAX_ATTACHMENT_BYTES must be a positive integer"
            )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> GmailConfig | None:
        """Build Gmail settings only when its dedicated flag is enabled."""
        values = environ if environ is not None else os.environ
        enabled = (
            values.get("GMAIL_ENABLED", "false").strip().lower() in GMAIL_ENABLED_VALUES
        )
        if not enabled:
            return None

        client_id = values.get("GMAIL_CLIENT_ID", "").strip()
        client_secret = values.get("GMAIL_CLIENT_SECRET", "").strip()
        encryption_key = values.get("GMAIL_TOKEN_ENCRYPTION_KEY", "").strip()

        missing = [
            name
            for name, value in (
                ("GMAIL_CLIENT_ID", client_id),
                ("GMAIL_CLIENT_SECRET", client_secret),
                ("GMAIL_TOKEN_ENCRYPTION_KEY", encryption_key),
            )
            if not value
        ]
        if missing:
            raise GmailConfigurationError(
                "Gmail configuration is incomplete: " + ", ".join(missing)
            )

        redirect_uri = resolve_gmail_redirect_uri(values)
        max_attachment_bytes = _max_attachment_bytes(values)
        try:
            TokenCipher(encryption_key, key_name="GMAIL_TOKEN_ENCRYPTION_KEY")
        except TokenEncryptionError as exc:
            raise GmailConfigurationError(str(exc)) from exc
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_encryption_key=encryption_key,
            max_attachment_bytes=max_attachment_bytes,
        )

    @property
    def cipher(self) -> TokenCipher:
        """Return the Fernet cipher for refresh tokens."""
        return TokenCipher(
            self.token_encryption_key,
            key_name="GMAIL_TOKEN_ENCRYPTION_KEY",
        )

    def authorization_url(self, state: str) -> str:
        """Build a Gmail OAuth URL using only the selected restricted scope."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "scope": " ".join(GMAIL_SCOPES),
            "state": state,
        }
        return f"{GMAIL_AUTHORIZATION_URL}?{urlencode(params)}"


def _validate_redirect_uri(redirect_uri: str) -> None:
    parsed = urlsplit(redirect_uri)
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}:
        return
    raise GmailConfigurationError(
        "GMAIL_REDIRECT_URI must use HTTPS, except for localhost development"
    )


def _max_attachment_bytes(values: Mapping[str, str]) -> int:
    raw_value = values.get("GMAIL_MAX_ATTACHMENT_BYTES", "").strip()
    if not raw_value:
        return GMAIL_MAX_ATTACHMENT_BYTES
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise GmailConfigurationError(
            "GMAIL_MAX_ATTACHMENT_BYTES must be a positive integer"
        ) from exc
    if value <= 0:
        raise GmailConfigurationError(
            "GMAIL_MAX_ATTACHMENT_BYTES must be a positive integer"
        )
    return value
