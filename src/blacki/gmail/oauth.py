"""User-bound Gmail OAuth flow for private Telegram chats."""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import cast

import httpx

from blacki.container import get_container

from .client import HTTP_TIMEOUT_SECONDS, GmailService, exchange_code_for_tokens
from .config import (
    GMAIL_STATE_TTL_SECONDS,
    GmailConfig,
    canonical_gmail_user_id,
)
from .errors import (
    GmailAlreadyConnectedError,
    GmailCredentialError,
    GmailInputError,
)
from .storage import SqliteGmailStorage

DEFAULT_GMAIL_REDIRECT_PATH = "/integrations/gmail/callback"
DEFAULT_GMAIL_REDIRECT_URI = "http://127.0.0.1:8080" + DEFAULT_GMAIL_REDIRECT_PATH
OAUTH_STATE_TTL_SECONDS = GMAIL_STATE_TTL_SECONDS


class GmailOAuthError(GmailCredentialError):
    """Raised when Gmail OAuth state or completion cannot be accepted."""


@dataclass(frozen=True, slots=True)
class GmailOAuthCompletion:
    """Result of one callback, without provider payloads."""

    telegram_user_id: str
    connected: bool


class GmailOAuthService:
    """Create and complete OAuth flows against one shared Gmail configuration."""

    def __init__(
        self,
        config: GmailConfig,
        storage: SqliteGmailStorage,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._gmail_service: GmailService | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        return self._http_client

    async def _service(self) -> GmailService:
        if self._gmail_service is None:
            self._gmail_service = GmailService(
                self.config,
                self.storage,
                http_client=await self._client(),
            )
        return self._gmail_service

    async def begin_authorization(self, telegram_user_id: str) -> str:
        """Store a hash of a one-time state and return its authorization URL."""
        user_id = _require_user_id(telegram_user_id)
        await self.storage.initialize()
        existing = await self.storage.get_connection(user_id)
        if existing and (
            existing.status == "connected" or existing.encrypted_refresh_token
        ):
            raise GmailAlreadyConnectedError(
                "Disconnect Gmail before connecting another account"
            )
        state = secrets.token_urlsafe(32)
        await self.storage.store_oauth_state(
            _hash_state(state),
            user_id,
            expires_at=time.time() + self.config.oauth_state_ttl_seconds,
        )
        return self.config.authorization_url(state)

    async def complete_authorization(
        self,
        *,
        state: str,
        code: str | None,
        error: str | None = None,
    ) -> GmailOAuthCompletion:
        """Consume state before exchanging a code, so callbacks are single-use."""
        state_hash = _hash_state(state)
        await self.storage.initialize()
        user_id = await self.storage.consume_oauth_state(state_hash)
        if user_id is None:
            raise GmailOAuthError("Gmail OAuth state is invalid or expired")
        if error:
            return GmailOAuthCompletion(telegram_user_id=user_id, connected=False)
        if not code:
            raise GmailOAuthError("Gmail OAuth callback did not include a code")
        if await self.storage.has_connection(user_id):
            raise GmailAlreadyConnectedError(
                "Disconnect Gmail before connecting another account"
            )

        tokens = await exchange_code_for_tokens(
            code=code,
            config=self.config,
            http_client=await self._client(),
        )
        refresh_token = cast(str, tokens["refresh_token"])
        scopes = tuple(str(scope) for scope in str(tokens["scope"]).split())
        await self.storage.save_connection(
            telegram_user_id=user_id,
            encrypted_refresh_token=self.config.cipher.encrypt(refresh_token),
            scopes=scopes,
        )
        return GmailOAuthCompletion(telegram_user_id=user_id, connected=True)

    async def disconnect(self, telegram_user_id: str) -> bool:
        """Attempt remote revocation and then remove exactly one local row."""
        user_id = _require_user_id(telegram_user_id)
        service = await self._service()
        return await service.disconnect(user_id)

    async def close(self) -> None:
        """Close the shared provider client."""
        if self._gmail_service is not None:
            await self._gmail_service.close()
            self._gmail_service = None
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


async def begin_gmail_authorization(
    telegram_user_id: str,
    *,
    config: GmailConfig,
    storage: SqliteGmailStorage,
) -> str:
    """Convenience wrapper for starting one private user's OAuth flow."""
    service = GmailOAuthService(config, storage)
    try:
        return await service.begin_authorization(telegram_user_id)
    finally:
        await service.close()


async def complete_web_authorization(
    *,
    code: str | None,
    state: str,
    error: str | None = None,
    config: GmailConfig | None = None,
    storage: SqliteGmailStorage | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> GmailOAuthCompletion:
    """Complete a callback using explicit dependencies or the app container."""
    resolved_config = config or GmailConfig.from_environment()
    if resolved_config is None:
        raise GmailOAuthError("Gmail is not configured on this Blacki server")
    if storage is None:
        try:
            storage = get_container().gmail_storage
        except RuntimeError as exc:
            raise GmailOAuthError("Gmail storage is not available") from exc
    service = GmailOAuthService(
        resolved_config,
        storage,
        http_client=http_client,
    )
    try:
        return await service.complete_authorization(
            state=state,
            code=code,
            error=error,
        )
    finally:
        await service.close()


def create_gmail_authorization_url(*, config: GmailConfig, state: str) -> str:
    """Build an authorization URL from an already validated configuration."""
    if not state or any(char.isspace() for char in state):
        raise GmailInputError("Gmail OAuth state is invalid")
    return config.authorization_url(state)


def _hash_state(state: str) -> str:
    if (
        not isinstance(state, str)
        or not state
        or len(state) > 512
        or any(char.isspace() for char in state)
    ):
        raise GmailOAuthError("Gmail OAuth state is missing or invalid")
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def _require_user_id(user_id: str) -> str:
    canonical = canonical_gmail_user_id(user_id)
    if canonical is None:
        raise GmailOAuthError(
            "Gmail authorization is available only to a private Telegram user"
        )
    return canonical
