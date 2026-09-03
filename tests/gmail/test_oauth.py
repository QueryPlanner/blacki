"""Tests for Gmail's user-bound OAuth lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import aiosqlite
import httpx
import pytest
from cryptography.fernet import Fernet

from blacki.gmail.client import GmailService
from blacki.gmail.config import GMAIL_SCOPE, GmailConfig
from blacki.gmail.errors import (
    GmailAlreadyConnectedError,
    GmailCredentialError,
    GmailInputError,
    GmailMissingScopeError,
)
from blacki.gmail.oauth import (
    GmailOAuthError,
    GmailOAuthService,
    _hash_state,
    begin_gmail_authorization,
    complete_web_authorization,
    create_gmail_authorization_url,
)
from blacki.gmail.storage import SqliteGmailStorage
from blacki.storage.sqlite import create_connection


def _config() -> GmailConfig:
    return GmailConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/gmail/callback",
        token_encryption_key=Fernet.generate_key().decode(),
        oauth_state_ttl_seconds=600,
    )


async def _storage(tmp_path: Path) -> tuple[aiosqlite.Connection, SqliteGmailStorage]:
    connection = await create_connection(tmp_path / "tools.db")
    storage = SqliteGmailStorage(connection, asyncio.Lock())
    await storage.initialize()
    return connection, storage


@pytest.mark.asyncio
async def test_oauth_state_is_bound_to_user_and_consumed_before_exchange(
    tmp_path: Path,
) -> None:
    connection, storage = await _storage(tmp_path)
    service = GmailOAuthService(_config(), storage)
    try:
        url = await service.begin_authorization("telegram-chat-42-thread-3")
        state = parse_qs(urlsplit(url).query)["state"][0]
        assert state
        async with connection.execute(
            "SELECT state_hash, telegram_user_id FROM gmail_oauth_states"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == hashlib.sha256(state.encode()).hexdigest()
        assert row[0] != state
        assert row[1] == "telegram-chat-42"
        with pytest.raises(GmailOAuthError, match="did not include a code"):
            await service.complete_authorization(state=state, code=None)
        assert await storage.consume_oauth_state(row[0], now=0) is None
    finally:
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_oauth_completion_encrypts_refresh_token_and_refuses_reconnect(
    tmp_path: Path,
) -> None:
    config = _config()
    connection, storage = await _storage(tmp_path)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "scope": GMAIL_SCOPE,
                "expires_in": 3600,
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = GmailOAuthService(config, storage, http_client=client)
    try:
        url = await service.begin_authorization("telegram-chat-42")
        state = parse_qs(urlsplit(url).query)["state"][0]
        completion = await service.complete_authorization(
            state=state,
            code="authorization-code",
        )
        assert completion.connected is True
        assert completion.telegram_user_id == "telegram-chat-42"
        connection_row = await storage.get_connection("telegram-chat-42")
        assert connection_row is not None
        ciphertext = connection_row.encrypted_refresh_token.encode()
        assert b"refresh-token" not in ciphertext
        assert config.cipher.decrypt(connection_row.encrypted_refresh_token) == (
            "refresh-token"
        )
        assert requests[0].url == httpx.URL("https://oauth2.googleapis.com/token")
        with pytest.raises(GmailAlreadyConnectedError):
            await service.begin_authorization("telegram-chat-42-thread-4")
    finally:
        await service.close()
        await client.aclose()
        await connection.close()


@pytest.mark.asyncio
async def test_oauth_cancellation_and_missing_scope_are_safe(tmp_path: Path) -> None:
    config = _config()
    connection, storage = await _storage(tmp_path)

    async def missing_scope(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "scope": "https://www.googleapis.com/auth/gmail.readonly",
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(missing_scope))
    service = GmailOAuthService(config, storage, http_client=client)
    try:
        cancel_url = await service.begin_authorization("telegram-chat-42")
        cancel_state = parse_qs(urlsplit(cancel_url).query)["state"][0]
        cancelled = await service.complete_authorization(
            state=cancel_state,
            code=None,
            error="access_denied",
        )
        assert cancelled.connected is False
        assert await storage.get_connection("telegram-chat-42") is None

        missing_url = await service.begin_authorization("telegram-chat-42")
        missing_state = parse_qs(urlsplit(missing_url).query)["state"][0]
        with pytest.raises(GmailMissingScopeError):
            await service.complete_authorization(
                state=missing_state,
                code="authorization-code",
            )
        with pytest.raises(GmailOAuthError):
            await service.complete_authorization(
                state=missing_state,
                code="authorization-code",
            )
    finally:
        await service.close()
        await client.aclose()
        await connection.close()


@pytest.mark.asyncio
async def test_disconnect_revokes_then_removes_only_one_user(tmp_path: Path) -> None:
    config = _config()
    connection, storage = await _storage(tmp_path)
    for user_id, token in (
        ("telegram-chat-42", "token-one"),
        ("telegram-chat-43", "token-two"),
    ):
        await storage.save_connection(
            telegram_user_id=user_id,
            encrypted_refresh_token=config.cipher.encrypt(token),
            scopes=(GMAIL_SCOPE,),
        )
    revoked: list[bytes] = []

    async def revoke_handler(request: httpx.Request) -> httpx.Response:
        revoked.append(request.content)
        return httpx.Response(200, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(revoke_handler))
    service = GmailOAuthService(config, storage, http_client=client)
    try:
        assert await service.disconnect("telegram-chat-42-thread-8") is True
        assert await storage.get_connection("telegram-chat-42") is None
        assert await storage.get_connection("telegram-chat-43") is not None
        assert revoked and b"token-one" in revoked[0]
        assert await service.disconnect("telegram-chat-42") is False

        async def failed_revoke(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, request=request)

        await storage.save_connection(
            telegram_user_id="telegram-chat-42",
            encrypted_refresh_token=config.cipher.encrypt("token-again"),
            scopes=(GMAIL_SCOPE,),
        )
        await client.aclose()
        failed_client = httpx.AsyncClient(transport=httpx.MockTransport(failed_revoke))
        failed_service = GmailService(config, storage, http_client=failed_client)
        assert await failed_service.disconnect("telegram-chat-42") is True
        assert await storage.get_connection("telegram-chat-42") is None
        await failed_service.close()
        await failed_client.aclose()
    finally:
        await service.close()
        await client.aclose()
        await connection.close()


@pytest.mark.asyncio
async def test_convenience_oauth_wrappers_use_explicit_dependencies(
    tmp_path: Path,
) -> None:
    connection, storage = await _storage(tmp_path)
    config = _config()
    try:
        url = await begin_gmail_authorization(
            "telegram-chat-42",
            config=config,
            storage=storage,
        )
        state = parse_qs(urlsplit(url).query)["state"][0]

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "scope": GMAIL_SCOPE,
                },
                request=request,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        completion = await complete_web_authorization(
            code="code",
            state=state,
            config=config,
            storage=storage,
            http_client=client,
        )
        assert completion.connected is True
        await client.aclose()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_oauth_owned_http_client_is_closed_and_private_ids_are_required(
    tmp_path: Path,
) -> None:
    connection, storage = await _storage(tmp_path)
    service = GmailOAuthService(_config(), storage)
    try:
        client = await service._client()
        assert client.is_closed is False
        with pytest.raises(GmailOAuthError):
            await service.begin_authorization("local")
    finally:
        await service.close()
        assert client.is_closed is True
        await connection.close()


@pytest.mark.asyncio
async def test_oauth_completion_refuses_existing_connection_and_missing_refresh_token(
    tmp_path: Path,
) -> None:
    config = _config()
    connection, storage = await _storage(tmp_path)
    service = GmailOAuthService(config, storage, http_client=AsyncMock())
    try:
        state_url = await service.begin_authorization("telegram-chat-42")
        state = parse_qs(urlsplit(state_url).query)["state"][0]
        await storage.save_connection(
            telegram_user_id="telegram-chat-42",
            encrypted_refresh_token=config.cipher.encrypt("existing"),
            scopes=(GMAIL_SCOPE,),
        )
        with pytest.raises(GmailAlreadyConnectedError):
            await service.complete_authorization(state=state, code="code")

        await storage.remove_connection("telegram-chat-42")
        next_url = await service.begin_authorization("telegram-chat-42")
        next_state = parse_qs(urlsplit(next_url).query)["state"][0]
        with (
            patch(
                "blacki.gmail.oauth.exchange_code_for_tokens",
                new=AsyncMock(return_value={"scope": GMAIL_SCOPE}),
            ),
            pytest.raises(GmailCredentialError, match="refresh token"),
        ):
            await service.complete_authorization(
                state=next_state,
                code="code",
            )
    finally:
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_complete_web_authorization_reports_missing_dependencies(
    tmp_path: Path,
) -> None:
    config = _config()
    connection, storage = await _storage(tmp_path)
    try:
        with (
            patch(
                "blacki.gmail.oauth.GmailConfig.from_environment",
                return_value=None,
            ),
            pytest.raises(GmailOAuthError, match="not configured"),
        ):
            await complete_web_authorization(
                code="code",
                state="state",
                config=None,
                storage=storage,
            )

        with (
            patch(
                "blacki.gmail.oauth.get_container",
                side_effect=RuntimeError("container secret"),
            ),
            pytest.raises(GmailOAuthError, match="storage is not available"),
        ):
            await complete_web_authorization(
                code="code",
                state="state",
                config=config,
                storage=None,
            )
    finally:
        await connection.close()


def test_oauth_state_and_authorization_url_validation() -> None:
    config = _config()
    with pytest.raises(GmailOAuthError):
        _hash_state("invalid state")
    with pytest.raises(GmailOAuthError):
        _hash_state("x" * 513)
    with pytest.raises(GmailInputError):
        create_gmail_authorization_url(config=config, state="invalid state")
    assert "state=valid" in create_gmail_authorization_url(
        config=config,
        state="valid",
    )
