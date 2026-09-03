"""Tests for Gmail configuration and SQLite credential isolation."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import aiosqlite
import pytest
from cryptography.fernet import Fernet

from blacki.gmail.config import (
    GMAIL_SCOPE,
    GmailConfig,
    canonical_gmail_user_id,
    gmail_user_id_for_chat,
    resolve_gmail_redirect_uri,
)
from blacki.gmail.errors import (
    GmailAlreadyConnectedError,
    GmailConfigurationError,
    GmailCredentialError,
)
from blacki.gmail.storage import SqliteGmailStorage, _connection_from_row
from blacki.storage.sqlite import create_connection


def _config() -> GmailConfig:
    return GmailConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/gmail/callback",
        token_encryption_key=Fernet.generate_key().decode(),
    )


async def _storage(tmp_path: Path) -> tuple[aiosqlite.Connection, SqliteGmailStorage]:
    connection = await create_connection(tmp_path / "tools.db")
    storage = SqliteGmailStorage(connection, asyncio.Lock())
    await storage.initialize()
    return connection, storage


def test_gmail_config_uses_one_restricted_scope_and_safe_redirect() -> None:
    key = Fernet.generate_key().decode()
    config = GmailConfig.from_environment(
        {
            "GOOGLE_HEALTH_CLIENT_ID": " client ",
            "GOOGLE_HEALTH_CLIENT_SECRET": " secret ",
            "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": key,
            "GOOGLE_HEALTH_REDIRECT_URI": "https://example.test/integrations/google-health/callback",
        }
    )
    assert config is not None
    assert config.client_id == "client"
    assert config.redirect_uri.endswith("/integrations/gmail/callback")
    params = parse_qs(urlsplit(config.authorization_url("state")).query)
    assert params["scope"] == [GMAIL_SCOPE]
    assert "mail.google.com" not in config.authorization_url("state")
    assert "include_granted_scopes" not in params


@pytest.mark.parametrize(
    ("user_id", "expected"),
    [
        ("telegram-chat-42", "telegram-chat-42"),
        ("telegram-chat-42-thread-7", "telegram-chat-42"),
        ("telegram-chat-0", None),
        ("telegram-chat--100", None),
        ("local", None),
        ("telegram-chat-42-thread-x", None),
    ],
)
def test_gmail_identity_is_private_and_topic_independent(
    user_id: str,
    expected: str | None,
) -> None:
    assert canonical_gmail_user_id(user_id) == expected


def test_gmail_config_rejects_missing_and_insecure_values() -> None:
    assert GmailConfig.from_environment({}) is None
    with pytest.raises(GmailConfigurationError, match="incomplete"):
        GmailConfig.from_environment({"GOOGLE_HEALTH_CLIENT_ID": "client"})
    with pytest.raises(GmailConfigurationError, match="HTTPS"):
        GmailConfig.from_environment(
            {
                "GOOGLE_HEALTH_CLIENT_ID": "client",
                "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
                "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
                "GMAIL_REDIRECT_URI": "http://example.test/callback",
            }
        )
    assert (
        resolve_gmail_redirect_uri({})
        == "http://127.0.0.1:8080/integrations/gmail/callback"
    )
    with pytest.raises(GmailConfigurationError):
        gmail_user_id_for_chat(-42)

    valid_environment = {
        "GOOGLE_HEALTH_CLIENT_ID": "client",
        "GOOGLE_HEALTH_CLIENT_SECRET": "secret",
        "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    }
    with pytest.raises(GmailConfigurationError, match="positive integer"):
        GmailConfig.from_environment(
            {**valid_environment, "GMAIL_MAX_ATTACHMENT_BYTES": "not-an-integer"}
        )
    with pytest.raises(GmailConfigurationError, match="positive integer"):
        GmailConfig.from_environment(
            {**valid_environment, "GMAIL_MAX_ATTACHMENT_BYTES": "0"}
        )
    configured = GmailConfig.from_environment(
        {**valid_environment, "GMAIL_MAX_ATTACHMENT_BYTES": "4"}
    )
    assert configured is not None
    assert configured.max_attachment_bytes == 4
    with pytest.raises(GmailConfigurationError, match="positive integer"):
        GmailConfig(
            client_id="client",
            client_secret="secret",
            redirect_uri="https://example.test/callback",
            token_encryption_key=Fernet.generate_key().decode(),
            max_attachment_bytes=0,
        )


def test_gmail_config_accepts_explicit_and_local_redirects() -> None:
    explicit = "https://gmail.example.test/callback"
    assert resolve_gmail_redirect_uri({"GMAIL_REDIRECT_URI": explicit}) == explicit
    assert (
        resolve_gmail_redirect_uri(
            {"GMAIL_REDIRECT_URI": "http://localhost:8080/callback"}
        )
        == "http://localhost:8080/callback"
    )
    assert canonical_gmail_user_id(42) is None
    assert gmail_user_id_for_chat(42) == "telegram-chat-42"


@pytest.mark.asyncio
async def test_gmail_storage_state_is_hashed_single_use_and_expiring(
    tmp_path: Path,
) -> None:
    connection, storage = await _storage(tmp_path)
    try:
        raw_state = "unreturned-oauth-state"
        state_hash = hashlib.sha256(raw_state.encode()).hexdigest()
        await storage.store_oauth_state(
            state_hash,
            "telegram-chat-42-thread-9",
            expires_at=200,
            now=100,
        )
        assert (
            await storage.consume_oauth_state(state_hash, now=150) == "telegram-chat-42"
        )
        assert await storage.consume_oauth_state(state_hash, now=150) is None

        expired_hash = hashlib.sha256(b"expired").hexdigest()
        await storage.store_oauth_state(
            expired_hash,
            "telegram-chat-43",
            expires_at=200,
            now=100,
        )
        assert await storage.consume_oauth_state(expired_hash, now=200) is None

        with pytest.raises(GmailCredentialError):
            await storage.store_oauth_state(
                "not-a-hash", "telegram-chat-42", expires_at=200
            )
        with pytest.raises(GmailCredentialError):
            await storage.store_oauth_state(
                state_hash,
                "telegram-chat-42",
                expires_at=100,
                now=100,
            )
        async with connection.execute(
            "SELECT COUNT(*) AS count FROM gmail_oauth_states"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0
    finally:
        await connection.close()


@pytest.mark.parametrize(
    "row",
    [
        {
            "telegram_user_id": "telegram-chat-42",
            "scopes_json": None,
            "encrypted_refresh_token": "ciphertext",
            "status": "connected",
            "connected_at": 1,
        },
        {
            "telegram_user_id": "telegram-chat-42",
            "scopes_json": "not-json",
            "encrypted_refresh_token": "ciphertext",
            "status": "connected",
            "connected_at": 1,
        },
        {
            "telegram_user_id": "telegram-chat-42",
            "scopes_json": "{}",
            "encrypted_refresh_token": "ciphertext",
            "status": "connected",
            "connected_at": 1,
        },
        {
            "telegram_user_id": "telegram-chat-42",
            "scopes_json": "[1]",
            "encrypted_refresh_token": "ciphertext",
            "status": "connected",
            "connected_at": 1,
        },
        {
            "telegram_user_id": "telegram-chat-42",
            "scopes_json": "[]",
            "encrypted_refresh_token": 1,
            "status": "connected",
            "connected_at": 1,
        },
        {
            "telegram_user_id": "telegram-chat-42",
            "scopes_json": "[]",
            "encrypted_refresh_token": "ciphertext",
            "status": 1,
            "connected_at": 1,
        },
    ],
)
def test_gmail_storage_rejects_corrupt_connection_rows(
    row: dict[str, Any],
) -> None:
    with pytest.raises(GmailCredentialError):
        _connection_from_row(row)


@pytest.mark.asyncio
async def test_gmail_storage_accepts_rotated_tokens_and_rejects_empty_rotation(
    tmp_path: Path,
) -> None:
    connection, storage = await _storage(tmp_path)
    config = _config()
    try:
        await storage.save_connection(
            telegram_user_id="telegram-chat-42",
            encrypted_refresh_token=config.cipher.encrypt("old-token"),
            scopes=(GMAIL_SCOPE,),
        )
        with pytest.raises(GmailCredentialError):
            await storage.replace_refresh_token("telegram-chat-42", "")
        rotated = config.cipher.encrypt("new-token")
        await storage.replace_refresh_token("telegram-chat-42", rotated)
        stored = await storage.get_connection("telegram-chat-42")
        assert stored is not None
        assert config.cipher.decrypt(stored.encrypted_refresh_token) == "new-token"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gmail_storage_encrypts_and_isolates_connections(tmp_path: Path) -> None:
    connection, storage = await _storage(tmp_path)
    config = _config()
    try:
        encrypted_one = config.cipher.encrypt("refresh-one")
        encrypted_two = config.cipher.encrypt("refresh-two")
        await storage.save_connection(
            telegram_user_id="telegram-chat-42-thread-1",
            encrypted_refresh_token=encrypted_one,
            scopes=(GMAIL_SCOPE,),
            connected_at=1,
        )
        await storage.save_connection(
            telegram_user_id="telegram-chat-43",
            encrypted_refresh_token=encrypted_two,
            scopes=(GMAIL_SCOPE,),
            connected_at=2,
        )
        assert await storage.has_connection("telegram-chat-42") is True
        assert await storage.has_connection("telegram-chat-43") is True
        with pytest.raises(GmailAlreadyConnectedError):
            await storage.save_connection(
                telegram_user_id="telegram-chat-42",
                encrypted_refresh_token=config.cipher.encrypt("replacement"),
                scopes=(GMAIL_SCOPE,),
            )

        async with connection.execute(
            "SELECT encrypted_refresh_token FROM gmail_connections "
            "WHERE telegram_user_id = 'telegram-chat-42'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == encrypted_one
        assert "refresh-one" not in row[0]

        await storage.mark_reauthorization_required("telegram-chat-42")
        assert await storage.has_connection("telegram-chat-42") is False
        await storage.save_connection(
            telegram_user_id="telegram-chat-42",
            encrypted_refresh_token=config.cipher.encrypt("refresh-new"),
            scopes=(GMAIL_SCOPE,),
        )
        assert await storage.remove_connection("telegram-chat-42") is not None
        assert await storage.has_connection("telegram-chat-42") is False
        assert await storage.has_connection("telegram-chat-43") is True

        with pytest.raises(GmailCredentialError):
            await storage.get_connection("local")
        with pytest.raises(GmailCredentialError):
            await storage.save_connection(
                telegram_user_id="telegram-chat-44",
                encrypted_refresh_token="",
                scopes=(GMAIL_SCOPE,),
            )
        with pytest.raises(GmailCredentialError):
            await storage.save_connection(
                telegram_user_id="telegram-chat-44",
                encrypted_refresh_token=config.cipher.encrypt("no-scope"),
                scopes=(),
            )
    finally:
        await connection.close()
