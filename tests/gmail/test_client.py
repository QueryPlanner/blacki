"""HTTP-boundary tests for Gmail REST operations and credential refresh."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cryptography.fernet import Fernet

from blacki.gmail.client import (
    MAX_BODY_LENGTH,
    MAX_LABEL_ID_LENGTH,
    MAX_LABEL_IDS,
    MAX_MESSAGE_BODY_BYTES,
    MAX_MIME_ATTACHMENTS,
    MAX_MIME_DEPTH,
    MAX_MIME_PARTS,
    MAX_PAGE_TOKEN_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_RECIPIENTS,
    GmailApiClient,
    GmailAttachmentDownload,
    GmailCredentialManager,
    GmailService,
    _add_page_token,
    _build_raw_message,
    _decode_attachment_data,
    _decode_body,
    _draft_matches,
    _find_message_part,
    _header_map,
    _headers,
    _normalize_message,
    _object,
    _object_list,
    _optional_int,
    _optional_string,
    _parse_recipients,
    _positive_expiry,
    _raise_api_error,
    _require_user_id,
    _required_string,
    _response_error_code,
    _response_object,
    _retry_after,
    _scope_set,
    _string_list,
    _validate_body,
    _validate_count,
    _validate_label_ids,
    _validate_label_name,
    _validate_query,
    _validate_resource_id,
    _validate_subject,
    exchange_code_for_tokens,
)
from blacki.gmail.config import GMAIL_SCOPE, GmailConfig
from blacki.gmail.errors import (
    GmailAccessDeniedError,
    GmailApiError,
    GmailAuthError,
    GmailAuthorizationRequiredError,
    GmailCredentialError,
    GmailDraftChangedError,
    GmailInputError,
    GmailMalformedResponseError,
    GmailMissingScopeError,
    GmailRateLimitError,
    GmailTransportError,
    safe_provider_error_code,
)
from blacki.gmail.storage import SqliteGmailStorage
from blacki.storage.sqlite import create_connection

USER_ID = "telegram-chat-42"


def _config() -> GmailConfig:
    return GmailConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/gmail/callback",
        token_encryption_key=Fernet.generate_key().decode(),
    )


def _encoded(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _message(
    message_id: str = "message-1",
    thread_id: str = "thread-1",
    *,
    subject: str = "Subject",
    to: str = "person@example.com",
    body: str = "hello",
    extra_parts: list[dict[str, Any]] | None = None,
    headers: list[Any] | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [
        {
            "mimeType": "text/plain",
            "filename": "",
            "body": {"data": _encoded(body), "size": len(body)},
        }
    ]
    if extra_parts:
        parts.extend(extra_parts)
    return {
        "id": message_id,
        "threadId": thread_id,
        "labelIds": ["INBOX"],
        "snippet": body[:30],
        "internalDate": "1700000000000",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": headers
            if headers is not None
            else [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": to},
                {"name": "Subject", "value": subject},
                {"name": "Message-ID", "value": "<message@example.com>"},
            ],
            "parts": parts,
        },
    }


async def _ready(
    tmp_path: Path,
    handler: Any,
    config: GmailConfig | None = None,
) -> tuple[httpx.AsyncClient, SqliteGmailStorage, GmailService, Any]:
    connection = await create_connection(tmp_path / "tools.db")
    storage = SqliteGmailStorage(connection, asyncio.Lock())
    await storage.initialize()
    config = config or _config()
    await storage.save_connection(
        telegram_user_id=USER_ID,
        encrypted_refresh_token=config.cipher.encrypt("refresh-token"),
        scopes=(GMAIL_SCOPE,),
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = GmailService(config, storage, http_client=client)
    return client, storage, service, connection


async def _close(
    client: httpx.AsyncClient,
    service: GmailService,
    connection: Any,
) -> None:
    await service.close()
    await client.aclose()
    await connection.close()


@pytest.mark.asyncio
async def test_search_read_thread_and_pagination(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "expires_in": 3600,
                    "scope": GMAIL_SCOPE,
                },
                request=request,
            )
        if request.url.path.endswith("/messages"):
            assert request.url.params["includeSpamTrash"] == "false"
            assert request.url.params["pageToken"] == "page-2"
            return httpx.Response(
                200,
                json={
                    "messages": [{"id": "message-1", "threadId": "thread-1"}],
                    "nextPageToken": "page-3",
                    "resultSizeEstimate": 1,
                },
                request=request,
            )
        if request.url.path.endswith("/messages/message-1"):
            return httpx.Response(200, json=_message(), request=request)
        if request.url.path.endswith("/threads/thread-1"):
            return httpx.Response(
                200,
                json={
                    "id": "thread-1",
                    "messages": [_message(), _message("message-2")],
                },
                request=request,
            )
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        result = await service.search_messages(
            USER_ID,
            query="from:sender@example.com",
            max_results=5,
            page_token="page-2",
        )
        assert result["messages"] == [{"id": "message-1", "thread_id": "thread-1"}]
        assert result["next_page_token"].startswith("page-")
        assert result["next_page_token"].endswith("3")
        message = await service.get_message(USER_ID, message_id="message-1")
        assert message["text_body"] == "hello"
        assert message["subject"] == "Subject"
        thread = await service.get_thread(USER_ID, thread_id="thread-1", max_messages=1)
        assert len(thread["messages"]) == 1
        assert len(calls) == 4
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "expected_headers"),
    [
        (
            [
                {"name": "From", "value": "sender@example.com"},
                {"name": "X-No-Value"},
                {"name": "Subject", "value": "Subject"},
            ],
            {"from": "sender@example.com", "x-no-value": "", "subject": "Subject"},
        ),
        (
            [
                {"name": "From", "value": "sender@example.com"},
                {"name": "X-Empty", "value": ""},
                {"name": "Subject", "value": "Subject"},
            ],
            {"from": "sender@example.com", "x-empty": "", "subject": "Subject"},
        ),
        (
            [
                {"name": None, "value": "ignored"},
                {"name": "To", "value": "person@example.com"},
                {"name": "Subject", "value": "Subject"},
            ],
            {"to": "person@example.com", "subject": "Subject"},
        ),
        (
            [
                {"name": "From", "value": "sender@example.com"},
                "malformed header entry",
                {"name": "Subject", "value": "Subject"},
            ],
            {"from": "sender@example.com", "subject": "Subject"},
        ),
    ],
)
async def test_message_headers_are_tolerant_per_entry(
    tmp_path: Path,
    headers: list[Any],
    expected_headers: dict[str, str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        return httpx.Response(
            200,
            json=_message(headers=headers),
            request=request,
        )

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        message = await service.get_message(USER_ID, message_id="message-1")
        assert message["text_body"] == "hello"
        assert message["headers"] == expected_headers
    finally:
        await _close(client, service, connection)


@pytest.mark.parametrize(
    "payload",
    [
        {"threadId": "thread-1", "payload": {}},
        {"id": "message-1", "payload": {}},
        {"id": "message-1", "threadId": "thread-1"},
    ],
)
def test_message_required_fields_remain_strict(payload: dict[str, Any]) -> None:
    with pytest.raises(GmailMalformedResponseError):
        _normalize_message(payload)


@pytest.mark.asyncio
async def test_mime_decoding_keeps_attachment_metadata_only(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        return httpx.Response(
            200,
            json=_message(
                extra_parts=[
                    {
                        "partId": "1",
                        "mimeType": "application/pdf",
                        "filename": "invoice.pdf",
                        "body": {"attachmentId": "attachment-1", "size": 4096},
                    }
                ]
            ),
            request=request,
        )

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        message = await service.get_message(USER_ID, message_id="message-1")
        assert message["attachments"] == [
            {
                "filename": "invoice.pdf",
                "mime_type": "application/pdf",
                "size": 4096,
                "attachment_id": "attachment-1",
                "part_id": "1",
            }
        ]
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_draft_creation_reply_and_send_recheck(tmp_path: Path) -> None:
    sent = False
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        requests.append(request)
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "expires_in": 3600,
                    "scope": GMAIL_SCOPE,
                },
                request=request,
            )
        if request.url.path.endswith("/messages/message-1"):
            return httpx.Response(
                200,
                json=_message(subject="Original", body="original"),
                request=request,
            )
        if request.url.path.endswith("/drafts") and request.method == "POST":
            body = json.loads(request.content)
            raw = body["message"]["raw"]
            decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
            assert "To: person@example.com" in decoded
            assert "In-Reply-To: <message@example.com>" in decoded
            assert body["message"]["threadId"] == "thread-1"
            return httpx.Response(
                200,
                json={"id": "draft-1", "message": _message(body="draft body")},
                request=request,
            )
        if request.url.path.endswith("/drafts/draft-1"):
            return httpx.Response(
                200,
                json={
                    "id": "draft-1",
                    "message": _message(
                        subject="Draft subject",
                        to="person@example.com",
                    ),
                },
                request=request,
            )
        if request.url.path.endswith("/drafts/send"):
            sent = True
            return httpx.Response(
                200,
                json={"id": "sent-1", "threadId": "thread-1"},
                request=request,
            )
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        draft = await service.create_draft(
            USER_ID,
            to="person@example.com",
            subject="Reply",
            body="draft body",
            reply_to_message_id="message-1",
        )
        assert draft["draft_id"] == "draft-1"
        result = await service.send_draft(
            USER_ID,
            draft_id="draft-1",
            expected_to="person@example.com",
            expected_cc="",
            expected_bcc="",
            expected_subject="Draft subject",
            expected_content_sha256=_normalize_message(
                _message(subject="Draft subject")
            )["content_sha256"],
        )
        assert result["sent"] is True
        assert result["message"]["id"] == "sent-1"
        assert sent is True
        assert any(request.url.path.endswith("/drafts/send") for request in requests)

        async def changed_handler(request: httpx.Request) -> httpx.Response:
            if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
                return httpx.Response(
                    200,
                    json={"access_token": "access", "expires_in": 3600},
                    request=request,
                )
            if request.url.path.endswith("/drafts/draft-1"):
                return httpx.Response(
                    200,
                    json={
                        "id": "draft-1",
                        "message": _message(subject="Changed"),
                    },
                    request=request,
                )
            raise AssertionError(request.url)

        await client.aclose()
        changed_client = httpx.AsyncClient(
            transport=httpx.MockTransport(changed_handler)
        )
        changed_service = GmailService(
            service.config, service.storage, http_client=changed_client
        )
        with pytest.raises(GmailDraftChangedError):
            await changed_service.send_draft(
                USER_ID,
                draft_id="draft-1",
                expected_to="person@example.com",
                expected_cc="",
                expected_bcc="",
                expected_subject="Draft subject",
                expected_content_sha256=_normalize_message(
                    _message(subject="Draft subject")
                )["content_sha256"],
            )
        await changed_service.close()
        await changed_client.aclose()
    finally:
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_draft_creation_includes_optional_recipients_without_message_id(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if request.url.path.endswith("/messages/original"):
            return httpx.Response(
                200,
                json=_message(
                    headers=[
                        {"name": "From", "value": "sender@example.com"},
                        {"name": "To", "value": "person@example.com"},
                        {"name": "Subject", "value": "Original"},
                    ]
                ),
                request=request,
            )
        if request.url.path.endswith("/drafts"):
            body = json.loads(request.content)
            raw = body["message"]["raw"]
            decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
            assert "Cc: copy@example.com" in decoded
            assert "Bcc: blind@example.com" in decoded
            assert "In-Reply-To:" not in decoded
            assert body["message"]["threadId"] == "thread-1"
            return httpx.Response(
                200,
                json={"id": "draft-1", "message": _message()},
                request=request,
            )
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        result = await service.create_draft(
            USER_ID,
            to="person@example.com",
            subject="Subject",
            body="body",
            cc="copy@example.com",
            bcc="blind@example.com",
            reply_to_message_id="original",
        )
        assert result["draft_id"] == "draft-1"
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_header", ["To", "Subject"])
async def test_draft_missing_recipient_or_subject_never_sends(
    tmp_path: Path,
    missing_header: str,
) -> None:
    headers: list[dict[str, str]] = [
        {"name": "From", "value": "sender@example.com"},
        {"name": "To", "value": "person@example.com"},
        {"name": "Subject", "value": "Subject"},
    ]
    headers = [header for header in headers if header["name"] != missing_header]
    send_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal send_calls
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if request.url.path.endswith("/drafts/draft-1"):
            return httpx.Response(
                200,
                json={"id": "draft-1", "message": _message(headers=headers)},
                request=request,
            )
        if request.url.path.endswith("/drafts/send"):
            send_calls += 1
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        with pytest.raises(GmailDraftChangedError):
            await service.send_draft(
                USER_ID,
                draft_id="draft-1",
                expected_to="person@example.com",
                expected_cc="",
                expected_bcc="",
                expected_subject="Subject",
                expected_content_sha256="0" * 64,
            )
        assert send_calls == 0
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_labels_and_input_bounds(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if request.url.path.endswith("/labels") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "labels": [{"id": "Label_1", "name": "Projects", "type": "user"}]
                },
                request=request,
            )
        if request.url.path.endswith("/labels"):
            return httpx.Response(
                200,
                json={"id": "Label_2", "name": "Later", "type": "user"},
                request=request,
            )
        if request.url.path.endswith("/messages/message-1") and request.method == "GET":
            return httpx.Response(200, json=_message(), request=request)
        if request.url.path.endswith("/messages/message-1/modify"):
            return httpx.Response(
                200,
                json={
                    "id": "message-1",
                    "threadId": "thread-1",
                    "labelIds": ["Label_1"],
                },
                request=request,
            )
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        assert (await service.list_labels(USER_ID))["labels"][0]["name"] == "Projects"
        assert (await service.create_label(USER_ID, name="Later"))["id"] == "Label_2"
        modified = await service.modify_message_labels(
            USER_ID,
            message_id="message-1",
            add_label_ids=["Label_1"],
            remove_label_ids=[],
        )
        assert modified["label_ids"] == ["Label_1"]
        with pytest.raises(GmailInputError):
            await service.search_messages(USER_ID, query="in:trash")
        with pytest.raises(GmailInputError):
            await service.modify_message_labels(
                USER_ID,
                message_id="message-1",
                add_label_ids=["TRASH"],
                remove_label_ids=[],
            )
        with pytest.raises(GmailInputError):
            await service.create_label(USER_ID, name="SENT")
        with pytest.raises(GmailInputError):
            await service.search_messages(USER_ID, query="x", max_results=101)
        with pytest.raises(GmailInputError):
            await service.search_messages(USER_ID, query="x", page_token=" ")
        with pytest.raises(GmailInputError):
            await service.api.modify_labels(
                USER_ID,
                resource="users",
                resource_id="message-1",
                add_label_ids=[],
                remove_label_ids=[],
            )
        with pytest.raises(GmailInputError):
            await service.api.modify_labels(
                USER_ID,
                resource="messages",
                resource_id="message-1",
                add_label_ids=["Label_1"],
                remove_label_ids=["Label_1"],
            )
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_refresh_is_serialized_and_invalid_grant_requires_reauth(
    tmp_path: Path,
) -> None:
    refresh_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_calls
        refresh_calls += 1
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={"access_token": "access", "expires_in": 3600},
            request=request,
        )

    client, storage, service, connection = await _ready(tmp_path, handler)
    try:
        tokens = await asyncio.gather(
            service.credentials.get_access_token(USER_ID),
            service.credentials.get_access_token(USER_ID),
        )
        assert list(tokens) == ["access", "access"]
        assert refresh_calls == 1

        async def invalid_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": "invalid_grant"},
                request=request,
            )

        await client.aclose()
        invalid_client = httpx.AsyncClient(
            transport=httpx.MockTransport(invalid_handler)
        )
        invalid_manager = GmailCredentialManager(
            service.config,
            storage,
            http_client=invalid_client,
        )
        invalid_manager.invalidate(USER_ID)
        with pytest.raises(GmailAuthorizationRequiredError):
            await invalid_manager.get_access_token(USER_ID)
        connection_row = await storage.get_connection(USER_ID)
        assert connection_row is not None
        assert connection_row.status == "reauthorization_required"
        await invalid_manager.close()
        await invalid_client.aclose()
    finally:
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_cached_access_token_tracks_current_connection(tmp_path: Path) -> None:
    refresh_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_calls
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            refresh_calls += 1
            return httpx.Response(
                200,
                json={
                    "access_token": f"access-{refresh_calls}",
                    "expires_in": 3600,
                },
                request=request,
            )
        raise AssertionError(request.url)

    client, storage, service, connection = await _ready(tmp_path, handler)
    try:
        assert await service.credentials.get_access_token(USER_ID) == "access-1"

        await connection.execute(
            "UPDATE gmail_connections SET encrypted_refresh_token = ?, "
            "status = 'connected' WHERE telegram_user_id = ?",
            (service.config.cipher.encrypt("replacement-refresh"), USER_ID),
        )
        await connection.commit()
        assert await service.credentials.get_access_token(USER_ID) == "access-2"
        assert refresh_calls == 2

        await connection.execute(
            "UPDATE gmail_connections SET status = 'reauthorization_required' "
            "WHERE telegram_user_id = ?",
            (USER_ID,),
        )
        await connection.commit()
        with pytest.raises(GmailAuthorizationRequiredError):
            await service.credentials.get_access_token(USER_ID)
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_api_errors_retry_auth_and_malformed_responses(tmp_path: Path) -> None:
    api_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal api_calls
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": f"access-{api_calls}", "expires_in": 3600},
                request=request,
            )
        api_calls += 1
        if api_calls == 1:
            return httpx.Response(
                401, json={"error": {"status": "UNAUTHENTICATED"}}, request=request
            )
        return httpx.Response(
            429,
            headers={"Retry-After": "2"},
            json={"error": {"status": "RATE_LIMITED"}},
            request=request,
        )

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        with pytest.raises(GmailRateLimitError) as error:
            await service.search_messages(USER_ID, query="x")
        assert error.value.retry_after_seconds == 2
        assert api_calls == 2

        async def malformed_handler(request: httpx.Request) -> httpx.Response:
            if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
                return httpx.Response(
                    200,
                    json={"access_token": "access", "expires_in": 3600},
                    request=request,
                )
            return httpx.Response(200, json={"messages": "not-a-list"}, request=request)

        await client.aclose()
        malformed_client = httpx.AsyncClient(
            transport=httpx.MockTransport(malformed_handler)
        )
        malformed_service = GmailService(
            service.config, service.storage, http_client=malformed_client
        )
        with pytest.raises(GmailMalformedResponseError):
            await malformed_service.search_messages(USER_ID, query="x")
        await malformed_service.close()
        await malformed_client.aclose()
    finally:
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_body_and_token_scope_limits(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        return httpx.Response(
            200,
            json=_message(body="x" * (MAX_MESSAGE_BODY_BYTES + 1)),
            request=request,
        )

    client, storage, service, connection = await _ready(tmp_path, handler)
    try:
        with pytest.raises(GmailInputError):
            await service.get_message(USER_ID, message_id="message-1")
        await storage.mark_reauthorization_required(USER_ID)
        service.credentials.invalidate(USER_ID)
        with pytest.raises(GmailAuthorizationRequiredError):
            await service.credentials.get_access_token(USER_ID)
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_drafts_and_thread_label_operations(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if request.url.path.endswith("/drafts") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "drafts": [{"id": "draft-1", "message": {"id": "message-1"}}],
                    "nextPageToken": "next",
                    "resultSizeEstimate": 1,
                },
                request=request,
            )
        if request.url.path.endswith("/drafts/draft-1"):
            return httpx.Response(
                200,
                json={"id": "draft-1", "message": _message()},
                request=request,
            )
        if request.url.path.endswith("/drafts") and request.method == "POST":
            return httpx.Response(
                200,
                json={"id": "draft-2", "message": _message()},
                request=request,
            )
        if request.url.path.endswith("/drafts/send"):
            return httpx.Response(204, request=request)
        if request.url.path.endswith("/threads/thread-1") and request.method == "GET":
            return httpx.Response(
                200,
                json={"id": "thread-1", "messages": [_message()]},
                request=request,
            )
        if request.url.path.endswith("/threads/thread-1/modify"):
            return httpx.Response(204, request=request)
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        drafts = await service.list_drafts(USER_ID, max_results=2, page_token="next-1")
        assert drafts == {
            "drafts": [{"id": "draft-1", "message": {"id": "message-1"}}],
            "next_page_token": "next",
            "result_size_estimate": 1,
        }
        draft = await service.get_draft(USER_ID, draft_id="draft-1")
        assert draft["draft_id"] == "draft-1"
        created = await service.api.create_draft(
            USER_ID,
            raw_message="raw-message",
        )
        assert created["draft_id"] == "draft-2"
        sent = await service.api.send_draft(USER_ID, draft_id="draft-1")
        assert sent == {"sent": True, "draft_id": "draft-1"}
        modified = await service.modify_thread_labels(
            USER_ID,
            thread_id="thread-1",
            add_label_ids=["Label_1"],
            remove_label_ids=[],
        )
        assert modified == {"id": "thread-1", "label_ids": []}
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_spam_and_trash_messages_are_not_read_or_modified(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/messages/spam-message"):
            message = _message("spam-message")
            message["labelIds"] = ["SPAM"]
            return httpx.Response(200, json=message, request=request)
        if request.url.path.endswith("/threads/trash-thread"):
            message = _message("trash-message", "trash-thread")
            message["labelIds"] = ["TRASH"]
            return httpx.Response(
                200,
                json={"id": "trash-thread", "messages": [message]},
                request=request,
            )
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        with pytest.raises(GmailAccessDeniedError):
            await service.get_message(USER_ID, message_id="spam-message")
        with pytest.raises(GmailAccessDeniedError):
            await service.modify_message_labels(
                USER_ID,
                message_id="spam-message",
                add_label_ids=["Label_1"],
                remove_label_ids=[],
            )
        with pytest.raises(GmailAccessDeniedError):
            await service.get_thread(USER_ID, thread_id="trash-thread")
        with pytest.raises(GmailAccessDeniedError):
            await service.modify_thread_labels(
                USER_ID,
                thread_id="trash-thread",
                add_label_ids=["Label_1"],
                remove_label_ids=[],
            )
        assert all(method == "GET" for method, _ in calls)
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_send_draft_normalizes_returned_message(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if request.url.path.endswith("/drafts/send"):
            return httpx.Response(
                200, json=_message(message_id="sent-1"), request=request
            )
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        result = await service.api.send_draft(USER_ID, draft_id="draft-1")
        assert result["message"]["id"] == "sent-1"
        assert result["sent"] is True
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_api_request_handles_empty_transport_and_auth_failures(
    tmp_path: Path,
) -> None:
    mode = "empty"
    api_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal api_calls
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if mode == "transport":
            raise httpx.ConnectError("network unavailable", request=request)
        api_calls += 1
        if mode == "auth":
            return httpx.Response(401, request=request)
        if request.url.path.endswith("/no-content"):
            return httpx.Response(204, request=request)
        return httpx.Response(200, content=b"", request=request)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        assert await service.api._request(USER_ID, "GET", "empty") == {}
        assert await service.api._request(USER_ID, "GET", "no-content") == {}

        mode = "transport"
        transport_service = GmailService(
            service.config, service.storage, http_client=client
        )
        with pytest.raises(GmailTransportError):
            await transport_service.api._request(USER_ID, "GET", "transport")
        await transport_service.close()

        mode = "auth"
        auth_service = GmailService(service.config, service.storage, http_client=client)
        with pytest.raises(GmailAuthError):
            await auth_service.api._request(USER_ID, "GET", "auth")
        assert api_calls >= 4
        await auth_service.close()
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_gmail_revocation_handles_transport_failure(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("revocation unavailable", request=request)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        assert await service.api.revoke("refresh") is False
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_gmail_clients_create_and_close_owned_http_clients(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "access", "expires_in": 3600},
            request=request,
        )

    client, storage, service, connection = await _ready(tmp_path, handler)
    del client
    try:
        manager = GmailCredentialManager(service.config, storage)
        owned_manager_client = await manager._client()
        await manager.close()
        assert owned_manager_client.is_closed is True

        api_credentials = GmailCredentialManager(service.config, storage)
        api = GmailApiClient(api_credentials)
        owned_api_client = await api._client()
        await api.close()
        assert owned_api_client.is_closed is True
        await api_credentials.close()
    finally:
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_credential_manager_rejects_invalid_connection_metadata_and_scopes(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "access", "expires_in": 3600},
            request=request,
        )

    client, storage, service, connection = await _ready(tmp_path, handler)
    try:
        missing = GmailCredentialManager(service.config, storage, http_client=client)
        with pytest.raises(GmailCredentialError):
            await missing.get_access_token("telegram-chat-43")

        await storage.mark_reauthorization_required(USER_ID)
        missing.invalidate(USER_ID)
        with pytest.raises(GmailAuthorizationRequiredError):
            await missing.get_access_token(USER_ID)

        await storage.remove_connection(USER_ID)
        await storage.save_connection(
            telegram_user_id=USER_ID,
            encrypted_refresh_token=service.config.cipher.encrypt("refresh"),
            scopes=(GMAIL_SCOPE,),
        )
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_credential_manager_handles_missing_scope_bad_ciphertext_and_rotation(
    tmp_path: Path,
) -> None:
    token_response: dict[str, Any] = {
        "access_token": "access",
        "expires_in": 3600,
        "scope": GMAIL_SCOPE,
        "refresh_token": "rotated-refresh",
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=token_response, request=request)

    client, storage, service, connection = await _ready(tmp_path, handler)
    try:
        await connection.execute(
            "UPDATE gmail_connections SET scopes_json = '[]' "
            "WHERE telegram_user_id = ?",
            (USER_ID,),
        )
        await connection.commit()
        with pytest.raises(GmailMissingScopeError):
            await service.credentials.get_access_token(USER_ID)
        stored = await storage.get_connection(USER_ID)
        assert stored is not None
        assert stored.status == "reauthorization_required"

        await connection.execute(
            "UPDATE gmail_connections SET encrypted_refresh_token = '', "
            "status = 'connected' WHERE telegram_user_id = ?",
            (USER_ID,),
        )
        await connection.commit()
        service.credentials.invalidate(USER_ID)
        with pytest.raises(GmailCredentialError):
            await service.credentials.get_access_token(USER_ID)

        await connection.execute(
            "UPDATE gmail_connections SET scopes_json = ?, "
            "encrypted_refresh_token = ?, status = 'connected' "
            "WHERE telegram_user_id = ?",
            (
                json.dumps([GMAIL_SCOPE]),
                service.config.cipher.encrypt("refresh"),
                USER_ID,
            ),
        )
        await connection.commit()
        service.credentials.invalidate(USER_ID)
        access_value = await service.credentials.get_access_token(USER_ID)
        assert access_value == "access"
        stored = await storage.get_connection(USER_ID)
        assert stored is not None
        assert service.config.cipher.decrypt(stored.encrypted_refresh_token) == (
            "rotated-refresh"
        )

        await connection.execute(
            "UPDATE gmail_connections SET encrypted_refresh_token = ?, "
            "status = 'connected' WHERE telegram_user_id = ?",
            ("not-a-valid-ciphertext", USER_ID),
        )
        await connection.commit()
        service.credentials.invalidate(USER_ID)
        with pytest.raises(GmailAuthorizationRequiredError):
            await service.credentials.get_access_token(USER_ID)
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_credential_refresh_maps_transport_rate_limit_and_malformed_results(
    tmp_path: Path,
) -> None:
    mode = "transport"

    async def handler(request: httpx.Request) -> httpx.Response:
        if mode == "transport":
            raise httpx.ConnectError("token endpoint unavailable", request=request)
        if mode == "rate_limit":
            return httpx.Response(
                429,
                headers={"Retry-After": "3"},
                json={"error": "slow_down"},
                request=request,
            )
        if mode == "api_error":
            return httpx.Response(500, json={"error": "backend"}, request=request)
        return httpx.Response(200, json={"expires_in": 3600}, request=request)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        with pytest.raises(GmailTransportError):
            await service.credentials._refresh("refresh")
        mode = "rate_limit"
        with pytest.raises(GmailRateLimitError) as rate_limit:
            await service.credentials._refresh("refresh")
        assert rate_limit.value.retry_after_seconds == 3
        mode = "api_error"
        with pytest.raises(GmailApiError):
            await service.credentials._refresh("refresh")
        mode = "malformed"
        with pytest.raises(GmailMalformedResponseError):
            await service.credentials._refresh("refresh")
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_credential_manager_requires_scope_returned_by_google(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "expires_in": 3600,
                "scope": "other.scope",
            },
            request=request,
        )

    client, storage, service, connection = await _ready(tmp_path, handler)
    try:
        with pytest.raises(GmailMissingScopeError):
            await service.credentials.get_access_token(USER_ID)
        stored = await storage.get_connection(USER_ID)
        assert stored is not None
        assert stored.status == "reauthorization_required"
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_exchange_code_rejects_bad_provider_results_and_transport_errors() -> (
    None
):
    config = _config()
    mode = "transport"

    async def handler(request: httpx.Request) -> httpx.Response:
        if mode == "transport":
            raise httpx.ConnectError("oauth unavailable", request=request)
        if mode == "invalid_grant":
            return httpx.Response(400, json={"error": "invalid_grant"}, request=request)
        if mode == "api_error":
            return httpx.Response(503, json={"error": "unavailable"}, request=request)
        if mode == "bad_json":
            return httpx.Response(200, content=b"not-json", request=request)
        if mode == "no_access":
            return httpx.Response(
                200, json={"refresh_token": "refresh"}, request=request
            )
        if mode == "no_refresh":
            return httpx.Response(200, json={"access_token": "access"}, request=request)
        if mode == "missing_scope":
            return httpx.Response(
                200,
                json={"access_token": "access", "refresh_token": "refresh"},
                request=request,
            )
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
    try:
        with pytest.raises(GmailInputError):
            await exchange_code_for_tokens(code="", config=config, http_client=client)
        with pytest.raises(GmailTransportError):
            await exchange_code_for_tokens(
                code="code", config=config, http_client=client
            )
        mode = "invalid_grant"
        with pytest.raises(GmailAuthorizationRequiredError):
            await exchange_code_for_tokens(
                code="code", config=config, http_client=client
            )
        mode = "api_error"
        with pytest.raises(GmailApiError):
            await exchange_code_for_tokens(
                code="code", config=config, http_client=client
            )
        mode = "bad_json"
        with pytest.raises(GmailMalformedResponseError):
            await exchange_code_for_tokens(
                code="code", config=config, http_client=client
            )
        mode = "no_access"
        with pytest.raises(GmailMalformedResponseError):
            await exchange_code_for_tokens(
                code="code", config=config, http_client=client
            )
        mode = "no_refresh"
        with pytest.raises(GmailCredentialError):
            await exchange_code_for_tokens(
                code="code", config=config, http_client=client
            )
        mode = "missing_scope"
        with pytest.raises(GmailMissingScopeError):
            await exchange_code_for_tokens(
                code="code", config=config, http_client=client
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_exchange_code_closes_an_owned_client() -> None:
    config = _config()

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

    owned_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("blacki.gmail.client.httpx.AsyncClient", return_value=owned_client):
        result = await exchange_code_for_tokens(code="code", config=config)
    expected_access = "access"
    assert result["access_token"] == expected_access
    assert owned_client.is_closed is True


def test_gmail_response_helpers_and_validation_bounds() -> None:
    assert _headers(None) == {}
    with pytest.raises(GmailMalformedResponseError):
        _headers("not-a-list")
    assert _header_map({"headers": {"Subject": "Hello", 1: "ignored"}}) == {
        "subject": "Hello"
    }
    assert _header_map({"headers": ["not-a-map"]}) == {}
    assert _optional_string(" value ") == "value"
    assert _optional_string(" ") is None
    assert _optional_string(1) is None
    assert _optional_int(None) is None
    assert _optional_int(2.9) == 2
    with pytest.raises(GmailMalformedResponseError):
        _optional_int("2")
    assert _positive_expiry(2) == 2.0
    with pytest.raises(GmailMalformedResponseError):
        _positive_expiry(0)
    assert _scope_set(None) == set()
    assert _scope_set(f"{GMAIL_SCOPE} other") == {GMAIL_SCOPE, "other"}
    with pytest.raises(GmailMalformedResponseError):
        _scope_set([GMAIL_SCOPE])
    assert _object({"id": "one"}, "object") == {"id": "one"}
    with pytest.raises(GmailMalformedResponseError):
        _object([], "object")
    assert _object_list(None, "items", "objects") == []
    with pytest.raises(GmailMalformedResponseError):
        _object_list({}, "items", "objects")
    assert _string_list(["one"], "items") == ["one"]
    with pytest.raises(GmailMalformedResponseError):
        _string_list([1], "items")
    assert _required_string({"id": " value "}, "id") == "value"
    with pytest.raises(GmailMalformedResponseError):
        _required_string({}, "id")
    with pytest.raises(GmailInputError):
        _decode_body("x" * (MAX_MESSAGE_BODY_BYTES * 2))
    with pytest.raises(GmailMalformedResponseError):
        _decode_body("not-base64!")
    with pytest.raises(GmailAccessDeniedError):
        _require_user_id("local")
    normalized = _normalize_message(
        _message(
            extra_parts=[
                {
                    "mimeType": "text/html",
                    "filename": "",
                    "body": {"data": _encoded("<p>hello</p>"), "size": 12},
                    "parts": None,
                }
            ]
        )
    )
    assert normalized["html_body"] == "<p>hello</p>"


def test_gmail_http_error_and_input_helpers_are_bounded() -> None:
    invalid_json_response = httpx.Response(500, content=b"bad")
    assert _response_error_code(invalid_json_response) is None
    assert _response_error_code(httpx.Response(500, json=[])) is None
    assert (
        _response_error_code(httpx.Response(500, json={"error": {"status": "STATUS"}}))
        == "STATUS"
    )
    assert _response_error_code(httpx.Response(500, json={"error": "backend"})) == (
        "backend"
    )
    assert _response_error_code(httpx.Response(500, json={"error": {}})) is None
    assert _retry_after(httpx.Response(429)) is None
    assert _retry_after(httpx.Response(429, headers={"Retry-After": "bad"})) is None
    assert _retry_after(httpx.Response(429, headers={"Retry-After": "999999"})) is None
    assert _retry_after(httpx.Response(429, headers={"Retry-After": "1.5"})) == 1.5
    with pytest.raises(GmailMalformedResponseError):
        _response_object(httpx.Response(200, content=b"bad"), "Gmail")
    with pytest.raises(GmailMalformedResponseError):
        _response_object(httpx.Response(200, json=[]), "Gmail")

    with pytest.raises(GmailAuthError):
        _raise_api_error(httpx.Response(401, json={"error": "auth"}))
    with pytest.raises(GmailApiError):
        _raise_api_error(httpx.Response(500, json={"error": "server"}))
    assert safe_provider_error_code(" safe_code-1") == "safe_code-1"
    assert safe_provider_error_code(1) is None
    assert safe_provider_error_code("café") is None
    assert safe_provider_error_code("bad code") is None
    assert safe_provider_error_code("x" * 81) is None

    params: dict[str, str] = {}
    _add_page_token(params, None)
    assert params == {}
    _add_page_token(params, "next")
    assert params == {"pageToken": "next"}
    with pytest.raises(GmailInputError):
        _add_page_token({}, " ")
    with pytest.raises(GmailInputError):
        _add_page_token({}, "x" * (MAX_PAGE_TOKEN_LENGTH + 1))
    with pytest.raises(GmailInputError):
        _validate_query("")
    with pytest.raises(GmailInputError):
        _validate_query("x\x00")
    with pytest.raises(GmailInputError):
        _validate_query("x" * (MAX_QUERY_LENGTH + 1))
    with pytest.raises(GmailInputError):
        _validate_count(0, 10, "count")
    with pytest.raises(GmailInputError):
        _validate_resource_id("bad id", "message_id")
    with pytest.raises(GmailInputError):
        _validate_subject("")
    with pytest.raises(GmailInputError):
        _validate_body("x" * (MAX_BODY_LENGTH + 1))
    with pytest.raises(GmailInputError):
        _parse_recipients(None, field_name="to", required=True)
    assert _parse_recipients(None, field_name="cc", required=False) == ()
    with pytest.raises(GmailInputError):
        _parse_recipients("bad address", field_name="to", required=True)
    too_many_recipients = ", ".join(
        f"person-{index}@example.com" for index in range(MAX_RECIPIENTS + 1)
    )
    with pytest.raises(GmailInputError):
        _parse_recipients(too_many_recipients, field_name="to", required=True)
    with pytest.raises(GmailInputError):
        _parse_recipients("x\x00@y.example", field_name="to", required=True)
    with pytest.raises(GmailInputError):
        _validate_label_name("TRASH")
    with pytest.raises(GmailInputError):
        _validate_label_ids("Label_1", "labels")
    with pytest.raises(GmailInputError):
        _validate_label_ids(["Label_1"] * (MAX_LABEL_IDS + 1), "labels")
    with pytest.raises(GmailInputError):
        _validate_label_ids(["x" * (MAX_LABEL_ID_LENGTH + 1)], "labels")


def test_draft_match_requires_send_metadata() -> None:
    content_sha256 = "a" * 64
    message = {
        "headers": {"to": "person@example.com", "subject": "Subject"},
        "content_sha256": content_sha256,
    }
    assert (
        _draft_matches(
            message,
            expected_to="person@example.com",
            expected_cc="",
            expected_bcc="",
            expected_subject="Subject",
            expected_content_sha256=content_sha256,
        )
        is True
    )
    for missing in ("to", "subject"):
        incomplete: dict[str, Any] = {
            "headers": dict(cast(dict[str, str], message["headers"]))
        }
        del incomplete["headers"][missing]
        assert (
            _draft_matches(
                incomplete,
                expected_to="person@example.com",
                expected_cc="",
                expected_bcc="",
                expected_subject="Subject",
                expected_content_sha256=content_sha256,
            )
            is False
        )
    assert (
        _draft_matches(
            message,
            expected_to="person@example.com",
            expected_cc="",
            expected_bcc="",
            expected_subject="",
            expected_content_sha256=content_sha256,
        )
        is False
    )
    assert (
        _draft_matches(
            message,
            expected_to="person@example.com",
            expected_cc="",
            expected_bcc="",
            expected_subject="Subject",
            expected_content_sha256="b" * 64,
        )
        is False
    )


@pytest.mark.asyncio
async def test_external_attachment_is_decoded_from_gmail_endpoint(
    tmp_path: Path,
) -> None:
    attachment_bytes = b"pdf-data"

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if request.url.path.endswith("/messages/message-1"):
            return httpx.Response(
                200,
                json=_message(
                    extra_parts=[
                        {
                            "partId": "1",
                            "mimeType": "application/pdf",
                            "filename": "invoice.pdf",
                            "body": {
                                "attachmentId": "attachment-1",
                                "size": len(attachment_bytes),
                            },
                        }
                    ]
                ),
                request=request,
            )
        if request.url.path.endswith("/attachments/attachment-1"):
            assert request.method == "GET"
            return httpx.Response(
                200,
                json={
                    "data": base64.urlsafe_b64encode(attachment_bytes)
                    .decode("ascii")
                    .rstrip("=")
                },
                request=request,
            )
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        result = await service.download_attachment(
            USER_ID,
            message_id="message-1",
            part_id="1",
        )
        assert result == GmailAttachmentDownload(
            filename="invoice.pdf",
            mime_type="application/pdf",
            size_bytes=len(attachment_bytes),
            data=attachment_bytes,
        )
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_inline_attachment_does_not_call_attachment_endpoint(
    tmp_path: Path,
) -> None:
    attachment_bytes = b"inline"
    attachment_endpoint_called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attachment_endpoint_called
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if "/attachments/" in request.url.path:
            attachment_endpoint_called = True
            raise AssertionError(request.url)
        return httpx.Response(
            200,
            json=_message(
                extra_parts=[
                    {
                        "partId": "1",
                        "mimeType": "text/plain",
                        "filename": "note.txt",
                        "body": {
                            "data": base64.urlsafe_b64encode(attachment_bytes)
                            .decode("ascii")
                            .rstrip("="),
                            "size": len(attachment_bytes),
                        },
                    }
                ]
            ),
            request=request,
        )

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        result = await service.download_attachment(
            USER_ID,
            message_id="message-1",
            part_id="1",
        )
        assert result.data == attachment_bytes
        assert result.filename == "note.txt"
        assert attachment_endpoint_called is False
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("part", "expected_error"),
    [
        (
            {
                "partId": "1",
                "mimeType": "application/pdf",
                "filename": "invoice.pdf",
                "body": {"attachmentId": "attachment-1", "size": 4},
            },
            GmailMalformedResponseError,
        ),
        (
            {
                "partId": "1",
                "mimeType": "application/pdf",
                "filename": "invoice.pdf",
                "body": {"attachmentId": "attachment-1", "size": 3},
            },
            GmailMalformedResponseError,
        ),
    ],
)
async def test_attachment_download_rejects_bad_provider_data(
    tmp_path: Path,
    part: dict[str, Any],
    expected_error: type[Exception],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        if request.url.path.endswith("/messages/message-1"):
            return httpx.Response(
                200,
                json=_message(extra_parts=[part]),
                request=request,
            )
        if request.url.path.endswith("/attachments/attachment-1"):
            body = {"data": "not-base64!"}
            if part["body"]["size"] == 3:
                body = {"data": _encoded("four")}
            return httpx.Response(200, json=body, request=request)
        raise AssertionError(request.url)

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        with pytest.raises(expected_error):
            await service.download_attachment(
                USER_ID,
                message_id="message-1",
                part_id="1",
            )
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_attachment_download_rejects_missing_part_and_oversize(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        return httpx.Response(
            200,
            json=_message(
                extra_parts=[
                    {
                        "partId": "1",
                        "mimeType": "application/octet-stream",
                        "filename": "large.bin",
                        "body": {"data": _encoded("12345"), "size": 5},
                    }
                ]
            ),
            request=request,
        )

    config = GmailConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/gmail/callback",
        token_encryption_key=Fernet.generate_key().decode(),
        max_attachment_bytes=4,
    )
    client, _, service, connection = await _ready(tmp_path, handler, config=config)
    try:
        with pytest.raises(GmailInputError):
            await service.download_attachment(
                USER_ID,
                message_id="message-1",
                part_id="missing",
            )
        with pytest.raises(GmailInputError):
            await service.download_attachment(
                USER_ID,
                message_id="message-1",
                part_id="1",
            )
    finally:
        await _close(client, service, connection)


def test_attachment_data_decoder_rejects_empty_malformed_and_oversized_values() -> None:
    with pytest.raises(GmailMalformedResponseError):
        _decode_attachment_data(None, max_bytes=4)
    with pytest.raises(GmailMalformedResponseError):
        _decode_attachment_data("not-base64!", max_bytes=4)
    with pytest.raises(GmailInputError):
        _decode_attachment_data("A" * 100, max_bytes=4)
    with (
        patch("blacki.gmail.client.base64.b64decode", return_value=b""),
        pytest.raises(GmailMalformedResponseError),
    ):
        _decode_attachment_data("AA", max_bytes=4)
    with pytest.raises(GmailInputError):
        _decode_attachment_data(_encoded("12345"), max_bytes=4)


def test_mime_traversal_and_normalization_limits_are_bounded() -> None:
    root: dict[str, Any] = {"partId": "root", "body": {}}
    current = root
    for index in range(MAX_MIME_DEPTH + 1):
        child: dict[str, Any] = {"partId": str(index), "body": {}}
        current["parts"] = [child]
        current = child
    with pytest.raises(GmailMalformedResponseError):
        _find_message_part({"payload": root}, "missing")

    with pytest.raises(GmailMalformedResponseError):
        _find_message_part(
            {"payload": {"body": {}, "parts": [{}] * (MAX_MIME_PARTS + 1)}},
            "missing",
        )
    assert _find_message_part({"payload": {"body": {}}}, "missing") is None

    deep_message = _message()
    deep_message["payload"] = {"body": {}}
    current = deep_message["payload"]
    for index in range(MAX_MIME_DEPTH + 1):
        child = {"body": {}, "partId": str(index)}
        current["parts"] = [child]
        current = child
    with pytest.raises(GmailMalformedResponseError):
        _normalize_message(deep_message)

    too_many_parts = _message()
    too_many_parts["payload"]["parts"] = [
        {"body": {}, "partId": str(index)} for index in range(MAX_MIME_PARTS)
    ]
    with pytest.raises(GmailMalformedResponseError):
        _normalize_message(too_many_parts)

    too_many_attachments = _message()
    too_many_attachments["payload"]["parts"] = [
        {
            "body": {},
            "filename": f"file-{index}.bin",
            "partId": str(index),
        }
        for index in range(MAX_MIME_ATTACHMENTS + 1)
    ]
    with pytest.raises(GmailMalformedResponseError):
        _normalize_message(too_many_attachments)

    with pytest.raises(GmailMalformedResponseError):
        _find_message_part(
            {
                "payload": {
                    "body": {},
                    "parts": [{}] * MAX_MIME_PARTS,
                }
            },
            "missing",
        )
    assert (
        _find_message_part({"payload": {"body": {}, "parts": None}}, "missing") is None
    )


@pytest.mark.asyncio
async def test_refresh_fails_closed_when_rotated_credential_was_replaced(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "refresh_token": "rotated",
                "scope": GMAIL_SCOPE,
                "expires_in": 3600,
            },
            request=request,
        )

    client, storage, service, connection = await _ready(tmp_path, handler)
    replacement = AsyncMock(return_value=False)
    try:
        with (
            patch.object(storage, "replace_refresh_token", replacement),
            pytest.raises(GmailAuthorizationRequiredError),
        ):
            await service.credentials.get_access_token(USER_ID)
        replacement.assert_awaited_once()
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_disconnect_removes_credential_when_stored_ciphertext_is_bad(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "access", "expires_in": 3600},
            request=request,
        )

    client, storage, service, connection = await _ready(tmp_path, handler)
    await connection.execute(
        "UPDATE gmail_connections SET encrypted_refresh_token = ? "
        "WHERE telegram_user_id = ?",
        ("bad-ciphertext", USER_ID),
    )
    await connection.commit()
    try:
        assert await service.disconnect(USER_ID) is True
        assert await storage.get_connection(USER_ID) is None
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_download_attachment_accepts_missing_declared_size_and_rejects_negative(
    tmp_path: Path,
) -> None:
    mode = "missing-size"

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        body: dict[str, Any] = {"data": _encoded("data")}
        if mode == "negative":
            body["size"] = -1
        return httpx.Response(
            200,
            json=_message(
                extra_parts=[
                    {
                        "partId": "1",
                        "filename": "data.bin",
                        "mimeType": "application/octet-stream",
                        "body": body,
                    }
                ]
            ),
            request=request,
        )

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        result = await service.download_attachment(
            USER_ID,
            message_id="message-1",
            part_id="1",
        )
        assert result.data == b"data"
        mode = "negative"
        with pytest.raises(GmailMalformedResponseError):
            await service.download_attachment(
                USER_ID,
                message_id="message-1",
                part_id="1",
            )
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_modify_empty_thread_fails_before_post(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL("https://oauth2.googleapis.com/token"):
            return httpx.Response(
                200,
                json={"access_token": "access", "expires_in": 3600},
                request=request,
            )
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={"id": "thread-1", "messages": []},
            request=request,
        )

    client, _, service, connection = await _ready(tmp_path, handler)
    try:
        with pytest.raises(GmailMalformedResponseError, match="no messages"):
            await service.modify_thread_labels(
                USER_ID,
                thread_id="thread-1",
                add_label_ids=["Label_1"],
                remove_label_ids=[],
            )
    finally:
        await _close(client, service, connection)


@pytest.mark.asyncio
async def test_build_raw_message_without_reply_skips_original_lookup() -> None:
    service = MagicMock()
    raw, thread_id = await _build_raw_message(
        service,
        USER_ID,
        to="person@example.com",
        subject="Subject",
        body="body",
        cc=None,
        bcc=None,
        reply_to_message_id=None,
    )
    assert raw
    assert thread_id is None
    service.get_message.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_removes_connection_without_a_refresh_token() -> None:
    service = GmailService.__new__(GmailService)
    service.storage = MagicMock()
    service.credentials = MagicMock()
    service.config = _config()
    service.api = MagicMock()
    service.storage.get_connection = AsyncMock(
        return_value=SimpleNamespace(encrypted_refresh_token="")
    )
    service.storage.remove_connection = AsyncMock()

    assert await service.disconnect(USER_ID) is True
    service.api.revoke.assert_not_called()
    service.storage.remove_connection.assert_awaited_once_with(USER_ID)
    service.credentials.invalidate.assert_called_once_with(USER_ID)
