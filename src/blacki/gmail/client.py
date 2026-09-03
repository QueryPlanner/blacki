"""Direct, user-scoped Gmail REST API access."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import EmailMessage
from email.policy import SMTP
from email.utils import getaddresses
from typing import Any, cast

import httpx

from .config import (
    GMAIL_API_BASE_URL,
    GMAIL_MAX_ATTACHMENT_BYTES,
    GMAIL_REVOCATION_URL,
    GMAIL_SCOPE,
    GMAIL_TOKEN_URL,
    TOKEN_EXPIRY_GRACE_SECONDS,
    GmailConfig,
    canonical_gmail_user_id,
)
from .errors import (
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
from .storage import SqliteGmailStorage

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_SECONDS = 30.0
MAX_RESULTS = 100
MAX_THREAD_MESSAGES = 100
MAX_PAGE_TOKEN_LENGTH = 2048
MAX_QUERY_LENGTH = 512
MAX_MESSAGE_BODY_BYTES = 128 * 1024
MAX_RECIPIENTS = 50
MAX_SUBJECT_LENGTH = 998
MAX_BODY_LENGTH = MAX_MESSAGE_BODY_BYTES
MAX_LABEL_NAME_LENGTH = 225
MAX_LABEL_IDS = 50
MAX_LABEL_ID_LENGTH = 256
MAX_MIME_DEPTH = 32
MAX_MIME_PARTS = 1_000
MAX_MIME_ATTACHMENTS = 100

_FORBIDDEN_SYSTEM_LABELS = frozenset(
    {
        "ALL",
        "DRAFT",
        "INBOX",
        "SENT",
        "SPAM",
        "STARRED",
        "TRASH",
        "UNREAD",
        "IMPORTANT",
        "CATEGORY_PERSONAL",
        "CATEGORY_SOCIAL",
        "CATEGORY_PROMOTIONS",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
    }
)
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_TRASH_OR_SPAM_QUERY_PATTERN = re.compile(
    r"(?:^|[\s({])-?(?:in|label):(trash|spam)(?:$|[\s)}])", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class _CachedAccessToken:
    value: str
    expires_at: float
    credential_fingerprint: str


@dataclass(frozen=True, slots=True)
class GmailAttachmentDownload:
    """Validated attachment bytes held only while a sandbox write is pending."""

    filename: str
    mime_type: str
    size_bytes: int
    data: bytes


class GmailCredentialManager:
    """Resolve and refresh one encrypted Gmail connection per Telegram user."""

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
        self._access_tokens: dict[str, _CachedAccessToken] = {}
        self._refresh_locks: dict[str, asyncio.Lock] = {}
        self._refresh_locks_lock = asyncio.Lock()
        self._owns_http_client = http_client is None

    async def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        return self._http_client

    async def _lock_for_user(self, user_id: str) -> asyncio.Lock:
        async with self._refresh_locks_lock:
            return self._refresh_locks.setdefault(user_id, asyncio.Lock())

    async def get_access_token(self, telegram_user_id: str) -> str:
        """Return a cached token or refresh the user's encrypted credential."""
        user_id = _require_user_id(telegram_user_id)
        user_lock = await self._lock_for_user(user_id)
        async with user_lock:
            connection = await self.storage.get_connection(user_id)
            if connection is None:
                raise GmailCredentialError(
                    "Gmail is not connected for this private Telegram user"
                )
            if connection.status != "connected":
                raise GmailAuthorizationRequiredError(
                    "Gmail authorization is required again"
                )
            if not connection.encrypted_refresh_token:
                raise GmailCredentialError(
                    "Gmail is not connected for this private Telegram user"
                )
            if GMAIL_SCOPE not in connection.scopes:
                await self.storage.mark_reauthorization_required(user_id)
                raise GmailMissingScopeError(
                    "Gmail authorization is missing the required scope"
                )
            credential_fingerprint = _credential_fingerprint(
                connection.encrypted_refresh_token
            )
            now = time.time()
            cached = self._access_tokens.get(user_id)
            if (
                cached is not None
                and cached.credential_fingerprint == credential_fingerprint
                and now < cached.expires_at - TOKEN_EXPIRY_GRACE_SECONDS
            ):
                return cached.value

            try:
                refresh_token = self.config.cipher.decrypt(
                    connection.encrypted_refresh_token
                )
            except Exception as exc:
                await self.storage.mark_reauthorization_required(user_id)
                raise GmailAuthorizationRequiredError(
                    "Stored Gmail authorization is no longer usable"
                ) from exc

            try:
                result = await self._refresh(refresh_token)
            except GmailAuthorizationRequiredError:
                await self.storage.mark_reauthorization_required(user_id)
                raise

            access_token = _required_string(result, "access_token")
            expires_in = _positive_expiry(result.get("expires_in", 3600))
            returned_scope = _scope_set(result.get("scope"))
            if returned_scope and GMAIL_SCOPE not in returned_scope:
                await self.storage.mark_reauthorization_required(user_id)
                raise GmailMissingScopeError(
                    "Gmail authorization is missing the required scope"
                )

            rotated_refresh_token = result.get("refresh_token")
            if isinstance(rotated_refresh_token, str) and rotated_refresh_token:
                encrypted = self.config.cipher.encrypt(rotated_refresh_token)
                if not await self.storage.replace_refresh_token(
                    user_id,
                    encrypted,
                    expected_encrypted_refresh_token=connection.encrypted_refresh_token,
                ):
                    raise GmailAuthorizationRequiredError(
                        "Gmail authorization changed during token refresh"
                    )
                credential_fingerprint = _credential_fingerprint(encrypted)

            self._access_tokens[user_id] = _CachedAccessToken(
                value=access_token,
                expires_at=time.time() + expires_in,
                credential_fingerprint=credential_fingerprint,
            )
            return access_token

    async def _refresh(self, refresh_token: str) -> dict[str, Any]:
        client = await self._client()
        try:
            response = await client.post(
                GMAIL_TOKEN_URL,
                data={
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
        except httpx.RequestError as exc:
            raise GmailTransportError(
                "Gmail token refresh could not reach Google"
            ) from exc

        if response.status_code >= 400:
            error_code = _response_error_code(response)
            if error_code == "invalid_grant":
                raise GmailAuthorizationRequiredError(
                    "Gmail authorization has expired or was revoked"
                )
            if response.status_code == 429:
                raise GmailRateLimitError(
                    "Google is rate limiting Gmail authorization",
                    status_code=response.status_code,
                    error_code=error_code,
                    retry_after_seconds=_retry_after(response),
                )
            raise GmailApiError(
                "Gmail token refresh failed",
                status_code=response.status_code,
                error_code=error_code,
            )

        payload = _response_object(response, "Gmail token refresh")
        if not isinstance(payload.get("access_token"), str):
            raise GmailMalformedResponseError(
                "Google returned an invalid Gmail access token response"
            )
        return payload

    def invalidate(self, telegram_user_id: str) -> None:
        """Discard one user's cached access token."""
        user_id = _require_user_id(telegram_user_id)
        self._access_tokens.pop(user_id, None)

    async def close(self) -> None:
        """Close an internally-created HTTP client."""
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


class GmailApiClient:
    """Small Gmail REST client with safe response normalization and retry."""

    def __init__(
        self,
        credentials: GmailCredentialManager,
        *,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = GMAIL_API_BASE_URL,
    ) -> None:
        self.credentials = credentials
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self.base_url = base_url.rstrip("/")

    async def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        return self._http_client

    async def _request(
        self,
        telegram_user_id: str,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_id = _require_user_id(telegram_user_id)
        client = await self._client()
        url = f"{self.base_url}/{path.lstrip('/')}"

        attempt = 0
        while True:
            token = await self.credentials.get_access_token(user_id)
            try:
                response = await client.request(
                    method,
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    json=json_body,
                )
            except httpx.RequestError as exc:
                raise GmailTransportError(
                    "Gmail request could not reach Google"
                ) from exc

            if response.status_code == 401:
                if attempt == 0:
                    self.credentials.invalidate(user_id)
                    attempt = 1
                    continue
                raise GmailAuthError(
                    "Gmail authentication failed after one retry",
                    status_code=401,
                )
            if response.status_code >= 400:
                _raise_api_error(response)

            if response.status_code == 204 or not response.content:
                return {}
            return _response_object(response, "Gmail API")

    async def list_messages(
        self,
        telegram_user_id: str,
        *,
        query: str,
        max_results: int = 10,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        _validate_query(query)
        params = {
            "q": query,
            "maxResults": str(_validate_count(max_results, MAX_RESULTS, "max_results")),
            "includeSpamTrash": "false",
        }
        _add_page_token(params, page_token)
        payload = await self._request(
            telegram_user_id,
            "GET",
            "messages",
            params=params,
        )
        messages = _object_list(
            payload.get("messages"), "messages", "Gmail message list"
        )
        normalized = []
        for message in messages:
            item = _object(message, "Gmail message summary")
            normalized.append(
                {
                    "id": _required_string(item, "id"),
                    "thread_id": _required_string(item, "threadId"),
                }
            )
        return {
            "messages": normalized,
            "next_page_token": _optional_string(payload.get("nextPageToken")),
            "result_size_estimate": _optional_int(payload.get("resultSizeEstimate")),
        }

    async def get_message(
        self,
        telegram_user_id: str,
        *,
        message_id: str,
    ) -> dict[str, Any]:
        _validate_resource_id(message_id, "message_id")
        payload = await self.get_full_message(telegram_user_id, message_id=message_id)
        return _normalize_message(payload)

    async def get_full_message(
        self,
        telegram_user_id: str,
        *,
        message_id: str,
    ) -> dict[str, Any]:
        """Fetch one full raw Gmail message for trusted part inspection."""
        _validate_resource_id(message_id, "message_id")
        payload = await self._request(
            telegram_user_id,
            "GET",
            f"messages/{message_id}",
            params={"format": "full"},
        )
        _validate_message_for_access(payload)
        return payload

    async def get_attachment(
        self,
        telegram_user_id: str,
        *,
        message_id: str,
        attachment_id: str,
    ) -> dict[str, Any]:
        """Fetch one external Gmail attachment body."""
        _validate_resource_id(message_id, "message_id")
        _validate_resource_id(attachment_id, "attachment_id")
        return await self._request(
            telegram_user_id,
            "GET",
            f"messages/{message_id}/attachments/{attachment_id}",
        )

    async def get_thread(
        self,
        telegram_user_id: str,
        *,
        thread_id: str,
        max_messages: int = 25,
    ) -> dict[str, Any]:
        _validate_resource_id(thread_id, "thread_id")
        count = _validate_count(max_messages, MAX_THREAD_MESSAGES, "max_messages")
        payload = await self._request(
            telegram_user_id,
            "GET",
            f"threads/{thread_id}",
            params={"format": "full"},
        )
        raw_messages = _object_list(payload.get("messages"), "messages", "Gmail thread")
        normalized_messages: list[dict[str, Any]] = []
        for raw_message in raw_messages:
            message = _object(raw_message, "Gmail thread message")
            _validate_message_for_access(message)
            if len(normalized_messages) < count:
                normalized_messages.append(_normalize_message(message))
        return {
            "id": _required_string(payload, "id"),
            "history_id": _optional_string(payload.get("historyId")),
            "messages": normalized_messages,
        }

    async def list_drafts(
        self,
        telegram_user_id: str,
        *,
        max_results: int = 10,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        params = {
            "maxResults": str(_validate_count(max_results, MAX_RESULTS, "max_results")),
        }
        _add_page_token(params, page_token)
        payload = await self._request(
            telegram_user_id,
            "GET",
            "drafts",
            params=params,
        )
        drafts = _object_list(payload.get("drafts"), "drafts", "Gmail draft list")
        normalized = []
        for draft in drafts:
            item = _object(draft, "Gmail draft summary")
            message = _object(item.get("message", {}), "Gmail draft message")
            normalized.append(
                {
                    "id": _required_string(item, "id"),
                    "message": {"id": _optional_string(message.get("id"))},
                }
            )
        return {
            "drafts": normalized,
            "next_page_token": _optional_string(payload.get("nextPageToken")),
            "result_size_estimate": _optional_int(payload.get("resultSizeEstimate")),
        }

    async def get_draft(
        self,
        telegram_user_id: str,
        *,
        draft_id: str,
    ) -> dict[str, Any]:
        _validate_resource_id(draft_id, "draft_id")
        payload = await self._request(
            telegram_user_id,
            "GET",
            f"drafts/{draft_id}",
            params={"format": "full"},
        )
        return _normalize_draft(payload)

    async def create_draft(
        self,
        telegram_user_id: str,
        *,
        raw_message: str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"raw": raw_message}
        if thread_id is not None:
            _validate_resource_id(thread_id, "thread_id")
            message["threadId"] = thread_id
        payload = await self._request(
            telegram_user_id,
            "POST",
            "drafts",
            json_body={"message": message},
        )
        return _normalize_draft(payload)

    async def send_draft(
        self,
        telegram_user_id: str,
        *,
        draft_id: str,
    ) -> dict[str, Any]:
        _validate_resource_id(draft_id, "draft_id")
        payload = await self._request(
            telegram_user_id,
            "POST",
            "drafts/send",
            json_body={"id": draft_id},
        )
        if not payload:
            return {"sent": True, "draft_id": draft_id}
        if "payload" in payload:
            sent = _normalize_message(payload)
        else:
            sent = {
                "id": _required_string(payload, "id"),
                "thread_id": _optional_string(payload.get("threadId")),
                "label_ids": _string_list(payload.get("labelIds", []), "labelIds"),
            }
        return {"sent": True, "draft_id": draft_id, "message": sent}

    async def list_labels(self, telegram_user_id: str) -> dict[str, Any]:
        payload = await self._request(telegram_user_id, "GET", "labels")
        labels = _object_list(payload.get("labels"), "labels", "Gmail label list")
        normalized = []
        for raw_label in labels:
            label = _object(raw_label, "Gmail label")
            normalized.append(
                {
                    "id": _required_string(label, "id"),
                    "name": _required_string(label, "name"),
                    "type": _optional_string(label.get("type")),
                }
            )
        return {"labels": normalized}

    async def create_label(
        self,
        telegram_user_id: str,
        *,
        name: str,
    ) -> dict[str, Any]:
        _validate_label_name(name)
        payload = await self._request(
            telegram_user_id,
            "POST",
            "labels",
            json_body={
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
        )
        label = _object(payload, "Gmail created label")
        return {
            "id": _required_string(label, "id"),
            "name": _required_string(label, "name"),
            "type": _optional_string(label.get("type")),
        }

    async def modify_labels(
        self,
        telegram_user_id: str,
        *,
        resource: str,
        resource_id: str,
        add_label_ids: Sequence[str],
        remove_label_ids: Sequence[str],
    ) -> dict[str, Any]:
        if resource not in {"messages", "threads"}:
            raise GmailInputError("Gmail resource is invalid")
        _validate_resource_id(resource_id, f"{resource}_id")
        add = _validate_label_ids(add_label_ids, "add_label_ids")
        remove = _validate_label_ids(remove_label_ids, "remove_label_ids")
        if set(add).intersection(remove):
            raise GmailInputError("A Gmail label cannot be added and removed together")
        if resource == "messages":
            await self.get_full_message(
                telegram_user_id,
                message_id=resource_id,
            )
        else:
            current_thread = await self._request(
                telegram_user_id,
                "GET",
                f"threads/{resource_id}",
                params={"format": "full"},
            )
            raw_messages = _object_list(
                current_thread.get("messages"),
                "messages",
                "Gmail thread",
            )
            _required_string(current_thread, "id")
            if not raw_messages:
                raise GmailMalformedResponseError("Gmail thread contains no messages")
            for raw_message in raw_messages:
                _validate_message_for_access(
                    _object(raw_message, "Gmail thread message")
                )
        payload = await self._request(
            telegram_user_id,
            "POST",
            f"{resource}/{resource_id}/modify",
            json_body={"addLabelIds": add, "removeLabelIds": remove},
        )
        if not payload:
            return {"id": resource_id, "label_ids": []}
        item = _object(payload, "Gmail label modification")
        return {
            "id": _required_string(item, "id"),
            "thread_id": _optional_string(item.get("threadId")),
            "label_ids": _string_list(item.get("labelIds", []), "labelIds"),
        }

    async def revoke(self, refresh_token: str) -> bool:
        """Ask Google to revoke a refresh token without exposing its value."""
        client = await self._client()
        try:
            response = await client.post(
                GMAIL_REVOCATION_URL,
                data={"token": refresh_token},
            )
        except httpx.RequestError:
            logger.warning("Gmail remote revocation could not reach Google")
            return False
        if response.status_code in (200, 204, 400):
            return response.status_code in (200, 204)
        logger.warning(
            "Gmail remote revocation failed with status %s",
            response.status_code,
        )
        return False

    async def close(self) -> None:
        """Close an internally-created HTTP client."""
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


class GmailService:
    """Application service that keeps every Gmail operation user-scoped."""

    def __init__(
        self,
        config: GmailConfig,
        storage: SqliteGmailStorage,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.credentials = GmailCredentialManager(
            config,
            storage,
            http_client=http_client,
        )
        self.api = GmailApiClient(
            self.credentials,
            http_client=http_client,
        )

    async def search_messages(
        self,
        user_id: str,
        *,
        query: str,
        max_results: int = 10,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        return await self.api.list_messages(
            _require_user_id(user_id),
            query=query,
            max_results=max_results,
            page_token=page_token,
        )

    async def get_message(self, user_id: str, *, message_id: str) -> dict[str, Any]:
        return await self.api.get_message(
            _require_user_id(user_id),
            message_id=message_id,
        )

    async def download_attachment(
        self,
        user_id: str,
        *,
        message_id: str,
        part_id: str,
    ) -> GmailAttachmentDownload:
        """Fetch and validate one attachment without exposing its bytes."""
        user_id = _require_user_id(user_id)
        _validate_resource_id(message_id, "message_id")
        _validate_resource_id(part_id, "part_id")
        payload = await self.api.get_full_message(user_id, message_id=message_id)
        _required_string(payload, "id")
        _required_string(payload, "threadId")
        _object(payload.get("payload"), "Gmail message payload")
        part = _find_message_part(payload, part_id)
        if part is None:
            raise GmailInputError("Gmail attachment part was not found")

        filename = _optional_string(part.get("filename")) or "attachment"
        mime_type = _optional_string(part.get("mimeType")) or "application/octet-stream"
        body = _object(part.get("body", {}), "Gmail attachment part body")
        declared_size = _optional_int(body.get("size"))
        if declared_size is not None:
            if declared_size < 0:
                raise GmailMalformedResponseError("Gmail attachment size is invalid")
            if declared_size > self.config.max_attachment_bytes:
                raise GmailInputError("Gmail attachment exceeds the size limit")

        attachment_id = _optional_string(body.get("attachmentId"))
        if attachment_id:
            attachment_payload = await self.api.get_attachment(
                user_id,
                message_id=message_id,
                attachment_id=attachment_id,
            )
            encoded_data = attachment_payload.get("data")
        else:
            encoded_data = body.get("data")
        data = _decode_attachment_data(
            encoded_data,
            max_bytes=self.config.max_attachment_bytes,
        )
        if declared_size is not None and declared_size != len(data):
            raise GmailMalformedResponseError(
                "Gmail attachment size does not match its data"
            )
        return GmailAttachmentDownload(
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(data),
            data=data,
        )

    async def get_thread(
        self,
        user_id: str,
        *,
        thread_id: str,
        max_messages: int = 25,
    ) -> dict[str, Any]:
        return await self.api.get_thread(
            _require_user_id(user_id),
            thread_id=thread_id,
            max_messages=max_messages,
        )

    async def list_drafts(
        self,
        user_id: str,
        *,
        max_results: int = 10,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        return await self.api.list_drafts(
            _require_user_id(user_id),
            max_results=max_results,
            page_token=page_token,
        )

    async def get_draft(self, user_id: str, *, draft_id: str) -> dict[str, Any]:
        return await self.api.get_draft(
            _require_user_id(user_id),
            draft_id=draft_id,
        )

    async def create_draft(
        self,
        user_id: str,
        *,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        bcc: str | None = None,
        reply_to_message_id: str | None = None,
    ) -> dict[str, Any]:
        user_id = _require_user_id(user_id)
        raw_message, thread_id = await _build_raw_message(
            self,
            user_id,
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            reply_to_message_id=reply_to_message_id,
        )
        return await self.api.create_draft(
            user_id,
            raw_message=raw_message,
            thread_id=thread_id,
        )

    async def send_draft(
        self,
        user_id: str,
        *,
        draft_id: str,
        expected_to: str,
        expected_cc: str,
        expected_bcc: str,
        expected_subject: str,
        expected_content_sha256: str,
    ) -> dict[str, Any]:
        user_id = _require_user_id(user_id)
        draft = await self.api.get_draft(user_id, draft_id=draft_id)
        message = _object(draft.get("message"), "Gmail draft message")
        if not _draft_matches(
            message,
            expected_to=expected_to,
            expected_cc=expected_cc,
            expected_bcc=expected_bcc,
            expected_subject=expected_subject,
            expected_content_sha256=expected_content_sha256,
        ):
            raise GmailDraftChangedError(
                "The Gmail draft changed after confirmation; nothing was sent"
            )
        return await self.api.send_draft(user_id, draft_id=draft_id)

    async def list_labels(self, user_id: str) -> dict[str, Any]:
        return await self.api.list_labels(_require_user_id(user_id))

    async def create_label(self, user_id: str, *, name: str) -> dict[str, Any]:
        return await self.api.create_label(_require_user_id(user_id), name=name)

    async def modify_message_labels(
        self,
        user_id: str,
        *,
        message_id: str,
        add_label_ids: Sequence[str],
        remove_label_ids: Sequence[str],
    ) -> dict[str, Any]:
        return await self.api.modify_labels(
            _require_user_id(user_id),
            resource="messages",
            resource_id=message_id,
            add_label_ids=add_label_ids,
            remove_label_ids=remove_label_ids,
        )

    async def modify_thread_labels(
        self,
        user_id: str,
        *,
        thread_id: str,
        add_label_ids: Sequence[str],
        remove_label_ids: Sequence[str],
    ) -> dict[str, Any]:
        return await self.api.modify_labels(
            _require_user_id(user_id),
            resource="threads",
            resource_id=thread_id,
            add_label_ids=add_label_ids,
            remove_label_ids=remove_label_ids,
        )

    async def disconnect(self, user_id: str) -> bool:
        """Attempt remote revocation, then remove only this user's row."""
        user_id = _require_user_id(user_id)
        connection = await self.storage.get_connection(user_id)
        if connection is None:
            return False
        if connection.encrypted_refresh_token:
            try:
                refresh_token = self.config.cipher.decrypt(
                    connection.encrypted_refresh_token
                )
            except Exception:
                logger.warning(
                    "Stored Gmail credential could not be decrypted during disconnect"
                )
            else:
                await self.api.revoke(refresh_token)
        await self.storage.remove_connection(user_id)
        self.credentials.invalidate(user_id)
        return True

    async def close(self) -> None:
        """Release provider clients owned by this service."""
        await self.api.close()
        await self.credentials.close()


async def exchange_code_for_tokens(
    *,
    code: str,
    config: GmailConfig,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Exchange one authorization code and require the selected Gmail scope."""
    if not code or _CONTROL_CHARACTER_PATTERN.search(code):
        raise GmailInputError("Gmail authorization code is invalid")
    client = http_client
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
    try:
        try:
            response = await client.post(
                GMAIL_TOKEN_URL,
                data={
                    "client_id": config.client_id,
                    "client_secret": config.client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": config.redirect_uri,
                },
            )
        except httpx.RequestError as exc:
            raise GmailTransportError(
                "Gmail authorization could not reach Google"
            ) from exc
        if response.status_code >= 400:
            error_code = _response_error_code(response)
            if error_code == "invalid_grant":
                raise GmailAuthorizationRequiredError(
                    "Gmail authorization code is no longer valid"
                )
            raise GmailApiError(
                "Gmail authorization exchange failed",
                status_code=response.status_code,
                error_code=error_code,
            )
        payload = _response_object(response, "Gmail authorization exchange")
        access_token = payload.get("access_token")
        refresh_token = payload.get("refresh_token")
        if not isinstance(access_token, str) or not access_token:
            raise GmailMalformedResponseError(
                "Google did not return a Gmail access token"
            )
        if not isinstance(refresh_token, str) or not refresh_token:
            raise GmailCredentialError(
                "Google did not return a Gmail refresh token; reconnect Gmail"
            )
        scopes = _scope_set(payload.get("scope"))
        if GMAIL_SCOPE not in scopes:
            raise GmailMissingScopeError(
                "Google did not grant the required Gmail scope"
            )
        return payload
    finally:
        if owns_client:
            await client.aclose()


async def _build_raw_message(
    service: GmailService,
    user_id: str,
    *,
    to: str,
    subject: str,
    body: str,
    cc: str | None,
    bcc: str | None,
    reply_to_message_id: str | None,
) -> tuple[str, str | None]:
    recipients = _parse_recipients(to, field_name="to", required=True)
    cc_recipients = _parse_recipients(cc, field_name="cc", required=False)
    bcc_recipients = _parse_recipients(bcc, field_name="bcc", required=False)
    _validate_subject(subject)
    _validate_body(body)

    message = EmailMessage(policy=SMTP)
    message["To"] = ", ".join(recipients)
    if cc_recipients:
        message["Cc"] = ", ".join(cc_recipients)
    if bcc_recipients:
        message["Bcc"] = ", ".join(bcc_recipients)
    message["Subject"] = subject
    thread_id = None
    if reply_to_message_id:
        _validate_resource_id(reply_to_message_id, "reply_to_message_id")
        original = await service.get_message(user_id, message_id=reply_to_message_id)
        thread_id = _optional_string(original.get("thread_id"))
        original_headers = _header_map(original)
        original_message_id = original_headers.get("message-id")
        if original_message_id:
            message["In-Reply-To"] = original_message_id
            references = original_headers.get("references", "").strip()
            message["References"] = f"{references} {original_message_id}".strip()
    message.set_content(body)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
    return raw, thread_id


def _draft_matches(
    message: Mapping[str, Any],
    *,
    expected_to: str,
    expected_cc: str,
    expected_bcc: str,
    expected_subject: str,
    expected_content_sha256: str,
) -> bool:
    headers = _header_map(message)
    if (
        "to" not in headers
        or "subject" not in headers
        or not _is_sha256(expected_content_sha256)
        or message.get("content_sha256") != expected_content_sha256
    ):
        return False
    try:
        actual_to = _parse_recipients(
            headers.get("to"),
            field_name="to",
            required=False,
        )
        actual_cc = _parse_recipients(
            headers.get("cc"),
            field_name="cc",
            required=False,
        )
        actual_bcc = _parse_recipients(
            headers.get("bcc"),
            field_name="bcc",
            required=False,
        )
        confirmed_to = _parse_recipients(
            expected_to,
            field_name="expected_to",
            required=True,
        )
        confirmed_cc = _parse_recipients(
            expected_cc,
            field_name="expected_cc",
            required=False,
        )
        confirmed_bcc = _parse_recipients(
            expected_bcc,
            field_name="expected_bcc",
            required=False,
        )
        _validate_subject(expected_subject)
    except GmailInputError:
        return False
    return (
        actual_to == confirmed_to
        and actual_cc == confirmed_cc
        and actual_bcc == confirmed_bcc
        and headers["subject"].strip() == expected_subject.strip()
    )


def _normalize_message(payload: Mapping[str, Any]) -> dict[str, Any]:
    message_id = _required_string(payload, "id")
    thread_id = _required_string(payload, "threadId")
    raw_payload = _object(payload.get("payload"), "Gmail message payload")
    header_map = _headers(raw_payload.get("headers"))
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[dict[str, Any]] = []
    decoded_size = 0
    part_count = 0
    attachment_count = 0

    def visit(part: Mapping[str, Any], depth: int = 0) -> None:
        nonlocal attachment_count, decoded_size, part_count
        if depth > MAX_MIME_DEPTH:
            raise GmailMalformedResponseError(
                "Gmail message MIME structure is too deep"
            )
        part_count += 1
        if part_count > MAX_MIME_PARTS:
            raise GmailMalformedResponseError("Gmail message has too many MIME parts")
        mime_type = _optional_string(part.get("mimeType")) or ""
        filename = _optional_string(part.get("filename")) or ""
        part_id = _optional_string(part.get("partId"))
        body = _object(part.get("body", {}), "Gmail message part body")
        attachment_id = _optional_string(body.get("attachmentId"))
        size = _optional_int(body.get("size")) or 0
        if filename or attachment_id:
            attachment_count += 1
            if attachment_count > MAX_MIME_ATTACHMENTS:
                raise GmailMalformedResponseError(
                    "Gmail message has too many attachments"
                )
            attachments.append(
                {
                    "filename": filename,
                    "mime_type": mime_type,
                    "size": size,
                    "attachment_id": attachment_id,
                    "part_id": part_id,
                }
            )
        data = body.get("data")
        if isinstance(data, str) and data and mime_type in {"text/plain", "text/html"}:
            decoded = _decode_body(data)
            decoded_size += len(decoded)
            if decoded_size > MAX_MESSAGE_BODY_BYTES:
                raise GmailInputError("Gmail message body exceeds the size limit")
            text = decoded.decode("utf-8", errors="replace")
            if mime_type == "text/plain":
                text_parts.append(text)
            else:
                html_parts.append(text)
        raw_parts = part.get("parts", [])
        if raw_parts is not None:
            for child in _object_list(raw_parts, "parts", "Gmail message parts"):
                visit(_object(child, "Gmail message part"), depth + 1)

    visit(raw_payload)
    return {
        "id": message_id,
        "thread_id": thread_id,
        "label_ids": _string_list(payload.get("labelIds", []), "labelIds"),
        "snippet": _optional_string(payload.get("snippet")),
        "internal_date": _optional_string(payload.get("internalDate")),
        "headers": header_map,
        "from": header_map.get("from", ""),
        "to": header_map.get("to", ""),
        "cc": header_map.get("cc", ""),
        "bcc": header_map.get("bcc", ""),
        "subject": header_map.get("subject", ""),
        "date": header_map.get("date", ""),
        "text_body": "\n".join(text_parts),
        "html_body": "\n".join(html_parts),
        "attachments": attachments,
        "content_sha256": _message_content_fingerprint(raw_payload),
    }


def _normalize_draft(payload: Mapping[str, Any]) -> dict[str, Any]:
    draft_id = _required_string(payload, "id")
    message = _normalize_message(_object(payload.get("message"), "Gmail draft message"))
    return {
        "id": draft_id,
        "draft_id": draft_id,
        "message": message,
        "thread_id": message["thread_id"],
    }


def _headers(raw_headers: object) -> dict[str, str]:
    if raw_headers is None:
        return {}
    if not isinstance(raw_headers, list):
        raise GmailMalformedResponseError("Gmail message headers are malformed")
    result: dict[str, str] = {}
    for raw_header in raw_headers:
        if not isinstance(raw_header, Mapping):
            continue
        raw_name = raw_header.get("name")
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
            or _CONTROL_CHARACTER_PATTERN.search(raw_name)
        ):
            continue
        name = raw_name.strip().lower()
        raw_value = raw_header.get("value")
        value = raw_value if isinstance(raw_value, str) else ""
        result[name] = value
    return result


def _header_map(message: Mapping[str, Any]) -> dict[str, str]:
    raw_headers = message.get("headers")
    if isinstance(raw_headers, Mapping):
        return {
            str(key).strip().lower(): str(value)
            for key, value in raw_headers.items()
            if isinstance(key, str) and isinstance(value, str)
        }
    return {}


def _decode_body(value: str) -> bytes:
    if len(value) > ((MAX_MESSAGE_BODY_BYTES + 2) * 4 // 3 + 8):
        raise GmailInputError("Gmail message body exceeds the size limit")
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GmailMalformedResponseError(
            "Gmail message body encoding is malformed"
        ) from exc


def _decode_attachment_data(
    value: object,
    *,
    max_bytes: int = GMAIL_MAX_ATTACHMENT_BYTES,
) -> bytes:
    if not isinstance(value, str) or not value:
        raise GmailMalformedResponseError("Gmail attachment data is empty")
    if len(value) > ((max_bytes + 2) * 4 // 3 + 8):
        raise GmailInputError("Gmail attachment exceeds the size limit")
    try:
        padded = value + "=" * (-len(value) % 4)
        data = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GmailMalformedResponseError(
            "Gmail attachment encoding is malformed"
        ) from exc
    if not data:
        raise GmailMalformedResponseError("Gmail attachment data is empty")
    if len(data) > max_bytes:
        raise GmailInputError("Gmail attachment exceeds the size limit")
    return data


def _find_message_part(
    payload: Mapping[str, Any],
    part_id: str,
) -> Mapping[str, Any] | None:
    root = _object(payload.get("payload"), "Gmail message payload")
    pending: list[tuple[Mapping[str, Any], int]] = [(root, 0)]
    part_count = 0
    while pending:
        part, depth = pending.pop()
        if depth > MAX_MIME_DEPTH:
            raise GmailMalformedResponseError(
                "Gmail message MIME structure is too deep"
            )
        part_count += 1
        if part_count > MAX_MIME_PARTS:
            raise GmailMalformedResponseError("Gmail message has too many MIME parts")
        if part.get("partId") == part_id:
            return part
        raw_parts = part.get("parts", [])
        if raw_parts is None:
            continue
        children = _object_list(raw_parts, "parts", "Gmail message parts")
        if len(children) > MAX_MIME_PARTS:
            raise GmailMalformedResponseError("Gmail message has too many MIME parts")
        pending.extend(
            (_object(child, "Gmail message part"), depth + 1)
            for child in reversed(children)
        )
    return None


def _require_user_id(user_id: str) -> str:
    canonical = canonical_gmail_user_id(user_id)
    if canonical is None:
        raise GmailAccessDeniedError(
            "Gmail is available only to a private Telegram user"
        )
    return canonical


def _credential_fingerprint(encrypted_refresh_token: str) -> str:
    """Identify the stored credential without retaining or logging its value."""
    return hashlib.sha256(encrypted_refresh_token.encode("utf-8")).hexdigest()


def _message_content_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash the trusted MIME payload without returning its content."""
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ensure_not_spam_or_trash(payload: Mapping[str, Any]) -> None:
    labels = _string_list(payload.get("labelIds", []), "labelIds")
    if any(label.upper() in {"SPAM", "TRASH"} for label in labels):
        raise GmailAccessDeniedError("Gmail cannot access spam or trash")


def _validate_message_for_access(payload: Mapping[str, Any]) -> None:
    _required_string(payload, "id")
    _required_string(payload, "threadId")
    _object(payload.get("payload"), "Gmail message payload")
    _ensure_not_spam_or_trash(payload)


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GmailMalformedResponseError(f"Gmail response is missing {key}")
    return value.strip()


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GmailMalformedResponseError("Gmail response contains an invalid count")
    return int(value)


def _positive_expiry(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise GmailMalformedResponseError(
            "Gmail token response contains an invalid expiry"
        )
    return float(value)


def _scope_set(value: object) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, str):
        raise GmailMalformedResponseError(
            "Gmail token response contains invalid scopes"
        )
    return set(value.split())


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GmailMalformedResponseError(f"{label} is malformed")
    return cast(dict[str, Any], dict(value))


def _object_list(value: object, key: str, label: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GmailMalformedResponseError(f"{label} contains an invalid {key} list")
    return value


def _string_list(value: object, key: str) -> list[str]:
    raw = _object_list(value, key, "Gmail response")
    if not all(isinstance(item, str) for item in raw):
        raise GmailMalformedResponseError(f"Gmail response contains invalid {key}")
    return [cast(str, item) for item in raw]


def _response_object(response: httpx.Response, label: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise GmailMalformedResponseError(
            f"{label} response is not valid JSON"
        ) from exc
    return _object(payload, f"{label} response")


def _response_error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if isinstance(error, Mapping):
        return (
            safe_provider_error_code(error.get("status"))
            or safe_provider_error_code(error.get("code"))
            or safe_provider_error_code(error.get("error"))
        )
    return safe_provider_error_code(error)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0 <= value <= 86_400 else None


def _raise_api_error(response: httpx.Response) -> None:
    code = _response_error_code(response)
    if response.status_code == 401:
        raise GmailAuthError(
            "Gmail authentication failed",
            status_code=response.status_code,
            error_code=code,
        )
    if response.status_code == 429:
        raise GmailRateLimitError(
            "Google is rate limiting Gmail",
            status_code=response.status_code,
            error_code=code,
            retry_after_seconds=_retry_after(response),
        )
    raise GmailApiError(
        "Gmail request failed",
        status_code=response.status_code,
        error_code=code,
    )


def _validate_count(value: int, maximum: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise GmailInputError(f"{name} must be between 1 and {maximum}")
    return value


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
        raise GmailInputError("Gmail search query is empty or too long")
    if _CONTROL_CHARACTER_PATTERN.search(query):
        raise GmailInputError("Gmail search query contains a control character")
    if _TRASH_OR_SPAM_QUERY_PATTERN.search(query):
        raise GmailInputError("Gmail search cannot target spam or trash")


def _add_page_token(params: dict[str, str], page_token: str | None) -> None:
    if page_token is None:
        return
    if (
        not isinstance(page_token, str)
        or not page_token.strip()
        or len(page_token) > MAX_PAGE_TOKEN_LENGTH
        or _CONTROL_CHARACTER_PATTERN.search(page_token)
    ):
        raise GmailInputError("Gmail page token is invalid or too long")
    params["pageToken"] = page_token


def _validate_resource_id(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or _CONTROL_CHARACTER_PATTERN.search(value)
        or any(char.isspace() for char in value)
        or any(char in value for char in "/\\?#")
    ):
        raise GmailInputError(f"Gmail {name} is invalid")


def _validate_subject(subject: str) -> None:
    if (
        not isinstance(subject, str)
        or not subject.strip()
        or len(subject) > MAX_SUBJECT_LENGTH
        or _CONTROL_CHARACTER_PATTERN.search(subject)
    ):
        raise GmailInputError("Gmail subject is empty, too long, or invalid")


def _validate_body(body: str) -> None:
    if not isinstance(body, str) or len(body.encode("utf-8")) > MAX_BODY_LENGTH:
        raise GmailInputError("Gmail message body exceeds the size limit")


def _parse_recipients(
    value: str | None,
    *,
    field_name: str,
    required: bool,
) -> tuple[str, ...]:
    if value is None or not isinstance(value, str) or not value.strip():
        if required:
            raise GmailInputError(f"Gmail {field_name} requires at least one recipient")
        return ()
    if _CONTROL_CHARACTER_PATTERN.search(value):
        raise GmailInputError(f"Gmail {field_name} contains an invalid character")
    parsed = getaddresses([value])
    if not parsed or len(parsed) > MAX_RECIPIENTS:
        raise GmailInputError(f"Gmail {field_name} contains too many recipients")
    addresses: list[str] = []
    for _, address in parsed:
        normalized = address.strip().casefold()
        if (
            not normalized
            or "@" not in normalized
            or normalized.startswith("@")
            or normalized.endswith("@")
            or any(char.isspace() for char in normalized)
        ):
            raise GmailInputError(f"Gmail {field_name} contains an invalid recipient")
        addresses.append(normalized)
    return tuple(sorted(addresses))


def _validate_label_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > MAX_LABEL_NAME_LENGTH
        or _CONTROL_CHARACTER_PATTERN.search(name)
        or name.strip().upper() in _FORBIDDEN_SYSTEM_LABELS
    ):
        raise GmailInputError("Gmail label name is invalid or system-controlled")


def _validate_label_ids(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise GmailInputError(f"Gmail {name} must be a list of label IDs")
    if len(values) > MAX_LABEL_IDS:
        raise GmailInputError(f"Gmail {name} contains too many labels")
    result: list[str] = []
    for value in values:
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > MAX_LABEL_ID_LENGTH
            or _CONTROL_CHARACTER_PATTERN.search(value)
            or value.strip().upper() in _FORBIDDEN_SYSTEM_LABELS
            or value.strip().upper().startswith("CATEGORY_")
        ):
            raise GmailInputError(
                f"Gmail {name} contains a destructive or system-controlled label"
            )
        result.append(value.strip())
    return result
