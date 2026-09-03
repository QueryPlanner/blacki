"""Tests for Gmail ADK tool exposure and confirmation boundaries."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from google.adk.tools.tool_confirmation import ToolConfirmation

from blacki.gmail.client import GmailAttachmentDownload, GmailService
from blacki.gmail.config import GMAIL_SCOPE, GmailConfig
from blacki.gmail.errors import (
    GmailApiError,
    GmailConfigurationError,
    GmailCredentialError,
    GmailError,
    GmailInputError,
)
from blacki.gmail.storage import SqliteGmailStorage
from blacki.gmail.tools import (
    _ACTIVE_SERVICE,
    GmailToolset,
    _gmail_body_present,
    _gmail_body_size,
    _materialize_large_gmail_result,
    _service_for_context,
    _strip_gmail_bodies,
    _truncate_gmail_bodies,
    _truncate_gmail_result,
    create_gmail_toolset,
    gmail_create_draft,
    gmail_create_label,
    gmail_download_attachment,
    gmail_get_draft,
    gmail_get_message,
    gmail_get_thread,
    gmail_list_drafts,
    gmail_list_labels,
    gmail_modify_message_labels,
    gmail_modify_thread_labels,
    gmail_search_messages,
    gmail_send_draft,
)
from blacki.storage.sqlite import create_connection


def _config() -> GmailConfig:
    return GmailConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/gmail/callback",
        token_encryption_key=Fernet.generate_key().decode(),
    )


class _ReadonlyContext:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.invocation_id = "invocation-1"
        self.state: dict[str, Any] = {}


class _ToolContext:
    def __init__(
        self,
        user_id: str,
        confirmation: ToolConfirmation | None = None,
    ) -> None:
        self.user_id = user_id
        self.tool_confirmation = confirmation
        self.state: dict[str, Any] = {}
        self.actions = SimpleNamespace(skip_summarization=False)
        self.confirmation_requested = False

    def request_confirmation(self, **_: Any) -> None:
        self.confirmation_requested = True


async def _ready(tmp_path: Path) -> tuple[Any, SqliteGmailStorage, Any]:
    connection = await create_connection(tmp_path / "tools.db")
    storage = SqliteGmailStorage(connection, asyncio.Lock())
    await storage.initialize()
    config = _config()
    await storage.save_connection(
        telegram_user_id="telegram-chat-42",
        encrypted_refresh_token=config.cipher.encrypt("refresh"),
        scopes=(GMAIL_SCOPE,),
    )
    service: Any = GmailService(_config(), storage)
    service.search_messages = AsyncMock(return_value={"messages": []})
    service.send_draft = AsyncMock(return_value={"sent": True})
    return connection, storage, service


@pytest.mark.asyncio
async def test_toolset_filters_private_connected_users_and_allows_topic_identity(
    tmp_path: Path,
) -> None:
    connection, storage, service = await _ready(tmp_path)
    toolset = GmailToolset(config=_config(), storage=storage, service=service)
    try:
        connected = await toolset.get_tools(
            _ReadonlyContext("telegram-chat-42-thread-9")  # type: ignore[arg-type]
        )
        assert {tool.name for tool in connected} == {
            "gmail_search_messages",
            "gmail_get_message",
            "gmail_get_thread",
            "gmail_download_attachment",
            "gmail_list_drafts",
            "gmail_get_draft",
            "gmail_create_draft",
            "gmail_send_draft",
            "gmail_list_labels",
            "gmail_create_label",
            "gmail_modify_message_labels",
            "gmail_modify_thread_labels",
        }

        assert await toolset.get_tools(_ReadonlyContext("telegram-chat-43")) == []  # type: ignore[arg-type]
        assert await toolset.get_tools(_ReadonlyContext("telegram-chat--100")) == []  # type: ignore[arg-type]
        assert await toolset.get_tools(_ReadonlyContext("local")) == []  # type: ignore[arg-type]
        assert await toolset.get_tools(None) == []
        assert toolset._use_invocation_cache is False
    finally:
        await toolset.close()
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_tool_calls_use_context_user_and_send_pauses_for_confirmation(
    tmp_path: Path,
) -> None:
    connection, storage, service = await _ready(tmp_path)
    toolset = GmailToolset(config=_config(), storage=storage, service=service)
    try:
        tools = await toolset.get_tools(
            _ReadonlyContext("telegram-chat-42")  # type: ignore[arg-type]
        )
        search = next(tool for tool in tools if tool.name == "gmail_search_messages")
        result = await search.run_async(
            args={"query": "invoices"},
            tool_context=_ToolContext("telegram-chat-42-thread-1"),  # type: ignore[arg-type]
        )
        assert result == {"messages": []}
        service.search_messages.assert_awaited_once_with(
            "telegram-chat-42",
            query="invoices",
            max_results=10,
            page_token=None,
        )

        send = next(tool for tool in tools if tool.name == "gmail_send_draft")
        pending_context = _ToolContext("telegram-chat-42")
        pending = await send.run_async(
            args={
                "draft_id": "draft-1",
                "expected_to": "person@example.com",
                "expected_cc": "",
                "expected_bcc": "",
                "expected_subject": "Subject",
                "expected_content_sha256": "a" * 64,
            },
            tool_context=pending_context,  # type: ignore[arg-type]
        )
        assert "requires confirmation" in pending["error"]
        assert pending_context.confirmation_requested is True
        service.send_draft.assert_not_awaited()

        approved = _ToolContext(
            "telegram-chat-42",
            ToolConfirmation(confirmed=True),
        )
        result = await send.run_async(
            args={
                "draft_id": "draft-1",
                "expected_to": "person@example.com",
                "expected_cc": "",
                "expected_bcc": "",
                "expected_subject": "Subject",
                "expected_content_sha256": "a" * 64,
            },
            tool_context=approved,  # type: ignore[arg-type]
        )
        assert result == {"sent": True}
        service.send_draft.assert_awaited_once_with(
            "telegram-chat-42",
            draft_id="draft-1",
            expected_to="person@example.com",
            expected_cc="",
            expected_bcc="",
            expected_subject="Subject",
            expected_content_sha256="a" * 64,
        )
    finally:
        await toolset.close()
        await service.close()
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        GmailInputError("Gmail query is invalid"),
        GmailApiError("Gmail request failed", error_code="backendError"),
    ],
)
async def test_gmail_tool_returns_expected_errors_safely(
    tmp_path: Path,
    error: Exception,
) -> None:
    connection, storage, service = await _ready(tmp_path)
    service.search_messages.side_effect = error
    toolset = GmailToolset(config=_config(), storage=storage, service=service)
    try:
        search = next(
            tool
            for tool in await toolset.get_tools(
                _ReadonlyContext("telegram-chat-42")  # type: ignore[arg-type]
            )
            if tool.name == "gmail_search_messages"
        )
        result = await search.run_async(
            args={"query": "invoices"},
            tool_context=_ToolContext("telegram-chat-42"),  # type: ignore[arg-type]
        )
        assert result["status"] == "error"
        assert result["error"] == str(error)
        if isinstance(error, GmailApiError):
            assert result["error_code"] == "backendError"
        assert _ACTIVE_SERVICE.get() is None
    finally:
        await toolset.close()
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_gmail_tool_converts_unexpected_errors_without_logging_content(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection, storage, service = await _ready(tmp_path)
    service.search_messages.side_effect = RuntimeError("private email body")
    toolset = GmailToolset(config=_config(), storage=storage, service=service)
    caplog.set_level(logging.ERROR, logger="blacki.gmail.tools")
    try:
        search = next(
            tool
            for tool in await toolset.get_tools(
                _ReadonlyContext("telegram-chat-42")  # type: ignore[arg-type]
            )
            if tool.name == "gmail_search_messages"
        )
        result = await search.run_async(
            args={"query": "invoices"},
            tool_context=_ToolContext("telegram-chat-42"),  # type: ignore[arg-type]
        )
        assert result == {
            "status": "error",
            "error": "Gmail operation failed unexpectedly",
        }
        assert "RuntimeError" in caplog.text
        assert "private email body" not in caplog.text
        assert _ACTIVE_SERVICE.get() is None
    finally:
        await toolset.close()
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_large_gmail_results_use_the_session_sandbox(tmp_path: Path) -> None:
    context = _ToolContext("telegram-chat-42")
    result = {
        "messages": [
            {
                "id": "message-1",
                "text_body": "x" * 9_000,
                "html_body": "<p>html</p>",
                "metadata": ["kept"],
            }
        ]
    }
    sandbox = MagicMock()
    sandbox.files.write_file = AsyncMock()
    manager = MagicMock()
    manager.get_or_create_sandbox = AsyncMock(
        return_value={"sandbox": sandbox, "error": None}
    )

    with patch("blacki.gmail.tools.get_sandbox_manager", return_value=manager):
        compact = await _materialize_large_gmail_result(
            result,
            identity="message:message-1",
            tool_context=context,  # type: ignore[arg-type]
        )

    assert compact["body_storage"] == "sandbox"
    assert compact["sandbox_path"].startswith("/workspace/uploads/gmail-result-")
    assert compact["body_size_bytes"] == len("x" * 9_000) + len("<p>html</p>")
    assert compact["messages"][0]["text_body"] == ""
    sandbox.files.write_file.assert_awaited_once()
    assert sandbox.files.write_file.await_args.args[0] == compact["sandbox_path"]
    assert json.loads(sandbox.files.write_file.await_args.args[1]) == result


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_result", [{"sandbox": None, "error": "disabled"}])
async def test_large_gmail_results_are_truncated_without_a_sandbox(
    sandbox_result: dict[str, Any],
) -> None:
    result = {
        "messages": [
            {"text_body": "a" * 3_000, "html_body": "b" * 3_000},
            {"text_body": "c" * 3_000},
        ]
    }
    manager = MagicMock()
    manager.get_or_create_sandbox = AsyncMock(return_value=sandbox_result)

    with patch("blacki.gmail.tools.get_sandbox_manager", return_value=manager):
        compact = await _materialize_large_gmail_result(
            result,
            identity="thread:thread-1",
            tool_context=_ToolContext("telegram-chat-42"),  # type: ignore[arg-type]
        )

    assert compact["body_storage"] == "inline_truncated"
    assert compact["body_truncated"] is True
    assert len(compact["messages"][0]["text_body"]) == 2_000
    assert len(compact["messages"][0]["html_body"]) == 2_000
    assert len(compact["messages"][1]["text_body"]) == 2_000


@pytest.mark.asyncio
async def test_large_gmail_results_fall_back_when_serialization_or_write_fails() -> (
    None
):
    result = {"text_body": "body", "bad": object()}
    assert _gmail_body_present(result) is True
    assert _gmail_body_size(result) == 4
    stripped = cast(dict[str, Any], _strip_gmail_bodies(result))
    assert stripped["text_body"] == ""
    assert _truncate_gmail_bodies(
        {"text_body": "body", "html_body": 1, "items": ["value"]},
        [2_000],
    ) == {"text_body": "body", "html_body": 1, "items": ["value"]}
    assert _truncate_gmail_result(result)["body_truncated"] is True

    context = _ToolContext("telegram-chat-42")
    with patch("blacki.gmail.tools.get_sandbox_manager") as get_manager:
        compact = await _materialize_large_gmail_result(
            result,
            identity="message:bad",
            tool_context=context,  # type: ignore[arg-type]
        )
        get_manager.assert_not_called()
    assert compact["body_storage"] == "inline_truncated"

    small = {"text_body": "small"}
    assert (
        await _materialize_large_gmail_result(
            small,
            identity="message:small",
            tool_context=context,  # type: ignore[arg-type]
        )
        is small
    )

    serializable_result = {"text_body": "x" * 9_000}
    sandbox = MagicMock()
    sandbox.files.write_file = AsyncMock(side_effect=RuntimeError("write failed"))
    manager = MagicMock()
    manager.get_or_create_sandbox = AsyncMock(
        return_value={"sandbox": sandbox, "error": None}
    )
    with patch("blacki.gmail.tools.get_sandbox_manager", return_value=manager):
        compact = await _materialize_large_gmail_result(
            serializable_result,
            identity="message:write-fails",
            tool_context=context,  # type: ignore[arg-type]
        )
    assert compact["body_storage"] == "inline_truncated"


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_gmail_attachment_write_failures_are_safe_and_cleaned(
    cleanup_fails: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = MagicMock()
    service.storage = MagicMock()
    service.storage.is_initialized = True
    service.storage.has_connection = AsyncMock(return_value=True)
    service.download_attachment = AsyncMock(
        return_value=GmailAttachmentDownload(
            filename="../private.pdf",
            mime_type="application/pdf",
            size_bytes=4,
            data=b"data",
        )
    )
    context = _ToolContext("telegram-chat-42")
    sandbox = MagicMock()
    sandbox.files.write_file = AsyncMock(side_effect=RuntimeError("write secret"))
    sandbox.files.delete_files = AsyncMock(
        side_effect=RuntimeError("cleanup secret") if cleanup_fails else None
    )
    manager = MagicMock()
    manager.get_or_create_sandbox = AsyncMock(
        return_value={"sandbox": sandbox, "error": None}
    )
    caplog.set_level(logging.WARNING, logger="blacki.gmail.tools")
    token = _ACTIVE_SERVICE.set(service)
    try:
        with (
            patch("blacki.gmail.tools.get_sandbox_manager", return_value=manager),
            pytest.raises(GmailError, match="could not be written"),
        ):
            await gmail_download_attachment(
                "message-1",
                "1",
                tool_context=context,  # type: ignore[arg-type]
            )
    finally:
        _ACTIVE_SERVICE.reset(token)
    sandbox.files.delete_files.assert_awaited_once_with(
        ["/workspace/uploads/gmail-ee189c10d3f76374-private.pdf"]
    )
    if cleanup_fails:
        assert "cleanup secret" not in caplog.text


@pytest.mark.asyncio
async def test_gmail_attachment_download_rejects_unavailable_sandbox() -> None:
    service = MagicMock()
    service.storage = MagicMock()
    service.storage.is_initialized = True
    service.storage.has_connection = AsyncMock(return_value=True)
    service.download_attachment = AsyncMock(
        return_value=GmailAttachmentDownload(
            filename="invoice.pdf",
            mime_type="application/pdf",
            size_bytes=4,
            data=b"data",
        )
    )
    manager = MagicMock()
    manager.get_or_create_sandbox = AsyncMock(
        return_value={"sandbox": None, "error": "sandbox disabled"}
    )
    token = _ACTIVE_SERVICE.set(service)
    try:
        with (
            patch("blacki.gmail.tools.get_sandbox_manager", return_value=manager),
            pytest.raises(GmailCredentialError, match="unavailable"),
        ):
            await gmail_download_attachment(
                "message-1",
                "1",
                tool_context=_ToolContext("telegram-chat-42"),  # type: ignore[arg-type]
            )
    finally:
        _ACTIVE_SERVICE.reset(token)


@pytest.mark.asyncio
async def test_all_gmail_wrappers_use_the_active_user_scoped_service(
    tmp_path: Path,
) -> None:
    connection, storage, service = await _ready(tmp_path)
    service.get_message = AsyncMock(return_value={"message": True})
    service.get_thread = AsyncMock(return_value={"thread": True})
    service.list_drafts = AsyncMock(return_value={"drafts": []})
    service.get_draft = AsyncMock(return_value={"draft": True})
    service.create_draft = AsyncMock(return_value={"created": True})
    service.list_labels = AsyncMock(return_value={"labels": []})
    service.create_label = AsyncMock(return_value={"label": True})
    service.modify_message_labels = AsyncMock(return_value={"message": True})
    service.modify_thread_labels = AsyncMock(return_value={"thread": True})
    service.download_attachment = AsyncMock(
        return_value=GmailAttachmentDownload(
            filename="invoice.pdf",
            mime_type="application/pdf",
            size_bytes=4,
            data=b"data",
        )
    )
    context = _ToolContext("telegram-chat-42-thread-5")
    token = _ACTIVE_SERVICE.set(service)
    try:
        assert await gmail_search_messages(
            "query",
            tool_context=context,  # type: ignore[arg-type]
        ) == {"messages": []}
        assert await gmail_get_message(
            "message-1",
            tool_context=context,  # type: ignore[arg-type]
        ) == {"message": True}
        assert await gmail_get_thread(
            "thread-1",
            tool_context=context,  # type: ignore[arg-type]
        ) == {"thread": True}
        sandbox = MagicMock()
        sandbox.files.write_file = AsyncMock()
        sandbox_manager = MagicMock()
        sandbox_manager.get_or_create_sandbox = AsyncMock(
            return_value={"sandbox": sandbox, "error": None}
        )
        with patch(
            "blacki.gmail.tools.get_sandbox_manager",
            return_value=sandbox_manager,
        ):
            assert await gmail_download_attachment(
                "message-1",
                "1",
                tool_context=context,  # type: ignore[arg-type]
            ) == {
                "status": "success",
                "filename": "invoice.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 4,
                "sandbox_path": "/workspace/uploads/gmail-ee189c10d3f76374-invoice.pdf",
            }
        assert await gmail_list_drafts(
            tool_context=context  # type: ignore[arg-type]
        ) == {"drafts": []}
        assert await gmail_get_draft(
            "draft-1",
            tool_context=context,  # type: ignore[arg-type]
        ) == {"draft": True}
        assert await gmail_create_draft(
            "person@example.com",
            "Subject",
            "body",
            tool_context=context,  # type: ignore[arg-type]
        ) == {"created": True}
        assert await gmail_send_draft(
            "draft-1",
            "person@example.com",
            "",
            "",
            "Subject",
            "a" * 64,
            tool_context=context,  # type: ignore[arg-type]
        ) == {"sent": True}
        assert await gmail_list_labels(
            tool_context=context  # type: ignore[arg-type]
        ) == {"labels": []}
        assert await gmail_create_label(
            "Projects",
            tool_context=context,  # type: ignore[arg-type]
        ) == {"label": True}
        assert await gmail_modify_message_labels(
            "message-1",
            ["Label_1"],
            [],
            tool_context=context,  # type: ignore[arg-type]
        ) == {"message": True}
        assert await gmail_modify_thread_labels(
            "thread-1",
            ["Label_1"],
            [],
            tool_context=context,  # type: ignore[arg-type]
        ) == {"thread": True}
    finally:
        _ACTIVE_SERVICE.reset(token)
        await service.close()
        await connection.close()

    service.get_message.assert_awaited_once_with(
        "telegram-chat-42", message_id="message-1"
    )
    service.get_thread.assert_awaited_once_with(
        "telegram-chat-42", thread_id="thread-1", max_messages=25
    )
    service.download_attachment.assert_awaited_once_with(
        "telegram-chat-42", message_id="message-1", part_id="1"
    )
    service.create_draft.assert_awaited_once_with(
        "telegram-chat-42",
        to="person@example.com",
        subject="Subject",
        body="body",
        cc=None,
        bcc=None,
        reply_to_message_id=None,
    )


@pytest.mark.asyncio
async def test_gmail_service_context_handles_missing_storage_and_configuration(
    tmp_path: Path,
) -> None:
    connection, storage, service = await _ready(tmp_path)
    try:
        token = _ACTIVE_SERVICE.set(service)
        try:
            with pytest.raises(GmailCredentialError):
                await gmail_search_messages(
                    "query",
                    tool_context=_ToolContext("telegram-chat-43"),  # type: ignore[arg-type]
                )
        finally:
            _ACTIVE_SERVICE.reset(token)

        with (
            patch("blacki.gmail.tools.GmailConfig.from_environment", return_value=None),
            pytest.raises(GmailConfigurationError),
        ):
            async with _service_for_context(
                _ToolContext("telegram-chat-42")  # type: ignore[arg-type]
            ):
                raise AssertionError("context should not open")

        with (
            patch(
                "blacki.gmail.tools.GmailConfig.from_environment",
                return_value=_config(),
            ),
            patch(
                "blacki.gmail.tools.get_container",
                side_effect=RuntimeError("container unavailable"),
            ),
            pytest.raises(GmailCredentialError),
        ):
            async with _service_for_context(
                _ToolContext("telegram-chat-42")  # type: ignore[arg-type]
            ):
                raise AssertionError("context should not open")

        with (
            patch(
                "blacki.gmail.tools.GmailConfig.from_environment",
                return_value=_config(),
            ),
            patch(
                "blacki.gmail.tools.get_container",
                return_value=SimpleNamespace(gmail_storage=storage),
            ),
        ):
            async with _service_for_context(
                _ToolContext("telegram-chat-42")  # type: ignore[arg-type]
            ) as (owned_service, user_id):
                assert user_id == "telegram-chat-42"
                assert owned_service is not service

        uninitialized_connection = await create_connection(
            tmp_path / "uninitialized.db"
        )
        uninitialized_storage = SqliteGmailStorage(
            uninitialized_connection, asyncio.Lock()
        )
        uninitialized_service = GmailService(_config(), uninitialized_storage)
        token = _ACTIVE_SERVICE.set(uninitialized_service)
        try:
            with pytest.raises(GmailCredentialError):
                await gmail_search_messages(
                    "query",
                    tool_context=_ToolContext("telegram-chat-42"),  # type: ignore[arg-type]
                )
            assert uninitialized_storage.is_initialized is True
        finally:
            _ACTIVE_SERVICE.reset(token)
            await uninitialized_service.close()
            await uninitialized_connection.close()
    finally:
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_gmail_service_context_rejects_mismatched_private_session_metadata(
    tmp_path: Path,
) -> None:
    connection, storage, service = await _ready(tmp_path)
    try:
        invalid_contexts = [
            _ToolContext("local"),
            _ToolContext("telegram-chat-42"),
            _ToolContext("telegram-chat-42"),
            _ToolContext("telegram-chat-42"),
            _ToolContext("telegram-chat-42"),
        ]
        invalid_contexts[1].state["telegram_chat_type"] = "group"
        invalid_contexts[2].state["user_id"] = "telegram-chat-43"
        invalid_contexts[3].state["telegram_chat_id"] = 43
        invalid_contexts[4].state.update(
            {"telegram_chat_id": 42, "temp:telegram_sender_user_id": 43}
        )
        for context in invalid_contexts:
            with pytest.raises(GmailCredentialError):
                async with _service_for_context(context):  # type: ignore[arg-type]
                    raise AssertionError("context should not open")

        with (
            patch(
                "blacki.gmail.tools.GmailConfig.from_environment",
                return_value=_config(),
            ),
            patch(
                "blacki.gmail.tools.get_container",
                return_value=SimpleNamespace(gmail_storage=storage),
            ),
            pytest.raises(GmailCredentialError),
        ):
            async with _service_for_context(
                _ToolContext("telegram-chat-43")  # type: ignore[arg-type]
            ):
                raise AssertionError("unconnected context should not open")
    finally:
        await service.close()
        await connection.close()


@pytest.mark.asyncio
async def test_gmail_toolset_and_factory_handle_owned_resources_and_failures(
    tmp_path: Path,
) -> None:
    connection, storage, service = await _ready(tmp_path)
    try:
        owned_toolset = GmailToolset(config=_config(), storage=storage)
        await owned_toolset.close()
        factory_toolset = create_gmail_toolset(config=_config(), storage=storage)
        await factory_toolset.close()

        with (
            patch("blacki.gmail.tools.GmailConfig.from_environment", return_value=None),
            pytest.raises(GmailConfigurationError),
        ):
            create_gmail_toolset()

        with (
            patch(
                "blacki.gmail.tools.get_container",
                side_effect=RuntimeError("container unavailable"),
            ),
            pytest.raises(GmailCredentialError),
        ):
            create_gmail_toolset(config=_config())
    finally:
        await service.close()
        await connection.close()
