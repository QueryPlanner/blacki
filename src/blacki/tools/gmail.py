"""ADK tools for the private, direct Gmail API connector."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from hashlib import sha256
from typing import Any, cast

import httpx
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from blacki.container import get_container
from blacki.gmail.client import GmailService
from blacki.gmail.config import GmailConfig, canonical_gmail_user_id
from blacki.gmail.errors import (
    GmailApiError,
    GmailConfigurationError,
    GmailCredentialError,
    GmailError,
)
from blacki.gmail.storage import SqliteGmailStorage
from blacki.sandbox.manager import get_sandbox_manager
from blacki.user_files.config import SENDER_STATE_KEY
from blacki.user_files.service import sanitize_display_name

logger = logging.getLogger(__name__)

GMAIL_RESULT_SPILL_THRESHOLD_BYTES = 8 * 1024
GMAIL_INLINE_BODY_CHAR_LIMIT = 2_000
GMAIL_INLINE_TOTAL_BODY_CHAR_LIMIT = 8_000

_ACTIVE_SERVICE: ContextVar[GmailService | None] = ContextVar(
    "blacki_active_gmail_service",
    default=None,
)


def _safe_gmail_error_result(error: GmailError) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "error",
        "error": str(error),
    }
    if isinstance(error, GmailApiError) and error.error_code:
        result["error_code"] = error.error_code
    return result


async def _ensure_storage_ready(storage: SqliteGmailStorage) -> None:
    if not storage.is_initialized:
        await storage.initialize()


async def _materialize_large_gmail_result(
    result: dict[str, Any],
    *,
    identity: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Keep large Gmail bodies out of the model context when a sandbox exists."""
    if not _gmail_body_present(result):
        return result

    try:
        serialized = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return _truncate_gmail_result(result)
    if len(serialized) <= GMAIL_RESULT_SPILL_THRESHOLD_BYTES:
        return result

    sandbox_path = (
        "/workspace/uploads/gmail-result-"
        f"{sha256(identity.encode('utf-8')).hexdigest()[:16]}.json"
    )
    try:
        sandbox_result = await get_sandbox_manager().get_or_create_sandbox(
            tool_context.state
        )
        sandbox = sandbox_result.get("sandbox")
        if sandbox_result.get("error") or sandbox is None:
            return _truncate_gmail_result(result)
        await sandbox.files.write_file(sandbox_path, serialized)
    except Exception:
        return _truncate_gmail_result(result)

    compact = cast(dict[str, Any], _strip_gmail_bodies(result))
    compact["body_storage"] = "sandbox"
    compact["sandbox_path"] = sandbox_path
    compact["body_size_bytes"] = _gmail_body_size(result)
    return compact


def _truncate_gmail_result(result: dict[str, Any]) -> dict[str, Any]:
    """Bound a Gmail result when its session sandbox cannot store the body."""
    compact = cast(
        dict[str, Any],
        _truncate_gmail_bodies(result, [GMAIL_INLINE_TOTAL_BODY_CHAR_LIMIT]),
    )
    compact["body_storage"] = "inline_truncated"
    compact["body_truncated"] = True
    return compact


def _gmail_body_present(value: object) -> bool:
    if isinstance(value, dict):
        if any(value.get(key) for key in ("text_body", "html_body")):
            return True
        return any(_gmail_body_present(item) for item in value.values())
    if isinstance(value, list):
        return any(_gmail_body_present(item) for item in value)
    return False


def _gmail_body_size(value: object) -> int:
    if isinstance(value, dict):
        size = 0
        for key in ("text_body", "html_body"):
            body = value.get(key)
            if isinstance(body, str):
                size += len(body.encode("utf-8"))
        return size + sum(
            _gmail_body_size(item)
            for key, item in value.items()
            if key not in {"text_body", "html_body"}
        )
    if isinstance(value, list):
        return sum(_gmail_body_size(item) for item in value)
    return 0


def _strip_gmail_bodies(value: object) -> dict[str, Any] | list[Any] | object:
    if isinstance(value, dict):
        return {
            key: "" if key in {"text_body", "html_body"} else _strip_gmail_bodies(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_strip_gmail_bodies(item) for item in value]
    return value


def _truncate_gmail_bodies(
    value: object,
    remaining_chars: list[int],
) -> dict[str, Any] | list[Any] | object:
    if isinstance(value, dict):
        truncated: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"text_body", "html_body"} and isinstance(item, str):
                limit = min(GMAIL_INLINE_BODY_CHAR_LIMIT, remaining_chars[0])
                truncated[key] = item[:limit]
                remaining_chars[0] -= limit
            else:
                truncated[key] = _truncate_gmail_bodies(item, remaining_chars)
        return truncated
    if isinstance(value, list):
        return [_truncate_gmail_bodies(item, remaining_chars) for item in value]
    return value


@asynccontextmanager
async def _service_for_context(
    tool_context: ToolContext,
) -> AsyncIterator[tuple[GmailService, str]]:
    user_id = canonical_gmail_user_id(tool_context.user_id)
    if user_id is None:
        raise GmailCredentialError("Gmail is available only to a private Telegram user")
    state = tool_context.state
    chat_type = state.get("telegram_chat_type")
    if chat_type is not None and chat_type != "private":
        raise GmailCredentialError("Gmail is available only in a private chat")
    state_user_id = state.get("user_id")
    if state_user_id is not None and state_user_id != tool_context.user_id:
        raise GmailCredentialError("Gmail user context does not match the session")
    chat_id = state.get("telegram_chat_id")
    if chat_id is not None and user_id != f"telegram-chat-{chat_id}":
        raise GmailCredentialError("Gmail user context does not match the chat")
    sender_user_id = state.get(SENDER_STATE_KEY)
    if sender_user_id is not None and str(sender_user_id) != str(chat_id):
        raise GmailCredentialError("Gmail sender context does not match the chat")
    active = _ACTIVE_SERVICE.get()
    if active is not None:
        await _ensure_storage_ready(active.storage)
        if not await active.storage.has_connection(user_id):
            raise GmailCredentialError("Gmail is not connected for this user")
        yield active, user_id
        return

    config = GmailConfig.from_environment()
    if config is None:
        raise GmailConfigurationError("Gmail is not configured on this Blacki server")
    try:
        storage = get_container().gmail_storage
    except RuntimeError as exc:
        raise GmailCredentialError("Gmail storage is not available") from exc
    await _ensure_storage_ready(storage)
    if not await storage.has_connection(user_id):
        raise GmailCredentialError("Gmail is not connected for this user")
    service = GmailService(config, storage)
    try:
        yield service, user_id
    finally:
        await service.close()


async def gmail_search_messages(
    query: str,
    max_results: int = 10,
    page_token: str | None = None,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Search the connected user's non-spam, non-trash Gmail messages."""
    async with _service_for_context(tool_context) as (service, user_id):
        return await service.search_messages(
            user_id,
            query=query,
            max_results=max_results,
            page_token=page_token,
        )


async def gmail_get_message(
    message_id: str,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Read one message, spilling large bodies into the session sandbox."""
    async with _service_for_context(tool_context) as (service, user_id):
        result = await service.get_message(user_id, message_id=message_id)
        return await _materialize_large_gmail_result(
            result,
            identity=f"message:{message_id}",
            tool_context=tool_context,
        )


async def gmail_get_thread(
    thread_id: str,
    max_messages: int = 25,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Read a bounded thread, spilling large bodies into the session sandbox."""
    async with _service_for_context(tool_context) as (service, user_id):
        result = await service.get_thread(
            user_id,
            thread_id=thread_id,
            max_messages=max_messages,
        )
        return await _materialize_large_gmail_result(
            result,
            identity=f"thread:{thread_id}",
            tool_context=tool_context,
        )


async def gmail_download_attachment(
    message_id: str,
    part_id: str,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Download one requested Gmail attachment into this session's sandbox."""
    async with _service_for_context(tool_context) as (service, user_id):
        attachment = await service.download_attachment(
            user_id,
            message_id=message_id,
            part_id=part_id,
        )
        sandbox_result = await get_sandbox_manager().get_or_create_sandbox(
            tool_context.state
        )
        sandbox = sandbox_result.get("sandbox")
        if sandbox_result.get("error") or sandbox is None:
            raise GmailCredentialError("Gmail attachment sandbox is unavailable")

        filename = sanitize_display_name(attachment.filename)
        digest = sha256(f"{message_id}\0{part_id}".encode()).hexdigest()[:16]
        sandbox_path = f"/workspace/uploads/gmail-{digest}-{filename}"
        try:
            await sandbox.files.write_file(sandbox_path, attachment.data)
        except Exception as exc:
            try:
                await sandbox.files.delete_files([sandbox_path])
            except Exception as cleanup_exc:
                logger.warning(
                    "Could not remove an incomplete Gmail sandbox file (%s)",
                    type(cleanup_exc).__name__,
                )
            raise GmailError(
                "Gmail attachment could not be written to the sandbox"
            ) from exc
        return {
            "status": "success",
            "filename": filename,
            "mime_type": attachment.mime_type,
            "size_bytes": attachment.size_bytes,
            "sandbox_path": sandbox_path,
        }


async def gmail_list_drafts(
    max_results: int = 10,
    page_token: str | None = None,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """List the connected user's Gmail drafts without sending anything."""
    async with _service_for_context(tool_context) as (service, user_id):
        return await service.list_drafts(
            user_id,
            max_results=max_results,
            page_token=page_token,
        )


async def gmail_get_draft(
    draft_id: str,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Read one draft, including bounded body text and recipient headers."""
    async with _service_for_context(tool_context) as (service, user_id):
        result = await service.get_draft(user_id, draft_id=draft_id)
        return await _materialize_large_gmail_result(
            result,
            identity=f"draft:{draft_id}",
            tool_context=tool_context,
        )


async def gmail_create_draft(
    to: str,
    subject: str,
    body: str,
    cc: str | None = None,
    bcc: str | None = None,
    reply_to_message_id: str | None = None,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Create a Gmail draft; this tool never sends it."""
    async with _service_for_context(tool_context) as (service, user_id):
        return await service.create_draft(
            user_id,
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            reply_to_message_id=reply_to_message_id,
        )


async def gmail_send_draft(
    draft_id: str,
    expected_to: str,
    expected_cc: str,
    expected_bcc: str,
    expected_subject: str,
    expected_content_sha256: str,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Send a confirmed draft only if its content still matches."""
    async with _service_for_context(tool_context) as (service, user_id):
        return await service.send_draft(
            user_id,
            draft_id=draft_id,
            expected_to=expected_to,
            expected_cc=expected_cc,
            expected_bcc=expected_bcc,
            expected_subject=expected_subject,
            expected_content_sha256=expected_content_sha256,
        )


async def gmail_list_labels(
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """List Gmail labels without changing mailbox state."""
    async with _service_for_context(tool_context) as (service, user_id):
        return await service.list_labels(user_id)


async def gmail_create_label(
    name: str,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Create one non-system Gmail label."""
    async with _service_for_context(tool_context) as (service, user_id):
        return await service.create_label(user_id, name=name)


async def gmail_modify_message_labels(
    message_id: str,
    add_label_ids: list[str],
    remove_label_ids: list[str],
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Modify only non-system labels on one Gmail message."""
    async with _service_for_context(tool_context) as (service, user_id):
        return await service.modify_message_labels(
            user_id,
            message_id=message_id,
            add_label_ids=add_label_ids,
            remove_label_ids=remove_label_ids,
        )


async def gmail_modify_thread_labels(
    thread_id: str,
    add_label_ids: list[str],
    remove_label_ids: list[str],
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Modify only non-system labels on one Gmail thread."""
    async with _service_for_context(tool_context) as (service, user_id):
        return await service.modify_thread_labels(
            user_id,
            thread_id=thread_id,
            add_label_ids=add_label_ids,
            remove_label_ids=remove_label_ids,
        )


_TOOL_FUNCTIONS: tuple[Callable[..., Any], ...] = (
    gmail_search_messages,
    gmail_get_message,
    gmail_get_thread,
    gmail_download_attachment,
    gmail_list_drafts,
    gmail_get_draft,
    gmail_create_draft,
    gmail_send_draft,
    gmail_list_labels,
    gmail_create_label,
    gmail_modify_message_labels,
    gmail_modify_thread_labels,
)


class GmailToolset(BaseToolset):
    """Expose Gmail tools only to connected private Telegram users."""

    def __init__(
        self,
        *,
        config: GmailConfig,
        storage: SqliteGmailStorage,
        http_client: httpx.AsyncClient | None = None,
        service: GmailService | None = None,
    ) -> None:
        super().__init__()
        # A user's connection may become available after skill activation.
        self._use_invocation_cache = False
        self.config = config
        self.storage = storage
        self._service = service or GmailService(
            config,
            storage,
            http_client=http_client,
        )
        self._owns_service = service is None
        self._tools = self._build_tools()

    def _build_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for function in _TOOL_FUNCTIONS:
            require_confirmation = function is gmail_send_draft

            @wraps(function)
            async def bound(
                *args: Any,
                _function: Callable[..., Any] = function,
                **kwargs: Any,
            ) -> Any:
                token = _ACTIVE_SERVICE.set(self._service)
                try:
                    return await _function(*args, **kwargs)
                except GmailError as exc:
                    return _safe_gmail_error_result(exc)
                except Exception as exc:
                    logger.error(
                        "Unexpected Gmail connector exception in %s (%s)",
                        _function.__name__,
                        type(exc).__name__,
                    )
                    return {
                        "status": "error",
                        "error": "Gmail operation failed unexpectedly",
                    }
                finally:
                    _ACTIVE_SERVICE.reset(token)

            tools.append(
                FunctionTool(
                    cast(Callable[..., Any], bound),
                    require_confirmation=require_confirmation,
                )
            )
        return tools

    async def get_tools(
        self,
        readonly_context: ReadonlyContext | None = None,
    ) -> list[BaseTool]:
        """Return tools only for a connected canonical private Telegram identity."""
        if readonly_context is None:
            return []
        user_id = canonical_gmail_user_id(readonly_context.user_id)
        if user_id is None:
            return []
        await _ensure_storage_ready(self.storage)
        if not await self.storage.has_connection(user_id):
            return []
        return list(self._tools)

    async def close(self) -> None:
        """Close the service owned by this toolset."""
        if self._owns_service:
            await self._service.close()


def create_gmail_toolset(
    *,
    config: GmailConfig | None = None,
    storage: SqliteGmailStorage | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> GmailToolset:
    """Create the private Gmail toolset from validated shared configuration."""
    resolved_config = config or GmailConfig.from_environment()
    if resolved_config is None:
        raise GmailConfigurationError(
            "Gmail is disabled or its shared OAuth configuration is incomplete"
        )
    if storage is None:
        try:
            storage = get_container().gmail_storage
        except RuntimeError as exc:
            raise GmailCredentialError("Gmail storage is not available") from exc
    return GmailToolset(
        config=resolved_config,
        storage=storage,
        http_client=http_client,
    )
