"""Private Zepto MCP toolset composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

from blacki.zepto.client import (
    BRIDGE_CONNECT_TIMEOUT_SECONDS,
    ZEPTO_TOOL_PREFIX,
    McpRemoteCredentialStore,
    ZeptoCredentialError,
    create_bridge_server_parameters,
)


def _is_allowed_private_telegram_user(
    readonly_context: ReadonlyContext | None,
    allowed_chat_ids: frozenset[str],
) -> bool:
    if readonly_context is None:
        return False
    user_id = readonly_context.user_id
    prefix = "telegram-chat-"
    if not user_id.startswith(prefix):
        return False
    chat_id = user_id.removeprefix(prefix)
    return chat_id.isdigit() and chat_id in allowed_chat_ids


def _is_supported_zepto_tool(
    tool: BaseTool,
    readonly_context: ReadonlyContext | None = None,
) -> bool:
    """Exclude Zepto's conversational wrapper from individual MCP tool use."""
    del readonly_context
    return tool.name != "zepto_shop"


def _requires_zepto_order_confirmation(
    confirmOrder: object = False,  # noqa: N803 - external Zepto MCP field
) -> bool:
    """Confirm only calls that can finalize a real Zepto order or payment."""
    return confirmOrder is not False


class AuthorizedZeptoToolset(McpToolset):
    """MCP toolset that refuses bridge startup for unauthorized identities."""

    def __init__(self, *, allowed_chat_ids: frozenset[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._allowed_chat_ids = allowed_chat_ids

    async def get_tools(
        self, readonly_context: ReadonlyContext | None = None
    ) -> list[BaseTool]:
        if not _is_allowed_private_telegram_user(
            readonly_context, self._allowed_chat_ids
        ):
            return []
        return cast(list[BaseTool], await super().get_tools(readonly_context))


def create_zepto_toolset(
    *,
    config_dir: Path,
    allowed_chat_ids: frozenset[str],
) -> AuthorizedZeptoToolset:
    """Build the private, root-only Zepto MCP toolset."""
    if not allowed_chat_ids:
        raise ZeptoCredentialError(
            "At least one private Telegram chat ID must be allowed for Zepto."
        )
    if any(not chat_id.isdigit() or int(chat_id) <= 0 for chat_id in allowed_chat_ids):
        raise ZeptoCredentialError(
            "Zepto allowlist accepts positive private Telegram chat IDs only."
        )

    store = McpRemoteCredentialStore(config_dir)
    store.validate_runtime_ready()
    in_docker = Path("/.dockerenv").exists()
    server_parameters = create_bridge_server_parameters(
        config_dir=store.config_dir,
        auth_timeout_seconds=2,
        allow_npx=not in_docker,
    )
    return AuthorizedZeptoToolset(
        allowed_chat_ids=allowed_chat_ids,
        connection_params=StdioConnectionParams(
            server_params=server_parameters,
            timeout=BRIDGE_CONNECT_TIMEOUT_SECONDS,
        ),
        tool_name_prefix=ZEPTO_TOOL_PREFIX,
        tool_filter=_is_supported_zepto_tool,
        require_confirmation=_requires_zepto_order_confirmation,
    )
