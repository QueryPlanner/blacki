"""Tests for the private Zepto mcp-remote bridge."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_tool import McpTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from mcp import StdioServerParameters
from mcp.types import ListToolsResult, Tool

from blacki.zepto.client import (
    _ALLOWED_FILENAMES,
    _SERVER_HASH,
    BRIDGE_CONNECT_TIMEOUT_SECONDS,
    FINAL_ORDER_PAYMENT_TOOL_NAMES,
    MCP_REMOTE_PACKAGE_VERSION,
    REQUIRED_SCOPE,
    ZEPTO_MCP_URL,
    AuthorizedZeptoToolset,
    McpRemoteCredentialStore,
    ZeptoCredentialError,
    _absolute,
    _bridge_environment,
    _is_allowed_private_telegram_user,
    _is_supported_zepto_tool,
    _locked_checkout_bridge,
    _read_private_object,
    _requires_zepto_order_confirmation,
    _resolve_bridge_command,
    _validate_owner_and_mode,
    _verified_bridge_binary,
    create_bridge_server_parameters,
    create_zepto_toolset,
)

ZEPTO_TOOL_NAMES = frozenset(
    {
        "add_saved_address",
        "check_payment_status",
        "create_online_payment_order",
        "create_order",
        "create_upi_reserve_pay_order",
        "create_wallet_order",
        "get_location_serviceability",
        "get_order_detail",
        "get_past_order_items",
        "get_payment_methods",
        "get_product_details",
        "get_user_details",
        "list_order_history",
        "list_saved_addresses",
        "search_multiple_products",
        "search_products",
        "select_saved_address",
        "select_store",
        "update_cart",
        "update_drop_zone",
        "update_user_name",
        "view_cart",
        "zepto_shop",
    }
)


def _write_private(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _ready_store(tmp_path: Path) -> McpRemoteCredentialStore:
    store = McpRemoteCredentialStore(tmp_path / "zepto")
    store.prepare_for_login()
    _write_private(
        store.version_dir / f"{_SERVER_HASH}_client_info.json",
        {
            "client_id": "client-id",
            "token_endpoint_auth_method": "none",
        },
    )
    _write_private(
        store.version_dir / f"{_SERVER_HASH}_tokens.json",
        {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "token_type": "Bearer",
            "scope": REQUIRED_SCOPE,
        },
    )
    return store


def _readonly_context(user_id: str) -> MagicMock:
    context = MagicMock(spec=ReadonlyContext)
    context.user_id = user_id
    return context


def test_store_prepares_and_validates_exact_private_state(tmp_path: Path) -> None:
    store = _ready_store(tmp_path)
    for suffix in ("code_verifier.txt", "lock.json"):
        _write_private(store.version_dir / f"{_SERVER_HASH}_{suffix}", {})

    store.validate_runtime_ready()

    assert stat.S_IMODE(store.config_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.version_dir.stat().st_mode) == 0o700
    assert {entry.name for entry in store.version_dir.iterdir()} == (_ALLOWED_FILENAMES)


def test_store_reports_missing_and_unreadable_directories(tmp_path: Path) -> None:
    store = McpRemoteCredentialStore(tmp_path / "missing")
    with pytest.raises(ZeptoCredentialError, match="missing or unreadable"):
        store.validate_runtime_ready()

    store.prepare_for_login()
    with pytest.raises(ZeptoCredentialError, match="not authenticated"):
        store.validate_runtime_ready()

    store.version_dir.chmod(0o755)
    with pytest.raises(ZeptoCredentialError, match="0700"):
        store.validate_runtime_ready()


def test_store_rejects_non_directories_and_iteration_failure(tmp_path: Path) -> None:
    store = McpRemoteCredentialStore(tmp_path / "credentials")
    non_directory = SimpleNamespace(st_mode=stat.S_IFREG)
    with (
        patch(
            "blacki.zepto.client._validate_owner_and_mode",
            return_value=non_directory,
        ),
        pytest.raises(ZeptoCredentialError, match="must be directories"),
    ):
        store.validate_runtime_ready()

    store = _ready_store(tmp_path / "ready")
    with (
        patch.object(Path, "iterdir", side_effect=OSError("unreadable")),
        pytest.raises(ZeptoCredentialError, match="directory is unreadable"),
    ):
        store.validate_runtime_ready()


def test_store_rejects_symlinked_path_chain_and_entries(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ZeptoCredentialError, match="symlinks"):
        McpRemoteCredentialStore(linked / "zepto").prepare_for_login()

    store = _ready_store(tmp_path / "files")
    tokens = store.version_dir / f"{_SERVER_HASH}_tokens.json"
    tokens.unlink()
    tokens.symlink_to(store.version_dir / f"{_SERVER_HASH}_client_info.json")
    with pytest.raises(ZeptoCredentialError, match="symlink"):
        store.validate_runtime_ready()
    with pytest.raises(ZeptoCredentialError, match="symlinks"):
        store.tighten_permissions()


def test_store_rejects_unsafe_entries_permissions_and_ownership(
    tmp_path: Path,
) -> None:
    store = _ready_store(tmp_path)
    unexpected = store.version_dir / "other.json"
    _write_private(unexpected, {})
    with pytest.raises(ZeptoCredentialError, match="unexpected"):
        store.validate_runtime_ready()
    unexpected.unlink()

    token_path = store.version_dir / f"{_SERVER_HASH}_tokens.json"
    token_path.chmod(0o644)
    with pytest.raises(ZeptoCredentialError, match="0600"):
        store.validate_runtime_ready()
    token_path.chmod(0o600)

    lock_path = store.version_dir / f"{_SERVER_HASH}_lock.json"
    lock_path.mkdir(mode=0o600)
    with pytest.raises(ZeptoCredentialError, match="regular files"):
        store.validate_runtime_ready()
    lock_path.rmdir()

    with (
        patch("blacki.zepto.client.os.geteuid", return_value=os.geteuid() + 1),
        pytest.raises(ZeptoCredentialError, match="owned by Blacki"),
    ):
        store.validate_runtime_ready()


@pytest.mark.parametrize(
    ("filename", "value", "message"),
    [
        ("client_info.json", [], "client registration is corrupt"),
        (
            "client_info.json",
            {"client_id": "", "token_endpoint_auth_method": "none"},
            "client registration is corrupt",
        ),
        (
            "client_info.json",
            {"client_id": "id", "token_endpoint_auth_method": "client_secret_post"},
            "client registration is corrupt",
        ),
        ("tokens.json", "bad", "OAuth tokens is corrupt"),
        (
            "tokens.json",
            {
                "access_token": "",
                "refresh_token": "refresh",
                "token_type": "Bearer",
                "scope": REQUIRED_SCOPE,
            },
            "OAuth tokens are corrupt",
        ),
        (
            "tokens.json",
            {
                "access_token": "access",
                "refresh_token": "",
                "token_type": "Bearer",
                "scope": REQUIRED_SCOPE,
            },
            "OAuth tokens are corrupt",
        ),
        (
            "tokens.json",
            {
                "access_token": "access",
                "refresh_token": "refresh",
                "token_type": "Basic",
                "scope": REQUIRED_SCOPE,
            },
            "OAuth tokens are corrupt",
        ),
        (
            "tokens.json",
            {
                "access_token": "access",
                "refresh_token": "refresh",
                "token_type": "Bearer",
                "scope": "openid",
            },
            "OAuth tokens are corrupt",
        ),
    ],
)
def test_store_rejects_corrupt_json_shapes(
    tmp_path: Path,
    filename: str,
    value: object,
    message: str,
) -> None:
    store = _ready_store(tmp_path)
    _write_private(store.version_dir / f"{_SERVER_HASH}_{filename}", value)

    with pytest.raises(ZeptoCredentialError, match=message):
        store.validate_runtime_ready()


def test_store_rejects_invalid_json_and_prepare_failure(tmp_path: Path) -> None:
    store = _ready_store(tmp_path)
    client = store.version_dir / f"{_SERVER_HASH}_client_info.json"
    client.write_text("{", encoding="utf-8")
    with pytest.raises(ZeptoCredentialError, match="corrupt"):
        store.validate_runtime_ready()

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ZeptoCredentialError, match="Could not prepare"):
        McpRemoteCredentialStore(blocked).prepare_for_login()

    with pytest.raises(ZeptoCredentialError, match="corrupt"):
        _read_private_object(tmp_path / "absent.json", "test file")


def test_store_tightens_bridge_permissions(tmp_path: Path) -> None:
    store = _ready_store(tmp_path)
    store.config_dir.chmod(0o755)
    store.version_dir.chmod(0o755)
    for entry in store.version_dir.iterdir():
        entry.chmod(0o644)

    store.tighten_permissions()
    store.validate_runtime_ready()

    absent = McpRemoteCredentialStore(tmp_path / "absent")
    absent.tighten_permissions()

    directory_entry = store.version_dir / f"{_SERVER_HASH}_lock.json"
    directory_entry.mkdir()
    store.tighten_permissions()

    with (
        patch.object(Path, "chmod", side_effect=OSError("permission denied")),
        pytest.raises(ZeptoCredentialError, match="Could not protect"),
    ):
        store.tighten_permissions()


def test_store_clear_authentication_is_narrow_and_explicit(tmp_path: Path) -> None:
    absent = McpRemoteCredentialStore(tmp_path / "absent")
    absent.clear_authentication()

    partial = McpRemoteCredentialStore(tmp_path / "partial")
    partial.config_dir.mkdir(mode=0o700)
    partial.clear_authentication()

    store = _ready_store(tmp_path / "ready")
    for suffix in ("code_verifier.txt", "lock.json"):
        _write_private(store.version_dir / f"{_SERVER_HASH}_{suffix}", {})
    store.clear_authentication()
    assert list(store.version_dir.iterdir()) == []

    store = _ready_store(tmp_path / "unexpected")
    _write_private(store.version_dir / "other.json", {})
    with pytest.raises(ZeptoCredentialError, match="unexpected files"):
        store.clear_authentication()

    store = _ready_store(tmp_path / "directory-entry")
    lock_path = store.version_dir / f"{_SERVER_HASH}_lock.json"
    lock_path.mkdir(mode=0o600)
    with pytest.raises(ZeptoCredentialError, match="regular files"):
        store.clear_authentication()


def test_store_clear_rejects_non_directory_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "config-file"
    config_file.write_text("x", encoding="utf-8")
    config_file.chmod(0o700)
    with pytest.raises(ZeptoCredentialError, match="must be a directory"):
        McpRemoteCredentialStore(config_file).clear_authentication()

    store = McpRemoteCredentialStore(tmp_path / "version-file")
    store.config_dir.mkdir(mode=0o700)
    store.version_dir.write_text("x", encoding="utf-8")
    store.version_dir.chmod(0o700)
    with pytest.raises(ZeptoCredentialError, match="must be a directory"):
        store.clear_authentication()


def test_low_level_path_validation_requires_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    file_path.chmod(0o700)
    assert _absolute(file_path) == file_path
    result = _validate_owner_and_mode(
        file_path,
        expected_mode=0o700,
        label="test path",
    )
    assert stat.S_ISREG(result.st_mode)

    store = McpRemoteCredentialStore(file_path)
    with pytest.raises(ZeptoCredentialError, match="missing or unreadable"):
        store.validate_runtime_ready()


def _fake_bridge_package(tmp_path: Path, *, version: str = "0.1.38") -> Path:
    package_dir = tmp_path / "node_modules" / "mcp-remote"
    executable = package_dir / "dist" / "proxy.js"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    executable.chmod(0o755)
    (package_dir / "package.json").write_text(
        json.dumps({"name": "mcp-remote", "version": version}),
        encoding="utf-8",
    )
    return executable


def test_bridge_binary_validation_requires_exact_runnable_package(
    tmp_path: Path,
) -> None:
    executable = _fake_bridge_package(tmp_path / "valid")
    assert _verified_bridge_binary(executable) == str(executable)

    wrong = _fake_bridge_package(tmp_path / "wrong", version="0.1.37")
    with pytest.raises(ZeptoCredentialError, match="exactly version"):
        _verified_bridge_binary(wrong)

    executable.chmod(0o600)
    with pytest.raises(ZeptoCredentialError, match="not runnable"):
        _verified_bridge_binary(executable)

    with pytest.raises(ZeptoCredentialError, match="metadata is unreadable"):
        _verified_bridge_binary(tmp_path / "missing")


def test_bridge_resolution_prefers_locked_install_then_pinned_npx(
    tmp_path: Path,
) -> None:
    locked = _fake_bridge_package(tmp_path / "locked")
    with (
        patch("blacki.zepto.client._locked_checkout_bridge", return_value=locked),
        patch("blacki.zepto.client.shutil.which") as which,
    ):
        assert _resolve_bridge_command(allow_npx=True) == (str(locked), [])
    which.assert_not_called()

    installed = _fake_bridge_package(tmp_path / "installed")
    with (
        patch("blacki.zepto.client._locked_checkout_bridge", return_value=None),
        patch("blacki.zepto.client.shutil.which", return_value=str(installed)),
    ):
        assert _resolve_bridge_command(allow_npx=True) == (
            str(installed),
            [],
        )

    def only_npx(command: str) -> str | None:
        return "/usr/bin/npx" if command == "npx" else None

    with (
        patch("blacki.zepto.client._locked_checkout_bridge", return_value=None),
        patch("blacki.zepto.client.shutil.which", side_effect=only_npx),
    ):
        command, args = _resolve_bridge_command(allow_npx=True)
    assert command == "/usr/bin/npx"
    assert args == [
        "--yes",
        "--package",
        f"mcp-remote@{MCP_REMOTE_PACKAGE_VERSION}",
        "mcp-remote",
    ]

    with (
        patch("blacki.zepto.client._locked_checkout_bridge", return_value=None),
        patch("blacki.zepto.client.shutil.which", return_value=None),
        pytest.raises(ZeptoCredentialError, match="unavailable"),
    ):
        _resolve_bridge_command(allow_npx=True)
    with (
        patch("blacki.zepto.client._locked_checkout_bridge", return_value=None),
        patch("blacki.zepto.client.shutil.which", return_value=None),
        pytest.raises(ZeptoCredentialError, match="unavailable"),
    ):
        _resolve_bridge_command(allow_npx=False)


def test_bridge_parameters_are_pinned_silent_and_secret_minimal(
    tmp_path: Path,
) -> None:
    with (
        patch(
            "blacki.zepto.client._resolve_bridge_command",
            return_value=("/usr/local/bin/mcp-remote", []),
        ),
        patch(
            "blacki.zepto.client.get_default_environment",
            return_value={"HOME": "/home/app", "PATH": "/usr/bin"},
        ),
    ):
        parameters = create_bridge_server_parameters(
            config_dir=tmp_path,
            auth_timeout_seconds=2,
            allow_npx=False,
        )

    assert parameters.command == "/usr/local/bin/mcp-remote"
    assert parameters.args == [
        ZEPTO_MCP_URL,
        "--transport",
        "http-only",
        "--auth-timeout",
        "2",
        "--silent",
    ]
    assert parameters.env == {
        "HOME": "/home/app",
        "PATH": "/usr/bin",
        "MCP_REMOTE_CONFIG_DIR": str(tmp_path),
        "npm_config_userconfig": os.devnull,
    }
    assert "OPENROUTER_API_KEY" not in parameters.env
    assert _bridge_environment(tmp_path)["MCP_REMOTE_CONFIG_DIR"] == str(tmp_path)


def test_locked_checkout_bridge_is_optional() -> None:
    with patch.object(Path, "exists", return_value=False):
        assert _locked_checkout_bridge() is None
    with patch.object(Path, "exists", return_value=True):
        assert _locked_checkout_bridge() is not None


@pytest.mark.parametrize(
    ("user_id", "expected"),
    [
        ("telegram-chat-123", True),
        ("telegram-chat--123", False),
        ("telegram-chat-123-thread-4", False),
        ("web-user", False),
    ],
)
def test_private_telegram_allowlist(user_id: str, expected: bool) -> None:
    assert (
        _is_allowed_private_telegram_user(
            _readonly_context(user_id),
            frozenset({"123"}),
        )
        is expected
    )
    assert _is_allowed_private_telegram_user(None, frozenset({"123"})) is False


@pytest.mark.asyncio
async def test_authorized_toolset_short_circuits_before_bridge() -> None:
    params = StdioConnectionParams(
        server_params=StdioServerParameters(command="/bin/false")
    )
    toolset = AuthorizedZeptoToolset(
        allowed_chat_ids=frozenset({"123"}),
        connection_params=params,
        tool_name_prefix="zepto",
        require_confirmation=True,
    )
    inherited_tool = MagicMock(spec=BaseTool)
    with patch.object(
        McpToolset,
        "get_tools",
        new=AsyncMock(return_value=[inherited_tool]),
    ) as inherited:
        assert await toolset.get_tools(_readonly_context("web-user")) == []
        assert await toolset.get_tools(_readonly_context("telegram-chat-999")) == []
        assert await toolset.get_tools(None) == []
        assert await toolset.get_tools(_readonly_context("telegram-chat-123")) == [
            inherited_tool
        ]
    inherited.assert_awaited_once()


@pytest.mark.asyncio
async def test_zepto_manifest_uses_final_order_confirmation_policy() -> None:
    toolset = AuthorizedZeptoToolset(
        allowed_chat_ids=frozenset({"123"}),
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command="/bin/false")
        ),
        tool_name_prefix="zepto",
        tool_filter=_is_supported_zepto_tool,
        require_confirmation=_requires_zepto_order_confirmation,
    )
    manifest = ListToolsResult(
        tools=[
            Tool(
                name=name,
                description=name,
                inputSchema={
                    "type": "object",
                    "properties": (
                        {"confirmOrder": {"type": "boolean"}}
                        if name in FINAL_ORDER_PAYMENT_TOOL_NAMES
                        else {}
                    ),
                },
            )
            for name in sorted(ZEPTO_TOOL_NAMES)
        ]
    )
    with patch.object(
        toolset,
        "_execute_with_session",
        new=AsyncMock(return_value=manifest),
    ):
        tools = await toolset.get_tools_with_prefix(
            _readonly_context("telegram-chat-123")
        )
    create_session = AsyncMock()

    assert {tool.name for tool in tools} == {
        f"zepto_{name}" for name in ZEPTO_TOOL_NAMES - {"zepto_shop"}
    }
    assert "zepto_zepto_shop" not in {tool.name for tool in tools}
    with patch.object(
        toolset._mcp_session_manager,
        "create_session",
        new=create_session,
    ):
        for tool in tools:
            assert isinstance(tool, McpTool)
            context = MagicMock(spec=ToolContext)
            context.tool_confirmation = None
            assert await tool.check_require_confirmation({}, context) is False

            tool_name = tool.name.removeprefix("zepto_")
            if tool_name not in FINAL_ORDER_PAYMENT_TOOL_NAMES:
                continue

            assert (
                await tool.check_require_confirmation(
                    {"confirmOrder": False},
                    context,
                )
                is False
            )
            assert (
                await tool.check_require_confirmation(
                    {"confirmOrder": True},
                    context,
                )
                is True
            )
            result = await tool.run_async(
                args={"confirmOrder": True},
                tool_context=context,
            )
            assert result == {
                "error": (
                    "This tool call requires confirmation, please approve or reject."
                )
            }
            context.request_confirmation.assert_called_once()
    create_session.assert_not_awaited()


@pytest.mark.parametrize("malformed_value", [None, 0, "false", {}, []])
def test_order_confirmation_fails_closed_for_malformed_values(
    malformed_value: object,
) -> None:
    assert _requires_zepto_order_confirmation() is False
    assert _requires_zepto_order_confirmation(confirmOrder=False) is False
    assert _requires_zepto_order_confirmation(confirmOrder=True) is True
    assert _requires_zepto_order_confirmation(confirmOrder=malformed_value) is True


def test_zepto_skill_uses_only_adk_final_confirmation() -> None:
    skill_path = (
        Path(__file__).parents[2] / "src" / "blacki" / "skills" / "zepto" / "SKILL.md"
    )
    instructions = skill_path.read_text(encoding="utf-8")

    assert "Every Zepto tool call requires" not in instructions
    assert "without asking for approval" in instructions
    assert "`confirmOrder=false`" in instructions
    assert "`confirmOrder=true`" in instructions
    assert "Do not ask separately" in instructions


def test_create_toolset_requires_private_allowlist_and_ready_store(
    tmp_path: Path,
) -> None:
    store = _ready_store(tmp_path)
    with pytest.raises(ZeptoCredentialError, match="At least one"):
        create_zepto_toolset(config_dir=store.config_dir, allowed_chat_ids=frozenset())
    with pytest.raises(ZeptoCredentialError, match="positive private"):
        create_zepto_toolset(
            config_dir=store.config_dir,
            allowed_chat_ids=frozenset({"-123"}),
        )

    server = StdioServerParameters(command="/usr/local/bin/mcp-remote")
    with patch(
        "blacki.zepto.client.create_bridge_server_parameters",
        return_value=server,
    ) as create_bridge:
        toolset = create_zepto_toolset(
            config_dir=store.config_dir,
            allowed_chat_ids=frozenset({"123"}),
        )

    assert isinstance(toolset, AuthorizedZeptoToolset)
    assert toolset.tool_name_prefix == "zepto"
    assert toolset.tool_filter is _is_supported_zepto_tool
    assert toolset._require_confirmation is _requires_zepto_order_confirmation
    connection = toolset._connection_params
    assert isinstance(connection, StdioConnectionParams)
    assert connection.server_params == server
    assert connection.timeout == BRIDGE_CONNECT_TIMEOUT_SECONDS
    assert create_bridge.call_args.kwargs["auth_timeout_seconds"] == 2


def test_create_toolset_forbids_npx_fallback_in_docker(tmp_path: Path) -> None:
    store = _ready_store(tmp_path)
    server = StdioServerParameters(command="/usr/local/bin/mcp-remote")
    original_exists = Path.exists

    def docker_exists(path: Path) -> bool:
        if str(path) == "/.dockerenv":
            return True
        return original_exists(path)

    with (
        patch("blacki.zepto.client.Path.exists", docker_exists),
        patch(
            "blacki.zepto.client.create_bridge_server_parameters",
            return_value=server,
        ) as create_bridge,
    ):
        create_zepto_toolset(
            config_dir=store.config_dir,
            allowed_chat_ids=frozenset({"123"}),
        )

    assert create_bridge.call_args.kwargs["allow_npx"] is False
