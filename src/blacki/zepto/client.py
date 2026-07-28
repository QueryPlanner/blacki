"""Permission-checked stdio bridge for Zepto's hosted MCP server."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, cast

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment

ZEPTO_MCP_URL = "https://mcp.zepto.co.in/mcp"
ZEPTO_TOOL_PREFIX = "zepto"
FINAL_ORDER_PAYMENT_TOOL_NAMES = frozenset(
    {
        "create_online_payment_order",
        "create_order",
        "create_upi_reserve_pay_order",
        "create_wallet_order",
    }
)
REQUIRED_SCOPE = "tools:read"
MCP_REMOTE_PACKAGE_VERSION = "0.1.38"
# mcp-remote 0.1.38's bundled storage namespace still reports 0.1.37.
MCP_REMOTE_STORAGE_VERSION = "0.1.37"
DEFAULT_CONFIG_DIR = Path("data/credentials/zepto-mcp-remote")
BRIDGE_CONNECT_TIMEOUT_SECONDS = 15.0
_SERVER_HASH = hashlib.md5(  # noqa: S324 - compatibility identifier, not crypto
    ZEPTO_MCP_URL.encode(),
    usedforsecurity=False,
).hexdigest()
_CREDENTIAL_FILENAMES = frozenset(
    {
        f"{_SERVER_HASH}_client_info.json",
        f"{_SERVER_HASH}_tokens.json",
    }
)
_ALLOWED_FILENAMES = _CREDENTIAL_FILENAMES | {
    f"{_SERVER_HASH}_code_verifier.txt",
    f"{_SERVER_HASH}_lock.json",
}


class ZeptoCredentialError(RuntimeError):
    """Raised when the bridge or its private credentials are not ready."""


def _absolute(path: Path) -> Path:
    """Return an absolute path without resolving away symlink evidence."""
    return path if path.is_absolute() else Path.cwd() / path


def _reject_symlink_chain(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ZeptoCredentialError(
                "Zepto credential directories must not contain symlinks."
            )


def _validate_owner_and_mode(
    path: Path,
    *,
    expected_mode: int,
    label: str,
) -> os.stat_result:
    if path.is_symlink():
        raise ZeptoCredentialError(f"Zepto {label} must not be a symlink.")
    try:
        result = path.stat()
    except OSError as exc:
        raise ZeptoCredentialError(f"Zepto {label} is missing or unreadable.") from exc
    if hasattr(os, "geteuid") and result.st_uid != os.geteuid():
        raise ZeptoCredentialError(f"Zepto {label} must be owned by Blacki.")
    if stat.S_IMODE(result.st_mode) != expected_mode:
        raise ZeptoCredentialError(f"Zepto {label} must use mode {expected_mode:04o}.")
    return result


def _read_private_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZeptoCredentialError(f"Zepto {label} is corrupt.") from exc
    if not isinstance(value, dict):
        raise ZeptoCredentialError(f"Zepto {label} is corrupt.")
    return cast(dict[str, Any], value)


class McpRemoteCredentialStore:
    """Validate mcp-remote's dedicated, permission-protected Zepto state."""

    def __init__(self, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
        self.config_dir = _absolute(config_dir)
        self.version_dir = self.config_dir / f"mcp-remote-{MCP_REMOTE_STORAGE_VERSION}"

    def prepare_for_login(self) -> None:
        """Create only the private directories needed by interactive OAuth."""
        _reject_symlink_chain(self.config_dir)
        for directory in (self.config_dir, self.version_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                directory.chmod(0o700)
            except OSError as exc:
                raise ZeptoCredentialError(
                    "Could not prepare the private Zepto credential directory."
                ) from exc

    def validate_runtime_ready(self) -> None:
        """Require exact Zepto files, private permissions, and valid JSON shape."""
        _reject_symlink_chain(self.config_dir)
        config_stat = _validate_owner_and_mode(
            self.config_dir,
            expected_mode=0o700,
            label="credential directory",
        )
        version_stat = _validate_owner_and_mode(
            self.version_dir,
            expected_mode=0o700,
            label="versioned credential directory",
        )
        if not stat.S_ISDIR(config_stat.st_mode) or not stat.S_ISDIR(
            version_stat.st_mode
        ):
            raise ZeptoCredentialError("Zepto credential paths must be directories.")

        try:
            entries = list(self.version_dir.iterdir())
        except OSError as exc:
            raise ZeptoCredentialError(
                "Zepto credential directory is unreadable."
            ) from exc
        names = {entry.name for entry in entries}
        if not _CREDENTIAL_FILENAMES.issubset(names):
            raise ZeptoCredentialError(
                "Zepto is not authenticated. Run the Zepto login command."
            )
        unexpected = names - _ALLOWED_FILENAMES
        if unexpected:
            raise ZeptoCredentialError(
                "Zepto credential directory contains unexpected files."
            )

        for entry in entries:
            result = _validate_owner_and_mode(
                entry,
                expected_mode=0o600,
                label="credential file",
            )
            if not stat.S_ISREG(result.st_mode):
                raise ZeptoCredentialError(
                    "Zepto credential entries must be regular files."
                )

        client = _read_private_object(
            self.version_dir / f"{_SERVER_HASH}_client_info.json",
            "client registration",
        )
        tokens = _read_private_object(
            self.version_dir / f"{_SERVER_HASH}_tokens.json",
            "OAuth tokens",
        )
        if (
            not isinstance(client.get("client_id"), str)
            or not client["client_id"]
            or client.get("token_endpoint_auth_method") != "none"
        ):
            raise ZeptoCredentialError("Zepto client registration is corrupt.")
        token_scope = tokens.get("scope")
        if (
            not isinstance(tokens.get("access_token"), str)
            or not tokens["access_token"]
            or not isinstance(tokens.get("refresh_token"), str)
            or not tokens["refresh_token"]
            or str(tokens.get("token_type", "")).lower() != "bearer"
            or not isinstance(token_scope, str)
            or REQUIRED_SCOPE not in token_scope.split()
        ):
            raise ZeptoCredentialError("Zepto OAuth tokens are corrupt.")

    def tighten_permissions(self) -> None:
        """Restore mcp-remote's state to the documented private modes."""
        _reject_symlink_chain(self.config_dir)
        try:
            for directory in (self.config_dir, self.version_dir):
                if directory.exists():
                    directory.chmod(0o700)
            if self.version_dir.is_dir():
                for entry in self.version_dir.iterdir():
                    if entry.is_symlink():
                        raise ZeptoCredentialError(
                            "Zepto credential files must not be symlinks."
                        )
                    if entry.is_file():
                        entry.chmod(0o600)
        except OSError as exc:
            raise ZeptoCredentialError(
                "Could not protect Zepto credential permissions."
            ) from exc

    def clear_authentication(self) -> None:
        """Delete only Zepto bridge state after an explicit force-login request."""
        if not self.config_dir.exists():
            return
        self.tighten_permissions()
        config_stat = _validate_owner_and_mode(
            self.config_dir,
            expected_mode=0o700,
            label="credential directory",
        )
        if not stat.S_ISDIR(config_stat.st_mode):
            raise ZeptoCredentialError("Zepto credential path must be a directory.")
        if not self.version_dir.exists():
            return
        version_stat = _validate_owner_and_mode(
            self.version_dir,
            expected_mode=0o700,
            label="versioned credential directory",
        )
        if not stat.S_ISDIR(version_stat.st_mode):
            raise ZeptoCredentialError("Zepto credential path must be a directory.")
        entries = list(self.version_dir.iterdir())
        if {entry.name for entry in entries} - _ALLOWED_FILENAMES:
            raise ZeptoCredentialError(
                "Zepto credential directory contains unexpected files."
            )
        for entry in entries:
            result = _validate_owner_and_mode(
                entry,
                expected_mode=0o600,
                label="credential file",
            )
            if not stat.S_ISREG(result.st_mode):
                raise ZeptoCredentialError(
                    "Zepto credential entries must be regular files."
                )
        for entry in entries:
            entry.unlink()


def _locked_checkout_bridge() -> Path | None:
    candidate = (
        Path(__file__).resolve().parents[3]
        / "mcp-bridge"
        / "node_modules"
        / ".bin"
        / "mcp-remote"
    )
    return candidate if candidate.exists() else None


def _verified_bridge_binary(path: Path) -> str:
    """Resolve a bridge executable and require the exact pinned package."""
    try:
        resolved = path.resolve(strict=True)
        package_path = resolved.parent.parent / "package.json"
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZeptoCredentialError(
            "Installed mcp-remote package metadata is unreadable."
        ) from exc
    if (
        not isinstance(package, dict)
        or package.get("name") != "mcp-remote"
        or package.get("version") != MCP_REMOTE_PACKAGE_VERSION
    ):
        raise ZeptoCredentialError(
            f"mcp-remote must be exactly version {MCP_REMOTE_PACKAGE_VERSION}."
        )
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ZeptoCredentialError("Pinned mcp-remote executable is not runnable.")
    return str(resolved)


def _resolve_bridge_command(*, allow_npx: bool) -> tuple[str, list[str]]:
    locked_checkout = _locked_checkout_bridge()
    if locked_checkout is not None:
        return _verified_bridge_binary(locked_checkout), []
    installed = shutil.which("mcp-remote")
    if installed:
        return _verified_bridge_binary(Path(installed)), []
    if allow_npx:
        npx = shutil.which("npx")
        if npx:
            return (
                str(_absolute(Path(npx))),
                [
                    "--yes",
                    "--package",
                    f"mcp-remote@{MCP_REMOTE_PACKAGE_VERSION}",
                    "mcp-remote",
                ],
            )
    raise ZeptoCredentialError(
        "Pinned mcp-remote is unavailable. Install the documented bridge."
    )


def _bridge_environment(config_dir: Path) -> dict[str, str]:
    environment = get_default_environment()
    environment["MCP_REMOTE_CONFIG_DIR"] = str(_absolute(config_dir))
    # Local npx fallback must not read or forward credentials from ~/.npmrc.
    environment["npm_config_userconfig"] = os.devnull
    return environment


def create_bridge_server_parameters(
    *,
    config_dir: Path,
    auth_timeout_seconds: int,
    allow_npx: bool,
) -> StdioServerParameters:
    """Build the pinned bridge process without inheriting Blacki secrets."""
    command, prefix_args = _resolve_bridge_command(allow_npx=allow_npx)
    return StdioServerParameters(
        command=command,
        args=[
            *prefix_args,
            ZEPTO_MCP_URL,
            "--transport",
            "http-only",
            "--auth-timeout",
            str(auth_timeout_seconds),
            "--silent",
        ],
        env=_bridge_environment(config_dir),
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
