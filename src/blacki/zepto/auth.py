"""Interactive login and non-mutating manifest probe for Zepto MCP."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from .client import (
    BRIDGE_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_CONFIG_DIR,
    McpRemoteCredentialStore,
    ZeptoCredentialError,
    create_bridge_server_parameters,
)


async def _list_tools(
    config_dir: Path,
    *,
    interactive_login: bool,
) -> list[str]:
    store = McpRemoteCredentialStore(config_dir)
    if interactive_login:
        store.prepare_for_login()
    else:
        store.validate_runtime_ready()
    server = create_bridge_server_parameters(
        config_dir=store.config_dir,
        auth_timeout_seconds=300 if interactive_login else 2,
        allow_npx=True,
    )
    try:
        async with asyncio.timeout(
            310 if interactive_login else BRIDGE_CONNECT_TIMEOUT_SECONDS
        ):
            async with (
                stdio_client(server) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                manifest = await session.list_tools()
    except Exception as exc:
        with suppress(ZeptoCredentialError):
            store.tighten_permissions()
        raise ZeptoCredentialError(
            "Zepto bridge could not connect. Run the login command again."
        ) from exc

    store.tighten_permissions()
    store.validate_runtime_ready()
    return sorted(tool.name for tool in manifest.tools)


async def login(config_dir: Path, *, force: bool = False) -> list[str]:
    """Open Zepto OAuth through the supported bridge and list all tools."""
    if force:
        McpRemoteCredentialStore(config_dir).clear_authentication()
    return await _list_tools(config_dir, interactive_login=True)


async def probe(config_dir: Path) -> list[str]:
    """List tools using stored credentials without calling a Zepto tool."""
    return await _list_tools(config_dir, interactive_login=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Authenticate or probe Zepto MCP.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Private mcp-remote directory shared with the Blacki runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    login_parser = subparsers.add_parser(
        "login", help="Run one-time Zepto OAuth in a browser."
    )
    login_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete only the stored Zepto bridge credentials before login.",
    )
    subparsers.add_parser("probe", help="List tools with stored credentials.")
    subparsers.add_parser("status", help="Validate local credentials only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "login":
            tools = asyncio.run(login(args.config_dir, force=args.force))
            print(json.dumps({"authenticated": True, "tools": tools}, indent=2))
        elif args.command == "probe":
            tools = asyncio.run(probe(args.config_dir))
            print(json.dumps({"authenticated": True, "tools": tools}, indent=2))
        else:
            McpRemoteCredentialStore(args.config_dir).validate_runtime_ready()
            print(json.dumps({"authenticated": True}))
    except ZeptoCredentialError as exc:
        if args.command == "status":
            print(json.dumps({"authenticated": False}))
        else:
            print(f"Zepto authentication error: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module command
    raise SystemExit(main())
