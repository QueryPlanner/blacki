"""Tests for Zepto bridge login and manifest probing."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp import StdioServerParameters

from blacki.zepto.auth import _list_tools, login, main, probe
from blacki.zepto.client import (
    _SERVER_HASH,
    REQUIRED_SCOPE,
    McpRemoteCredentialStore,
    ZeptoCredentialError,
)


def _write_private(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _make_ready(config_dir: Path) -> McpRemoteCredentialStore:
    store = McpRemoteCredentialStore(config_dir)
    store.prepare_for_login()
    _write_private(
        store.version_dir / f"{_SERVER_HASH}_client_info.json",
        {"client_id": "client-id", "token_endpoint_auth_method": "none"},
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


class _AsyncContext:
    def __init__(
        self,
        value: Any,
        *,
        on_enter: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.value = value
        self.on_enter = on_enter
        self.exited = False

    async def __aenter__(self) -> Any:
        if self.on_enter is not None:
            await self.on_enter()
        return self.value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        del exc_type, exc, traceback
        self.exited = True


class _FakeSession(_AsyncContext):
    def __init__(self, tool_names: list[str], *, fail: bool = False) -> None:
        super().__init__(self)
        self.tool_names = tool_names
        self.fail = fail
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> SimpleNamespace:
        if self.fail:
            raise RuntimeError("secret remote failure")
        assert self.initialized is True
        return SimpleNamespace(
            tools=[SimpleNamespace(name=name) for name in self.tool_names]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [True, False])
async def test_list_tools_uses_bridge_and_closes_both_contexts(
    tmp_path: Path,
    interactive: bool,
) -> None:
    config_dir = tmp_path / str(interactive)
    if not interactive:
        _make_ready(config_dir)
    stream_context = _AsyncContext((object(), object()))
    session = _FakeSession(["update_cart", "search_products"])

    async def make_credentials() -> None:
        _make_ready(config_dir)

    if interactive:
        stream_context.on_enter = make_credentials

    with (
        patch(
            "blacki.zepto.auth.create_bridge_server_parameters",
            return_value=StdioServerParameters(command="/bin/false"),
        ) as create_bridge,
        patch("blacki.zepto.auth.stdio_client", return_value=stream_context),
        patch("blacki.zepto.auth.ClientSession", return_value=session),
    ):
        tools = await _list_tools(config_dir, interactive_login=interactive)

    assert tools == ["search_products", "update_cart"]
    assert stream_context.exited is True
    assert session.exited is True
    assert create_bridge.call_args.kwargs["auth_timeout_seconds"] == (
        300 if interactive else 2
    )


@pytest.mark.asyncio
async def test_list_tools_hides_bridge_failure_and_closes_process(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "zepto"
    _make_ready(config_dir)
    stream_context = _AsyncContext((object(), object()))
    session = _FakeSession([], fail=True)
    with (
        patch(
            "blacki.zepto.auth.create_bridge_server_parameters",
            return_value=StdioServerParameters(command="/bin/false"),
        ),
        patch("blacki.zepto.auth.stdio_client", return_value=stream_context),
        patch("blacki.zepto.auth.ClientSession", return_value=session),
        pytest.raises(ZeptoCredentialError, match="could not connect") as error,
    ):
        await probe(config_dir)

    assert "secret remote failure" not in str(error.value)
    assert stream_context.exited is True
    assert session.exited is True


@pytest.mark.asyncio
async def test_interrupted_login_restores_private_permissions(tmp_path: Path) -> None:
    config_dir = tmp_path / "zepto"

    async def write_partial_state() -> None:
        store = _make_ready(config_dir)
        store.config_dir.chmod(0o755)
        store.version_dir.chmod(0o755)
        for entry in store.version_dir.iterdir():
            entry.chmod(0o644)

    stream_context = _AsyncContext(
        (object(), object()),
        on_enter=write_partial_state,
    )
    with (
        patch(
            "blacki.zepto.auth.create_bridge_server_parameters",
            return_value=StdioServerParameters(command="/bin/false"),
        ),
        patch("blacki.zepto.auth.stdio_client", return_value=stream_context),
        patch(
            "blacki.zepto.auth.ClientSession", return_value=_FakeSession([], fail=True)
        ),
        pytest.raises(ZeptoCredentialError, match="could not connect"),
    ):
        await _list_tools(config_dir, interactive_login=True)

    store = McpRemoteCredentialStore(config_dir)
    store.validate_runtime_ready()


@pytest.mark.asyncio
async def test_login_and_probe_delegate_to_manifest_flow(tmp_path: Path) -> None:
    with patch(
        "blacki.zepto.auth._list_tools",
        new=AsyncMock(return_value=["search_products"]),
    ) as list_tools:
        assert await login(tmp_path) == ["search_products"]
        assert await probe(tmp_path) == ["search_products"]

    assert list_tools.await_args_list[0].kwargs == {"interactive_login": True}
    assert list_tools.await_args_list[1].kwargs == {"interactive_login": False}


@pytest.mark.asyncio
async def test_force_login_clears_only_bridge_state(tmp_path: Path) -> None:
    store = _make_ready(tmp_path)
    with (
        patch.object(store, "clear_authentication") as clear,
        patch("blacki.zepto.auth.McpRemoteCredentialStore", return_value=store),
        patch(
            "blacki.zepto.auth._list_tools",
            new=AsyncMock(return_value=["search_products"]),
        ),
    ):
        assert await login(tmp_path, force=True) == ["search_products"]
    clear.assert_called_once_with()


@pytest.mark.asyncio
async def test_real_stdio_process_is_terminated_after_probe(tmp_path: Path) -> None:
    config_dir = tmp_path / "credentials"
    _make_ready(config_dir)
    marker = tmp_path / "child-closed"
    server_script = tmp_path / "fake_mcp.py"
    server_script.write_text(
        "import json, pathlib, sys\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    method = message.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': '2025-06-18', "
        "'capabilities': {}, 'serverInfo': {'name': 'fake', 'version': '1'}}\n"
        "    elif method == 'tools/list':\n"
        "        result = {'tools': []}\n"
        "    else:\n"
        "        continue\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': message['id'], "
        "'result': result}), flush=True)\n"
        "marker.write_text('closed')\n",
        encoding="utf-8",
    )
    parameters = StdioServerParameters(
        command=os.fspath(Path(sys.executable)),
        args=[str(server_script)],
    )

    with patch(
        "blacki.zepto.auth.create_bridge_server_parameters",
        return_value=parameters,
    ):
        assert await probe(config_dir) == []

    assert marker.read_text(encoding="utf-8") == "closed"


def test_main_status_error_and_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = tmp_path / "credentials"
    assert main(["--config-dir", str(config_dir), "status"]) == 1
    assert json.loads(capsys.readouterr().out) == {"authenticated": False}

    _make_ready(config_dir)
    assert main(["--config-dir", str(config_dir), "status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"authenticated": True}

    with patch(
        "blacki.zepto.auth.probe",
        new=AsyncMock(side_effect=ZeptoCredentialError("authenticate again")),
    ):
        assert main(["--config-dir", str(config_dir), "probe"]) == 1
    output = capsys.readouterr().out
    assert "authenticate again" in output
    assert "access-token" not in output


def test_main_login_and_probe_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = tmp_path / "credentials"
    with patch(
        "blacki.zepto.auth.login",
        new=AsyncMock(return_value=["search_products"]),
    ) as login_mock:
        assert main(["--config-dir", str(config_dir), "login", "--force"]) == 0
    login_mock.assert_awaited_once_with(config_dir, force=True)
    assert json.loads(capsys.readouterr().out) == {
        "authenticated": True,
        "tools": ["search_products"],
    }

    with patch(
        "blacki.zepto.auth.probe",
        new=AsyncMock(return_value=["update_cart"]),
    ):
        assert main(["--config-dir", str(config_dir), "probe"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "authenticated": True,
        "tools": ["update_cart"],
    }
