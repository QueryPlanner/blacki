"""Browser tool contracts through real ADK and sandbox routing."""

import asyncio
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.adk.tools import ToolContext
from google.adk.tools.function_tool import FunctionTool
from opensandbox.exceptions import SandboxException
from opensandbox.models.execd import (
    Execution,
    ExecutionError,
    ExecutionLogs,
    OutputMessage,
    RunCommandOpts,
)

from blacki.tools.agent_browser import agent_browser
from blacki.tools.registry import ToolConfig, build_tools


@pytest.mark.parametrize("enabled", [False, True])
async def test_browser_tool_is_available_only_with_sandbox(enabled: bool) -> None:
    """Registration exposes a usable ADK tool only with sandbox capability."""
    tools = build_tools(ToolConfig(sandbox_enabled=enabled, weather_enabled=False))
    browser_tools = [tool for tool in tools if tool is agent_browser]
    assert len(browser_tools) == int(enabled)
    if enabled:
        declaration = FunctionTool(browser_tools[0])._get_declaration()
        assert declaration is not None
        assert declaration.name == "agent_browser"
        schema = declaration.parameters_json_schema
        assert schema is not None
        assert schema["required"] == ["args"]
        assert schema["properties"]["args"]["type"] == "array"
        assert schema["properties"]["args"]["items"]["type"] == "string"


async def test_browser_arguments_remain_literal_in_the_remote_shell(
    browser_context: ToolContext,
    browser_commands: tuple[MagicMock, MagicMock],
    tmp_path: Path,
) -> None:
    """The transport command preserves argv without executing shell metacharacters."""
    commands, connect = browser_commands
    executable = tmp_path / "agent-browser"
    executable.write_text("#!/bin/sh\nprintf '%s\\0' \"$@\"\n")
    executable.chmod(0o700)
    marker = tmp_path / "injected"
    args = [
        "eval",
        "hello world",
        "a'b",
        'a"b',
        f"; touch {marker}",
        f"$(touch {marker})",
        f"`touch {marker}`",
        "a\nb",
        "",
        "--session",
        "other",
    ]

    async def run_stub(command: str, *, opts: RunCommandOpts) -> Execution:
        assert opts.timeout == timedelta(seconds=90)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            env={"PATH": str(tmp_path)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode == 0, stderr.decode()
        return Execution(
            logs=ExecutionLogs(
                stdout=[OutputMessage(text=stdout.decode(), timestamp=0)]
            )
        )

    commands.run.side_effect = run_stub
    result = await FunctionTool(agent_browser).run_async(
        args={"args": args},
        tool_context=browser_context,
    )

    assert result["status"] == "success"
    assert result["output"].split("\x00")[:-1] == [
        "--session",
        "conversation-a",
        "--content-boundaries",
        *args,
    ]
    assert not marker.exists()
    assert connect.await_args.args == ("sandbox-a",)
    assert browser_context.state["__sandbox_id__"] == "sandbox-a"


@pytest.mark.parametrize("args", [[], "open", None, [42], ["open", "bad\x00url"]])
async def test_invalid_browser_arguments_never_reach_the_sandbox(
    args: object,
    browser_context: ToolContext,
    browser_commands: tuple[MagicMock, MagicMock],
) -> None:
    commands, connect = browser_commands
    result = await FunctionTool(agent_browser).run_async(
        args={"args": args},
        tool_context=browser_context,
    )
    assert result["status"] == "error"
    assert result["output"] is None
    connect.assert_not_awaited()
    commands.run.assert_not_awaited()


@pytest.mark.parametrize("failure", ["disabled", "command", "transport"])
async def test_browser_preserves_sandbox_errors(
    failure: str,
    browser_context: ToolContext,
    browser_commands: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, connect = browser_commands
    if failure == "disabled":
        monkeypatch.setenv("SANDBOX_ENABLED", "false")
    elif failure == "transport":
        commands.run.side_effect = SandboxException("sandbox unavailable")
    else:
        commands.run.return_value = Execution(
            error=ExecutionError(
                name="CommandError", value="agent-browser: not found", timestamp=0
            ),
        )
    result = await agent_browser(["--help"], browser_context)
    assert result["status"] == "error"
    assert result["error"]
    if failure == "disabled":
        connect.assert_not_awaited()
    else:
        commands.run.assert_awaited_once()
        assert (
            "sandbox unavailable" if failure == "transport" else "not found"
        ) in result["error"]
