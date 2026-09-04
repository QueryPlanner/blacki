"""Tests for sandbox code interpreter."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opensandbox.exceptions import SandboxException

from blacki.tools.sandbox_code import sandbox_execute_code


@pytest.fixture
def mock_sandbox_manager() -> Generator[MagicMock, None, None]:
    """Mock sandbox manager."""
    with patch("blacki.tools.sandbox_code.get_sandbox_manager") as mock:
        manager = MagicMock()
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_sandbox_execute_code_success(mock_sandbox_manager: MagicMock) -> None:
    """Test successful code execution."""
    mock_sandbox = MagicMock()
    mock_sandbox_manager.get_or_create_sandbox = AsyncMock(
        return_value={
            "sandbox": mock_sandbox,
            "error": None,
        }
    )

    mock_interpreter = MagicMock()
    mock_execution = MagicMock()

    mock_msg1 = MagicMock()
    mock_msg1.text = "output1"
    mock_execution.logs.stdout = [mock_msg1]
    mock_execution.logs.stderr = []

    mock_result1 = MagicMock()
    mock_result1.text = "result1"
    mock_execution.result = [mock_result1]
    mock_execution.error = None

    mock_interpreter.codes.run = AsyncMock(return_value=mock_execution)

    with patch(
        "blacki.tools.sandbox_code.CodeInterpreter.create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_interpreter

        tool_context = MagicMock()
        tool_context.state = {}

        result = await sandbox_execute_code("print('test')", tool_context)

        assert result["status"] == "success"
        assert "output1" in result["output"]
        assert "result1" in result["output"]


@pytest.mark.asyncio
async def test_sandbox_execute_code_with_stderr(
    mock_sandbox_manager: MagicMock,
) -> None:
    """Test code execution with stderr."""
    mock_sandbox = MagicMock()
    mock_sandbox_manager.get_or_create_sandbox = AsyncMock(
        return_value={
            "sandbox": mock_sandbox,
            "error": None,
        }
    )

    mock_interpreter = MagicMock()
    mock_execution = MagicMock()

    mock_execution.logs.stdout = []
    mock_err_msg = MagicMock()
    mock_err_msg.text = "stderr1"
    mock_execution.logs.stderr = [mock_err_msg]
    mock_execution.result = []
    mock_execution.error = None

    mock_interpreter.codes.run = AsyncMock(return_value=mock_execution)

    with patch(
        "blacki.tools.sandbox_code.CodeInterpreter.create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_interpreter

        tool_context = MagicMock()
        tool_context.state = {}

        result = await sandbox_execute_code("print('test')", tool_context)

        assert result["status"] == "success"
        assert "stderr1" in result["output"]


@pytest.mark.asyncio
async def test_sandbox_execute_code_timeout(
    mock_sandbox_manager: MagicMock,
) -> None:
    """Test code execution timeout."""
    mock_sandbox = MagicMock()
    mock_sandbox_manager.get_or_create_sandbox = AsyncMock(
        return_value={
            "sandbox": mock_sandbox,
            "error": None,
        }
    )

    mock_interpreter = MagicMock()
    mock_interpreter.codes.run = AsyncMock(side_effect=TimeoutError())

    with patch(
        "blacki.tools.sandbox_code.CodeInterpreter.create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_interpreter

        tool_context = MagicMock()
        tool_context.state = {}

        result = await sandbox_execute_code("while True: pass", tool_context, timeout=1)

        assert result["status"] == "error"
        assert "timed out after 1 seconds" in result["error"]


@pytest.mark.asyncio
async def test_sandbox_execute_code_with_execution_error(
    mock_sandbox_manager: MagicMock,
) -> None:
    """Test code execution with execution error."""
    mock_sandbox = MagicMock()
    mock_sandbox_manager.get_or_create_sandbox = AsyncMock(
        return_value={
            "sandbox": mock_sandbox,
            "error": None,
        }
    )

    mock_interpreter = MagicMock()
    mock_execution = MagicMock()

    mock_execution.logs.stdout = []
    mock_execution.logs.stderr = []
    mock_execution.result = []
    mock_err = MagicMock()
    mock_err.name = "SyntaxError"
    mock_err.value = "invalid syntax"
    mock_execution.error = mock_err

    mock_interpreter.codes.run = AsyncMock(return_value=mock_execution)

    with patch(
        "blacki.tools.sandbox_code.CodeInterpreter.create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_interpreter

        tool_context = MagicMock()
        tool_context.state = {}

        result = await sandbox_execute_code("invalid python", tool_context)

        assert result["status"] == "error"
        assert "SyntaxError: invalid syntax" in result["error"]


@pytest.mark.asyncio
async def test_sandbox_execute_code_manager_error(
    mock_sandbox_manager: MagicMock,
) -> None:
    """Test code execution when sandbox manager returns an error."""
    mock_sandbox_manager.get_or_create_sandbox = AsyncMock(
        return_value={
            "sandbox": None,
            "error": "Failed to create sandbox",
        }
    )

    tool_context = MagicMock()
    tool_context.state = {}

    result = await sandbox_execute_code("print('test')", tool_context)

    assert result["status"] == "error"
    assert result["error"] == "Failed to create sandbox"


@pytest.mark.asyncio
async def test_sandbox_execute_code_sandbox_exception(
    mock_sandbox_manager: MagicMock,
) -> None:
    """Test handling of SandboxException during execution."""
    mock_sandbox = MagicMock()
    mock_sandbox_manager.get_or_create_sandbox = AsyncMock(
        return_value={
            "sandbox": mock_sandbox,
            "error": None,
        }
    )

    with patch(
        "blacki.tools.sandbox_code.CodeInterpreter.create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = SandboxException("API error")

        tool_context = MagicMock()
        tool_context.state = {}

        result = await sandbox_execute_code("print('test')", tool_context)

        assert result["status"] == "error"
        assert "Sandbox code execution error" in result["error"]


@pytest.mark.asyncio
async def test_sandbox_execute_code_unexpected_exception(
    mock_sandbox_manager: MagicMock,
) -> None:
    """Test handling of unexpected exception during execution."""
    mock_sandbox = MagicMock()
    mock_sandbox_manager.get_or_create_sandbox = AsyncMock(
        return_value={
            "sandbox": mock_sandbox,
            "error": None,
        }
    )

    with patch(
        "blacki.tools.sandbox_code.CodeInterpreter.create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = RuntimeError("Unexpected error")

        tool_context = MagicMock()
        tool_context.state = {}

        result = await sandbox_execute_code("print('test')", tool_context)

        assert result["status"] == "error"
        assert "Unexpected error in code execution" in result["error"]
