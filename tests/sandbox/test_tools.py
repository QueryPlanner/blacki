"""Tests for OpenSandbox tools module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opensandbox.exceptions import SandboxException

from blacki.sandbox.tools import (
    _format_command_output,
    sandbox_enabled,
    sandbox_list_files,
    sandbox_read_file,
    sandbox_run_command,
    sandbox_send_file_to_user,
    sandbox_write_file,
)


class TestFormatCommandOutput:
    """Tests for _format_command_output helper."""

    def test_stdout_only(self) -> None:
        """Test output with stdout only."""
        execution = MagicMock()
        execution.logs.stdout = [MagicMock(text="Hello"), MagicMock(text="World")]
        execution.logs.stderr = []
        execution.error = None

        result = _format_command_output(execution)

        assert result == "Hello\nWorld"

    def test_stdout_and_stderr(self) -> None:
        """Test output with both stdout and stderr."""
        execution = MagicMock()
        execution.logs.stdout = [MagicMock(text="Output")]
        execution.logs.stderr = [MagicMock(text="Warning")]
        execution.error = None

        result = _format_command_output(execution)

        assert "Output" in result
        assert "[stderr]" in result
        assert "Warning" in result

    def test_with_error(self) -> None:
        """Test output with error."""
        execution = MagicMock()
        execution.logs.stdout = [MagicMock(text="Output")]
        execution.logs.stderr = []
        execution.error = MagicMock()
        execution.error.name = "RuntimeError"
        execution.error.value = "Something went wrong"

        result = _format_command_output(execution)

        assert "Output" in result
        assert "[error]" in result
        assert "RuntimeError" in result

    def test_no_output(self) -> None:
        """Test output with no content."""
        execution = MagicMock()
        execution.logs.stdout = []
        execution.logs.stderr = []
        execution.error = None

        result = _format_command_output(execution)

        assert result == "(no output)"


class TestSandboxRunCommand:
    """Tests for sandbox_run_command tool."""

    @pytest.mark.asyncio
    async def test_disabled_sandbox(self) -> None:
        """Test error when sandbox is disabled."""
        tool_context = MagicMock()
        tool_context.state = {}

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": None, "error": "Sandbox tools are disabled"}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_run_command("ls", tool_context)

        assert result["status"] == "error"
        assert "disabled" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_command(self) -> None:
        """Test successful command execution."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_execution = MagicMock()
        mock_execution.logs.stdout = [MagicMock(text="file1.txt\nfile2.txt")]
        mock_execution.logs.stderr = []
        mock_execution.error = None
        mock_sandbox.commands.run = AsyncMock(return_value=mock_execution)

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_run_command("ls", tool_context)

        assert result["status"] == "success"
        assert "file1.txt" in result["output"]

    @pytest.mark.asyncio
    async def test_command_with_error(self) -> None:
        """Test command execution with error."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_execution = MagicMock()
        mock_execution.logs.stdout = []
        mock_execution.logs.stderr = [MagicMock(text="Command not found")]
        mock_execution.error = MagicMock()
        mock_execution.error.name = "ExitError"
        mock_execution.error.value = "127"
        mock_sandbox.commands.run = AsyncMock(return_value=mock_execution)

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_run_command("invalidcmd", tool_context)

        assert result["status"] == "error"
        assert "Command failed" in result["error"]

    @pytest.mark.asyncio
    async def test_command_sandbox_exception(self) -> None:
        """Test command with SandboxException."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.commands.run = AsyncMock(
            side_effect=SandboxException("Command failed")
        )

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_run_command("ls", tool_context)

        assert result["status"] == "error"
        assert "Sandbox error" in result["error"]

    @pytest.mark.asyncio
    async def test_command_unexpected_exception(self) -> None:
        """Test command with unexpected exception."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.commands.run = AsyncMock(side_effect=RuntimeError("Unexpected"))

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_run_command("ls", tool_context)

        assert result["status"] == "error"
        assert "Unexpected error" in result["error"]


class TestSandboxWriteFile:
    """Tests for sandbox_write_file tool."""

    @pytest.mark.asyncio
    async def test_successful_write(self) -> None:
        """Test successful file write."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.files.write_file = AsyncMock()

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_write_file("/tmp/test.txt", "Hello", tool_context)

        assert result["status"] == "success"
        assert result["path"] == "/tmp/test.txt"
        assert result["bytes_written"] == 5

    @pytest.mark.asyncio
    async def test_write_disabled_sandbox(self) -> None:
        """Test write when sandbox is disabled."""
        tool_context = MagicMock()
        tool_context.state = {}

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": None, "error": "Sandbox tools are disabled"}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_write_file("/tmp/test.txt", "Hello", tool_context)

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_write_sandbox_exception(self) -> None:
        """Test write with SandboxException."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.files.write_file = AsyncMock(
            side_effect=SandboxException("Write failed")
        )

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_write_file("/tmp/test.txt", "Hello", tool_context)

        assert result["status"] == "error"
        assert "Failed to write file" in result["error"]

    @pytest.mark.asyncio
    async def test_write_unexpected_exception(self) -> None:
        """Test write with unexpected exception."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.files.write_file = AsyncMock(
            side_effect=RuntimeError("Unexpected")
        )

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_write_file("/tmp/test.txt", "Hello", tool_context)

        assert result["status"] == "error"
        assert "Unexpected error" in result["error"]


class TestSandboxReadFile:
    """Tests for sandbox_read_file tool."""

    @pytest.mark.asyncio
    async def test_successful_read(self) -> None:
        """Test successful file read."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_file = AsyncMock(return_value="File content")

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_read_file("/tmp/test.txt", tool_context)

        assert result["status"] == "success"
        assert result["content"] == "File content"
        assert result["path"] == "/tmp/test.txt"

    @pytest.mark.asyncio
    async def test_read_disabled_sandbox(self) -> None:
        """Test read when sandbox is disabled."""
        tool_context = MagicMock()
        tool_context.state = {}

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": None, "error": "Sandbox tools are disabled"}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_read_file("/tmp/test.txt", tool_context)

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_read_sandbox_exception(self) -> None:
        """Test read with SandboxException."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_file = AsyncMock(
            side_effect=SandboxException("Read failed")
        )

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_read_file("/tmp/test.txt", tool_context)

        assert result["status"] == "error"
        assert "Failed to read file" in result["error"]

    @pytest.mark.asyncio
    async def test_read_unexpected_exception(self) -> None:
        """Test read with unexpected exception."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_file = AsyncMock(side_effect=RuntimeError("Unexpected"))

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_read_file("/tmp/test.txt", tool_context)

        assert result["status"] == "error"
        assert "Unexpected error" in result["error"]


class TestSandboxListFiles:
    """Tests for sandbox_list_files tool."""

    @pytest.mark.asyncio
    async def test_successful_list(self) -> None:
        """Test successful file listing."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_entry = MagicMock()
        mock_entry.path = "/workspace/main.py"
        mock_entry.size = 1024
        mock_entry.modified_at.isoformat.return_value = "2025-01-01T00:00:00"

        mock_sandbox = MagicMock()
        mock_sandbox.files.search = AsyncMock(return_value=[mock_entry])

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_list_files("/workspace", "*.py", tool_context)

        assert result["status"] == "success"
        assert len(result["files"]) == 1
        assert result["files"][0]["path"] == "/workspace/main.py"

    @pytest.mark.asyncio
    async def test_list_disabled_sandbox(self) -> None:
        """Test list when sandbox is disabled."""
        tool_context = MagicMock()
        tool_context.state = {}

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": None, "error": "Sandbox tools are disabled"}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_list_files("/workspace", "*.py", tool_context)

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_list_sandbox_exception(self) -> None:
        """Test list with SandboxException."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.files.search = AsyncMock(
            side_effect=SandboxException("Search failed")
        )

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_list_files("/workspace", "*.py", tool_context)

        assert result["status"] == "error"
        assert "Failed to list files" in result["error"]

    @pytest.mark.asyncio
    async def test_list_unexpected_exception(self) -> None:
        """Test list with unexpected exception."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.files.search = AsyncMock(side_effect=RuntimeError("Unexpected"))

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_list_files("/workspace", "*.py", tool_context)

        assert result["status"] == "error"
        assert "Unexpected error" in result["error"]


class TestSandboxEnabled:
    """Tests for sandbox_enabled function."""

    def test_enabled_true(self) -> None:
        """Test when sandbox is enabled."""
        with patch("blacki.sandbox.config.load_sandbox_config") as mock_load:
            mock_load.return_value = MagicMock(enabled=True)

            result = sandbox_enabled()

        assert result is True

    def test_enabled_false(self) -> None:
        """Test when sandbox is disabled."""
        with patch("blacki.sandbox.config.load_sandbox_config") as mock_load:
            mock_load.return_value = MagicMock(enabled=False)

            result = sandbox_enabled()

        assert result is False


class TestSandboxSendFileToUser:
    """Tests for sandbox_send_file_to_user tool."""

    @pytest.mark.asyncio
    async def test_disabled_sandbox(self) -> None:
        """Test when sandbox is disabled."""
        tool_context = MagicMock()
        tool_context.state = {}

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": None, "error": "Sandbox disabled"}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_send_file_to_user("/tmp/test.txt", tool_context)

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_not_in_telegram_session(self) -> None:
        """Test when not in a Telegram session."""
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_send_file_to_user("/tmp/test.txt", tool_context)

        assert result["status"] == "error"
        assert "Not in a Telegram session" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_telegram_token(self) -> None:
        """Test when TELEGRAM_BOT_TOKEN is not set."""
        tool_context = MagicMock()
        tool_context.state = {"telegram_chat_id": "12345"}

        mock_sandbox = MagicMock()

        with patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            with patch.dict("os.environ", {}, clear=True):
                result = await sandbox_send_file_to_user("/tmp/test.txt", tool_context)

        assert result["status"] == "error"
        assert "TELEGRAM_BOT_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_send(self) -> None:
        """Test successful file send."""
        tool_context = MagicMock()
        tool_context.state = {"telegram_chat_id": "12345"}

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_bytes = AsyncMock(return_value=b"file contents")

        with (
            patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager,
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}),
            patch("blacki.telegram.api.TelegramApiClient") as mock_api_client_cls,
        ):
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            mock_api = MagicMock()
            mock_api.send_document = AsyncMock()
            mock_api_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_api
            )
            mock_api_client_cls.return_value.__aexit__ = AsyncMock()

            result = await sandbox_send_file_to_user("/tmp/test.txt", tool_context)

        assert result["status"] == "success"
        assert "test.txt" in result["message"]
        mock_api.send_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_sandbox_exception(self) -> None:
        """Test with SandboxException during read."""
        tool_context = MagicMock()
        tool_context.state = {"telegram_chat_id": "12345"}

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_bytes = AsyncMock(
            side_effect=SandboxException("Read failed")
        )

        with (
            patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager,
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}),
        ):
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_send_file_to_user("/tmp/test.txt", tool_context)

        assert result["status"] == "error"
        assert "Failed to read file from sandbox" in result["error"]

    @pytest.mark.asyncio
    async def test_unexpected_exception(self) -> None:
        """Test with unexpected exception."""
        tool_context = MagicMock()
        tool_context.state = {"telegram_chat_id": "12345"}

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_bytes = AsyncMock(
            side_effect=RuntimeError("Unexpected")
        )

        with (
            patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager,
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}),
        ):
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            result = await sandbox_send_file_to_user("/tmp/test.txt", tool_context)

        assert result["status"] == "error"
        assert "Unexpected error" in result["error"]

    @pytest.mark.asyncio
    async def test_with_thread_id(self) -> None:
        """Test successful file send with thread_id."""
        tool_context = MagicMock()
        tool_context.state = {
            "telegram_chat_id": "12345",
            "telegram_thread_id": "67890",
        }

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_bytes = AsyncMock(return_value=b"file contents")

        with (
            patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager,
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}),
            patch("blacki.telegram.api.TelegramApiClient") as mock_api_client_cls,
        ):
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            mock_api = MagicMock()
            mock_api.send_document = AsyncMock()
            mock_api_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_api
            )
            mock_api_client_cls.return_value.__aexit__ = AsyncMock()

            result = await sandbox_send_file_to_user("/tmp/test.txt", tool_context)

        assert result["status"] == "success"
        call_kwargs = mock_api.send_document.call_args.kwargs
        assert call_kwargs["message_thread_id"] == 67890

    @pytest.mark.asyncio
    async def test_send_png_file(self) -> None:
        """Test sending a PNG image file."""
        tool_context = MagicMock()
        tool_context.state = {"telegram_chat_id": "12345"}

        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_bytes = AsyncMock(return_value=png_bytes)

        with (
            patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager,
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}),
            patch("blacki.telegram.api.TelegramApiClient") as mock_api_client_cls,
        ):
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            mock_api = MagicMock()
            mock_api.send_document = AsyncMock()
            mock_api_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_api
            )
            mock_api_client_cls.return_value.__aexit__ = AsyncMock()

            result = await sandbox_send_file_to_user("/tmp/image.png", tool_context)

        assert result["status"] == "success"
        assert "image.png" in result["message"]
        call_kwargs = mock_api.send_document.call_args.kwargs
        assert call_kwargs["filename"] == "image.png"
        assert call_kwargs["document_bytes"] == png_bytes

    @pytest.mark.asyncio
    async def test_send_jpeg_file(self) -> None:
        """Test sending a JPEG image file."""
        tool_context = MagicMock()
        tool_context.state = {"telegram_chat_id": "12345"}

        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00"

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_bytes = AsyncMock(return_value=jpeg_bytes)

        with (
            patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager,
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}),
            patch("blacki.telegram.api.TelegramApiClient") as mock_api_client_cls,
        ):
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            mock_api = MagicMock()
            mock_api.send_document = AsyncMock()
            mock_api_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_api
            )
            mock_api_client_cls.return_value.__aexit__ = AsyncMock()

            result = await sandbox_send_file_to_user("/tmp/photo.jpg", tool_context)

        assert result["status"] == "success"
        assert "photo.jpg" in result["message"]
        call_kwargs = mock_api.send_document.call_args.kwargs
        assert call_kwargs["filename"] == "photo.jpg"
        assert call_kwargs["document_bytes"] == jpeg_bytes

    @pytest.mark.asyncio
    async def test_send_pdf_file(self) -> None:
        """Test sending a PDF file."""
        tool_context = MagicMock()
        tool_context.state = {"telegram_chat_id": "12345"}

        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog"

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_bytes = AsyncMock(return_value=pdf_bytes)

        with (
            patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager,
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}),
            patch("blacki.telegram.api.TelegramApiClient") as mock_api_client_cls,
        ):
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            mock_api = MagicMock()
            mock_api.send_document = AsyncMock()
            mock_api_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_api
            )
            mock_api_client_cls.return_value.__aexit__ = AsyncMock()

            result = await sandbox_send_file_to_user("/tmp/document.pdf", tool_context)

        assert result["status"] == "success"
        assert "document.pdf" in result["message"]
        call_kwargs = mock_api.send_document.call_args.kwargs
        assert call_kwargs["filename"] == "document.pdf"
        assert call_kwargs["document_bytes"] == pdf_bytes

    @pytest.mark.asyncio
    async def test_send_text_file_regression(self) -> None:
        """Test that text files still work after binary file support."""
        tool_context = MagicMock()
        tool_context.state = {"telegram_chat_id": "12345"}

        text_bytes = b"Hello, World!\nThis is a text file."

        mock_sandbox = MagicMock()
        mock_sandbox.files.read_bytes = AsyncMock(return_value=text_bytes)

        with (
            patch("blacki.sandbox.tools.get_sandbox_manager") as mock_get_manager,
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}),
            patch("blacki.telegram.api.TelegramApiClient") as mock_api_client_cls,
        ):
            manager = MagicMock()
            manager.get_or_create_sandbox = AsyncMock(
                return_value={"sandbox": mock_sandbox, "error": None}
            )
            mock_get_manager.return_value = manager

            mock_api = MagicMock()
            mock_api.send_document = AsyncMock()
            mock_api_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_api
            )
            mock_api_client_cls.return_value.__aexit__ = AsyncMock()

            result = await sandbox_send_file_to_user("/tmp/test.txt", tool_context)

        assert result["status"] == "success"
        assert "test.txt" in result["message"]
        call_kwargs = mock_api.send_document.call_args.kwargs
        assert call_kwargs["filename"] == "test.txt"
        assert call_kwargs["document_bytes"] == text_bytes
