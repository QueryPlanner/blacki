"""Tests for OpenSandbox manager module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blacki.sandbox.config import SandboxConfig
from blacki.sandbox.manager import (
    SandboxManager,
    get_sandbox_manager,
    reset_sandbox_manager,
)


class TestSandboxManager:
    """Tests for SandboxManager."""

    def test_init(self) -> None:
        """Test manager initialization."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)

        assert manager.config == config

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_disabled(self) -> None:
        """Test error when sandbox is disabled."""
        config = SandboxConfig(enabled=False)
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}

        result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] is None
        assert "disabled" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_create_new(self) -> None:
        """Test creating a new sandbox."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "test-sandbox-id"

        with patch(
            "blacki.sandbox.manager.Sandbox.create",
            new_callable=AsyncMock,
            return_value=mock_sandbox,
        ):
            result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] == mock_sandbox
        assert result["error"] is None
        assert tool_context.state["__sandbox_id__"] == "test-sandbox-id"

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_reuse_existing(self) -> None:
        """Test reusing an existing sandbox."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {"__sandbox_id__": "existing-sandbox-id"}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "existing-sandbox-id"

        with patch(
            "blacki.sandbox.manager.Sandbox.connect",
            new_callable=AsyncMock,
            return_value=mock_sandbox,
        ):
            result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] == mock_sandbox
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_reconnect_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test fallback when reconnect fails."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {"__sandbox_id__": "dead-sandbox-id"}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "new-sandbox-id"

        from opensandbox.exceptions import SandboxException

        with (
            patch(
                "blacki.sandbox.manager.Sandbox.connect",
                new_callable=AsyncMock,
                side_effect=SandboxException("reconnect-credential-canary"),
            ),
            patch(
                "blacki.sandbox.manager.Sandbox.create",
                new_callable=AsyncMock,
                return_value=mock_sandbox,
            ),
        ):
            result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] == mock_sandbox
        assert tool_context.state["__sandbox_id__"] == "new-sandbox-id"
        assert "reconnect-credential-canary" not in caplog.text

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_timeout_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handling SandboxReadyTimeoutException."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}

        from opensandbox.exceptions import SandboxReadyTimeoutException

        with patch(
            "blacki.sandbox.manager.Sandbox.create",
            new_callable=AsyncMock,
            side_effect=SandboxReadyTimeoutException("timeout-credential-canary"),
        ):
            result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] is None
        assert "timed out" in result["error"].lower()
        assert "timeout-credential-canary" not in result["error"]
        assert "timeout-credential-canary" not in caplog.text

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_generic_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test handling generic SandboxException."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}

        from opensandbox.exceptions import SandboxException

        with patch(
            "blacki.sandbox.manager.Sandbox.create",
            new_callable=AsyncMock,
            side_effect=SandboxException("sdk-credential-canary"),
        ):
            result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] is None
        assert "Failed to create sandbox" in result["error"]
        assert "sdk-credential-canary" not in result["error"]
        assert "sdk-credential-canary" not in caplog.text

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_never_injects_credentials(self) -> None:
        """The sandbox creation request must always omit process credentials."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "test-sandbox-id"

        with patch(
            "blacki.sandbox.manager.Sandbox.create",
            new_callable=AsyncMock,
            return_value=mock_sandbox,
        ) as mock_create:
            result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] == mock_sandbox
        assert result["error"] is None

        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["env"] is None

    @pytest.mark.asyncio
    async def test_sdk_error_details_are_redacted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Tool responses and logs must not echo SDK exception details."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}
        canary = "credential-canary-value"

        with patch(
            "blacki.sandbox.manager.Sandbox.create",
            new_callable=AsyncMock,
            side_effect=RuntimeError(canary),
        ):
            result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] is None
        assert canary not in result["error"]
        assert canary not in caplog.text

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing manager."""
        config = SandboxConfig(enabled=True)
        manager = SandboxManager(config)

        await manager.close()

    @pytest.mark.asyncio
    async def test_close_with_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test closing manager handles exceptions gracefully."""
        from unittest.mock import AsyncMock

        config = SandboxConfig(enabled=True)
        manager = SandboxManager(config)

        original_method = manager._connection_config.close_transport_if_owned
        error_mock = AsyncMock(side_effect=RuntimeError("close-credential-canary"))
        object.__setattr__(
            manager._connection_config, "close_transport_if_owned", error_mock
        )

        await manager.close()
        assert "close-credential-canary" not in caplog.text

        object.__setattr__(
            manager._connection_config, "close_transport_if_owned", original_method
        )

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_unexpected_exception(self) -> None:
        """Test handling unexpected exception during sandbox creation."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}

        with patch(
            "blacki.sandbox.manager.Sandbox.create",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = await manager.get_or_create_sandbox(tool_context.state)

        assert result["sandbox"] is None
        assert "Unexpected error" in result["error"]

    @pytest.mark.asyncio
    async def test_clear_gmail_artifacts_deletes_only_matching_files(self) -> None:
        manager = SandboxManager(SandboxConfig(enabled=True, domain="localhost:9090"))
        state = {"__sandbox_id__": "sandbox-1"}
        sandbox = MagicMock()
        sandbox.files.search = AsyncMock(
            return_value=[
                SimpleNamespace(path="/workspace/uploads/gmail-result-1.json"),
                SimpleNamespace(path="/workspace/uploads/gmail-2-invoice.pdf"),
            ]
        )
        sandbox.files.delete_files = AsyncMock()
        with patch.object(
            manager,
            "get_or_create_sandbox",
            new_callable=AsyncMock,
            return_value={"sandbox": sandbox, "error": None},
        ):
            await manager.clear_gmail_artifacts(state)

        sandbox.files.search.assert_awaited_once()
        sandbox.files.delete_files.assert_awaited_once_with(
            [
                "/workspace/uploads/gmail-result-1.json",
                "/workspace/uploads/gmail-2-invoice.pdf",
            ]
        )

    @pytest.mark.asyncio
    async def test_clear_gmail_artifacts_handles_absent_sandbox_and_errors(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        manager = SandboxManager(SandboxConfig(enabled=True, domain="localhost:9090"))
        await manager.clear_gmail_artifacts({})

        with patch.object(
            manager,
            "get_or_create_sandbox",
            new_callable=AsyncMock,
            return_value={"sandbox": None, "error": "disabled"},
        ):
            await manager.clear_gmail_artifacts({"__sandbox_id__": "sandbox-1"})

        sandbox = MagicMock()
        sandbox.files.search = AsyncMock(side_effect=RuntimeError("file secret"))
        with (
            patch.object(
                manager,
                "get_or_create_sandbox",
                new_callable=AsyncMock,
                return_value={"sandbox": sandbox, "error": None},
            ),
            caplog.at_level("WARNING", logger="blacki.sandbox.manager"),
        ):
            await manager.clear_gmail_artifacts({"__sandbox_id__": "sandbox-1"})
        assert "file secret" not in caplog.text

        sandbox.files.search = AsyncMock(return_value=[])
        with patch.object(
            manager,
            "get_or_create_sandbox",
            new_callable=AsyncMock,
            return_value={"sandbox": sandbox, "error": None},
        ):
            await manager.clear_gmail_artifacts({"__sandbox_id__": "sandbox-1"})
        sandbox.files.delete_files.assert_not_called()


class TestGetSandboxManager:
    """Tests for get_sandbox_manager singleton."""

    def test_returns_singleton(self) -> None:
        """Test that get_sandbox_manager returns the same instance."""
        manager1 = get_sandbox_manager()
        manager2 = get_sandbox_manager()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_reset_sandbox_manager(self) -> None:
        """Test reset_sandbox_manager clears the singleton."""
        manager1 = get_sandbox_manager()

        await reset_sandbox_manager()

        manager2 = get_sandbox_manager()

        assert manager1 is not manager2

    @pytest.mark.asyncio
    async def test_reset_sandbox_manager_when_none(self) -> None:
        """Test reset_sandbox_manager when manager is None."""
        await reset_sandbox_manager()

        await reset_sandbox_manager()
