"""Tests for OpenSandbox manager module."""

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

        result = await manager.get_or_create_sandbox(tool_context)

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
            result = await manager.get_or_create_sandbox(tool_context)

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
            result = await manager.get_or_create_sandbox(tool_context)

        assert result["sandbox"] == mock_sandbox
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_reconnect_fallback(self) -> None:
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
                side_effect=SandboxException("Connection failed"),
            ),
            patch(
                "blacki.sandbox.manager.Sandbox.create",
                new_callable=AsyncMock,
                return_value=mock_sandbox,
            ),
        ):
            result = await manager.get_or_create_sandbox(tool_context)

        assert result["sandbox"] == mock_sandbox
        assert tool_context.state["__sandbox_id__"] == "new-sandbox-id"

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_timeout_error(self) -> None:
        """Test handling SandboxReadyTimeoutException."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}

        from opensandbox.exceptions import SandboxReadyTimeoutException

        with patch(
            "blacki.sandbox.manager.Sandbox.create",
            new_callable=AsyncMock,
            side_effect=SandboxReadyTimeoutException("Timeout"),
        ):
            result = await manager.get_or_create_sandbox(tool_context)

        assert result["sandbox"] is None
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_or_create_sandbox_generic_error(self) -> None:
        """Test handling generic SandboxException."""
        config = SandboxConfig(enabled=True, domain="localhost:9090")
        manager = SandboxManager(config)
        tool_context = MagicMock()
        tool_context.state = {}

        from opensandbox.exceptions import SandboxException

        with patch(
            "blacki.sandbox.manager.Sandbox.create",
            new_callable=AsyncMock,
            side_effect=SandboxException("Generic error"),
        ):
            result = await manager.get_or_create_sandbox(tool_context)

        assert result["sandbox"] is None
        assert "Failed to create sandbox" in result["error"]

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing manager."""
        config = SandboxConfig(enabled=True)
        manager = SandboxManager(config)

        await manager.close()

    @pytest.mark.asyncio
    async def test_close_with_exception(self) -> None:
        """Test closing manager handles exceptions gracefully."""
        from unittest.mock import AsyncMock

        config = SandboxConfig(enabled=True)
        manager = SandboxManager(config)

        original_method = manager._connection_config.close_transport_if_owned
        error_mock = AsyncMock(side_effect=RuntimeError("Close failed"))
        object.__setattr__(
            manager._connection_config, "close_transport_if_owned", error_mock
        )

        await manager.close()

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
            result = await manager.get_or_create_sandbox(tool_context)

        assert result["sandbox"] is None
        assert "Unexpected error" in result["error"]


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
