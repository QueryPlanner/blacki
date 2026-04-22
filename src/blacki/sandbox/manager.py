"""Sandbox manager for lifecycle management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.exceptions import (
    SandboxException,
    SandboxReadyTimeoutException,
)

from .config import SANDBOX_STATE_KEY, SandboxConfig, load_sandbox_config

if TYPE_CHECKING:
    from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

_manager: SandboxManager | None = None


def get_sandbox_manager() -> SandboxManager:
    """Get or create the singleton SandboxManager instance."""
    global _manager
    if _manager is None:
        config = load_sandbox_config()
        _manager = SandboxManager(config)
    return _manager


async def reset_sandbox_manager() -> None:
    """Reset the singleton manager (for testing)."""
    global _manager
    if _manager is not None:
        await _manager.close()
        _manager = None


class SandboxManager:
    """Manages sandbox lifecycle with session-scoped sandboxes.

    Uses lazy creation: sandboxes are created on first tool call,
    then reused within the same session via ToolContext.state.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._connection_config = ConnectionConfig(
            domain=config.domain,
            api_key=config.api_key,
        )

    @property
    def config(self) -> SandboxConfig:
        """Get the sandbox configuration."""
        return self._config

    async def get_or_create_sandbox(self, tool_context: ToolContext) -> dict[str, Any]:
        """Get existing sandbox or create a new one for this session.

        Args:
            tool_context: ADK tool context with session state.

        Returns:
            Dict with sandbox instance or error information.
        """
        if not self._config.enabled:
            return {
                "error": (
                    "Sandbox tools are disabled. Set SANDBOX_ENABLED=true to enable."
                ),
                "sandbox": None,
            }

        sandbox_id = tool_context.state.get(SANDBOX_STATE_KEY)
        if sandbox_id:
            try:
                sandbox = await Sandbox.connect(
                    sandbox_id, connection_config=self._connection_config
                )
                logger.debug("Reconnected to existing sandbox: %s", sandbox_id)
                return {"sandbox": sandbox, "error": None}
            except SandboxException as e:
                logger.warning(
                    "Failed to reconnect to sandbox %s, creating new one: %s",
                    sandbox_id,
                    e,
                )

        try:
            sandbox = await Sandbox.create(
                self._config.image,
                connection_config=self._connection_config,
                entrypoint=self._config.entrypoint,
                timeout=self._config.timeout,
                resource=self._config.resource,
            )
            tool_context.state[SANDBOX_STATE_KEY] = sandbox.id
            logger.info("Created new sandbox: %s", sandbox.id)
            return {"sandbox": sandbox, "error": None}
        except SandboxReadyTimeoutException as e:
            error_msg = (
                f"Sandbox startup timed out. "
                f"Ensure opensandbox-server is running at {self._config.domain}. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            return {"sandbox": None, "error": error_msg}
        except SandboxException as e:
            error_msg = f"Failed to create sandbox: {e}"
            logger.exception(error_msg)
            return {"sandbox": None, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error creating sandbox: {e}"
            logger.exception(error_msg)
            return {"sandbox": None, "error": error_msg}

    async def close(self) -> None:
        """Close the connection config transport if owned."""
        try:
            await self._connection_config.close_transport_if_owned()
        except Exception:
            logger.exception("Error closing connection config transport")
