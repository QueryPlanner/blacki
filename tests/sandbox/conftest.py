"""Real ADK context and external OpenSandbox boundary for browser tests."""

from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import ToolContext
from opensandbox import Sandbox
from opensandbox.services.command import Commands

from blacki.sandbox.manager import reset_sandbox_manager


@pytest.fixture
async def browser_context(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[ToolContext]:
    """Select an existing session sandbox through the actual ADK state object."""
    await reset_sandbox_manager()
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    session = Session(
        id="conversation-a",
        user_id="user-a",
        app_name="blacki",
        state={"__sandbox_id__": "sandbox-a"},
    )
    yield ToolContext(
        InvocationContext(
            invocation_id="browser-test",
            session=session,
            session_service=InMemorySessionService(),
        )
    )
    await reset_sandbox_manager()


@pytest.fixture
def browser_commands() -> Iterator[tuple[MagicMock, MagicMock]]:
    """Replace only the remote sandbox connection and command service."""
    commands = create_autospec(Commands, instance=True, spec_set=True)
    sandbox = create_autospec(Sandbox, instance=True, spec_set=True)
    sandbox.commands = commands
    with patch.object(Sandbox, "connect", autospec=True, spec_set=True) as connect:
        connect.return_value = sandbox
        yield commands, connect
