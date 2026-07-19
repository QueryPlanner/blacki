# mypy: disable-error-code="no-untyped-def"
"""Tests for server configuration."""

import sys
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_dependencies() -> Generator[MagicMock]:
    """Mock external dependencies to prevent side effects during import."""
    with (
        patch("google.adk.cli.fast_api.get_fast_api_app") as mock_get_app,
        patch("blacki.utils.initialize_environment") as mock_init_env,
        patch("blacki.utils.configure_otel_resource"),
        patch("openinference.instrumentation.google_adk.GoogleADKInstrumentor"),
        patch("blacki.utils.setup_logging"),
    ):
        mock_env = MagicMock()
        mock_env.session_uri = None
        mock_env.allow_origins_list = ["*"]
        mock_env.serve_web_interface = True
        mock_env.reload_agents = False
        mock_env.sqlite_path = None
        mock_env.agent_dir = "src"

        mock_env.host = "127.0.0.1"
        mock_env.port = 8080

        mock_init_env.return_value = mock_env

        yield mock_get_app


def test_server_session_service_uri_is_none(mock_dependencies: MagicMock) -> None:
    """Verify session_service_uri is None for default SQLite sessions."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server  # noqa: F401

    mock_dependencies.assert_called_once()
    call_kwargs = mock_dependencies.call_args[1]

    assert call_kwargs["session_service_uri"] is None


@pytest.mark.asyncio
async def test_server_lifespan_closes_search_clients(
    mock_dependencies: MagicMock,
) -> None:
    """Verify both managed search clients are closed during shutdown."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    container = MagicMock()
    container.initialize_all_storages = AsyncMock()
    init_container = AsyncMock(return_value=container)
    close_container = AsyncMock()
    close_brave = AsyncMock()
    close_exa = AsyncMock()
    close_notify = AsyncMock()

    with (
        patch.object(server, "init_container", new=init_container),
        patch.object(server, "close_container", new=close_container),
        patch.object(server, "_start_telegram_bot", new=AsyncMock()),
        patch.object(server, "_stop_telegram_bot", new=AsyncMock()),
        patch.object(server, "_stop_reminder_scheduler", new=AsyncMock()),
        patch.object(server.validation, "validate_configuration", return_value=[]),
        patch("blacki.tools.close_shared_brave_search_client", new=close_brave),
        patch("blacki.search.close_shared_exa_search_client", new=close_exa),
        patch("blacki.callbacks.close_shared_notify_client", new=close_notify),
    ):
        async with server.lifespan(server.app):
            pass

    container.initialize_all_storages.assert_awaited_once()
    close_container.assert_awaited_once()
    close_brave.assert_awaited_once()
    close_exa.assert_awaited_once()
    close_notify.assert_awaited_once()
