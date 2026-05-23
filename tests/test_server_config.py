# mypy: disable-error-code="no-untyped-def"
"""Tests for server configuration."""

import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

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
