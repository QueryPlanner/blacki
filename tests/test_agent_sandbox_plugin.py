"""Tests for sandbox-specific app plugin wiring."""

from unittest.mock import patch

from google.adk.plugins.logging_plugin import LoggingPlugin

from blacki.agent import create_app, root_agent


def test_create_app_registers_multimodal_sandbox_bridge() -> None:
    with patch("blacki.sandbox.config.load_sandbox_config") as load_config:
        load_config.return_value.enabled = True
        app = create_app(root_agent)

    assert app.plugins is not None
    assert "sandbox_multimodal_results" in {plugin.name for plugin in app.plugins}
    assert app.plugins[0].name == "sandbox_multimodal_results"


def test_create_app_does_not_install_print_based_logging_plugin() -> None:
    """ADK's plugin prints content directly and bypasses structured logging."""
    app = create_app(root_agent)

    assert app.plugins is not None
    assert not any(isinstance(plugin, LoggingPlugin) for plugin in app.plugins)
