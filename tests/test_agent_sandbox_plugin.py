"""Tests for sandbox-specific app plugin wiring."""

from unittest.mock import patch

from blacki.agent import create_app, root_agent


def test_create_app_registers_multimodal_sandbox_bridge() -> None:
    with patch("blacki.sandbox.config.load_sandbox_config") as load_config:
        load_config.return_value.enabled = True
        app = create_app(root_agent)

    assert app.plugins is not None
    assert "sandbox_multimodal_results" in {plugin.name for plugin in app.plugins}
    assert app.plugins[0].name == "sandbox_multimodal_results"
