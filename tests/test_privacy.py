"""Tests for private-tool logging and tracing privacy controls."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from google.adk.tools.base_tool import BaseTool

from blacki.privacy import (
    PrivacyAwareLoggingPlugin,
    configure_private_tool_privacy,
    configure_zepto_privacy,
    is_private_tool,
    kokoro_tts_enabled,
    private_tool_privacy_enabled,
    zepto_mcp_enabled,
)


def _tool(name: str) -> MagicMock:
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    return tool


def test_configure_zepto_privacy_is_explicit_and_forces_safe_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZEPTO_MCP_ENABLED", "false")
    assert zepto_mcp_enabled() is False
    assert configure_zepto_privacy() is False

    monkeypatch.setenv("ZEPTO_MCP_ENABLED", "yes")
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "true")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    assert zepto_mcp_enabled() is True
    assert configure_zepto_privacy() is True
    assert (
        logging.getLogger("google_adk.google.adk.tools.mcp_tool").level == logging.INFO
    )
    assert logging.getLogger("google.adk.tools.mcp_tool").level == logging.INFO
    import os

    assert os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] == "false"
    assert os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"


def test_private_tool_identification_uses_zepto_prefix() -> None:
    assert is_private_tool(_tool("zepto_search_products")) is True
    assert is_private_tool(_tool("send_text_to_speech")) is True
    assert is_private_tool(_tool("search_products")) is False


def test_kokoro_tts_enables_content_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZEPTO_MCP_ENABLED", "false")
    monkeypatch.delenv("KOKORO_TTS_BASE_URL", raising=False)
    assert kokoro_tts_enabled() is False
    assert private_tool_privacy_enabled() is False
    assert configure_private_tool_privacy() is False

    monkeypatch.setenv("KOKORO_TTS_BASE_URL", " http://kokoro.internal:8880 ")
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "true")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    assert kokoro_tts_enabled() is True
    assert private_tool_privacy_enabled() is True
    assert configure_private_tool_privacy() is True
    import os

    assert os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] == "false"
    assert os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"


def test_secure_zepto_app_removes_content_logging_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Secure mode must not register ADK's content-printing plugin."""
    from blacki.agent import create_app, root_agent

    monkeypatch.setenv("ZEPTO_MCP_ENABLED", "true")
    secure_app = create_app(root_agent)

    assert secure_app.plugins is not None
    assert "logging_plugin" not in {plugin.name for plugin in secure_app.plugins}


@pytest.mark.asyncio
async def test_privacy_logging_plugin_redacts_private_payloads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin = PrivacyAwareLoggingPlugin()
    context = MagicMock()
    context.function_call_id = "call-1"
    context.agent_name = "blacki"
    secret_args = {"address": "private-address"}
    secret_result = {"phone": "private-phone"}
    secret_error = RuntimeError("private-error")
    tool = _tool("zepto_update_cart")

    assert (
        await plugin.before_tool_callback(
            tool=tool,
            tool_args=secret_args,
            tool_context=context,
        )
        is None
    )
    assert (
        await plugin.after_tool_callback(
            tool=tool,
            tool_args=secret_args,
            tool_context=context,
            result=secret_result,
        )
        is None
    )
    assert (
        await plugin.on_tool_error_callback(
            tool=tool,
            tool_args=secret_args,
            tool_context=context,
            error=secret_error,
        )
        is None
    )

    output = capsys.readouterr().out
    assert "zepto_update_cart" in output
    for private_value in ("private-address", "private-phone", "private-error"):
        assert private_value not in output


@pytest.mark.asyncio
async def test_privacy_logging_plugin_redacts_tts_payloads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin = PrivacyAwareLoggingPlugin()
    context = MagicMock()
    context.function_call_id = "call-tts"
    context.agent_name = "blacki"
    tool = _tool("send_text_to_speech")

    await plugin.before_tool_callback(
        tool=tool,
        tool_args={"text": "private speech"},
        tool_context=context,
    )
    await plugin.after_tool_callback(
        tool=tool,
        tool_args={"text": "private speech"},
        tool_context=context,
        result={"status": "success"},
    )

    output = capsys.readouterr().out
    assert "send_text_to_speech" in output
    assert "private speech" not in output


@pytest.mark.asyncio
async def test_privacy_logging_plugin_preserves_normal_tool_logging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin = PrivacyAwareLoggingPlugin()
    context = MagicMock()
    context.function_call_id = "call-2"
    context.agent_name = "blacki"

    await plugin.before_tool_callback(
        tool=_tool("weather"),
        tool_args={"city": "Pune"},
        tool_context=context,
    )
    await plugin.after_tool_callback(
        tool=_tool("weather"),
        tool_args={"city": "Pune"},
        tool_context=context,
        result={"temperature": 25},
    )
    await plugin.on_tool_error_callback(
        tool=_tool("weather"),
        tool_args={"city": "Pune"},
        tool_context=context,
        error=RuntimeError("weather failed"),
    )

    output = capsys.readouterr().out
    assert "Pune" in output
    assert "temperature" in output
    assert "weather failed" in output
