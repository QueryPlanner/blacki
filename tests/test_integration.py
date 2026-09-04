# mypy: disable-error-code="no-untyped-def"
"""Integration tests for agent configuration and component wiring.

This module validates the basic structure and wiring of ADK app components.
Tests are pattern-based and validate integration points regardless of specific
implementation choices (plugins, tools, etc.).

Future: Container-based smoke tests for CI/CD will be added here.
"""

from collections.abc import Sequence
from typing import Any, Protocol, cast

from google.adk.apps.app import EventsCompactionConfig

from blacki import app
from blacki.agent import (
    AUTO_COMPACTION_EVENT_RETENTION_SIZE,
    AUTO_COMPACTION_TOKEN_THRESHOLD,
    _build_model,
)
from blacki.observability.costs import CostAwareLiteLLMClient


class AgentConfigLike(Protocol):
    """Minimal agent surface needed for integration assertions."""

    name: str
    model: Any
    instruction: str | None
    description: str | None
    tools: Sequence[object] | None


def as_agent_config(agent: object) -> AgentConfigLike:
    """Treat runtime agent instances as a typed config surface."""
    return cast(AgentConfigLike, agent)


class TestAppIntegration:
    """Pattern-based integration tests for App configuration and wiring."""

    def test_app_is_properly_instantiated(self) -> None:
        """Verify app container is properly instantiated."""
        assert app is not None
        assert app.name is not None
        assert isinstance(app.name, str)
        assert len(app.name) > 0

    def test_app_has_root_agent(self) -> None:
        """Verify app is wired to root agent."""
        assert app.root_agent is not None

    def test_app_uses_token_based_context_compaction(self) -> None:
        """Keep long conversations bounded without replacing their sessions."""
        config = app.events_compaction_config

        assert isinstance(config, EventsCompactionConfig)
        assert config.token_threshold == AUTO_COMPACTION_TOKEN_THRESHOLD
        assert config.event_retention_size == AUTO_COMPACTION_EVENT_RETENTION_SIZE
        assert config.overlap_size == 0
        assert config.compaction_interval > config.token_threshold

    def test_app_plugins_are_valid_if_configured(self) -> None:
        """Verify plugins (if any) are properly initialized."""
        # Plugins are optional - if configured, they should be a list
        if app.plugins is not None:
            assert isinstance(app.plugins, list)
            # Each plugin should be an object instance
            for plugin in app.plugins:
                assert plugin is not None
                assert hasattr(plugin, "__class__")

    def test_prompt_plugins_preserve_precedence_order(self) -> None:
        """Safety leads and stored preferences follow developer policy/schema."""
        assert app.plugins is not None
        plugin_names = [plugin.name for plugin in app.plugins]

        assert plugin_names == [
            "telegram_model_override",
            "global_instruction",
            "domain_policy",
            "declarative_db",
            "stored_preferences",
            "response_policy",
            "logging_plugin",
        ]


class TestAgentIntegration:
    """Pattern-based integration tests for Agent configuration."""

    def test_agent_has_required_configuration(self) -> None:
        """Verify agent has required configuration fields."""
        agent = app.root_agent
        assert agent is not None
        typed_agent = as_agent_config(agent)

        # Required: agent name
        assert typed_agent.name is not None
        assert isinstance(typed_agent.name, str)
        assert len(typed_agent.name) > 0

        # Required: agent model
        assert typed_agent.model is not None
        # model can be a string name or a model object (e.g. LiteLlm)
        if isinstance(typed_agent.model, str):
            assert len(typed_agent.model) > 0
        else:
            # If it's an object, it should have a model attribute that is a string
            assert hasattr(typed_agent.model, "model")
            assert isinstance(typed_agent.model.model, str)
            assert len(typed_agent.model.model) > 0

    def test_agent_instructions_are_valid_if_configured(self) -> None:
        """Verify agent instructions (if configured) are valid strings."""
        agent = app.root_agent
        assert agent is not None
        typed_agent = as_agent_config(agent)

        # Instruction is optional - if configured, should be non-empty string
        if typed_agent.instruction is not None:
            assert isinstance(typed_agent.instruction, str)
            assert len(typed_agent.instruction) > 0

        # Description is optional - if configured, should be non-empty string
        if typed_agent.description is not None:
            assert isinstance(typed_agent.description, str)
            assert len(typed_agent.description) > 0

    def test_agent_tools_are_valid_if_configured(self) -> None:
        """Verify agent tools (if any) are properly initialized."""
        agent = app.root_agent
        assert agent is not None
        typed_agent = as_agent_config(agent)

        # Tools are optional - if configured, should be a list
        if typed_agent.tools is not None:
            assert isinstance(typed_agent.tools, list)
            # Each tool should be an object instance
            for tool in typed_agent.tools:
                assert tool is not None
                assert hasattr(tool, "__class__")

    def test_memory_tools_are_registered(self) -> None:
        """Verify memory tools stay available to ADK."""
        agent = app.root_agent
        assert agent is not None
        typed_agent = as_agent_config(agent)

        assert typed_agent.tools is not None
        tool_names = {
            getattr(tool, "name", None) or getattr(tool, "__name__", "")
            for tool in typed_agent.tools
        }

        assert "save_memory" in tool_names
        assert "search_memory" in tool_names

    def test_litellm_models_use_cost_aware_client(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("ROOT_AGENT_MODEL", "google/gemini-2.5-flash")

        model = _build_model()

        assert not isinstance(model, str)
        assert isinstance(model.llm_client, CostAwareLiteLLMClient)
