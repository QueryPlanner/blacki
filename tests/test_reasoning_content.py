"""Tests for the DeepSeek reasoning content preservation plugin."""

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from blacki.adk_runtime import DeepSeekReasoningPlugin


class MockAgent:
    """Mock agent for callback context."""

    class MockModel:
        def __init__(self, model_name: str) -> None:
            self.model = model_name

    def __init__(self, model_name: str) -> None:
        self.model = self.MockModel(model_name)
        self.name = "test_agent"


class MockCallbackContext(CallbackContext):
    """Mock callback context."""

    def __init__(self, agent: MockAgent) -> None:
        self.agent = agent


def _create_part(text: str, thought: bool) -> types.Part:
    part = types.Part.from_text(text=text)
    part.thought = thought
    return part


def test_deepseek_reasoning_plugin_preserves_thoughts(monkeypatch: Any) -> None:
    """Test that the plugin converts thought parts to <think> text parts."""
    monkeypatch.setenv("MODEL_ID", "openrouter/deepseek/deepseek-r1")
    plugin = DeepSeekReasoningPlugin(name="deepseek_reasoning")

    # Create mock request with thoughts
    content = types.Content(
        role="assistant",
        parts=[
            _create_part(text="I am thinking...", thought=True),
            _create_part(text="This is my final answer.", thought=False),
        ]
    )
    request = LlmRequest(contents=[content])

    # Run plugin with a deepseek model
    agent = MockAgent("openrouter/deepseek/deepseek-r1")
    ctx = MockCallbackContext(agent)

    plugin.before_model(ctx, request)

    # Verify the thought was nested in the content block
    assert len(request.contents) == 1
    assert request.contents[0].parts is not None
    assert len(request.contents[0].parts) == 1

    final_text = request.contents[0].parts[0].text
    assert final_text is not None
    assert final_text.startswith("<think>\nI am thinking...\n</think>\n")
    assert "This is my final answer." in final_text

    # Verify no part has thought=True
    for p in request.contents[0].parts:
        assert getattr(p, "thought", False) is False


def test_deepseek_reasoning_plugin_ignores_non_deepseek_models(
    monkeypatch: Any,
) -> None:
    """Test that the plugin does nothing for non-DeepSeek models."""
    monkeypatch.setenv("MODEL_ID", "google/gemini-2.5-flash")
    plugin = DeepSeekReasoningPlugin(name="deepseek_reasoning")

    content = types.Content(
        role="assistant",
        parts=[
            _create_part(text="I am thinking...", thought=True),
            _create_part(text="This is my final answer.", thought=False),
        ]
    )
    request = LlmRequest(contents=[content])

    agent = MockAgent("google/gemini-2.5-flash")
    ctx = MockCallbackContext(agent)

    plugin.before_model(ctx, request)

    # Verify the thought was NOT modified
    assert request.contents[0].parts is not None
    assert len(request.contents[0].parts) == 2
    assert getattr(request.contents[0].parts[0], "thought", False) is True

