"""Single-root agent composition and Telegram failure recovery."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from cryptography.fernet import Fernet
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from blacki.agent import create_agent, create_app
from blacki.runtime.adk import AdkRuntime, SessionLocator
from blacki.telegram.progress_callbacks import recover_telegram_tool_error


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("legacy_flag", [None, "true", "false"])
def test_root_agent_has_direct_tools_without_delegation(
    private: bool,
    legacy_flag: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old worker flags cannot restore delegation or broaden private tools."""
    if legacy_flag is None:
        monkeypatch.delenv("TASK_WORKER_ENABLED", raising=False)
    else:
        monkeypatch.setenv("TASK_WORKER_ENABLED", legacy_flag)
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    monkeypatch.setenv("KOKORO_TTS_BASE_URL", "http://kokoro.test:8880")
    monkeypatch.setenv("GOOGLE_HEALTH_CLIENT_ID", "test-client")
    monkeypatch.setenv("GOOGLE_HEALTH_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv(
        "GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode()
    )
    agent = create_agent(include_user_scoped_tools=private)
    names = {
        getattr(tool, "name", None) or getattr(tool, "__name__", None)
        for tool in agent.tools
    }
    assert agent.sub_agents == []
    assert "task_worker" not in names
    assert "agent_browser" in names
    assert "preload_memory" in names
    assert ("send_text_to_speech" in names) is private
    assert ("get_health_summary" in names) is private
    assert agent.on_tool_error_callback is (
        recover_telegram_tool_error if private else None
    )


def _function_call_response(
    *, call_id: str, name: str, args: dict[str, str]
) -> LlmResponse:
    """Build one deterministic model function-call response."""
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=call_id,
                        name=name,
                        args=args,
                    )
                )
            ],
        )
    )


class ToolFailureLlm(BaseLlm):
    """Deterministic model that continues after a failed Telegram tool."""

    calls: int = 0
    saw_recoverable_tool_error: bool = False

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Call one failing tool, then produce a final response."""
        del stream
        self.calls += 1
        if self.calls <= 2:
            yield _function_call_response(
                call_id=f"tool-failure-{self.calls}",
                name="failing_tool",
                args={},
            )
            return

        self.saw_recoverable_tool_error = any(
            part.function_response is not None
            and isinstance(part.function_response.response, dict)
            and part.function_response.response.get("status") == "error"
            for content in llm_request.contents
            for part in content.parts or []
        )
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Recovered after tool failure.")],
            )
        )


@pytest.mark.asyncio
async def test_telegram_tool_failure_does_not_fail_the_root_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception from a Telegram tool returns control to the model."""

    tool_invocations = 0

    def failing_tool(tool_context: ToolContext) -> dict[str, str]:
        nonlocal tool_invocations
        del tool_context
        tool_invocations += 1
        raise RuntimeError("sandbox endpoint unavailable")

    model = ToolFailureLlm(model="tool-failure-test")
    agent = create_agent(include_user_scoped_tools=True)
    agent.model = model
    agent.tools.append(failing_tool)
    assert agent.on_tool_error_callback is recover_telegram_tool_error

    runtime = AdkRuntime(
        InMemorySessionService(),
        agent_app=create_app(agent),
    )
    response = await runtime.run_user_turn_with_thoughts(
        locator=SessionLocator(
            user_id="telegram-user",
            session_id_prefix="telegram-tool-failure",
        ),
        message_text="Use the failing tool.",
        state={"telegram_chat_id": "123"},
    )

    assert model.calls == 3
    assert tool_invocations == 1
    assert model.saw_recoverable_tool_error is True
    assert response.content == "Recovered after tool failure."
