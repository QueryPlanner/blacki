"""Behavior tests for always-on live Telegram tool progress."""

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import MockBaseTool, MockState, MockToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.genai.types import Content, FunctionCall, Part

import blacki.callbacks as callbacks_module
from blacki.callbacks import (
    block_telegram_tool_retry,
    clear_telegram_tool_failure,
    notify_telegram_after_agent,
    notify_telegram_after_model,
    notify_telegram_before_tool,
    recover_telegram_tool_error,
    reset_telegram_tool_notify_rate_limiter_for_tests,
    telegram_live_tool_progress_enabled,
)
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiError


@pytest.fixture(autouse=True)
async def reset_live_tool_progress() -> AsyncIterator[None]:
    await reset_telegram_tool_notify_rate_limiter_for_tests()
    yield None
    await reset_telegram_tool_notify_rate_limiter_for_tests()


def configure_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "1s"), (4.9, "4s"), (65, "1m 5s"), (125, "2m 5s")],
)
def test_format_elapsed_duration(seconds: float, expected: str) -> None:
    assert callbacks_module._format_elapsed_duration(seconds) == expected


def test_live_progress_requires_configured_telegram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert telegram_live_tool_progress_enabled() is False

    configure_telegram(monkeypatch)
    monkeypatch.setenv("TELEGRAM_TOOL_NOTIFICATIONS", "false")
    monkeypatch.setenv("TELEGRAM_TOOL_PROGRESS_MODE", "off")
    assert telegram_live_tool_progress_enabled() is True


def test_telegram_tool_failure_becomes_a_recoverable_result() -> None:
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))

    result = recover_telegram_tool_error(
        cast(BaseTool, MockBaseTool("restore_user_file")),
        {"object_id": "opaque"},
        cast(ToolContext, context),
        RuntimeError("sandbox endpoint unavailable"),
    )

    assert result == {
        "status": "error",
        "error": (
            "The restore_user_file tool is unavailable right now. "
            "Do not retry it in this turn. Continue without it when possible "
            "and explain the limitation to the user."
        ),
    }


def test_telegram_tool_failure_is_suppressed_for_the_current_invocation() -> None:
    context = MockToolContext(
        invocation_id="turn-1",
        state=MockState({"telegram_chat_id": "42"}),
    )
    tool = cast(BaseTool, MockBaseTool("restore_user_file"))

    clear_telegram_tool_failure(cast(CallbackContext, context))
    recover_telegram_tool_error(tool, {}, cast(ToolContext, context), RuntimeError())
    blocked = block_telegram_tool_retry(
        tool,
        {},
        cast(ToolContext, context),
    )

    assert blocked == {
        "status": "error",
        "error": (
            "The restore_user_file tool already failed in this turn. "
            "Do not call it again. Continue without it when possible "
            "and explain the limitation to the user."
        ),
    }

    clear_telegram_tool_failure(cast(CallbackContext, context))
    assert block_telegram_tool_retry(tool, {}, cast(ToolContext, context)) is None


def test_telegram_tool_failure_recovery_skips_non_telegram_sessions() -> None:
    result = recover_telegram_tool_error(
        cast(BaseTool, MockBaseTool("restore_user_file")),
        {},
        cast(ToolContext, MockToolContext(state=MockState({}))),
        RuntimeError("sandbox endpoint unavailable"),
    )

    assert result is None
    assert (
        block_telegram_tool_retry(
            cast(BaseTool, MockBaseTool("restore_user_file")),
            {},
            cast(ToolContext, MockToolContext(state=MockState({}))),
        )
        is None
    )


def test_telegram_config_ignores_removed_progress_options() -> None:
    config = TelegramConfig.model_validate(
        {
            "TELEGRAM_ENABLED": "true",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_TOOL_NOTIFICATIONS": "false",
            "TELEGRAM_TOOL_PROGRESS_MODE": "off",
        }
    )

    assert config.is_configured() is True
    assert not hasattr(config, "telegram_tool_notifications")
    assert not hasattr(config, "telegram_tool_progress_mode")


@pytest.mark.asyncio
async def test_live_progress_is_inert_without_telegram_configuration() -> None:
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    with patch("blacki.callbacks.TelegramApiClient") as client_class:
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "test"},
            cast(ToolContext, context),
        )

    client_class.assert_not_called()


@pytest.mark.asyncio
async def test_live_progress_skips_non_telegram_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock()

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "test"},
            cast(ToolContext, MockToolContext(state=MockState({}))),
        )

    client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_live_progress_uses_one_message_for_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    client.edit_message_text = AsyncMock(return_value=MagicMock(message_id=7))
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    context.invocation_id = "turn-1"

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "first"},
            cast(ToolContext, context),
        )
        callbacks_module._INTERMEDIATE_NOTIFY_LAST["42"] = 0
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_calorie_summary")),
            {},
            cast(ToolContext, context),
        )

    client.send_message.assert_awaited_once()
    assert (
        client.send_message.await_args.kwargs["text"]
        == "Searching the web for *first*…"
    )
    client.edit_message_text.assert_awaited_once()
    assert (
        client.edit_message_text.await_args.kwargs["text"]
        == "Checking calorie summary…"
    )


@pytest.mark.asyncio
async def test_live_progress_uses_model_preamble_then_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    client.edit_message_text = AsyncMock(return_value=MagicMock(message_id=7))
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    context.invocation_id = "turn-2"
    response = LlmResponse(
        content=Content(
            parts=[
                Part.from_text(text="Checking your meals"),
                Part(function_call=FunctionCall(name="get_calorie_summary", args={})),
            ]
        )
    )
    tool_context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    tool_context.invocation_id = "turn-2"

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_after_model(context, response)
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_calorie_summary")),
            {},
            cast(ToolContext, tool_context),
        )
        await notify_telegram_after_agent(context)

    assert client.send_message.await_args.kwargs["text"] == "Checking your meals"
    assert client.edit_message_text.await_args.kwargs["text"].startswith(
        "✓ Worked for "
    )
    assert (42, None, "turn-2") not in callbacks_module._LIVE_STATUS_SESSIONS


@pytest.mark.asyncio
async def test_live_progress_redacts_private_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("send_text_to_speech")),
            {"text": "private spoken content"},
            cast(ToolContext, context),
        )

    assert client.send_message.await_args.kwargs["text"] == "Generating speech…"


@pytest.mark.asyncio
async def test_live_progress_telegram_errors_do_not_block_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(
        side_effect=TelegramApiError("bad request", error_code=400)
    )
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "test"},
            cast(ToolContext, context),
        )

    client.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_progress_ignores_thought_only_model_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    context.invocation_id = "turn-3"
    response = LlmResponse(
        content=Content(
            parts=[
                Part(text="internal", thought=True),
                Part(function_call=FunctionCall(name="brave_search", args={})),
            ]
        )
    )

    await notify_telegram_after_model(context, response)

    assert (42, None, "turn-3") not in callbacks_module._LIVE_STATUS_SESSIONS


def test_live_progress_treats_blank_token_as_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "   ")

    assert telegram_live_tool_progress_enabled() is False


def test_parse_optional_int_variants() -> None:
    assert callbacks_module._parse_optional_int(None) is None
    assert callbacks_module._parse_optional_int("  ") is None
    assert callbacks_module._parse_optional_int("not-a-number") is None
    assert callbacks_module._parse_optional_int("42") == 42
    assert callbacks_module._parse_optional_int(7) == 7


def test_evict_oldest_rate_limit_entries() -> None:
    storage = {"a": 1.0, "b": 2.0, "c": 3.0}
    callbacks_module._evict_oldest_rate_limit_entries(storage, 0)
    assert storage == {"a": 1.0, "b": 2.0, "c": 3.0}

    callbacks_module._evict_oldest_rate_limit_entries(storage, 2)
    assert storage == {"c": 3.0}

    callbacks_module._evict_oldest_rate_limit_entries({}, 5)


def test_evict_oldest_live_status_sessions() -> None:
    session_a = callbacks_module._LiveStatusSession(last_sent_time=1.0)
    session_b = callbacks_module._LiveStatusSession(last_sent_time=2.0)
    key_a: tuple[int, int | None, str | None] = (1, None, "a")
    key_b: tuple[int, int | None, str | None] = (2, None, "b")
    storage = {key_a: session_a, key_b: session_b}

    callbacks_module._evict_oldest_live_status_sessions(storage, 0)
    assert storage == {key_a: session_a, key_b: session_b}

    callbacks_module._evict_oldest_live_status_sessions(storage, 1)
    assert storage == {key_b: session_b}

    callbacks_module._evict_oldest_live_status_sessions({}, 5)


@pytest.mark.asyncio
async def test_rate_limit_allows_notification_too_soon() -> None:
    storage: dict[str, float] = {"chat": 100.0}
    allowed = await callbacks_module._rate_limit_allows_notification(
        "chat",
        100.1,
        storage=storage,
        min_interval=0.35,
        max_entries=8192,
        lock=callbacks_module._INTERMEDIATE_NOTIFY_LOCK,
    )

    assert allowed is False


@pytest.mark.asyncio
async def test_rate_limit_allows_notification_evicts_when_full() -> None:
    storage: dict[str, float] = {"old": 0.0, "older": 0.0}
    allowed = await callbacks_module._rate_limit_allows_notification(
        "new-chat",
        100.0,
        storage=storage,
        min_interval=0.35,
        max_entries=2,
        lock=callbacks_module._INTERMEDIATE_NOTIFY_LOCK,
    )

    assert allowed is True
    assert "new-chat" in storage


@pytest.mark.asyncio
async def test_get_or_create_live_status_session_evicts_when_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(callbacks_module, "_MAX_LIVE_STATUS_SESSIONS", 8)
    for i in range(8):
        callbacks_module._LIVE_STATUS_SESSIONS[(i, None, "seed")] = (
            callbacks_module._LiveStatusSession(last_sent_time=float(i))
        )

    await callbacks_module._get_or_create_live_status_session((99, None, "c"))

    assert (0, None, "seed") not in callbacks_module._LIVE_STATUS_SESSIONS
    assert (99, None, "c") in callbacks_module._LIVE_STATUS_SESSIONS


@pytest.mark.asyncio
async def test_safe_send_status_message_unexpected_error() -> None:
    client = MagicMock()
    client.send_message = AsyncMock(side_effect=RuntimeError("boom"))

    result = await callbacks_module._safe_send_status_message(
        client, chat_id=1, text="hi"
    )

    assert result is None


@pytest.mark.asyncio
async def test_safe_edit_message_text_error_branches() -> None:
    client = MagicMock()
    client.edit_message_text = AsyncMock(
        side_effect=TelegramApiError("bad request", error_code=400)
    )

    result = await callbacks_module._safe_edit_message_text(
        client, chat_id=1, message_id=7, text="hi"
    )

    assert result is False


@pytest.mark.asyncio
async def test_safe_edit_message_text_treats_not_modified_as_success() -> None:
    client = MagicMock()
    client.edit_message_text = AsyncMock(
        side_effect=TelegramApiError("message is not modified", error_code=400)
    )

    result = await callbacks_module._safe_edit_message_text(
        client, chat_id=1, message_id=7, text="hi"
    )

    assert result is True


@pytest.mark.asyncio
async def test_safe_edit_message_text_unexpected_error() -> None:
    client = MagicMock()
    client.edit_message_text = AsyncMock(side_effect=RuntimeError("boom"))

    result = await callbacks_module._safe_edit_message_text(
        client, chat_id=1, message_id=7, text="hi"
    )

    assert result is False


def test_schedule_shared_notify_client_close_for_tests_no_running_loop() -> None:
    client = MagicMock()
    client.close = AsyncMock()
    callbacks_module._shared_notify_client = client
    callbacks_module._shared_notify_token = "test-bot-token"  # noqa: S105

    callbacks_module._schedule_shared_notify_client_close_for_tests()

    assert callbacks_module._shared_notify_client is None


@pytest.mark.asyncio
async def test_schedule_shared_notify_client_close_create_task_fails() -> None:
    client = MagicMock()
    client.close = AsyncMock()
    callbacks_module._shared_notify_client = client
    callbacks_module._shared_notify_token = "test-bot-token"  # noqa: S105

    fake_loop = MagicMock()
    fake_loop.create_task = MagicMock(side_effect=RuntimeError("closed loop"))

    with patch("asyncio.get_running_loop", return_value=fake_loop):
        callbacks_module._schedule_shared_notify_client_close_for_tests()

    assert callbacks_module._shared_notify_client is None


@pytest.mark.asyncio
async def test_close_shared_notify_client() -> None:
    client = MagicMock()
    client.close = AsyncMock()
    callbacks_module._shared_notify_client = client
    callbacks_module._shared_notify_token = "test-bot-token"  # noqa: S105

    await callbacks_module.close_shared_notify_client()

    client.close.assert_awaited_once()
    assert callbacks_module._shared_notify_client is None


@pytest.mark.asyncio
async def test_close_shared_notify_client_swallows_close_error() -> None:
    client = MagicMock()
    client.close = AsyncMock(side_effect=RuntimeError("boom"))
    callbacks_module._shared_notify_client = client
    callbacks_module._shared_notify_token = "test-bot-token"  # noqa: S105

    await callbacks_module.close_shared_notify_client()

    assert callbacks_module._shared_notify_client is None


@pytest.mark.asyncio
async def test_close_shared_notify_client_when_none_is_noop() -> None:
    callbacks_module._shared_notify_client = None
    callbacks_module._shared_notify_token = None

    await callbacks_module.close_shared_notify_client()

    assert callbacks_module._shared_notify_client is None


@pytest.mark.asyncio
async def test_shared_telegram_notify_client_swaps_on_token_change() -> None:
    old_client = MagicMock()
    old_client.close = AsyncMock()
    callbacks_module._shared_notify_client = old_client
    callbacks_module._shared_notify_token = "old-bot-token"  # noqa: S105

    with patch("blacki.callbacks.TelegramApiClient") as client_class:
        new_client = await callbacks_module._shared_telegram_notify_client("new-token")

    old_client.close.assert_awaited_once()
    assert new_client is client_class.return_value
    await callbacks_module.close_shared_notify_client()


@pytest.mark.asyncio
async def test_before_tool_invalid_chat_id_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MockToolContext(state=MockState({"telegram_chat_id": "not-a-number"}))

    with patch("blacki.callbacks.TelegramApiClient") as client_class:
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "test"},
            cast(ToolContext, context),
        )

    client_class.assert_not_called()


@pytest.mark.asyncio
async def test_before_tool_blank_token_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))

    with patch(
        "blacki.callbacks.telegram_live_tool_progress_enabled", return_value=True
    ):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        with patch("blacki.callbacks.TelegramApiClient") as client_class:
            await notify_telegram_before_tool(
                cast(BaseTool, MockBaseTool("brave_search")),
                {"query": "test"},
                cast(ToolContext, context),
            )

    client_class.assert_not_called()


@pytest.mark.asyncio
async def test_before_tool_skips_edit_when_label_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    client.edit_message_text = AsyncMock(return_value=MagicMock(message_id=7))
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    context.invocation_id = "turn-same-label"

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "same"},
            cast(ToolContext, context),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "same"},
            cast(ToolContext, context),
        )

    client.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_before_tool_coalesces_rapid_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    client.edit_message_text = AsyncMock(return_value=MagicMock(message_id=7))
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    context.invocation_id = "turn-coalesce"

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "first"},
            cast(ToolContext, context),
        )
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_calorie_summary")),
            {},
            cast(ToolContext, context),
        )

    client.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_before_tool_edit_failure_keeps_previous_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    client = MagicMock()
    client.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    client.edit_message_text = AsyncMock(
        side_effect=TelegramApiError("bad request", error_code=400)
    )
    context = MockToolContext(state=MockState({"telegram_chat_id": "42"}))
    context.invocation_id = "turn-edit-fails"

    with patch("blacki.callbacks.TelegramApiClient", return_value=client):
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("brave_search")),
            {"query": "first"},
            cast(ToolContext, context),
        )
        callbacks_module._INTERMEDIATE_NOTIFY_LAST["42"] = 0
        await notify_telegram_before_tool(
            cast(BaseTool, MockBaseTool("get_calorie_summary")),
            {},
            cast(ToolContext, context),
        )

    session = callbacks_module._LIVE_STATUS_SESSIONS[(42, None, "turn-edit-fails")]
    assert session.last_sent_text == "Searching the web for *first*…"


@pytest.mark.asyncio
async def test_after_model_disabled_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    response = LlmResponse(content=None)

    await notify_telegram_after_model(context, response)


@pytest.mark.asyncio
async def test_after_model_without_content_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    response = LlmResponse(content=None)

    await notify_telegram_after_model(context, response)


@pytest.mark.asyncio
async def test_after_model_without_function_call_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    response = LlmResponse(content=Content(parts=[Part.from_text(text="just text")]))

    await notify_telegram_after_model(context, response)

    assert callbacks_module._LIVE_STATUS_SESSIONS == {}


@pytest.mark.asyncio
async def test_after_model_missing_chat_id_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({})
    response = LlmResponse(
        content=Content(
            parts=[
                Part.from_text(text="hello"),
                Part(function_call=FunctionCall(name="brave_search", args={})),
            ]
        )
    )

    await notify_telegram_after_model(context, response)

    assert callbacks_module._LIVE_STATUS_SESSIONS == {}


@pytest.mark.asyncio
async def test_after_model_invalid_chat_id_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "not-a-number"})
    response = LlmResponse(
        content=Content(
            parts=[
                Part.from_text(text="hello"),
                Part(function_call=FunctionCall(name="brave_search", args={})),
            ]
        )
    )

    await notify_telegram_after_model(context, response)

    assert callbacks_module._LIVE_STATUS_SESSIONS == {}


@pytest.mark.asyncio
async def test_after_agent_disabled_returns_early() -> None:
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})

    await notify_telegram_after_agent(context)


@pytest.mark.asyncio
async def test_after_agent_missing_chat_id_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({})

    await notify_telegram_after_agent(context)


@pytest.mark.asyncio
async def test_after_agent_invalid_chat_id_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "not-a-number"})

    await notify_telegram_after_agent(context)


@pytest.mark.asyncio
async def test_after_agent_no_session_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    context.invocation_id = "no-such-turn"

    await notify_telegram_after_agent(context)


@pytest.mark.asyncio
async def test_after_agent_blank_token_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_telegram(monkeypatch)
    context = MagicMock(spec=CallbackContext)
    context.state = MockState({"telegram_chat_id": "42"})
    context.invocation_id = "turn-blank-token"
    session_key = (42, None, "turn-blank-token")
    callbacks_module._LIVE_STATUS_SESSIONS[session_key] = (
        callbacks_module._LiveStatusSession(message_id=99)
    )

    with patch(
        "blacki.callbacks.telegram_live_tool_progress_enabled", return_value=True
    ):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        with patch("blacki.callbacks.TelegramApiClient") as client_class:
            await notify_telegram_after_agent(context)

    client_class.assert_not_called()
