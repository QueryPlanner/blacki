# mypy: ignore-errors
"""Telegram model commands through real preferences, ADK, and provider transport."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import create_autospec

import aiosqlite
import httpx
import pytest
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.sessions import InMemorySessionService
from openai import AsyncOpenAI

from blacki.agent import TelegramModelOverridePlugin
from blacki.models.catalog import supported_model_ids
from blacki.models.inference import (
    INFERENCE_PROFILE_PREFERENCE_KEY,
    LEGACY_MODEL_PREFERENCE_KEY,
    load_inference_profile,
)
from blacki.models.openai_responses import DEFAULT_BASE_URL, OpenAIResponsesLlm
from blacki.runtime.adk import AdkRuntime, SessionLocator
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import CallbackQuery, Chat, Message, Update, User
from blacki.utils.preferences import SqlitePreferencesStorage

DEPLOYMENT_MODEL = "gpt-6.1-terra"
API_KEY = "test-key"


def _reply(model: str) -> dict[str, object]:
    """A completed Responses API turn with one final answer message."""
    return {
        "id": "resp_command_test",
        "object": "response",
        "created_at": 1_790_000_000,
        "model": model,
        "status": "completed",
        "output": [
            {
                "id": "msg_command_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [
                    {"type": "output_text", "text": "Remembered.", "annotations": []}
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 12,
        },
        "error": None,
        "incomplete_details": None,
        "store": False,
    }


def _plain(text: str) -> str:
    """Drop MarkdownV2 escapes so assertions read the visible words."""
    return text.replace("\\", "")


@pytest.fixture
async def command_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TelegramBot, SqlitePreferencesStorage, list[dict]]]:
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setenv("ROOT_AGENT_MODEL", DEPLOYMENT_MODEL)
    monkeypatch.delenv("ROOT_AGENT_REASONING_EFFORT", raising=False)
    records: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        records.append(json.loads(request.content))
        return httpx.Response(200, json=_reply(records[-1]["model"]))

    def client_factory(timeout: float, max_retries: int) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=API_KEY,
            base_url=DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )

    model = OpenAIResponsesLlm(
        DEPLOYMENT_MODEL, api_key=API_KEY, client_factory=client_factory
    )
    runtime = AdkRuntime(
        InMemorySessionService(),
        agent_app=App(
            name="model_command_test",
            root_agent=LlmAgent(name="root_agent", model=model),
            plugins=[TelegramModelOverridePlugin(root_is_openai=True)],
        ),
    )
    bot = TelegramBot(
        TelegramConfig(telegram_enabled=True, telegram_bot_token="test-token"), runtime
    )
    bot._api = create_autospec(TelegramApiClient, instance=True, spec_set=True)
    async with aiosqlite.connect(
        tmp_path / "preferences.db", isolation_level=None
    ) as conn:
        conn.row_factory = aiosqlite.Row
        storage = SqlitePreferencesStorage(conn, asyncio.Lock())
        await storage.initialize()
        monkeypatch.setattr(
            "blacki.telegram.bot.get_preferences_storage", lambda: storage
        )
        monkeypatch.setattr(
            "blacki.telegram.settings_menu.get_preferences_storage", lambda: storage
        )
        yield bot, storage, records
        await bot._settings_menu.aclose()
        await storage.close()


async def send(bot: TelegramBot, text: str, *, chat_id: int = 123) -> None:
    await bot._handle_update(
        Update(
            update_id=1,
            message=Message(
                message_id=42,
                date="2026-10-23T00:00:00Z",
                chat=Chat(id=chat_id, type="private"),
                message_thread_id=7,
                text=text,
            ),
        )
    )


@pytest.mark.parametrize(
    ("command", "api_model"),
    [
        ("/model gpt-6-luna", "gpt-6-luna"),
        ("/model openrouter/openai/gpt-6-luna", "gpt-6-luna"),
        ("/model@blacki_bot\t  GPT-6.1-Sol  ", "gpt-6.1-sol"),
    ],
)
async def test_selection_reaches_provider_with_existing_history(
    command_harness, command: str, api_model: str
) -> None:
    bot, storage, records = command_harness
    await send(bot, "Remember the word apricot")
    locator = SessionLocator(
        user_id="telegram-chat-123-thread-7",
        session_id_prefix="telegram-chat-123-thread-7",
    )
    original = await bot.runtime.get_or_create_session(locator=locator)
    await storage.set(
        "123", INFERENCE_PROFILE_PREFERENCE_KEY, {"reasoning": {"effort": "high"}}
    )

    await send(bot, command)
    saved = await load_inference_profile(storage, "123")
    assert saved.model == api_model
    assert saved.reasoning is None
    assert (await load_inference_profile(storage, "456")).model is None
    assert "next turn" in bot.api.send_message.await_args.kwargs["text"]
    assert bot.api.send_message.await_args.kwargs["message_thread_id"] == 7

    await send(bot, "What word did I ask you to remember?")

    assert [request["model"] for request in records] == [
        DEPLOYMENT_MODEL,
        api_model,
    ]
    assert any(
        item.get("role") == "user" and "apricot" in str(item.get("content"))
        for item in records[-1]["input"]
    )
    assert (await bot.runtime.get_or_create_session(locator=locator)).id == original.id


@pytest.mark.parametrize("command", ["/model", "/models", "/model  "])
async def test_panel_has_usage_and_no_preset_models(
    command_harness, command: str
) -> None:
    bot, storage, records = command_harness
    await send(bot, command)
    reply = bot.api.send_message.await_args.kwargs
    assert "GPT-6.1 Terra" in _plain(reply["text"])
    assert "/model" in reply["text"]
    assert [
        button.callback_data
        for row in reply["reply_markup"].inline_keyboard
        for button in row
    ] == ["s:t", "s:x"]
    assert await storage.get("123", INFERENCE_PROFILE_PREFERENCE_KEY) is None
    assert records == []


@pytest.mark.parametrize(
    "model_id",
    [
        "model-name",
        "acme/",
        "/model",
        "acme/model extra",
        "acme/model\nextra",
        "https://example.com/model",
        "acme/model/extra",
        "../model",
        "acme/..",
        "acme/model\x00",
        "acme/model?key=value",
        "openrouter/anthropic/claude-x",
    ],
)
async def test_invalid_model_lists_supported_ids_and_saves_nothing(
    command_harness, model_id
) -> None:
    bot, storage, records = command_harness
    await send(bot, f"/model {model_id}")
    assert await storage.get("123", INFERENCE_PROFILE_PREFERENCE_KEY) is None
    text = bot.api.send_message.await_args.kwargs["text"]
    assert "Use /model <model-id> or /model default." in text
    for supported in supported_model_ids():
        assert supported in text
    assert records == []


async def test_default_uses_deployment_model_and_suppresses_legacy_override(
    command_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, storage, records = command_harness
    await storage.set(
        "123", LEGACY_MODEL_PREFERENCE_KEY, "openrouter/openai/gpt-6-luna"
    )
    await send(bot, "/model gpt-6-astra")
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "low")
    await send(bot, "/model default")
    assert await storage.get("123", INFERENCE_PROFILE_PREFERENCE_KEY) == {
        "model": None,
        "reasoning": None,
    }
    profile = await load_inference_profile(storage, "123")
    assert profile.model is None
    assert profile.reasoning.effort == "low"
    await send(bot, "Use the configured default")
    assert records[-1]["model"] == DEPLOYMENT_MODEL


@pytest.mark.parametrize("key", ["", "   "])
async def test_missing_openai_key_preserves_selection_and_allows_default(
    command_harness, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    bot, storage, _ = command_harness
    await send(bot, "/model gpt-6-luna")
    monkeypatch.setenv("OPENAI_API_KEY", key)
    await send(bot, "/model gpt-6.1-sol")
    assert (await load_inference_profile(storage, "123")).model == "gpt-6-luna"
    assert "OpenAI is not configured" in bot.api.send_message.await_args.kwargs["text"]
    await send(bot, "/model default")
    assert (await load_inference_profile(storage, "123")).model is None


async def test_save_failure_reports_error(command_harness) -> None:
    bot, storage, _ = command_harness
    await storage.conn.execute("PRAGMA query_only = ON")
    await send(bot, "/model gpt-6-luna")
    assert "Could not save" in bot.api.send_message.await_args.kwargs["text"]
    assert (await load_inference_profile(storage, "123")).model is None


async def test_reply_failure_does_not_undo_saved_model(command_harness, caplog) -> None:
    bot, storage, _ = command_harness
    bot.api.send_message.side_effect = RuntimeError("Telegram unavailable")
    await send(bot, "/model gpt-6-luna")
    assert (await load_inference_profile(storage, "123")).model == "gpt-6-luna"
    assert "Could not save" not in caplog.text
    assert "Failed to send model selection result" in caplog.text


@pytest.mark.parametrize("data", ["mod:m1", "s:m:m11", "s:m:unknown"])
async def test_expired_preset_cannot_replace_current_selection(
    command_harness, data
) -> None:
    bot, storage, _ = command_harness
    await send(bot, "/model gpt-6-luna")
    await bot._settings_menu.handle_callback(
        CallbackQuery(
            id="old-preset",
            from_user=User(id=123, is_bot=False, first_name="Test"),
            chat_instance="instance",
            data=data,
            message=Message(
                message_id=1,
                date="2026-10-23T00:00:00Z",
                chat=Chat(id=123, type="private"),
            ),
        )
    )
    assert (await load_inference_profile(storage, "123")).model == "gpt-6-luna"
    assert "expired" in bot.api.answer_callback_query.await_args.kwargs["text"]


async def test_similar_command_does_not_select_model(command_harness) -> None:
    bot, storage, _ = command_harness
    await send(bot, "/modelx gpt-6-luna")
    assert await storage.get("123", INFERENCE_PROFILE_PREFERENCE_KEY) is None
    bot.api.send_message.assert_not_awaited()


@pytest.mark.parametrize("prompt_count", [1, 2])
@pytest.mark.parametrize(
    ("command", "expected_model"),
    [
        ("/model gpt-6-luna", "gpt-6-luna"),
        ("/models@blacki_bot gpt-6-luna", "gpt-6-luna"),
        ("/model default", DEPLOYMENT_MODEL),
    ],
)
async def test_rapid_prompt_waits_for_model_save(
    command_harness, command: str, expected_model: str, prompt_count: int
) -> None:
    bot, storage, records = command_harness
    await send(bot, "Remember apricot")
    await send(bot, "/model gpt-6-astra")
    bot.api.send_message.reset_mock()

    def update(seq: int, text: str) -> Update:
        return Update(
            update_id=seq,
            message=Message(
                message_id=seq,
                date="2026-10-23T00:00:00Z",
                chat=Chat(id=123, type="private"),
                message_thread_id=7,
                text=text,
            ),
        )

    # Hold the real transaction lock so the command pauses before its write.
    tasks = []
    try:
        async with storage._lock:
            command_task = asyncio.create_task(
                bot._safe_handle_update(update(10, command))
            )
            tasks.append(command_task)
            await asyncio.sleep(0)
            for seq in range(11, 11 + prompt_count):
                tasks.append(
                    asyncio.create_task(
                        bot._safe_handle_update(update(seq, "Recall apricot"))
                    )
                )
                await asyncio.sleep(0)
            assert records[-1]["model"] == DEPLOYMENT_MODEL

        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=5
        )
        assert results[0] is None
        assert results[-1] is None
        assert records[-1]["model"] == expected_model
        assert any(
            item.get("role") == "user"
            and "Remember apricot" in str(item.get("content"))
            for item in records[-1]["input"]
        )
        assert any(
            "Model saved for the next turn" in call.kwargs["text"]
            for call in bot.api.send_message.await_args_list
        )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
