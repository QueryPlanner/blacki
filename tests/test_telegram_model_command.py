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
from blacki.models.inference import (
    INFERENCE_PROFILE_PREFERENCE_KEY,
    LEGACY_MODEL_PREFERENCE_KEY,
    load_inference_profile,
)
from blacki.models.openrouter import OpenRouterLlm
from blacki.runtime.adk import AdkRuntime, SessionLocator
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import CallbackQuery, Chat, Message, Update, User
from blacki.utils.preferences import SqlitePreferencesStorage


@pytest.fixture
async def command_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TelegramBot, SqlitePreferencesStorage, list[dict]]]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ROOT_AGENT_MODEL", "openrouter/acme/deployment-default")
    monkeypatch.delenv("ROOT_AGENT_REASONING_EFFORT", raising=False)
    records: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        records.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion-test",
                "model": records[-1]["model"],
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Remembered."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def client_factory(timeout: float, max_retries: int) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )

    model = OpenRouterLlm(
        model="openrouter/acme/deployment-default",
        api_key="test-key",
        ledger_path=tmp_path / "usage.db",
        client_factory=client_factory,
    )
    runtime = AdkRuntime(
        InMemorySessionService(),
        agent_app=App(
            name="model_command_test",
            root_agent=LlmAgent(name="root_agent", model=model),
            plugins=[TelegramModelOverridePlugin()],
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
        ("/model acme/new-model", "acme/new-model"),
        ("/model openrouter/acme/new-model:free", "acme/new-model:free"),
        ("/model openrouter/auto", "openrouter/auto"),
        ("/model openrouter/free", "openrouter/free"),
        ("/model@blacki_bot\t  Acme/Model-v1.2:extended  ", "Acme/Model-v1.2:extended"),
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
    assert saved.model == f"openrouter/{api_model}"
    assert saved.reasoning is None
    assert (await load_inference_profile(storage, "456")).model is None
    assert "next turn" in bot.api.send_message.await_args.kwargs["text"]
    assert bot.api.send_message.await_args.kwargs["message_thread_id"] == 7

    await send(bot, "What word did I ask you to remember?")

    assert [request["model"] for request in records] == [
        "acme/deployment-default",
        api_model,
    ]
    assert any(
        message["role"] == "user" and "apricot" in str(message["content"])
        for message in records[-1]["messages"]
    )
    assert (await bot.runtime.get_or_create_session(locator=locator)).id == original.id


@pytest.mark.parametrize("command", ["/model", "/models", "/model  "])
async def test_panel_has_usage_and_no_preset_models(
    command_harness, command: str
) -> None:
    bot, storage, records = command_harness
    await send(bot, command)
    reply = bot.api.send_message.await_args.kwargs
    assert "acme/deployment" in reply["text"]
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
    ],
)
async def test_invalid_model_leaves_preferences_unchanged(
    command_harness, model_id
) -> None:
    bot, storage, records = command_harness
    await send(bot, f"/model {model_id}")
    assert await storage.get("123", INFERENCE_PROFILE_PREFERENCE_KEY) is None
    assert "Use /model" in bot.api.send_message.await_args.kwargs["text"]
    assert records == []


async def test_default_uses_deployment_model_and_suppresses_legacy_override(
    command_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot, storage, records = command_harness
    await storage.set("123", LEGACY_MODEL_PREFERENCE_KEY, "openrouter/acme/legacy")
    await send(bot, "/model acme/selected")
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
    assert records[-1]["model"] == "acme/deployment-default"


@pytest.mark.parametrize("key", ["", "   "])
async def test_missing_openrouter_key_preserves_selection_and_allows_default(
    command_harness, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    bot, storage, _ = command_harness
    await send(bot, "/model acme/selected")
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    await send(bot, "/model acme/unusable")
    assert (
        await load_inference_profile(storage, "123")
    ).model == "openrouter/acme/selected"
    assert "OPENROUTER_API_KEY" in bot.api.send_message.await_args.kwargs["text"]
    await send(bot, "/model default")
    assert (await load_inference_profile(storage, "123")).model is None


async def test_save_failure_reports_error(command_harness) -> None:
    bot, storage, _ = command_harness
    await storage.conn.execute("PRAGMA query_only = ON")
    await send(bot, "/model acme/selected")
    assert "Could not save" in bot.api.send_message.await_args.kwargs["text"]
    assert (await load_inference_profile(storage, "123")).model is None


async def test_reply_failure_does_not_undo_saved_model(command_harness, caplog) -> None:
    bot, storage, _ = command_harness
    bot.api.send_message.side_effect = RuntimeError("Telegram unavailable")
    await send(bot, "/model acme/selected")
    assert (
        await load_inference_profile(storage, "123")
    ).model == "openrouter/acme/selected"
    assert "Could not save" not in caplog.text
    assert "Failed to send model selection result" in caplog.text


@pytest.mark.parametrize("data", ["mod:m1", "s:m:m11", "s:m:unknown"])
async def test_expired_preset_cannot_replace_current_selection(
    command_harness, data
) -> None:
    bot, storage, _ = command_harness
    await send(bot, "/model acme/selected")
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
    assert (
        await load_inference_profile(storage, "123")
    ).model == "openrouter/acme/selected"
    assert "expired" in bot.api.answer_callback_query.await_args.kwargs["text"]


async def test_similar_command_does_not_select_model(command_harness) -> None:
    bot, storage, _ = command_harness
    await send(bot, "/modelx acme/selected")
    assert await storage.get("123", INFERENCE_PROFILE_PREFERENCE_KEY) is None
    bot.api.send_message.assert_not_awaited()
