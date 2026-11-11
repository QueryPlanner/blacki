"""Telegram wiki controls, command menu, help text, and scheduled reminders."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types

from blacki.agent import create_agent, create_app
from blacki.reminders.storage import Reminder
from blacki.runtime.adk import AdkRuntime
from blacki.telegram import TelegramConfig
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.types import Update
from blacki.wiki.storage import WikiStore
from blacki.wiki.tools import TOOL_NAMES, WikiTurn

from .conftest import OWNER, PATH


def _bot(
    store: WikiStore | None, runtime: AdkRuntime | None = None
) -> tuple[TelegramBot, Any]:
    bot = TelegramBot(
        TelegramConfig.model_validate({}),
        runtime or AdkRuntime(InMemorySessionService()),
        wiki_store=store,
    )
    api = create_autospec(TelegramApiClient, instance=True, spec_set=True)
    bot._api = api
    return bot, api


def _update(
    text: str,
    message_id: int,
    *,
    chat_type: str = "private",
    sender: int = OWNER,
    forwarded: bool = False,
) -> Update:
    return Update.model_validate(
        {
            "update_id": message_id,
            "message": {
                "message_id": message_id,
                "date": "2026-11-09T01:00:00Z",
                "chat": {"id": OWNER, "type": chat_type},
                "from": {"id": sender, "first_name": "User"},
                "text": text,
                "forward_origin": {"type": "hidden_user"} if forwarded else None,
            },
        }
    )


async def _send(bot: TelegramBot, text: str, message_id: int, **kwargs: Any) -> None:
    await bot._safe_handle_update(_update(text, message_id, **kwargs))


async def test_memory_on_off_and_list_work_through_private_chat(
    store: WikiStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    bot, api = _bot(store)
    store.set_enabled(OWNER, False)

    await _send(bot, "/memory on", 1, chat_type="group")
    assert not store.enabled(OWNER)
    api.send_message.assert_not_called()

    await _send(bot, "/memory on", 2)
    assert store.enabled(OWNER)
    assert api.send_message.call_args.kwargs["text"].startswith("Memory enabled.")

    await _send(bot, "/memory list", 3)
    assert api.send_message.call_args.kwargs["text"] == "Your wiki is empty."

    short_replies = "Short replies."
    store.edit(OWNER, PATH, short_replies, "", read=False)
    await _send(bot, "/memory list", 4)
    assert api.send_message.call_args.kwargs["text"] == (
        f"{PATH} ({len(short_replies)} chars)"
    )

    await _send(bot, "/memory off", 5)
    assert api.send_message.call_args.kwargs["text"] == "Memory paused."
    assert not store.enabled(OWNER)

    await _send(bot, "/memory list", 6)
    assert api.send_message.call_args.kwargs["text"] == (
        f"{PATH} ({len(short_replies)} chars)"
    )

    await _send(bot, "/memory forget people/ana.md tone", 7)
    assert api.send_message.call_args.kwargs["text"] == (
        "Use /memory on, /memory off, or /memory list."
    )
    assert store.get_page(OWNER, PATH) is not None

    await _send(bot, "/memory on", 8, forwarded=True)
    assert not store.enabled(OWNER)

    await _send(bot, "/memory", 9)
    assert api.send_message.call_args.kwargs["text"].startswith("Use /memory on")


async def test_memory_list_stays_within_one_telegram_message(
    store: WikiStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    for number in range(80):
        store.edit(OWNER, f"{'p' * 140}/{number:02}.md", "x", "", read=False)
    bot, api = _bot(store)
    await _send(bot, "/memory list", 1)
    text = api.send_message.call_args.kwargs["text"]
    assert len(text) <= 4096
    assert text.endswith("more.")
    assert text.count("\n") < 79


async def test_a_failing_wiki_store_reports_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    # The store path is a directory, so every database operation fails.
    bot, api = _bot(WikiStore(tmp_path))
    await _send(bot, "/memory on", 1)
    assert api.send_message.call_args.kwargs["text"] == "Memory storage is unavailable."


async def test_memory_is_ignored_without_a_wiki_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    bot, api = _bot(None)
    await _send(bot, "/memory on", 1)
    api.send_message.assert_not_called()


@pytest.mark.parametrize(
    ("chat_type", "sender", "forwarded", "expected_message_id"),
    [
        ("private", OWNER, False, 7),
        ("private", OWNER, True, None),
        ("group", OWNER, False, "none"),
        ("private", 99, False, "none"),
    ],
)
async def test_text_turns_belong_only_to_the_owners_private_chat(
    store: WikiStore,
    chat_type: str,
    sender: int,
    forwarded: bool,
    expected_message_id: int | None | str,
) -> None:
    bot, _ = _bot(store)
    message = _update(
        "hello", 7, chat_type=chat_type, sender=sender, forwarded=forwarded
    ).message
    assert message is not None
    turn = bot._wiki_text_turn(message)
    if expected_message_id == "none":
        assert turn is None
    else:
        assert isinstance(turn, WikiTurn)
        assert turn.owner == OWNER
        assert turn.message_id == expected_message_id


async def test_memory_command_appears_only_when_wiki_is_configured(
    store: WikiStore,
) -> None:
    with_store, api = _bot(store)
    await with_store._register_commands()
    names = {command.command for command in api.set_my_commands.call_args.args[0]}
    assert "memory" in names

    without, api = _bot(None)
    await without._register_commands()
    names = {command.command for command in api.set_my_commands.call_args.args[0]}
    assert "memory" not in names


async def test_help_lists_memory_only_when_wiki_is_configured(store: WikiStore) -> None:
    with_store, api = _bot(store)
    await with_store._send_help_message(OWNER)
    assert (
        "/memory on, /memory off, /memory list"
        in api.send_message.call_args.kwargs["text"]
    )

    without, api = _bot(None)
    await without._send_help_message(OWNER)
    assert "/memory" not in api.send_message.call_args.kwargs["text"]


class ReminderModel(BaseLlm):
    """Fails the test if wiki tools or the wiki policy reach a scheduled run."""

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        assert TOOL_NAMES.isdisjoint(llm_request.tools_dict)
        assert "personal knowledge base" not in str(
            llm_request.config.system_instruction
        )
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text="Reminder delivered.")]
            )
        )


async def test_scheduled_reminders_run_without_wiki_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "1")
    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    store = WikiStore(tmp_path / "wiki.db")
    store.set_enabled(OWNER, True)
    agent = create_agent(include_user_scoped_tools=True)
    agent.model = ReminderModel(model="test")
    runtime = AdkRuntime(InMemorySessionService(), agent_app=create_app(agent))
    bot, api = _bot(store, runtime)
    reminder = Reminder(
        user_id="telegram-chat-12",
        message="Review the deployment wiki.",
        trigger_time="2026-11-09T18:00:00Z",
        created_at="2026-11-09T12:00:00Z",
    )
    await bot.handle_scheduled_reminder(reminder)
    assert "Reminder delivered" in api.send_message.call_args.kwargs["text"]
    assert store.list_pages(OWNER) == []
