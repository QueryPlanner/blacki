"""Telegram wiki controls, command menu, help text, and scheduled reminders."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types

from blacki import container
from blacki.agent import create_agent, create_app
from blacki.container import AppContainer
from blacki.reminders import scheduler
from blacki.reminders.storage import SqliteReminderStorage
from blacki.runtime.adk import AdkRuntime, SessionLocator
from blacki.telegram import TelegramConfig
from blacki.telegram.album_buffer import _BufferedAlbum
from blacki.telegram.api import TelegramApiClient
from blacki.telegram.bot import TelegramBot
from blacki.telegram.transcription import CloudflareWhisperTranscriber
from blacki.telegram.types import ChatType, PhotoSize, Update, Voice
from blacki.wiki.storage import WikiStore
from blacki.wiki.tools import TOOL_NAMES, WikiTurn

from .conftest import OWNER, PATH


def _bot(
    store: WikiStore | None,
    runtime: AdkRuntime | None = None,
    *,
    access_code: str | None = None,
) -> tuple[TelegramBot, Any]:
    bot = TelegramBot(
        TelegramConfig(TELEGRAM_ACCESS_CODE=access_code),
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
    turn = bot._wiki_message_turn(message)
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
    """Use real ADK tools to schedule a reminder and read its wiki page."""

    reminder_message: str = "Review the deployment wiki."
    delivery_requests: int = 0

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        assert llm_request.contents
        parts = llm_request.contents[-1].parts or []
        response = next(
            (part.function_response for part in parts if part.function_response), None
        )
        if response and response.name == "schedule_reminder":
            assert response.response and response.response["status"] == "success"
            part = types.Part(text="Reminder scheduled.")
        elif response and response.name == "wiki_read":
            assert response.response and response.response["status"] == "ok"
            part = types.Part(text=response.response["markdown"])
        elif any((part.text or "").startswith("[Scheduled Event]") for part in parts):
            self.delivery_requests += 1
            part = (
                types.Part.from_function_call(name="wiki_read", args={"path": PATH})
                if "wiki_read" in llm_request.tools_dict
                else types.Part(text="Wiki unavailable.")
            )
        else:
            part = types.Part.from_function_call(
                name="schedule_reminder",
                args={"message": self.reminder_message, "recurrence": "0 8 * * *"},
            )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.fixture
async def reminder_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[SqliteReminderStorage, None]:
    app = await AppContainer.create(tmp_path / "app.db")
    try:
        await app.initialize_all_storages()
        monkeypatch.setattr(container, "_container", app)
        monkeypatch.setattr(scheduler, "_scheduler", None)
        yield app.reminder_storage
    finally:
        await app.close()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("private", "Updated deployment notes"),
        ("legacy_private", "Updated deployment notes"),
        ("paused", "Wiki unavailable\\."),
        ("access_revoked", "Review the deployment wiki\\."),
        ("public", "Review the deployment wiki\\."),
        ("legacy_public", "Review the deployment wiki\\."),
    ],
)
async def test_private_scheduled_reminder_reads_latest_wiki(
    store: WikiStore,
    reminder_storage: SqliteReminderStorage,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected: str,
) -> None:
    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "1")
    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    agent = create_agent(include_user_scoped_tools=True)
    model = ReminderModel(model="test")
    agent.model = model
    runtime = AdkRuntime(InMemorySessionService(), agent_app=create_app(agent))
    bot, api = _bot(store, runtime)
    user_id = f"telegram-chat-{OWNER}"
    locator = SessionLocator(user_id=user_id, session_id_prefix=user_id)
    if source.endswith("public"):
        public_agent = create_agent()
        public_agent.model = ReminderModel(model="test")
        public_runtime = AdkRuntime(
            InMemorySessionService(), agent_app=create_app(public_agent)
        )
        result = await public_runtime.run_user_turn(
            locator=locator,
            message_text="Schedule a wiki review.",
            state={
                "telegram_chat_id": str(OWNER),
                "telegram_chat_type": "private",
                "temp:telegram_sender_user_id": str(OWNER),
            },
        )
        assert result == "Reminder scheduled."
    else:
        await _send(bot, "Schedule a wiki review.", 1)
    [reminder] = await reminder_storage.get_user_reminders(user_id)
    assert reminder.private_telegram == (not source.endswith("public"))
    if source.startswith("legacy"):
        reminder = reminder.model_copy(update={"private_telegram": None})
        if source == "legacy_public":
            # A private reminder with the same text has a different receipt.
            await _send(bot, "Schedule a wiki review.", 1)
        else:
            await runtime.create_next_session(locator=locator)
    if source == "paused":
        await _send(bot, "/memory off", 2)
    if source == "access_revoked":
        bot, api = _bot(store, runtime, access_code="replacement-code")
    store.edit(OWNER, PATH, "Updated deployment notes", "", read=False)
    store.edit(99, PATH, "Another owner's notes", "", read=False)

    await bot.handle_scheduled_reminder(reminder)

    assert expected in api.send_message.call_args.kwargs["text"]
    assert model.delivery_requests == (
        0 if source.endswith("public") or source == "access_revoked" else 1
    )
    session = await runtime.get_or_create_session(locator=locator)
    scheduled_events = [
        event
        for event in session.events
        if event.author == "user"
        and event.content
        and any(
            (part.text or "").startswith("[Scheduled Event]")
            for part in event.content.parts or []
        )
    ]
    assert len(scheduled_events) == model.delivery_requests
    assert all(
        event.custom_metadata == {"blacki_wiki": {"kind": "synthetic"}}
        for event in scheduled_events
    )


class MediaWikiModel(BaseLlm):
    """Record the effective model request for a private turn."""

    request: LlmRequest | None = None

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.request = llm_request
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Media read.")])
        )


async def test_confirmed_miniapp_turn_receives_owner_wiki(
    store: WikiStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "1")
    agent = create_agent(include_user_scoped_tools=True)
    model = MediaWikiModel(model="test")
    agent.model = model
    runtime = AdkRuntime(InMemorySessionService(), agent_app=create_app(agent))
    bot, api = _bot(store, runtime)
    store.edit(OWNER, "memory.md", "An owner preference", "", read=False)

    await bot.submit_miniapp_prompt(OWNER, "Use my saved preferences.")
    [turn] = bot._background_tasks
    await turn

    assert model.request is not None
    assert model.request.tools_dict.keys() >= TOOL_NAMES
    assert "An owner preference" in str(model.request.config.system_instruction)
    assert api.send_message.call_args.kwargs["chat_id"] == OWNER


@pytest.mark.parametrize("media", ["voice", "album", "forwarded_album", "mixed_album"])
async def test_media_turn_wiki_tool_registration(
    store: WikiStore, monkeypatch: pytest.MonkeyPatch, media: str
) -> None:
    monkeypatch.setenv("BLACKI_WIKI_ENABLED", "1")
    monkeypatch.delenv("TELEGRAM_ACCESS_CODE", raising=False)
    assert (
        store.edit(OWNER, "memory.md", "A remembered detail.", "", read=False).status
        == "saved"
    )
    agent = create_agent(include_user_scoped_tools=True)
    model = MediaWikiModel(model="test")
    agent.model = model
    runtime = AdkRuntime(InMemorySessionService(), agent_app=create_app(agent))
    bot, api = _bot(store, runtime)
    api.get_file.return_value = {"file_path": "telegram-media"}
    if media == "voice":
        transcriber = create_autospec(
            CloudflareWhisperTranscriber, spec_set=True, instance=True
        )
        transcriber.transcribe.return_value = "A voice transcript."
        bot._voice_transcriber = transcriber
        api.download_file.return_value = b"voice"
        update = _update("unused", 1)
        assert update.message is not None
        update.message.text = None
        update.message.voice = Voice(file_id="voice", file_unique_id="1", duration=1)
        await bot._safe_handle_update(update)
    else:
        api.download_file.return_value = bytes.fromhex(
            "ffd8ffc0000b080001000101011100ffd9"
        )
        first = _update("unused", 1).message
        second = _update(
            "unused",
            2,
            forwarded=media == "forwarded_album",
            sender=99 if media == "mixed_album" else OWNER,
        ).message
        assert first is not None and second is not None
        for message in (first, second):
            message.photo = [
                PhotoSize(file_id="photo", file_unique_id="1", width=1, height=1)
            ]
        await bot._handle_album_turn(
            _BufferedAlbum(OWNER, None, "album", ChatType.PRIVATE, [first, second])
        )

    request = model.request
    assert request is not None and request.config is not None
    if media == "mixed_album":
        assert TOOL_NAMES.isdisjoint(request.tools_dict)
        assert "<owner_memory>" not in str(request.config.system_instruction)
    else:
        assert request.tools_dict.keys() >= TOOL_NAMES
        assert "A remembered detail." in str(request.config.system_instruction)
