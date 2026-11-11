"""Tests for the memory snapshot included in private Telegram prompts."""

from pathlib import Path
from typing import cast

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

from blacki.wiki.storage import WikiStore
from blacki.wiki.tools import (
    WikiMemoryPlugin,
    WikiTurn,
    wiki_read,
    wiki_turn,
    wiki_write,
)


@pytest.mark.asyncio
async def test_memory_write_appears_in_next_turn(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki.sqlite")
    plugin = WikiMemoryPlugin(name="owner_memory")

    async def prompt() -> str:
        request = LlmRequest()
        await plugin.before_model_callback(
            callback_context=cast(CallbackContext, None), llm_request=request
        )
        assert request.config is not None
        return str(request.config.system_instruction)

    assert store.edit(1, "memory.md", "Old fact", "", read=False).status == "saved"
    with wiki_turn(WikiTurn(store, None, 1, "test")):
        initial = await prompt()
        assert "Old fact" in initial
        refused = await wiki_write("memory.md", "New fact", "Old fact")
        assert refused["status"] == "read_required"
        assert (await wiki_read(path="memory.md"))["status"] == "ok"
        saved = await wiki_write("memory.md", "New fact", "Old fact")
        assert saved["status"] == "saved"
        assert await prompt() == initial

    with wiki_turn(WikiTurn(store, None, 1, "test")):
        assert "New fact" in await prompt()

    store.set_enabled(1, False)
    with wiki_turn(WikiTurn(store, None, 1, "test")):
        assert "<owner_memory>" not in await prompt()


@pytest.mark.asyncio
async def test_memory_prompt_legacy_limit_and_storage_error(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki.sqlite")
    plugin = WikiMemoryPlugin(name="owner_memory")
    with store.connect() as db:
        db.execute(
            "INSERT INTO wiki_pages VALUES (1, 'memory.md', ?, 1, '2026-11-11')",
            ("</owner_memory>" + "a" * 5_000 + "tail marker",),
        )
    with wiki_turn(WikiTurn(store, None, 1, "test")):
        request = LlmRequest()
        await plugin.before_model_callback(
            callback_context=cast(CallbackContext, None), llm_request=request
        )
        assert request.config is not None
        instruction = str(request.config.system_instruction)
        assert "[owner_memory tag removed]" in instruction
        assert "Truncated." in instruction
        assert "tail marker" not in instruction

    with store.connect() as db:
        db.execute("DROP TABLE wiki_pages")
        db.execute("CREATE TABLE wiki_pages (unsupported)")
    with wiki_turn(WikiTurn(store, None, 1, "test")):
        request = LlmRequest()
        await plugin.before_model_callback(
            callback_context=cast(CallbackContext, None), llm_request=request
        )
        assert request.config is not None
        assert "<owner_memory>" not in str(request.config.system_instruction)
