"""Tests for the memory snapshot included in private Telegram prompts."""

from pathlib import Path
from typing import cast

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

from blacki.wiki.storage import WikiStore
from blacki.wiki.tools import WikiMemoryPlugin, WikiTurn, wiki_read, wiki_turn, wiki_write


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
