"""Tenant-isolation contract tests for agent-facing memory tools."""

from __future__ import annotations

import inspect
from typing import Any, cast
from unittest.mock import patch

import pytest
from conftest import MockToolContext
from google.adk.tools import ToolContext

from blacki.tools.memory import (
    INACCESSIBLE_MEMORY_ERROR,
    delete_memory,
    get_all_memories,
    get_memory,
    save_memory,
    search_memory,
    update_memory,
)


class FakeMemoryClient:
    """Small in-memory stand-in for the external Mem0 boundary."""

    def __init__(self) -> None:
        self.memories: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self._next_id = 1

    def add(self, text: str, *, user_id: str) -> dict[str, str]:
        memory_id = f"mem-{self._next_id}"
        self._next_id += 1
        self.memories[memory_id] = {
            "id": memory_id,
            "memory": text,
            "user_id": user_id,
        }
        self.calls.append(("add", user_id))
        return {"id": memory_id}

    def search(
        self,
        *,
        query: str,
        user_id: str,
        limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        self.calls.append(("search", user_id))
        matches = [
            {**memory, "score": 1.0}
            for memory in self.memories.values()
            if memory["user_id"] == user_id
            and query.casefold() in str(memory["memory"]).casefold()
        ]
        return {"results": matches[:limit]}

    def get_all(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        self.calls.append(("get_all", user_id))
        matches = [
            memory for memory in self.memories.values() if memory["user_id"] == user_id
        ]
        return {"results": matches[:limit]}

    def get(self, *, memory_id: str) -> dict[str, Any] | None:
        return self.memories.get(memory_id)

    def update(self, memory_id: str, *, data: str) -> None:
        self.memories[memory_id]["memory"] = data

    def delete(self, *, memory_id: str) -> None:
        del self.memories[memory_id]


def _context(user_id: str | None) -> ToolContext:
    return cast(ToolContext, MockToolContext(user_id=user_id))


@pytest.mark.asyncio
async def test_every_memory_operation_is_scoped_to_tool_context() -> None:
    """A second ADK user cannot discover or mutate another user's memories."""
    client = FakeMemoryClient()
    alice = _context("alice")
    bob = _context("bob")

    with patch("blacki.tools.memory.get_memory_client", return_value=client):
        alice_save = await save_memory("Alice secret", alice)
        bob_save = await save_memory("Bob note", bob)
        alice_id = alice_save["result"]["id"]
        bob_id = bob_save["result"]["id"]

        alice_search = await search_memory("secret", alice)
        bob_search = await search_memory("secret", bob)
        alice_list = await get_all_memories(alice)
        bob_list = await get_all_memories(bob)

        assert [item["id"] for item in alice_search["results"]] == [alice_id]
        assert bob_search["results"] == []
        assert [item["id"] for item in alice_list["results"]] == [alice_id]
        assert [item["id"] for item in bob_list["results"]] == [bob_id]
        assert client.calls == [
            ("add", "alice"),
            ("add", "bob"),
            ("search", "alice"),
            ("search", "bob"),
            ("get_all", "alice"),
            ("get_all", "bob"),
        ]

        missing = await get_memory("guessed-id", bob)
        foreign_get = await get_memory(alice_id, bob)
        foreign_update = await update_memory(alice_id, "stolen", bob)
        foreign_delete = await delete_memory(alice_id, bob)

        for result in (missing, foreign_get, foreign_update, foreign_delete):
            assert result == {
                "status": "error",
                "error": INACCESSIBLE_MEMORY_ERROR,
            }

        assert client.memories[alice_id]["memory"] == "Alice secret"

        owner_get = await get_memory(alice_id, alice)
        owner_update = await update_memory(alice_id, "Updated secret", alice)
        owner_delete = await delete_memory(alice_id, alice)

        assert owner_get["status"] == "success"
        assert owner_update["status"] == "success"
        assert owner_delete["status"] == "success"
        assert alice_id not in client.memories


def test_agent_facing_signatures_do_not_accept_user_id() -> None:
    """The model cannot supply a tenant identifier in memory tool schemas."""
    for tool in (save_memory, search_memory, get_all_memories):
        assert "user_id" not in inspect.signature(tool).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (save_memory, ("text",)),
        (search_memory, ("query",)),
        (get_all_memories, ()),
        (get_memory, ("memory-id",)),
        (update_memory, ("memory-id", "text")),
        (delete_memory, ("memory-id",)),
    ],
)
async def test_memory_tools_reject_missing_authenticated_user(
    tool: Any,
    args: tuple[Any, ...],
) -> None:
    """Every operation fails before touching Mem0 when ADK identity is absent."""
    client = FakeMemoryClient()
    with patch("blacki.tools.memory.get_memory_client", return_value=client):
        result = await tool(*args, _context(None))

    assert result["status"] == "error"
    assert result["error"] == "Missing user_id in tool_context."
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed",
    [
        None,
        [],
        {},
        {"id": "memory-id", "memory": "missing owner"},
        {"id": "memory-id", "memory": "foreign", "user_id": "bob"},
    ],
)
async def test_id_operations_fail_closed_for_malformed_or_foreign_records(
    malformed: Any,
) -> None:
    """Malformed Mem0 records never bypass exact ownership checks."""
    client = FakeMemoryClient()
    with (
        patch("blacki.tools.memory.get_memory_client", return_value=client),
        patch.object(client, "get", return_value=malformed),
    ):
        get_result = await get_memory("memory-id", _context("alice"))
        update_result = await update_memory("memory-id", "new", _context("alice"))
        delete_result = await delete_memory("memory-id", _context("alice"))

    assert get_result["error"] == INACCESSIBLE_MEMORY_ERROR
    assert update_result["error"] == INACCESSIBLE_MEMORY_ERROR
    assert delete_result["error"] == INACCESSIBLE_MEMORY_ERROR
