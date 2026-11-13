"""Tests for the stored OpenAI item contract on ADK parts."""

from __future__ import annotations

import pytest
from google.genai import types

from blacki.models.openai_items import (
    METADATA_KEY,
    StoredItem,
    is_commentary,
    item_metadata,
    stored_item,
    without_openai_metadata,
)

REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e"}


def test_round_trip_stores_model_and_item() -> None:
    part = types.Part(
        text="", thought=True, part_metadata=item_metadata("gpt-6-luna", REASONING)
    )

    stored = stored_item(part)

    assert stored == StoredItem(model="gpt-6-luna", item=REASONING)
    assert stored is not None
    assert stored.item_type == "reasoning"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"other": 1},
        {METADATA_KEY: "not a mapping"},
        {METADATA_KEY: {"model": 1, "item": REASONING}},
        {METADATA_KEY: {"model": "gpt-6-luna", "item": "x"}},
    ],
)
def test_stored_item_rejects_malformed_metadata(metadata: object) -> None:
    part = types.Part(text="hi", part_metadata=metadata)

    assert stored_item(part) is None


def test_item_type_is_none_when_missing() -> None:
    assert StoredItem(model="m", item={"id": "x"}).item_type is None


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"type": "message", "phase": "commentary"}, True),
        ({"type": "message", "phase": "final_answer"}, False),
        ({"type": "message"}, False),
        ({"type": "reasoning", "phase": "commentary"}, False),
    ],
)
def test_is_commentary(item: dict[str, str], expected: bool) -> None:
    part = types.Part(text="x", part_metadata=item_metadata("gpt-6-luna", item))

    assert is_commentary(part) is expected


def test_is_commentary_false_without_metadata() -> None:
    assert is_commentary(types.Part(text="x")) is False


def test_without_openai_metadata_removes_only_our_key() -> None:
    metadata = {**item_metadata("gpt-6-luna", REASONING), "keep": True}
    part = types.Part(text="", thought=True, part_metadata=metadata)

    cleaned = without_openai_metadata(part)

    assert cleaned.part_metadata == {"keep": True}
    assert part.part_metadata == metadata


def test_without_openai_metadata_clears_empty_metadata() -> None:
    part = types.Part(text="x", part_metadata=item_metadata("gpt-6-luna", REASONING))

    assert without_openai_metadata(part).part_metadata is None


@pytest.mark.parametrize("metadata", [None, {"keep": True}])
def test_without_openai_metadata_returns_part_unchanged(metadata: object) -> None:
    part = types.Part(text="x", part_metadata=metadata)

    assert without_openai_metadata(part) is part
