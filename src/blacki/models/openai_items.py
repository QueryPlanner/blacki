"""Shared contract for OpenAI Responses items stored on ADK parts.

Each Responses output item becomes one ADK ``Part``. The raw item, exactly as
OpenAI returned it, is kept in ``part.part_metadata[METADATA_KEY]`` together
with the normalized model id that was *requested* (not the dated snapshot id
OpenAI reports). The request builder replays the raw item unchanged when the
next request goes to the same model. See design decision D2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from google.genai import types

METADATA_KEY: Final = "blacki_openai"
COMMENTARY_PHASE: Final = "commentary"


@dataclass(frozen=True, slots=True)
class StoredItem:
    """One raw Responses item and the model that produced it."""

    model: str
    item: Mapping[str, Any]

    @property
    def item_type(self) -> str | None:
        value = self.item.get("type")
        return value if isinstance(value, str) else None


def item_metadata(model: str, item: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``part_metadata`` value that stores ``item`` for ``model``."""

    return {METADATA_KEY: {"model": model, "item": dict(item)}}


def stored_item(part: types.Part) -> StoredItem | None:
    """Return the stored Responses item on ``part``, or ``None``."""

    metadata = part.part_metadata
    if not isinstance(metadata, Mapping):
        return None
    entry = metadata.get(METADATA_KEY)
    if not isinstance(entry, Mapping):
        return None
    model = entry.get("model")
    item = entry.get("item")
    if not isinstance(model, str) or not isinstance(item, Mapping):
        return None
    return StoredItem(model=model, item=item)


def is_commentary(part: types.Part) -> bool:
    """Return whether ``part`` holds an assistant message in commentary phase."""

    stored = stored_item(part)
    return (
        stored is not None
        and stored.item_type == "message"
        and stored.item.get("phase") == COMMENTARY_PHASE
    )


def without_openai_metadata(part: types.Part) -> types.Part:
    """Return a copy of ``part`` with the stored OpenAI item removed.

    Other ``part_metadata`` keys are kept. The copy has no metadata at all when
    the OpenAI item was its only entry.
    """

    metadata = part.part_metadata
    if not isinstance(metadata, Mapping) or METADATA_KEY not in metadata:
        return part
    remaining = {key: value for key, value in metadata.items() if key != METADATA_KEY}
    return part.model_copy(update={"part_metadata": remaining or None})
