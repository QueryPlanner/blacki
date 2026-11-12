"""Static catalog of the OpenAI models Blacki can select.

Each entry records the reasoning efforts the Responses API accepts for the
model, the effort used when none is chosen, whether sampling parameters are
accepted at effort ``none``, and the token prices used for cost estimates.

Prices are USD per 1M tokens at OpenAI's standard processing rates. They do
not include the surcharge for prompts over 272K input tokens, the Batch and
Flex discounts, fast mode, or regional processing premiums, so estimates made
from them can be low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_OPENROUTER_OPENAI_PREFIX: Final = "openrouter/openai/"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One supported OpenAI model and its request and pricing rules.

    ``efforts`` is ordered from lowest to highest.
    """

    id: str
    display_name: str
    efforts: tuple[str, ...]
    default_effort: str
    accepts_sampling_at_none: bool
    input: float
    cached_input: float
    cache_write: float | None
    output: float | None


CATALOG: Final[tuple[ModelEntry, ...]] = (
    # Source: https://developers.openai.com/api/docs/models/gpt-6-luna
    # Read 2026-11-12.
    ModelEntry(
        id="gpt-6-luna",
        display_name="GPT-6 Luna",
        efforts=("none", "low", "medium", "high", "xhigh", "max"),
        default_effort="medium",
        accepts_sampling_at_none=True,
        input=0.10,
        cached_input=0.01,
        cache_write=0.125,
        output=0.50,
    ),
    # Source: https://developers.openai.com/api/docs/models/gpt-6.1-terra
    # Read 2026-11-12. The cache-write price is the 30-minute retention rate;
    # the page lists $2.50 for 12-hour retention.
    ModelEntry(
        id="gpt-6.1-terra",
        display_name="GPT-6.1 Terra",
        efforts=("none", "low", "medium", "high", "xhigh", "max"),
        default_effort="medium",
        accepts_sampling_at_none=True,
        input=1.00,
        cached_input=0.05,
        cache_write=1.25,
        output=5.00,
    ),
    # Source: https://developers.openai.com/api/docs/models/gpt-6.1-sol
    # Read 2026-11-12.
    ModelEntry(
        id="gpt-6.1-sol",
        display_name="GPT-6.1 Sol",
        efforts=("low", "medium", "high", "xhigh", "max"),
        default_effort="medium",
        accepts_sampling_at_none=False,
        input=2.00,
        cached_input=0.10,
        cache_write=2.50,
        output=10.00,
    ),
    # Source: https://developers.openai.com/api/docs/models/gpt-6-astra
    # Read 2026-11-12. The page lists the accepted efforts but marks no default;
    # "medium" is the default set in the catalog spec, not read from the page.
    ModelEntry(
        id="gpt-6-astra",
        display_name="GPT-6 Astra",
        efforts=("low", "medium", "high", "xhigh", "max"),
        default_effort="medium",
        accepts_sampling_at_none=False,
        input=10.00,
        cached_input=1.00,
        cache_write=12.50,
        output=50.00,
    ),
)

_ENTRIES_BY_ID: Final[dict[str, ModelEntry]] = {entry.id: entry for entry in CATALOG}


def normalize_model_id(value: str) -> str:
    """Return the catalog spelling of a model reference.

    Strips surrounding whitespace, lowercases, and removes one leading
    ``openrouter/openai/`` prefix. The result is not checked against the
    catalog; use ``get_model`` for that.
    """

    normalized = value.strip().lower()
    return normalized.removeprefix(_OPENROUTER_OPENAI_PREFIX)


def get_model(model_id: str) -> ModelEntry | None:
    """Return the catalog entry for ``model_id``, or ``None`` if unsupported.

    ``model_id`` may use any spelling that ``normalize_model_id`` accepts.
    """

    return _ENTRIES_BY_ID.get(normalize_model_id(model_id))


def supported_model_ids() -> tuple[str, ...]:
    """Return every catalog model id, in catalog order."""

    return tuple(entry.id for entry in CATALOG)


def lowest_effort_above_none(entry: ModelEntry) -> str:
    """Return the lowest effort in ``entry`` that is not ``none``.

    Raises ``ValueError`` when the entry offers no effort other than ``none``.
    """

    for effort in entry.efforts:
        if effort != "none":
            return effort
    raise ValueError(f"model {entry.id} offers no effort above none")
