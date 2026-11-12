"""Tests for the static OpenAI model catalog and model id normalization."""

from __future__ import annotations

import pytest

from blacki.models.catalog import (
    CATALOG,
    ModelEntry,
    get_model,
    lowest_effort_above_none,
    normalize_model_id,
    supported_model_ids,
)

LUNA = "gpt-6-luna"
TERRA = "gpt-6.1-terra"
SOL = "gpt-6.1-sol"
ASTRA = "gpt-6-astra"

EFFORT_LADDER = ("none", "low", "medium", "high", "xhigh", "max")


def test_catalog_lists_the_four_supported_models() -> None:
    assert supported_model_ids() == (LUNA, TERRA, SOL, ASTRA)


@pytest.mark.parametrize("entry", CATALOG, ids=lambda entry: entry.id)
def test_catalog_efforts_are_unique_ordered_and_contain_the_default(
    entry: ModelEntry,
) -> None:
    positions = [EFFORT_LADDER.index(effort) for effort in entry.efforts]

    assert positions == sorted(set(positions))
    assert entry.default_effort in entry.efforts


@pytest.mark.parametrize(
    ("model_id", "offers_none"),
    [(LUNA, True), (TERRA, True), (SOL, False), (ASTRA, False)],
)
def test_none_effort_is_offered_only_where_sampling_is_accepted_at_none(
    model_id: str,
    offers_none: bool,
) -> None:
    entry = get_model(model_id)

    assert entry is not None
    assert ("none" in entry.efforts) is offers_none
    assert entry.accepts_sampling_at_none is offers_none


@pytest.mark.parametrize(
    ("model_id", "input_price", "cached_price", "cache_write_price", "output_price"),
    [
        (LUNA, 0.10, 0.01, 0.125, 0.50),
        (TERRA, 1.00, 0.05, 1.25, 5.00),
        (SOL, 2.00, 0.10, 2.50, 10.00),
        (ASTRA, 10.00, 1.00, 12.50, 50.00),
    ],
)
def test_prices_match_the_openai_model_pages(
    model_id: str,
    input_price: float,
    cached_price: float,
    cache_write_price: float,
    output_price: float,
) -> None:
    entry = get_model(model_id)

    assert entry is not None
    assert entry.input == input_price
    assert entry.cached_input == cached_price
    assert entry.cache_write == cache_write_price
    assert entry.output == output_price


@pytest.mark.parametrize(
    ("model_id", "expected_id"),
    [
        (LUNA, LUNA),
        (TERRA, TERRA),
        (SOL, SOL),
        (ASTRA, ASTRA),
        (f"openrouter/openai/{LUNA}", LUNA),
        (f"  OPENROUTER/OpenAI/{TERRA.upper()}  ", TERRA),
        (f"\n{ASTRA}\t", ASTRA),
    ],
)
def test_get_model_accepts_plain_prefixed_and_mixed_case_ids(
    model_id: str,
    expected_id: str,
) -> None:
    entry = get_model(model_id)

    assert entry is not None
    assert entry.id == expected_id


@pytest.mark.parametrize(
    "model_id",
    [
        "gemini-2.5-flash",
        "openrouter/google/gemini-2.5-flash",
        "openrouter/anthropic/claude-x",
        "openrouter/gpt-6-luna",
        "openrouter/openai/openrouter/openai/gpt-6-luna",
        "gpt-5.6-luna",
        "openrouter/openai/",
        "",
    ],
)
def test_get_model_returns_none_for_ids_outside_the_catalog(model_id: str) -> None:
    assert get_model(model_id) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gpt-6-luna", "gpt-6-luna"),
        ("  GPT-6.1-Terra \n", "gpt-6.1-terra"),
        ("openrouter/openai/gpt-6-luna", "gpt-6-luna"),
        ("OPENROUTER/OpenAI/GPT-6-Luna", "gpt-6-luna"),
        ("  openrouter/openai/gpt-6.1-sol  ", "gpt-6.1-sol"),
        (
            "openrouter/openai/openrouter/openai/gpt-6-luna",
            "openrouter/openai/gpt-6-luna",
        ),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("openrouter/google/gemini-2.5-flash", "openrouter/google/gemini-2.5-flash"),
        ("openrouter/anthropic/claude-x", "openrouter/anthropic/claude-x"),
        ("   ", ""),
    ],
)
def test_normalize_model_id_strips_space_case_and_one_openai_prefix(
    value: str,
    expected: str,
) -> None:
    assert normalize_model_id(value) == expected


@pytest.mark.parametrize("model_id", [LUNA, TERRA, SOL, ASTRA])
def test_lowest_effort_above_none_is_low_for_every_model(model_id: str) -> None:
    entry = get_model(model_id)

    assert entry is not None
    assert lowest_effort_above_none(entry) == "low"


def test_lowest_effort_above_none_rejects_an_entry_with_only_none() -> None:
    entry = ModelEntry(
        id="test-none-only",
        display_name="Test none only",
        efforts=("none",),
        default_effort="none",
        accepts_sampling_at_none=True,
        input=0.10,
        cached_input=0.01,
        cache_write=None,
        output=None,
    )

    with pytest.raises(ValueError, match="offers no effort above none"):
        lowest_effort_above_none(entry)
