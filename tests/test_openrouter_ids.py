"""Tests for canonical OpenRouter model identifiers."""

from __future__ import annotations

import pytest

from blacki.models.openrouter_ids import (
    canonical_openrouter_model_id,
    is_openrouter_model_id,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("openrouter/openai/gpt-4o", "openai/gpt-4o"),
        ("openrouter/google/gemini-2.5-flash", "google/gemini-2.5-flash"),
        ("OPENROUTER/OpenAI/GPT-4O", "OpenAI/GPT-4O"),
        ("openrouter/auto", "openrouter/auto"),
        ("openrouter/free", "openrouter/free"),
        ("google/gemini-2.5-flash", "google/gemini-2.5-flash"),
        ("  openrouter/openai/gpt-4o  ", "openai/gpt-4o"),
        ("custom-model", "custom-model"),
        ("", ""),
    ],
)
def test_canonical_openrouter_model_id(value: str, expected: str) -> None:
    assert canonical_openrouter_model_id(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("openrouter/openai/gpt-4o", True),
        ("openai/gpt-4o", True),
        ("openrouter/auto", True),
        ("openrouter/free", True),
        ("custom-model", False),
        ("", False),
        ("openrouter/", False),
        ("openrouter/../gpt", False),
        ("openrouter/gpt/..", False),
        ("author/slug/extra", False),
    ],
)
def test_is_openrouter_model_id(value: str, expected: bool) -> None:
    assert is_openrouter_model_id(value) is expected
