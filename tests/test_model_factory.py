"""Tests for model selection and construction boundaries."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from blacki.models.factory import build_model, normalize_model_for_openrouter


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        (
            "openrouter/openai/gpt-5.6-luna",
            "openrouter/openai/gpt-5.6-luna",
        ),
        (
            "google/gemini-2.5-flash",
            "openrouter/google/gemini-2.5-flash",
        ),
        (
            "gemini-2.5-flash",
            "openrouter/google/gemini-2.5-flash",
        ),
        ("custom-model", "custom-model"),
    ],
)
def test_normalize_model_for_openrouter(model_name: str, expected: str) -> None:
    assert normalize_model_for_openrouter(model_name) == expected


def test_build_model_keeps_native_model_without_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")

    assert build_model() == "gemini-2.5-flash"


def test_build_model_uses_litellm_without_openrouter_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "google/gemini-2.5-flash")

    model = build_model()

    assert not isinstance(model, str)
    assert model.model == "google/gemini-2.5-flash"
    assert model.llm_client is not None


def test_build_model_falls_back_when_litellm_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "google/gemini-2.5-flash")

    with patch("builtins.__import__", side_effect=ImportError("LiteLlm missing")):
        assert build_model() == "google/gemini-2.5-flash"
