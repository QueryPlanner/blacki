"""Tests for model selection and construction boundaries."""

from __future__ import annotations

import pytest

from blacki.models.factory import build_model, normalize_model_for_openrouter
from blacki.models.openrouter import OpenRouterLlm
from blacki.utils.exceptions import ConfigurationError


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


def test_build_model_uses_openrouter_adapter_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ROOT_AGENT_MODEL", "google/gemini-2.5-flash")

    model = build_model()

    assert isinstance(model, OpenRouterLlm)
    assert model.model == "openrouter/google/gemini-2.5-flash"


def test_build_model_keeps_existing_openrouter_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ROOT_AGENT_MODEL", "openrouter/openai/gpt-4o")

    model = build_model()

    assert isinstance(model, OpenRouterLlm)
    assert model.model == "openrouter/openai/gpt-4o"


def test_build_model_rejects_openrouter_model_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "openrouter/google/gemini-2.5-flash")

    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY is required"):
        build_model()


def test_build_model_rejects_provider_style_model_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "google/gemini-2.5-flash")

    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY is required"):
        build_model()


def test_build_model_rejects_invalid_openrouter_id_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ROOT_AGENT_MODEL", "custom-model")

    with pytest.raises(ConfigurationError, match="not a valid OpenRouter"):
        build_model()


def test_build_model_rejects_unknown_native_model_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "custom-model")

    with pytest.raises(ConfigurationError, match="not a native Gemini model"):
        build_model()


def test_build_model_accepts_models_prefixed_native_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "models/gemini-2.5-flash")

    assert build_model() == "models/gemini-2.5-flash"
