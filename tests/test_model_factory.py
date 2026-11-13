"""Tests for root-model selection and construction."""

from __future__ import annotations

import logging

import pytest

from blacki.models.catalog import supported_model_ids
from blacki.models.factory import build_model
from blacki.models.openai_responses import OpenAIResponsesLlm
from blacki.utils.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def isolated_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ROOT_AGENT_MODEL", raising=False)


@pytest.mark.parametrize("model_id", supported_model_ids())
def test_catalog_id_builds_openai_adapter_with_key(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ROOT_AGENT_MODEL", model_id)

    model = build_model()

    assert isinstance(model, OpenAIResponsesLlm)
    assert model.model == model_id


def test_catalog_id_is_normalized_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ROOT_AGENT_MODEL", "  GPT-6.1-Terra  ")

    model = build_model()

    assert isinstance(model, OpenAIResponsesLlm)
    assert model.model == "gpt-6.1-terra"


def test_legacy_openrouter_openai_id_is_converted_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ROOT_AGENT_MODEL", "openrouter/openai/gpt-6-luna")

    with caplog.at_level(logging.INFO, logger="blacki.agent"):
        model = build_model()

    assert isinstance(model, OpenAIResponsesLlm)
    assert model.model == "gpt-6-luna"
    assert "legacy OpenRouter form" in caplog.text


@pytest.mark.parametrize(
    "model_name",
    ["gpt-6-luna", "openrouter/openai/gpt-6-luna"],
)
def test_openai_model_without_key_names_the_missing_setting(
    monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    monkeypatch.setenv("ROOT_AGENT_MODEL", model_name)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_model()


@pytest.mark.parametrize(
    "model_name",
    ["gemini-2.5-flash", "gemma-3-27b-it", "models/gemini-2.5-flash"],
)
def test_native_gemini_model_is_returned_without_an_openai_key(
    monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    monkeypatch.setenv("ROOT_AGENT_MODEL", model_name)

    assert build_model() == model_name


def test_native_gemini_model_stays_native_when_openai_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")

    assert build_model() == "gemini-2.5-flash"


def test_unset_model_defaults_to_native_gemini_without_openai_key() -> None:
    assert build_model() == "gemini-2.5-flash"


def test_blank_model_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROOT_AGENT_MODEL", "   ")

    assert build_model() == "gemini-2.5-flash"


def test_unset_model_defaults_to_luna_when_openai_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = build_model()

    assert isinstance(model, OpenAIResponsesLlm)
    assert model.model == "gpt-6-luna"


@pytest.mark.parametrize(
    "model_name",
    [
        "openrouter/anthropic/claude-x",
        "openrouter/google/gemini-2.5-flash",
        "google/gemini-2.5-flash",
        "custom-model",
    ],
)
@pytest.mark.parametrize("openai_key", [None, "test-key"])
def test_unsupported_model_fails_listing_supported_ids(
    monkeypatch: pytest.MonkeyPatch, model_name: str, openai_key: str | None
) -> None:
    if openai_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    monkeypatch.setenv("ROOT_AGENT_MODEL", model_name)

    with pytest.raises(ConfigurationError) as info:
        build_model()

    for model_id in supported_model_ids():
        assert model_id in str(info.value)
