"""Tests for request-scoped model and reasoning controls."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import ValidationError

from blacki.agent import TelegramModelOverridePlugin
from blacki.inference import (
    InferenceProfile,
    ReasoningConfig,
    ReasoningEffort,
    apply_inference_profile,
    get_active_inference_profile,
    inference_profile_context,
    inference_profile_from_environment,
    inference_profile_from_mapping,
    load_inference_profile,
    parse_inference_profile,
    update_inference_profile,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("none", ReasoningEffort.NONE),
        ("minimal", ReasoningEffort.MINIMAL),
        ("low", ReasoningEffort.LOW),
        ("medium", ReasoningEffort.MEDIUM),
        ("high", ReasoningEffort.HIGH),
        ("xhigh", ReasoningEffort.XHIGH),
        ("max", ReasoningEffort.MAX),
    ],
)
def test_environment_reasoning_effort_accepts_gateway_values(
    raw: str, expected: ReasoningEffort
) -> None:
    profile = inference_profile_from_environment({"ROOT_AGENT_REASONING_EFFORT": raw})

    assert profile.reasoning is not None
    assert profile.reasoning.effort is expected


def test_environment_reasoning_effort_invalid_value_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        profile = inference_profile_from_environment(
            {"ROOT_AGENT_REASONING_EFFORT": "unsupported"}
        )

    assert profile == InferenceProfile()
    assert "Ignoring unsupported ROOT_AGENT_REASONING_EFFORT" in caplog.text


def test_environment_reasoning_effort_unset_means_inherit() -> None:
    assert inference_profile_from_environment({}) == InferenceProfile()
    assert inference_profile_from_environment(
        {"ROOT_AGENT_REASONING_EFFORT": "  "}
    ) == (InferenceProfile())


def test_reasoning_config_supports_token_budget_and_rejects_ambiguous_values() -> None:
    assert ReasoningConfig(max_tokens=512).model_dump(
        mode="json", exclude_none=True
    ) == {"max_tokens": 512}

    with pytest.raises(ValidationError):
        ReasoningConfig()
    with pytest.raises(ValidationError):
        ReasoningConfig(effort=ReasoningEffort.MAX, max_tokens=512)


def test_inference_profile_is_frozen() -> None:
    profile = InferenceProfile(model="openrouter/openai/gpt-5.6-luna")

    with pytest.raises(ValidationError):
        profile.model = "other"


def test_apply_profile_merges_reasoning_without_clobbering_extra_body() -> None:
    original_extra_body = {
        "provider": {"order": ["openai"]},
        "custom_flag": True,
    }
    request = LlmRequest(
        model="old-model",
        config=types.GenerateContentConfig(
            http_options=types.HttpOptions(extra_body=original_extra_body)
        ),
    )
    profile = InferenceProfile(
        model="openrouter/openai/gpt-5.6-luna",
        reasoning=ReasoningConfig(effort=ReasoningEffort.MAX),
    )

    apply_inference_profile(request, profile)

    assert request.model == "openrouter/openai/gpt-5.6-luna"
    assert request.config.http_options is not None
    assert request.config.http_options.extra_body == {
        "provider": {"order": ["openai"]},
        "custom_flag": True,
        "reasoning": {"effort": "max"},
    }
    assert original_extra_body == {
        "provider": {"order": ["openai"]},
        "custom_flag": True,
    }


def test_apply_profile_can_explicitly_disable_reasoning() -> None:
    request = LlmRequest(model="model")

    apply_inference_profile(
        request,
        InferenceProfile(reasoning=ReasoningConfig(effort=ReasoningEffort.NONE)),
    )

    assert request.config.http_options is not None
    assert request.config.http_options.extra_body == {"reasoning": {"effort": "none"}}


def test_apply_profile_inherit_leaves_existing_reasoning_untouched() -> None:
    request = LlmRequest(
        model="model",
        config=types.GenerateContentConfig(
            http_options=types.HttpOptions(extra_body={"reasoning": {"effort": "low"}})
        ),
    )

    apply_inference_profile(request, InferenceProfile())

    assert request.config.http_options is not None
    assert request.config.http_options.extra_body == {"reasoning": {"effort": "low"}}


def test_apply_profile_ignores_non_mapping_extra_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = LlmRequest(
        model="model",
        config=types.GenerateContentConfig(
            http_options=types.HttpOptions(extra_body={"stale": True})
        ),
    )
    assert request.config.http_options is not None
    request.config.http_options.extra_body = cast(Any, ["not-a-mapping"])

    with caplog.at_level("WARNING"):
        apply_inference_profile(
            request,
            InferenceProfile(reasoning=ReasoningConfig(effort=ReasoningEffort.MAX)),
        )

    assert request.config.http_options.extra_body == {"reasoning": {"effort": "max"}}
    assert "Ignoring non-mapping LLM extra_body" in caplog.text


def test_parse_inference_profile_fails_closed_for_malformed_data() -> None:
    assert parse_inference_profile({"reasoning": {"effort": "invalid"}}) is None
    assert parse_inference_profile({"unknown": True}) is None
    assert parse_inference_profile(None) is None


def test_parse_inference_profile_returns_typed_instance_unchanged() -> None:
    profile = InferenceProfile(model="selected-model")

    assert parse_inference_profile(profile) is profile


class _PreferenceStore:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], object] = {}

    async def get(self, user_id: str, key: str, default: object = None) -> object:
        return self.values.get((user_id, key), default)

    async def update_dict(
        self, user_id: str, key: str, updates: dict[str, object]
    ) -> dict[str, object]:
        current = self.values.get((user_id, key), {})
        merged = dict(current) if isinstance(current, dict) else {}
        merged.update(updates)
        self.values[(user_id, key)] = merged
        return merged


@pytest.mark.asyncio
async def test_load_profile_prefers_canonical_explicit_inherit_over_legacy() -> None:
    storage = _PreferenceStore()
    await storage.update_dict("123", "telegram_model_override", {"value": "old"})
    storage.values[("123", "telegram_model_override")] = "old-model"
    storage.values[("123", "telegram_inference_profile")] = None

    profile = await load_inference_profile(storage, "123")  # type: ignore[arg-type]

    assert profile == InferenceProfile()


@pytest.mark.asyncio
async def test_load_profile_falls_back_to_legacy_model() -> None:
    storage = _PreferenceStore()
    storage.values[("123", "telegram_model_override")] = "old-model"

    profile = await load_inference_profile(storage, "123")  # type: ignore[arg-type]

    assert profile == InferenceProfile(model="old-model")


@pytest.mark.asyncio
async def test_load_profile_falls_back_to_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _PreferenceStore()
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")

    profile = await load_inference_profile(storage, "123")  # type: ignore[arg-type]

    assert profile == InferenceProfile(
        reasoning=ReasoningConfig(effort=ReasoningEffort.MAX)
    )


@pytest.mark.asyncio
async def test_load_profile_merges_environment_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _PreferenceStore()
    storage.values[("123", "telegram_inference_profile")] = {
        "model": "openrouter/openai/gpt-5.6-luna",
        "reasoning": None,
    }
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")

    profile = await load_inference_profile(storage, "123")  # type: ignore[arg-type]

    assert profile.model == "openrouter/openai/gpt-5.6-luna"
    assert profile.reasoning == ReasoningConfig(effort=ReasoningEffort.MAX)


@pytest.mark.asyncio
async def test_load_profile_preserves_explicit_reasoning_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _PreferenceStore()
    storage.values[("123", "telegram_inference_profile")] = {
        "reasoning": {"effort": "high"},
    }
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")

    profile = await load_inference_profile(storage, "123")  # type: ignore[arg-type]

    assert profile.reasoning == ReasoningConfig(effort=ReasoningEffort.HIGH)


@pytest.mark.asyncio
async def test_update_profile_validates_and_serializes_reasoning() -> None:
    storage = _PreferenceStore()

    profile = await update_inference_profile(
        storage,  # type: ignore[arg-type]
        "123",
        {
            "model": "selected-model",
            "reasoning": ReasoningConfig(effort=ReasoningEffort.MAX),
        },
    )

    assert profile.model == "selected-model"
    assert profile.reasoning is not None
    assert profile.reasoning.effort is ReasoningEffort.MAX
    assert storage.values[("123", "telegram_inference_profile")] == {
        "model": "selected-model",
        "reasoning": {"effort": "max"},
    }

    with pytest.raises(ValueError, match="Unsupported inference profile fields"):
        await update_inference_profile(
            storage,  # type: ignore[arg-type]
            "123",
            {"provider": "openai"},
        )


@pytest.mark.asyncio
async def test_update_profile_accepts_model_only_update() -> None:
    storage = _PreferenceStore()

    profile = await update_inference_profile(
        storage,  # type: ignore[arg-type]
        "123",
        {"model": "selected-model"},
    )

    assert profile == InferenceProfile(model="selected-model")
    assert storage.values[("123", "telegram_inference_profile")] == {
        "model": "selected-model",
    }


@pytest.mark.asyncio
async def test_update_profile_serializes_explicit_inherit() -> None:
    storage = _PreferenceStore()

    profile = await update_inference_profile(
        storage,  # type: ignore[arg-type]
        "123",
        {"reasoning": None},
    )

    assert profile == InferenceProfile(reasoning=None)
    assert storage.values[("123", "telegram_inference_profile")] == {
        "reasoning": None,
    }


class _MalformedUpdateStore(_PreferenceStore):
    async def update_dict(
        self, user_id: str, key: str, updates: dict[str, object]
    ) -> dict[str, object]:
        del user_id, key, updates
        return {"reasoning": {"effort": "unsupported"}}


@pytest.mark.asyncio
async def test_update_profile_rejects_malformed_stored_result() -> None:
    storage = _MalformedUpdateStore()

    with pytest.raises(ValueError, match="Stored inference profile is invalid"):
        await update_inference_profile(
            storage,  # type: ignore[arg-type]
            "123",
            {"model": "selected-model"},
        )


def test_inference_profile_from_mapping_delegates_to_parser() -> None:
    profile = inference_profile_from_mapping({"model": "selected-model"})

    assert profile == InferenceProfile(model="selected-model")


@pytest.mark.asyncio
async def test_inference_profile_context_isolated_between_concurrent_turns() -> None:
    first_ready = asyncio.Event()
    second_ready = asyncio.Event()

    async def read_profile(
        profile: InferenceProfile, ready: asyncio.Event, other: asyncio.Event
    ) -> InferenceProfile | None:
        with inference_profile_context(profile):
            ready.set()
            await other.wait()
            return get_active_inference_profile()

    first_profile = InferenceProfile(model="first")
    second_profile = InferenceProfile(model="second")
    first_task = asyncio.create_task(
        read_profile(first_profile, first_ready, second_ready)
    )
    second_task = asyncio.create_task(
        read_profile(second_profile, second_ready, first_ready)
    )

    assert await first_task == first_profile
    assert await second_task == second_profile
    assert get_active_inference_profile() is None


class _CallbackContext:
    session = None


@pytest.mark.asyncio
async def test_plugin_applies_environment_reasoning_without_telegram_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")
    plugin = TelegramModelOverridePlugin()
    plugin.normalize_openrouter = False
    request = LlmRequest(model="openrouter/openai/gpt-5.6-luna")

    await plugin.before_model_callback(
        callback_context=cast(CallbackContext, _CallbackContext()),
        llm_request=request,
    )

    assert request.config.http_options is not None
    assert request.config.http_options.extra_body == {"reasoning": {"effort": "max"}}


@pytest.mark.asyncio
async def test_plugin_ignores_openrouter_reasoning_for_native_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")
    plugin = TelegramModelOverridePlugin()
    plugin.normalize_openrouter = False
    request = LlmRequest(model="gemini-2.5-flash")

    await plugin.before_model_callback(
        callback_context=cast(CallbackContext, _CallbackContext()),
        llm_request=request,
    )

    assert request.config.http_options is None


@pytest.mark.asyncio
async def test_plugin_prefers_active_profile_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")
    plugin = TelegramModelOverridePlugin()
    plugin.normalize_openrouter = False
    request = LlmRequest(model="model")
    profile = InferenceProfile(model="selected", reasoning=None)

    with inference_profile_context(profile):
        await plugin.before_model_callback(
            callback_context=cast(CallbackContext, _CallbackContext()),
            llm_request=request,
        )

    assert request.model == "selected"
    assert request.config.http_options is None
