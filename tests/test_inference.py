"""Tests for request-scoped model and reasoning controls."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
from typing import Any, cast

import pytest
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import ValidationError

import blacki.agent as agent_module
from blacki.agent import TelegramModelOverridePlugin
from blacki.models.inference import (
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
from blacki.models.openai_items import item_metadata
from blacki.models.openai_responses import SESSION_LABEL_KEY


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
        self,
        user_id: str,
        key: str,
        updates: dict[str, object],
        *,
        defaults: dict[str, object] | None = None,
        expected: dict[str, object] | None = None,
    ) -> dict[str, object]:
        current = self.values.get((user_id, key), {})
        merged = dict(current) if isinstance(current, dict) else {}
        for field, value in (defaults or {}).items():
            merged.setdefault(field, value)
        for field, value in (expected or {}).items():
            if merged.get(field) != value:
                raise RuntimeError("preference conflict")
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
async def test_load_profile_backfills_legacy_model_when_canonical_omits_it() -> None:
    storage = _PreferenceStore()
    storage.values[("123", "telegram_model_override")] = "old-model"
    storage.values[("123", "telegram_inference_profile")] = {
        "reasoning": {"effort": "high"}
    }

    profile = await load_inference_profile(storage, "123")  # type: ignore[arg-type]

    assert profile == InferenceProfile(
        model="old-model",
        reasoning=ReasoningConfig(effort=ReasoningEffort.HIGH),
    )


@pytest.mark.asyncio
async def test_load_profile_explicit_default_model_suppresses_legacy() -> None:
    storage = _PreferenceStore()
    storage.values[("123", "telegram_model_override")] = "old-model"
    storage.values[("123", "telegram_inference_profile")] = {
        "model": None,
        "reasoning": {"effort": "high"},
    }

    profile = await load_inference_profile(storage, "123")  # type: ignore[arg-type]

    assert profile == InferenceProfile(
        model=None,
        reasoning=ReasoningConfig(effort=ReasoningEffort.HIGH),
    )


@pytest.mark.asyncio
async def test_load_profile_recovers_legacy_model_from_malformed_canonical() -> None:
    storage = _PreferenceStore()
    storage.values[("123", "telegram_model_override")] = "old-model"
    storage.values[("123", "telegram_inference_profile")] = ["invalid"]

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
async def test_update_profile_seeds_model_from_base_profile() -> None:
    storage = _PreferenceStore()
    base_profile = InferenceProfile(model="legacy-model")

    profile = await update_inference_profile(
        storage,  # type: ignore[arg-type]
        "123",
        {"reasoning": ReasoningConfig(effort=ReasoningEffort.MAX)},
        base_profile=base_profile,
    )

    assert profile == InferenceProfile(
        model="legacy-model",
        reasoning=ReasoningConfig(effort=ReasoningEffort.MAX),
    )


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
        self,
        user_id: str,
        key: str,
        updates: dict[str, object],
        *,
        defaults: dict[str, object] | None = None,
        expected: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del user_id, key, updates, defaults, expected
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


_MODEL = "gpt-6-luna"
_TEST_LABEL_SECRET = b"t" * 32
_CALL_ID = "call_test_0001"
_REASONING_ITEM: dict[str, Any] = {
    "id": "rs_0001",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "Checked the log."}],
    "encrypted_content": "ENC_plugin_fixture",
}
_EMPTY_REASONING_ITEM: dict[str, Any] = {
    "id": "rs_0002",
    "type": "reasoning",
    "summary": [],
    "encrypted_content": "ENC_plugin_empty",
}
_FUNCTION_CALL_ITEM: dict[str, Any] = {
    "id": "fc_0001",
    "type": "function_call",
    "call_id": _CALL_ID,
    "name": "list_meals",
    "arguments": '{"date": "2026-11-12"}',
}
_MESSAGE_ITEM: dict[str, Any] = {
    "id": "msg_0001",
    "type": "message",
    "role": "assistant",
    "phase": "final_answer",
    "content": [
        {"type": "output_text", "text": "You logged lunch.", "annotations": []}
    ],
}


def _model_turn_with_stored_items() -> types.Content:
    """A model turn whose parts each carry the item OpenAI returned."""

    return types.Content(
        role="model",
        parts=[
            types.Part(
                text="Checked the log.",
                thought=True,
                part_metadata=item_metadata(_MODEL, _REASONING_ITEM),
            ),
            types.Part(
                function_call=types.FunctionCall(
                    id=_CALL_ID, name="list_meals", args={"date": "2026-11-12"}
                ),
                part_metadata=item_metadata(_MODEL, _FUNCTION_CALL_ITEM),
            ),
            types.Part(
                text="You logged lunch.",
                part_metadata=item_metadata(_MODEL, _MESSAGE_ITEM),
            ),
        ],
    )


async def _callback_context(
    *,
    user_id: str = "telegram-chat-1",
    session_id: str = "session-1",
) -> Context:
    """Return a real ADK callback context for one session."""

    service = InMemorySessionService()
    session = await service.create_session(
        app_name="blacki", user_id=user_id, session_id=session_id
    )
    invocation = InvocationContext(
        session_service=service,
        invocation_id="invocation-1",
        agent=LlmAgent(name="blacki", model="gemini-2.5-flash"),
        session=session,
    )
    return Context(invocation)


def _openai_plugin() -> TelegramModelOverridePlugin:
    return TelegramModelOverridePlugin(root_is_openai=True)


async def _run_plugin(
    plugin: TelegramModelOverridePlugin,
    request: LlmRequest,
    *,
    context: Context | None = None,
) -> None:
    callback_context = context if context is not None else await _callback_context()
    await plugin.before_model_callback(
        callback_context=callback_context,
        llm_request=request,
    )


@pytest.mark.asyncio
async def test_plugin_applies_environment_reasoning_to_catalog_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")
    request = LlmRequest(model="gpt-6-luna")

    await _run_plugin(_openai_plugin(), request)

    assert request.model == "gpt-6-luna"
    assert request.config.http_options is not None
    assert request.config.http_options.extra_body == {"reasoning": {"effort": "max"}}


@pytest.mark.asyncio
async def test_plugin_ignores_reasoning_for_native_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")
    request = LlmRequest(model="gemini-2.5-flash")

    with caplog.at_level(logging.INFO, logger="blacki.agent"):
        await _run_plugin(TelegramModelOverridePlugin(), request)

    assert request.config.http_options is None
    assert "Ignoring reasoning controls for native model" in caplog.text


@pytest.mark.asyncio
async def test_plugin_prefers_active_profile_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOT_AGENT_REASONING_EFFORT", "max")
    request = LlmRequest(model="gpt-6-luna")
    profile = InferenceProfile(model="gpt-6.1-terra", reasoning=None)

    with inference_profile_context(profile):
        await _run_plugin(_openai_plugin(), request)

    assert request.model == "gpt-6.1-terra"
    assert request.config.http_options is None


@pytest.mark.asyncio
async def test_legacy_openai_selection_uses_its_catalog_model() -> None:
    request = LlmRequest(model="gpt-6-luna")
    profile = InferenceProfile(model="openrouter/openai/gpt-6.1-terra")

    with inference_profile_context(profile):
        await _run_plugin(_openai_plugin(), request)

    assert request.model == "gpt-6.1-terra"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "saved_model",
    [
        "openrouter/google/gemini-2.5-flash",
        "gemini-2.5-flash",
        "openrouter/anthropic/claude-x",
        "openrouter/openai/gpt-not-in-catalog",
    ],
)
async def test_saved_non_catalog_selection_falls_back_to_root_model(
    saved_model: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = LlmRequest(model="gpt-6-luna")
    profile = InferenceProfile(model=saved_model)

    with (
        caplog.at_level(logging.WARNING, logger="blacki.agent"),
        inference_profile_context(profile),
    ):
        await _run_plugin(_openai_plugin(), request)

    assert request.model == "gpt-6-luna"
    assert "Ignoring saved model selection" in caplog.text
    assert saved_model not in caplog.text


@pytest.mark.asyncio
async def test_saved_plain_gemini_selection_is_ignored_with_an_openai_root(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = LlmRequest(model="gpt-6-luna")
    profile = InferenceProfile(model="gemini-2.5-flash")

    with (
        caplog.at_level(logging.WARNING, logger="blacki.agent"),
        inference_profile_context(profile),
    ):
        await _run_plugin(_openai_plugin(), request)

    assert request.model == "gpt-6-luna"
    assert "Ignoring saved model selection" in caplog.text


@pytest.mark.asyncio
async def test_native_root_ignores_a_saved_openai_selection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = LlmRequest(model="gemini-2.5-flash")
    profile = InferenceProfile(model="gpt-6-luna")

    with (
        caplog.at_level(logging.WARNING, logger="blacki.agent"),
        inference_profile_context(profile),
    ):
        await _run_plugin(TelegramModelOverridePlugin(), request)

    assert request.model == "gemini-2.5-flash"
    assert "Ignoring saved model selection" in caplog.text


@pytest.mark.asyncio
async def test_default_selection_uses_root_model_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = LlmRequest(model="gpt-6-luna")
    profile = InferenceProfile(model="default")

    with (
        caplog.at_level(logging.WARNING, logger="blacki.agent"),
        inference_profile_context(profile),
    ):
        await _run_plugin(_openai_plugin(), request)

    assert request.model == "gpt-6-luna"
    assert caplog.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning", "expected_extra_body"),
    [
        pytest.param(
            ReasoningConfig(effort=ReasoningEffort.HIGH),
            {"reasoning": {"effort": "high"}},
            id="supported-effort-kept",
        ),
        pytest.param(
            ReasoningConfig(effort=ReasoningEffort.NONE),
            None,
            id="unsupported-none-dropped",
        ),
        pytest.param(
            ReasoningConfig(max_tokens=512),
            {"reasoning": {"max_tokens": 512}},
            id="token-budget-passes-through",
        ),
    ],
)
async def test_saved_effort_is_checked_against_the_selected_model(
    reasoning: ReasoningConfig,
    expected_extra_body: dict[str, Any] | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = LlmRequest(model="gpt-6-luna")
    profile = InferenceProfile(model="gpt-6.1-sol", reasoning=reasoning)

    with (
        caplog.at_level(logging.WARNING, logger="blacki.agent"),
        inference_profile_context(profile),
    ):
        await _run_plugin(_openai_plugin(), request)

    assert request.model == "gpt-6.1-sol"
    if expected_extra_body is None:
        assert request.config.http_options is None
        assert "does not accept it" in caplog.text
    else:
        assert request.config.http_options is not None
        assert request.config.http_options.extra_body == expected_extra_body
        assert caplog.text == ""


@pytest.mark.asyncio
async def test_openai_request_carries_hmac_session_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "_SESSION_LABEL_SECRET", _TEST_LABEL_SECRET)
    context = await _callback_context(user_id="chat-7", session_id="session-9")
    request = LlmRequest(model="gpt-6-luna")

    await _run_plugin(_openai_plugin(), request, context=context)

    identity = json.dumps(["blacki", "chat-7", "session-9"])
    expected = hmac.new(
        _TEST_LABEL_SECRET, identity.encode(), hashlib.sha256
    ).hexdigest()
    assert request.config.labels == {SESSION_LABEL_KEY: expected}


@pytest.mark.asyncio
async def test_session_label_is_stable_per_session_and_distinct_otherwise() -> None:
    sessions = [
        ("chat-1", "session-1"),
        ("chat-1", "session-1"),
        ("chat-1", "session-2"),
        ("chat-2", "session-1"),
    ]
    labels: list[str] = []
    for user_id, session_id in sessions:
        request = LlmRequest(model="gpt-6-luna")
        context = await _callback_context(user_id=user_id, session_id=session_id)
        await _run_plugin(_openai_plugin(), request, context=context)
        labels.append((request.config.labels or {})[SESSION_LABEL_KEY])

    assert labels[0] == labels[1]
    assert len(set(labels)) == 3
    for label in labels:
        assert re.fullmatch(r"[0-9a-f]{64}", label)
        assert "chat-1" not in label


@pytest.mark.asyncio
async def test_openai_request_keeps_other_labels_and_stored_items() -> None:
    request = LlmRequest(
        model="gpt-6-luna",
        contents=[_model_turn_with_stored_items()],
        config=types.GenerateContentConfig(labels={"adk_agent_name": "blacki"}),
    )

    await _run_plugin(_openai_plugin(), request)

    labels = request.config.labels or {}
    assert labels["adk_agent_name"] == "blacki"
    assert SESSION_LABEL_KEY in labels
    stored = request.contents[0].parts or []
    assert [part.part_metadata is not None for part in stored] == [True, True, True]


@pytest.mark.asyncio
async def test_native_request_drops_session_label_and_keeps_other_labels() -> None:
    request = LlmRequest(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            labels={"adk_agent_name": "blacki", SESSION_LABEL_KEY: "stale-label"}
        ),
    )

    await _run_plugin(TelegramModelOverridePlugin(), request)

    assert request.config.labels == {"adk_agent_name": "blacki"}


@pytest.mark.asyncio
async def test_native_request_drops_openai_items_and_empty_thoughts() -> None:
    empty_reasoning = types.Part(
        text="",
        thought=True,
        part_metadata=item_metadata(_MODEL, _EMPTY_REASONING_ITEM),
    )
    signed_thought = types.Part(text="", thought=True, thought_signature=b"sig")
    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="Log lunch")]),
            types.Content(role="model", parts=[empty_reasoning]),
            _model_turn_with_stored_items(),
            types.Content(role="model", parts=[signed_thought]),
        ],
    )

    await _run_plugin(TelegramModelOverridePlugin(), request)

    contents = request.contents
    assert len(contents) == 3
    assert contents[0].parts == [types.Part.from_text(text="Log lunch")]
    turn = contents[1].parts or []
    assert [part.part_metadata for part in turn] == [None, None, None]
    assert turn[0].text == "Checked the log." and turn[0].thought is True
    assert turn[1].function_call == types.FunctionCall(
        id=_CALL_ID, name="list_meals", args={"date": "2026-11-12"}
    )
    assert turn[2].text == "You logged lunch."
    assert contents[2].parts == [signed_thought]


@pytest.mark.asyncio
async def test_native_request_without_contents_or_labels_is_unchanged() -> None:
    request = LlmRequest(model="gemini-2.5-flash")

    await _run_plugin(TelegramModelOverridePlugin(), request)

    assert request.contents == []
    assert request.config.labels is None


@pytest.mark.asyncio
async def test_native_request_keeps_a_turn_that_has_no_parts() -> None:
    empty_turn = types.Content(role="user", parts=[])
    request = LlmRequest(model="gemini-2.5-flash", contents=[empty_turn])

    await _run_plugin(TelegramModelOverridePlugin(), request)

    assert request.contents == [empty_turn]
