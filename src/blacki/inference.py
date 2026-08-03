"""Request-scoped model and reasoning controls for ADK turns.

The ADK ``LlmRequest`` does not expose a provider-neutral field for arbitrary
LiteLLM parameters.  OpenRouter's reasoning object is therefore carried in
``GenerateContentConfig.http_options.extra_body`` and translated by ADK's
LiteLLM adapter at the request boundary.  This module keeps the profile
immutable and request-scoped so concurrent Telegram turns cannot change one
another's model settings.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Any

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from blacki.utils.preferences import SqlitePreferencesStorage

INFERENCE_PROFILE_PREFERENCE_KEY = "telegram_inference_profile"
LEGACY_MODEL_PREFERENCE_KEY = "telegram_model_override"


class ReasoningEffort(str, Enum):
    """Gateway effort values accepted by the OpenRouter reasoning object."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


# Descriptive alias for callers that want to distinguish gateway values from
# provider-specific future reasoning settings.
GatewayReasoningEffort = ReasoningEffort


class ReasoningConfig(BaseModel):
    """Provider-neutral reasoning settings.

    ``effort`` is the currently supported OpenRouter gateway control.  The
    optional ``max_tokens`` field leaves room for providers that expose a
    token budget instead of an effort enum.  Exactly one control is accepted
    at a time so an invalid mixed provider request cannot be sent silently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    effort: ReasoningEffort | None = None
    max_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_one_control(self) -> ReasoningConfig:
        if self.effort is None and self.max_tokens is None:
            raise ValueError("reasoning requires effort or max_tokens")
        if self.effort is not None and self.max_tokens is not None:
            raise ValueError("reasoning effort and max_tokens are mutually exclusive")
        return self


class InferenceProfile(BaseModel):
    """Immutable model and reasoning settings for one complete ADK turn.

    ``reasoning=None`` means inherit the model/provider default.  To
    explicitly disable reasoning, use ``ReasoningConfig(effort='none')``.
    ``InferenceProfile()`` is consequently a useful explicit per-chat
    snapshot that suppresses the process-wide environment default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = None
    reasoning: ReasoningConfig | None = None


_ACTIVE_PROFILE: ContextVar[InferenceProfile | None] = ContextVar(
    "blacki_active_inference_profile", default=None
)


@contextmanager
def inference_profile_context(
    profile: InferenceProfile | None,
) -> Iterator[None]:
    """Install a profile for the current async context and always restore it."""

    token = _ACTIVE_PROFILE.set(profile)
    try:
        yield
    finally:
        _ACTIVE_PROFILE.reset(token)


def get_active_inference_profile() -> InferenceProfile | None:
    """Return the profile snapshot active in this task, if any."""

    return _ACTIVE_PROFILE.get()


def reasoning_payload(reasoning: ReasoningConfig) -> dict[str, Any]:
    """Serialize reasoning settings to the OpenRouter request shape."""

    return reasoning.model_dump(mode="json", exclude_none=True)


def apply_inference_profile(
    llm_request: LlmRequest,
    profile: InferenceProfile,
) -> None:
    """Apply a profile to one ADK LLM request without mutating shared config.

    The model is applied to ``llm_request.model``.  Explicit reasoning is
    merged into a copied ``http_options.extra_body`` mapping; unrelated
    provider parameters remain intact.  Inherited reasoning intentionally
    leaves an existing request-level value unchanged.
    """

    if profile.model:
        llm_request.model = profile.model

    if profile.reasoning is None:
        return

    config = llm_request.config
    http_options = config.http_options
    existing_extra_body: Mapping[str, Any] = {}
    if http_options is not None and http_options.extra_body is not None:
        candidate_extra_body: object = http_options.extra_body
        if isinstance(candidate_extra_body, Mapping):
            existing_extra_body = candidate_extra_body
        else:
            logger.warning(
                "Ignoring non-mapping LLM extra_body while applying reasoning"
            )

    merged_extra_body = dict(existing_extra_body)
    merged_extra_body["reasoning"] = reasoning_payload(profile.reasoning)

    if http_options is None:
        config.http_options = types.HttpOptions(extra_body=merged_extra_body)
    else:
        config.http_options = http_options.model_copy(
            update={"extra_body": merged_extra_body}
        )


def parse_inference_profile(value: Any) -> InferenceProfile | None:
    """Parse a stored preference, failing closed for malformed values."""

    if value is None:
        return None
    if isinstance(value, InferenceProfile):
        return value
    try:
        return InferenceProfile.model_validate(value)
    except (TypeError, ValueError, ValidationError):
        logger.warning("Ignoring malformed inference profile preference")
        return None


async def load_inference_profile(
    storage: SqlitePreferencesStorage,
    chat_id: str,
) -> InferenceProfile:
    """Load a chat profile with canonical and legacy preference fallback.

    Presence of the canonical key, including an explicit ``null`` value,
    takes precedence over the legacy model-only key.  This prevents an old
    model override from being resurrected.  A profile without an explicit
    reasoning setting still inherits the optional process fallback.
    """

    missing = object()
    stored_profile = await storage.get(
        chat_id,
        INFERENCE_PROFILE_PREFERENCE_KEY,
        missing,
    )
    if stored_profile is not missing:
        profile = parse_inference_profile(stored_profile)
        return _merge_environment_reasoning(profile or InferenceProfile())

    legacy_model = await storage.get(chat_id, LEGACY_MODEL_PREFERENCE_KEY, missing)
    if legacy_model is not missing and legacy_model not in (None, "", "default"):
        return _merge_environment_reasoning(InferenceProfile(model=str(legacy_model)))

    return inference_profile_from_environment()


def _merge_environment_reasoning(profile: InferenceProfile) -> InferenceProfile:
    """Apply the process fallback only when the chat has no explicit effort."""

    if profile.reasoning is not None:
        return profile
    fallback = inference_profile_from_environment()
    if fallback.reasoning is None:
        return profile
    return profile.model_copy(update={"reasoning": fallback.reasoning})


async def update_inference_profile(
    storage: SqlitePreferencesStorage,
    chat_id: str,
    updates: Mapping[str, Any],
) -> InferenceProfile:
    """Atomically update and validate one chat's canonical profile."""

    unknown_fields = set(updates) - {"model", "reasoning"}
    if unknown_fields:
        fields = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unsupported inference profile fields: {fields}")

    normalized_updates = dict(updates)
    if "reasoning" in normalized_updates:
        raw_reasoning = normalized_updates["reasoning"]
        if raw_reasoning is None:
            normalized_updates["reasoning"] = None
        else:
            reasoning = (
                raw_reasoning
                if isinstance(raw_reasoning, ReasoningConfig)
                else ReasoningConfig.model_validate(raw_reasoning)
            )
            normalized_updates["reasoning"] = reasoning.model_dump(
                mode="json", exclude_none=True
            )

    # Validate the shape before writing.  ``InferenceProfile`` supplies the
    # nested Pydantic conversion for dictionary reasoning values.
    InferenceProfile.model_validate(normalized_updates)
    merged = await storage.update_dict(
        chat_id,
        INFERENCE_PROFILE_PREFERENCE_KEY,
        normalized_updates,
    )
    profile = parse_inference_profile(merged)
    if profile is None:
        raise ValueError("Stored inference profile is invalid")
    return profile


def inference_profile_from_environment(
    environ: Mapping[str, str] | None = None,
) -> InferenceProfile:
    """Build the process-wide fallback from ``ROOT_AGENT_REASONING_EFFORT``.

    Empty or unset values inherit the provider default.  Invalid values fail
    closed to inherit and emit a warning instead of sending an unsafe request.
    """

    values = os.environ if environ is None else environ
    raw_effort = values.get("ROOT_AGENT_REASONING_EFFORT", "").strip().lower()
    if not raw_effort:
        return InferenceProfile()

    try:
        effort = ReasoningEffort(raw_effort)
    except ValueError:
        logger.warning(
            "Ignoring unsupported ROOT_AGENT_REASONING_EFFORT=%r; using inherit",
            raw_effort,
        )
        return InferenceProfile()

    return InferenceProfile(reasoning=ReasoningConfig(effort=effort))


def inference_profile_from_mapping(
    value: Mapping[str, Any],
) -> InferenceProfile | None:
    """Parse a mapping while keeping a typed helper for storage callers."""

    return parse_inference_profile(value)
