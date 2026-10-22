"""Model selection and construction for the Blacki agent."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from blacki.utils.exceptions import ConfigurationError

from .openrouter_ids import is_openrouter_model_id

logger = logging.getLogger("blacki.agent")

if TYPE_CHECKING:
    from .openrouter import OpenRouterLlm


def normalize_model_for_openrouter(model_name: str) -> str:
    """Map common model IDs to the OpenRouter transport form.

    The ``openrouter/`` prefix is retained for compatibility with stored
    Telegram preferences and documented deployment samples.  The adapter
    strips it before calling OpenRouter's API.
    """
    normalized = model_name.strip()
    lower = normalized.lower()
    if lower.startswith("openrouter/"):
        return normalized
    if "/" in normalized:
        return f"openrouter/{normalized}"
    if normalized.startswith("gemini-"):
        return f"openrouter/google/{normalized}"
    return normalized


def build_model() -> str | OpenRouterLlm:
    """Build the configured native Gemini or OpenRouter model."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash").strip()

    if openrouter_api_key:
        normalized = normalize_model_for_openrouter(model_name)
        if not is_openrouter_model_id(normalized):
            raise ConfigurationError(
                f"ROOT_AGENT_MODEL={model_name!r} is not a valid OpenRouter "
                "model ID. Use an author/slug identifier such as "
                "google/gemini-2.5-flash."
            )
        from .openrouter import OpenRouterLlm

        logger.info("Using OpenRouter model: %s", normalized)
        return OpenRouterLlm(model=normalized, api_key=openrouter_api_key)

    if model_name.lower().startswith("openrouter/") or (
        "/" in model_name and not model_name.startswith("models/")
    ):
        raise ConfigurationError(
            f"OPENROUTER_API_KEY is required for model {model_name!r}. "
            "Set the key or use a native Gemini model such as "
            "gemini-2.5-flash."
        )

    if not model_name.lower().startswith(("gemini-", "gemma-", "models/")):
        raise ConfigurationError(
            f"ROOT_AGENT_MODEL={model_name!r} is not a native Gemini model. "
            "Set OPENROUTER_API_KEY for OpenRouter models."
        )

    return model_name
