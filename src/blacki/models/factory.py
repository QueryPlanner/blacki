"""Model selection and construction for the Blacki agent."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("blacki.agent")

if TYPE_CHECKING:
    from google.adk.models.lite_llm import LiteLlm


def normalize_model_for_openrouter(model_name: str) -> str:
    """Map common model IDs to the OpenRouter/LiteLLM form."""
    normalized = model_name.strip()
    lower = normalized.lower()
    if lower.startswith("openrouter/"):
        return normalized
    if "/" in normalized:
        return f"openrouter/{normalized}"
    if normalized.startswith("gemini-"):
        return f"openrouter/google/{normalized}"
    return normalized


def build_model() -> str | LiteLlm:
    """Build the configured native or LiteLLM model."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")
    model: str | LiteLlm = model_name

    use_litellm = openrouter_api_key is not None or "/" in model_name.lower()
    if openrouter_api_key:
        model_name = normalize_model_for_openrouter(model_name)

    if use_litellm:
        try:
            from google.adk.models import LiteLlm

            litellm_kwargs: dict[str, Any] = {}
            if model_name.lower().startswith("openrouter/") and openrouter_api_key:
                litellm_kwargs["api_key"] = openrouter_api_key
            from blacki.observability.costs import CostAwareLiteLLMClient

            litellm_kwargs["llm_client"] = CostAwareLiteLLMClient()

            logger.info("Using LiteLlm for model: %s", model_name)
            return LiteLlm(model=model_name, **litellm_kwargs)
        except ImportError:
            logger.warning(
                "LiteLlm not available, falling back to string model name. "
                "OpenRouter models may not work."
            )

    return model
