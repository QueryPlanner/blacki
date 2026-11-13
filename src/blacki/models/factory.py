"""Model selection and construction for the Blacki root agent."""

from __future__ import annotations

import logging
import os
from typing import Final

from blacki.utils.exceptions import ConfigurationError

from .catalog import get_model, supported_model_ids
from .openai_responses import OpenAIResponsesLlm

logger = logging.getLogger("blacki.agent")

DEFAULT_NATIVE_MODEL: Final = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL: Final = "gpt-6-luna"
_LEGACY_OPENAI_PREFIX: Final = "openrouter/openai/"
_NATIVE_GEMINI_PREFIXES: Final = ("gemini-", "gemma-", "models/")


def build_model() -> str | OpenAIResponsesLlm:
    """Build the root model from ``ROOT_AGENT_MODEL``.

    A catalog OpenAI id (or its legacy ``openrouter/openai/`` form) returns an
    ``OpenAIResponsesLlm`` and requires ``OPENAI_API_KEY``. A native Gemini id
    (``gemini-*``, ``gemma-*`` or ``models/*``) is returned as a plain string for
    ADK's native Gemini path. When ``ROOT_AGENT_MODEL`` is unset, the default is
    ``gpt-6-luna`` if ``OPENAI_API_KEY`` is set, and ``gemini-2.5-flash`` otherwise.

    Raises:
        ConfigurationError: The OpenAI key is missing for an OpenAI model, or the
            model id is neither a catalog id nor a native Gemini id.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("ROOT_AGENT_MODEL", "").strip()
    if not model_name:
        model_name = DEFAULT_OPENAI_MODEL if api_key else DEFAULT_NATIVE_MODEL

    entry = get_model(model_name)
    if entry is not None:
        if not api_key:
            raise ConfigurationError(
                f"ROOT_AGENT_MODEL={model_name!r} is an OpenAI model and requires "
                "OPENAI_API_KEY. Set OPENAI_API_KEY, or use a native Gemini model "
                "such as gemini-2.5-flash."
            )
        if model_name.lower().startswith(_LEGACY_OPENAI_PREFIX):
            logger.info(
                "ROOT_AGENT_MODEL=%s uses the legacy OpenRouter form; using %s.",
                model_name,
                entry.id,
            )
        logger.info("Using OpenAI model: %s", entry.id)
        return OpenAIResponsesLlm(model=entry.id, api_key=api_key)

    if model_name.lower().startswith(_NATIVE_GEMINI_PREFIXES):
        return model_name

    raise ConfigurationError(
        f"ROOT_AGENT_MODEL={model_name!r} is not supported. Supported OpenAI model "
        f"ids: {', '.join(supported_model_ids())}. Or use a native Gemini model "
        "such as gemini-2.5-flash."
    )
