"""Provider-neutral model capability discovery.

The runtime only needs a small, stable description of model reasoning
controls. This module exposes that contract to Telegram and ADK code. The
static OpenAI model catalog is the only source of capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .catalog import get_model


@dataclass(frozen=True, slots=True)
class ReasoningCapabilities:
    """Normalized reasoning controls exposed by a model.

    ``supported_efforts`` is ``None`` when a model accepts an effort selector
    without publishing the accepted values. An empty tuple means the model
    published an empty list. ``supports_effort`` keeps those cases distinct
    from models that do not publish an effort selector.
    """

    supports_effort: bool = False
    supported_efforts: tuple[str, ...] | None = None
    default_effort: str | None = None
    default_enabled: bool | None = None
    supports_max_tokens: bool = False
    mandatory: bool = False


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Provider-neutral capabilities for one canonical model identifier."""

    model_id: str
    name: str | None = None
    reasoning: ReasoningCapabilities | None = None


class ModelCapabilitiesResolver(Protocol):
    """Resolve capabilities for a model selected by a caller."""

    async def resolve(self, model_id: str) -> ModelCapabilities | None:
        """Return capabilities, or ``None`` when they are unavailable."""


class CatalogCapabilitiesResolver:
    """Resolve reasoning efforts from the static OpenAI model catalog.

    The catalog is the source of truth for supported OpenAI models, so this
    resolver makes no network calls and holds no state.
    """

    async def resolve(self, model_id: str) -> ModelCapabilities | None:
        """Return catalog capabilities for ``model_id``, or ``None``.

        Catalog ids are accepted with or without the legacy
        ``openrouter/openai/`` prefix.
        """

        entry = get_model(model_id)
        if entry is None:
            return None
        return ModelCapabilities(
            model_id=entry.id,
            name=entry.display_name,
            reasoning=ReasoningCapabilities(
                supports_effort=True,
                supported_efforts=entry.efforts,
                default_effort=entry.default_effort,
            ),
        )

    async def aclose(self) -> None:
        """Release nothing; the resolver holds no HTTP client."""
