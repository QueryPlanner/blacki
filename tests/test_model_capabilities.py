"""Tests for provider-neutral model capability discovery."""

from __future__ import annotations

import inspect

import pytest

from blacki.models.capabilities import (
    CatalogCapabilitiesResolver,
    ModelCapabilities,
    ReasoningCapabilities,
)


def test_protocol_shape_is_async_callable() -> None:
    assert inspect.iscoroutinefunction(CatalogCapabilitiesResolver.resolve)


@pytest.mark.asyncio
async def test_catalog_resolver_returns_luna_metadata_and_efforts() -> None:
    result = await CatalogCapabilitiesResolver().resolve("gpt-6-luna")

    assert result == ModelCapabilities(
        model_id="gpt-6-luna",
        name="GPT-6 Luna",
        reasoning=ReasoningCapabilities(
            supports_effort=True,
            supported_efforts=("none", "low", "medium", "high", "xhigh", "max"),
            default_effort="medium",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_id", "offers_none"),
    [
        ("gpt-6-luna", True),
        ("gpt-6.1-terra", True),
        ("gpt-6.1-sol", False),
        ("gpt-6-astra", False),
    ],
)
async def test_catalog_resolver_offers_none_only_where_accepted(
    model_id: str,
    offers_none: bool,
) -> None:
    result = await CatalogCapabilitiesResolver().resolve(model_id)

    assert result is not None
    assert result.reasoning is not None
    assert result.reasoning.supports_effort is True
    assert result.reasoning.default_effort == "medium"
    efforts = result.reasoning.supported_efforts or ()
    assert ("none" in efforts) is offers_none
    assert "low" in efforts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_id", "canonical_id"),
    [
        ("openrouter/openai/gpt-6-luna", "gpt-6-luna"),
        ("GPT-6-LUNA", "gpt-6-luna"),
        (" openrouter/openai/gpt-6.1-terra ", "gpt-6.1-terra"),
    ],
)
async def test_catalog_resolver_accepts_legacy_and_mixed_case_ids(
    model_id: str,
    canonical_id: str,
) -> None:
    result = await CatalogCapabilitiesResolver().resolve(model_id)

    assert result is not None
    assert result.model_id == canonical_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id",
    [
        "unknown-model",
        "gemini-2.5-flash",
        "openrouter/anthropic/claude-x",
        "openrouter/openai/gpt-5.6-luna",
    ],
)
async def test_catalog_resolver_returns_none_outside_the_catalog(
    model_id: str,
) -> None:
    assert await CatalogCapabilitiesResolver().resolve(model_id) is None


@pytest.mark.asyncio
async def test_catalog_resolver_aclose_keeps_resolver_usable() -> None:
    resolver = CatalogCapabilitiesResolver()

    await resolver.aclose()

    result = await resolver.resolve("gpt-6-astra")
    assert result is not None
    assert result.model_id == "gpt-6-astra"
