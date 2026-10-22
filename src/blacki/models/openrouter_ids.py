"""Canonical OpenRouter model identifier handling."""

from __future__ import annotations

OPENROUTER_PREFIX = "openrouter/"


def canonical_openrouter_model_id(model: str) -> str:
    """Return the model identifier accepted by OpenRouter's API.

    Blacki historically stored model references with an ``openrouter/``
    transport prefix.  Strip exactly one prefix when the remainder still looks
    like an ``author/slug`` identifier.  Dynamic router identifiers such as
    ``openrouter/auto`` are already valid API IDs and must not be reduced to
    ``auto``.
    """

    stripped = model.strip()
    if stripped.lower().startswith(OPENROUTER_PREFIX):
        remainder = stripped[len(OPENROUTER_PREFIX) :]
        if "/" in remainder:
            return remainder
    return stripped


def is_openrouter_model_id(model: str) -> bool:
    """Return whether ``model`` is a usable OpenRouter ``author/slug`` id."""

    candidate = canonical_openrouter_model_id(model)
    if candidate.count("/") != 1:
        return False
    author, slug = candidate.split("/", 1)
    return bool(
        author and slug and author not in {".", ".."} and slug not in {".", ".."}
    )
