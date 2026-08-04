"""Provider-neutral model capability discovery.

The runtime only needs a small, stable description of model reasoning
controls.  This module keeps the provider-specific OpenRouter response shape at
the edge so Telegram and ADK code can consume the same capability contract as
other providers are added later.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_BASE_URL = "https://openrouter.ai/api/v1/model"
DEFAULT_CACHE_TTL_SECONDS = 15 * 60.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class ReasoningCapabilities:
    """Normalized reasoning controls exposed by a model.

    ``supported_efforts`` is ``None`` when OpenRouter explicitly says that all
    gateway effort values are accepted.  An empty tuple means the provider
    published an empty list.  ``supports_effort`` keeps those cases distinct
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

    async def resolve(
        self,
        model_id: str,
        *,
        openrouter_routed: bool = False,
    ) -> ModelCapabilities | None:
        """Return capabilities, or ``None`` when they are unavailable."""


def normalize_openrouter_model_id(
    model_id: object,
    *,
    openrouter_routed: bool = False,
) -> str | None:
    """Return the OpenRouter API identifier for a model reference.

    LiteLLM model strings use ``openrouter/<author>/<slug>`` while the
    OpenRouter Models API uses ``<author>/<slug>``.  Exactly one transport
    prefix is removed.  Bare provider IDs are accepted only when the caller
    explicitly states that the model is routed through OpenRouter; this avoids
    making a native Google or future provider lookup by accident.
    """

    if not isinstance(model_id, str):
        return None

    normalized = model_id.strip()
    if not normalized:
        return None

    prefix = "openrouter/"
    if normalized.lower().startswith(prefix):
        normalized = normalized[len(prefix) :]
    elif not openrouter_routed:
        return None

    # OpenRouter model IDs are author/slug.  Keep variant suffixes (for
    # example ``:free``) in the slug, but reject dynamic/unqualified values.
    if normalized.count("/") != 1:
        return None
    author, slug = normalized.split("/", 1)
    if not author or not slug or author in {".", ".."} or slug in {".", ".."}:
        return None
    return normalized


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    capabilities: ModelCapabilities
    expires_at: float


class OpenRouterModelCapabilitiesResolver:
    """Resolve model metadata from OpenRouter with a bounded stale cache.

    The resolver performs one-model lookups, rather than downloading the full
    catalog for every Telegram interaction.  Successful responses are cached
    for ``cache_ttl_seconds``.  If a refresh fails, an expired successful entry
    is returned so a temporary catalog outage never removes existing controls.
    An injected ``httpx.AsyncClient`` is never closed by this class; when no
    client is supplied, the resolver owns and closes the client in ``aclose``.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        base_url: str = OPENROUTER_MODELS_BASE_URL,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        openrouter_routed: bool = False,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._clock = clock
        self._cache_ttl_seconds = float(cache_ttl_seconds)
        self._timeout_seconds = float(timeout_seconds)
        self._base_url = base_url.rstrip("/")
        self._openrouter_routed = openrouter_routed
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._owns_client = client is None
        # Model metadata is public. Only attach credentials when a caller
        # explicitly requests authenticated, user-filtered discovery.
        configured_key = api_key.strip() if api_key is not None else ""
        self._headers = (
            {"Authorization": f"Bearer {configured_key}"} if configured_key else {}
        )

        if client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout_seconds,
                headers=self._headers or None,
            )
        else:
            self._client = client

    async def resolve(
        self,
        model_id: str,
        *,
        openrouter_routed: bool | None = None,
    ) -> ModelCapabilities | None:
        """Return normalized capabilities for ``model_id``.

        ``openrouter_routed`` overrides the constructor default for one call.
        A model that is not identifiable as ``author/slug`` is unavailable and
        does not trigger a network request.
        """

        routed = (
            self._openrouter_routed if openrouter_routed is None else openrouter_routed
        )
        canonical_id = normalize_openrouter_model_id(
            model_id,
            openrouter_routed=routed,
        )
        if canonical_id is None:
            return None

        now = self._clock()
        cached = self._cache.get(canonical_id)
        if cached is not None and now < cached.expires_at:
            return cached.capabilities

        async with self._lock:
            # Another coroutine may have refreshed while this one waited.
            now = self._clock()
            cached = self._cache.get(canonical_id)
            if cached is not None and now < cached.expires_at:
                return cached.capabilities

            try:
                capabilities = await self._fetch(canonical_id)
            except (
                httpx.HTTPError,
                TimeoutError,
                ValueError,
                TypeError,
                KeyError,
            ) as exc:
                if cached is not None:
                    logger.warning(
                        "OpenRouter model capability refresh failed for %s; "
                        "using stale metadata: %s",
                        canonical_id,
                        exc,
                    )
                    return cached.capabilities
                logger.warning(
                    "OpenRouter model capability lookup failed for %s: %s",
                    canonical_id,
                    exc,
                )
                return None

            self._cache[canonical_id] = _CacheEntry(
                capabilities=capabilities,
                expires_at=self._clock() + self._cache_ttl_seconds,
            )
            return capabilities

    async def get_capabilities(
        self,
        model_id: str,
        *,
        openrouter_routed: bool | None = None,
    ) -> ModelCapabilities | None:
        """Compatibility alias for callers that prefer an explicit name."""

        return await self.resolve(
            model_id,
            openrouter_routed=openrouter_routed,
        )

    async def aclose(self) -> None:
        """Close the HTTP client when this resolver created it."""

        if self._owns_client:
            await self._client.aclose()

    async def _fetch(self, canonical_id: str) -> ModelCapabilities:
        encoded_id = quote(canonical_id, safe="/")
        url = f"{self._base_url}/{encoded_id}"
        request_kwargs: dict[str, Any] = {"timeout": self._timeout_seconds}
        if self._headers:
            request_kwargs["headers"] = self._headers
        response = await self._client.get(url, **request_kwargs)
        response.raise_for_status()
        payload = response.json()
        return _parse_model_capabilities(payload, fallback_model_id=canonical_id)


def _parse_model_capabilities(
    payload: object,
    *,
    fallback_model_id: str,
) -> ModelCapabilities:
    """Parse one OpenRouter model response without trusting optional fields."""

    if not isinstance(payload, dict):
        raise ValueError("OpenRouter model response must be an object")
    data = payload.get("data")
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        raise ValueError("OpenRouter model response has no model data")

    model_id = data.get("id")
    normalized_id = model_id.strip() if isinstance(model_id, str) else fallback_model_id
    if not normalized_id:
        normalized_id = fallback_model_id

    name = data.get("name")
    display_name = name.strip() if isinstance(name, str) and name.strip() else None
    supported_parameters = _string_set(data.get("supported_parameters"))
    reasoning_raw = data.get("reasoning")
    reasoning: ReasoningCapabilities | None = None

    if isinstance(reasoning_raw, dict):
        efforts_value = reasoning_raw.get("supported_efforts", _MISSING)
        supported_efforts: tuple[str, ...] | None
        if efforts_value is _MISSING:
            supports_effort = False
            supported_efforts = None
        elif efforts_value is None:
            supports_effort = True
            supported_efforts = None
        elif isinstance(efforts_value, list):
            supports_effort = True
            supported_efforts = _normalize_efforts(efforts_value)
        else:
            supports_effort = False
            supported_efforts = None

        default_effort = _normalize_effort(reasoning_raw.get("default_effort"))
        default_enabled = reasoning_raw.get("default_enabled")
        if not isinstance(default_enabled, bool):
            default_enabled = None
        supports_max_tokens = reasoning_raw.get("supports_max_tokens") is True
        mandatory = reasoning_raw.get("mandatory") is True
        reasoning = ReasoningCapabilities(
            supports_effort=supports_effort,
            supported_efforts=supported_efforts,
            default_effort=default_effort,
            default_enabled=default_enabled,
            supports_max_tokens=supports_max_tokens,
            mandatory=mandatory,
        )
    elif "reasoning" in supported_parameters:
        # The model accepts a reasoning object, but it does not advertise an
        # effort selector. Keep the capability visible without inventing UI
        # options that the metadata did not publish.
        reasoning = ReasoningCapabilities(
            supports_effort=False,
            supported_efforts=None,
        )

    return ModelCapabilities(
        model_id=normalized_id,
        name=display_name,
        reasoning=reasoning,
    )


_MISSING = object()


def _string_set(value: object) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {
        item.strip().lower() for item in value if isinstance(item, str) and item.strip()
    }


def _normalize_efforts(value: list[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        effort = _normalize_effort(item)
        if effort is not None and effort not in seen:
            normalized.append(effort)
            seen.add(effort)
    return tuple(normalized)


def _normalize_effort(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None
