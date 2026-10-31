"""Minimal OpenRouter model adapter for Google ADK.

This replaces LiteLLM for Blacki's OpenRouter traffic.  The adapter uses the
OpenAI-compatible chat-completions API directly, translates ADK requests at the
edge, and records provider-reported usage in the local cost ledger.

Plaintext reasoning is echoed back on assistant messages.  Encrypted
``reasoning_details`` signatures are not preserved, matching the previous
adapter's behavior.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from openai import AsyncOpenAI
from pydantic import PrivateAttr

from blacki.observability.costs import observe_response
from blacki.observability.ledger import default_usage_ledger_path

from .openrouter_ids import canonical_openrouter_model_id
from .openrouter_request import build_chat_request
from .openrouter_response import (
    OpenRouterStreamAccumulator,
    response_to_llm_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2

ClientFactory = Callable[[float, int], AsyncOpenAI]


class OpenRouterLlm(BaseLlm):
    """ADK model backed by OpenRouter's OpenAI-compatible endpoint.

    Credentials and transport settings are private attributes so they never
    appear in ``repr()``, ``model_dump()``, or serialized ADK state.  A client
    is created per model call and closed in a ``finally`` block; storing an
    ``AsyncOpenAI`` instance on a pydantic model would break deep copying and
    leak unclosed transports across event loops.
    """

    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr()
    _timeout_seconds: float = PrivateAttr()
    _max_retries: int = PrivateAttr()
    _ledger_path: Path = PrivateAttr()
    _client_factory: ClientFactory | None = PrivateAttr(default=None)

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        ledger_path: Path | str | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        super().__init__(model=model)

        key = api_key.strip()
        if not key:
            raise ValueError("OpenRouter API key is required.")
        if timeout_seconds <= 0:
            raise ValueError("OpenRouter timeout must be positive.")
        if max_retries < 0:
            raise ValueError("OpenRouter max_retries must be non-negative.")

        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._ledger_path = (
            Path(ledger_path)
            if ledger_path is not None
            else default_usage_ledger_path()
        )
        self._client_factory = client_factory

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate one model turn through OpenRouter."""

        self._maybe_append_user_content(llm_request)

        requested_model = llm_request.model or self.model
        api_model = canonical_openrouter_model_id(requested_model)
        request = build_chat_request(llm_request, model=api_model)

        timeout_seconds = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._timeout_seconds
        )
        max_retries = (
            request.max_retries
            if request.max_retries is not None
            else self._max_retries
        )
        client = self._create_client(timeout_seconds, max_retries)
        try:
            if stream:
                async for response in self._stream(
                    client,
                    request.create_kwargs(stream=True),
                    ledger_model=requested_model,
                ):
                    yield response
            else:
                completion = await client.chat.completions.create(
                    **request.create_kwargs(stream=False)
                )
                raw = completion.model_dump(exclude_none=True)
                await observe_response(raw, requested_model, self._ledger_path)
                yield response_to_llm_response(raw)
        finally:
            await client.close()

    async def _stream(
        self,
        client: AsyncOpenAI,
        create_kwargs: dict[str, Any],
        *,
        ledger_model: str,
    ) -> AsyncGenerator[LlmResponse, None]:
        stream = await client.chat.completions.create(**create_kwargs)
        accumulator = OpenRouterStreamAccumulator()
        try:
            async for chunk in stream:
                raw = chunk.model_dump(exclude_none=True)
                await observe_response(raw, ledger_model, self._ledger_path)
                for response in accumulator.add(raw):
                    yield response
        finally:
            await stream.close()
        yield accumulator.finish()

    def _create_client(self, timeout_seconds: float, max_retries: int) -> AsyncOpenAI:
        if self._client_factory is not None:
            return self._client_factory(timeout_seconds, max_retries)
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
