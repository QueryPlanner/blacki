"""OpenAI Responses model adapter for Google ADK.

``OpenAIResponsesLlm`` sends each ADK model call to OpenAI's Responses API.
Request and response translation live in ``openai_request.py`` and
``openai_response.py``. This module owns the transport, the fallback for
rejected reasoning (design D7), and the mapping of failures to
``ProviderError``.

Design decisions are recorded in
``openspec/changes/replace-openrouter-with-openai-responses/design.md``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from contextlib import aclosing, contextmanager
from pathlib import Path
from typing import Any, Final

import openai
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from openai import AsyncOpenAI
from pydantic import PrivateAttr

from blacki.observability.costs import observe_openai_usage
from blacki.observability.ledger import default_usage_ledger_path
from blacki.utils.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderPermissionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
)

from .catalog import normalize_model_id
from .openai_request import ResponsesRequest, build_responses_request
from .openai_response import (
    OpenAIResponseError,
    OpenAIStreamAccumulator,
    response_to_llm_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2

ClientFactory = Callable[[float, int], AsyncOpenAI]

# Label written by Blacki's override plugin on each OpenAI request. It holds a
# non-reversible per-session key and is never sent to OpenAI.
SESSION_LABEL_KEY: Final = "blacki_session"

_REASONING = "reasoning"
_INVALID_ENCRYPTED_CONTENT = "invalid_encrypted_content"
_INPUT_INDEX_PARAM = re.compile(r"input\[(\d+)\]")


class OpenAIResponsesLlm(BaseLlm):
    """ADK model backed by OpenAI's Responses API.

    Credentials and transport settings are private attributes, so they never
    appear in ``repr()``, ``model_dump()``, or serialized ADK state. A client
    is created for each model call and closed in a ``finally`` block.

    When OpenAI rejects a replayed reasoning item, the call is retried once
    without reasoning, and later calls for the same session skip reasoning until
    the process restarts (D7). The session is identified by the
    ``SESSION_LABEL_KEY`` label on the request. A request without that label falls
    back to its model id. The fallback is kept per adapter instance.
    """

    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr()
    _timeout_seconds: float = PrivateAttr()
    _max_retries: int = PrivateAttr()
    _client_factory: ClientFactory | None = PrivateAttr(default=None)
    _ledger_path: Path = PrivateAttr()
    _reasoning_downgraded: set[str] = PrivateAttr(default_factory=set)

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client_factory: ClientFactory | None = None,
        ledger_path: Path | str | None = None,
    ) -> None:
        super().__init__(model=model)

        key = api_key.strip()
        if not key:
            raise ValueError("OpenAI API key is required.")
        if timeout_seconds <= 0:
            raise ValueError("OpenAI timeout must be positive.")
        if max_retries < 0:
            raise ValueError("OpenAI max_retries must be non-negative.")

        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._client_factory = client_factory
        self._ledger_path = (
            Path(ledger_path)
            if ledger_path is not None
            else default_usage_ledger_path()
        )

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate one model turn through OpenAI's Responses API."""

        self._maybe_append_user_content(llm_request)

        model = normalize_model_id(llm_request.model or self.model)
        downgrade_key = _downgrade_key(llm_request, model)
        request = build_responses_request(
            llm_request,
            model=model,
            replay_reasoning=downgrade_key not in self._reasoning_downgraded,
        )
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
                # aclosing ends the inner stream when the consumer stops early,
                # before the client is closed below.
                async with aclosing(
                    self._stream(
                        client,
                        llm_request,
                        request,
                        model=model,
                        downgrade_key=downgrade_key,
                    )
                ) as responses:
                    async for response in responses:
                        yield response
            else:
                yield await self._complete(
                    client,
                    llm_request,
                    request,
                    model=model,
                    downgrade_key=downgrade_key,
                )
        finally:
            await client.close()

    async def _complete(
        self,
        client: AsyncOpenAI,
        llm_request: LlmRequest,
        request: ResponsesRequest,
        *,
        model: str,
        downgrade_key: str,
    ) -> LlmResponse:
        payload = await self._create(
            client,
            llm_request,
            request,
            model=model,
            stream=False,
            downgrade_key=downgrade_key,
        )
        with _provider_errors():
            response = response_to_llm_response(
                payload.model_dump(exclude_none=True, mode="json"),
                model=model,
            )
        await self._after_final_response(response, model=model)
        return response

    async def _stream(
        self,
        client: AsyncOpenAI,
        llm_request: LlmRequest,
        request: ResponsesRequest,
        *,
        model: str,
        downgrade_key: str,
    ) -> AsyncGenerator[LlmResponse, None]:
        events = await self._create(
            client,
            llm_request,
            request,
            model=model,
            stream=True,
            downgrade_key=downgrade_key,
        )
        accumulator = OpenAIStreamAccumulator(model)
        try:
            with _provider_errors():
                async for event in events:
                    payload = event.model_dump(exclude_none=True, mode="json")
                    for partial in accumulator.add(payload):
                        yield partial
                final = accumulator.finish()
        finally:
            await events.close()
        await self._after_final_response(final, model=model)
        yield final

    async def _create(
        self,
        client: AsyncOpenAI,
        llm_request: LlmRequest,
        request: ResponsesRequest,
        *,
        model: str,
        stream: bool,
        downgrade_key: str,
    ) -> Any:
        """Send one Responses call, retrying once without reasoning on D7.

        The downgrade key is recorded only after the retry succeeds, so a retry
        that also fails leaves later calls for that session free to try reasoning
        again.
        """

        with _provider_errors():
            try:
                return await client.responses.create(
                    **request.create_kwargs(stream=stream)
                )
            except openai.BadRequestError as exc:
                if not _rejects_reasoning(exc, request):
                    raise
                rejection_code = exc.code

            logger.warning(
                "OpenAI rejected replayed reasoning (code=%s); retrying without it.",
                rejection_code,
            )
            retry = build_responses_request(
                llm_request, model=model, replay_reasoning=False
            )
            result = await client.responses.create(**retry.create_kwargs(stream=stream))
            self._reasoning_downgraded.add(downgrade_key)
            return result

    async def _after_final_response(self, response: LlmResponse, *, model: str) -> None:
        """Record usage and estimated cost for one final response.

        Runs once per model call, streamed or not, before the response is
        yielded so the ADK after-model callback can attach the observation.
        """

        metadata = response.custom_metadata or {}
        usage = metadata.get("openai_usage")
        if isinstance(usage, Mapping):
            await observe_openai_usage(usage, model, self._ledger_path)

    def _create_client(self, timeout_seconds: float, max_retries: int) -> AsyncOpenAI:
        if self._client_factory is not None:
            return self._client_factory(timeout_seconds, max_retries)
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )


@contextmanager
def _provider_errors() -> Iterator[None]:
    """Re-raise OpenAI SDK and Responses failures as ``ProviderError``.

    ``from None`` hides the SDK exception's chained traceback. The SDK message
    and body can echo the API key or prompt text, so neither is copied.
    """

    try:
        yield
    except openai.APIError as exc:
        raise _provider_error(exc) from None
    except OpenAIResponseError as exc:
        raise _response_error(exc) from None


def _provider_error(exc: openai.APIError) -> ProviderError:
    """Map an OpenAI SDK error to a provider error.

    Only the status, error code and request id are copied. A connection failure
    never received a response, so it carries no status.
    """

    if isinstance(exc, openai.APIConnectionError):
        return ProviderServerError()
    if isinstance(exc, openai.APIStatusError):
        error_type = _status_error_type(exc.status_code)
        return error_type(
            status=exc.status_code,
            code=exc.code,
            request_id=exc.request_id,
        )
    return ProviderError(code=exc.code)


def _status_error_type(status: int) -> type[ProviderError]:
    if status == 401:
        return ProviderAuthenticationError
    if status == 403:
        return ProviderPermissionError
    if status == 429:
        return ProviderRateLimitError
    if 400 <= status < 500:
        return ProviderRequestError
    if 500 <= status < 600:
        return ProviderServerError
    return ProviderError


def _response_error(exc: OpenAIResponseError) -> ProviderError:
    """Map a failed or malformed Responses payload. OpenAI's text is not copied."""

    return ProviderError(code=exc.code)


def _downgrade_key(llm_request: LlmRequest, model: str) -> str:
    """Return the key that scopes a reasoning downgrade to one session (D7).

    Blacki's override plugin labels each OpenAI request with its session. Without
    that label, the normalized model id is the key, so all unlabelled requests to
    the model share one downgrade.
    """

    labels = llm_request.config.labels or {}
    session_key = labels.get(SESSION_LABEL_KEY)
    if session_key:
        return f"session:{session_key}"
    return f"model:{model}"


def _rejects_reasoning(exc: openai.BadRequestError, request: ResponsesRequest) -> bool:
    """Return whether a 400 is OpenAI rejecting a replayed reasoning item.

    The request must hold a reasoning item, and either the error code or the
    ``param`` (for example ``input[3]``) must point at one. Without a reasoning
    item, dropping reasoning would leave the request unchanged.
    """

    if not any(item.get("type") == _REASONING for item in request.input):
        return False
    if exc.code == _INVALID_ENCRYPTED_CONTENT:
        return True
    param = exc.param
    if not isinstance(param, str):
        return False
    match = _INPUT_INDEX_PARAM.match(param)
    if match is None:
        return False
    index = int(match.group(1))
    return index < len(request.input) and (
        request.input[index].get("type") == _REASONING
    )
