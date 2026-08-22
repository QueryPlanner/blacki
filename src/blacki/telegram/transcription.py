"""Cloudflare Workers AI transcription for Telegram voice notes."""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

CLOUDFLARE_WHISPER_MODEL = "@cf/openai/whisper-large-v3-turbo"
CLOUDFLARE_WHISPER_RUN_PATH = "/ai/run/"
MAX_CLOUDFLARE_WHISPER_AUDIO_BYTES = 8 * 1024 * 1024
CLOUDFLARE_WHISPER_CONNECT_TIMEOUT_SECONDS = 5.0
CLOUDFLARE_WHISPER_READ_TIMEOUT_SECONDS = 90.0
MAX_CONCURRENT_CLOUDFLARE_TRANSCRIPTIONS = 1


class CloudflareWhisperError(RuntimeError):
    """Raised when Cloudflare returns an unusable transcription response."""


@dataclass(frozen=True, slots=True)
class CloudflareWhisperConfig:
    """Credentials for the Cloudflare Workers AI REST API."""

    account_id: str
    api_token: str = field(repr=False)


@dataclass(slots=True)
class CloudflareWhisperTranscriber:
    """Call Cloudflare's hosted Whisper model with transient audio bytes."""

    config: CloudflareWhisperConfig
    http_transport: httpx.AsyncBaseTransport | None = field(
        default=None,
        repr=False,
    )
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(
            MAX_CONCURRENT_CLOUDFLARE_TRANSCRIPTIONS
        ),
        init=False,
        repr=False,
    )

    @classmethod
    def from_environment(cls) -> CloudflareWhisperTranscriber | None:
        """Create a transcriber when both Cloudflare credentials are present."""
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        if not account_id or not api_token:
            return None
        return cls(CloudflareWhisperConfig(account_id, api_token))

    @property
    def endpoint(self) -> str:
        """Return the Workers AI model-run endpoint."""
        account_id = quote(self.config.account_id, safe="")
        model = quote(CLOUDFLARE_WHISPER_MODEL, safe="@/")
        return (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}{CLOUDFLARE_WHISPER_RUN_PATH}{model}"
        )

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, creating it lazily."""
        if self._client is None:
            timeout = httpx.Timeout(
                CLOUDFLARE_WHISPER_READ_TIMEOUT_SECONDS,
                connect=CLOUDFLARE_WHISPER_CONNECT_TIMEOUT_SECONDS,
            )
            self._client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                transport=self.http_transport,
            )
        return self._client

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe one OGG voice note without persisting its audio."""
        if not audio_bytes:
            raise CloudflareWhisperError("Cloudflare transcription audio is empty")
        if len(audio_bytes) > MAX_CLOUDFLARE_WHISPER_AUDIO_BYTES:
            raise CloudflareWhisperError("Cloudflare transcription audio is too large")

        async with self._semaphore:
            payload = {
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
                "task": "transcribe",
            }
            try:
                response = await self._get_client().post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.config.api_token}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise CloudflareWhisperError(
                    "Cloudflare transcription request failed"
                ) from exc
            finally:
                payload["audio"] = ""

        if not isinstance(body, dict) or body.get("success") is not True:
            raise CloudflareWhisperError(
                "Cloudflare returned an unsuccessful transcription response"
            )

        result = body.get("result")
        if not isinstance(result, dict):
            raise CloudflareWhisperError("Cloudflare returned no transcription result")

        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise CloudflareWhisperError("Cloudflare returned an empty transcription")
        return text.strip()

    async def close(self) -> None:
        """Close the shared HTTP client if a transcription created it."""
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
