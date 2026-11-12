"""Kokoro speech synthesis, voice listing, and Telegram delivery tools."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from google.adk.tools import ToolContext

from blacki.telegram.api import TelegramApiClient

logger = logging.getLogger(__name__)

KOKORO_SPEECH_PATH = "/v1/audio/speech"
KOKORO_VOICES_PATH = "/v1/audio/voices"
KOKORO_MODEL = "kokoro"
KOKORO_RESPONSE_FORMAT = "mp3"
KOKORO_AUDIO_MIME_TYPE = "audio/mpeg"
DEFAULT_KOKORO_VOICE = "af_heart"
MAX_TTS_TEXT_CHARS = 4096
# The agent targets 512 MiB-class VPS instances. Keep the response far below
# Telegram's transport ceiling because bytearray-to-bytes conversion and the
# multipart upload can temporarily retain more than one copy.
MAX_TTS_AUDIO_BYTES = 8 * 1024 * 1024
MAX_CONCURRENT_TTS_DELIVERIES = 1
TTS_CONNECT_TIMEOUT_SECONDS = 5.0
TTS_READ_TIMEOUT_SECONDS = 60.0
VOICES_READ_TIMEOUT_SECONDS = 10.0
_VOICE_PATTERN = re.compile(r"[A-Za-z0-9_.+-]{1,128}")


class KokoroTtsResponseError(RuntimeError):
    """Raised when Kokoro returns an unusable audio or voice response."""


@dataclass(frozen=True, slots=True)
class KokoroTtsConfig:
    """Trusted deployment configuration for one Kokoro endpoint."""

    base_url: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url.strip())
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "KOKORO_TTS_BASE_URL must be an http(s) base URL without "
                "credentials, query, or fragment"
            )

    @property
    def speech_url(self) -> str:
        """Return the configured OpenAI-compatible speech endpoint."""
        return f"{self.base_url.strip().rstrip('/')}{KOKORO_SPEECH_PATH}"

    @property
    def voices_url(self) -> str:
        """Return the configured Kokoro voice-list endpoint."""
        return f"{self.base_url.strip().rstrip('/')}{KOKORO_VOICES_PATH}"


def _parse_voices(payload: Any) -> tuple[str, ...]:
    """Extract sorted voice IDs from a Kokoro voice-list response."""
    entries = payload.get("voices") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise KokoroTtsResponseError("Kokoro returned an invalid voice list")

    voices: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            voice: Any = entry
        elif isinstance(entry, dict):
            voice = entry.get("id")
        else:
            voice = None
        if isinstance(voice, str) and _VOICE_PATTERN.fullmatch(voice):
            voices.append(voice)

    if not voices:
        raise KokoroTtsResponseError("Kokoro returned no usable voices")
    return tuple(sorted(set(voices)))


async def _synthesize_mp3(
    text: str,
    config: KokoroTtsConfig,
    client: httpx.AsyncClient,
    *,
    voice: str,
) -> bytes:
    """Synthesize a bounded MP3 response without persisting it."""
    payload: dict[str, Any] = {
        "model": KOKORO_MODEL,
        "input": text,
        "voice": voice,
        "response_format": KOKORO_RESPONSE_FORMAT,
        "stream": False,
    }
    async with client.stream("POST", config.speech_url, json=payload) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        media_type = content_type.partition(";")[0].strip().casefold()
        if media_type != KOKORO_AUDIO_MIME_TYPE:
            raise KokoroTtsResponseError("Kokoro returned a non-MP3 response")

        content_length = response.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise KokoroTtsResponseError(
                    "Kokoro returned an invalid Content-Length"
                ) from exc
            if declared_size > MAX_TTS_AUDIO_BYTES:
                raise KokoroTtsResponseError("Kokoro audio exceeds the size limit")

        audio = bytearray()
        async for chunk in response.aiter_bytes():
            audio.extend(chunk)
            if len(audio) > MAX_TTS_AUDIO_BYTES:
                raise KokoroTtsResponseError("Kokoro audio exceeds the size limit")

    if not audio:
        raise KokoroTtsResponseError("Kokoro returned empty audio")
    return bytes(audio)


def _telegram_target(tool_context: ToolContext) -> tuple[int, int | None]:
    """Resolve a strict Telegram delivery target from ADK session state."""
    chat_id_raw = tool_context.state.get("telegram_chat_id")
    if chat_id_raw is None or not str(chat_id_raw).strip():
        raise ValueError("This tool is available only in a Telegram conversation")
    try:
        chat_id = int(str(chat_id_raw).strip())
    except ValueError as exc:
        raise ValueError("The Telegram chat context is invalid") from exc

    thread_id_raw = tool_context.state.get("telegram_thread_id")
    if thread_id_raw is None or not str(thread_id_raw).strip():
        return chat_id, None
    try:
        return chat_id, int(str(thread_id_raw).strip())
    except ValueError as exc:
        raise ValueError("The Telegram thread context is invalid") from exc


def create_send_text_to_speech_tool(
    config: KokoroTtsConfig,
    *,
    http_transport: httpx.AsyncBaseTransport | None = None,
    telegram_client_factory: Callable[[str], TelegramApiClient] | None = None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Create the private, Telegram-scoped Kokoro speech tool."""
    delivery_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS_DELIVERIES)

    async def send_text_to_speech(
        text: str,
        tool_context: ToolContext,
        voice: str = "",
    ) -> dict[str, Any]:
        """Synthesize text and send the MP3 to the current Telegram chat.

        Use this when the user asks for a spoken or audio response.

        Args:
            text: The exact text to speak, from 1 to 4096 characters.
            voice: Optional Kokoro voice ID. Omit it to use `af_heart`, the
                American English default. Call `list_speech_voices` to
                discover the available voice IDs.

        Returns:
            A small delivery-status dictionary. Audio bytes are never returned.
        """
        normalized_text = text.strip()
        if not normalized_text:
            return {"status": "error", "error": "Speech text must not be empty."}
        if len(normalized_text) > MAX_TTS_TEXT_CHARS:
            return {
                "status": "error",
                "error": (
                    f"Speech text must be at most {MAX_TTS_TEXT_CHARS} characters."
                ),
            }

        resolved_voice = voice.strip() or DEFAULT_KOKORO_VOICE
        if not _VOICE_PATTERN.fullmatch(resolved_voice):
            return {"status": "error", "error": "The requested voice ID is invalid."}

        try:
            chat_id, thread_id = _telegram_target(tool_context)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return {
                "status": "error",
                "error": "Telegram audio delivery is not configured.",
            }

        async with delivery_semaphore:
            timeout = httpx.Timeout(
                TTS_READ_TIMEOUT_SECONDS,
                connect=TTS_CONNECT_TIMEOUT_SECONDS,
            )
            try:
                async with httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=False,
                    transport=http_transport,
                ) as client:
                    audio_bytes = await _synthesize_mp3(
                        normalized_text,
                        config,
                        client,
                        voice=resolved_voice,
                    )
            except Exception as exc:
                logger.warning(
                    "Kokoro TTS synthesis failed (%s)",
                    type(exc).__name__,
                )
                return {"status": "error", "error": "Speech synthesis failed."}

            client_factory = telegram_client_factory or TelegramApiClient
            try:
                async with client_factory(token) as telegram:
                    await telegram.send_audio(
                        chat_id=chat_id,
                        audio_bytes=audio_bytes,
                        filename="blacki-speech.mp3",
                        message_thread_id=thread_id,
                    )
            except Exception as exc:
                logger.warning(
                    "Telegram TTS delivery failed (%s)",
                    type(exc).__name__,
                )
                return {
                    "status": "error",
                    "error": "Speech was generated but Telegram delivery failed.",
                }

            result = {
                "status": "success",
                "message": "Speech audio sent to the current Telegram chat.",
                "format": KOKORO_RESPONSE_FORMAT,
                "voice": resolved_voice,
                "bytes": len(audio_bytes),
            }

        return result

    return send_text_to_speech


def create_list_speech_voices_tool(
    config: KokoroTtsConfig,
    *,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Create the private tool that lists the endpoint's Kokoro voices."""

    async def list_speech_voices() -> dict[str, Any]:
        """List the Kokoro voices available for `send_text_to_speech`.

        Use this before choosing a non-default voice.

        Returns:
            A status dictionary with the default voice and every voice ID.
        """
        timeout = httpx.Timeout(
            VOICES_READ_TIMEOUT_SECONDS,
            connect=TTS_CONNECT_TIMEOUT_SECONDS,
        )
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                transport=http_transport,
            ) as client:
                response = await client.get(config.voices_url)
                response.raise_for_status()
                voices = _parse_voices(response.json())
        except Exception as exc:
            logger.warning(
                "Kokoro voice list request failed (%s)",
                type(exc).__name__,
            )
            return {
                "status": "error",
                "error": "The Kokoro voice list is unavailable.",
            }

        return {
            "status": "success",
            "default_voice": DEFAULT_KOKORO_VOICE,
            "voices": list(voices),
        }

    return list_speech_voices
