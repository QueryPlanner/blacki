"""Kokoro speech synthesis and Telegram delivery tool."""

from __future__ import annotations

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
KOKORO_MODEL = "kokoro"
KOKORO_RESPONSE_FORMAT = "mp3"
KOKORO_AUDIO_MIME_TYPE = "audio/mpeg"
MAX_TTS_TEXT_CHARS = 4096
# Telegram currently documents a 50 MB sendAudio ceiling. Keep headroom for
# multipart framing and fail before attempting an oversized upload.
MAX_TTS_AUDIO_BYTES = 49 * 1024 * 1024
TTS_CONNECT_TIMEOUT_SECONDS = 5.0
TTS_READ_TIMEOUT_SECONDS = 60.0
_VOICE_PATTERN = re.compile(r"[A-Za-z0-9_.+-]{1,128}")


class KokoroTtsResponseError(RuntimeError):
    """Raised when Kokoro returns an unusable audio response."""


@dataclass(frozen=True, slots=True)
class KokoroTtsConfig:
    """Trusted deployment configuration for one Kokoro endpoint."""

    base_url: str
    voice: str = "af_heart"

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
        if not _VOICE_PATTERN.fullmatch(self.voice.strip()):
            raise ValueError("KOKORO_TTS_VOICE is invalid")

    @property
    def speech_url(self) -> str:
        """Return the configured OpenAI-compatible speech endpoint."""
        return f"{self.base_url.strip().rstrip('/')}{KOKORO_SPEECH_PATH}"

    @property
    def normalized_voice(self) -> str:
        """Return the whitespace-normalized voice ID."""
        return self.voice.strip()


async def _synthesize_mp3(
    text: str,
    config: KokoroTtsConfig,
    client: httpx.AsyncClient,
) -> bytes:
    """Synthesize a bounded MP3 response without persisting it."""
    payload: dict[str, Any] = {
        "model": KOKORO_MODEL,
        "input": text,
        "voice": config.normalized_voice,
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
) -> Callable[[str, ToolContext], Awaitable[dict[str, Any]]]:
    """Create the private, Telegram-scoped Kokoro speech tool."""

    async def send_text_to_speech(
        text: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Synthesize text and send the MP3 to the current Telegram chat.

        Use this when the user asks for a spoken or audio response.

        Args:
            text: The exact text to speak, from 1 to 4096 characters.

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

        return {
            "status": "success",
            "message": "Speech audio sent to the current Telegram chat.",
            "format": KOKORO_RESPONSE_FORMAT,
            "voice": config.normalized_voice,
            "bytes": len(audio_bytes),
        }

    return send_text_to_speech
