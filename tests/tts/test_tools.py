"""Tests for the private Kokoro speech-delivery tool."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import cast
from unittest.mock import AsyncMock, MagicMock, create_autospec

import httpx
import pytest
from conftest import MockState, MockToolContext
from google.adk.tools import ToolContext

import blacki.tts.tools as tts_module
from blacki.telegram.api import TelegramApiClient
from blacki.tts.tools import (
    KOKORO_AUDIO_MIME_TYPE,
    MAX_TTS_AUDIO_BYTES,
    MAX_TTS_TEXT_CHARS,
    KokoroTtsConfig,
    KokoroTtsResponseError,
    _synthesize_mp3,
    create_send_text_to_speech_tool,
)


def _audio_response(content: bytes = b"ID3-audio") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "audio/mpeg; charset=binary"},
        content=content,
    )


def _telegram_factory() -> tuple[
    Callable[[str], TelegramApiClient],
    MagicMock,
    list[str],
]:
    client: MagicMock = create_autospec(TelegramApiClient, spec_set=True, instance=True)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.send_audio = AsyncMock()
    tokens: list[str] = []

    def factory(token: str) -> TelegramApiClient:
        tokens.append(token)
        return cast(TelegramApiClient, client)

    return factory, client, tokens


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://kokoro.internal",
        "http:///missing-host",
        "http://user@kokoro.internal",
        "http://user:pass@kokoro.internal",
        "http://kokoro.internal?debug=true",
        "http://kokoro.internal#fragment",
    ],
)
def test_config_rejects_untrusted_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="KOKORO_TTS_BASE_URL"):
        KokoroTtsConfig(base_url=base_url)


def test_config_normalizes_endpoint_and_voice() -> None:
    config = KokoroTtsConfig(
        base_url=" http://100.77.130.71:8880/ ",
        voice=" af_heart ",
    )

    assert config.speech_url == "http://100.77.130.71:8880/v1/audio/speech"
    assert config.normalized_voice == "af_heart"


def test_config_rejects_invalid_voice() -> None:
    with pytest.raises(ValueError, match="KOKORO_TTS_VOICE"):
        KokoroTtsConfig(base_url="http://kokoro.internal", voice="bad voice")


@pytest.mark.asyncio
async def test_synthesize_mp3_uses_verified_openai_contract() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _audio_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        audio = await _synthesize_mp3(
            "Speak this",
            KokoroTtsConfig("http://kokoro.internal", voice="af_heart"),
            client,
        )

    assert audio == b"ID3-audio"
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == "http://kokoro.internal/v1/audio/speech"
    assert json.loads(seen[0].content) == {
        "model": "kokoro",
        "input": "Speak this",
        "voice": "af_heart",
        "response_format": "mp3",
        "stream": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            httpx.Response(200, headers={"content-type": "application/json"}),
            "non-MP3",
        ),
        (_audio_response(b""), "empty audio"),
        (
            httpx.Response(
                200,
                headers={
                    "content-type": KOKORO_AUDIO_MIME_TYPE,
                    "content-length": "invalid",
                },
                content=b"audio",
            ),
            "invalid Content-Length",
        ),
        (
            httpx.Response(
                200,
                headers={
                    "content-type": KOKORO_AUDIO_MIME_TYPE,
                    "content-length": str(MAX_TTS_AUDIO_BYTES + 1),
                },
                content=b"audio",
            ),
            "size limit",
        ),
    ],
)
async def test_synthesize_mp3_rejects_invalid_responses(
    response: httpx.Response,
    error: str,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response)
    ) as client:
        with pytest.raises(KokoroTtsResponseError, match=error):
            await _synthesize_mp3(
                "hello",
                KokoroTtsConfig("http://kokoro.internal"),
                client,
            )


@pytest.mark.asyncio
async def test_synthesize_mp3_rejects_oversized_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"12345"

    monkeypatch.setattr(tts_module, "MAX_TTS_AUDIO_BYTES", 4)
    response = httpx.Response(
        200,
        headers={"content-type": KOKORO_AUDIO_MIME_TYPE},
        stream=OversizedStream(),
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response)
    ) as client:
        with pytest.raises(KokoroTtsResponseError, match="size limit"):
            await _synthesize_mp3(
                "hello",
                KokoroTtsConfig("http://kokoro.internal"),
                client,
            )


@pytest.mark.asyncio
async def test_tool_synthesizes_and_sends_to_current_telegram_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-secret")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _audio_response()

    factory, telegram, tokens = _telegram_factory()
    tool = create_send_text_to_speech_tool(
        KokoroTtsConfig("http://kokoro.internal", voice="af_heart"),
        http_transport=httpx.MockTransport(handler),
        telegram_client_factory=factory,
    )
    context = MockToolContext(
        state=MockState({"telegram_chat_id": "-12345", "telegram_thread_id": " 77 "})
    )

    result = await tool("  Hello from Blacki  ", cast(ToolContext, context))

    assert result == {
        "status": "success",
        "message": "Speech audio sent to the current Telegram chat.",
        "format": "mp3",
        "voice": "af_heart",
        "bytes": len(b"ID3-audio"),
    }
    assert tokens == ["telegram-secret"]
    telegram.send_audio.assert_awaited_once_with(
        chat_id=-12345,
        audio_bytes=b"ID3-audio",
        filename="blacki-speech.mp3",
        message_thread_id=77,
    )
    assert json.loads(requests[0].content)["input"] == "Hello from Blacki"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "state", "expected_error"),
    [
        ("   ", {"telegram_chat_id": "1"}, "must not be empty"),
        (
            "x" * (MAX_TTS_TEXT_CHARS + 1),
            {"telegram_chat_id": "1"},
            "at most 4096",
        ),
        ("hello", {}, "only in a Telegram conversation"),
        ("hello", {"telegram_chat_id": "bad"}, "chat context is invalid"),
        (
            "hello",
            {"telegram_chat_id": "1", "telegram_thread_id": "bad"},
            "thread context is invalid",
        ),
    ],
)
async def test_tool_rejects_invalid_input_or_context_without_network(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    state: dict[str, str],
    expected_error: str,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _audio_response()

    tool = create_send_text_to_speech_tool(
        KokoroTtsConfig("http://kokoro.internal"),
        http_transport=httpx.MockTransport(handler),
    )

    result = await tool(
        text,
        cast(ToolContext, MockToolContext(state=MockState(state))),
    )

    assert result["status"] == "error"
    assert expected_error in result["error"]
    assert requests == []


@pytest.mark.asyncio
async def test_tool_rejects_missing_telegram_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    tool = create_send_text_to_speech_tool(KokoroTtsConfig("http://kokoro.internal"))

    result = await tool(
        "hello",
        cast(
            ToolContext,
            MockToolContext(state=MockState({"telegram_chat_id": "1"})),
        ),
    )

    assert result == {
        "status": "error",
        "error": "Telegram audio delivery is not configured.",
    }


@pytest.mark.asyncio
async def test_tool_sanitizes_synthesis_failure_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    private_text = "private speech contents"
    tool = create_send_text_to_speech_tool(
        KokoroTtsConfig("http://kokoro.internal"),
        http_transport=httpx.MockTransport(lambda _: httpx.Response(503)),
    )

    result = await tool(
        private_text,
        cast(
            ToolContext,
            MockToolContext(state=MockState({"telegram_chat_id": "1"})),
        ),
    )

    assert result == {"status": "error", "error": "Speech synthesis failed."}
    assert "HTTPStatusError" in caplog.text
    assert private_text not in caplog.text
    assert "secret-token" not in caplog.text
    assert "kokoro.internal" not in caplog.text


@pytest.mark.asyncio
async def test_tool_sanitizes_telegram_delivery_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token")
    factory, telegram, _ = _telegram_factory()
    telegram.send_audio.side_effect = RuntimeError("provider body with private text")
    tool = create_send_text_to_speech_tool(
        KokoroTtsConfig("http://kokoro.internal"),
        http_transport=httpx.MockTransport(lambda _: _audio_response()),
        telegram_client_factory=factory,
    )

    result = await tool(
        "private text",
        cast(
            ToolContext,
            MockToolContext(state=MockState({"telegram_chat_id": "1"})),
        ),
    )

    assert result == {
        "status": "error",
        "error": "Speech was generated but Telegram delivery failed.",
    }
    assert "RuntimeError" in caplog.text
    assert "private text" not in caplog.text
    assert "secret-token" not in caplog.text
    assert "provider body" not in caplog.text
