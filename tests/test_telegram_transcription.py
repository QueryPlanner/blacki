"""Tests for the Cloudflare Workers AI Telegram transcription client."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from blacki.telegram.transcription import (
    CLOUDFLARE_WHISPER_MODEL,
    MAX_CLOUDFLARE_WHISPER_AUDIO_BYTES,
    CloudflareWhisperConfig,
    CloudflareWhisperError,
    CloudflareWhisperTranscriber,
)


def _config() -> CloudflareWhisperConfig:
    return CloudflareWhisperConfig(
        account_id="account-id",
        api_token="cloudflare-token",
    )


@pytest.mark.asyncio
async def test_transcriber_uses_cloudflare_rest_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"success": True, "result": {"text": "  Hello from audio  "}},
        )

    transcriber = CloudflareWhisperTranscriber(
        _config(),
        http_transport=httpx.MockTransport(handler),
    )

    assert await transcriber.transcribe(b"\x00\x01\xff") == "Hello from audio"
    assert await transcriber.transcribe(b"\x02") == "Hello from audio"
    await transcriber.close()

    assert len(requests) == 2
    assert requests[0].method == "POST"
    assert str(requests[0].url) == (
        "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run/"
        f"{CLOUDFLARE_WHISPER_MODEL}"
    )
    assert requests[0].headers["authorization"] == "Bearer cloudflare-token"
    assert json.loads(requests[0].content) == {
        "audio": base64.b64encode(b"\x00\x01\xff").decode("ascii"),
        "task": "transcribe",
    }


def test_from_environment_requires_both_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    assert CloudflareWhisperTranscriber.from_environment() is None

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-id")
    assert CloudflareWhisperTranscriber.from_environment() is None

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", " cloudflare-token ")
    transcriber = CloudflareWhisperTranscriber.from_environment()
    assert transcriber is not None
    assert transcriber.config.account_id == "account-id"
    assert transcriber.config.api_token == "cloudflare-token"  # noqa: S105


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "error"),
    [
        (
            {"success": False, "errors": [{"message": "provider detail"}]},
            "unsuccessful",
        ),
        ({"success": True}, "no transcription result"),
        ({"success": True, "result": {"text": "   "}}, "empty transcription"),
    ],
)
async def test_transcriber_rejects_invalid_cloudflare_results(
    body: dict[str, object],
    error: str,
) -> None:
    transcriber = CloudflareWhisperTranscriber(
        _config(),
        http_transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    )

    with pytest.raises(CloudflareWhisperError, match=error):
        await transcriber.transcribe(b"audio")
    await transcriber.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, content=b"not-json"),
    ],
)
async def test_transcriber_sanitizes_http_and_json_failures(
    response: httpx.Response,
) -> None:
    transcriber = CloudflareWhisperTranscriber(
        _config(),
        http_transport=httpx.MockTransport(lambda _: response),
    )

    with pytest.raises(CloudflareWhisperError, match="request failed"):
        await transcriber.transcribe(b"private audio")
    await transcriber.close()


@pytest.mark.asyncio
async def test_transcriber_sanitizes_network_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private provider detail", request=request)

    transcriber = CloudflareWhisperTranscriber(
        _config(),
        http_transport=httpx.MockTransport(handler),
    )

    with pytest.raises(CloudflareWhisperError, match="request failed"):
        await transcriber.transcribe(b"audio")
    await transcriber.close()


@pytest.mark.asyncio
async def test_transcriber_rejects_empty_and_oversized_audio_without_network() -> None:
    transcriber = CloudflareWhisperTranscriber(_config())

    with pytest.raises(CloudflareWhisperError, match="audio is empty"):
        await transcriber.transcribe(b"")
    with pytest.raises(CloudflareWhisperError, match="audio is too large"):
        await transcriber.transcribe(b"x" * (MAX_CLOUDFLARE_WHISPER_AUDIO_BYTES + 1))

    await transcriber.close()
