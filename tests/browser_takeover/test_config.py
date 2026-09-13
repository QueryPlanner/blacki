"""Tests for browser takeover configuration."""

import pytest

from blacki.browser_takeover.config import (
    BrowserTakeoverConfig,
    BrowserTakeoverConfigurationError,
)


def test_config_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BROWSER_TAKEOVER_PUBLIC_URL", raising=False)

    assert BrowserTakeoverConfig.from_environment() is None


def test_valid_https_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "BROWSER_TAKEOVER_PUBLIC_URL",
        "https://blacki.example.ts.net/browser-takeover/",
    )
    monkeypatch.setenv("BROWSER_TAKEOVER_TTL_SECONDS", "420")
    monkeypatch.setenv("BROWSER_TAKEOVER_STREAM_PORT", "9323")

    config = BrowserTakeoverConfig.from_environment()

    assert config is not None
    assert config.public_url == "https://blacki.example.ts.net/browser-takeover"
    assert config.public_origin == "https://blacki.example.ts.net"
    assert config.secure_cookie is True
    assert config.ttl_seconds == 420
    assert config.stream_port == 9323


def test_loopback_http_is_allowed() -> None:
    config = BrowserTakeoverConfig("http://127.0.0.1:8080/browser-takeover")

    config.validate()

    assert config.secure_cookie is False


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://example.test/browser-takeover", "must use HTTPS"),
        ("https:///browser-takeover", "absolute URL"),
        ("https://user:pass@example.test/browser-takeover", "without credentials"),
        ("https://example.test/browser-takeover?x=1", "query or fragment"),
        ("https://example.test/takeover", "must end with /browser-takeover"),
    ],
)
def test_unsafe_urls_are_rejected(url: str, message: str) -> None:
    with pytest.raises(BrowserTakeoverConfigurationError, match=message):
        BrowserTakeoverConfig(url).validate()


@pytest.mark.parametrize(
    "config",
    [
        BrowserTakeoverConfig("https://example.test/browser-takeover", ttl_seconds=59),
        BrowserTakeoverConfig("https://example.test/browser-takeover", ttl_seconds=901),
        BrowserTakeoverConfig("https://example.test/browser-takeover", stream_port=80),
        BrowserTakeoverConfig(
            "https://example.test/browser-takeover", stream_port=70000
        ),
    ],
)
def test_unsafe_limits_are_rejected(config: BrowserTakeoverConfig) -> None:
    with pytest.raises(BrowserTakeoverConfigurationError):
        config.validate()


def test_non_integer_environment_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BROWSER_TAKEOVER_PUBLIC_URL",
        "https://example.test/browser-takeover",
    )
    monkeypatch.setenv("BROWSER_TAKEOVER_TTL_SECONDS", "five")

    with pytest.raises(BrowserTakeoverConfigurationError, match="must be integers"):
        BrowserTakeoverConfig.from_environment()
