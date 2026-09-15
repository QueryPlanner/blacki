"""Configuration for the private browser takeover endpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit


class BrowserTakeoverConfigurationError(ValueError):
    """Raised when browser takeover configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class BrowserTakeoverConfig:
    """Validated browser takeover configuration."""

    public_url: str
    ttl_seconds: int = 300
    stream_port: int = 9223

    @classmethod
    def from_environment(cls) -> BrowserTakeoverConfig | None:
        """Build configuration when the optional takeover URL is present."""
        public_url = os.getenv("BROWSER_TAKEOVER_PUBLIC_URL", "").strip()
        if not public_url:
            return None

        try:
            ttl_seconds = int(os.getenv("BROWSER_TAKEOVER_TTL_SECONDS", "300"))
            stream_port = int(os.getenv("BROWSER_TAKEOVER_STREAM_PORT", "9223"))
        except ValueError as exc:
            raise BrowserTakeoverConfigurationError(
                "Browser takeover TTL and stream port must be integers"
            ) from exc

        config = cls(
            public_url=public_url.rstrip("/"),
            ttl_seconds=ttl_seconds,
            stream_port=stream_port,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject public URLs and limits that could expose a browser session."""
        parsed = urlsplit(self.public_url)
        is_loopback_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "localhost",
        }
        if parsed.scheme != "https" and not is_loopback_http:
            raise BrowserTakeoverConfigurationError(
                "BROWSER_TAKEOVER_PUBLIC_URL must use HTTPS"
            )
        if not parsed.netloc or parsed.username or parsed.password:
            raise BrowserTakeoverConfigurationError(
                "BROWSER_TAKEOVER_PUBLIC_URL must be an absolute URL "
                "without credentials"
            )
        if parsed.query or parsed.fragment:
            raise BrowserTakeoverConfigurationError(
                "BROWSER_TAKEOVER_PUBLIC_URL cannot contain a query or fragment"
            )
        if parsed.path != "/browser-takeover":
            raise BrowserTakeoverConfigurationError(
                "BROWSER_TAKEOVER_PUBLIC_URL must end with /browser-takeover"
            )
        if not 60 <= self.ttl_seconds <= 900:
            raise BrowserTakeoverConfigurationError(
                "BROWSER_TAKEOVER_TTL_SECONDS must be between 60 and 900"
            )
        if not 1024 <= self.stream_port <= 65535:
            raise BrowserTakeoverConfigurationError(
                "BROWSER_TAKEOVER_STREAM_PORT must be between 1024 and 65535"
            )

    @property
    def public_origin(self) -> str:
        """Return the exact origin allowed to redeem takeover tokens."""
        parsed = urlsplit(self.public_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @property
    def secure_cookie(self) -> bool:
        """Require Secure cookies outside loopback development."""
        return urlsplit(self.public_url).scheme == "https"
