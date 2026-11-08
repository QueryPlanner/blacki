"""Configuration for the Telegram Mini App.

The Mini App runs as its own small FastAPI app on its own port, so a tunnel or
reverse proxy for Telegram exposes only ``/miniapp`` and never the dashboard
or the ADK endpoints on the main port.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit


class MiniAppConfigurationError(ValueError):
    """Raised when Mini App settings are unsafe or incomplete."""


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class MiniAppConfig:
    """Validated Mini App settings."""

    host: str = "127.0.0.1"
    port: int = 8090
    public_url: str | None = None
    dev_telegram_user_id: int | None = None
    allow_dev_sign_in: bool = False
    init_data_max_age_seconds: int = 900
    session_ttl_seconds: int = 7200

    @classmethod
    def from_environment(cls) -> MiniAppConfig:
        """Read and validate the Mini App settings. The Mini App is always on."""
        try:
            port = int(os.getenv("MINIAPP_PORT", "8090"))
            max_age = int(os.getenv("MINIAPP_INIT_DATA_MAX_AGE_SECONDS", "900"))
        except ValueError as exc:
            raise MiniAppConfigurationError("MINIAPP_PORT must be an integer") from exc
        public_url = os.getenv("MINIAPP_PUBLIC_URL", "").strip().rstrip("/") or None
        dev_raw = os.getenv("MINIAPP_DEV_TELEGRAM_USER_ID", "").strip()
        dev_user = int(dev_raw) if dev_raw.isdigit() else None
        config = cls(
            host=os.getenv("MINIAPP_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=port,
            public_url=public_url,
            dev_telegram_user_id=dev_user,
            allow_dev_sign_in=_flag("MINIAPP_ALLOW_DEV_SIGN_IN"),
            init_data_max_age_seconds=max_age,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.allow_dev_sign_in:
            if self.dev_telegram_user_id is None:
                raise MiniAppConfigurationError(
                    "MINIAPP_ALLOW_DEV_SIGN_IN needs MINIAPP_DEV_TELEGRAM_USER_ID"
                )
            if self.host not in LOOPBACK_HOSTS:
                raise MiniAppConfigurationError(
                    "MINIAPP_ALLOW_DEV_SIGN_IN needs MINIAPP_HOST to be a loopback "
                    "address (127.0.0.1, ::1 or localhost)"
                )
        if not 1024 <= self.port <= 65535:
            raise MiniAppConfigurationError(
                "MINIAPP_PORT must be between 1024 and 65535"
            )
        if not 60 <= self.init_data_max_age_seconds <= 86400:
            raise MiniAppConfigurationError(
                "MINIAPP_INIT_DATA_MAX_AGE_SECONDS must be between 60 and 86400"
            )
        if self.public_url:
            parsed = urlsplit(self.public_url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise MiniAppConfigurationError(
                    "MINIAPP_PUBLIC_URL must be an https URL"
                )
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise MiniAppConfigurationError(
                    "MINIAPP_PUBLIC_URL can't contain credentials, a query or a fragment"  # noqa: E501
                )

    @property
    def dev_shortcut_enabled(self) -> bool:
        """Local test sign-in: explicit opt-in, loopback bind, no public URL.

        A reverse proxy on the same host can rewrite Host and hide its origin, so
        the request checks alone cannot prove a caller is local. The explicit
        opt-in flag is the real gate; the loopback bind limits the blast radius.
        """
        return (
            self.allow_dev_sign_in
            and self.dev_telegram_user_id is not None
            and self.public_url is None
            and self.host in LOOPBACK_HOSTS
        )

    @property
    def launch_url(self) -> str | None:
        """The URL Telegram buttons open."""
        return f"{self.public_url}/miniapp/" if self.public_url else None
