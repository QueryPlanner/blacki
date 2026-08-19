"""Configuration for durable Telegram attachments in Cloudflare R2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

_ENABLED_VALUES = frozenset({"1", "true", "yes"})


def user_files_enabled() -> bool:
    """Return whether durable user files are explicitly enabled."""
    return os.getenv("R2_FILES_ENABLED", "false").strip().lower() in _ENABLED_VALUES


@dataclass(frozen=True, slots=True)
class R2FileConfig:
    """Trusted configuration for a private R2 bucket."""

    enabled: bool = False
    endpoint_url: str = ""
    bucket_name: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    owner_hmac_secret: str = ""
    key_prefix: str = "blacki/user-files"
    retention_days: int | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        parsed = urlsplit(self.endpoint_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("R2_ENDPOINT_URL must be a credential-free HTTPS URL")
        required = {
            "R2_BUCKET_NAME": self.bucket_name,
            "R2_ACCESS_KEY_ID": self.access_key_id,
            "R2_SECRET_ACCESS_KEY": self.secret_access_key,
            "R2_OWNER_HMAC_SECRET": self.owner_hmac_secret,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"Missing R2 file configuration: {', '.join(missing)}")
        if self.retention_days is not None and not 1 <= self.retention_days <= 3650:
            raise ValueError("R2_FILE_RETENTION_DAYS must be between 1 and 3650")
        if not self.key_prefix.strip("/"):
            raise ValueError("R2_FILE_KEY_PREFIX cannot be empty")

    @property
    def normalized_prefix(self) -> str:
        """Return a slash-normalized object prefix."""
        return self.key_prefix.strip("/")


def load_r2_file_config() -> R2FileConfig:
    """Load R2 attachment configuration from environment variables."""
    raw_retention_days = os.getenv("R2_FILE_RETENTION_DAYS", "").strip()
    return R2FileConfig(
        enabled=user_files_enabled(),
        endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip(),
        bucket_name=os.getenv("R2_BUCKET_NAME", "").strip(),
        access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
        owner_hmac_secret=os.getenv("R2_OWNER_HMAC_SECRET", "").strip(),
        key_prefix=os.getenv("R2_FILE_KEY_PREFIX", "blacki/user-files").strip(),
        retention_days=int(raw_retention_days) if raw_retention_days else None,
    )
