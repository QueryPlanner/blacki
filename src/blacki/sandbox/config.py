"""Configuration for OpenSandbox integration."""

from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Final

from pydantic import BaseModel, Field, field_validator

DEFAULT_SANDBOX_IMAGE: Final[str] = "ghcr.io/queryplanner/blacki-sandbox:latest"


class SandboxConfig(BaseModel):
    """Configuration for OpenSandbox sandbox operations.

    All settings can be overridden via environment variables.

    Attributes:
        enabled: Whether sandbox tools are enabled.
        domain: OpenSandbox server domain (e.g., localhost:9090).
        api_key: API key for authentication (optional for local dev).
        timeout_minutes: Sandbox TTL in minutes.
        memory_limit: Memory limit per sandbox (e.g., 1Gi).
        cpu_limit: CPU limit per sandbox (e.g., 0.5).
        image: Docker image to use for sandboxes.
    """

    enabled: bool = Field(default=False, description="Enable sandbox tools")
    domain: str = Field(
        default="localhost:9090", description="OpenSandbox server domain"
    )
    api_key: str | None = Field(default=None, description="API key for authentication")
    timeout_minutes: int = Field(
        default=30, description="Sandbox TTL in minutes", ge=1, le=1440
    )
    memory_limit: str = Field(default="1Gi", description="Memory limit per sandbox")
    cpu_limit: str = Field(default="0.5", description="CPU limit per sandbox")
    image: str = Field(
        default=DEFAULT_SANDBOX_IMAGE,
        description="Docker image for sandboxes",
    )
    entrypoint: list[str] = Field(
        default=[
            "/bin/sh",
            "-c",
            "if command -v blacki-sandbox-start >/dev/null 2>&1; then "
            "exec blacki-sandbox-start; "
            "elif command -v athena-sandbox-start >/dev/null 2>&1; then "
            "exec athena-sandbox-start; "
            "elif [ -x /opt/opensandbox/code-interpreter.sh ]; then "
            "exec /opt/opensandbox/code-interpreter.sh; "
            "else exec tail -f /dev/null; fi",
        ],
        description="Entrypoint command for the sandbox image",
    )

    @field_validator("memory_limit")
    @classmethod
    def validate_memory_limit(cls, v: str) -> str:
        if not re.match(r"^\d+(Mi|Gi|M|G)$", v):
            raise ValueError("memory_limit must be like 1Gi, 512Mi, 256M, or 2G")
        return v

    @field_validator("cpu_limit")
    @classmethod
    def validate_cpu_limit(cls, v: str) -> str:
        try:
            float(v)
        except ValueError as e:
            raise ValueError("cpu_limit must be a number (e.g., 0.5 or 1)") from e
        return v

    @property
    def timeout(self) -> timedelta:
        """Get timeout as timedelta."""
        return timedelta(minutes=self.timeout_minutes)

    @property
    def resource(self) -> dict[str, str]:
        """Get resource limits as dict for Sandbox.create()."""
        return {"cpu": self.cpu_limit, "memory": self.memory_limit}


def load_sandbox_config() -> SandboxConfig:
    """Load sandbox configuration from environment variables.

    Environment variables:
        SANDBOX_ENABLED: Enable sandbox tools (default: false)
        SANDBOX_DOMAIN: OpenSandbox server domain (default: localhost:9090)
        SANDBOX_API_KEY: API key for authentication
        SANDBOX_TIMEOUT_MINUTES: Sandbox TTL in minutes (default: 30)
        SANDBOX_MEMORY_LIMIT: Memory limit per sandbox (default: 1Gi)
        SANDBOX_CPU_LIMIT: CPU limit per sandbox (default: 0.5)
        SANDBOX_IMAGE: Docker image for sandboxes (default: the Blacki
            Chromium/agent-browser image)

    Returns:
        SandboxConfig instance with values from environment.
    """
    enabled_str = os.getenv("SANDBOX_ENABLED", "false").strip().lower()
    enabled = enabled_str in ("true", "1", "yes")

    return SandboxConfig(
        enabled=enabled,
        domain=os.getenv("SANDBOX_DOMAIN", "localhost:9090").strip(),
        api_key=os.getenv("SANDBOX_API_KEY", "").strip() or None,
        timeout_minutes=int(os.getenv("SANDBOX_TIMEOUT_MINUTES", "30").strip()),
        memory_limit=os.getenv("SANDBOX_MEMORY_LIMIT", "1Gi").strip(),
        cpu_limit=os.getenv("SANDBOX_CPU_LIMIT", "0.5").strip(),
        image=os.getenv("SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE).strip(),
    )


SANDBOX_STATE_KEY: Final[str] = "__sandbox_id__"


def sandbox_enabled() -> bool:
    """Return whether sandbox tools are enabled."""
    return load_sandbox_config().enabled
