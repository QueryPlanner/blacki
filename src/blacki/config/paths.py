"""Stable paths shared by application composition and runtime modules."""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """Return the installed ``blacki`` package directory."""
    return Path(__file__).resolve().parent.parent


def skills_directory() -> Path:
    """Return the bundled skills directory."""
    return package_root() / "skills"


def agent_root() -> Path:
    """Return the configured agent root, preserving relative paths."""
    return Path(os.getenv("AGENT_DIR", str(package_root().parent)))


def application_data_directory() -> Path:
    """Return the application data directory under the agent root."""
    return agent_root() / ".adk"
