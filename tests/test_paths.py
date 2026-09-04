"""Tests for stable application path ownership."""

from pathlib import Path

import pytest

from blacki.config.paths import (
    agent_root,
    application_data_directory,
    package_root,
    skills_directory,
)


def test_default_paths_are_independent_of_registry_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default paths resolve from the package root rather than registry.py."""
    monkeypatch.delenv("AGENT_DIR", raising=False)

    expected_package_root = Path(__file__).parents[1] / "src" / "blacki"

    assert package_root() == expected_package_root
    assert skills_directory() == expected_package_root / "skills"
    assert agent_root() == expected_package_root.parent
    assert application_data_directory() == expected_package_root.parent / ".adk"


def test_configured_agent_root_controls_application_data_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENT_DIR continues to control the default SQLite data location."""
    configured_root = Path("/tmp/blacki-configured-agent")
    monkeypatch.setenv("AGENT_DIR", str(configured_root))

    assert agent_root() == configured_root
    assert application_data_directory() == configured_root / ".adk"
    assert skills_directory() == package_root() / "skills"
