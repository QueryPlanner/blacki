"""Tests for the narrow import boundary of blacki.tools."""

import os
import subprocess
import sys
from pathlib import Path


def test_importing_tools_does_not_import_runtime_or_domain_barrels() -> None:
    """The tool package must not eagerly load application runtime modules."""
    source_root = Path(__file__).parents[1] / "src"
    child_environment = os.environ.copy()
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else f"{source_root}{os.pathsep}{existing_pythonpath}"
    )
    script = """
import sys

import blacki.tools

for prefix in (
    "blacki.agent",
    "blacki.server",
    "blacki.telegram",
    "blacki.calories",
    "blacki.gmail",
    "blacki.health",
    "blacki.memory",
    "blacki.reminders",
    "blacki.sandbox",
    "blacki.user_files",
    "blacki.zepto",
):
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in sys.modules
    ), prefix
"""

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        env=child_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
