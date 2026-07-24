"""Compatibility tests for Google ADK observability instrumentation."""

import subprocess
import sys


def test_openinference_instruments_google_adk_in_isolated_process() -> None:
    """The locked instrumentor must patch and restore the installed ADK."""
    script = """
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

instrumentor = GoogleADKInstrumentor()
instrumentor.instrument()
instrumentor.uninstrument()
"""

    result = subprocess.run(  # noqa: S603 - current test interpreter is trusted
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
