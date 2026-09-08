"""Tests for PR3 package ownership and stable application entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"

CANONICAL_MODULES = (
    "blacki.models.factory",
    "blacki.models.inference",
    "blacki.models.capabilities",
    "blacki.prompts.instructions",
    "blacki.prompts.policies",
    "blacki.runtime.adk",
)
REMOVED_ROOT_MODULES = (
    "adk_runtime.py",
    "inference.py",
    "model_capabilities.py",
    "prompt.py",
)


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SOURCE_ROOT)
        if not existing_pythonpath
        else f"{SOURCE_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    environment.update(
        {
            "AGENT_NAME": "layout-test-agent",
            "OPENROUTER_API_KEY": "layout-test-key",
            "ROOT_AGENT_MODEL": "openrouter/openai/gpt-5.6-luna",
            "TELEGRAM_ENABLED": "false",
            "SERVE_WEB_INTERFACE": "false",
            "RELOAD_AGENTS": "false",
        }
    )
    return environment


def test_application_modules_have_one_canonical_package_owner() -> None:
    """Reject reintroduced root modules after the package move."""
    assert all(
        not (SOURCE_ROOT / "blacki" / name).exists() for name in REMOVED_ROOT_MODULES
    )

    expected_paths = (
        "models/factory.py",
        "models/inference.py",
        "models/capabilities.py",
        "prompts/instructions.py",
        "prompts/policies.py",
        "runtime/adk.py",
    )
    assert all((SOURCE_ROOT / "blacki" / path).is_file() for path in expected_paths)


def test_canonical_packages_do_not_import_application_entrypoints() -> None:
    """Models, prompts, and runtime helpers must remain import-direction safe."""
    script = f"""
import importlib
import sys

for module_name in {CANONICAL_MODULES!r}:
    importlib.import_module(module_name)

assert "blacki.agent" not in sys.modules
assert "blacki.server" not in sys.modules
"""
    result = subprocess.run(  # noqa: S603 - current test interpreter is trusted
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_adk_discovery_and_executable_entrypoints_remain_stable() -> None:
    """Keep ADK discovery and documented process entrypoints unchanged."""
    script = f"""
import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False

from google.adk.apps import App
from google.adk.cli.utils.agent_loader import AgentLoader

import blacki
import blacki.agent as agent_module
from blacki.server import main

assert blacki.app is agent_module.app
assert agent_module.root_agent is agent_module.app.root_agent
assert callable(main)
loaded = AgentLoader({str(SOURCE_ROOT)!r}).load_agent("blacki")
assert isinstance(loaded, App)
assert loaded.name == "blacki"
assert loaded.root_agent is not None
"""
    result = subprocess.run(  # noqa: S603 - current test interpreter is trusted
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    compose = (REPO_ROOT / "compose.yaml").read_text()
    systemd = (REPO_ROOT / "systemd/agent.service").read_text()
    assert 'server = "blacki.server:main"' in pyproject
    assert 'CMD ["python", "-m", "blacki.server"]' in dockerfile
    assert "command: python -m blacki.server" in compose
    assert "ExecStart=/home/ubuntu/blacki/.venv/bin/python -m blacki.server" in systemd
