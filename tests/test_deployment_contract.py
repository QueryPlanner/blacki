"""Contract tests for the documented Docker Compose deployment path."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from blacki.utils.config import ServerEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
BASH_EXECUTABLE = shutil.which("bash")
DOCKER_EXECUTABLE = shutil.which("docker")


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def _load_yaml(relative_path: str) -> dict[str, Any]:
    loaded = yaml.safe_load(_read(relative_path))
    assert isinstance(loaded, dict)
    return loaded


def _parse_env(relative_path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in _read(relative_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key] = value
    return values


def _nav_targets(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [target for item in value for target in _nav_targets(item)]
    if isinstance(value, dict):
        return [target for item in value.values() for target in _nav_targets(item)]
    return []


def test_docker_context_is_an_explicit_allowlist() -> None:
    """Secrets, state, and unrelated files must stay out of Docker builds."""
    rules = {
        line
        for line in _read(".dockerignore").splitlines()
        if line and not line.startswith("#")
    }

    assert rules == {
        "*",
        "!Dockerfile",
        "!entrypoint.sh",
        "!pyproject.toml",
        "!uv.lock",
        "!src/",
        "!src/**",
    }
    assert "!.env" not in rules


def test_minimal_environment_is_safe_and_complete() -> None:
    """The golden-path sample selects one provider and private host defaults."""
    values = _parse_env(".env.minimal")

    assert values["AGENT_NAME"] == "replace-me"
    assert values["ROOT_AGENT_MODEL"].startswith("openrouter/")
    assert values["OPENROUTER_API_KEY"] == "replace-me"
    assert "GOOGLE_API_KEY" not in values
    assert values["TELEGRAM_ENABLED"] == "true"
    assert values["TELEGRAM_BOT_TOKEN"] == "replace-me"  # noqa: S105
    assert "BIND_ADDRESS" not in values
    assert values["SERVE_WEB_INTERFACE"] == "false"
    assert values["RELOAD_AGENTS"] == "false"


def test_full_environment_does_not_activate_fake_optional_credentials() -> None:
    """The full sample must not enable multiple providers or Telegram by accident."""
    values = _parse_env(".env.example")

    assert values["OPENROUTER_API_KEY"] == "replace-me"
    assert "GOOGLE_API_KEY" not in values
    assert values["TELEGRAM_ENABLED"] == "false"
    assert "TELEGRAM_BOT_TOKEN" not in values
    assert values["HOST"] == "127.0.0.1"


def test_compose_defaults_are_private_persistent_and_live() -> None:
    """Compose must encode the supported VPS safety and persistence contract."""
    service = _load_yaml("compose.yaml")["services"]["agent"]

    assert service["image"] == "${IMAGE:-blacki:local}"
    assert "ports" not in service
    assert service["environment"]["AGENT_NAME"].startswith("${AGENT_NAME:?")
    assert service["environment"]["SERVE_WEB_INTERFACE"] == (
        "${SERVE_WEB_INTERFACE:-false}"
    )
    assert service["environment"]["RELOAD_AGENTS"] == "${RELOAD_AGENTS:-false}"
    assert service["env_file"] == ["${ENV_FILE:-.env}"]
    assert service["restart"] == "${RESTART_POLICY:-unless-stopped}"

    assert set(service["volumes"]) == {
        "./.adk_state:/app/src/.adk",
        "./data:/app/data",
        "./logs:/app/logs",
    }

    health_command = " ".join(service["healthcheck"]["test"])
    assert "urllib.request.urlopen" in health_command
    assert "/ready" in health_command


def test_runtime_prepares_every_persistent_mount() -> None:
    """Dockerfile and entrypoint ownership setup must match Compose volumes."""
    dockerfile = _read("Dockerfile")
    entrypoint = _read("entrypoint.sh")

    for container_path in ("/app/src/.adk", "/app/data", "/app/logs"):
        assert container_path in dockerfile
        assert container_path in entrypoint

    assert 'exec runuser -u app -- "$@"' in entrypoint


def test_mkdocs_navigation_targets_existing_files() -> None:
    """Every page in the declared information architecture must exist."""
    config = _load_yaml("mkdocs.yml")
    targets = _nav_targets(config["nav"])

    assert targets
    assert "index.md" in targets
    missing = [
        target for target in targets if not (REPO_ROOT / "docs" / target).is_file()
    ]
    assert missing == []


def test_server_environment_is_covered_by_configuration_reference() -> None:
    """Required runtime settings should not silently disappear from the docs."""
    reference = _read("docs/base-infra/environment-variables.md")
    aliases = {
        field.alias
        for field in ServerEnv.model_fields.values()
        if isinstance(field.alias, str)
    }

    assert aliases
    assert all(f"`{alias}`" in reference for alias in aliases)


def test_readme_exposes_the_complete_first_run() -> None:
    """A developer should reach a running container from the repository front page."""
    readme = _read("README.md")

    for command in (
        "cp .env.minimal .env",
        "docker compose -f compose.yaml -f compose.prod.yaml config --quiet",
        "docker compose -f compose.yaml -f compose.prod.yaml up --build -d",
        "docker compose -f compose.yaml -f compose.prod.yaml ps",
    ):
        assert command in readme

    assert "127.0.0.1" in readme
    assert "docs/DEPLOYMENT.md" in readme


def test_deployment_ci_covers_the_contract_and_native_image_build() -> None:
    """Relevant files must trigger checks that exercise docs, Compose, and Docker."""
    workflow = _read(".github/workflows/developer-experience.yml")

    for path in (
        '".dockerignore"',
        '".env.example"',
        '"Dockerfile"',
        '"compose.yaml"',
        '"compose.dev.yaml"',
        '"compose.prod.yaml"',
        '"docs/**"',
        '"setup.sh"',
    ):
        assert workflow.count(path) == 2

    for command in (
        "mkdocs build --strict",
        "pytest tests/test_deployment_contract.py",
        "-f compose.yaml -f compose.prod.yaml config --quiet",
        "-f compose.yaml -f compose.dev.yaml config --quiet",
        "docker build --tag blacki:contract-test .",
        "validate_observability_environment",
    ):
        assert command in workflow


def test_production_deployment_preflights_before_stopping_service() -> None:
    """The candidate image and environment must pass before downtime begins."""
    workflow = _read(".github/workflows/docker-publish.yml")
    workflow_config = _load_yaml(".github/workflows/docker-publish.yml")
    deploy_concurrency = workflow_config["jobs"]["deploy"]["concurrency"]
    ordered_steps = (
        'git -c advice.detachedHead=false checkout --detach "$DEPLOY_SHA"',
        "write_app_env .env.next",
        'docker pull "$IMAGE_NAME"',
        "ENV_FILE=.env.next docker compose --env-file .env.next",
        "validate_observability_environment",
        "down --remove-orphans",
        "mv .env.next .env",
        "docker compose -f compose.yaml -f compose.prod.yaml up -d",
        "curl -sf http://localhost:${HOST_PORT}/ready",
        "docker image prune -af",
    )
    positions = [workflow.index(step) for step in ordered_steps]

    assert positions == sorted(positions)
    assert deploy_concurrency == {
        "group": "blacki-production",
        "cancel-in-progress": False,
    }
    assert "trap cleanup EXIT" in workflow
    assert 'rm -f "$PROJECT_DIR/.env.next" /tmp/deploy.env' in workflow
    assert "down --remove-orphans 2>/dev/null || true" not in workflow
    assert "git pull" not in workflow
    assert "image-digest: ${{ steps.build-push.outputs.digest }}" in workflow
    assert "DEPLOY_SHA: ${{ github.sha }}" in workflow
    assert "IMAGE_DIGEST: ${{ needs.build.outputs.image-digest }}" in workflow
    assert 'IMAGE_NAME="ghcr.io/queryplanner/blacki@${IMAGE_DIGEST}"' in workflow
    assert '"$IMAGE_NAME" -c' in workflow
    assert "--env OPENROUTER_API_KEY" not in workflow

    for setting in (
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ):
        assert workflow.count(f"{setting}: ${{{{ secrets.{setting} }}}}") == 1
        assert workflow.count(f'{setting}="${{{setting}}}"') == 2


def test_production_deployment_shell_is_valid_bash() -> None:
    """Nested staging and preflight heredocs must remain valid shell syntax."""
    assert BASH_EXECUTABLE is not None
    workflow = _load_yaml(".github/workflows/docker-publish.yml")
    steps = workflow["jobs"]["deploy"]["steps"]
    deploy_step = next(
        step for step in steps if step["name"] == "Deploy to Server via Tailscale"
    )
    deploy_script = deploy_step["run"]
    assert isinstance(deploy_script, str)

    subprocess.run(  # noqa: S603 - executable resolved with shutil.which
        [BASH_EXECUTABLE, "-n"],
        input=deploy_script,
        check=True,
        capture_output=True,
        text=True,
    )


def test_production_overlay_defeats_hostile_environment_overrides() -> None:
    """Rendered production Compose must stay private and disable dev features."""
    assert DOCKER_EXECUTABLE is not None
    environment = {
        **os.environ,
        "BIND_ADDRESS": "0.0.0.0",  # noqa: S104 - hostile input under test
        "ENV_FILE": ".env.minimal",
        "RELOAD_AGENTS": "true",
        "SERVE_WEB_INTERFACE": "true",
    }
    result = subprocess.run(  # noqa: S603 - executable resolved with shutil.which
        [
            DOCKER_EXECUTABLE,
            "compose",
            "--env-file",
            ".env.minimal",
            "-f",
            "compose.yaml",
            "-f",
            "compose.prod.yaml",
            "config",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    service = yaml.safe_load(result.stdout)["services"]["agent"]

    assert service["environment"]["SERVE_WEB_INTERFACE"] == "false"
    assert service["environment"]["RELOAD_AGENTS"] == "false"
    assert service["ports"] == [
        {
            "mode": "ingress",
            "host_ip": "127.0.0.1",
            "target": 8080,
            "published": "8080",
            "protocol": "tcp",
        }
    ]


def test_development_overlay_is_loopback_only() -> None:
    """Local development enables the UI without exposing it on every interface."""
    assert DOCKER_EXECUTABLE is not None
    result = subprocess.run(  # noqa: S603 - executable resolved with shutil.which
        [
            DOCKER_EXECUTABLE,
            "compose",
            "--env-file",
            ".env.minimal",
            "-f",
            "compose.yaml",
            "-f",
            "compose.dev.yaml",
            "config",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "BIND_ADDRESS": "0.0.0.0",  # noqa: S104 - hostile input under test
            "ENV_FILE": ".env.minimal",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    service = yaml.safe_load(result.stdout)["services"]["agent"]

    assert service["environment"]["SERVE_WEB_INTERFACE"] == "true"
    assert service["environment"]["RELOAD_AGENTS"] == "true"
    assert service["ports"][0]["host_ip"] == "127.0.0.1"


def test_owner_deployment_cannot_run_from_a_fork() -> None:
    """The Tailscale deployment is private to the QueryPlanner repository."""
    workflow = _read(".github/workflows/docker-publish.yml")

    assert "github.repository == 'QueryPlanner/blacki'" in workflow


def test_legacy_host_service_and_setup_are_not_presented_as_golden_paths() -> None:
    """Hardcoded host automation must carry an explicit warning."""
    service = _read("systemd/agent.service")
    deployment_guide = _read("docs/DEPLOYMENT.md")

    assert service.startswith("# Legacy example only.")
    assert "Do not run setup.sh unattended" in deployment_guide
    assert "not the supported first-deployment path" in deployment_guide
    assert "docker compose -f compose.yaml -f compose.prod.yaml up -d" in _read(
        "setup.sh"
    )
