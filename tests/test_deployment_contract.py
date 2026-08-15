"""Contract tests for the documented Docker Compose deployment path."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from blacki.utils.config import ServerEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
BASH_EXECUTABLE = shutil.which("bash")
DOCKER_EXECUTABLE = shutil.which("docker")
COMPOSE_ENV_WRITER = REPO_ROOT / "scripts" / "write_compose_env.py"


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
        "!mcp-bridge/",
        "!mcp-bridge/package.json",
        "!mcp-bridge/package-lock.json",
        "!pyproject.toml",
        "!uv.lock",
        "!src/",
        "!src/**",
        "src/.adk/",
        "src/**/.adk/",
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
    assert values["HOST_BIND_IP"] == "127.0.0.1"
    assert values["SERVE_WEB_INTERFACE"] == "false"
    assert values["RELOAD_AGENTS"] == "false"
    assert values["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"
    assert values["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] == "false"


def test_full_environment_does_not_activate_fake_optional_credentials() -> None:
    """The full sample must not enable multiple providers or Telegram by accident."""
    values = _parse_env(".env.example")

    assert values["OPENROUTER_API_KEY"] == "replace-me"
    assert "GOOGLE_API_KEY" not in values
    assert values["TELEGRAM_ENABLED"] == "false"
    assert "TELEGRAM_BOT_TOKEN" not in values
    assert values["HOST_BIND_IP"] == "127.0.0.1"
    assert values["HOST"] == "127.0.0.1"
    assert values["ZEPTO_MCP_ENABLED"] == "false"
    assert values["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "false"
    assert values["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] == "false"


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


def test_zepto_bridge_is_locked_and_baked_into_the_image() -> None:
    """Production must not download or resolve the Zepto bridge at runtime."""
    package = json.loads(_read("mcp-bridge/package.json"))
    lock = json.loads(_read("mcp-bridge/package-lock.json"))
    dockerfile = _read("Dockerfile")

    assert package["dependencies"] == {"mcp-remote": "0.1.38"}
    assert lock["packages"][""]["dependencies"] == {"mcp-remote": "0.1.38"}
    assert lock["packages"]["node_modules/mcp-remote"]["version"] == "0.1.38"
    for required in (
        "FROM node:22-bookworm-slim AS mcp-bridge",
        "npm ci --omit=dev --ignore-scripts",
        "COPY --from=mcp-bridge /usr/local/bin/node /usr/local/bin/node",
        "/usr/local/bin/mcp-remote",
    ):
        assert required in dockerfile
    assert "npx" not in dockerfile


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


def test_documentation_deployment_triggers_on_its_own_workflow() -> None:
    """Pages changes must deploy the workflow that defines them."""
    workflow = _read(".github/workflows/docs-pages.yml")

    assert workflow.count('".github/workflows/docs-pages.yml"') == 1

    for required in (
        "mkdocs build --strict",
        "actions/upload-pages-artifact@v4",
        "actions/deploy-pages@v4",
    ):
        assert required in workflow


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


def test_operations_document_automated_deployment_safety() -> None:
    """Operators must know the smoke-test boundary and rollback behavior."""
    operations = _read("docs/operations.md")

    for required in (
        "compose.smoke.yaml",
        "forces Telegram off",
        "exact image digest",
        "automatically restores",
        "Automatic rollback is unavailable",
    ):
        assert required in operations


def test_compose_env_serializer_preserves_secret_characters(tmp_path: Path) -> None:
    """Transferred secrets must reach a container without interpolation."""
    assert DOCKER_EXECUTABLE is not None
    sample_value = "dollar$VAR${OTHER} quote\" single' backslash\\ space & equals="
    env_file = tmp_path / "deploy.env"
    compose_file = tmp_path / "compose.yaml"
    project_name = f"blacki-env-{os.getpid()}"

    subprocess.run(  # noqa: S603 - current test interpreter is trusted
        [sys.executable, str(COMPOSE_ENV_WRITER), str(env_file), "SECRET"],
        env={**os.environ, "SECRET": sample_value},
        check=True,
        capture_output=True,
        text=True,
    )
    compose_file.write_text(
        "services:\n"
        "  agent:\n"
        "    image: blacki:contract-test\n"
        f"    env_file:\n      - {env_file}\n"
        "    entrypoint: []\n"
        "    command:\n"
        "      - python\n"
        "      - -c\n"
        "      - import os; print(os.environ['SECRET'], end='')\n"
    )
    result = subprocess.run(  # noqa: S603 - executable resolved with shutil.which
        [
            DOCKER_EXECUTABLE,
            "compose",
            "--project-name",
            project_name,
            "--env-file",
            str(env_file),
            "-f",
            str(compose_file),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    service = yaml.safe_load(result.stdout)["services"]["agent"]

    assert env_file.stat().st_mode & 0o777 == 0o600
    # Canonical Compose output escapes literal dollars for safe re-parsing.
    assert service["environment"]["SECRET"] == sample_value.replace("$", "$$")
    assert env_file.read_text() == (
        'SECRET="dollar$$VAR$${OTHER} quote\\" single\' '
        'backslash\\\\ space & equals="\n'
    )

    image = subprocess.run(  # noqa: S603 - executable resolved with shutil.which
        [DOCKER_EXECUTABLE, "image", "inspect", "blacki:contract-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    if image.returncode != 0:
        pytest.skip(
            "runtime round-trip requires the deployment-contract production image"
        )
    try:
        runtime = subprocess.run(  # noqa: S603 - executable resolved with shutil.which
            [
                DOCKER_EXECUTABLE,
                "compose",
                "--project-name",
                project_name,
                "--env-file",
                str(env_file),
                "-f",
                str(compose_file),
                "run",
                "--rm",
                "--no-deps",
                "agent",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        subprocess.run(  # noqa: S603 - executable resolved with shutil.which
            [
                DOCKER_EXECUTABLE,
                "compose",
                "--project-name",
                project_name,
                "--env-file",
                str(env_file),
                "-f",
                str(compose_file),
                "down",
                "--remove-orphans",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert runtime.stdout == sample_value


@pytest.mark.parametrize("unsafe_character", ["\r", "\n"])
def test_compose_env_serializer_rejects_line_injection(
    tmp_path: Path,
    unsafe_character: str,
) -> None:
    """Control characters must not create extra dotenv assignments."""
    env_file = tmp_path / "deploy.env"
    result = subprocess.run(  # noqa: S603 - current test interpreter is trusted
        [sys.executable, str(COMPOSE_ENV_WRITER), str(env_file), "SECRET"],
        env={**os.environ, "SECRET": f"before{unsafe_character}after"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "cannot contain NUL or line breaks" in result.stderr
    assert not env_file.exists()


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
        '"compose.smoke.yaml"',
        '"docs/**"',
        '"mcp-bridge/**"',
        '"scripts/write_compose_env.py"',
        '"setup.sh"',
        '"src/**"',
    ):
        assert workflow.count(path) == 2

    for command in (
        "mkdocs build --strict",
        "pytest tests/test_deployment_contract.py",
        "-f compose.yaml -f compose.prod.yaml config --quiet",
        "-f compose.yaml -f compose.dev.yaml config --quiet",
        "docker build --tag blacki:contract-test .",
        "-f compose.smoke.yaml up -d",
        'docker exec "$CONTAINER_ID" python -c',
        "validate_observability_environment",
    ):
        assert command in workflow

    assert workflow.index("docker build --tag blacki:contract-test .") < (
        workflow.index("pytest tests/test_deployment_contract.py")
    )


def test_smoke_overlay_is_isolated_and_side_effect_free() -> None:
    """Startup smoke tests must not share state or start external integrations."""
    service = _load_yaml("compose.smoke.yaml")["services"]["agent"]

    assert service["image"].startswith("${IMAGE:")
    assert service["env_file"] == ["${ENV_FILE:-.env}"]
    assert service["restart"] == "no"
    assert "volumes" not in service
    assert "ports" not in service
    assert service["environment"] == {
        "HOST": "0.0.0.0",  # noqa: S104 - no host port is published
        "PORT": 8080,
        "RELOAD_AGENTS": "false",
        "SERVE_WEB_INTERFACE": "false",
        "TELEGRAM_ENABLED": "false",
        "TELEGRAM_TOOL_NOTIFICATIONS": "false",
    }


def test_production_deployment_preflights_before_stopping_service() -> None:
    """The candidate image and environment must pass before downtime begins."""
    workflow = _read(".github/workflows/docker-publish.yml")
    workflow_config = _load_yaml(".github/workflows/docker-publish.yml")
    deploy_concurrency = workflow_config["jobs"]["deploy"]["concurrency"]
    deployment = workflow[
        workflow.index(
            'git -c advice.detachedHead=false checkout --detach "$DEPLOY_SHA"'
        ) :
    ]
    ordered_steps = (
        'git -c advice.detachedHead=false checkout --detach "$DEPLOY_SHA"',
        'cp "$DEPLOY_ENV_FILE" .env.next',
        'docker pull "$IMAGE_NAME"',
        "--env-file .env.next -f compose.smoke.yaml up -d",
        '"$SMOKE_CONTAINER_ID" "Candidate"',
        "cleanup_smoke",
        "PROMOTION_STARTED=true",
        "down --remove-orphans",
        "mv .env.next .env",
        "-f compose.yaml -f compose.prod.yaml up -d",
        '"$HOST_PORT" "$PROMOTED_CONTAINER_ID" "Promoted deployment"',
        "DEPLOYMENT_HEALTHY=true",
        "docker image prune -af",
    )
    positions = [deployment.index(step) for step in ordered_steps]

    assert positions == sorted(positions)
    assert deploy_concurrency == {
        "group": "blacki-production",
        "cancel-in-progress": False,
    }
    assert "trap cleanup EXIT" in workflow
    assert "trap cleanup_transfer EXIT" in workflow
    assert 'rm -f "$PROJECT_DIR/.env.next"' in workflow
    assert 'rm -f "$ROLLBACK_ENV"' in workflow
    assert "/tmp/deploy.env" not in workflow
    assert "/tmp/ghcr.token" not in workflow
    assert "umask 077" in workflow
    assert 'mktemp -d "${RUNNER_TEMP}/blacki-deploy.XXXXXX"' in workflow
    assert 'mktemp -d "$HOME/.blacki-deploy/run.XXXXXX"' in workflow
    assert 'chmod 600 "$transfer_dir/deploy.env"' in workflow
    assert "python3 scripts/write_compose_env.py" in workflow
    assert "PREVIOUS_REVISION=$(git rev-parse HEAD)" in workflow
    assert "PREVIOUS_IMAGE=$(" in workflow
    assert 'cp .env "$ROLLBACK_ENV"' in workflow
    assert "rollback_deployment" in workflow
    assert "Stopping failed promoted deployment" in workflow
    assert "CHECKOUT_CHANGED=true" in workflow
    assert 'elif [ "$CHECKOUT_CHANGED" = true ]' in workflow
    assert "Automatic rollback failed" in workflow
    assert "Preserved rollback environment at $ROLLBACK_ENV" in workflow
    assert "down --remove-orphans 2>/dev/null || true" not in workflow
    assert "docker system prune" not in workflow
    assert "--volumes" not in workflow
    assert "git pull" not in workflow
    assert "image-digest: ${{ steps.build-push.outputs.digest }}" in workflow
    assert "DEPLOY_SHA: ${{ github.sha }}" in workflow
    assert "IMAGE_DIGEST: ${{ needs.build.outputs.image-digest }}" in workflow
    assert 'IMAGE_NAME="ghcr.io/queryplanner/blacki@${IMAGE_DIGEST}"' in workflow
    assert 'printf \'IMAGE="%s"\\n\' "$IMAGE_NAME"' in workflow
    assert 'printf \'DEPLOY_SHA="%s"\\n\' "$DEPLOY_SHA"' in workflow
    assert "HOST_BIND_IP=$(" in workflow
    assert "HOST_BIND_IP=127.0.0.1" in workflow
    assert 'printf \'HOST_BIND_IP="%s"\\n\' "$HOST_BIND_IP"' in workflow
    assert 'SMOKE_PROJECT="${PROJECT_NAME}-smoke-${DEPLOY_SHA:0:12}-$$"' in workflow
    assert "source /tmp/deploy.env" not in workflow
    assert '--password-stdin < "$GHCR_TOKEN_FILE"' in workflow
    assert "TELEGRAM_ENABLED=false" not in workflow
    assert "--env OPENROUTER_API_KEY" not in workflow

    for setting in (
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ):
        assert workflow.count(f"{setting}: ${{{{ secrets.{setting} }}}}") == 1
        assert f'{setting}="${{{setting}}}"' not in workflow


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


def test_production_deployment_serializes_zepto_settings() -> None:
    """Zepto feature flags and paths must survive automated deployments."""
    workflow = _load_yaml(".github/workflows/docker-publish.yml")
    deploy_step = next(
        step
        for step in workflow["jobs"]["deploy"]["steps"]
        if step["name"] == "Deploy to Server via Tailscale"
    )
    deploy_script = deploy_step["run"]
    writer_start = deploy_script.index("python3 scripts/write_compose_env.py")
    writer_end = deploy_script.index("printf '%s' \"$GH_TOKEN\"")
    writer_names = deploy_script[writer_start:writer_end].split()

    for setting in (
        "ZEPTO_MCP_ENABLED",
        "ZEPTO_MCP_CONFIG_DIR",
        "ZEPTO_MCP_ALLOWED_TELEGRAM_CHAT_IDS",
    ):
        assert deploy_step["env"][setting] == f"${{{{ secrets.{setting} }}}}"
        assert setting in writer_names


def test_production_deployment_serializes_kokoro_tts_settings() -> None:
    """The private Kokoro endpoint and voice must survive deployment."""
    workflow = _load_yaml(".github/workflows/docker-publish.yml")
    deploy_step = next(
        step
        for step in workflow["jobs"]["deploy"]["steps"]
        if step["name"] == "Deploy to Server via Tailscale"
    )
    deploy_script = deploy_step["run"]
    writer_start = deploy_script.index("python3 scripts/write_compose_env.py")
    writer_end = deploy_script.index("printf '%s' \"$GH_TOKEN\"")
    writer_names = deploy_script[writer_start:writer_end].split()

    for setting in ("KOKORO_TTS_BASE_URL", "KOKORO_TTS_VOICE"):
        assert deploy_step["env"][setting] == f"${{{{ secrets.{setting} }}}}"
        assert setting in writer_names


def test_ci_startup_smoke_shell_is_valid_bash() -> None:
    """The production-image startup check must remain valid executable Bash."""
    assert BASH_EXECUTABLE is not None
    workflow = _load_yaml(".github/workflows/developer-experience.yml")
    steps = workflow["jobs"]["deployment-contract"]["steps"]
    smoke_step = next(
        step for step in steps if step["name"] == "Smoke test production startup"
    )
    smoke_script = smoke_step["run"]
    assert isinstance(smoke_script, str)

    subprocess.run(  # noqa: S603 - executable resolved with shutil.which
        [BASH_EXECUTABLE, "-n"],
        input=smoke_script,
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


def test_production_overlay_supports_a_specific_tailscale_bind() -> None:
    """Operators can publish only the host's private Tailscale interface."""
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
            "compose.prod.yaml",
            "config",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "ENV_FILE": ".env.minimal",
            "HOST_BIND_IP": "100.64.0.42",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    service = yaml.safe_load(result.stdout)["services"]["agent"]

    assert service["ports"][0]["host_ip"] == "100.64.0.42"


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
