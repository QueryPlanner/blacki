# Local development

## Requirements

- Python 3.11, 3.12, or 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git
- optionally, Docker Engine with Docker Compose

## Configure

```bash
git clone https://github.com/QueryPlanner/blacki.git
cd blacki
cp .env.example .env
```

Set a unique `AGENT_NAME`, activate exactly one model provider, and replace its
key. The development Compose overlay enables the ADK web interface and agent
reload while keeping the published port on loopback.

## Install and run with Python

```bash
uv sync
uv run python -m blacki.server
```

The default local bind address is `127.0.0.1:8080`.

## Run with Docker Compose

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build
```

Compose does not currently configure file-watch synchronization. Rebuild the
image after source changes:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build
```

Use `Ctrl+C` in attached mode, or run `docker compose down` from another
terminal.

## Quality checks

Run the same sequence expected by CI:

```bash
uv run ruff format
uv run ruff check
uv run mypy .
uv run pytest --cov=src
```

If any command changes files or you fix a failure, restart the sequence from
`ruff format`.

The coverage threshold is 100% branch coverage. Tests should exercise real
internal classes and mock only external boundaries.

## Add a tool

Put a Blacki-owned callable tool or toolset in `src/blacki/tools/`. Keep the
provider client, OAuth service, storage, and other supporting code in its
existing domain package. Register the tool in `src/blacki/tools/registry.py`
and add it to the appropriate public, private, or worker exposure path.

Do not import `blacki.agent`, `blacki.server`, or `blacki.telegram` from a tool
module. Do not add a tool to a domain package `__init__.py` as a shortcut. Add
tests for its successful result, its provider-missing or failure path, its
exposure boundary, and any confirmation metadata. Run the full quality-check
sequence after changing the registry or an exposure rule.

## Documentation

Install the documentation group and start the live preview:

```bash
uv sync --group docs
uv run mkdocs serve
```

Build exactly as CI does:

```bash
uv run mkdocs build --strict
```

Strict mode validates MkDocs configuration and internal documentation
references that emit warnings. It does not check whether external URLs are
reachable.

## Deployment contract checks

The targeted checks cover Compose defaults, environment samples, build-context
isolation, documentation navigation, and owner-only deployment gating:

```bash
uv run pytest tests/test_deployment_contract.py
ENV_FILE=.env.minimal docker compose --env-file .env.minimal \
  -f compose.yaml -f compose.prod.yaml config --quiet
ENV_FILE=.env.minimal docker compose --env-file .env.minimal \
  -f compose.yaml -f compose.dev.yaml config --quiet
bash -n setup.sh entrypoint.sh
docker build --tag blacki:contract-test .
```

The Docker build is required for deployment-related changes.

## Repository automation

Pull requests run code quality and developer-experience workflows. A merge to
`main` builds the container image. The Tailscale production deployment is
gated to `QueryPlanner/blacki`, so forks do not attempt to use the owner's
infrastructure secrets.

Merging to `main` in the owner repository still triggers that production
deployment. Review its workflow and required secrets before merging.
