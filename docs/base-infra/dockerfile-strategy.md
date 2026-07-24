# Docker image

Blacki uses a multi-stage Dockerfile based on `python:3.13-slim`.

## Build context

`.dockerignore` starts with `*` and allowlists only:

- `Dockerfile`;
- `entrypoint.sh`;
- `pyproject.toml`;
- `uv.lock`; and
- `src/`.

This prevents `.env`, `.git`, runtime databases, memory data, logs, caches, and
unrelated working-tree files from entering the Docker build context.

## Builder stage

The builder:

1. installs the pinned uv version;
2. copies `pyproject.toml` and `uv.lock`;
3. runs `uv sync --locked --no-install-project --no-dev`;
4. copies `src/`; and
5. installs Blacki non-editably with development dependencies excluded.

Dependency files are copied before source code so Docker can reuse the expensive
dependency layer when only application code changes. A BuildKit cache mount
reuses uv downloads across builds.

The project metadata references `README.md`, but documentation is intentionally
excluded from the build context. The Dockerfile creates an empty build-only
README before installing the project.

## Runtime stage

The runtime stage starts from a fresh `python:3.13-slim` image and copies the
application plus virtual environment from the builder. Build tools and uv are
not required at runtime.

It creates:

- a non-root user and group named `app` with UID/GID 1000;
- `/app/src/.adk` for SQLite and artifacts;
- `/app/data` for optional local memory; and
- `/app/logs` for JSON logs and traces.

The container initially runs `entrypoint.sh` as root so bind mounts created by
Docker can be assigned to `app`. The script then uses `exec runuser` to replace
itself with the server process as the non-root user.

## Runtime environment

The image sets:

| Variable | Value |
| --- | --- |
| `VIRTUAL_ENV` | `/app/.venv` |
| `AGENT_DIR` | `/app/src` |
| `HOST` | `0.0.0.0` |
| `PORT` | `8080` |
| `PYTHONUNBUFFERED` | `1` |

`AGENT_DIR=/app/src` ensures ADK discovers Blacki's source directory instead of
scanning the virtual environment's site-packages.

## Build and inspect

```bash
docker build --tag blacki:local .
docker image inspect blacki:local
```

The CI deployment contract performs a native Linux build for every relevant
pull request. The main publishing workflow separately builds multi-platform
images after merge.

## Design tradeoff

The root entrypoint is required to fix ownership on host bind mounts. The
long-running server still executes as `app`. Removing the root entrypoint would
require a different volume-permission strategy and is an architectural change,
not a Dockerfile cleanup.
