# Docker Compose contract

`compose.yaml` defines one service named `agent`. The same file supports a
source build and an explicitly configured prebuilt image.

## Resolve the configuration

Before every first start or configuration change:

```bash
docker compose config --quiet
```

To validate a different environment file:

```bash
ENV_FILE=.env.production docker compose --env-file .env.production config --quiet
```

Compose interpolation and container environment injection are separate. Pass
the same file through `--env-file` and `ENV_FILE` when its name is not `.env`.

## Source-build path

The default image name is `blacki:local`, and the service includes `build: .`.

```bash
docker compose up --build -d
```

Use this path unless a registry image and its access policy have been verified.

## Prebuilt-image path

Set an exact registry reference:

```dotenv
IMAGE=ghcr.io/your-owner/blacki:your-tag
```

Then:

```bash
docker compose pull
docker compose up --no-build -d
```

The `--no-build` flag prevents an accidental local rebuild under the registry
tag.

## Network defaults

The mapping is:

```yaml
ports:
  - "${BIND_ADDRESS:-127.0.0.1}:${HOST_PORT:-8080}:8080"
```

The process listens on `0.0.0.0:8080` inside the container, while the VPS
publishes it only on loopback by default. Telegram long polling needs no inbound
port.

For temporary browser access, enable the web interface and use an SSH tunnel.
Treat `BIND_ADDRESS=0.0.0.0` as an explicit public-network decision.

## Persistent mounts

| Host | Container | Data |
| --- | --- | --- |
| `./.adk_state` | `/app/src/.adk` | SQLite and ADK artifacts |
| `./data` | `/app/data` | Optional local memory data |
| `./logs` | `/app/logs` | Application JSON logs and traces |

The entrypoint starts as root only long enough to create and assign these bind
mounts, then executes the server as the non-root `app` user.

## Liveness

The healthcheck uses Python's standard `socket` module to connect to
`127.0.0.1:8080` inside the container. It proves the server is accepting TCP
connections without calling the richer `/health` endpoint or initializing
optional Mem0 configuration.

```bash
docker compose ps
```

Use `/health` manually when you need database and memory details.

## Lifecycle

```bash
# Attached source build
docker compose up --build

# Detached source build
docker compose up --build -d

# Status
docker compose ps

# Logs
docker compose logs --follow agent

# Restart
docker compose restart agent

# Stop and remove the container and network
docker compose down
```

`docker compose down` does not delete the bind-mounted host directories.

## Source changes

Compose Watch is not configured. Rebuild after source or dependency changes:

```bash
docker compose up --build
```

For faster Python iteration, use the
[local uv workflow](../development.md#install-and-run-with-python).
