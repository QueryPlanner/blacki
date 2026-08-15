# Docker Compose contract

`compose.yaml` defines one service named `agent`. The same file supports a
source build and an explicitly configured prebuilt image.

## Resolve the configuration

Before every first start or configuration change:

```bash
docker compose -f compose.yaml -f compose.prod.yaml config --quiet
```

To validate a different environment file:

```bash
ENV_FILE=.env.production docker compose --env-file .env.production \
  -f compose.yaml -f compose.prod.yaml config --quiet
```

Compose interpolation and container environment injection are separate. Pass
the same file through `--env-file` and `ENV_FILE` when its name is not `.env`.

## Source-build path

The default image name is `blacki:local`, and the service includes `build: .`.

```bash
docker compose -f compose.yaml -f compose.prod.yaml up --build -d
```

Use this path unless a registry image and its access policy have been verified.

## Prebuilt-image path

Set an exact registry reference:

```dotenv
IMAGE=ghcr.io/your-owner/blacki:your-tag
```

Then:

```bash
docker compose -f compose.yaml -f compose.prod.yaml pull
docker compose -f compose.yaml -f compose.prod.yaml up --no-build -d
```

The `--no-build` flag prevents an accidental local rebuild under the registry
tag.

## Network defaults

The mapping is:

```yaml
ports:
  - "${HOST_BIND_IP:-127.0.0.1}:${HOST_PORT:-8080}:8080"
```

The process listens on `0.0.0.0:8080` inside the container, while the VPS
publishes it only on host loopback by default. Telegram long polling needs no
inbound port.

For direct private dashboard access over Tailscale, set the production host
bind in `.env` to the VPS's Tailscale IPv4 address:

```dotenv
HOST_BIND_IP=100.x.y.z
```

Then recreate the production service and open
`http://100.x.y.z:${HOST_PORT:-8080}/dashboard` from an allowed tailnet device.
The dashboard has no application authentication, so configure Tailscale ACLs
for the operator devices and never use `0.0.0.0` for this setting. Tailscale
Serve remains an alternative when HTTPS and a loopback-only Docker bind are
preferred.

The base file publishes no host port. `compose.prod.yaml` adds the loopback
mapping by default and forces the web interface and reload off even if hostile
shell or `.env` values request them.
For local browser development, use `compose.dev.yaml`; it enables the
development features but also forces loopback. Use an SSH tunnel for temporary
remote access. Neither overlay provides authentication for public exposure.

## Persistent mounts

| Host | Container | Data |
| --- | --- | --- |
| `./.adk_state` | `/app/src/.adk` | SQLite and ADK artifacts |
| `./data` | `/app/data` | Optional local memory data |
| `./logs` | `/app/logs` | Application JSON logs and traces |

The entrypoint starts as root only long enough to create and assign these bind
mounts, then executes the server as the non-root `app` user.

## Health probes

The healthcheck uses Python's standard HTTP client to call `/ready` inside the
container. Readiness returns HTTP 503 until SQLite is initialized and whenever
its side-effect-free `SELECT 1` check fails. `/live` checks only that the
application event loop is serving requests. Optional Mem0 is not
readiness-critical.

```bash
docker compose -f compose.yaml -f compose.prod.yaml ps
```

`/health` remains a compatibility alias for `/ready`.

## Lifecycle

```bash
# Attached source build
docker compose -f compose.yaml -f compose.dev.yaml up --build

# Detached source build
docker compose -f compose.yaml -f compose.prod.yaml up --build -d

# Status
docker compose -f compose.yaml -f compose.prod.yaml ps

# Logs
docker compose -f compose.yaml -f compose.prod.yaml logs --follow agent

# Restart
docker compose -f compose.yaml -f compose.prod.yaml restart agent

# Stop and remove the container and network
docker compose -f compose.yaml -f compose.prod.yaml down
```

`docker compose down` does not delete the bind-mounted host directories.

## Source changes

Compose Watch is not configured. Rebuild after source or dependency changes:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build
```

For faster Python iteration, use the
[local uv workflow](../development.md#install-and-run-with-python).
