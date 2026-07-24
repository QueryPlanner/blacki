# Troubleshooting

Start with:

```bash
docker compose -f compose.yaml -f compose.prod.yaml ps
docker compose -f compose.yaml -f compose.prod.yaml logs --tail=200 agent
docker compose -f compose.yaml -f compose.prod.yaml config --quiet
```

Do not post `.env` or unredacted logs publicly.

## Compose says AGENT_NAME is required

`AGENT_NAME` is the only required `ServerEnv` field and Compose validates it
before startup. Confirm `.env` exists in the repository root and contains a
non-empty value:

```dotenv
AGENT_NAME=my-blacki
```

If you use a different file, pass it explicitly:

```bash
ENV_FILE=.env.production docker compose --env-file .env.production \
  -f compose.yaml -f compose.prod.yaml config --quiet
ENV_FILE=.env.production docker compose --env-file .env.production \
  -f compose.yaml -f compose.prod.yaml up -d
```

## Startup says no model API key is configured

Activate one provider:

=== "OpenRouter"

    ```dotenv
    ROOT_AGENT_MODEL=openrouter/google/gemini-2.5-flash
    OPENROUTER_API_KEY=replace-me
    ```

=== "Google AI Studio"

    ```dotenv
    ROOT_AGENT_MODEL=gemini-2.5-flash
    GOOGLE_API_KEY=replace-me
    ```

Do not leave an active placeholder for the provider you are not using.

## Telegram token validation fails

The token must be present when `TELEGRAM_ENABLED=true` and match the
`number:string` shape returned by BotFather. Remove surrounding quotes and
trailing whitespace. Regenerate the token if its value is uncertain.

## The container is unhealthy

The Compose probe calls `/ready` inside the container. Inspect the startup
exception:

```bash
docker compose -f compose.yaml -f compose.prod.yaml logs --tail=200 agent
```

Common causes are missing configuration, an invalid Telegram token, a database
permission problem, or an exception while an enabled integration starts.

## Startup rejects the OTLP trace protocol

Blacki supports `grpc` and `http/protobuf`. Match the protocol to the collector:

```dotenv
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_ENDPOINT=https://collector.example.com:4318
```

A global HTTP endpoint receives `/v1/traces` automatically. A trace-specific
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` must be the complete signal URL and is used
exactly as configured. gRPC endpoints cannot contain a path.

Validate the environment against the candidate image without starting the
service:

```bash
ENV_FILE=.env docker compose --env-file .env \
  -f compose.yaml -f compose.prod.yaml run --rm --no-deps --no-build \
  --entrypoint python agent -c \
  'from blacki.utils.observability import validate_observability_environment; validate_observability_environment()'
```

The production workflow runs this preflight before stopping the existing
container.

## /ready returns HTTP 503

`status: starting` means application startup has not finished. `status:
degraded` with `database: unhealthy` means SQLite is unavailable. Optional
Mem0 configuration does not affect readiness. `/health` is an exact
compatibility alias; `/live` can still return HTTP 200 while readiness is 503.

## The web interface is unreachable

VPS defaults intentionally set:

```dotenv
SERVE_WEB_INTERFACE=false
```

The supported overlays bind only to loopback. Enable the interface with the
development overlay and use an SSH tunnel as described in
[First VPS deployment](DEPLOYMENT.md#secure-browser-access). A loopback bind is
not reachable directly from another computer.

## Host port 8080 is already in use

Change only the host port:

```dotenv
HOST_PORT=8081
```

The container still listens on port 8080. Recreate it and adjust the SSH tunnel
or local health URL to the new host port.

## Permission denied under /app

The entrypoint creates and assigns ownership for `/app/src/.adk`, `/app/data`,
and `/app/logs` before dropping to the non-root `app` user. Rebuild the current
image:

```bash
docker compose -f compose.yaml -f compose.prod.yaml up --build -d
```

Then inspect the host directory ownership. Avoid making the state directories
world-writable.

## State disappeared after restart

Confirm all three bind mounts are present in the resolved configuration:

```bash
docker compose -f compose.yaml -f compose.prod.yaml config
```

ADK HTTP and web sessions are intentionally in memory and do not persist.
SQLite-backed tools use `.adk_state/`; optional local memory uses `data/`.

## The Docker build is killed or stalls

The first source build resolves a large locked Python environment. Check VPS
memory and disk pressure, then retry after correcting the resource constraint.
Alternatively, publish a verified image from CI and use the explicit
prebuilt-image path.

## docker compose is not a command

Install the maintained Docker Compose plugin from Docker's package repository.
The legacy `docker-compose` command is not used by this project.
