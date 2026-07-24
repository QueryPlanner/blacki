# Troubleshooting

Start with:

```bash
docker compose ps
docker compose logs --tail=200 agent
docker compose config --quiet
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
ENV_FILE=.env.production docker compose --env-file .env.production config --quiet
ENV_FILE=.env.production docker compose --env-file .env.production up -d
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

The Compose probe checks whether port 8080 accepts a TCP connection inside the
container. Inspect the startup exception:

```bash
docker compose logs --tail=200 agent
```

Common causes are missing configuration, an invalid Telegram token, a database
permission problem, or an exception while an enabled integration starts.

## /health reports degraded

The application endpoint reports optional memory state as well as SQLite.
`memory_service: degraded` or `unavailable` is expected when Mem0 is not fully
configured. `database: unhealthy` requires investigation.

The endpoint returns HTTP 200 for a degraded payload. Read the JSON body; do
not use only the status code as a readiness decision.

## The web interface is unreachable

VPS defaults intentionally set:

```dotenv
BIND_ADDRESS=127.0.0.1
SERVE_WEB_INTERFACE=false
```

Enable the interface and use an SSH tunnel as described in
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
docker compose up --build -d
```

Then inspect the host directory ownership. Avoid making the state directories
world-writable.

## State disappeared after restart

Confirm all three bind mounts are present in the resolved configuration:

```bash
docker compose config
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
