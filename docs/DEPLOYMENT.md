# First VPS deployment

This is the supported first-deployment path: an Ubuntu or Debian VPS, Docker
Engine, Docker Compose, one model provider, and Telegram long polling.

No inbound application port is required for Telegram. Blacki binds its HTTP
port to the VPS loopback interface unless you explicitly opt out.

## Before you start

You need:

- a VPS you can reach over SSH;
- a user with permission to run Docker;
- Git;
- an [OpenRouter](https://openrouter.ai/keys) key or a
  [Google AI Studio](https://aistudio.google.com/apikey) key; and
- for Telegram, a token from
  [BotFather](https://core.telegram.org/bots/features#botfather).

Install Docker Engine from Docker's official instructions for
[Ubuntu](https://docs.docker.com/engine/install/ubuntu/) or
[Debian](https://docs.docker.com/engine/install/debian/). Install the
[Docker Compose plugin](https://docs.docker.com/compose/install/linux/) from
Docker's repository so it receives package updates.

Verify the host:

```bash
docker --version
docker compose version
git --version
```

!!! warning "Do not run setup.sh unattended"

    The repository's legacy `setup.sh` performs broad root-level changes,
    including an OS upgrade, firewall changes, Docker daemon configuration,
    and docker-group membership. It is not the supported first-deployment path.
    Review it line by line before using it on a disposable host.

## 1. Clone Blacki

```bash
git clone https://github.com/QueryPlanner/blacki.git
cd blacki
cp .env.minimal .env
chmod 600 .env
```

The Docker build context is allowlisted by `.dockerignore`; `.env`, Git data,
runtime state, logs, and unrelated local files are not sent to the builder.

## 2. Configure the assistant

Open `.env` in your editor and replace every `replace-me` value:

```dotenv
AGENT_NAME=my-blacki
ROOT_AGENT_MODEL=openrouter/google/gemini-2.5-flash
OPENROUTER_API_KEY=replace-me
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=replace-me
```

`AGENT_NAME` is required. Keep only the API key for the provider you use. The
full [configuration reference](base-infra/environment-variables.md) shows the
Google AI Studio alternative and optional integrations.

The safe deployment defaults are:

```dotenv
HOST_PORT=8080
SERVE_WEB_INTERFACE=false
RELOAD_AGENTS=false
RESTART_POLICY=unless-stopped
```

## 3. Validate before starting

```bash
docker compose -f compose.yaml -f compose.prod.yaml config --quiet
```

This catches missing required Compose values and invalid YAML without starting
the service. It does not validate model or Telegram credentials.

## 4. Build and start

```bash
docker compose -f compose.yaml -f compose.prod.yaml up --build -d
docker compose -f compose.yaml -f compose.prod.yaml ps
```

The first build installs the locked Python dependencies and can take several
minutes. `docker compose ps` should eventually report the `agent` service as
healthy. The Compose healthcheck calls `/ready`, which returns success only
after the required SQLite resource is initialized and answering queries.

If startup fails:

```bash
docker compose -f compose.yaml -f compose.prod.yaml logs --tail=200 agent
```

## 5. Verify the deployment

Inspect the application health details from the VPS:

```bash
curl --fail http://127.0.0.1:8080/ready
```

`/live` is a process-only liveness endpoint. `/ready` and its compatibility
alias `/health` treat SQLite as critical and return HTTP 503 during startup or
when the database is unavailable. Optional Mem0 memory is deliberately not a
readiness dependency.

## Secure dashboard access over Tailscale

The read-only dashboard is available at `/dashboard` in the production image.
The default bind remains loopback. To make it reachable directly from your
tailnet, set the server's Tailscale IPv4 address in `.env`:

```dotenv
HOST_BIND_IP=100.x.y.z
HOST_PORT=8080
```

Recreate the service after changing `.env`:

```bash
docker compose -f compose.yaml -f compose.prod.yaml config --quiet
docker compose -f compose.yaml -f compose.prod.yaml up -d
```

Then open `http://100.x.y.z:8080/dashboard` from an allowed Tailscale device.
This binds Docker to that specific host interface; it does not bind the port
to every interface. The dashboard contains private conversations and has no
application authentication, so restrict it with Tailscale ACLs and device
policy. Never set `HOST_BIND_IP=0.0.0.0`.

If you prefer HTTPS while keeping Docker on loopback, leave `HOST_BIND_IP` at
its default and use Tailscale Serve on the VPS:

```bash
tailscale serve --bg localhost:${HOST_PORT:-8080}
```

Do not use Tailscale Funnel for this dashboard.

## Secure browser access

The ADK web interface is a development interface and is disabled in the VPS
sample. To inspect it temporarily:

1. Start the development overlay with
   `docker compose -f compose.yaml -f compose.dev.yaml up -d`.
2. From your computer, open an SSH tunnel:

   ```bash
   ssh -L 8080:127.0.0.1:8080 your-user@your-vps
   ```

3. Open `http://127.0.0.1:8080` locally.
4. Return to the production overlay when finished.

The ADK web interface is a development surface. Keep it disabled in production
and use an authenticated reverse proxy only as a separate, deliberate
deployment decision.

## Use a prebuilt image only when verified

The source-build path above works without a container registry. If you publish
an image from your own fork, verify its tag and visibility first, then set:

```dotenv
IMAGE=ghcr.io/your-owner/blacki:your-tag
```

Authenticate to GHCR if the package is private, then use:

```bash
docker compose -f compose.yaml -f compose.prod.yaml pull
docker compose -f compose.yaml -f compose.prod.yaml up --no-build -d
```

`--no-build` makes the prebuilt-image path explicit. Do not assume that the
QueryPlanner package is anonymously pullable; that has not been verified.

## Important change for existing installations

Older Compose defaults could publish port 8080 on all host interfaces and
enable the web UI and agent reload. The base file now publishes no host port;
both supported overlays force loopback. Production also forces the UI and
reload off, while the development overlay enables both without public
exposure. After the first successful start, continue with
[Day-two operations](operations.md).
