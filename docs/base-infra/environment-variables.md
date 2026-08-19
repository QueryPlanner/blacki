# Configuration

Copy `.env.minimal` for the Telegram golden path or `.env.example` for all
documented options. Keep `.env` private.

## Required values

**`AGENT_NAME`**

:   Required unique service identifier. Compose fails during interpolation when
    it is absent.

**One model API key**

:   Set `OPENROUTER_API_KEY` or `GOOGLE_API_KEY`. Startup validation fails when
    both are absent.

An explicit `ROOT_AGENT_MODEL` is strongly recommended so provider routing does
not depend on a default.

## Docker Compose host settings

These values are consumed by Compose rather than `ServerEnv`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST_BIND_IP` | `127.0.0.1` | Production host interface for the published port |
| `HOST_PORT` | `8080` | Host-side port |
| `RESTART_POLICY` | `unless-stopped` | Compose restart policy |
| `IMAGE` | `blacki:local` | Image name or verified registry reference |
| `ENV_FILE` | `.env` | File injected into the container |

`HOST` and `PORT` are different: they control the process inside the container.
Compose sets them to `0.0.0.0` and `8080`; the supported overlays publish only
to host loopback by default. The production overlay uses `HOST_BIND_IP` when it
is explicitly set, and `HOST_PORT` selects the host-side port. Set
`HOST_BIND_IP` to the server's Tailscale IPv4 address for direct tailnet access
to `/dashboard`. Do not set it to `0.0.0.0`; that publishes the private-data
surface on every host interface.

## Server

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENT_NAME` | None | Required identity for the service and telemetry |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `HOST` | `127.0.0.1` | Process bind address outside Compose |
| `PORT` | `8080` | Process port outside Compose |
| `AGENT_DIR` | `src` | ADK agents and local-state base directory |
| `SERVE_WEB_INTERFACE` | `false` | Enable the ADK development web interface |
| `RELOAD_AGENTS` | `false` | Reload agent definitions; development only |
| `ALLOW_ORIGINS` | local origins JSON | JSON array of CORS origins |
| `AGENT_ENGINE` | unset | Optional Agent Engine identifier |
| `SQLITE_PATH` | `{AGENT_DIR}/.adk/tools.db` | SQLite file for application tools |

For `ALLOW_ORIGINS`, use a JSON array string:

```dotenv
ALLOW_ORIGINS=["http://127.0.0.1","http://127.0.0.1:8080"]
```

## Model providers

=== "OpenRouter"

    ```dotenv
    ROOT_AGENT_MODEL=openrouter/google/gemini-2.5-flash
    OPENROUTER_API_KEY=replace-me
    ```

    Blacki builds a LiteLLM model and normalizes common model identifiers to
    OpenRouter form when this key is present.

    Set `ROOT_AGENT_REASONING_EFFORT` to configure the process-wide reasoning
    fallback. Supported values are `max`, `xhigh`, `high`, `medium`, `low`,
    `minimal`, and `none`; leave it unset to inherit the provider default.
    A Telegram chat's explicit thinking selection overrides this fallback for
    that chat. Only configure an effort advertised by the selected model.

=== "Google AI Studio"

    ```dotenv
    ROOT_AGENT_MODEL=gemini-2.5-flash
    GOOGLE_API_KEY=replace-me
    ```

Do not leave a fake `OPENROUTER_API_KEY` active in a Google-only configuration;
its presence changes model routing.

## Telegram

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEGRAM_ENABLED` | `false` | Start Telegram long polling |
| `TELEGRAM_BOT_TOKEN` | unset | Token from BotFather |
| `KOKORO_TTS_BASE_URL` | unset | Register private Kokoro speech delivery for Telegram |
| `KOKORO_TTS_VOICE` | `af_heart` | Kokoro voice ID used for generated MP3 audio |

The token is required and format-validated when Telegram is enabled.

`KOKORO_TTS_BASE_URL` is an optional HTTP or HTTPS base URL without a path to
`/v1/audio/speech`; Blacki appends that fixed endpoint. The URL must be
reachable from inside the Blacki container. Do not use `localhost` for a
Kokoro process on another Tailscale host; configure that host's Tailscale IP or
MagicDNS name instead. Plain HTTP is suitable only across a trusted private
network such as the encrypted Tailscale connection.

### Google Health connector

The optional connector is enabled when all three required values are present.
It is exposed only to the Telegram root agent and only accepts private-chat
identities.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_HEALTH_CLIENT_ID` | unset | Server-side Google OAuth web-client ID |
| `GOOGLE_HEALTH_CLIENT_SECRET` | unset | Server-side Google OAuth web-client secret |
| `GOOGLE_HEALTH_REDIRECT_URI` | `http://127.0.0.1:8080/integrations/google-health/callback` | OAuth callback; use HTTPS in production |
| `GOOGLE_HEALTH_TOKEN_ENCRYPTION_KEY` | unset | Fernet key used to encrypt refresh tokens at rest |
| `GOOGLE_HEALTH_SYNC_INTERVAL_HOURS` | `12` | Background sync interval |
| `GOOGLE_HEALTH_MANUAL_REFRESH_COOLDOWN_SECONDS` | `3600` | Per-user on-demand refresh cooldown |
| `GOOGLE_HEALTH_OAUTH_STATE_TTL_SECONDS` | `600` | Lifetime of one-time OAuth state |

Blacki requests the current Google Health read-only activity/fitness,
health-metrics/measurements, and sleep scopes. Do not paste the client secret
or Fernet key into logs, chat, or source control. The callback URL must exactly
match the Google Cloud OAuth client configuration. The Apple Health-to-Google
Health or Fitbit import step is configured separately by the user and may be
incomplete; Blacki does not access HealthKit directly.

## Search and browser tools

| Variable | Default | Purpose |
| --- | --- | --- |
| `EXA_API_KEY` | unset | Primary Exa search integration |
| `BRAVE_SEARCH_API_KEY` | unset | Brave search fallback |
| `BROWSER_USE_API_KEY` | unset | Browser Use Cloud automation |

These integrations are optional. Their absence should not replace the required
model key.

## Mem0 memory

Mem0 is optional. The current samples support:

- `MEM0_LLM_PROVIDER`, `MEM0_LLM_MODEL`, `MEM0_LLM_API_KEY`,
  `MEM0_LLM_TEMPERATURE`, and `MEM0_LLM_MAX_TOKENS`;
- `MEM0_EMBEDDER_PROVIDER`, `MEM0_EMBEDDER_MODEL`,
  `MEM0_EMBEDDER_DIMS`, and `MEM0_EMBEDDER_API_KEY`;
- `MEM0_USER_ID`, `MEM0_COLLECTION_NAME`, and `MEM0_SEARCH_LIMIT`;
- Qdrant Cloud through `MEM0_QDRANT_URL` and `MEM0_QDRANT_API_KEY`;
- local embedded Qdrant through `MEM0_QDRANT_PATH`; or
- a remote server through `MEM0_QDRANT_HOST` and `MEM0_QDRANT_PORT`.

Compose mounts `./data` at `/app/data`. Use a path under `/app/data` for local
memory that must persist:

```dotenv
MEM0_QDRANT_PATH=/app/data/qdrant
```

If Qdrant Cloud values are present, the provider stores the vectors remotely.

## Delegated task worker

| Variable | Default | Purpose |
| --- | --- | --- |
| `TASK_WORKER_ENABLED` | `true` | Register a same-privilege ADK task worker |

By default, the root agent can delegate one complex task at a time to a
registered ADK task-mode child. Set `TASK_WORKER_ENABLED=false` to opt out. The
worker receives an independently built copy of the root agent's user-facing
toolset and shares the same ADK session state, so sandbox tools reconnect to the
same sandbox ID.

This is not a security boundary or a background worker pool: the worker has the
root agent's privileges, runs within the request, and cannot delegate
recursively. Delegation can add another model and tool turn, increasing latency
and model usage.

## OpenSandbox

| Variable | Default | Purpose |
| --- | --- | --- |
| `SANDBOX_ENABLED` | `false` | Register code-execution tools |
| `SANDBOX_DOMAIN` | `localhost:9090` | OpenSandbox server address |
| `SANDBOX_API_KEY` | unset | Optional server credential |
| `SANDBOX_TIMEOUT_MINUTES` | `30` | Sandbox lifetime |
| `SANDBOX_MEMORY_LIMIT` | `512Mi` | Per-sandbox memory setting |
| `SANDBOX_CPU_LIMIT` | `0.5` | Per-sandbox CPU setting |
| `SANDBOX_IMAGE` | project default | Code-interpreter image |

Running a local OpenSandbox server adds Docker and resource requirements beyond
the Blacki golden path.

## Cloudflare R2 user files

| Variable | Default | Purpose |
| --- | --- | --- |
| `R2_FILES_ENABLED` | `false` | Persist supported Telegram attachments |
| `R2_ENDPOINT_URL` | unset | Account or jurisdiction-specific S3 endpoint |
| `R2_BUCKET_NAME` | unset | Private attachment bucket |
| `R2_ACCESS_KEY_ID` | unset | Bucket-scoped S3 access key |
| `R2_SECRET_ACCESS_KEY` | unset | Bucket-scoped S3 secret |
| `R2_OWNER_HMAC_SECRET` | unset | Secret used to hide Telegram IDs in object keys |
| `R2_FILE_KEY_PREFIX` | `blacki/user-files` | Private object-key prefix |
| `R2_FILE_RETENTION_DAYS` | `90` | Application availability window |

Create a private R2 bucket, grant Blacki only Object Read & Write permission
for that bucket, and add an R2 lifecycle rule that deletes
`blacki/user-files/` objects after 90 days. Keep the lifecycle setting aligned
with `R2_FILE_RETENTION_DAYS`. Files are catalogued in the persistent SQLite
volume; include that database in backups. R2 credentials remain in the Blacki
host and are never copied into a sandbox.

If R2 is unavailable, Telegram processing can continue with an explicit
temporary-storage warning. If the sandbox is unavailable, a successfully
stored object remains available for a later restore.

## Zepto MCP

| Variable | Default | Purpose |
| --- | --- | --- |
| `ZEPTO_MCP_ENABLED` | `false` | Enable the root-only Zepto skill after OAuth |
| `ZEPTO_MCP_ALLOWED_TELEGRAM_CHAT_IDS` | unset | Comma-separated positive private chat IDs allowed to use the shared account |
| `ZEPTO_MCP_CONFIG_DIR` | `data/credentials/zepto-mcp-remote` | Permission-protected bridge credential directory shared by host and container |

The allowlist deliberately rejects Telegram groups, topics, and arbitrary HTTP
user IDs. All allowed chats share one Zepto account and cart. Complete the
one-time OAuth flow before enabling the integration; see
[Zepto MCP](../zepto-mcp.md).

## Observability

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEMETRY_NAMESPACE` | `local` | OpenTelemetry service namespace |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | Allow instrumentors to capture message content |
| `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` | `false` | Allow legacy ADK spans to capture prompts and tool data |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | unset | Preferred complete trace collector URL |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Global collector URL fallback; HTTP adds `/v1/traces` |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | `grpc` | Preferred trace protocol: `grpc` or `http/protobuf` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | Global protocol fallback |
| `OTEL_EXPORTER_OTLP_TRACES_HEADERS` | unset | Preferred trace authentication headers |
| `OTEL_EXPORTER_OTLP_HEADERS` | unset | Global authentication-header fallback |

Blacki always exports spans to local JSON. A validated trace-specific or global
endpoint adds gRPC or HTTP/protobuf OTLP export; trace-specific values take
precedence. See [Observability](observability.md).

When Zepto is enabled, Blacki forces both content-capture variables to `false`
and disables the OpenInference Google ADK instrumentor because it otherwise
records raw tool parameters.

## Precedence

For Docker Compose:

1. shell values and `--env-file` drive Compose interpolation;
2. the service's `environment` mapping overrides matching `env_file` values;
3. remaining values come from `ENV_FILE`, which defaults to `.env`.

For local Python, `initialize_environment` loads the nearest `.env` with
`override=True`, so values in that file can replace existing process values.

## Secret handling

- Keep `.env` out of Git; it is already ignored.
- Run `chmod 600 .env` on a multi-user VPS.
- Never include secrets in `docker compose config` output shared with others.
- Rotate a token immediately if it appears in Git, logs, an issue, or chat.
- Prefer provider-scoped, least-privilege credentials.
