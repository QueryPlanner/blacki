# Observability

Blacki always provides local structured logs and local OpenTelemetry span files.
It can additionally export spans to a remote collector through gRPC or
HTTP/protobuf OTLP.

## Outputs

Inside the container:

| File | Contents |
| --- | --- |
| `/app/logs/blacki-telemetry.log` | JSON application log records |
| `/app/logs/blacki-traces.log` | JSON Lines OpenTelemetry spans |

Compose maps `/app/logs` to `./logs` on the host. Human-readable application
logs also go to stdout and are available through:

```bash
docker compose -f compose.yaml -f compose.prod.yaml logs --follow agent
```

## Instrumentation

At startup Blacki:

1. sets `OTEL_RESOURCE_ATTRIBUTES`;
2. instruments Google ADK with `GoogleADKInstrumentor` unless a private-tool
   mode is enabled;
3. configures stdout and JSON file logging; and
4. registers a `TracerProvider` with `JSONFileSpanExporter`; and
5. adds the selected OTLP exporter when a validated trace endpoint is configured.

The resource contains:

| Attribute | Source |
| --- | --- |
| `service.name` | `AGENT_NAME` |
| `service.namespace` | `TELEMETRY_NAMESPACE`, default `local` |
| `service.version` | `K_REVISION`, default `local` |
| `service.instance.id` | Process ID plus generated UUID |

## Message content

Both `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false` and
`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false` are required to disable current
and legacy ADK content capture. Capturing prompts and responses can expose
personal data, credentials, and tool results.

When any private-tool integration is configured, Blacki forces both values to
`false` and disables the OpenInference Google ADK instrumentor. The ADK session
database still retains tool calls and results so interrupted turns and
confirmations can resume correctly.

## Remote OTLP export

Local-only mode is the default and never creates a network exporter. To add
remote gRPC span export, set:

```dotenv
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://collector.example.com:4317
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_TRACES_HEADERS=authorization=replace-me
```

For HTTP/protobuf, set:

```dotenv
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://collector.example.com:4318/v1/traces
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_TRACES_HEADERS=authorization=Bearer%20replace-me
```

You can instead use the global `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_PROTOCOL`, and `OTEL_EXPORTER_OTLP_HEADERS` fallbacks.
Trace-specific values take precedence. The protocol defaults to `grpc` when
omitted.

The endpoint must use `http` or `https`, identify a host, and contain no
embedded username, password, query, or fragment. gRPC endpoints cannot contain a
path. A trace-specific HTTP/protobuf endpoint is used exactly as configured; a
global HTTP/protobuf endpoint receives `/v1/traces` after any existing base
path. Headers use comma-separated, URL-encoded `name=value` pairs.
Use `https` whenever headers contain credentials. Plain `http` sends headers
without transport encryption and is appropriate only for a trusted local
collector.

Protocol or header settings without an endpoint fail startup with a secret-free
configuration error. The application logs only `local`, `local+otlp-grpc`, or
`local+otlp-http-protobuf` once; it never logs the endpoint or headers.

## Retention

The application files append and have no built-in size or time rotation.
Docker's stdout log-driver rotation does not rotate `./logs/*.log`.

Monitor them:

```bash
du -sh logs
```

Configure host-side retention before long-running use. Archive or clear files
only during a maintenance window after preserving anything needed for an
incident.

## Failure behavior

If the application cannot create its log directory or file handler, it reports
the error and continues with stdout logging. Treat that as degraded
observability, not a successful persistence setup.

## Private dashboard

The server always mounts the read-only operator dashboard at `/dashboard`. It
reads the local ADK session database, application JSON logs, and JSON Lines
traces from the same persistent volumes. The dashboard can filter users and
sessions and inspect individual conversations and traces;
Telegram `/reset` starts the next versioned ADK session and leaves earlier
session rows available. The append-only log and trace files are not deleted by
`/reset`, so the dashboard continues to show that history.

The Sessions view can show the recorded system instructions and tool
descriptions for an individual model request. It verifies each capture against
the selected ADK session and filters common credential patterns. The dashboard
uses existing local captures. Blacki's normal private-tool settings suppress
content capture, so recent requests can have conversation history and trace
metadata with no captured system instructions.

The User wiki tab reads saved claims for the selected direct Telegram user
and respects the user's memory setting. It uses the same local `wiki.db` as
Telegram, beside the tools database.

Both views are disabled by default. For an existing private dashboard, set
`BLACKI_DASHBOARD_PRIVATE_HOSTS` to the exact hostnames or IP addresses operators
use in their browser, without schemes or ports. For example, use `127.0.0.1`
for a loopback-only dashboard or the server's exact Tailscale Serve hostname.
Use a comma to separate multiple hosts. For GitHub-managed deployments, set
the `BLACKI_DASHBOARD_PRIVATE_HOSTS` secret in the production environment.
Restrict access to the dashboard with the loopback or Tailscale controls below.
The host setting protects browser access and does not authenticate operators.

For OpenRouter-backed requests, Blacki also writes a content-free SQLite usage
ledger at `{AGENT_DIR}/.adk/costs.db` (override with
`BLACKI_COST_LEDGER_PATH`). The ledger stores identity, model, token, provider
response, and fixed-point cost fields, but never prompts, responses, or tool
arguments. Provider-reported OpenRouter account cost and upstream inference
cost are kept separately; historical gateway estimates remain labelled
estimated, while new OpenRouter rows use provider-reported costs only.
The dashboard uses UTC calendar months for monthly totals and averages. The
average is across users with an exact or estimated cost in the current month;
users with unavailable cost are excluded and the reported coverage is shown.
Records created before cost capture, or responses without a provider cost,
remain unavailable rather than being treated as zero.

This is an admin-only, private-data surface. The application does not add
HTTP Basic auth, cookies, or Tailscale identity-header authentication. For
direct tailnet access, set `HOST_BIND_IP` to the host's Tailscale IPv4 address
and restrict access with Tailscale ACLs and device posture. Alternatively,
keep the production Compose loopback binding and use Tailscale Serve to reverse
proxy the service to your tailnet over HTTPS:

```bash
HOST_PORT="$(sed -n 's/^HOST_PORT="\{0,1\}\([0-9][0-9]*\)"\{0,1\}$/\1/p' .env | tail -n 1)"
tailscale serve --bg "localhost:${HOST_PORT:-8080}"
```

Run the commands from the Blacki repository directory; the first line reads
the configured host port from `.env` and falls back to 8080 when absent.

The `--bg` flag keeps the Serve configuration across host reboots and service
restarts while the endpoint remains tailnet-only. Use `tailscale serve`, never
`tailscale funnel`, for this dashboard. Funnel is public internet exposure and
is not an acceptable boundary for conversation logs. If you use the direct
bind, never set `HOST_BIND_IP=0.0.0.0`. The current command syntax is documented
in the official
[Tailscale Serve CLI reference](https://tailscale.com/docs/reference/tailscale-cli/serve).
