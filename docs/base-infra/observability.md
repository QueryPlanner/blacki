# Observability

Blacki always provides local structured logs and local OpenTelemetry span files.
It can additionally export spans to a remote collector through gRPC or
HTTP/protobuf OTLP.

## Outputs

Inside the container:

| File | Contents |
| --- | --- |
| `/app/logs/blacki-telemetry.log` | JSON application log records (rotated) |
| `/app/logs/blacki-traces.log` | Slim JSON Lines span overview (no content) |
| `/app/logs/traces/YYYY-MM-DD.jsonl` | Full JSON Lines spans for that UTC day |

The slim overview keeps only allowlisted metadata such as model, token, and
cost attributes. The daily files keep the complete span attributes, including
the prompt, tool declarations, tool arguments, and tool responses that ADK
records. Both trace files are created with mode `0600` and are intended for the
private operator dashboard.

Compose maps `/app/logs` to `./logs` on the host. Human-readable application
logs also go to stdout and are available through:

```bash
docker compose -f compose.yaml -f compose.prod.yaml logs --follow agent
```

## Instrumentation

At startup Blacki:

1. sets `OTEL_RESOURCE_ATTRIBUTES`;
2. configures stdout and rotating JSON file logging;
3. registers a `TracerProvider` with `FlatFileSpanExporter`; and
4. adds the selected OTLP exporter when a validated trace endpoint is configured.

Blacki relies on ADK's built-in span instrumentation. It does not install the
OpenInference Google ADK instrumentor. The resource contains:

| Attribute | Source |
| --- | --- |
| `service.name` | `AGENT_NAME` |
| `service.namespace` | `TELEMETRY_NAMESPACE`, default `local` |
| `service.version` | `K_REVISION`, default `local` |
| `service.instance.id` | Process ID plus generated UUID |

## Message content

`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` is on by default. Blacki keeps full
prompts, tool declarations, tool calls, and model responses in the local daily
trace files so the private dashboard can show exactly what the model received
and returned. Only the slim overview omits content.

Content capture is a deliberate operator feature, not an external export.
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` stays `false` so the same
content is not duplicated into OTel log events. Treat the local log directory as
private user data: it can contain personal messages, tool arguments, and tool
results. Common credential patterns are redacted before writes, but redaction is
best-effort and is not a substitute for protecting the log volume.

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

`blacki-telemetry.log` rotates at UTC midnight and keeps fourteen daily backups
(`blacki-telemetry.log.YYYY-MM-DD`). The slim trace log and the daily
`traces/YYYY-MM-DD.jsonl` files are append-only and are not rotated by the
application. Docker's stdout log-driver rotation does not rotate `./logs/*.log`.

Monitor them:

```bash
du -sh logs
```

Configure host-side retention for the trace directory before long-running use.
Archive or clear files only during a maintenance window after preserving
anything needed for an incident.

## Failure behavior

If the application cannot create its log directory or file handler, it reports
the error and continues with stdout logging. Tracing still installs a
`TracerProvider` so ADK cannot fall back to an unbounded in-memory exporter.
Treat either condition as degraded observability, not a successful persistence
setup.

## Private dashboard

The server always mounts the read-only operator dashboard at `/dashboard`. It
reads the local ADK session database, application JSON logs, and daily JSON Lines
traces from the same persistent volumes. The dashboard can filter users and
sessions and inspect individual conversations and traces;
Telegram `/reset` starts the next versioned ADK session and leaves earlier
session rows available. The append-only log and trace files are not deleted by
`/reset`, so the dashboard continues to show that history.

The Sessions view reads the system instruction and tool declarations from ADK's
`gcp.vertex.agent.llm_request` span attribute in the daily trace files. It
verifies each capture against the selected ADK session and filters common
credential patterns. Because content capture is on by default, recent requests
normally include their captured system instructions and tool descriptions.

The User wiki tab lists the saved Markdown pages for the selected direct
Telegram user and shows the selected page as plain text. It respects the
user's memory setting. It uses the same local `wiki.db` as Telegram, beside the
tools database.

Both views are available by default. Open the dashboard at a loopback address,
a Tailscale device IP, or the full hostname provided by Tailscale Serve.
The application checks the browser's request host and blocks cross-origin
reads for both views. Operator access still depends on the loopback or
Tailscale controls below.

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
