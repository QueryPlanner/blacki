# Observability

Blacki uses ADK's official OpenTelemetry setup and the OpenTelemetry SDK's
exporters. Spans and log records go to two local JSON Lines files. Nothing is
filtered or redacted.

## Outputs

Inside the container:

| File | Contents |
| --- | --- |
| `/app/logs/blacki-traces.log` | Every OpenTelemetry span, one JSON object per line |
| `/app/logs/blacki-telemetry.log` | Every OpenTelemetry log record, one JSON object per line |

Each line is the SDK's own JSON for a span (`ReadableSpan.to_json`) or log
record (`ReadableLogRecord.to_json`, plus the instrumentation scope that holds
the Python logger name). Log records carry the trace and span IDs of the span
that was active when they were written. Both files are created with mode `0600`.

Compose maps `/app/logs` to `./logs` on the host. Human-readable application
logs also go to stdout and are available through:

```bash
docker compose -f compose.yaml -f compose.prod.yaml logs --follow agent
```

## Instrumentation

At startup Blacki:

1. sets `OTEL_RESOURCE_ATTRIBUTES`;
2. sets the GenAI telemetry defaults below, unless already set;
3. calls ADK's `google.adk.telemetry.setup.maybe_set_otel_providers` with a
   `BatchSpanProcessor` and a `BatchLogRecordProcessor`, each wrapping the
   SDK console exporter pointed at its file; and
4. adds the SDK `LoggingHandler`, so Python `logging` records reach
   `blacki-telemetry.log`.

The resource contains:

| Attribute | Source |
| --- | --- |
| `service.name` | `AGENT_NAME` |
| `service.namespace` | `TELEMETRY_NAMESPACE`, default `local` |
| `service.version` | `K_REVISION`, default `local` |
| `service.instance.id` | Process ID plus generated UUID |

## Message content

Blacki opts in to ADK's experimental OpenTelemetry GenAI semantic conventions:

| Variable | Default |
| --- | --- |
| `OTEL_SEMCONV_STABILITY_OPT_IN` | `gen_ai_latest_experimental` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `SPAN_ONLY` |
| `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` | `false` |

Each `generate_content` span then carries `gen_ai.system_instructions`,
`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.tool.definitions`,
and token usage. Each `execute_tool` span carries `gen_ai.tool.name` and
`gen_ai.tool.call.id`. ADK's legacy `gcp.vertex.agent.llm_request` and
`gcp.vertex.agent.tool_call_args` content is turned off, because it would store
every request twice.

These attributes follow the OpenTelemetry GenAI conventions, which are still
experimental. A future ADK or OpenTelemetry release may rename them; the
dashboard's tests pin the format it reads.

Treat the log directory as private user data. It holds full prompts, personal
messages, tool arguments, and tool results, unredacted. Protecting the log
volume is the control.

At `LOG_LEVEL=DEBUG`, `LoggingCallbacks` also logs full model request contents
and tool payloads, including Gmail and health data, to stdout and
`blacki-telemetry.log`.

## Remote export with OTLP

`opentelemetry-exporter-otlp-proto-http` is a declared dependency, so the image
can export over OTLP/HTTP. ADK adds span, log, and metric exporters when
`OTEL_EXPORTER_OTLP_ENDPOINT` or a signal-specific endpoint is set. The local
JSON Lines files keep working alongside them, so this is a dual export rather
than a replacement.

Use the base endpoint when you want traces, logs, and metrics:

| Variable | Example |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://openobserve.internal:5080/api/default` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `Authorization=Basic%20<base64(email:password)>` |

The HTTP exporter appends `/v1/traces`, `/v1/logs`, and `/v1/metrics` to the
base endpoint. Do not add a trailing slash and do not include `/v1/...` in
`OTEL_EXPORTER_OTLP_ENDPOINT`.

Signal-specific variables are used verbatim, so they must include the full
path. Use them when you want only some signals:

```bash
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://openobserve.internal:5080/api/default/v1/traces
OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://openobserve.internal:5080/api/default/v1/logs
```

### OpenObserve

OpenObserve accepts OTLP/HTTP at `http://<host>:5080/api/<organization>`. For
the `default` organization:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://100.77.130.71:5080/api/default
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<token>
```

Build the Basic token from the OpenObserve login, then URL-encode the space:

```bash
printf '%s' 'user@example.com:password' | base64
# Prefix the result with: Authorization=Basic%20
```

OpenObserve Cloud uses `https://api.openobserve.ai/api/<organization>`.

When OpenObserve runs on the Docker host, use `host.docker.internal`; Compose
already maps it:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:5080/api/default
```

You can select a stream with an extra header:

```bash
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<token>,stream-name=blacki
```

### Production secrets

The deployment workflow writes these GitHub Actions secrets into the generated
`.env`:

| Secret | Value |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base OTLP endpoint, for example `http://100.77.130.71:5080/api/default` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `Authorization=Basic%20<base64(email:password)>` |

Leave all three unset to keep local files only. Do not also set
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`,
`OTEL_EXPORTER_OTLP_TRACES_HEADERS`, or `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`
unless you intentionally want to override the base endpoint for one signal.
The deployment workflow does not pass those signal-specific variables.

### What leaves the host

The OTLP export sends the same spans and log records as the local files. That
includes full prompts, personal messages, tool arguments, tool results, Gmail
and health data, and any content logged at `LOG_LEVEL=DEBUG`. There is no
redaction. Before enabling remote export:

- use HTTPS or a private network path you trust;
- restrict access to the OpenObserve organization and its credentials;
- choose a retention period that matches the data;
- keep `LOG_LEVEL` at `INFO` unless you intend to ship full lifecycle payloads.

The local files remain the fallback, but OpenObserve becomes a second copy of
the same private data.

### Verify ingestion

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  -X POST "$OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces" \
  -H "Authorization: Basic <token>" \
  -H "Content-Type: application/json" \
  -d '{"resourceSpans":[]}'
```

A `200` means the endpoint and credentials are correct. In OpenObserve, check
**Streams → Traces**, **Streams → Logs**, and **Streams → Metrics**.

## Retention

Neither file is rotated, trimmed, or deleted by the application. Docker's stdout
log-driver rotation does not touch `./logs`. Each model-call span repeats the
conversation sent to the model, so `blacki-traces.log` grows quickly. Monitor
disk usage:

```bash
du -sh logs
```

The dashboard reads only the last 128 MiB of the trace file and the last 16 MiB
of the log file, so page loads stay bounded. Older lines stay on disk.

## Failure behavior

If either file cannot be opened, startup prints the error and continues without
file exporters. A `TracerProvider` is still installed, so ADK cannot fall back
to an unbounded in-memory exporter. Treat this as degraded observability.

## Private dashboard

The server always mounts the read-only operator dashboard at `/dashboard`. It
reads the local ADK session database and the two telemetry files from the same
persistent volumes. Telegram `/reset` starts the next versioned ADK session and
leaves earlier session rows and telemetry lines in place.

The Logs and Traces tabs show records exactly as written, with every attribute.
The Sessions view opens a model-call inspector for each `generate_content` span.
It shows the system instructions, tool definitions, the messages sent, the
model's output, each tool call with its result and execution span, and the raw
span. It checks each span against the selected ADK session before showing it.

The User wiki tab lists the saved Markdown pages for the selected direct
Telegram user and shows the selected page as plain text. It respects the
user's memory setting. It uses the same local `wiki.db` as Telegram, beside the
tools database.

The Context, Wiki, Logs, and Traces views open only from a loopback address, a
Tailscale device IP, or the full hostname provided by Tailscale Serve. The
application checks the browser's request host and blocks cross-origin reads.
Operator access still depends on the loopback or Tailscale controls below.

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
