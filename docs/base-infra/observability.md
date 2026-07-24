# Observability

Blacki always provides local structured logs and local OpenTelemetry span files.
It can additionally export spans to a remote collector through gRPC OTLP.

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
2. instruments Google ADK with `GoogleADKInstrumentor`;
3. configures stdout and JSON file logging; and
4. registers a `TracerProvider` with `JSONFileSpanExporter`; and
5. adds a gRPC OTLP exporter when a validated trace endpoint is configured.

The resource contains:

| Attribute | Source |
| --- | --- |
| `service.name` | `AGENT_NAME` |
| `service.namespace` | `TELEMETRY_NAMESPACE`, default `local` |
| `service.version` | `K_REVISION`, default `local` |
| `service.instance.id` | Process ID plus generated UUID |

## Message content

`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false` is the safe default.
Capturing prompts and responses can expose personal data, credentials, and
tool results. Enable it only for a deliberate debugging session with an
appropriate retention policy.

## Remote gRPC OTLP export

Local-only mode is the default and never creates a network exporter. To add
remote span export, set either:

```dotenv
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://collector.example.com:4317
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_TRACES_HEADERS=authorization=replace-me
```

or the global `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`, and
`OTEL_EXPORTER_OTLP_HEADERS` fallbacks. Trace-specific values take precedence.
The protocol defaults to `grpc` when omitted. HTTP/protobuf is not installed or
accepted.

The endpoint must use `http` or `https`, identify a host, and contain no
embedded username, password, path, query, or fragment. Protocol or header
settings without an endpoint fail startup with a secret-free configuration
error. The application logs only `local` or `local+otlp-grpc` once; it never
logs the endpoint or headers.

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
