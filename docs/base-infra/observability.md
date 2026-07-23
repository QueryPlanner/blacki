# Observability

Blacki currently provides local structured logs and local OpenTelemetry span
files. It does not configure a remote OTLP exporter from environment variables
alone.

## Outputs

Inside the container:

| File | Contents |
| --- | --- |
| `/app/logs/blacki-telemetry.log` | JSON application log records |
| `/app/logs/blacki-traces.log` | JSON Lines OpenTelemetry spans |

Compose maps `/app/logs` to `./logs` on the host. Human-readable application
logs also go to stdout and are available through:

```bash
docker compose logs --follow agent
```

## Instrumentation

At startup Blacki:

1. sets `OTEL_RESOURCE_ATTRIBUTES`;
2. instruments Google ADK with `GoogleADKInstrumentor`;
3. configures stdout and JSON file logging; and
4. registers a `TracerProvider` with `JSONFileSpanExporter`.

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

## Remote export is not configured

Although OTLP exporter packages are installed, `setup_tracing()` currently
registers only the local `JSONFileSpanExporter`. Setting
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`, or related
variables does not add a network span processor.

Remote OTLP support requires an implementation change with tests. Do not assume
that traces reach Axiom, Honeycomb, Langfuse, Jaeger, or Google Cloud because an
environment variable is present.

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
