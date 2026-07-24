"""OpenTelemetry and Logging setup for bare-metal adaptation.

This module provides consolidated observability configuration using standard
OpenTelemetry environment variables for vendor-neutral operation.
"""

import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpOTLPSpanExporter,
)
from opentelemetry.sdk.resources import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_NAMESPACE,
    SERVICE_VERSION,
    Resource,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OTLPTraceConfig:
    """Validated, trace-specific OTLP exporter settings."""

    endpoint: str
    endpoint_is_trace_specific: bool
    protocol: str
    headers: Mapping[str, str] | None


_OTLP_ENDPOINT_KEYS = (
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)
_OTLP_PROTOCOL_KEYS = (
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
)
_OTLP_HEADER_KEYS = (
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_HEADERS",
)
_OTLP_PROTOCOLS = frozenset({"grpc", "http/protobuf"})
_HTTP_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_HTTP_HEADER_VALUE = re.compile(r"^[\x20-\x7e]*$")
_HTTP_TRACE_PATH = "v1/traces"


def configure_otel_resource(agent_name: str) -> None:
    """Configure OpenTelemetry resource via environment variables.

    Sets standard OTel resource attributes. Exporters are configured via
    standard OTel environment variables (OTEL_EXPORTER_OTLP_*) set by the user.

    Args:
        agent_name: Unique service identifier
    """
    print("🔭 Setting OpenTelemetry Resource attributes environment variable...")
    instance_id = f"worker-{os.getpid()}-{uuid.uuid4().hex}"
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (
        f"{SERVICE_INSTANCE_ID}={instance_id},"
        f"{SERVICE_NAME}={agent_name},"
        f"{SERVICE_NAMESPACE}={os.getenv('TELEMETRY_NAMESPACE', 'local')},"
        f"{SERVICE_VERSION}={os.getenv('K_REVISION', 'local')}"
    )


def get_log_dir() -> Path:
    """Get the appropriate log directory based on environment."""
    return Path("/app/logs") if Path("/.dockerenv").exists() else Path("./logs")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "process_id": record.process,
            "thread_id": record.thread,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, default=str)


def setup_logging(log_level: str) -> None:
    """Set up basic logging with local JSON file export.

    Configures Python logging to output to stdout and append to a local JSON file.

    Args:
        log_level: Logging verbosity level as string
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure stdout handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    handlers: list[logging.Handler] = [console_handler]

    # Configure local JSON file handler
    log_dir = get_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "blacki-telemetry.log")
        file_handler.setFormatter(JSONFormatter())
        handlers.append(file_handler)
    except OSError as e:
        print(f"⚠️ Failed to create log directory or file handler: {e}")
        print("   Continuing with stdout logging only...")

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Set levels for some noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class JSONFileSpanExporter(SpanExporter):
    """Exports OpenTelemetry Spans to a local JSON Lines file."""

    def __init__(self, log_path: str):
        self.log_path = log_path

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with Path(self.log_path).open("a") as f:
                for span in spans:
                    # Convert span attributes and events to serializable dicts
                    span_data = {
                        "name": span.name,
                        "context": {
                            "trace_id": format(span.context.trace_id, "032x"),
                            "span_id": format(span.context.span_id, "016x"),
                        },
                        "parent_id": format(span.parent.span_id, "016x")
                        if span.parent
                        else None,
                        "kind": span.kind.name if span.kind else None,
                        "start_time": datetime.fromtimestamp(
                            span.start_time / 1e9, tz=UTC
                        ).isoformat()
                        if span.start_time
                        else None,
                        "end_time": datetime.fromtimestamp(
                            span.end_time / 1e9, tz=UTC
                        ).isoformat()
                        if span.end_time
                        else None,
                        "status": {
                            "status_code": span.status.status_code.name
                            if span.status
                            else None,
                            "description": span.status.description
                            if span.status
                            else None,
                        },
                        "attributes": dict(span.attributes) if span.attributes else {},
                        "events": [
                            {
                                "name": event.name,
                                "timestamp": datetime.fromtimestamp(
                                    event.timestamp / 1e9, tz=UTC
                                ).isoformat()
                                if event.timestamp
                                else None,
                                "attributes": dict(event.attributes)
                                if event.attributes
                                else {},
                            }
                            for event in span.events
                        ]
                        if span.events
                        else [],
                    }
                    f.write(json.dumps(span_data, default=str) + "\n")
            return SpanExportResult.SUCCESS
        except Exception as e:
            print(f"⚠️ Failed to write trace to {self.log_path}: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def _first_environment_setting(keys: Sequence[str]) -> tuple[str, str] | None:
    """Return the first non-empty environment setting in precedence order."""
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return key, value
    return None


def _parse_otlp_headers(headers: str) -> Mapping[str, str]:
    """Parse OTLP environment headers without logging credential material."""
    parsed_headers: dict[str, str] = {}
    for raw_header in headers.split(","):
        header = raw_header.strip()
        if not header or "=" not in header:
            raise ConfigurationError(
                "Invalid OTLP trace headers. Use comma-separated name=value pairs."
            )

        raw_name, raw_value = header.split("=", maxsplit=1)
        name = unquote(raw_name).strip().lower()
        value = unquote(raw_value).strip()
        if (
            not name
            or _HTTP_HEADER_NAME.fullmatch(name) is None
            or _HTTP_HEADER_VALUE.fullmatch(value) is None
        ):
            raise ConfigurationError(
                "Invalid OTLP trace headers. Use comma-separated name=value pairs."
            )
        parsed_headers[name] = value

    return parsed_headers


def _normalize_otlp_endpoint(
    endpoint: str,
    *,
    protocol: str,
    trace_specific: bool,
) -> str:
    """Validate an OTLP endpoint and apply protocol-specific path semantics."""
    try:
        parsed = urlparse(endpoint)
        _ = parsed.port
    except ValueError:
        raise ConfigurationError(
            "Invalid OTLP trace endpoint. Use an http(s) collector URL."
        ) from None

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(
            "Invalid OTLP trace endpoint. Use an http(s) collector URL "
            "without credentials, queries, or fragments."
        )

    if protocol == "grpc":
        if parsed.params or parsed.path not in {"", "/"}:
            raise ConfigurationError(
                "Invalid gRPC OTLP trace endpoint. Use a collector URL without a path."
            )
        return endpoint

    if trace_specific:
        return endpoint
    return f"{endpoint.rstrip('/')}/{_HTTP_TRACE_PATH}"


def _load_otlp_trace_config() -> OTLPTraceConfig | None:
    """Load OTLP trace settings with trace-specific values taking precedence."""
    endpoint_setting = _first_environment_setting(_OTLP_ENDPOINT_KEYS)
    protocol_setting = _first_environment_setting(_OTLP_PROTOCOL_KEYS)
    header_setting = _first_environment_setting(_OTLP_HEADER_KEYS)

    if endpoint_setting is None:
        if protocol_setting is not None or header_setting is not None:
            raise ConfigurationError(
                "OTLP trace protocol or headers require "
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or "
                "OTEL_EXPORTER_OTLP_ENDPOINT."
            )
        return None

    endpoint_key, endpoint = endpoint_setting
    selected_protocol = (
        protocol_setting[1].casefold() if protocol_setting is not None else "grpc"
    )
    if selected_protocol not in _OTLP_PROTOCOLS:
        raise ConfigurationError(
            "Unsupported OTLP trace protocol. This deployment supports "
            "grpc and http/protobuf."
        )

    trace_specific = endpoint_key == "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    normalized_endpoint = _normalize_otlp_endpoint(
        endpoint,
        protocol=selected_protocol,
        trace_specific=trace_specific,
    )
    headers = (
        _parse_otlp_headers(header_setting[1]) if header_setting is not None else None
    )
    return OTLPTraceConfig(
        endpoint=normalized_endpoint,
        endpoint_is_trace_specific=trace_specific,
        protocol=selected_protocol,
        headers=headers,
    )


def _create_otlp_exporter(
    otlp_config: OTLPTraceConfig,
) -> tuple[SpanExporter, str]:
    """Construct the selected network exporter without sending any spans."""
    exporter_headers = dict(otlp_config.headers) if otlp_config.headers else None
    if otlp_config.protocol == "grpc":
        otlp_exporter: SpanExporter = GrpcOTLPSpanExporter(
            endpoint=otlp_config.endpoint,
            headers=exporter_headers,
        )
    else:
        otlp_exporter = HttpOTLPSpanExporter(
            endpoint=otlp_config.endpoint,
            headers=exporter_headers,
        )
    mode = "grpc" if otlp_config.protocol == "grpc" else "http-protobuf"
    return otlp_exporter, mode


def _create_tracer_provider(log_path: Path) -> tuple[TracerProvider, str]:
    """Create a provider with local JSON and optional OTLP span processors."""
    provider = TracerProvider(resource=Resource.create())

    local_exporter = JSONFileSpanExporter(str(log_path))
    provider.add_span_processor(BatchSpanProcessor(local_exporter))

    otlp_config = _load_otlp_trace_config()
    if otlp_config is None:
        return provider, "local"

    otlp_exporter, mode = _create_otlp_exporter(otlp_config)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    return provider, f"local+otlp-{mode}"


def validate_observability_environment() -> None:
    """Construct and close the configured exporter without sending spans."""
    otlp_config = _load_otlp_trace_config()
    if otlp_config is None:
        return

    otlp_exporter, _ = _create_otlp_exporter(otlp_config)
    otlp_exporter.shutdown()


def setup_tracing() -> TracerProvider | None:
    """Set up local JSON tracing and opt-in remote OTLP export."""
    log_dir = get_log_dir()
    log_path = log_dir / "blacki-traces.log"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"⚠️ Failed to create trace directory: {e}")
        return None

    provider, mode = _create_tracer_provider(log_path)
    trace.set_tracer_provider(provider)
    logger.info("Trace exporter mode configured: %s", mode)
    return provider
