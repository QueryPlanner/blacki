"""OpenTelemetry and Logging setup for bare-metal adaptation.

This module provides consolidated observability configuration.  Spans are
written to local flat files; the optional OTLP path is retained until the
dedicated removal change lands.
"""

import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
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

from ..utils.exceptions import ConfigurationError
from .attributes import TRACE_ATTRIBUTE_ALLOWLIST
from .redaction import RedactionFilter, redact_secrets

logger = logging.getLogger("blacki.utils.observability")

ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"
_LOG_FILE_NAME = "blacki-telemetry.log"
_SLIM_TRACE_FILE_NAME = "blacki-traces.log"
_TRACE_DIRECTORY_NAME = "traces"
_LOG_BACKUP_COUNT = 14


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


def adk_span_content_capture_enabled() -> bool:
    """Return whether ADK records content on its own spans.

    ADK defaults this setting on; only the two documented falsy values turn it
    off.
    """
    value = os.getenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, "true").strip().lower()
    return value not in {"0", "false"}


class JSONFormatter(logging.Formatter):
    """Format log records as one JSON object with trace correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": redact_secrets(record.getMessage()),
            "process_id": record.process,
            "thread_id": record.thread,
        }
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            log_record["trace_id"] = format(span_context.trace_id, "032x")
            log_record["span_id"] = format(span_context.span_id, "016x")
        if record.exc_info:
            log_record["exception"] = redact_secrets(
                self.formatException(record.exc_info)
            )
        return json.dumps(log_record, default=str, ensure_ascii=False)


def setup_logging(log_level: str) -> None:
    """Set up basic logging with local JSON file export.

    Configures Python logging to output to stdout and append to a local JSON
    file.  The file rotates at UTC midnight and keeps fourteen days.

    Args:
        log_level: Logging verbosity level as string
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    redaction_filter = RedactionFilter()

    # Configure stdout handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    console_handler.addFilter(redaction_filter)

    handlers: list[logging.Handler] = [console_handler]

    # Configure local JSON file handler
    log_dir = get_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            log_dir / _LOG_FILE_NAME,
            when="midnight",
            utc=True,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.addFilter(redaction_filter)
        handlers.append(file_handler)
    except OSError as e:
        print(f"⚠️ Failed to create log directory or file handler: {e}")
        print("   Continuing with stdout logging only...")

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Set levels for some noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _iso_timestamp(nanoseconds: int | None) -> str | None:
    if not nanoseconds:
        return None
    return datetime.fromtimestamp(nanoseconds / 1e9, tz=UTC).isoformat()


def _span_end_day(span: ReadableSpan) -> str:
    """Return the UTC date used to name a span's daily trace file."""
    end = span.end_time or span.start_time
    if end:
        return datetime.fromtimestamp(end / 1e9, tz=UTC).date().isoformat()
    return datetime.now(tz=UTC).date().isoformat()


def _span_record(
    span: ReadableSpan,
    *,
    include_content: bool,
) -> dict[str, object]:
    attributes = dict(span.attributes) if span.attributes else {}
    if not include_content:
        attributes = {
            name: value
            for name, value in attributes.items()
            if name in TRACE_ATTRIBUTE_ALLOWLIST
        }
    events = []
    if include_content and span.events:
        events = [
            {
                "name": event.name,
                "timestamp": _iso_timestamp(event.timestamp),
                "attributes": dict(event.attributes) if event.attributes else {},
            }
            for event in span.events
        ]
    return {
        "name": span.name,
        "context": {
            "trace_id": format(span.context.trace_id, "032x"),
            "span_id": format(span.context.span_id, "016x"),
        },
        "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
        "kind": span.kind.name if span.kind else None,
        "start_time": _iso_timestamp(span.start_time),
        "end_time": _iso_timestamp(span.end_time),
        "status": {
            "status_code": span.status.status_code.name if span.status else None,
            "description": span.status.description if span.status else None,
        },
        "attributes": attributes,
        "events": events,
    }


def _append_private_lines(path: Path, lines: list[str]) -> None:
    """Append UTF-8 lines to a file readable only by the owner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        path.chmod(0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            descriptor = -1
            for line in lines:
                handle.write(line + "\n")
    finally:
        if descriptor >= 0:  # pragma: no cover - defensive close on fdopen failure
            os.close(descriptor)


class FlatFileSpanExporter(SpanExporter):
    """Write a slim overview log and full daily trace files."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.slim_path = self.log_dir / _SLIM_TRACE_FILE_NAME
        self.trace_dir = self.log_dir / _TRACE_DIRECTORY_NAME

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            slim_lines: list[str] = []
            daily_lines: dict[str, list[str]] = defaultdict(list)
            for span in spans:
                slim_lines.append(
                    json.dumps(
                        _span_record(span, include_content=False),
                        default=str,
                        ensure_ascii=False,
                    )
                )
                daily_lines[_span_end_day(span)].append(
                    json.dumps(
                        _span_record(span, include_content=True),
                        default=str,
                        ensure_ascii=False,
                    )
                )
            if slim_lines:
                _append_private_lines(self.slim_path, slim_lines)
            for day, lines in daily_lines.items():
                _append_private_lines(self.trace_dir / f"{day}.jsonl", lines)
            return SpanExportResult.SUCCESS
        except Exception as e:
            print(f"⚠️ Failed to write traces under {self.log_dir}: {e}")
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


def _create_tracer_provider(log_dir: Path) -> tuple[TracerProvider, str]:
    """Create a provider with the flat-file exporter and optional OTLP export."""
    provider = TracerProvider(resource=Resource.create())
    provider.add_span_processor(BatchSpanProcessor(FlatFileSpanExporter(log_dir)))

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


def setup_tracing() -> TracerProvider:
    """Set up the local flat-file tracer provider.

    The provider is installed even when the log directory cannot be written.
    Leaving the global provider unset would let ADK install an in-memory
    exporter that retains every content-bearing span forever.
    """
    log_dir = get_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / _TRACE_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"⚠️ Failed to create trace directory: {e}")

    provider, mode = _create_tracer_provider(log_dir)
    trace.set_tracer_provider(provider)
    logger.info("Trace exporter mode configured: %s", mode)
    logger.info(
        "ADK span content capture is %s",
        "enabled" if adk_span_content_capture_enabled() else "disabled",
    )
    return provider


def shutdown_tracing(provider: TracerProvider | None) -> None:
    """Flush and release an OpenTelemetry provider when one was configured."""
    if provider is None:
        return
    provider.shutdown()
