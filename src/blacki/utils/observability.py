"""OpenTelemetry and Logging setup for bare-metal adaptation.

This module provides consolidated observability configuration using standard
OpenTelemetry environment variables for vendor-neutral operation.
"""

import json
import logging
import os
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from opentelemetry import trace
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


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


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
    # We use ./logs locally and /app/logs inside Docker, so check both
    log_dir = "/app/logs" if Path("/.dockerenv").exists() else "./logs"
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "blacki-telemetry.log")
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
                        "kind": span.kind.name if span.kind else None,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
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
                                "timestamp": event.timestamp,
                                "attributes": dict(event.attributes)
                                if event.attributes
                                else {},
                            }
                            for event in span.events
                        ]
                        if span.events
                        else [],
                    }
                    f.write(json.dumps(span_data) + "\n")
            return SpanExportResult.SUCCESS
        except Exception as e:
            print(f"⚠️ Failed to write trace to {self.log_path}: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def setup_tracing() -> None:
    """Set up OpenTelemetry tracing with local JSON file export."""
    log_dir = "/app/logs" if Path("/.dockerenv").exists() else "./logs"
    log_path = Path(log_dir) / "blacki-traces.log"

    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"⚠️ Failed to create trace directory: {e}")
        return

    # Extract resource attributes if they exist
    resource_attrs = {}
    if "OTEL_RESOURCE_ATTRIBUTES" in os.environ:
        for pair in os.environ["OTEL_RESOURCE_ATTRIBUTES"].split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                resource_attrs[key] = value

    resource = Resource.create(resource_attrs)

    # Set up tracer provider
    provider = TracerProvider(resource=resource)

    # Add our custom JSON exporter
    exporter = JSONFileSpanExporter(str(log_path))
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Register global tracer provider
    trace.set_tracer_provider(provider)
    print(f"✅ Local trace export configured to: {log_path}")
