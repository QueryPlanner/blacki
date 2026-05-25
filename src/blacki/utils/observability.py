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
from datetime import UTC, datetime
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


def setup_tracing() -> None:
    """Set up OpenTelemetry tracing with local JSON file export."""
    log_dir = get_log_dir()
    log_path = log_dir / "blacki-traces.log"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"⚠️ Failed to create trace directory: {e}")
        return

    # Resource automatically reads OTEL_RESOURCE_ATTRIBUTES from environment variables
    resource = Resource.create()

    # Set up tracer provider
    provider = TracerProvider(resource=resource)

    # Add our custom JSON exporter
    exporter = JSONFileSpanExporter(str(log_path))
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Register global tracer provider
    trace.set_tracer_provider(provider)
    print(f"✅ Local trace export configured to: {log_path}")
