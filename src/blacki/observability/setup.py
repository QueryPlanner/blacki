"""OpenTelemetry and logging setup.

Blacki uses ADK's OpenTelemetry setup (``google.adk.telemetry.setup``) with the
experimental GenAI semantic conventions, so every model call carries its system
instructions, input messages, output messages, and tool definitions as
standard ``gen_ai.*`` span attributes.

The OpenTelemetry SDK console exporters write two local JSON Lines files:

* ``blacki-traces.log``: every span, unfiltered.
* ``blacki-telemetry.log``: every log record, unfiltered.

Both files are created with mode ``0600`` and are never rotated or trimmed.
ADK also adds its OTLP exporters when ``OTEL_EXPORTER_OTLP_*`` is set, which
requires installing an OTLP exporter package first.
"""

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from google.adk.telemetry.setup import OTelHooks, maybe_set_otel_providers
from opentelemetry import _logs, trace
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogRecordExporter,
)
from opentelemetry.sdk.resources import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_NAMESPACE,
    SERVICE_VERSION,
    Resource,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger("blacki.utils.observability")

LOG_FILE_NAME = "blacki-telemetry.log"
TRACE_FILE_NAME = "blacki-traces.log"

OTEL_SEMCONV_STABILITY_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)
ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"
GENAI_EXPERIMENTAL_OPT_IN = "gen_ai_latest_experimental"

# Defaults applied unless the operator already set the variable.  Content goes
# on spans in the experimental GenAI format.  ADK's legacy content attributes
# would store a second copy of every request, so they are off.
GENAI_TELEMETRY_DEFAULTS = {
    OTEL_SEMCONV_STABILITY_OPT_IN: GENAI_EXPERIMENTAL_OPT_IN,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY",
    ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS: "false",
}
_SPAN_CONTENT_MODES = frozenset({"SPAN_ONLY", "SPAN_AND_EVENT"})


@dataclass(frozen=True)
class TelemetryProviders:
    """The process-wide providers installed by :func:`setup_tracing`."""

    tracer: TracerProvider
    logger: LoggerProvider | None


def configure_otel_resource(agent_name: str) -> None:
    """Configure OpenTelemetry resource via environment variables.

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


def configure_genai_telemetry() -> None:
    """Apply the GenAI telemetry defaults without overriding operator values."""
    for name, value in GENAI_TELEMETRY_DEFAULTS.items():
        os.environ.setdefault(name, value)


def genai_content_capture_enabled() -> bool:
    """Return whether ADK writes GenAI content onto spans.

    The dashboard builds its model-call views from these attributes, so the
    views are empty when this is off.
    """
    opt_ins = os.getenv(OTEL_SEMCONV_STABILITY_OPT_IN, "")
    mode = os.getenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "")
    return (
        GENAI_EXPERIMENTAL_OPT_IN in {item.strip() for item in opt_ins.split(",")}
        and mode.strip().upper() in _SPAN_CONTENT_MODES
    )


def get_log_dir() -> Path:
    """Get the appropriate log directory based on environment."""
    return Path("/app/logs") if Path("/.dockerenv").exists() else Path("./logs")


def setup_logging(log_level: str) -> None:
    """Send Python logging to stdout at the configured level.

    :func:`setup_tracing` adds the OpenTelemetry handler that writes the same
    records to ``blacki-telemetry.log``.

    Args:
        log_level: Logging verbosity level as string
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.basicConfig(level=level, handlers=[console_handler], force=True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def span_line(span: ReadableSpan) -> str:
    """Format one span as the SDK's JSON on a single line."""
    return str(span.to_json(indent=None)) + "\n"


def log_line(record: ReadableLogRecord) -> str:
    """Format one log record as the SDK's JSON on a single line.

    The SDK's JSON leaves out the instrumentation scope, which holds the Python
    logger name, so it is added here.
    """
    data = json.loads(record.to_json(indent=None))
    scope = record.instrumentation_scope
    data["instrumentation_scope"] = (
        {"name": scope.name, "version": scope.version} if scope else None
    )
    return json.dumps(data, ensure_ascii=False) + "\n"


def open_private_append(path: Path) -> TextIO:
    """Open a UTF-8 file for appending, readable only by its owner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        path.chmod(0o600)
        return os.fdopen(descriptor, "a", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


class FileSpanExporter(ConsoleSpanExporter):
    """The SDK console span exporter, closing its file on shutdown."""

    def shutdown(self) -> None:
        self.out.close()


class FileLogRecordExporter(ConsoleLogRecordExporter):
    """The SDK console log exporter, closing its file on shutdown."""

    def shutdown(self) -> None:
        self.out.close()


def _otel_hooks(log_dir: Path) -> OTelHooks | None:
    """Build the file exporters, or return None when the files cannot open."""
    try:
        traces = open_private_append(log_dir / TRACE_FILE_NAME)
        try:
            logs = open_private_append(log_dir / LOG_FILE_NAME)
        except OSError:
            traces.close()
            raise
    except OSError as error:
        print(f"⚠️ Failed to open telemetry files under {log_dir}: {error}")
        return None
    return OTelHooks(
        span_processors=[
            BatchSpanProcessor(FileSpanExporter(out=traces, formatter=span_line))
        ],
        log_record_processors=[
            BatchLogRecordProcessor(FileLogRecordExporter(out=logs, formatter=log_line))
        ],
    )


def setup_tracing() -> TelemetryProviders:
    """Install ADK's OpenTelemetry providers with the local file exporters.

    A tracer provider is installed even when the files cannot be opened.
    Otherwise ADK's web server would install its own in-memory exporter, which
    keeps every span in memory.
    """
    configure_genai_telemetry()
    hooks = _otel_hooks(get_log_dir())
    try:
        maybe_set_otel_providers(
            [hooks] if hooks else [], otel_resource=Resource.create()
        )
    except ImportError as error:
        raise RuntimeError(
            "OTEL_EXPORTER_OTLP_* is set, but no OTLP exporter package is "
            "installed. Install opentelemetry-exporter-otlp-proto-http or unset "
            "the variables."
        ) from error

    tracer = trace.get_tracer_provider()
    if not isinstance(tracer, TracerProvider):
        tracer = TracerProvider(resource=Resource.create())
        trace.set_tracer_provider(tracer)
    current_log_provider = _logs.get_logger_provider()
    log_provider = (
        current_log_provider
        if isinstance(current_log_provider, LoggerProvider)
        else None
    )
    if log_provider is not None:
        logging.getLogger().addHandler(LoggingHandler(logger_provider=log_provider))

    logger.info("OpenTelemetry file exporters configured")
    if not genai_content_capture_enabled():
        logger.warning(
            "GenAI content capture is off, so the dashboard cannot show model "
            "calls. Set %s=%s and %s=SPAN_ONLY.",
            OTEL_SEMCONV_STABILITY_OPT_IN,
            GENAI_EXPERIMENTAL_OPT_IN,
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
        )
    return TelemetryProviders(tracer=tracer, logger=log_provider)


def shutdown_tracing(providers: TelemetryProviders | None) -> None:
    """Flush and release the providers installed by :func:`setup_tracing`."""
    if providers is None:
        return
    providers.tracer.shutdown()
    if providers.logger is not None:
        providers.logger.shutdown()
