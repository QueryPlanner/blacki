"""OpenTelemetry and logging setup.

Blacki uses ADK's OpenTelemetry setup (``google.adk.telemetry.setup``) with the
experimental GenAI semantic conventions, so every model call carries its system
instructions, input messages, output messages, and tool definitions as
standard ``gen_ai.*`` span attributes.

The OpenTelemetry SDK console exporters write two local JSON Lines files:

* ``blacki-traces.log``: OpenTelemetry spans.
* ``blacki-telemetry.log``: Python log records.

Each file rolls over at its byte limit and retains one private backup.
ADK also adds its OTLP exporters when ``OTEL_EXPORTER_OTLP_*`` is set, which
requires installing an OTLP exporter package first.
"""

import io
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TextIO, cast

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
LOG_FILE_MAX_BYTES = 16 * 1024 * 1024
TRACE_FILE_MAX_BYTES = 128 * 1024 * 1024

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
# ADK treats only these values of its legacy span-content flag as off.
_LEGACY_CAPTURE_OFF_VALUES = frozenset({"0", "false"})


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
    """Apply the GenAI telemetry defaults and keep legacy span content off.

    Operator values for the other variables are kept. The legacy content flag is
    the exception: ADK's legacy span format writes whole message parts, including
    encrypted reasoning, so it is forced off (design D11).
    """
    for name, value in GENAI_TELEMETRY_DEFAULTS.items():
        os.environ.setdefault(name, value)
    _force_legacy_span_content_off()


def _force_legacy_span_content_off() -> None:
    """Set ADK's legacy span-content flag to false, warning on an override."""
    requested = os.environ.get(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS)
    os.environ[ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS] = "false"
    if (
        requested is not None
        and requested.strip().lower() not in _LEGACY_CAPTURE_OFF_VALUES
    ):
        logger.warning(
            "%s is overridden to false. The legacy span format would export whole "
            "message parts, including encrypted reasoning.",
            ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS,
        )


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
    # Polling drivers log each database operation and HTTP exchange at DEBUG.
    logging.getLogger("aiosqlite").setLevel(max(level, logging.INFO))
    for name in ("httpcore", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


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


class RotatingJsonlFile(io.TextIOBase):
    """Write complete UTF-8 records with one owner-only rollover file.

    Each file has a single writer, its OpenTelemetry batch processor.
    """

    def __init__(self, path: Path, max_bytes: int) -> None:
        self._path = path
        self._backup = path.with_suffix(path.suffix + ".1")
        self._max_bytes = max_bytes
        self._file: TextIO | None = None
        self._file = open_private_append(path)
        self._size = self._file.tell()

    def write(self, text: str) -> int:
        size = len(text.encode("utf-8"))
        if size > self._max_bytes:
            # The independently configured OTLP exporter can keep this record.
            return len(text)
        if self._file is None:
            self._file = open_private_append(self._path)
            self._size = self._file.tell()
        if self._size + size > self._max_bytes:
            file, self._file = self._file, None
            file.close()
            self._path.replace(self._backup)
            self._file = open_private_append(self._path)
            self._size = 0
        written = self._file.write(text)
        self._size += size
        return written

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._file is not None:
                file, self._file = self._file, None
                file.close()


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
        traces = RotatingJsonlFile(log_dir / TRACE_FILE_NAME, TRACE_FILE_MAX_BYTES)
        try:
            logs = RotatingJsonlFile(log_dir / LOG_FILE_NAME, LOG_FILE_MAX_BYTES)
        except OSError:
            traces.close()
            raise
    except OSError as error:
        print(f"⚠️ Failed to open telemetry files under {log_dir}: {error}")
        return None
    return OTelHooks(
        span_processors=[
            BatchSpanProcessor(
                FileSpanExporter(out=cast("IO[str]", traces), formatter=span_line)
            )
        ],
        log_record_processors=[
            BatchLogRecordProcessor(
                FileLogRecordExporter(out=cast("IO[str]", logs), formatter=log_line)
            )
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
        handler = LoggingHandler(logger_provider=log_provider)
        # Export failures still reach stdout but cannot feed the failing exporter.
        handler.addFilter(
            lambda record: record.name != "opentelemetry.sdk._shared_internal"
        )
        logging.getLogger().addHandler(handler)

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
