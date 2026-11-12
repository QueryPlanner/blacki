"""Tests for local and OTLP observability configuration."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from threading import Thread
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import grpc
import pytest
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from blacki.observability.setup import (
    FlatFileSpanExporter,
    JSONFormatter,
    _create_tracer_provider,
    _load_otlp_trace_config,
    _span_end_day,
    adk_span_content_capture_enabled,
    configure_otel_resource,
    get_log_dir,
    setup_logging,
    setup_tracing,
    shutdown_tracing,
    validate_observability_environment,
)
from blacki.utils.exceptions import ConfigurationError

OTLP_ENVIRONMENT_KEYS = (
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_HEADERS",
)


@pytest.fixture(autouse=True)
def clear_otlp_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every exporter test independent from the developer's shell."""
    for key in OTLP_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_observability_setup_has_stable_import_path() -> None:
    """Deployment preflight must import the relocated setup module."""
    module = importlib.import_module("blacki.observability.setup")

    assert (
        module.validate_observability_environment is validate_observability_environment
    )


def test_shutdown_tracing_handles_missing_and_configured_providers() -> None:
    """Tracing teardown should be safe when setup could not create a provider."""
    provider = MagicMock()

    shutdown_tracing(None)
    shutdown_tracing(provider)

    provider.shutdown.assert_called_once_with()


@pytest.fixture
def preserve_logging_configuration() -> Iterator[None]:
    """Restore pytest's logging handlers after forceful setup tests."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    urllib3_logger = logging.getLogger("urllib3")
    original_urllib3_level = urllib3_logger.level
    try:
        yield
    finally:
        for handler in root_logger.handlers:
            if handler not in original_handlers:
                handler.close()
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        urllib3_logger.setLevel(original_urllib3_level)


class RecordingCollector(trace_service_pb2_grpc.TraceServiceServicer):
    """Minimal local gRPC collector that records export requests."""

    def __init__(self) -> None:
        self.requests: list[trace_service_pb2.ExportTraceServiceRequest] = []
        self.authorizations: list[str | bytes | None] = []

    def Export(  # noqa: N802
        self,
        request: trace_service_pb2.ExportTraceServiceRequest,
        context: grpc.ServicerContext,
    ) -> trace_service_pb2.ExportTraceServiceResponse:
        self.requests.append(request)
        metadata = dict(context.invocation_metadata())
        self.authorizations.append(metadata.get("authorization"))
        return trace_service_pb2.ExportTraceServiceResponse()


class RecordingHttpHandler(BaseHTTPRequestHandler):
    """Minimal local HTTP/protobuf collector that records export requests."""

    requests: ClassVar[
        list[
            tuple[
                str, str | None, str | None, trace_service_pb2.ExportTraceServiceRequest
            ]
        ]
    ] = []

    def do_POST(self) -> None:  # noqa: N802
        """Decode and record one OTLP HTTP/protobuf export request."""
        content_length = int(self.headers["Content-Length"])
        request = trace_service_pb2.ExportTraceServiceRequest.FromString(
            self.rfile.read(content_length)
        )
        self.requests.append(
            (
                self.path,
                self.headers.get("Content-Type"),
                self.headers.get("Authorization"),
                request,
            )
        )
        self.send_response(200)
        self.end_headers()

    def log_message(self, _format: str, *args: object) -> None:
        """Keep the fake collector silent during tests."""
        _ = args


def _span_names(collector: RecordingCollector) -> list[str]:
    return [
        span.name
        for request in collector.requests
        for resource_spans in request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]


def _request_span_names(
    requests: list[trace_service_pb2.ExportTraceServiceRequest],
) -> list[str]:
    """Return every span name recorded by HTTP or gRPC requests."""
    return [
        span.name
        for request in requests
        for resource_spans in request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]


def test_local_only_provider_writes_span_without_otlp(
    tmp_path: Path,
) -> None:
    """Default mode writes slim and full local traces without a network exporter."""
    with (
        patch("blacki.observability.setup.GrpcOTLPSpanExporter") as mock_grpc_exporter,
        patch("blacki.observability.setup.HttpOTLPSpanExporter") as mock_http_exporter,
    ):
        provider, mode = _create_tracer_provider(tmp_path)
        tracer = provider.get_tracer("test.local")
        with tracer.start_as_current_span("agent.local"):
            pass
        assert provider.force_flush()
        provider.shutdown()

    assert mode == "local"
    mock_grpc_exporter.assert_not_called()
    mock_http_exporter.assert_not_called()
    slim = [
        json.loads(line)
        for line in (tmp_path / "blacki-traces.log").read_text().splitlines()
    ]
    assert [span["name"] for span in slim] == ["agent.local"]
    full_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(full_files) == 1
    full = [json.loads(line) for line in full_files[0].read_text().splitlines()]
    assert [span["name"] for span in full] == ["agent.local"]


def test_otlp_export_reaches_local_grpc_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A representative agent span reaches a real local OTLP gRPC service."""
    collector = RecordingCollector()
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(collector, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"http://127.0.0.1:{port}",
    )
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "authorization=Bearer%20grpc-token",
    )
    provider, mode = _create_tracer_provider(tmp_path)
    try:
        tracer = provider.get_tracer("blacki.agent")
        with tracer.start_as_current_span("agent.turn") as span:
            span.set_attribute("blacki.channel", "telegram")
        assert provider.force_flush(timeout_millis=5_000)
    finally:
        provider.shutdown()
        server.stop(grace=None).wait(timeout=5)

    assert mode == "local+otlp-grpc"
    assert "agent.turn" in _span_names(collector)
    assert collector.authorizations == ["Bearer grpc-token"]


def test_otlp_export_reaches_local_http_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A representative span reaches a real OTLP HTTP/protobuf endpoint."""
    RecordingHttpHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingHttpHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"http://127.0.0.1:{port}/custom/v1/traces",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "authorization=Bearer%20test-token",
    )
    provider, mode = _create_tracer_provider(tmp_path)
    try:
        tracer = provider.get_tracer("blacki.agent")
        with tracer.start_as_current_span("agent.http-turn"):
            pass
        assert provider.force_flush(timeout_millis=5_000)
    finally:
        provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert mode == "local+otlp-http-protobuf"
    assert len(RecordingHttpHandler.requests) == 1
    path, content_type, authorization, request = RecordingHttpHandler.requests[0]
    assert path == "/custom/v1/traces"
    assert content_type == "application/x-protobuf"
    assert authorization == "Bearer test-token"
    assert "agent.http-turn" in _request_span_names([request])


def test_trace_specific_otlp_values_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace-specific endpoint, protocol, and headers override global values."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://global.example:4317")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "https://traces.example:4317",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "global=value")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "trace=value")

    config = _load_otlp_trace_config()

    assert config is not None
    assert config.endpoint == "https://traces.example:4317"
    assert config.endpoint_is_trace_specific
    assert config.protocol == "grpc"
    assert config.headers == {"trace": "value"}


def test_otlp_protocol_defaults_to_grpc(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endpoint without an explicit protocol uses the installed gRPC exporter."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example:4317",
    )

    config = _load_otlp_trace_config()

    assert config is not None
    assert config.protocol == "grpc"


def test_http_global_endpoint_appends_trace_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A global HTTP endpoint is a base URL and receives the signal path."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example/otlp/",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    config = _load_otlp_trace_config()

    assert config is not None
    assert config.endpoint == "https://collector.example/otlp/v1/traces"
    assert not config.endpoint_is_trace_specific
    assert config.protocol == "http/protobuf"


def test_http_trace_endpoint_is_used_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trace-specific HTTP endpoint already includes its complete path."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "https://collector.example/custom/traces",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    config = _load_otlp_trace_config()

    assert config is not None
    assert config.endpoint == "https://collector.example/custom/traces"
    assert config.endpoint_is_trace_specific


def test_otlp_headers_are_parsed_for_both_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment headers become the mapping required by the HTTP exporter."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example:4318",
    )
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "authorization=Bearer%20token,x-api-key=value%3Dwith-equals",
    )

    config = _load_otlp_trace_config()

    assert config is not None
    assert config.headers == {
        "authorization": "Bearer token",
        "x-api-key": "value=with-equals",
    }


def test_observability_preflight_constructs_and_closes_selected_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deployment validation proves the selected exporter can be constructed."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example:4318",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    with (
        patch("blacki.observability.setup.GrpcOTLPSpanExporter") as mock_grpc_exporter,
        patch("blacki.observability.setup.HttpOTLPSpanExporter") as mock_http_exporter,
    ):
        validate_observability_environment()

    mock_grpc_exporter.assert_not_called()
    mock_http_exporter.assert_called_once_with(
        endpoint="https://collector.example:4318/v1/traces",
        headers=None,
    )
    mock_http_exporter.return_value.shutdown.assert_called_once_with()


def test_observability_preflight_local_mode_constructs_no_exporter() -> None:
    """Local-only preflight has no network-exporter lifecycle."""
    with (
        patch("blacki.observability.setup.GrpcOTLPSpanExporter") as mock_grpc_exporter,
        patch("blacki.observability.setup.HttpOTLPSpanExporter") as mock_http_exporter,
    ):
        validate_observability_environment()

    mock_grpc_exporter.assert_not_called()
    mock_http_exporter.assert_not_called()


@pytest.mark.parametrize(
    ("protocol", "endpoint"),
    [
        ("grpc", "not-a-url"),
        ("grpc", "ftp://collector.example:4317"),
        ("grpc", "https://user:password@collector.example:4317"),
        ("grpc", "https://collector.example:4317/path"),
        ("http/protobuf", "https://collector.example/path;token=endpoint-canary"),
        (
            "http/protobuf",
            "https://collector.example/path?token=endpoint-canary",
        ),
        ("http/protobuf", "https://collector.example:invalid-port"),
    ],
)
def test_invalid_otlp_endpoint_is_rejected_without_echoing_value(
    protocol: str,
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed and credential-bearing endpoints fail with secret-free text."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", protocol)

    with pytest.raises(ConfigurationError) as error:
        _load_otlp_trace_config()

    formatted_error = "".join(
        traceback.format_exception(error.type, error.value, error.tb)
    )
    assert endpoint not in formatted_error
    assert "password" not in formatted_error
    assert "endpoint-canary" not in formatted_error


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"),
        ("OTEL_EXPORTER_OTLP_HEADERS", "authorization=header-canary"),
    ],
)
def test_otlp_options_without_endpoint_fail_startup(
    key: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protocol or headers alone must not silently create a default network path."""
    monkeypatch.setenv(key, value)

    with pytest.raises(ConfigurationError) as error:
        _load_otlp_trace_config()

    assert value not in str(error.value)
    assert "require" in str(error.value)


@pytest.mark.parametrize(
    "header_canary",
    [
        "authorization header-credential-canary",
        "invalid header=header-credential-canary",
        "authorization=Bearer%20header-credential-canary%0D%0AX-Injected%3Ayes",
    ],
)
def test_malformed_otlp_headers_fail_without_echoing_value(
    header_canary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed headers fail startup without exposing credential material."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example:4317",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", header_canary)

    with pytest.raises(ConfigurationError, match="Invalid OTLP trace headers") as error:
        _load_otlp_trace_config()

    formatted_error = "".join(
        traceback.format_exception(error.type, error.value, error.tb)
    )
    assert header_canary not in formatted_error
    assert "header-credential-canary" not in formatted_error


@pytest.mark.parametrize("protocol", ["http/json", "json", "grpc/protobuf"])
def test_unsupported_otlp_protocol_fails_startup(
    protocol: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the two installed OTLP transports may be selected."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example:4317",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", protocol)

    with pytest.raises(ConfigurationError, match="grpc and http/protobuf"):
        _load_otlp_trace_config()


@pytest.mark.parametrize(
    ("protocol", "endpoint_suffix", "expected_mode"),
    [
        ("grpc", ":4317", "local+otlp-grpc"),
        ("http/protobuf", ":4318/v1/traces", "local+otlp-http-protobuf"),
    ],
)
def test_setup_tracing_logs_only_selected_mode(
    protocol: str,
    endpoint_suffix: str,
    expected_mode: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup logs exporter mode once without endpoint or header material."""
    endpoint_canary = "collector-canary.example"
    header_canary = "header-credential-canary"
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"https://{endpoint_canary}{endpoint_suffix}",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", protocol)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        f"authorization={header_canary}",
    )
    caplog.set_level(logging.INFO)

    with (
        patch("blacki.observability.setup.get_log_dir", return_value=tmp_path),
        patch("blacki.observability.setup.trace.set_tracer_provider"),
    ):
        provider = setup_tracing()

    assert provider is not None
    provider.shutdown()
    mode_records = [
        record
        for record in caplog.records
        if "Trace exporter mode configured" in record.message
    ]
    assert len(mode_records) == 1
    assert expected_mode in mode_records[0].message
    assert endpoint_canary not in caplog.text
    assert header_canary not in caplog.text


def test_observability_formatters_and_resource_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resource attributes and JSON log records remain structured."""
    monkeypatch.setenv("TELEMETRY_NAMESPACE", "tests")
    monkeypatch.setenv("K_REVISION", "revision-1")
    configure_otel_resource("blacki-test")

    attributes = os.getenv("OTEL_RESOURCE_ATTRIBUTES")
    assert attributes is not None
    assert "service.name=blacki-test" in attributes
    assert "service.namespace=tests" in attributes
    assert "service.version=revision-1" in attributes

    record = logging.LogRecord(
        name="blacki.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    formatted = json.loads(JSONFormatter().format(record))
    assert formatted["message"] == "hello"
    assert formatted["name"] == "blacki.test"


def test_json_formatter_includes_exception_details() -> None:
    """Structured logs should retain exception type and message."""
    try:
        raise ValueError("formatting failed")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="blacki.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )

    formatted = json.loads(JSONFormatter().format(record))

    assert "ValueError: formatting failed" in formatted["exception"]


@pytest.mark.parametrize(
    ("in_container", "expected"),
    [
        (True, Path("/app/logs")),
        (False, Path("./logs")),
    ],
)
def test_get_log_dir_selects_container_or_local_path(
    in_container: bool,
    expected: Path,
) -> None:
    """Log paths should follow the runtime environment."""
    with patch.object(Path, "exists", return_value=in_container):
        assert get_log_dir() == expected


def test_setup_logging_adds_json_file_handler(
    tmp_path: Path,
    preserve_logging_configuration: None,
) -> None:
    """A writable log directory should receive structured file logs."""
    with patch("blacki.observability.setup.get_log_dir", return_value=tmp_path):
        setup_logging("DEBUG")
        logging.getLogger("blacki.test").info("file message")
        for handler in logging.getLogger().handlers:
            handler.flush()

    log_path = tmp_path / "blacki-telemetry.log"
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[-1]["message"] == "file message"
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("urllib3").level == logging.WARNING


def test_setup_logging_falls_back_to_stdout(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    preserve_logging_configuration: None,
) -> None:
    """An unavailable local log directory does not prevent process logging."""
    blocked = Path("/blocked")
    with (
        patch("blacki.observability.setup.get_log_dir", return_value=blocked),
        patch.object(Path, "mkdir", side_effect=OSError("permission denied")),
    ):
        setup_logging("INFO")

    assert "Continuing with stdout logging only" in capsys.readouterr().out


def test_setup_tracing_installs_provider_for_unwritable_directory() -> None:
    """Tracing still installs a provider so ADK cannot keep spans in memory."""
    blocked = Path("/blocked")
    with (
        patch("blacki.observability.setup.get_log_dir", return_value=blocked),
        patch.object(Path, "mkdir", side_effect=OSError("permission denied")),
        patch("blacki.observability.setup.trace.set_tracer_provider"),
    ):
        provider = setup_tracing()

    assert provider is not None
    provider.shutdown()


def test_flat_file_exporter_reports_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local exporter failure is reported without crashing the application."""

    def fail_write(_path: Path, _lines: list[str]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("blacki.observability.setup._append_private_lines", fail_write)
    exporter = FlatFileSpanExporter(tmp_path)
    span = _make_span("agent.failure")

    assert exporter.export([span]).name == "FAILURE"
    exporter.shutdown()


def _make_span(name: str) -> ReadableSpan:
    """Create a completed in-memory span for exporter unit tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test.exporter")
    with tracer.start_as_current_span(name):
        pass
    spans = exporter.get_finished_spans()
    provider.shutdown()
    assert len(spans) == 1
    return spans[0]


def _make_span_with(
    name: str,
    attributes: dict[str, Any] | None = None,
    events: list[tuple[str, dict[str, Any]]] | None = None,
) -> ReadableSpan:
    """Create a completed span carrying the supplied attributes and events."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test.exporter")
    with tracer.start_as_current_span(name) as span:
        if attributes:
            span.set_attributes(attributes)
        for event_name, event_attributes in events or []:
            span.add_event(event_name, attributes=event_attributes)
    spans = exporter.get_finished_spans()
    provider.shutdown()
    return spans[0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("0", False),
        ("false", False),
        ("FALSE", False),
    ],
)
def test_adk_span_content_capture_defaults_on(
    value: str | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADK span content capture is opt-out and on by default."""
    if value is None:
        monkeypatch.delenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", raising=False)
    else:
        monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", value)

    assert adk_span_content_capture_enabled() is expected


def test_exporter_keeps_slim_and_full_traces_separate(tmp_path: Path) -> None:
    """The slim overview omits content while the daily full trace retains it."""
    exporter = FlatFileSpanExporter(tmp_path)
    span = _make_span_with(
        "call_llm",
        {
            "gen_ai.request.model": "openrouter/test",
            "gcp.vertex.agent.llm_request": '{"contents": "secret prompt"}',
            "gen_ai.tool.name": "should-not-be-allowlisted",
        },
        [("content.event", {"payload": "secret event"})],
    )

    assert exporter.export([span]).name == "SUCCESS"

    slim = json.loads((tmp_path / "blacki-traces.log").read_text())
    assert slim["attributes"] == {
        "gen_ai.request.model": "openrouter/test",
        "gen_ai.tool.name": "should-not-be-allowlisted",
    }
    assert slim["events"] == []
    assert "secret" not in json.dumps(slim)

    full_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(full_files) == 1
    full = json.loads(full_files[0].read_text())
    assert full["attributes"]["gcp.vertex.agent.llm_request"] == (
        '{"contents": "secret prompt"}'
    )
    assert full["events"] == [
        {
            "name": "content.event",
            "timestamp": full["events"][0]["timestamp"],
            "attributes": {"payload": "secret event"},
        }
    ]


def test_trace_files_are_owner_readable_only(tmp_path: Path) -> None:
    """Trace and slim files are created with mode 0600."""
    exporter = FlatFileSpanExporter(tmp_path)
    exporter.export([_make_span("agent.private")])

    slim_mode = (tmp_path / "blacki-traces.log").stat().st_mode & 0o777
    full_file = next((tmp_path / "traces").glob("*.jsonl"))
    assert slim_mode == 0o600
    assert full_file.stat().st_mode & 0o777 == 0o600


def test_daily_trace_file_uses_utc_end_date(tmp_path: Path) -> None:
    """Full traces are grouped into UTC-dated files."""
    exporter = FlatFileSpanExporter(tmp_path)
    span = _make_span("agent.dated")
    exporter.export([span])

    end = span.end_time
    assert end is not None
    expected = datetime.fromtimestamp(end / 1e9, tz=UTC).date().isoformat()
    assert (tmp_path / "traces" / f"{expected}.jsonl").is_file()


def test_json_formatter_redacts_and_correlates_trace_ids() -> None:
    """Structured logs redact credentials and include the active trace IDs."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test.logging")
    record = logging.LogRecord(
        name="blacki.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=super-secret-value",
        args=(),
        exc_info=None,
    )
    with tracer.start_as_current_span("log.parent"):
        formatted = json.loads(JSONFormatter().format(record))
        context = trace.get_current_span().get_span_context()
    provider.shutdown()

    assert formatted["message"] == "token=[REDACTED]"
    assert formatted["trace_id"] == format(context.trace_id, "032x")
    assert formatted["span_id"] == format(context.span_id, "016x")


def test_setup_logging_rotates_at_utc_midnight(
    tmp_path: Path,
    preserve_logging_configuration: None,
) -> None:
    """The file handler rotates daily in UTC and keeps fourteen backups."""
    with patch("blacki.observability.setup.get_log_dir", return_value=tmp_path):
        setup_logging("DEBUG")

    handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, TimedRotatingFileHandler)
    ]
    assert len(handlers) == 1
    handler = handlers[0]
    assert handler.when == "MIDNIGHT"
    assert handler.utc is True
    assert handler.backupCount == 14
    assert Path(handler.baseFilename) == tmp_path / "blacki-telemetry.log"


def test_span_end_day_falls_back_to_current_utc_date() -> None:
    """A span without timestamps is filed under today's UTC date."""
    from types import SimpleNamespace

    span = SimpleNamespace(end_time=None, start_time=None)
    expected = datetime.now(tz=UTC).date().isoformat()
    assert _span_end_day(span) == expected  # type: ignore[arg-type]


def test_exporter_handles_empty_batches(tmp_path: Path) -> None:
    """An empty export is a success and writes no files."""
    exporter = FlatFileSpanExporter(tmp_path)

    assert exporter.export([]).name == "SUCCESS"
    assert not (tmp_path / "blacki-traces.log").exists()


def test_redaction_filter_rewrites_messages_with_secrets() -> None:
    """The logging filter rewrites a rendered record before a handler writes it."""
    from blacki.observability.redaction import RedactionFilter

    record = logging.LogRecord(
        name="blacki.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="api_key=%s",
        args=("sk-or-v1-" + "a" * 24,),
        exc_info=None,
    )

    assert RedactionFilter().filter(record) is True
    assert record.args == ()
    assert "sk-or-v1-" not in record.msg
    assert "[REDACTED]" in record.msg


def test_redaction_filter_keeps_unparsable_records() -> None:
    """A record whose arguments cannot be formatted is passed through untouched."""
    from blacki.observability.redaction import RedactionFilter

    record = logging.LogRecord(
        name="blacki.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="%d",
        args=("not-an-int",),
        exc_info=None,
    )

    assert RedactionFilter().filter(record) is True


def test_iso_timestamp_returns_none_for_missing_nanoseconds() -> None:
    """A span without a timestamp maps to a null ISO value."""
    from blacki.observability.setup import _iso_timestamp

    assert _iso_timestamp(None) is None
