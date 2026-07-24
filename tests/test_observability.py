"""Tests for local and OTLP observability configuration."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import grpc
import pytest
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)

from blacki.utils.exceptions import ConfigurationError
from blacki.utils.observability import (
    JSONFileSpanExporter,
    JSONFormatter,
    _create_tracer_provider,
    _load_otlp_trace_config,
    configure_otel_resource,
    get_log_dir,
    setup_logging,
    setup_tracing,
)

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

    def Export(  # noqa: N802
        self,
        request: trace_service_pb2.ExportTraceServiceRequest,
        context: grpc.ServicerContext,
    ) -> trace_service_pb2.ExportTraceServiceResponse:
        self.requests.append(request)
        return trace_service_pb2.ExportTraceServiceResponse()


def _span_names(collector: RecordingCollector) -> list[str]:
    return [
        span.name
        for request in collector.requests
        for resource_spans in request.resource_spans
        for scope_spans in resource_spans.scope_spans
        for span in scope_spans.spans
    ]


def test_local_only_provider_writes_span_without_otlp(
    tmp_path: Path,
) -> None:
    """Default mode writes local JSON and never constructs a network exporter."""
    log_path = tmp_path / "traces.jsonl"

    with patch("blacki.utils.observability.OTLPSpanExporter") as mock_otlp_exporter:
        provider, mode = _create_tracer_provider(log_path)
        tracer = provider.get_tracer("test.local")
        with tracer.start_as_current_span("agent.local"):
            pass
        assert provider.force_flush()
        provider.shutdown()

    assert mode == "local"
    mock_otlp_exporter.assert_not_called()
    written = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [span["name"] for span in written] == ["agent.local"]


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
    provider, mode = _create_tracer_provider(tmp_path / "traces.jsonl")
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
    assert config.protocol == "grpc"
    assert config.headers == "trace=value"


def test_otlp_protocol_defaults_to_grpc(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endpoint without an explicit protocol uses the installed gRPC exporter."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example:4317",
    )

    config = _load_otlp_trace_config()

    assert config is not None
    assert config.protocol == "grpc"


@pytest.mark.parametrize(
    "endpoint",
    [
        "not-a-url",
        "ftp://collector.example:4317",
        "https://user:password@collector.example:4317",
        "https://collector.example:4317/path?token=endpoint-canary",
        "https://collector.example:invalid-port",
    ],
)
def test_invalid_otlp_endpoint_is_rejected_without_echoing_value(
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed and credential-bearing endpoints fail with secret-free text."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)

    with pytest.raises(ConfigurationError) as error:
        _load_otlp_trace_config()

    assert endpoint not in str(error.value)
    assert "password" not in str(error.value)
    assert "endpoint-canary" not in str(error.value)


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


def test_unsupported_otlp_protocol_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed exporter cannot claim HTTP/protobuf support."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example:4317",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    with pytest.raises(ConfigurationError, match="supports grpc"):
        _load_otlp_trace_config()


def test_setup_tracing_logs_only_selected_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup logs exporter mode once without endpoint or header material."""
    endpoint_canary = "collector-canary.example"
    header_canary = "header-credential-canary"
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"https://{endpoint_canary}:4317",
    )
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        f"authorization={header_canary}",
    )
    caplog.set_level(logging.INFO)

    with (
        patch("blacki.utils.observability.get_log_dir", return_value=tmp_path),
        patch("blacki.utils.observability.trace.set_tracer_provider"),
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
    assert "local+otlp-grpc" in mode_records[0].message
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
    with patch("blacki.utils.observability.get_log_dir", return_value=tmp_path):
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
        patch("blacki.utils.observability.get_log_dir", return_value=blocked),
        patch.object(Path, "mkdir", side_effect=OSError("permission denied")),
    ):
        setup_logging("INFO")

    assert "Continuing with stdout logging only" in capsys.readouterr().out


def test_setup_tracing_returns_none_for_unwritable_directory() -> None:
    """Tracing should degrade safely when its directory cannot be created."""
    blocked = Path("/blocked")
    with (
        patch("blacki.utils.observability.get_log_dir", return_value=blocked),
        patch.object(Path, "mkdir", side_effect=OSError("permission denied")),
    ):
        provider = setup_tracing()

    assert provider is None


def test_json_span_exporter_reports_write_failure(tmp_path: Path) -> None:
    """Local exporter failure is reported without crashing the application."""
    exporter = JSONFileSpanExporter(str(tmp_path))

    assert exporter.export([]).name == "FAILURE"
    exporter.shutdown()
