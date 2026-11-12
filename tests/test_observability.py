"""Tests for local observability configuration."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from blacki.observability.setup import (
    FlatFileSpanExporter,
    JSONFormatter,
    _create_tracer_provider,
    _span_end_day,
    adk_span_content_capture_enabled,
    configure_otel_resource,
    get_log_dir,
    setup_logging,
    setup_tracing,
    shutdown_tracing,
)


@pytest.fixture(autouse=True)
def clear_otlp_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep exporter-warning tests independent from the developer's shell."""
    for name in [name for name in os.environ if name.startswith("OTEL_EXPORTER_OTLP_")]:
        monkeypatch.delenv(name, raising=False)


def test_observability_setup_has_stable_import_path() -> None:
    """Deployment preflight must import the relocated setup module."""
    module = importlib.import_module("blacki.observability.setup")

    assert module.setup_tracing is setup_tracing
    assert module.FlatFileSpanExporter is FlatFileSpanExporter


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


def test_local_provider_writes_span(tmp_path: Path) -> None:
    """The provider writes slim and full local traces without a network exporter."""
    provider = _create_tracer_provider(tmp_path)
    tracer = provider.get_tracer("test.local")
    with tracer.start_as_current_span("agent.local"):
        pass
    assert provider.force_flush()
    provider.shutdown()

    slim = [
        json.loads(line)
        for line in (tmp_path / "blacki-traces.log").read_text().splitlines()
    ]
    assert [span["name"] for span in slim] == ["agent.local"]
    full_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(full_files) == 1
    full = [json.loads(line) for line in full_files[0].read_text().splitlines()]
    assert [span["name"] for span in full] == ["agent.local"]


def test_setup_tracing_warns_about_ignored_otlp_settings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote exporter settings are named but never logged with their values."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://collector.example/endpoint-canary",
    )
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "authorization=header-credential-canary",
    )
    caplog.set_level(logging.WARNING)

    with (
        patch("blacki.observability.setup.get_log_dir", return_value=tmp_path),
        patch("blacki.observability.setup.trace.set_tracer_provider"),
    ):
        provider = setup_tracing()

    provider.shutdown()
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in caplog.text
    assert "OTEL_EXPORTER_OTLP_HEADERS" in caplog.text
    assert "endpoint-canary" not in caplog.text
    assert "header-credential-canary" not in caplog.text


def test_setup_tracing_stays_quiet_without_otlp_settings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Local-only startup does not emit a remote-exporter warning."""
    caplog.set_level(logging.WARNING)

    with (
        patch("blacki.observability.setup.get_log_dir", return_value=tmp_path),
        patch("blacki.observability.setup.trace.set_tracer_provider"),
    ):
        provider = setup_tracing()

    provider.shutdown()
    assert "Ignoring remote OTLP" not in caplog.text


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
