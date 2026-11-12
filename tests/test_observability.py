"""Tests for the OpenTelemetry file exporters and GenAI telemetry defaults."""

from __future__ import annotations

import io
import json
import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider

from blacki.observability.setup import (
    ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS,
    GENAI_TELEMETRY_DEFAULTS,
    LOG_FILE_NAME,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_SEMCONV_STABILITY_OPT_IN,
    TRACE_FILE_NAME,
    FileLogRecordExporter,
    FileSpanExporter,
    TelemetryProviders,
    _otel_hooks,
    configure_genai_telemetry,
    configure_otel_resource,
    genai_content_capture_enabled,
    get_log_dir,
    open_private_append,
    setup_logging,
    setup_tracing,
    shutdown_tracing,
)

_SETUP = "blacki.observability.setup"


@pytest.fixture
def genai_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Start every test without GenAI telemetry variables from the shell."""
    for name in GENAI_TELEMETRY_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


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


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_genai_defaults_fill_only_unset_variables(
    genai_environment: pytest.MonkeyPatch,
) -> None:
    genai_environment.setenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "")
    configure_genai_telemetry()

    assert os.environ[OTEL_SEMCONV_STABILITY_OPT_IN] == "gen_ai_latest_experimental"
    assert os.environ[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] == ""
    assert os.environ[ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS] == "false"


@pytest.mark.parametrize("operator_value", ["true", "TRUE", " 1 ", "yes", ""])
def test_legacy_span_capture_is_forced_off_with_warning(
    genai_environment: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    operator_value: str,
) -> None:
    """An operator value that enables legacy span content is overridden."""
    genai_environment.setenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, operator_value)

    with caplog.at_level(logging.WARNING, logger="blacki.utils.observability"):
        configure_genai_telemetry()

    assert os.environ[ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS] == "false"
    assert "is overridden to false" in caplog.text
    assert "encrypted reasoning" in caplog.text


@pytest.mark.parametrize("operator_value", [None, "false", "FALSE", "0"])
def test_legacy_span_capture_off_values_are_kept_without_warning(
    genai_environment: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    operator_value: str | None,
) -> None:
    """Values that already disable legacy span content need no warning."""
    if operator_value is None:
        genai_environment.delenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, raising=False)
    else:
        genai_environment.setenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, operator_value)

    with caplog.at_level(logging.WARNING, logger="blacki.utils.observability"):
        configure_genai_telemetry()

    assert os.environ[ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS] == "false"
    assert "overridden" not in caplog.text


@pytest.mark.parametrize(
    ("opt_in", "mode", "expected"),
    [
        ("gen_ai_latest_experimental", "SPAN_ONLY", True),
        ("http, gen_ai_latest_experimental", "span_and_event", True),
        ("gen_ai_latest_experimental", "EVENT_ONLY", False),
        ("gen_ai_latest_experimental", "", False),
        ("", "SPAN_ONLY", False),
    ],
)
def test_genai_content_capture_needs_experimental_span_content(
    genai_environment: pytest.MonkeyPatch, opt_in: str, mode: str, expected: bool
) -> None:
    genai_environment.setenv(OTEL_SEMCONV_STABILITY_OPT_IN, opt_in)
    genai_environment.setenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, mode)

    assert genai_content_capture_enabled() is expected


def test_file_exporters_write_every_span_and_log_record(tmp_path: Path) -> None:
    hooks = _otel_hooks(tmp_path)
    assert hooks is not None

    tracer_provider = TracerProvider()
    for span_processor in hooks.span_processors:
        tracer_provider.add_span_processor(span_processor)
    logger_provider = LoggerProvider()
    for log_processor in hooks.log_record_processors:
        logger_provider.add_log_record_processor(log_processor)
    test_logger = logging.getLogger("blacki.test.exporters")
    test_logger.propagate = False
    test_logger.setLevel(logging.INFO)
    handler = LoggingHandler(logger_provider=logger_provider)
    test_logger.addHandler(handler)
    prompt = "password: example-secret\n" * 1000
    try:
        with tracer_provider.get_tracer("test").start_as_current_span("model") as span:
            span.set_attribute("gen_ai.system_instructions", prompt)
            test_logger.info("inside %s", "span")
    finally:
        test_logger.removeHandler(handler)
        tracer_provider.shutdown()
        logger_provider.shutdown()

    [span_record] = [
        json.loads(line)
        for line in (tmp_path / TRACE_FILE_NAME).read_text().splitlines()
    ]
    assert span_record["name"] == "model"
    assert span_record["attributes"]["gen_ai.system_instructions"] == prompt
    assert span_record["context"]["trace_id"].startswith("0x")
    [log_record] = [
        json.loads(line) for line in (tmp_path / LOG_FILE_NAME).read_text().splitlines()
    ]
    assert log_record["body"] == "inside span"
    assert log_record["severity_text"] == "INFO"
    assert log_record["trace_id"] == span_record["context"]["trace_id"]
    assert log_record["instrumentation_scope"]["name"] == "blacki.test.exporters"
    assert _mode(tmp_path / TRACE_FILE_NAME) == 0o600
    assert _mode(tmp_path / LOG_FILE_NAME) == 0o600


def test_private_append_tightens_existing_files_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "telemetry.log"
    path.parent.mkdir()
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o644)

    with open_private_append(path) as handle:
        handle.write("new\n")

    assert path.read_text(encoding="utf-8") == "old\nnew\n"
    assert _mode(path) == 0o600


def test_private_append_closes_descriptor_when_wrapping_fails(
    tmp_path: Path,
) -> None:
    with (
        patch(f"{_SETUP}.os.fdopen", side_effect=OSError("wrap failed")),
        patch(f"{_SETUP}.os.close", wraps=os.close) as close,
        pytest.raises(OSError, match="wrap failed"),
    ):
        open_private_append(tmp_path / "telemetry.log")

    close.assert_called_once()


def test_hooks_are_skipped_when_either_file_cannot_open(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")
    assert _otel_hooks(blocked) is None

    (tmp_path / LOG_FILE_NAME).mkdir()
    assert _otel_hooks(tmp_path) is None
    assert (tmp_path / TRACE_FILE_NAME).exists()


def test_setup_tracing_installs_adk_providers_and_bridges_logging(
    tmp_path: Path,
    genai_environment: pytest.MonkeyPatch,
    preserve_logging_configuration: None,
) -> None:
    tracer_provider, logger_provider = TracerProvider(), LoggerProvider()
    with (
        patch(f"{_SETUP}.get_log_dir", return_value=tmp_path),
        patch(f"{_SETUP}.maybe_set_otel_providers", autospec=True) as install,
        patch(f"{_SETUP}.trace.get_tracer_provider", return_value=tracer_provider),
        patch(f"{_SETUP}._logs.get_logger_provider", return_value=logger_provider),
    ):
        providers = setup_tracing()

    [hooks] = install.call_args.args[0]
    assert len(hooks.span_processors) == len(hooks.log_record_processors) == 1
    assert providers == TelemetryProviders(tracer_provider, logger_provider)
    assert any(isinstance(h, LoggingHandler) for h in logging.getLogger().handlers)
    assert genai_content_capture_enabled()
    shutdown_tracing(providers)
    for processor in hooks.span_processors:
        processor.shutdown()
    for processor in hooks.log_record_processors:
        processor.shutdown()


def test_setup_tracing_installs_a_provider_without_files_and_warns(
    tmp_path: Path,
    genai_environment: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("", encoding="utf-8")
    genai_environment.setenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "")
    with (
        patch(f"{_SETUP}.get_log_dir", return_value=blocked),
        patch(f"{_SETUP}.maybe_set_otel_providers", autospec=True) as install,
        patch(
            f"{_SETUP}.trace.get_tracer_provider", return_value=ProxyTracerProvider()
        ),
        patch(f"{_SETUP}.trace.set_tracer_provider") as set_provider,
        patch(f"{_SETUP}._logs.get_logger_provider", return_value=object()),
        caplog.at_level(logging.WARNING, logger="blacki.utils.observability"),
    ):
        providers = setup_tracing()

    assert install.call_args.args[0] == []
    set_provider.assert_called_once_with(providers.tracer)
    assert providers.logger is None
    assert "GenAI content capture is off" in caplog.text
    shutdown_tracing(providers)


def test_setup_tracing_explains_missing_otlp_exporter(tmp_path: Path) -> None:
    with (
        patch(f"{_SETUP}.get_log_dir", return_value=tmp_path),
        patch(
            f"{_SETUP}.maybe_set_otel_providers",
            autospec=True,
            side_effect=ImportError("opentelemetry.exporter"),
        ),
        pytest.raises(RuntimeError, match="no OTLP exporter package"),
    ):
        setup_tracing()


def test_shutdown_tracing_accepts_missing_providers() -> None:
    shutdown_tracing(None)


def test_resource_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEMETRY_NAMESPACE", "tests")
    monkeypatch.setenv("K_REVISION", "revision-1")
    configure_otel_resource("blacki-test")

    attributes = os.environ["OTEL_RESOURCE_ATTRIBUTES"]
    assert "service.name=blacki-test" in attributes
    assert "service.namespace=tests" in attributes
    assert "service.version=revision-1" in attributes


@pytest.mark.parametrize(
    ("in_container", "expected"),
    [(True, Path("/app/logs")), (False, Path("./logs"))],
)
def test_get_log_dir_selects_container_or_local_path(
    in_container: bool, expected: Path
) -> None:
    with patch.object(Path, "exists", return_value=in_container):
        assert get_log_dir() == expected


def test_setup_logging_writes_to_stdout_at_the_requested_level(
    preserve_logging_configuration: None,
) -> None:
    setup_logging("DEBUG")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert [type(handler) for handler in root.handlers] == [logging.StreamHandler]
    assert logging.getLogger("urllib3").level == logging.WARNING


def test_file_exporters_close_their_file_on_shutdown() -> None:
    traces, logs = io.StringIO(), io.StringIO()
    FileSpanExporter(out=traces).shutdown()
    FileLogRecordExporter(out=logs).shutdown()
    assert traces.closed and logs.closed
