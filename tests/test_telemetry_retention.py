"""Exercise retention through the telemetry exporters and dashboard."""

import json
import logging
import stat
from pathlib import Path

import pytest
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk.trace import TracerProvider

from blacki.dashboard.data import DashboardStore
from blacki.observability import setup


@pytest.mark.asyncio
async def test_local_telemetry_rolls_over_and_dashboard_reads_both_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 4096
    monkeypatch.setattr(setup, "LOG_FILE_MAX_BYTES", limit, raising=False)
    monkeypatch.setattr(setup, "TRACE_FILE_MAX_BYTES", limit, raising=False)
    for name in (setup.LOG_FILE_NAME, setup.TRACE_FILE_NAME):
        (tmp_path / name).write_text("legacy" * limit)
    hooks = setup._otel_hooks(tmp_path)
    assert hooks is not None
    tracer_provider = TracerProvider()
    logger_provider = LoggerProvider()
    for span_processor in hooks.span_processors:
        tracer_provider.add_span_processor(span_processor)
    for log_processor in hooks.log_record_processors:
        logger_provider.add_log_record_processor(log_processor)
    test_logger = logging.Logger("test.retention", logging.INFO)
    test_logger.addHandler(LoggingHandler(logger_provider=logger_provider))
    tracer = tracer_provider.get_tracer("test")
    payload = "é" * 900
    try:
        for index in range(6):
            with tracer.start_as_current_span(f"span-{index}") as span:
                span.set_attribute("test.message", "é" * 480)
                test_logger.info("log-%d %s", index, payload)
        with tracer.start_as_current_span("oversized-span") as span:
            span.set_attribute("test.message", "é" * limit)
            test_logger.info("oversized-log %s", "é" * limit)
    finally:
        tracer_provider.shutdown()
        logger_provider.shutdown()

    for name in (setup.LOG_FILE_NAME, setup.TRACE_FILE_NAME):
        files = sorted(tmp_path.glob(f"{name}*"))
        assert [path.name for path in files] == [name, f"{name}.1"]
        for file in files:
            assert 0 < file.stat().st_size <= limit
            assert stat.S_IMODE(file.stat().st_mode) == 0o600
            assert all(json.loads(line) for line in file.read_text().splitlines())

    store = DashboardStore(tmp_path / "missing.db", tmp_path)
    assert (await store.list_logs(None, "log-5", 10))["items"]
    assert (await store.list_logs(None, "log-4", 10))["items"]
    assert (await store.list_logs(None, "log-0", 10))["items"] == []
    assert (await store.list_traces(None, "span-5", 10))["items"]
    assert (await store.list_traces(None, "span-4", 10))["items"]
    assert (await store.list_traces(None, "span-0", 10))["items"] == []


def test_file_exporter_resumes_after_rollover_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    limit = 1024
    monkeypatch.setattr(setup, "LOG_FILE_MAX_BYTES", limit)
    path = tmp_path / setup.LOG_FILE_NAME
    backup = path.with_suffix(".log.1")
    path.write_text("legacy" * limit)
    backup.mkdir()
    hooks = setup._otel_hooks(tmp_path)
    assert hooks is not None
    provider = LoggerProvider()
    for processor in hooks.log_record_processors:
        provider.add_log_record_processor(processor)
    test_logger = logging.Logger("test.recovery", logging.INFO)
    test_logger.addHandler(LoggingHandler(logger_provider=provider))
    try:
        test_logger.info("before recovery")
        provider.force_flush()
        assert "Exception while exporting" in caplog.text
        assert path.read_text() == "legacy" * limit

        backup.rmdir()
        test_logger.info("after recovery")
        provider.force_flush()
        assert json.loads(path.read_text())["body"] == "after recovery"
        assert backup.read_text() == "legacy" * limit
    finally:
        provider.shutdown()
        for span_processor in hooks.span_processors:
            span_processor.shutdown()
