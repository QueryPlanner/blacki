"""Tests for the content-free model usage ledger."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from blacki.config.paths import package_root
from blacki.observability.ledger import (
    COST_LEDGER_WARNING,
    UsageRecord,
    _fixed_cost,
    _summary_from_row,
    _where_clause,
    default_usage_ledger_path,
    fixed_cost_to_usd,
    read_usage_ledger,
    write_usage_record,
)


def _record(
    key: str,
    *,
    timestamp: float,
    user_id: str = "user-1",
    session_id: str = "session-1",
    cost: float | None = None,
    upstream: float | None = None,
    estimate: float | None = None,
    kind: str = "reported",
) -> UsageRecord:
    return UsageRecord(
        dedupe_key=key,
        observed_at=timestamp,
        user_id=user_id,
        session_id=session_id,
        invocation_id=f"inv-{key}",
        model="openrouter/test-model",
        provider_response_id=f"response-{key}",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cost_usd=cost,
        upstream_cost_usd=upstream,
        estimated_cost_usd=estimate,
        cost_kind=kind,
        cost_source="provider_usage" if kind == "reported" else "test",
    )


def test_ledger_aggregates_fixed_point_costs_and_groups_users(tmp_path: Path) -> None:
    path = tmp_path / "costs.db"
    write_usage_record(
        path,
        _record("one", timestamp=100, cost=0.000000001, upstream=0.000000001),
    )
    write_usage_record(
        path,
        _record(
            "two",
            timestamp=250,
            user_id="user-2",
            session_id="session-2",
            cost=0.25,
            upstream=0.2,
        ),
    )
    write_usage_record(
        path,
        _record(
            "three",
            timestamp=300,
            user_id="user-1",
            session_id="session-1",
            estimate=0.5,
            kind="estimated",
        ),
    )

    snapshot = read_usage_ledger(
        path,
        selected_since=150,
        selected_until=350,
        month_start=200,
        now=350,
    )

    assert snapshot.available is True
    assert snapshot.selected.records == 2
    assert snapshot.cumulative.records == 3
    assert fixed_cost_to_usd(snapshot.cumulative.cost_nano_usd) == 0.250000001
    assert fixed_cost_to_usd(snapshot.monthly.cost_nano_usd) == 0.25
    assert snapshot.cumulative.reported_records == 2
    assert snapshot.cumulative.estimated_records == 1
    assert snapshot.users["user-2"].total_tokens == 15
    assert snapshot.monthly_users["user-2"].records == 1
    assert snapshot.sessions[("user-1", "session-1")].records == 2
    assert snapshot.monthly_sessions[("user-1", "session-1")].records == 1


def test_ledger_upsert_does_not_double_count_stream_chunks(tmp_path: Path) -> None:
    path = tmp_path / "costs.db"
    write_usage_record(
        path,
        _record(
            "same-response",
            timestamp=100,
            estimate=0.01,
            kind="estimated",
        ),
    )
    write_usage_record(
        path,
        _record(
            "same-response",
            timestamp=100,
            cost=0.02,
            upstream=0.018,
            kind="reported",
        ),
    )

    snapshot = read_usage_ledger(
        path,
        selected_since=None,
        selected_until=200,
        month_start=1,
        now=200,
    )

    assert snapshot.cumulative.records == 1
    assert snapshot.cumulative.reported_records == 1
    assert snapshot.cumulative.estimated_records == 0
    assert fixed_cost_to_usd(snapshot.cumulative.cost_nano_usd) == 0.02
    assert fixed_cost_to_usd(snapshot.cumulative.upstream_cost_nano_usd) == 0.018


def test_ledger_missing_and_corrupt_files_are_non_mutating(tmp_path: Path) -> None:
    missing = read_usage_ledger(
        tmp_path / "missing.db",
        selected_since=None,
        selected_until=10,
        month_start=1,
        now=10,
    )
    assert missing.available is False
    assert missing.warnings == ()

    corrupt_path = tmp_path / "corrupt.db"
    corrupt_path.write_text("not sqlite", encoding="utf-8")
    corrupt = read_usage_ledger(
        corrupt_path,
        selected_since=None,
        selected_until=10,
        month_start=1,
        now=10,
    )
    assert corrupt.available is False
    assert corrupt.warnings == (COST_LEDGER_WARNING,)


def test_default_ledger_path_honors_explicit_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured.db"
    monkeypatch.setenv("BLACKI_COST_LEDGER_PATH", str(configured))
    assert default_usage_ledger_path() == configured

    monkeypatch.delenv("BLACKI_COST_LEDGER_PATH")
    assert default_usage_ledger_path(tmp_path) == tmp_path / ".adk" / "costs.db"


@pytest.mark.parametrize("blank_agent_dir", ["", "   "])
def test_default_ledger_path_ignores_blank_explicit_agent_dir(
    monkeypatch: pytest.MonkeyPatch,
    blank_agent_dir: str,
) -> None:
    """Blank dashboard overrides must use the stable configured root."""
    monkeypatch.delenv("BLACKI_COST_LEDGER_PATH", raising=False)
    monkeypatch.delenv("AGENT_DIR", raising=False)

    assert default_usage_ledger_path(blank_agent_dir) == (
        package_root().parent / ".adk" / "costs.db"
    )


def test_default_ledger_path_uses_stable_agent_root_after_relocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relocating the ledger module must not move the default database."""
    monkeypatch.delenv("BLACKI_COST_LEDGER_PATH", raising=False)
    monkeypatch.delenv("AGENT_DIR", raising=False)

    assert default_usage_ledger_path() == package_root().parent / ".adk" / "costs.db"


def test_ledger_write_preserves_existing_schema(tmp_path: Path) -> None:
    """The relocated writer must keep every existing ledger column."""
    path = tmp_path / "costs.db"
    write_usage_record(path, _record("schema", timestamp=100, cost=0.01))

    connection = sqlite3.connect(path)
    try:
        columns = [
            str(row[1])
            for row in connection.execute("PRAGMA table_info(llm_usage_ledger)")
        ]
    finally:
        connection.close()

    assert columns == [
        "dedupe_key",
        "observed_at",
        "user_id",
        "session_id",
        "invocation_id",
        "model",
        "provider_response_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_nano_usd",
        "upstream_cost_nano_usd",
        "estimated_cost_nano_usd",
        "cost_kind",
        "cost_source",
        "currency",
    ]


def test_ledger_defensive_helpers_cover_invalid_values_and_filters(
    tmp_path: Path,
) -> None:
    assert _fixed_cost(-1) is None
    assert _fixed_cost(1.7e308) is None
    assert _summary_from_row(None).records == 0

    where, values = _where_clause(
        since=1,
        until=2,
        user_id="user-1",
        session_id="session-1",
    )
    assert where == (
        "observed_at >= ? AND observed_at <= ? AND user_id = ? AND session_id = ?"
    )
    assert values == (1, 2, "user-1", "session-1")
    assert _where_clause(since=None, until=None) == ("1 = 1", ())

    empty_path = tmp_path / "empty.db"
    connection = sqlite3.connect(empty_path)
    connection.close()
    empty = read_usage_ledger(
        empty_path,
        selected_since=None,
        selected_until=10,
        month_start=1,
        now=10,
    )
    assert empty.available is False
    assert empty.warnings == ()
