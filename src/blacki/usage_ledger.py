"""Content-free, idempotent local ledger for model usage and cost."""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

LEDGER_TABLE = "llm_usage_ledger"
COST_SCALE = 1_000_000_000
COST_LEDGER_WARNING = "Cost ledger is temporarily unavailable."


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One billable model response without prompts, outputs, or tool data."""

    dedupe_key: str
    observed_at: float
    user_id: str
    session_id: str
    invocation_id: str
    model: str
    provider_response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    upstream_cost_usd: float | None
    estimated_cost_usd: float | None
    cost_kind: str
    cost_source: str
    currency: str = "USD"


@dataclass(frozen=True, slots=True)
class LedgerSummary:
    """Fixed-point aggregates returned by the usage ledger."""

    records: int = 0
    reported_records: int = 0
    estimated_records: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_nano_usd: int | None = None
    upstream_cost_nano_usd: int | None = None
    estimated_cost_nano_usd: int | None = None


@dataclass(frozen=True, slots=True)
class UsageLedgerSnapshot:
    """All dashboard aggregates needed for one refresh."""

    selected: LedgerSummary
    cumulative: LedgerSummary
    monthly: LedgerSummary
    users: dict[str, LedgerSummary]
    monthly_users: dict[str, LedgerSummary]
    sessions: dict[tuple[str, str], LedgerSummary]
    monthly_sessions: dict[tuple[str, str], LedgerSummary]
    available: bool
    warnings: tuple[str, ...]


def default_usage_ledger_path(agent_dir: Path | str | None = None) -> Path:
    """Return the configurable content-free usage ledger path."""
    configured = os.environ.get("BLACKI_COST_LEDGER_PATH", "").strip()
    if configured:
        return Path(configured)
    configured_agent_dir = os.environ.get("AGENT_DIR", "").strip()
    base = Path(agent_dir) if agent_dir is not None else Path(configured_agent_dir)
    if agent_dir is None and not configured_agent_dir:
        base = Path(__file__).resolve().parent.parent
    return base / ".adk" / "costs.db"


def _fixed_cost(value: float | None) -> int | None:
    if value is None or isinstance(value, bool) or not math.isfinite(value):
        return None
    if value < 0:
        return None
    try:
        decimal_value = Decimal(str(value)) * COST_SCALE
        return int(decimal_value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def fixed_cost_to_usd(value: int | None) -> float | None:
    """Convert fixed-point nanodollars to a JSON-safe USD float."""
    if value is None:
        return None
    return round(value / COST_SCALE, 9)


def _schema_sql() -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
            dedupe_key TEXT PRIMARY KEY,
            observed_at REAL NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            invocation_id TEXT NOT NULL,
            model TEXT NOT NULL,
            provider_response_id TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            cost_nano_usd INTEGER,
            upstream_cost_nano_usd INTEGER,
            estimated_cost_nano_usd INTEGER,
            cost_kind TEXT NOT NULL,
            cost_source TEXT NOT NULL,
            currency TEXT NOT NULL
        )
    """


def write_usage_record(path: Path, record: UsageRecord) -> None:
    """Upsert one usage record, preserving provider-response idempotency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = (
        record.dedupe_key,
        record.observed_at,
        record.user_id,
        record.session_id,
        record.invocation_id,
        record.model,
        record.provider_response_id,
        record.input_tokens,
        record.output_tokens,
        record.total_tokens,
        _fixed_cost(record.cost_usd),
        _fixed_cost(record.upstream_cost_usd),
        _fixed_cost(record.estimated_cost_usd),
        record.cost_kind,
        record.cost_source,
        record.currency,
    )
    connection = sqlite3.connect(path, timeout=5.0)
    try:
        with connection:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(_schema_sql())
            connection.execute(  # noqa: S608
                f"""
            INSERT INTO {LEDGER_TABLE} (
                dedupe_key, observed_at, user_id, session_id, invocation_id,
                model, provider_response_id, input_tokens, output_tokens,
                total_tokens, cost_nano_usd, upstream_cost_nano_usd,
                estimated_cost_nano_usd, cost_kind, cost_source, currency
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dedupe_key) DO UPDATE SET
                provider_response_id = COALESCE(
                    excluded.provider_response_id,
                    {LEDGER_TABLE}.provider_response_id
                ),
                input_tokens = COALESCE(
                    excluded.input_tokens, {LEDGER_TABLE}.input_tokens
                ),
                output_tokens = COALESCE(
                    excluded.output_tokens, {LEDGER_TABLE}.output_tokens
                ),
                total_tokens = COALESCE(
                    excluded.total_tokens, {LEDGER_TABLE}.total_tokens
                ),
                cost_nano_usd = COALESCE(
                    excluded.cost_nano_usd, {LEDGER_TABLE}.cost_nano_usd
                ),
                upstream_cost_nano_usd = COALESCE(
                    excluded.upstream_cost_nano_usd,
                    {LEDGER_TABLE}.upstream_cost_nano_usd
                ),
                estimated_cost_nano_usd = COALESCE(
                    excluded.estimated_cost_nano_usd,
                    {LEDGER_TABLE}.estimated_cost_nano_usd
                ),
                cost_kind = CASE
                    WHEN {LEDGER_TABLE}.cost_kind = 'reported'
                    THEN {LEDGER_TABLE}.cost_kind
                    ELSE excluded.cost_kind
                END,
                cost_source = CASE
                    WHEN {LEDGER_TABLE}.cost_kind = 'reported'
                    THEN {LEDGER_TABLE}.cost_source
                    ELSE excluded.cost_source
                END
            """,  # noqa: S608
                values,
            )
    finally:
        connection.close()


_SUMMARY_COLUMNS = """
    COUNT(*) AS records,
    SUM(CASE WHEN cost_nano_usd IS NOT NULL THEN 1 ELSE 0 END)
        AS reported_records,
    SUM(CASE WHEN estimated_cost_nano_usd IS NOT NULL
        AND cost_nano_usd IS NULL THEN 1 ELSE 0 END) AS estimated_records,
    COALESCE(SUM(input_tokens), 0) AS input_tokens,
    COALESCE(SUM(output_tokens), 0) AS output_tokens,
    COALESCE(SUM(total_tokens), 0) AS total_tokens,
    SUM(cost_nano_usd) AS cost_nano_usd,
    SUM(upstream_cost_nano_usd) AS upstream_cost_nano_usd,
    SUM(estimated_cost_nano_usd) AS estimated_cost_nano_usd
"""


def _summary_from_row(row: sqlite3.Row | None) -> LedgerSummary:
    if row is None:
        return LedgerSummary()
    return LedgerSummary(
        records=int(row["records"] or 0),
        reported_records=int(row["reported_records"] or 0),
        estimated_records=int(row["estimated_records"] or 0),
        input_tokens=int(row["input_tokens"] or 0),
        output_tokens=int(row["output_tokens"] or 0),
        total_tokens=int(row["total_tokens"] or 0),
        cost_nano_usd=(
            int(row["cost_nano_usd"]) if row["cost_nano_usd"] is not None else None
        ),
        upstream_cost_nano_usd=(
            int(row["upstream_cost_nano_usd"])
            if row["upstream_cost_nano_usd"] is not None
            else None
        ),
        estimated_cost_nano_usd=(
            int(row["estimated_cost_nano_usd"])
            if row["estimated_cost_nano_usd"] is not None
            else None
        ),
    )


def _where_clause(
    *,
    since: float | None,
    until: float | None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    values: list[Any] = []
    if since is not None:
        clauses.append("observed_at >= ?")
        values.append(since)
    if until is not None:
        clauses.append("observed_at <= ?")
        values.append(until)
    if user_id is not None:
        clauses.append("user_id = ?")
        values.append(user_id)
    if session_id is not None:
        clauses.append("session_id = ?")
        values.append(session_id)
    return " AND ".join(clauses) or "1 = 1", tuple(values)


def _query_summary(
    connection: sqlite3.Connection,
    *,
    since: float | None,
    until: float | None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> LedgerSummary:
    where, values = _where_clause(
        since=since,
        until=until,
        user_id=user_id,
        session_id=session_id,
    )
    row = connection.execute(  # noqa: S608
        f"SELECT {_SUMMARY_COLUMNS} FROM {LEDGER_TABLE} WHERE {where}",  # noqa: S608
        values,
    ).fetchone()
    return _summary_from_row(row)


def _query_grouped(
    connection: sqlite3.Connection,
    *,
    group_columns: tuple[str, ...],
    since: float | None,
    until: float | None,
) -> dict[Any, LedgerSummary]:
    where, values = _where_clause(since=since, until=until)
    group_sql = ", ".join(group_columns)
    rows = connection.execute(  # noqa: S608
        f"""
        SELECT {group_sql}, {_SUMMARY_COLUMNS}
        FROM {LEDGER_TABLE}
        WHERE {where}
        GROUP BY {group_sql}
        """,  # noqa: S608
        values,
    ).fetchall()
    result: dict[Any, LedgerSummary] = {}
    for row in rows:
        key = (
            row[group_columns[0]]
            if len(group_columns) == 1
            else tuple(row[column] for column in group_columns)
        )
        result[key] = _summary_from_row(row)
    return result


def _empty_snapshot(
    *, available: bool, warnings: tuple[str, ...] = ()
) -> UsageLedgerSnapshot:
    empty = LedgerSummary()
    return UsageLedgerSnapshot(
        selected=empty,
        cumulative=empty,
        monthly=empty,
        users={},
        monthly_users={},
        sessions={},
        monthly_sessions={},
        available=available,
        warnings=warnings,
    )


def read_usage_ledger(
    path: Path,
    *,
    selected_since: float | None,
    selected_until: float | None,
    month_start: float,
    now: float,
) -> UsageLedgerSnapshot:
    """Read aggregate usage without exposing ledger rows to the dashboard."""
    path = Path(path)
    try:
        if not path.is_file():
            return _empty_snapshot(available=False)
        uri = f"file:{quote(path.resolve().as_posix(), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        try:
            connection.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if LEDGER_TABLE not in tables:
                return _empty_snapshot(available=False)
            selected = _query_summary(
                connection,
                since=selected_since,
                until=selected_until,
            )
            cumulative = _query_summary(connection, since=None, until=now)
            monthly = _query_summary(
                connection,
                since=month_start,
                until=now,
            )
            users = _query_grouped(
                connection,
                group_columns=("user_id",),
                since=None,
                until=now,
            )
            monthly_users = _query_grouped(
                connection,
                group_columns=("user_id",),
                since=month_start,
                until=now,
            )
            sessions = _query_grouped(
                connection,
                group_columns=("user_id", "session_id"),
                since=None,
                until=now,
            )
            monthly_sessions = _query_grouped(
                connection,
                group_columns=("user_id", "session_id"),
                since=month_start,
                until=now,
            )
            return UsageLedgerSnapshot(
                selected=selected,
                cumulative=cumulative,
                monthly=monthly,
                users={str(key): value for key, value in users.items()},
                monthly_users={str(key): value for key, value in monthly_users.items()},
                sessions={
                    (str(key[0]), str(key[1])): value for key, value in sessions.items()
                },
                monthly_sessions={
                    (str(key[0]), str(key[1])): value
                    for key, value in monthly_sessions.items()
                },
                available=True,
                warnings=(),
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError):
        return _empty_snapshot(available=False, warnings=(COST_LEDGER_WARNING,))
