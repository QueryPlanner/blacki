"""SQLite storage for the private wiki's owner-scoped Markdown pages."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from blacki.utils.timezone import now_utc

from .pages import EditError, Page, apply_edit

# Claim-based tables from the previous wiki design. No data is migrated.
_DROP_LEGACY_TABLES = """
DROP TABLE IF EXISTS wiki_claims;
DROP TABLE IF EXISTS wiki_evidence;
DROP TABLE IF EXISTS wiki_links;
DROP TABLE IF EXISTS wiki_forgotten;
"""
_SCHEMA = """
CREATE TABLE IF NOT EXISTS wiki_owners (
    owner INTEGER PRIMARY KEY, enabled INTEGER NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS wiki_pages (
    owner INTEGER NOT NULL, path TEXT NOT NULL, markdown TEXT NOT NULL,
    revision INTEGER NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (owner, path)
);
"""
_UPSERT_PAGE = """
INSERT INTO wiki_pages(owner, path, markdown, revision, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(owner, path) DO UPDATE SET
    markdown=excluded.markdown,
    revision=excluded.revision,
    updated_at=excluded.updated_at
"""


@dataclass(frozen=True)
class EditOutcome:
    """Result of one edit attempt. Only saved and deleted change the wiki."""

    status: str
    path: str
    chars: int = 0
    revision: int = 0
    hint: str = ""


def _page(row: sqlite3.Row) -> Page:
    return Page(
        path=row["path"],
        markdown=row["markdown"],
        revision=row["revision"],
        updated_at=row["updated_at"],
    )


class WikiStore:
    """Owner-scoped page storage. Every page belongs to one Telegram chat ID."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA secure_delete=ON")
            db.executescript(_DROP_LEGACY_TABLES + _SCHEMA)
            yield db

    def enabled(self, owner: int) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT enabled FROM wiki_owners WHERE owner=?", (owner,)
            ).fetchone()
            return row is None or bool(row["enabled"])

    def set_enabled(self, owner: int, enabled: bool) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO wiki_owners(owner, enabled) VALUES (?, ?)
                ON CONFLICT(owner) DO UPDATE SET
                    enabled=excluded.enabled, revision=revision+1
                """,
                (owner, enabled),
            )

    def get_page(self, owner: int, path: str) -> Page | None:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT path, markdown, revision, updated_at FROM wiki_pages
                WHERE owner=? AND path=?
                """,
                (owner, path),
            ).fetchone()
        return None if row is None else _page(row)

    def list_pages(self, owner: int) -> list[Page]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT path, markdown, revision, updated_at FROM wiki_pages
                WHERE owner=? ORDER BY path
                """,
                (owner,),
            ).fetchall()
        return [_page(row) for row in rows]

    def edit(
        self,
        owner: int,
        path: str,
        new_text: str,
        old_text: str,
        *,
        read: bool,
    ) -> EditOutcome:
        """Apply one edit inside a single immediate write transaction.

        read is True only when the caller fully read this page earlier in the
        same turn. Editing an existing page without that read is refused. The
        owner revision changes only when the stored wiki changes.
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                "SELECT enabled FROM wiki_owners WHERE owner=?", (owner,)
            ).fetchone()
            if state is not None and not state["enabled"]:
                return EditOutcome("disabled", path)
            row = db.execute(
                "SELECT markdown, revision FROM wiki_pages WHERE owner=? AND path=?",
                (owner, path),
            ).fetchone()
            if row is None and old_text:
                return EditOutcome(
                    "not_found",
                    path,
                    hint="No page exists at this path. Leave old_text empty to "
                    "create it.",
                )
            if row is not None and not read:
                return EditOutcome(
                    "read_required",
                    path,
                    chars=len(row["markdown"]),
                    hint="Read this page with wiki_read(path=...) in this turn "
                    "before editing it.",
                )
            current = row["markdown"] if row is not None else ""
            try:
                markdown = apply_edit(current, new_text, old_text)
            except EditError as error:
                return EditOutcome(
                    error.status, path, chars=len(current), hint=error.hint
                )
            if markdown == current:
                return EditOutcome(
                    "unchanged",
                    path,
                    chars=len(current),
                    revision=row["revision"],
                    hint="The page already has this text. Nothing was saved.",
                )
            if markdown.strip():
                revision = (row["revision"] if row is not None else 0) + 1
                timestamp = now_utc().isoformat(timespec="seconds")
                db.execute(_UPSERT_PAGE, (owner, path, markdown, revision, timestamp))
                outcome = EditOutcome(
                    "saved", path, chars=len(markdown), revision=revision
                )
            else:
                db.execute(
                    "DELETE FROM wiki_pages WHERE owner=? AND path=?", (owner, path)
                )
                outcome = EditOutcome("deleted", path)
            db.execute(
                "INSERT OR IGNORE INTO wiki_owners(owner, enabled) VALUES (?, 1)",
                (owner,),
            )
            db.execute(
                "UPDATE wiki_owners SET revision=revision+1 WHERE owner=?", (owner,)
            )
            return outcome
