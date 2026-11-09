"""Owner-scoped wiki storage. Markdown is rendered from sourced claims."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Source:
    ref: str
    text: str
    date: str
    role: str = "user"
    session: str = ""
    writable: bool = True


class WikiStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA secure_delete=ON")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS wiki_owners (
                    owner INTEGER PRIMARY KEY, enabled INTEGER NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS wiki_claims (
                    owner INTEGER NOT NULL, path TEXT NOT NULL, fact TEXT NOT NULL,
                    claim TEXT NOT NULL, source TEXT NOT NULL, excerpt TEXT NOT NULL,
                    source_date TEXT NOT NULL, disputed INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (owner, path, fact, source)
                );
                CREATE TABLE IF NOT EXISTS wiki_evidence (
                    owner INTEGER NOT NULL, path TEXT NOT NULL, fact TEXT NOT NULL,
                    source TEXT NOT NULL, PRIMARY KEY(owner, path, fact, source)
                );
                CREATE TABLE IF NOT EXISTS wiki_links (
                    owner INTEGER NOT NULL, path TEXT NOT NULL, fact TEXT NOT NULL,
                    target TEXT NOT NULL, PRIMARY KEY(owner, path, fact, target)
                );
                CREATE TABLE IF NOT EXISTS wiki_forgotten (
                    owner INTEGER NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
                    PRIMARY KEY (owner, kind, value)
                );
                """)
            yield db

    def enabled(self, owner: int) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT enabled FROM wiki_owners WHERE owner=?", (owner,)
            ).fetchone()
            return bool(row and row["enabled"])

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

    def read(self, owner: int, mode: str, value: str) -> dict[str, Any]:
        with self.connect() as db:
            state = db.execute(
                "SELECT enabled, revision FROM wiki_owners WHERE owner=?", (owner,)
            ).fetchone()
            if not state or not state["enabled"]:
                return {"status": "disabled"}
            revision = state["revision"]
            if mode == "page":
                rows = db.execute(
                    """
                    SELECT fact, claim, group_concat(source, ' ') AS sources, disputed
                    FROM wiki_claims WHERE owner=? AND path=?
                    GROUP BY fact, claim, disputed ORDER BY fact, claim LIMIT 13
                    """,
                    (owner, value),
                ).fetchall()
                links = db.execute(
                    """
                    SELECT DISTINCT links.target FROM wiki_links links
                    WHERE links.owner=? AND links.path=? AND EXISTS (
                        SELECT 1 FROM wiki_claims c
                        WHERE c.owner=links.owner AND c.path=links.target
                    ) ORDER BY links.target LIMIT 8
                    """,
                    (owner, value),
                ).fetchall()
                return {
                    "revision": revision,
                    "markdown": "\n".join(
                        [f"# {value}"]
                        + [
                            f"- {row['claim']} (key: {row['fact']}) "
                            + " ".join(f"[{ref}]" for ref in row["sources"].split()[:4])
                            + (" (disputed)" if row["disputed"] else "")
                            for row in rows[:12]
                        ]
                        + [f"- Related: [[{link['target']}]]" for link in links]
                    ),
                    "truncated": len(rows) > 12,
                }
            if mode == "source":
                row = db.execute(
                    """
                    SELECT excerpt, source_date FROM wiki_claims
                    WHERE owner=? AND source=? AND NOT EXISTS (
                        SELECT 1 FROM wiki_forgotten f
                        WHERE f.owner=? AND f.kind='source' AND f.value=?
                    ) ORDER BY path, fact LIMIT 1
                    """,
                    (owner, value, owner, value),
                ).fetchone()
                return {"revision": revision, "source": dict(row) if row else None}
            db.create_function("casefold", 1, str.casefold, deterministic=True)
            needle = value.casefold()
            paths = db.execute(
                """
                SELECT DISTINCT path FROM wiki_claims WHERE owner=? AND (
                    (?='prefix' AND instr(path, ?)=1)
                    OR (?='search' AND (
                        instr(path, ?)>0 OR (disputed=0 AND instr(casefold(claim), ?)>0)
                    ))
                ) ORDER BY path LIMIT 13
                """,
                (owner, mode, needle, mode, needle, needle),
            ).fetchall()
            return {
                "revision": revision,
                "pages": [row["path"] for row in paths[:12]],
                "truncated": len(paths) > 12,
            }

    def write(
        self,
        owner: int,
        path: str,
        fact: str,
        claim: str,
        source: Source,
        revision: int,
        explicit: bool,
        links: tuple[str, ...] = (),
    ) -> str:
        if not re.fullmatch(r"(?:[\w-]+/)*[\w-]+\.md", path) or not re.fullmatch(
            r"[\w-]+", fact
        ):
            return "invalid_path_or_fact"
        key = hashlib.sha256(f"{path}:{fact}".encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                "SELECT enabled, revision FROM wiki_owners WHERE owner=?", (owner,)
            ).fetchone()
            if not state or not state["enabled"]:
                return "disabled"
            rows = db.execute(
                """
                SELECT claim, source, source_date, disputed FROM wiki_claims
                WHERE owner=? AND path=? AND fact=?
                """,
                (owner, path, fact),
            ).fetchall()
            if any(row["source"] == source.ref for row in rows):
                return "unchanged"
            blocked = db.execute(
                """
                SELECT kind FROM wiki_forgotten WHERE owner=? AND (
                    (kind='source' AND value=?) OR (kind='fact' AND value=?))
                """,
                (owner, source.ref, key),
            ).fetchall()
            if any(row["kind"] == "source" or not explicit for row in blocked):
                return "forgotten"
            if revision != state["revision"]:
                return "stale_revision"
            # Replace only when the evidence is a newer, explicit correction.
            correction = source.text.lstrip().casefold().startswith(
                ("actually ", "actually,", "correction:", "update:", "remember ")
            ) and all(source.date > row["source_date"] for row in rows)
            different = any(row["claim"].casefold() != claim.casefold() for row in rows)
            disputed = (
                different or any(row["disputed"] for row in rows)
            ) and not correction
            if correction:
                db.execute(
                    "DELETE FROM wiki_claims WHERE owner=? AND path=? AND fact=?",
                    (owner, path, fact),
                )
            elif disputed:
                db.execute(
                    """
                    UPDATE wiki_claims SET disputed=1
                    WHERE owner=? AND path=? AND fact=?
                    """,
                    (owner, path, fact),
                )
            db.execute(
                """
                INSERT INTO wiki_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner,
                    path,
                    fact,
                    claim,
                    source.ref,
                    source.text,
                    source.date,
                    disputed,
                ),
            )
            db.execute(
                "INSERT OR IGNORE INTO wiki_evidence VALUES (?, ?, ?, ?)",
                (owner, path, fact, source.ref),
            )
            if correction:
                db.execute(
                    "DELETE FROM wiki_links WHERE owner=? AND path=? AND fact=?",
                    (owner, path, fact),
                )
            db.executemany(
                """
                INSERT OR IGNORE INTO wiki_links
                SELECT ?, ?, ?, ? WHERE EXISTS (
                    SELECT 1 FROM wiki_claims WHERE owner=? AND path=?
                )
                """,
                [
                    (owner, path, fact, target, owner, target)
                    for target in links
                    if target != path
                ],
            )
            if explicit:
                db.execute(
                    """
                    DELETE FROM wiki_forgotten WHERE owner=? AND kind='fact' AND value=?
                    """,
                    (owner, key),
                )
            db.execute(
                "UPDATE wiki_owners SET revision=revision+1 WHERE owner=?", (owner,)
            )
            return "disputed" if disputed else "saved"

    def suppressed_sources(self, owner: int) -> set[str]:
        with self.connect() as db:
            return {
                row["value"]
                for row in db.execute(
                    "SELECT value FROM wiki_forgotten WHERE owner=? AND kind='source'",
                    (owner,),
                )
            }

    def inspect(
        self,
        owner: int,
        path: str | None = None,
        fact: str | None = None,
    ) -> tuple[int, list[tuple[str, str]]]:
        with self.connect() as db:
            row = db.execute(
                "SELECT revision FROM wiki_owners WHERE owner=?", (owner,)
            ).fetchone()
            entries = db.execute(
                """
                SELECT DISTINCT path, fact FROM wiki_claims WHERE owner=?
                AND (? IS NULL OR path=?) AND (? IS NULL OR fact=?)
                ORDER BY path, fact LIMIT 24
                """,
                (owner, path, path, fact, fact),
            ).fetchall()
            return (
                row["revision"] if row else 0,
                [(entry["path"], entry["fact"]) for entry in entries],
            )

    def forget(self, owner: int, path: str, fact: str, revision: int) -> bool:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                "SELECT revision FROM wiki_owners WHERE owner=?", (owner,)
            ).fetchone()
            if not state or state["revision"] != revision:
                return False
            sources = db.execute(
                """
                SELECT source FROM wiki_evidence WHERE owner=? AND path=? AND fact=?
                """,
                (owner, path, fact),
            ).fetchall()
            if not sources:
                return False
            db.executemany(
                "INSERT OR IGNORE INTO wiki_forgotten VALUES (?, ?, ?)",
                [(owner, "fact", hashlib.sha256(f"{path}:{fact}".encode()).hexdigest())]
                + [(owner, "source", row["source"]) for row in sources],
            )
            db.execute(
                "DELETE FROM wiki_claims WHERE owner=? AND path=? AND fact=?",
                (owner, path, fact),
            )
            db.execute(
                "UPDATE wiki_owners SET revision=revision+1 WHERE owner=?", (owner,)
            )
            db.execute(
                "DELETE FROM wiki_evidence WHERE owner=? AND path=? AND fact=?",
                (owner, path, fact),
            )
            db.execute(
                "DELETE FROM wiki_links WHERE owner=? AND path=? AND fact=?",
                (owner, path, fact),
            )
            db.executemany(
                "UPDATE wiki_claims SET excerpt='' WHERE owner=? AND source=?",
                [(owner, row["source"]) for row in sources],
            )
            return True
