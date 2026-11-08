"""SQLite storage for Mini App pages, versions, chat cards and preferences.

Built-in pages live as source files in ``blacki/miniapp/pages`` and are never
stored or editable. Agent pages (``foryou`` and pages Blacki creates) keep
every version so the user can undo. All writes take the shared container lock
and run in one transaction, so a version check and its write are atomic.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from typing import Any

from blacki.storage.base import SqlStorage

from .catalog import AGENT_HOME_PAGE, BUILTIN_PAGES

MAX_AGENT_PAGES = 12
MAX_VERSIONS = 30
CARD_TTL_DAYS = 14
BUILTIN_TITLES = {
    "today": "Today",
    "health": "Health",
    "food": "Food",
    "training": "Training",
    "files": "Files",
}


class StaleVersionError(ValueError):
    """The page changed since the caller read it."""


class WriteBatchError(Exception):
    """One write in a batch failed, so the whole batch was rolled back."""

    def __init__(self, index: int, cause: Exception) -> None:
        self.index = index
        self.cause = cause
        super().__init__(f"write {index} failed: {cause}")


@dataclass(frozen=True)
class PageSave:
    """Save a new version of an agent page."""

    owner: str
    page_id: str
    title: str
    source: str
    ir: dict[str, Any]
    base_version: int | None
    created_by: str
    note: str | None = None


@dataclass(frozen=True)
class PageDelete:
    owner: str
    page_id: str


@dataclass(frozen=True)
class PageMove:
    owner: str
    page_id: str
    position: int


@dataclass(frozen=True)
class CardCreate:
    owner: str
    title: str
    source: str
    ir: dict[str, Any]


StoreWrite = PageSave | PageDelete | PageMove | CardCreate


_LIST_CARDS_ALL_SQL = (
    "SELECT card_id, title, pinned, created_at FROM miniapp_cards "
    "WHERE owner = ? AND (pinned = 1 OR expires_at >= ?) "
    "ORDER BY created_at DESC LIMIT 30"
)
_LIST_CARDS_PINNED_SQL = (
    "SELECT card_id, title, pinned, created_at FROM miniapp_cards "
    "WHERE owner = ? AND (pinned = 1 OR expires_at >= ?) "
    "AND pinned = 1 ORDER BY created_at DESC LIMIT 30"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def builtin_source(page_id: str) -> str:
    """Return the shipped source of a built-in (or seeded) page."""
    return (
        resources.files("blacki.miniapp")
        .joinpath("pages", f"{page_id}.blk")
        .read_text(encoding="utf-8")
    )


class MiniAppStore(SqlStorage):
    """Owner-scoped page, card and preference storage."""

    async def _create_tables(self) -> None:
        await self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS miniapp_pages (
                owner TEXT NOT NULL,
                page_id TEXT NOT NULL,
                title TEXT NOT NULL,
                position INTEGER NOT NULL DEFAULT 100,
                version INTEGER NOT NULL,
                source TEXT NOT NULL,
                ir TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner, page_id)
            );
            CREATE TABLE IF NOT EXISTS miniapp_page_versions (
                owner TEXT NOT NULL,
                page_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                created_by TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (owner, page_id, version)
            );
            CREATE TABLE IF NOT EXISTS miniapp_cards (
                owner TEXT NOT NULL,
                card_id TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                ir TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                PRIMARY KEY (owner, card_id)
            );
            CREATE TABLE IF NOT EXISTS miniapp_prefs (
                owner TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner, key)
            );
            """
        )
        await self._conn.commit()

    # ---------------------------------------------------------- pages

    async def list_pages(self, owner: str) -> list[dict[str, Any]]:
        await self.initialize()
        pages: list[dict[str, Any]] = [
            {"id": pid, "title": BUILTIN_TITLES[pid], "kind": "builtin", "version": 0}
            for pid in BUILTIN_PAGES
        ]
        rows = await self._fetch_all(
            "SELECT page_id, title, version, position FROM miniapp_pages "
            "WHERE owner = ? AND deleted = 0 ORDER BY position, page_id",
            (owner,),
        )
        seen = {r["page_id"] for r in rows}
        if AGENT_HOME_PAGE not in seen:
            pages.append(
                {
                    "id": AGENT_HOME_PAGE,
                    "title": "For you",
                    "kind": "agent",
                    "version": 0,
                }
            )
        pages += [
            {
                "id": r["page_id"],
                "title": r["title"],
                "kind": "agent",
                "version": r["version"],
            }
            for r in rows
        ]
        home = next(p for p in pages if p["id"] == AGENT_HOME_PAGE)
        return (
            [p for p in pages if p is not home][:5]
            + [home]
            + [p for p in pages if p is not home][5:]
        )

    async def get_page(self, owner: str, page_id: str) -> dict[str, Any] | None:
        """Return an agent page's current row, or None if it doesn't exist."""
        await self.initialize()
        row = await self._fetch_one(
            "SELECT page_id, title, version, source, ir FROM miniapp_pages "
            "WHERE owner = ? AND page_id = ? AND deleted = 0",
            (owner, page_id),
        )
        if row is None:
            return None
        return {
            "id": row["page_id"],
            "title": row["title"],
            "kind": "agent",
            "version": row["version"],
            "source": row["source"],
            "ir": json.loads(row["ir"]),
        }

    async def save_page(
        self,
        owner: str,
        page_id: str,
        *,
        title: str,
        source: str,
        ir: dict[str, Any],
        base_version: int | None,
        created_by: str,
        note: str | None = None,
    ) -> int:
        """Save a new version; raises StaleVersionError on a version mismatch."""
        version: int = await self._single(
            PageSave(
                owner,
                page_id,
                title,
                source,
                ir,
                base_version,
                created_by,
                note,
            )
        )
        return version

    async def delete_page(self, owner: str, page_id: str) -> bool:
        try:
            await self._single(PageDelete(owner, page_id))
        except LookupError:
            return False
        return True

    async def move_page(self, owner: str, page_id: str, position: int) -> bool:
        try:
            await self._single(PageMove(owner, page_id, position))
        except LookupError:
            return False
        return True

    async def apply_writes(self, writes: Sequence[StoreWrite]) -> list[Any]:
        """Run writes in one transaction: every write commits, or none does.

        Returns one result per write: the new version for a page save, the card
        id for a card, and None for a delete or move. A failing write raises
        WriteBatchError with its index, and nothing from this call is saved.
        """
        await self.initialize()
        results: list[Any] = []
        async with self._lock:
            await self._conn.execute("BEGIN")
            try:
                for index, write in enumerate(writes):
                    try:
                        results.append(await self._apply(write))
                    except (StaleVersionError, ValueError, LookupError) as exc:
                        raise WriteBatchError(index, exc) from exc
                await self._conn.commit()
            except BaseException:
                await self._conn.rollback()
                raise
        return results

    async def _single(self, write: StoreWrite) -> Any:
        """Run one write on its own; its own failure is raised unwrapped."""
        try:
            (result,) = await self.apply_writes([write])
        except WriteBatchError as failed:
            raise failed.cause from None
        return result

    async def _apply(self, write: StoreWrite) -> Any:
        """Run one write inside the open transaction; the caller commits."""
        if isinstance(write, PageSave):
            return await self._save_page_in_tx(write)
        if isinstance(write, PageDelete):
            return await self._delete_page_in_tx(write)
        if isinstance(write, PageMove):
            return await self._move_page_in_tx(write)
        return await self._create_card_in_tx(write)

    async def _save_page_in_tx(self, write: PageSave) -> int:
        async with self._conn.execute(
            "SELECT version, deleted FROM miniapp_pages WHERE owner = ? AND page_id = ?",  # noqa: E501
            (write.owner, write.page_id),
        ) as cur:
            row = await cur.fetchone()
        current = int(row[0]) if row and not row[1] else 0
        if write.base_version is not None and write.base_version != current:
            raise StaleVersionError(
                f"page {write.page_id} is at version {current}, not {write.base_version}; read it again"  # noqa: E501
            )
        if row is None or row[1]:
            async with self._conn.execute(
                "SELECT COUNT(*) FROM miniapp_pages WHERE owner = ? AND deleted = 0",
                (write.owner,),
            ) as cur:
                count_row = await cur.fetchone()
            if count_row and count_row[0] >= MAX_AGENT_PAGES:
                raise ValueError(f"you can have at most {MAX_AGENT_PAGES} Blacki pages")
        version = (int(row[0]) if row else 0) + 1
        now = _now()
        await self._conn.execute(
            """
            INSERT INTO miniapp_pages
                (owner, page_id, title, version, source, ir, deleted,
                 updated_at, position)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, 100)
            ON CONFLICT(owner, page_id) DO UPDATE SET
                title = excluded.title, version = excluded.version,
                source = excluded.source, ir = excluded.ir, deleted = 0,
                updated_at = excluded.updated_at
            """,
            (
                write.owner,
                write.page_id,
                write.title,
                version,
                write.source,
                json.dumps(write.ir),
                now,
            ),
        )
        await self._conn.execute(
            """
            INSERT INTO miniapp_page_versions
                (owner, page_id, version, title, source, created_by, note,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                write.owner,
                write.page_id,
                version,
                write.title,
                write.source,
                write.created_by,
                write.note,
                now,
            ),
        )
        await self._conn.execute(
            """
            DELETE FROM miniapp_page_versions
            WHERE owner = ? AND page_id = ? AND version <= ?
            """,
            (write.owner, write.page_id, version - MAX_VERSIONS),
        )
        return version

    async def _delete_page_in_tx(self, write: PageDelete) -> None:
        cursor = await self._conn.execute(
            "UPDATE miniapp_pages SET deleted = 1, updated_at = ? "
            "WHERE owner = ? AND page_id = ? AND deleted = 0",
            (_now(), write.owner, write.page_id),
        )
        if cursor.rowcount == 0:
            raise LookupError(f"page {write.page_id} not found")

    async def _move_page_in_tx(self, write: PageMove) -> None:
        cursor = await self._conn.execute(
            "UPDATE miniapp_pages SET position = ? WHERE owner = ? AND page_id = ? AND deleted = 0",  # noqa: E501
            (write.position, write.owner, write.page_id),
        )
        if cursor.rowcount == 0:
            raise LookupError(f"page {write.page_id} not found")

    async def history(self, owner: str, page_id: str) -> list[dict[str, Any]]:
        await self.initialize()
        return await self._fetch_all(
            "SELECT version, title, created_by, note, created_at FROM miniapp_page_versions "  # noqa: E501
            "WHERE owner = ? AND page_id = ? ORDER BY version DESC LIMIT 30",
            (owner, page_id),
        )

    async def version_source(
        self, owner: str, page_id: str, version: int
    ) -> dict[str, Any] | None:
        await self.initialize()
        return await self._fetch_one(
            "SELECT title, source FROM miniapp_page_versions "
            "WHERE owner = ? AND page_id = ? AND version = ?",
            (owner, page_id, version),
        )

    # ---------------------------------------------------------- cards

    async def create_card(
        self, owner: str, *, title: str, source: str, ir: dict[str, Any]
    ) -> str:
        card_id: str = await self._single(CardCreate(owner, title, source, ir))
        return card_id

    async def _create_card_in_tx(self, write: CardCreate) -> str:
        card_id = secrets.token_urlsafe(9)
        now = datetime.now(UTC)
        await self._conn.execute(
            "DELETE FROM miniapp_cards WHERE owner = ? AND pinned = 0 AND expires_at < ?",  # noqa: E501
            (write.owner, now.isoformat(timespec="seconds")),
        )
        await self._conn.execute(
            """
            INSERT INTO miniapp_cards
                (owner, card_id, title, source, ir, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                write.owner,
                card_id,
                write.title,
                write.source,
                json.dumps(write.ir),
                now.isoformat(timespec="seconds"),
                (now + timedelta(days=CARD_TTL_DAYS)).isoformat(timespec="seconds"),
            ),
        )
        return card_id

    async def get_card(self, owner: str, card_id: str) -> dict[str, Any] | None:
        await self.initialize()
        row = await self._fetch_one(
            "SELECT card_id, title, source, ir, pinned, created_at, expires_at FROM miniapp_cards "  # noqa: E501
            "WHERE owner = ? AND card_id = ? AND (pinned = 1 OR expires_at >= ?)",
            (owner, card_id, _now()),
        )
        if row is None:
            return None
        return {
            "id": row["card_id"],
            "title": row["title"],
            "source": row["source"],
            "ir": json.loads(row["ir"]),
            "pinned": bool(row["pinned"]),
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
        }

    async def list_cards(
        self, owner: str, *, pinned_only: bool = False
    ) -> list[dict[str, Any]]:
        await self.initialize()
        rows = await self._fetch_all(
            _LIST_CARDS_PINNED_SQL if pinned_only else _LIST_CARDS_ALL_SQL,
            (owner, _now()),
        )
        return [
            {
                "id": r["card_id"],
                "title": r["title"],
                "pinned": bool(r["pinned"]),
                "created": r["created_at"],
            }
            for r in rows
        ]

    async def set_card_pinned(self, owner: str, card_id: str, pinned: bool) -> bool:
        await self.initialize()
        async with self._lock:
            cursor = await self._conn.execute(
                "UPDATE miniapp_cards SET pinned = ? WHERE owner = ? AND card_id = ?",
                (1 if pinned else 0, owner, card_id),
            )
            await self._conn.commit()
            return cursor.rowcount > 0

    async def delete_card(self, owner: str, card_id: str) -> bool:
        await self.initialize()
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM miniapp_cards WHERE owner = ? AND card_id = ?",
                (owner, card_id),
            )
            await self._conn.commit()
            return cursor.rowcount > 0

    # ---------------------------------------------------------- preferences

    async def get_pref(self, owner: str, key: str) -> Any:
        await self.initialize()
        row = await self._fetch_one(
            "SELECT value FROM miniapp_prefs WHERE owner = ? AND key = ?", (owner, key)
        )
        return json.loads(row["value"]) if row else None

    async def set_pref(self, owner: str, key: str, value: Any) -> None:
        await self.initialize()
        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO miniapp_prefs (owner, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner, key) DO UPDATE SET value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (owner, key, json.dumps(value), _now()),
            )
            await self._conn.commit()
