"""SQLite wiki storage: edits, revisions, owner isolation, and legacy migration."""

import sqlite3
from contextlib import closing
from pathlib import Path

from blacki.wiki.pages import PAGE_CHAR_LIMIT
from blacki.wiki.storage import WikiStore

PATH = "people/ana.md"
OWNER_TWO = 34


def _owner_revision(store: WikiStore, owner: int) -> int:
    with store.connect() as db:
        row = db.execute(
            "SELECT revision FROM wiki_owners WHERE owner=?", (owner,)
        ).fetchone()
    return int(row["revision"])


def test_new_owner_has_memory_enabled_and_an_empty_wiki(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki.db")
    assert store.enabled(42)
    assert store.list_pages(42) == []
    assert store.get_page(42, PATH) is None


def test_pause_and_resume_are_stored_per_owner(store: WikiStore) -> None:
    store.set_enabled(OWNER_TWO, False)
    assert not store.enabled(OWNER_TWO)
    assert store.enabled(12)
    store.set_enabled(OWNER_TWO, True)
    assert store.enabled(OWNER_TWO)


def test_create_needs_no_prior_read(store: WikiStore) -> None:
    outcome = store.edit(12, PATH, "# Ana\n", "", read=False)
    assert (outcome.status, outcome.chars, outcome.revision) == ("saved", 6, 1)
    page = store.get_page(12, PATH)
    assert page is not None
    assert (page.markdown, page.revision) == ("# Ana\n", 1)


def test_first_edit_for_a_new_owner_bumps_the_owner_revision(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki.db")
    assert store.edit(42, PATH, "# Ana", "", read=False).status == "saved"
    assert _owner_revision(store, 42) == 1


def test_editing_an_existing_page_requires_a_full_read(store: WikiStore) -> None:
    store.edit(12, PATH, "# Ana\n", "", read=False)
    outcome = store.edit(12, PATH, "- Likes tea", "", read=False)
    assert outcome.status == "read_required"
    assert outcome.chars == 6
    page = store.get_page(12, PATH)
    assert page is not None
    assert page.markdown == "# Ana\n"


def test_old_text_on_a_missing_page_is_not_found(store: WikiStore) -> None:
    outcome = store.edit(12, PATH, "new", "old", read=False)
    assert outcome.status == "not_found"
    assert outcome.hint
    assert store.get_page(12, PATH) is None


def test_append_increments_the_page_and_owner_revisions(store: WikiStore) -> None:
    store.edit(12, PATH, "# Ana", "", read=False)
    before = _owner_revision(store, 12)
    outcome = store.edit(12, PATH, "- Likes tea", "", read=True)
    assert (outcome.status, outcome.revision) == ("saved", 2)
    page = store.get_page(12, PATH)
    assert page is not None
    assert page.markdown == "# Ana\n- Likes tea"
    assert _owner_revision(store, 12) == before + 1


def test_replace_and_delete_text_inside_a_page(store: WikiStore) -> None:
    store.edit(12, PATH, "- Lives in Pune\n- Drinks tea\n", "", read=False)
    store.edit(12, PATH, "- Lives in Mumbai", "- Lives in Pune", read=True)
    store.edit(12, PATH, "", "- Drinks tea\n", read=True)
    page = store.get_page(12, PATH)
    assert page is not None
    assert page.markdown == "- Lives in Mumbai\n"


def test_emptying_a_page_deletes_it(store: WikiStore) -> None:
    store.edit(12, PATH, "only line", "", read=False)
    outcome = store.edit(12, PATH, "", "only line", read=True)
    assert (outcome.status, outcome.revision) == ("deleted", 0)
    assert store.get_page(12, PATH) is None


def test_an_unchanged_edit_saves_nothing_and_keeps_revisions(
    store: WikiStore,
) -> None:
    store.edit(12, PATH, "- Likes tea", "", read=False)
    before = _owner_revision(store, 12)
    outcome = store.edit(12, PATH, "- Likes tea", "- Likes tea", read=True)
    assert outcome.status == "unchanged"
    assert outcome.revision == 1
    assert _owner_revision(store, 12) == before


def test_no_match_and_ambiguous_edits_leave_the_page_unchanged(
    store: WikiStore,
) -> None:
    store.edit(12, PATH, "- tea\n- tea", "", read=False)
    assert store.edit(12, PATH, "x", "- milk", read=True).status == "no_match"
    assert store.edit(12, PATH, "x", "- tea", read=True).status == "ambiguous"
    page = store.get_page(12, PATH)
    assert page is not None
    assert page.markdown == "- tea\n- tea"


def test_an_edit_over_the_page_limit_is_not_saved(store: WikiStore) -> None:
    store.edit(12, PATH, "a" * PAGE_CHAR_LIMIT, "", read=False)
    outcome = store.edit(12, PATH, "b", "", read=True)
    assert outcome.status == "too_large"
    page = store.get_page(12, PATH)
    assert page is not None
    assert len(page.markdown) == PAGE_CHAR_LIMIT


def test_a_paused_owner_cannot_be_edited(store: WikiStore) -> None:
    store.set_enabled(12, False)
    outcome = store.edit(12, PATH, "# Ana", "", read=False)
    assert outcome.status == "disabled"
    assert store.get_page(12, PATH) is None


def test_pages_are_isolated_by_owner_and_listed_in_path_order(
    store: WikiStore,
) -> None:
    store.set_enabled(OWNER_TWO, True)
    store.edit(12, "work.md", "private", "", read=False)
    store.edit(12, "food.md", "tea", "", read=False)
    store.edit(OWNER_TWO, PATH, "other owner", "", read=False)
    assert [page.path for page in store.list_pages(12)] == ["food.md", "work.md"]
    assert [page.path for page in store.list_pages(OWNER_TWO)] == [PATH]
    assert store.get_page(OWNER_TWO, "work.md") is None


def test_connections_enable_secure_delete(store: WikiStore) -> None:
    with store.connect() as db:
        assert db.execute("PRAGMA secure_delete").fetchone()[0] == 1


def test_connect_drops_claim_tables_from_a_legacy_database(tmp_path: Path) -> None:
    path = tmp_path / "wiki.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE wiki_owners (
                owner INTEGER PRIMARY KEY, enabled INTEGER NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE wiki_claims (
                owner INTEGER NOT NULL, path TEXT NOT NULL, fact TEXT NOT NULL,
                claim TEXT NOT NULL, source TEXT NOT NULL, excerpt TEXT NOT NULL,
                source_date TEXT NOT NULL, disputed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (owner, path, fact, source)
            );
            CREATE TABLE wiki_evidence (
                owner INTEGER NOT NULL, path TEXT NOT NULL, fact TEXT NOT NULL,
                source TEXT NOT NULL, PRIMARY KEY(owner, path, fact, source)
            );
            CREATE TABLE wiki_links (
                owner INTEGER NOT NULL, path TEXT NOT NULL, fact TEXT NOT NULL,
                target TEXT NOT NULL, PRIMARY KEY(owner, path, fact, target)
            );
            CREATE TABLE wiki_forgotten (
                owner INTEGER NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY (owner, kind, value)
            );
            INSERT INTO wiki_owners VALUES (12, 0, 5);
            INSERT INTO wiki_claims VALUES
                (12, 'food.md', 'diet', 'Vegetarian.', 'tg:1', 'Vegetarian.',
                 '2026-11-08', 0);
            INSERT INTO wiki_forgotten VALUES (12, 'source', 'tg:1');
            """)

    store = WikiStore(path)
    assert not store.enabled(12)  # Pause state survives the migration.
    assert store.list_pages(12) == []
    with closing(sqlite3.connect(path)) as db:
        names = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        revision = db.execute(
            "SELECT revision FROM wiki_owners WHERE owner=12"
        ).fetchone()[0]
    assert {"wiki_claims", "wiki_evidence", "wiki_links", "wiki_forgotten"}.isdisjoint(
        names
    )
    assert {"wiki_owners", "wiki_pages"} <= names
    assert revision == 5
