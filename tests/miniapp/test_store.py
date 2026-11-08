from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite
import pytest

from blacki.miniapp.catalog import BUILTIN_PAGES
from blacki.miniapp.store import (
    CARD_TTL_DAYS,
    MAX_AGENT_PAGES,
    MAX_VERSIONS,
    MiniAppStore,
    StaleVersionError,
    builtin_source,
)


@pytest.fixture
async def store() -> Any:
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    lock = asyncio.Lock()
    st = MiniAppStore(conn, lock)
    await st.initialize()
    try:
        yield st
    finally:
        await conn.close()


def test_builtin_source() -> None:
    for page_id in BUILTIN_PAGES:
        src = builtin_source(page_id)
        assert isinstance(src, str)
        assert len(src) > 0


async def test_list_and_get_pages(store: MiniAppStore) -> None:
    owner = "user1"
    # Initially: 5 builtin pages + 1 "For you" agent page (home)
    pages = await store.list_pages(owner)
    assert len(pages) == 6
    assert [p["id"] for p in pages[:5]] == [
        "today",
        "health",
        "food",
        "training",
        "files",
    ]
    assert pages[5]["id"] == "foryou"
    assert pages[5]["kind"] == "agent"
    assert pages[5]["version"] == 0

    # Non-existent agent page
    assert await store.get_page(owner, "nonexistent") is None

    # Save an agent page
    ir = {"version": 1, "title": "My Page", "root": {"type": "Page"}}
    v1 = await store.save_page(
        owner,
        "mypage",
        title="My Page",
        source='root = Page("My Page", [])',
        ir=ir,
        base_version=None,
        created_by="agent",
        note="Initial version",
    )
    assert v1 == 1

    # Get page
    page = await store.get_page(owner, "mypage")
    assert page is not None
    assert page["id"] == "mypage"
    assert page["title"] == "My Page"
    assert page["version"] == 1
    assert page["source"] == 'root = Page("My Page", [])'
    assert page["ir"] == ir

    # List pages again: home page 'foryou' is inserted at index 5
    pages2 = await store.list_pages(owner)
    assert any(p["id"] == "mypage" for p in pages2)

    # Save page with base_version mismatch raises StaleVersionError
    with pytest.raises(StaleVersionError, match="is at version 1, not 0"):
        await store.save_page(
            owner,
            "mypage",
            title="My Page Updated",
            source="src2",
            ir=ir,
            base_version=0,
            created_by="user",
        )

    # Save page with matching base_version succeeds
    v2 = await store.save_page(
        owner,
        "mypage",
        title="My Page Updated",
        source="src2",
        ir=ir,
        base_version=1,
        created_by="user",
        note="Second version",
    )
    assert v2 == 2

    # Check history
    hist = await store.history(owner, "mypage")
    assert len(hist) == 2
    assert hist[0]["version"] == 2
    assert hist[0]["created_by"] == "user"
    assert hist[1]["version"] == 1
    assert hist[1]["created_by"] == "agent"

    # Version source
    vs2 = await store.version_source(owner, "mypage", 2)
    assert vs2 is not None
    assert vs2["title"] == "My Page Updated"
    assert vs2["source"] == "src2"

    # Test list_pages when AGENT_HOME_PAGE ('foryou') IS in seen
    await store.save_page(
        owner,
        "foryou",
        title="For you custom",
        source="src_foryou",
        ir=ir,
        base_version=None,
        created_by="agent",
    )
    pages_with_foryou = await store.list_pages(owner)
    foryou_entry = next(p for p in pages_with_foryou if p["id"] == "foryou")
    assert foryou_entry["title"] == "For you custom"
    assert foryou_entry["version"] == 1


async def test_save_page_max_limit_and_version_pruning(store: MiniAppStore) -> None:
    owner = "user_limit"
    ir = {"version": 1}

    # Create up to MAX_AGENT_PAGES
    for i in range(MAX_AGENT_PAGES):
        await store.save_page(
            owner,
            f"page_{i}",
            title=f"Page {i}",
            source=f"src_{i}",
            ir=ir,
            base_version=None,
            created_by="agent",
        )

    # Attempting to create one more raises ValueError
    with pytest.raises(ValueError, match=f"at most {MAX_AGENT_PAGES} Blacki pages"):
        await store.save_page(
            owner,
            "one_too_many",
            title="Extra",
            source="src",
            ir=ir,
            base_version=None,
            created_by="agent",
        )

    # But updating an existing page is allowed
    v_up = await store.save_page(
        owner,
        "page_0",
        title="Page 0 Updated",
        source="src_0_v2",
        ir=ir,
        base_version=1,
        created_by="user",
    )
    assert v_up == 2

    # Test version pruning: save page_0 more than MAX_VERSIONS times
    current_v = 2
    for _ in range(MAX_VERSIONS + 5):
        current_v = await store.save_page(
            owner,
            "page_0",
            title="Page 0 V",
            source="src",
            ir=ir,
            base_version=current_v,
            created_by="agent",
        )
    # Old versions should be pruned to MAX_VERSIONS
    hist = await store.history(owner, "page_0")
    assert len(hist) == MAX_VERSIONS
    assert hist[0]["version"] == current_v
    # Earliest version in history should be current_v - MAX_VERSIONS + 1
    assert hist[-1]["version"] == current_v - MAX_VERSIONS + 1


async def test_page_move_and_delete(store: MiniAppStore) -> None:
    owner = "user_move"
    ir = {"version": 1}
    await store.save_page(
        owner,
        "p1",
        title="P1",
        source="s1",
        ir=ir,
        base_version=None,
        created_by="agent",
    )

    # Move page
    assert await store.move_page(owner, "p1", 50) is True
    assert await store.move_page(owner, "nonexistent", 50) is False

    # Delete page
    assert await store.delete_page(owner, "p1") is True
    assert await store.delete_page(owner, "p1") is False  # already deleted
    assert await store.delete_page(owner, "nonexistent") is False

    # Deleted page is not returned by get_page or list_pages
    assert await store.get_page(owner, "p1") is None
    pages = await store.list_pages(owner)
    assert not any(p["id"] == "p1" for p in pages)

    # Re-saving a deleted page is allowed (version increments)
    v_new = await store.save_page(
        owner,
        "p1",
        title="P1 Restored",
        source="s1_new",
        ir=ir,
        base_version=None,
        created_by="agent",
    )
    assert v_new == 2
    assert (await store.get_page(owner, "p1")) is not None


async def test_cards_lifecycle(store: MiniAppStore) -> None:
    owner = "user_cards"
    ir = {"title": "Card"}

    # Create card
    cid1 = await store.create_card(owner, title="Card 1", source="src1", ir=ir)
    cid2 = await store.create_card(owner, title="Card 2", source="src2", ir=ir)

    # Get card
    c1 = await store.get_card(owner, cid1)
    assert c1 is not None
    assert c1["id"] == cid1
    assert c1["title"] == "Card 1"
    assert c1["source"] == "src1"
    assert c1["ir"] == ir
    assert c1["pinned"] is False
    assert c1["created_at"]
    assert c1["expires_at"]

    assert await store.get_card(owner, "missing_card") is None

    # List cards (all vs pinned)
    all_cards = await store.list_cards(owner)
    assert len(all_cards) == 2
    assert not any(c["pinned"] for c in all_cards)

    pinned_cards = await store.list_cards(owner, pinned_only=True)
    assert len(pinned_cards) == 0

    # Pin card 1
    assert await store.set_card_pinned(owner, cid1, True) is True
    assert await store.set_card_pinned(owner, "missing", True) is False

    c1_pinned = await store.get_card(owner, cid1)
    assert c1_pinned is not None
    assert c1_pinned["pinned"] is True

    pinned_after = await store.list_cards(owner, pinned_only=True)
    assert len(pinned_after) == 1
    assert pinned_after[0]["id"] == cid1

    # Unpin card 1
    assert await store.set_card_pinned(owner, cid1, False) is True

    # Delete card 2
    assert await store.delete_card(owner, cid2) is True
    assert await store.delete_card(owner, "missing") is False
    assert await store.get_card(owner, cid2) is None


async def test_card_expiration(store: MiniAppStore) -> None:
    owner = "user_exp"
    ir = {"card": True}
    cid = await store.create_card(owner, title="Expiring", source="src", ir=ir)

    # Manually expire the card in the database
    past = (datetime.now(UTC) - timedelta(days=CARD_TTL_DAYS + 1)).isoformat(
        timespec="seconds"
    )
    await store._conn.execute(
        "UPDATE miniapp_cards SET expires_at = ? WHERE card_id = ?", (past, cid)
    )
    await store._conn.commit()

    # Expired unpinned card should not be returned by get_card or list_cards
    assert await store.get_card(owner, cid) is None
    cards = await store.list_cards(owner)
    assert len(cards) == 0

    # If pinned, it should still be returned even if expired
    await store._conn.execute(
        "UPDATE miniapp_cards SET pinned = 1 WHERE card_id = ?", (cid,)
    )
    await store._conn.commit()
    assert (await store.get_card(owner, cid)) is not None
    assert len(await store.list_cards(owner)) == 1

    # When creating a new card, expired unpinned cards are pruned
    await store._conn.execute(
        "UPDATE miniapp_cards SET pinned = 0 WHERE card_id = ?", (cid,)
    )
    await store._conn.commit()
    await store.create_card(owner, title="New", source="src", ir=ir)
    # Check that old cid was deleted from database
    row = await store._fetch_one(
        "SELECT card_id FROM miniapp_cards WHERE card_id = ?", (cid,)
    )
    assert row is None


async def test_preferences(store: MiniAppStore) -> None:
    owner = "user_pref"
    assert await store.get_pref(owner, "theme") is None

    await store.set_pref(owner, "theme", "dark")
    assert await store.get_pref(owner, "theme") == "dark"

    # Complex object
    await store.set_pref(owner, "settings", {"notifications": True, "volume": 80})
    assert await store.get_pref(owner, "settings") == {
        "notifications": True,
        "volume": 80,
    }

    # Update preference
    await store.set_pref(owner, "theme", "light")
    assert await store.get_pref(owner, "theme") == "light"
