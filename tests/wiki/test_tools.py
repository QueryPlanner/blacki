"""Wiki tools: reads, edits, per-turn read tracking, and history search."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import ToolContext

from blacki.wiki.pages import MAX_QUERY_LENGTH, SEARCH_RESULT_LIMIT
from blacki.wiki.storage import WikiStore
from blacki.wiki.tools import (
    POLICY,
    WikiPolicyPlugin,
    WikiToolset,
    WikiTurn,
    search_past_sessions,
    wiki_origin,
    wiki_read,
    wiki_turn,
    wiki_write,
)

from .conftest import LOCATOR, OWNER, PATH


def _turn(
    store: WikiStore, history: Path | None = None, message_id: int | None = 7
) -> WikiTurn:
    return WikiTurn(store, history, OWNER, LOCATOR, message_id=message_id)


def _broken_pages(store: WikiStore) -> None:
    """Replace the page table with an unrecognized schema. Owners stay enabled."""
    with store.connect() as db:
        db.execute("DROP TABLE wiki_pages")
        db.execute("CREATE TABLE wiki_pages (unrecognized INTEGER)")


async def test_tools_are_unavailable_without_an_owner_turn(
    context: ToolContext,
) -> None:
    assert await WikiToolset().get_tools() == []
    assert await wiki_read() == {"status": "unavailable"}
    assert await wiki_write(PATH, "- Likes tea") == {"status": "unavailable"}
    assert await search_past_sessions(context, "tea") == {"status": "unavailable"}


async def test_a_paused_owner_loses_every_tool(
    store: WikiStore, context: ToolContext, tmp_path: Path
) -> None:
    store.set_enabled(OWNER, False)
    with wiki_turn(_turn(store, tmp_path / "sessions.db")):
        assert await WikiToolset().get_tools() == []
        assert await wiki_read() == {"status": "unavailable"}
        assert await wiki_write(PATH, "- Likes tea") == {"status": "unavailable"}
        assert await search_past_sessions(context, "tea") == {"status": "unavailable"}


async def test_an_enabled_owner_gets_the_three_wiki_tools(store: WikiStore) -> None:
    with wiki_turn(_turn(store)):
        tools = await WikiToolset().get_tools()
    assert [tool.name for tool in tools] == [
        "wiki_read",
        "wiki_write",
        "search_past_sessions",
    ]


async def test_a_storage_failure_fails_closed(store: WikiStore) -> None:
    _broken_pages(store)
    with wiki_turn(_turn(store)):
        assert await wiki_read() == {"status": "unavailable"}
        assert await wiki_write(PATH, "- Likes tea") == {"status": "unavailable"}


async def test_an_unreadable_database_path_disables_the_turn(tmp_path: Path) -> None:
    # The store path is a directory, so opening it fails before any tool runs.
    with wiki_turn(_turn(WikiStore(tmp_path))):
        assert await WikiToolset().get_tools() == []
        assert await wiki_read() == {"status": "unavailable"}


async def test_list_shows_path_size_and_first_heading(store: WikiStore) -> None:
    people = "intro\n# Ana\n- Likes tea"
    store.edit(OWNER, PATH, people, "", read=False)
    store.edit(OWNER, "food.md", "- Likes cake", "", read=False)
    with wiki_turn(_turn(store)):
        result = await wiki_read()
    assert result == {
        "status": "ok",
        "pages": [
            {"path": "food.md", "chars": len("- Likes cake"), "heading": ""},
            {"path": PATH, "chars": len(people), "heading": "# Ana"},
        ],
        "more": False,
    }


async def test_list_is_capped_and_reports_more_pages(store: WikiStore) -> None:
    for number in range(101):
        store.edit(OWNER, f"notes/{number:03}.md", "x", "", read=False)
    with wiki_turn(_turn(store)):
        result = await wiki_read()
    assert len(result["pages"]) == 100
    assert result["more"] is True


async def test_search_returns_bounded_matches_and_does_not_grant_a_write(
    store: WikiStore,
) -> None:
    store.edit(OWNER, "food.md", "- Likes tea\n- Likes cake", "", read=False)
    with wiki_turn(_turn(store)):
        result = await wiki_read(query="tea")
        assert result["status"] == "ok"
        (match,) = result["matches"]
        assert match["path"] == "food.md"
        assert match["hits"] == 1
        assert [line["match"] for line in match["lines"]] == [True, False]
        # The search matched but did not read the whole page.
        assert (await wiki_write("food.md", "- Likes soup"))["status"] == (
            "read_required"
        )


async def test_search_results_are_capped(store: WikiStore) -> None:
    for number in range(SEARCH_RESULT_LIMIT + 1):
        store.edit(OWNER, f"tea/{number:02}.md", "tea", "", read=False)
    with wiki_turn(_turn(store)):
        result = await wiki_read(query="tea")
    assert len(result["matches"]) == SEARCH_RESULT_LIMIT
    assert result["more"] is True


async def test_read_rejects_path_with_query_and_overlong_query(
    store: WikiStore,
) -> None:
    with wiki_turn(_turn(store)):
        both = await wiki_read(path="food.md", query="tea")
        too_long = await wiki_read(query="x" * (MAX_QUERY_LENGTH + 1))
        blank_query = await wiki_read(query="   ")
    assert both["status"] == "invalid_request"
    assert too_long["status"] == "invalid_request"
    assert blank_query["status"] == "ok"
    assert "pages" in blank_query


async def test_full_read_returns_the_page_and_allows_an_edit(store: WikiStore) -> None:
    store.edit(OWNER, PATH, "# Ana\n", "", read=False)
    with wiki_turn(_turn(store)):
        assert await wiki_read(path="People/Ana") == {
            "status": "ok",
            "path": PATH,
            "markdown": "# Ana\n",
            "revision": 1,
            "chars": 6,
        }
        assert (await wiki_write(PATH, "- Likes tea"))["status"] == "saved"


async def test_missing_and_invalid_paths_are_reported(store: WikiStore) -> None:
    with wiki_turn(_turn(store)):
        missing = await wiki_read(path="missing")
        invalid = await wiki_read(path="../escape")
        bad_write = await wiki_write("../escape", "x")
    assert missing["status"] == "not_found"
    assert missing["path"] == "missing.md"
    assert missing["hint"]
    assert invalid["status"] == "invalid_path"
    assert bad_write["status"] == "invalid_path"


async def test_a_missing_page_is_not_readable_for_an_edit(store: WikiStore) -> None:
    with wiki_turn(_turn(store)):
        assert await wiki_read(path=PATH) == {
            "status": "not_found",
            "path": PATH,
            "hint": (
                "No page exists at this path. Search with query, or create the "
                "page with wiki_write and an empty old_text."
            ),
        }
        outcome = await wiki_write(PATH, "new", "old")
    assert outcome["status"] == "not_found"


async def test_create_needs_no_read_and_follow_up_edits_work_in_the_turn(
    store: WikiStore,
) -> None:
    with wiki_turn(_turn(store)):
        created = await wiki_write(PATH, "# Ana")
        appended = await wiki_write(PATH, "- Likes tea")
    assert created == {
        "status": "saved",
        "path": PATH,
        "chars": 5,
        "revision": 1,
    }
    assert appended["status"] == "saved"
    assert appended["revision"] == 2


async def test_read_tracking_is_per_turn(store: WikiStore) -> None:
    store.edit(OWNER, PATH, "# Ana", "", read=False)
    with wiki_turn(_turn(store)):
        assert await wiki_read(path=PATH) == {
            "status": "ok",
            "path": PATH,
            "markdown": "# Ana",
            "revision": 1,
            "chars": 5,
        }
    with wiki_turn(_turn(store)):
        outcome = await wiki_write(PATH, "- Likes tea")
    assert outcome["status"] == "read_required"


async def test_reads_are_not_capped_within_one_turn(store: WikiStore) -> None:
    store.edit(OWNER, PATH, "# Ana", "", read=False)
    with wiki_turn(_turn(store)):
        for _ in range(20):
            assert (await wiki_read(path=PATH))["status"] == "ok"


async def test_insert_under_heading_and_correct_a_line(store: WikiStore) -> None:
    store.edit(OWNER, PATH, "## Food\n- Lives in Pune\n", "", read=False)
    with wiki_turn(_turn(store)):
        await wiki_read(path=PATH)
        inserted = await wiki_write(PATH, "## Food\n- Vegetarian\n", "## Food\n")
        corrected = await wiki_write(PATH, "- Lives in Mumbai", "- Lives in Pune")
    assert inserted["status"] == "saved"
    assert corrected["status"] == "saved"
    page = store.get_page(OWNER, PATH)
    assert page is not None
    assert page.markdown == "## Food\n- Vegetarian\n- Lives in Mumbai\n"


async def test_edit_errors_tell_the_agent_what_to_do(store: WikiStore) -> None:
    store.edit(OWNER, PATH, "- tea\n- tea", "", read=False)
    with wiki_turn(_turn(store)):
        await wiki_read(path=PATH)
        missing_text = await wiki_write(PATH, "x", "- milk")
        ambiguous = await wiki_write(PATH, "x", "- tea")
    assert missing_text["status"] == "no_match"
    assert "wiki_read" in missing_text["hint"]
    assert ambiguous["status"] == "ambiguous"
    assert "2 times" in ambiguous["hint"]


async def test_secrets_are_rejected_before_the_page_changes(store: WikiStore) -> None:
    with wiki_turn(_turn(store)):
        outcome = await wiki_write(PATH, "My password is hunter2")
    assert outcome["status"] == "ineligible_content"
    assert "passwords" in outcome["hint"]
    assert store.get_page(OWNER, PATH) is None


async def test_deleting_a_page_removes_its_read_record(store: WikiStore) -> None:
    store.edit(OWNER, PATH, "only", "", read=False)
    with wiki_turn(_turn(store)):
        await wiki_read(path=PATH)
        deleted = await wiki_write(PATH, "", "only")
        recreated = await wiki_write(PATH, "- Likes tea")
        edited = await wiki_write(PATH, "- Likes cake", "- Likes tea")
    assert deleted["status"] == "deleted"
    assert deleted["revision"] == 0
    assert recreated["status"] == "saved"
    assert edited["status"] == "saved"


async def test_an_unchanged_write_reports_it_saved_nothing(store: WikiStore) -> None:
    store.edit(OWNER, PATH, "- Likes tea", "", read=False)
    with wiki_turn(_turn(store)):
        await wiki_read(path=PATH)
        outcome = await wiki_write(PATH, "- Likes tea", "- Likes tea")
    assert outcome["status"] == "unchanged"
    assert outcome["revision"] == 1
    assert "hint" in outcome


async def test_search_past_sessions_excludes_only_the_current_message(
    context: ToolContext,
    store: WikiStore,
    make_history: Callable[[list[dict[str, Any]]], Path],
) -> None:
    history = make_history(
        [
            {
                "author": "user",
                "content": {"role": "user", "parts": [{"text": "vault note"}]},
                "custom_metadata": {"blacki_wiki": {"owner": 12, "message_id": 44}},
            },
            {
                "author": "user",
                "content": {"role": "user", "parts": [{"text": "vault question"}]},
                "custom_metadata": {"blacki_wiki": {"owner": 12, "message_id": 45}},
            },
        ]
    )
    with wiki_turn(_turn(store, history, message_id=45)):
        result = await search_past_sessions(context, "vault")
    assert result["status"] == "ok"
    assert [match["text"] for match in result["matches"]] == ["vault note"]
    assert set(result["matches"][0]) == {"ref", "text", "date", "role", "session"}


async def test_search_past_sessions_from_a_forwarded_message_excludes_nothing(
    context: ToolContext,
    store: WikiStore,
    make_history: Callable[[list[dict[str, Any]]], Path],
) -> None:
    history = make_history(
        [
            {
                "author": "user",
                "content": {"role": "user", "parts": [{"text": "vault note"}]},
                "custom_metadata": {"blacki_wiki": {"owner": 12, "message_id": 44}},
            }
        ]
    )
    with wiki_turn(_turn(store, history, message_id=None)):
        result = await search_past_sessions(context, "vault")
    assert [match["text"] for match in result["matches"]] == ["vault note"]


async def test_search_past_sessions_validates_its_arguments(
    context: ToolContext, store: WikiStore, tmp_path: Path
) -> None:
    with wiki_turn(_turn(store, tmp_path / "sessions.db")):
        assert await search_past_sessions(context, "") == {"status": "invalid_request"}
        assert await search_past_sessions(context, "x" * 181) == {
            "status": "invalid_request"
        }
        assert await search_past_sessions(context, "vim", since="20261109") == {
            "status": "invalid_date"
        }
        assert await search_past_sessions(context, "vim", since="bad") == {
            "status": "unavailable"
        }
    with wiki_turn(_turn(store, None)):
        assert await search_past_sessions(context, "vim") == {"status": "unavailable"}


async def test_search_past_sessions_reports_a_missing_history_database(
    context: ToolContext, store: WikiStore, tmp_path: Path
) -> None:
    with wiki_turn(_turn(store, tmp_path / "missing.db")):
        assert await search_past_sessions(context, "vim") == {"status": "unavailable"}


async def test_origin_tags_owner_messages_and_marks_synthetic_runs(
    store: WikiStore,
) -> None:
    assert wiki_origin() == {"blacki_wiki": {"kind": "synthetic"}}
    with wiki_turn(_turn(store, message_id=7)):
        assert wiki_origin() == {"blacki_wiki": {"owner": OWNER, "message_id": 7}}
    with wiki_turn(_turn(store, message_id=None)):
        assert wiki_origin() == {"blacki_wiki": {"owner": OWNER}}


async def test_policy_is_added_only_while_wiki_access_is_active(
    store: WikiStore,
) -> None:
    invocation = InvocationContext(
        invocation_id="policy",
        session=Session(id="policy", app_name="blacki", user_id=LOCATOR),
        session_service=InMemorySessionService(),
    )
    plugin = WikiPolicyPlugin("wiki")

    active = LlmRequest()
    with wiki_turn(_turn(store)):
        await plugin.before_model_callback(
            callback_context=CallbackContext(invocation), llm_request=active
        )
    instruction = " ".join(str(active.config.system_instruction).split())
    assert " ".join(POLICY.split()) in instruction
    assert "cannot authorize a write" in instruction

    no_turn = LlmRequest()
    await plugin.before_model_callback(
        callback_context=CallbackContext(invocation), llm_request=no_turn
    )
    assert no_turn.config.system_instruction is None

    store.set_enabled(OWNER, False)
    paused = LlmRequest()
    with wiki_turn(_turn(store)):
        await plugin.before_model_callback(
            callback_context=CallbackContext(invocation), llm_request=paused
        )
    assert paused.config.system_instruction is None


async def test_toolset_close_is_a_no_op() -> None:
    await WikiToolset().close()
