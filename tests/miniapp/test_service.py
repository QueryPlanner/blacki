from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import aiosqlite
import pytest

from blacki.container import AppContainer
from blacki.declarative_db.storage import SqliteDeclarativeDbStorage
from blacki.health.storage import (
    SqliteGoogleHealthStorage,
)
from blacki.miniapp.catalog import AGENT_HOME_PAGE
from blacki.miniapp.lang import LangError
from blacki.miniapp.service import (
    ActionError,
    MiniAppService,
    _param_text,
    _retitle,
    _shift,
    _split_blocks,
    _walk_actions,
    get_miniapp_service,
    hooks,
    reset_miniapp_service,
)
from blacki.miniapp.sources import MiniAppUser
from blacki.miniapp.store import MiniAppStore
from blacki.user_files.service import UserFileService


@pytest.fixture
async def app_container(tmp_path: Path) -> AsyncGenerator[AppContainer, None]:
    db_path = tmp_path / "service_test.db"
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    lock = asyncio.Lock()

    container = AppContainer(conn, lock)

    await container.calorie_storage.initialize()
    await container.workout_storage.initialize()
    await container.declarative_db_storage.initialize()

    real_gh = SqliteGoogleHealthStorage(conn, lock)
    gh_storage = create_autospec(real_gh, spec_set=True)
    container._google_health_storage = gh_storage

    try:
        yield container
    finally:
        await conn.close()


@pytest.fixture
async def miniapp_store(app_container: AppContainer) -> MiniAppStore:
    lock = asyncio.Lock()
    st = MiniAppStore(app_container.conn, lock)
    await st.initialize()
    return st


@pytest.fixture
def service(app_container: AppContainer, miniapp_store: MiniAppStore) -> MiniAppService:
    return MiniAppService(app_container, miniapp_store)


def test_service_singletons(app_container: AppContainer) -> None:
    reset_miniapp_service()
    with patch("blacki.miniapp.service.get_container", return_value=app_container):
        s1 = get_miniapp_service()
        s2 = get_miniapp_service()
        assert s1 is s2
    reset_miniapp_service()


def test_helpers() -> None:
    from blacki.miniapp.sources import SourceParam

    assert _param_text(SourceParam("enum", options=("a", "b"))) == "one of a, b"
    assert (
        _param_text(SourceParam("int", default=5, min=1, max=10))
        == "whole number 1..10, default 5"
    )
    assert _param_text(SourceParam("text")) == "lowercase name"

    err_with_line = LangError("bad syntax", line=5)
    assert _shift(err_with_line, 10) == "line 15: bad syntax"
    err_no_line = LangError("generic error", line=None)
    assert _shift(err_no_line, 10) == "generic error"

    # _walk_actions
    tree = {
        "title": "Hi",
        "btn": {"a": "SaveTarget", "args": {"kind": "steps", "value": 10000}},
        "nested": [{"a": "Log", "args": {"template": "run"}}],
    }
    actions = _walk_actions(tree)
    assert len(actions) == 2

    # _retitle
    from blacki.miniapp.lang import parse

    p = parse('root = Page("Old Title", [])')
    _retitle(p, "New Title")
    assert 'Page("New Title", [])' in p.to_source()

    p_non_page = parse("root = Stack([])")
    _retitle(p_non_page, "Ignored")  # Should not raise


def test_split_blocks_syntax() -> None:
    # 1. Start without command
    with pytest.raises(LangError, match="start with a command"):
        _split_blocks("x = 10\npage foo")

    # 2. Empty code
    with pytest.raises(LangError, match="nothing to do"):
        _split_blocks("# just a comment")

    # 3. Invalid page id (line 753)
    # _COMMAND regex matches [a-z][a-z0-9-]{1,31}. If _COMMAND allows broader chars:
    import re

    import blacki.miniapp.service as svc

    orig_cmd = svc._COMMAND
    try:
        svc._COMMAND = re.compile(
            r"^(?:(page)\s+([A-Za-z0-9_-]{1,31})(?:\s+\"([^\"]{1,60})\")?(\s+replace)?"
            r"|(card)(?:\s+\"([^\"]{1,60})\")?"
            r"|(remove)\s+([A-Za-z_][A-Za-z0-9_\s,$]*)"
            r"|(delete)\s+page\s+([a-z][a-z0-9-]{1,31})"
            r"|(revert)\s+([a-z][a-z0-9-]{1,31})\s+to\s+(\d{1,4})"
            r"|(move)\s+([a-z][a-z0-9-]{1,31})\s+to\s+(\d{1,3})"
            r"|(note)\s+\"([^\"]{1,120})\""
            r"|(noop))\s*$"
        )
        with pytest.raises(LangError, match="page ids are lowercase letters"):
            _split_blocks('page Invalid_Page "Title"\nx = 1')
        # Also exercise branch 785->723 (where match succeeds without g[16])
        blocks_noop = _split_blocks(
            'noop\npage custom "Custom"\nroot = Page("Custom", [])'
        )
        assert len(blocks_noop) == 1
    finally:
        svc._COMMAND = orig_cmd

    # Test line 731 raised for invalid command start:
    with pytest.raises(LangError, match="start with a command such as"):
        _split_blocks('page INVALID_PAGE "Title"\nx = 1')

    # 4. Remove outside page
    with pytest.raises(LangError, match="remove belongs inside a page block"):
        _split_blocks('remove x\npage foo "Foo"')

    with pytest.raises(LangError, match="remove belongs inside a page block"):
        _split_blocks('card "Card"\nremove x')

    # 5. Note outside page
    with pytest.raises(LangError, match="note belongs inside a page block"):
        _split_blocks('note "a note"')

    # 6. Valid full script
    script = """
    # Setup
    page custom "My Custom Page" replace
        note "Initial version"
        q = Query("health.days", {days: 7})
        root = Page("My Custom Page", [])
        remove old_item, another_item

    card "Summary Card"
        root = Page("Summary Card", [])

    move custom to 1
    revert custom to 1
    delete page custom
    """
    blocks = _split_blocks(script)
    assert len(blocks) == 5
    assert blocks[0]["command"] == "page"
    assert blocks[0]["target"] == "custom"
    assert blocks[0]["replace"] is True
    assert blocks[0]["note"] == "Initial version"
    assert blocks[0]["remove"] == ["old_item", "another_item"]
    assert blocks[1]["command"] == "card"
    assert blocks[2]["command"] == "move"
    assert blocks[2]["position"] == 1
    assert blocks[3]["command"] == "revert"
    assert blocks[3]["version"] == 1
    assert blocks[4]["command"] == "delete"


async def test_pages_and_nav(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # 1. Builtin IR caching
    ir1 = service.builtin_ir("today")
    ir2 = service.builtin_ir("today")
    assert ir1 is ir2

    # 2. Load builtin page
    loaded_today = await service.load(user, "today")
    assert loaded_today["id"] == "today"
    assert loaded_today["kind"] == "builtin"
    assert "data" in loaded_today

    # 3. Load agent home page default when not created yet
    foryou = await service.load(user, AGENT_HOME_PAGE)
    assert foryou["id"] == AGENT_HOME_PAGE
    assert foryou["kind"] == "agent"
    assert foryou["version"] == 0

    # 4. Load nonexistent agent page -> LookupError
    with pytest.raises(LookupError, match="page not found"):
        await service.load(user, "nonexistent")

    # 5. Nav
    nav_items = await service.nav(user)
    assert len(nav_items) == 6  # 5 builtins + foryou


async def test_cards_load_and_pinned(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # 1. Nonexistent card
    with pytest.raises(LookupError, match="card not found"):
        await service.load_card(user, "c_none")

    # 2. Create card via write tool
    res = await service.tool_write(
        user, 'card "Sleep Card"\nroot = Page("Sleep Card", [Header("Hi")])', None
    )
    assert res["status"] == "ok"
    card_id = res["results"][0]["card_id"]

    # 3. Load card
    loaded = await service.load_card(user, card_id)
    assert loaded["id"] == card_id
    assert loaded["title"] == "Sleep Card"
    assert loaded["pinned"] == 0

    # 4. Pin card and test _pinned_cards resolution
    await service.run_action(
        user, {"action": "Pin", "card": card_id, "args": {"pinned": True}}
    )
    pinned = await service._pinned_cards(user)
    assert len(pinned) == 1
    assert pinned[0]["id"] == card_id

    # Test _pinned_cards when card disappears (line 123)
    orig_get_card = service.store.get_card
    try:
        service.store.get_card = AsyncMock(return_value=None)  # type: ignore[method-assign]
        pinned_missing = await service._pinned_cards(user)
        assert len(pinned_missing) == 0
    finally:
        service.store.get_card = orig_get_card  # type: ignore[method-assign]

    # Test card deletion and unpin action
    del_res = await service.run_action(
        user, {"action": "card.delete", "args": {"id": card_id}}
    )
    assert del_res["status"] == "ok"

    # Delete again -> ActionError
    with pytest.raises(ActionError, match="card not found"):
        await service.run_action(
            user, {"action": "card.delete", "args": {"id": card_id}}
        )

    # Pin nonexistent -> ActionError
    with pytest.raises(ActionError, match="card not found"):
        await service.run_action(user, {"action": "Pin", "card": card_id})


async def test_run_action_save_target(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # Invalid kind
    with pytest.raises(ActionError, match="invalid target"):
        await service.run_action(
            user, {"action": "SaveTarget", "args": {"kind": "invalid", "value": 100}}
        )

    # Invalid value type (bool)
    with pytest.raises(ActionError, match="invalid target"):
        await service.run_action(
            user, {"action": "SaveTarget", "args": {"kind": "steps", "value": True}}
        )

    # Out of range
    with pytest.raises(ActionError, match="target is out of range"):
        await service.run_action(
            user, {"action": "SaveTarget", "args": {"kind": "steps", "value": -1}}
        )

    with pytest.raises(ActionError, match="target is out of range"):
        await service.run_action(
            user, {"action": "SaveTarget", "args": {"kind": "steps", "value": 100001}}
        )

    # Valid
    res = await service.run_action(
        user, {"action": "SaveTarget", "args": {"kind": "steps", "value": 10000}}
    )
    assert res["status"] == "ok"
    assert "10000" in res["message"]


async def test_run_action_tracker(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # 1. Invalid template name or values
    with pytest.raises(ActionError, match="invalid tracker entry"):
        await service.run_action(
            user, {"action": "Log", "args": {"template": "BAD!", "values": {}}}
        )

    with pytest.raises(ActionError, match="invalid tracker entry"):
        await service.run_action(
            user,
            {"action": "Log", "args": {"template": "valid", "values": "not a dict"}},
        )

    # 2. Too many values or value too large
    too_many = {f"k{i}": i for i in range(13)}
    with pytest.raises(ActionError, match="tracker values are too large"):
        await service.run_action(
            user, {"action": "Log", "args": {"template": "valid", "values": too_many}}
        )

    with pytest.raises(ActionError, match="tracker values are too large"):
        await service.run_action(
            user,
            {
                "action": "Log",
                "args": {"template": "valid", "values": {"k": "x" * 501}},
            },
        )

    with pytest.raises(ActionError, match="tracker values are too large"):
        await service.run_action(
            user,
            {"action": "Log", "args": {"template": "valid", "values": {"k": [1, 2]}}},
        )

    # 3. Nonexistent or non-INSERT template
    with pytest.raises(ActionError, match="that tracker can't be logged from the app"):
        await service.run_action(
            user,
            {"action": "Log", "args": {"template": "not_there", "values": {"k": 1}}},
        )

    await service.container.conn.execute(
        "INSERT INTO saved_query_templates "
        "(user_id, template_name, table_name, query_type) VALUES (?, ?, ?, ?)",
        (user.chat_user_id, "sel_temp", "t", "SELECT"),
    )
    await service.container.conn.commit()

    with pytest.raises(ActionError, match="that tracker can't be logged from the app"):
        await service.run_action(
            user,
            {"action": "Log", "args": {"template": "sel_temp", "values": {"k": 1}}},
        )

    # 4. Valid INSERT template but execute raises ValueError
    decl_mock = create_autospec(SqliteDeclarativeDbStorage, spec_set=True)
    service.container._declarative_db_storage = decl_mock
    await service.container.conn.execute(
        "INSERT INTO saved_query_templates "
        "(user_id, template_name, table_name, query_type) VALUES (?, ?, ?, ?)",
        (user.chat_user_id, "ins_temp", "t", "INSERT"),
    )
    await service.container.conn.commit()

    decl_mock.execute_query_template.side_effect = ValueError("Insert failed")
    with pytest.raises(ActionError, match="Insert failed"):
        await service.run_action(
            user,
            {"action": "Log", "args": {"template": "ins_temp", "values": {"k": 1}}},
        )

    # 5. Success
    decl_mock.execute_query_template.side_effect = None
    res = await service.run_action(
        user, {"action": "Log", "args": {"template": "ins_temp", "values": {"k": 1}}}
    )
    assert res["status"] == "ok"


async def test_run_action_ask_blacki(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # 1. Invalid text
    with pytest.raises(ActionError, match="message is empty or too long"):
        await service.run_action(user, {"action": "AskBlacki", "args": {"text": ""}})

    with pytest.raises(ActionError, match="message is empty or too long"):
        await service.run_action(
            user, {"action": "AskBlacki", "args": {"text": "a" * 301}}
        )

    # 2. Hooks send_prompt is None
    orig_prompt = hooks.send_prompt
    hooks.send_prompt = None
    try:
        with pytest.raises(ActionError, match="Blacki's Telegram chat isn't running"):
            await service.run_action(
                user, {"action": "AskBlacki", "args": {"text": "Hello"}}
            )
    finally:
        hooks.send_prompt = orig_prompt

    # 3. Success and Rate Limiting
    sent: list[tuple[int, str]] = []

    async def _send(uid: int, text: str) -> None:
        sent.append((uid, text))

    hooks.send_prompt = _send
    try:
        res = await service.run_action(
            user, {"action": "AskBlacki", "args": {"text": "Hello 1"}}
        )
        assert res["status"] == "ok"
        assert len(sent) == 1

        # Second call immediately (< 10s) -> rate limit
        with pytest.raises(ActionError, match="wait a moment before asking again"):
            await service.run_action(
                user, {"action": "AskBlacki", "args": {"text": "Hello 2"}}
            )

        # Fill with 3 prompts within 60s
        service._prompt_times[user.telegram_user_id] = [
            time.monotonic() - 30,
            time.monotonic() - 20,
            time.monotonic() - 11,
        ]
        with pytest.raises(ActionError, match="wait a moment before asking again"):
            await service.run_action(
                user, {"action": "AskBlacki", "args": {"text": "Hello 4"}}
            )
    finally:
        hooks.send_prompt = orig_prompt


async def test_run_action_builtins_guard_and_meals(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # 1. Destructive actions rejected if not on builtin page
    for act in ("meal.edit", "meal.delete", "file.delete", "workout.save"):
        with pytest.raises(ActionError, match="only available on built-in pages"):
            await service.run_action(
                user, {"action": act, "page": "my-page", "args": {}}
            )

    # 2. Meal edit & delete: invalid meal id or meal not found
    with pytest.raises(ActionError, match="invalid meal"):
        await service.run_action(
            user, {"action": "meal.edit", "page": "food", "args": {"id": -1}}
        )

    with pytest.raises(ActionError, match="meal not found"):
        await service.run_action(
            user, {"action": "meal.edit", "page": "food", "args": {"id": 9999}}
        )

    # 3. Insert real meal
    await service.container.conn.execute(
        "INSERT INTO calorie_logs "
        "(id, user_id, logged_date, logged_at, description, calories, "
        "protein_g, carbs_g, fat_g, meal_type) "
        "VALUES (50, ?, '2026-05-10', '2026-05-10T12:00:00', 'Salad', "
        "200, 10, 20, 5, 'lunch')",
        (user.chat_user_id,),
    )
    await service.container.conn.commit()

    # 4. Meal edit validation
    with pytest.raises(ActionError, match="kcal is out of range"):
        await service.run_action(
            user,
            {
                "action": "meal.edit",
                "page": "food",
                "args": {"id": 50, "changes": {"kcal": -10}},
            },
        )

    with pytest.raises(ActionError, match="meal name is empty or too long"):
        await service.run_action(
            user,
            {
                "action": "meal.edit",
                "page": "food",
                "args": {"id": 50, "changes": {"name": ""}},
            },
        )

    with pytest.raises(ActionError, match="nothing to save"):
        await service.run_action(
            user,
            {"action": "meal.edit", "page": "food", "args": {"id": 50, "changes": {}}},
        )

    # 5. Mock meal service for edit & delete success
    from unittest.mock import AsyncMock

    meal_service_mock = MagicMock()
    meal_service_mock.mutate = AsyncMock(return_value=(50, "synced"))
    with patch(
        "blacki.calories.service.get_meal_service", return_value=meal_service_mock
    ):
        res_edit = await service.run_action(
            user,
            {
                "action": "meal.edit",
                "page": "food",
                "args": {
                    "id": 50,
                    "changes": {
                        "name": "Big Salad",
                        "kcal": 300,
                        "protein": 15,
                        "carbs": 25,
                        "fat": 8,
                    },
                },
            },
        )
        assert res_edit["status"] == "ok"
        assert res_edit["sync"] == "synced"

        # Mutate error
        meal_service_mock.mutate.side_effect = ValueError("Mutate error")
        with pytest.raises(ActionError, match="Mutate error"):
            await service.run_action(
                user,
                {
                    "action": "meal.edit",
                    "page": "food",
                    "args": {"id": 50, "changes": {"kcal": 350}},
                },
            )

        with pytest.raises(ActionError, match="Mutate error"):
            await service.run_action(
                user, {"action": "meal.delete", "page": "food", "args": {"id": 50}}
            )

        # Mutate delete success
        meal_service_mock.mutate.side_effect = None
        res_del = await service.run_action(
            user, {"action": "meal.delete", "page": "food", "args": {"id": 50}}
        )
        assert res_del["status"] == "ok"


async def test_run_action_files_workout_theme_unknown(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # 1. file.delete
    with pytest.raises(ActionError, match="invalid file"):
        await service.run_action(
            user, {"action": "file.delete", "page": "files", "args": {"id": "BAD ID!"}}
        )

    user_file_mock = create_autospec(UserFileService, spec_set=True)
    with patch(
        "blacki.user_files.service.get_user_file_service", return_value=user_file_mock
    ):
        user_file_mock.delete.side_effect = RuntimeError("R2 unavailable")
        with pytest.raises(ActionError, match="R2 unavailable"):
            await service.run_action(
                user, {"action": "file.delete", "page": "files", "args": {"id": "f1"}}
            )

        user_file_mock.delete.side_effect = Exception("General error")
        with pytest.raises(ActionError, match="the file may be only partly deleted"):
            await service.run_action(
                user, {"action": "file.delete", "page": "files", "args": {"id": "f1"}}
            )

        user_file_mock.delete.side_effect = None
        user_file_mock.delete.return_value = False
        with pytest.raises(ActionError, match="file not found"):
            await service.run_action(
                user, {"action": "file.delete", "page": "files", "args": {"id": "f1"}}
            )

        user_file_mock.delete.return_value = True
        res_file = await service.run_action(
            user, {"action": "file.delete", "page": "files", "args": {"id": "f1"}}
        )
        assert res_file["status"] == "ok"

    # 2. workout.save
    with pytest.raises(ActionError, match="invalid workout"):
        await service.run_action(
            user,
            {
                "action": "workout.save",
                "page": "training",
                "args": {"exercise": "", "sets": []},
            },
        )

    with pytest.raises(ActionError, match="invalid set"):
        await service.run_action(
            user,
            {
                "action": "workout.save",
                "page": "training",
                "args": {"exercise": "Bench", "sets": ["not a dict"]},
            },
        )

    with pytest.raises(ActionError, match="sets need kg 0..500 and reps 1..100"):
        await service.run_action(
            user,
            {
                "action": "workout.save",
                "page": "training",
                "args": {"exercise": "Bench", "sets": [{"kg": -1, "reps": 10}]},
            },
        )

    with (
        patch(
            "blacki.tools.workouts.log_training_for_user",
            return_value={"status": "error", "message": "Failed to log"},
        ),
        pytest.raises(ActionError, match="Failed to log"),
    ):
        await service.run_action(
            user,
            {
                "action": "workout.save",
                "page": "training",
                "args": {"exercise": "Bench", "sets": [{"kg": 80, "reps": 8}]},
            },
        )

    with patch(
        "blacki.tools.workouts.log_training_for_user",
        return_value={"status": "success"},
    ):
        res_w = await service.run_action(
            user,
            {
                "action": "workout.save",
                "page": "training",
                "args": {"exercise": "Bench", "sets": [{"kg": 80, "reps": 8}]},
            },
        )
        assert res_w["status"] == "ok"

    # 3. prefs.theme
    with pytest.raises(ActionError, match="invalid theme"):
        await service.run_action(
            user, {"action": "prefs.theme", "args": {"theme": "not a dict"}}
        )

    with pytest.raises(ActionError, match="invalid theme"):
        await service.run_action(
            user, {"action": "prefs.theme", "args": {"theme": {"k": "x" * 4001}}}
        )

    res_theme = await service.run_action(
        user, {"action": "prefs.theme", "args": {"theme": {"dark": True}}}
    )
    assert res_theme["status"] == "ok"

    # 4. Unknown action
    with pytest.raises(ActionError, match="unknown action"):
        await service.run_action(user, {"action": "unknown.action"})


async def test_tool_read(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # 1. help / empty
    res = await service.tool_read(user, "")
    assert res["status"] == "ok"
    assert "commands" in res

    res = await service.tool_read(user, "help")
    assert res["status"] == "ok"

    # 2. catalog
    res = await service.tool_read(user, "catalog")
    assert res["status"] == "ok"
    assert "catalog" in res

    # 3. sources
    res = await service.tool_read(user, "sources")
    assert res["status"] == "ok"
    assert len(res["sources"]) > 0

    # 4. pages and cards
    res = await service.tool_read(user, "pages")
    assert res["status"] == "ok"

    res = await service.tool_read(user, "cards")
    assert res["status"] == "ok"

    # 5. card <id>
    # Existing card
    res_write_card = await service.tool_write(
        user, 'card "Read Me"\nroot = Page("Read Me", [])', None
    )
    c_id = res_write_card["results"][0]["card_id"]
    res_card_ok = await service.tool_read(user, f"card {c_id}")
    assert res_card_ok["status"] == "ok"
    assert res_card_ok["card"]["title"] == "Read Me"

    res = await service.tool_read(user, "card nonexistent")
    assert res["status"] == "error"
    assert res["error"] == "card not found"

    # 6. history <id>
    res = await service.tool_read(user, "history nonexistent")
    assert res["status"] == "ok"
    assert res["versions"] == []

    # 7. card <id>
    res_card_err = await service.tool_read(user, "card nonexistent")
    assert res_card_err["status"] == "error"
    assert res_card_err["error"] == "card not found"

    # page <builtin> and page <agent>
    res = await service.tool_read(user, "page today")
    assert res["status"] == "ok"
    assert res["page"]["locked"] is True

    res = await service.tool_read(user, "page foryou")
    assert res["status"] == "ok"
    assert res["page"]["id"] == "foryou"

    res = await service.tool_read(user, "page notfound")
    assert res["status"] == "error"

    # 8. version <id> <n>
    res = await service.tool_read(user, "version foryou 1")
    assert res["status"] == "error"

    # 9. Query(...) peek
    res = await service.tool_read(user, 'Query("health.sleep", {days: 7})')
    assert "rows" in res
    assert "row_count" in res

    # Bad Query peek
    res = await service.tool_read(user, 'Query("bad.source", {})')
    assert res["status"] == "error"

    # 10. Unknown read command
    res = await service.tool_read(user, "unknown read command")
    assert res["status"] == "error"


async def test_tool_write_and_check_templates(service: MiniAppService) -> None:
    user = MiniAppUser(100)

    # CompileError in tool_write block execution (lines 505-510)
    res_compile_err = await service.tool_write(
        user,
        'page custom "Custom Page"\n'
        'btn = Button("Click", @UnknownAction())\n'
        'root = Page("Custom Page", [btn])',
        None,
    )
    assert res_compile_err["status"] == "error"
    assert any("line " in e for e in res_compile_err["results"][0]["errors"])

    # Card without title (lines 560->562)
    res_card_notitle = await service.tool_write(
        user,
        'card\nroot = Page("Card No Title", [Header("Hi")])',
        None,
    )
    assert res_card_notitle["status"] == "ok"

    # Page without title in block command (lines 599->601)
    res_page_notitle = await service.tool_write(
        user,
        'page notitle\nroot = Page("Default Title", [Header("Hi")])',
        None,
    )
    assert res_page_notitle["status"] == "ok"

    # _retitle when args is empty (lines 686->688)
    from blacki.miniapp.lang import parse
    from blacki.miniapp.service import _retitle

    prog_empty_args = parse("root = Page()")
    _retitle(prog_empty_args, "New Title")

    # _split_blocks with note inside block (lines 785->723)
    blocks_note = _split_blocks(
        'page withnote "With Note"\n'
        'note "my note"\n'
        'btn = Button("Hi", @Navigate("today"))\n'
        'root = Page("With Note", [btn])'
    )
    assert blocks_note[0]["note"] == "my note"

    # 1. LangError in _split_blocks
    res = await service.tool_write(user, "invalid script without command", None)
    assert res["status"] == "error"

    # 2. Cannot edit builtin page
    res = await service.tool_write(
        user, 'page today "Today"\nroot = Page("Today", [])', None
    )
    assert res["status"] == "error"
    assert any("built-in page" in e for e in res["results"][0]["errors"])

    # 3. Create a custom page
    res = await service.tool_write(
        user,
        'page custom "Custom Page"\n'
        'note "test note"\n'
        'root = Page("Custom Page", [Header("Custom")])',
        None,
    )
    assert res["status"] == "ok"
    assert res["results"][0]["version"] == 1

    # 4. Cannot update without base_version when version > 0
    res = await service.tool_write(
        user,
        'page custom "Custom Page"\nroot = Page("Custom Page", [Header("Custom v2")])',
        None,
    )
    assert res["status"] == "error"
    assert any("pass base_version" in e for e in res["results"][0]["errors"])

    # 5. Partial update with remove line that doesn't exist
    res = await service.tool_write(
        user,
        'page custom "Custom Page"\n'
        "remove non_existent_item\n"
        'root = Page("Custom Page", [Header("Custom v2")])',
        1,
    )
    assert res["status"] == "error"
    assert any(
        "no line named non_existent_item" in e for e in res["results"][0]["errors"]
    )

    # 6. Valid update with remove
    res = await service.tool_write(
        user,
        'page custom "Custom Page"\n'
        'banner = Header("Custom v2")\n'
        'root = Page("Custom Page", [banner])',
        1,
    )
    assert res["status"] == "ok"
    assert res["results"][0]["version"] == 2

    # Now update removing banner
    res = await service.tool_write(
        user,
        'page custom "Custom Page"\nremove banner\nroot = Page("Custom Page", [])',
        2,
    )
    assert res["status"] == "ok"
    assert res["results"][0]["version"] == 3

    # 7. Check templates validation: SELECT template for tracker query
    res = await service.tool_write(
        user,
        'page custom "Custom Page"\n'
        'q = Query("tracker", {template: "missing_sel"})\n'
        'tbl = Table(q, [Column("x", "X")])\n'
        'root = Page("Custom Page", [tbl])',
        3,
    )
    assert res["status"] == "error"
    assert any(
        "must exist and be a SELECT template" in e for e in res["results"][0]["errors"]
    )

    # Check templates validation: INSERT template for @Log
    res = await service.tool_write(
        user,
        'page custom "Custom Page"\n'
        'btn = Button("Log", @Log("missing_ins", {k: 1}))\n'
        'root = Page("Custom Page", [btn])',
        3,
    )
    assert res["status"] == "error"
    assert any(
        "must exist and be a INSERT template" in e for e in res["results"][0]["errors"]
    )

    # Check templates success (lines 644->638, 630->629, 633->632)
    # Seed templates in saved_query_templates
    await service.container.declarative_db_storage.initialize()
    await service.container.conn.execute(
        "INSERT INTO saved_query_templates "
        "(user_id, template_name, table_name, query_type) VALUES (?, ?, ?, ?)",
        (user.chat_user_id, "my_sel", "t", "SELECT"),
    )
    await service.container.conn.execute(
        "INSERT INTO saved_query_templates "
        "(user_id, template_name, table_name, query_type) VALUES (?, ?, ?, ?)",
        (user.chat_user_id, "my_ins", "t", "INSERT"),
    )
    await service.container.conn.commit()

    res_templates_ok = await service.tool_write(
        user,
        'page custom "Custom Page"\n'
        'q1 = Query("health.days", {days: 7})\n'
        'q2 = Query("tracker", {template: "my_sel"})\n'
        'tbl = Table(q2, [Column("x", "X")])\n'
        'btn1 = Button("Nav", @Navigate("today"))\n'
        'btn2 = Button("Log", @Log("my_ins", {k: 1}))\n'
        'root = Page("Custom Page", [tbl, btn1, btn2])',
        3,
    )
    assert res_templates_ok["status"] == "ok"

    # Call _check_templates directly with mixed queries to test branch 630->629
    ir_mixed = {
        "queries": {
            "q1": {"source": "health.days", "params": {"days": 7}},
            "q2": {"source": "tracker", "params": {"template": "my_sel"}},
        },
        "root": {"t": "Page", "c": []},
    }
    await service._check_templates(user, ir_mixed)

    # Revert page
    res_rev = await service.tool_write(user, "revert custom to 1", 4)
    assert res_rev["status"] == "ok"
    assert res_rev["results"][0]["version"] == 5

    # Revert to invalid version
    res_rev_err = await service.tool_write(user, "revert custom to 999", 5)
    assert res_rev_err["status"] == "error"

    # 9. Move page
    res_move = await service.tool_write(user, "move custom to 1", None)
    assert res_move["status"] == "ok"

    # Move page with missing target (line 539)
    res_move_err = await service.tool_write(user, "move nonexistent to 1", None)
    assert res_move_err["status"] == "error"
    assert "page nonexistent not found" in res_move_err["results"][0]["errors"][0]

    # 10. Delete page: builtin/home page rejection
    res_del_home = await service.tool_write(user, "delete page foryou", None)
    assert res_del_home["status"] == "error"

    res_del_today = await service.tool_write(user, "delete page today", None)
    assert res_del_today["status"] == "error"

    # Delete custom page
    res_del = await service.tool_write(user, "delete page custom", None)
    assert res_del["status"] == "ok"

    # Delete non-existent page
    res_del_err = await service.tool_write(user, "delete page custom", None)
    assert res_del_err["status"] == "error"

    # 11. Card with unused warning
    res_card = await service.tool_write(
        user,
        'card "Card Title"\nunused_var = 123\nroot = Page("Card Title", [])',
        None,
    )
    assert res_card["status"] == "ok"
    assert "unused_var" in res_card["results"][0]["warning"]

    # Page with unused warning
    res_page = await service.tool_write(
        user,
        'page unused-page "Unused Page"\n'
        "unused_var = 456\n"
        'root = Page("Unused Page", [])',
        None,
    )
    assert res_page["status"] == "ok"
    assert "unused_var" in res_page["results"][0]["warning"]
