"""Tests for Telegram tool progress label generation and MarkdownV2 validity."""

from typing import Any

import pytest

from blacki.security.tool_privacy import is_private_tool
from blacki.telegram.formatting import MARKDOWN_SPECIAL_CHARS
from blacki.telegram.progress import describe_tool


class _DummyTool:
    def __init__(self, name: str) -> None:
        self.name = name


def validate_markdown_v2(text: str) -> None:
    """Validate that text adheres strictly to Telegram MarkdownV2 syntax.

    Asserts:
    1. Text is non-empty.
    2. No empty bold entities ('**') exist.
    3. Every character in MARKDOWN_SPECIAL_CHARS is escaped with a preceding '\\'
       unless it is the intentional '*' entity delimiter.
    4. All bold entity markers ('*') are properly paired and closed.
    5. Escapes do not dangle at the end of the string.
    """
    assert text, "MarkdownV2 text must not be empty"
    assert "**" not in text, f"Empty bold entity found in: {text!r}"

    i = 0
    in_bold = False
    while i < len(text):
        char = text[i]

        if char == "\\":
            assert i + 1 < len(text), f"Dangling escape at end of: {text!r}"
            escaped_char = text[i + 1]
            assert escaped_char in MARKDOWN_SPECIAL_CHARS, (
                f"Invalid escaped character {escaped_char!r} in: {text!r}"
            )
            i += 2
            continue

        if char == "*":
            in_bold = not in_bold
            i += 1
            continue

        if char in MARKDOWN_SPECIAL_CHARS:
            raise AssertionError(
                f"Unescaped MarkdownV2 character {char!r} at index {i} in: {text!r}"
            )

        i += 1

    assert not in_bold, f"Unclosed bold entity in: {text!r}"


def test_describe_tool_known_search_tools() -> None:
    """Search tools interpolate query and escape markdown."""
    res1 = describe_tool("brave_search", {"query": "high protein foods"}, private=False)
    assert res1 == "Searching the web for *high protein foods*…"
    validate_markdown_v2(res1)

    res2 = describe_tool("exa_search", {"query": "python 3.13 features"}, private=False)
    assert res2 == "Searching the web for *python 3\\.13 features*…"
    validate_markdown_v2(res2)

    # Without query arg
    res3 = describe_tool("brave_search", {}, private=False)
    assert res3 == "Searching the web…"
    validate_markdown_v2(res3)

    res4 = describe_tool("exa_search", {"num_results": 5}, private=False)
    assert res4 == "Searching the web…"
    validate_markdown_v2(res4)


def test_describe_tool_reminders() -> None:
    """Reminder tools produce human progress phrases."""
    res1 = describe_tool(
        "schedule_reminder",
        {"message": "Take vitamins at 9am"},
        private=False,
    )
    assert res1 == "Scheduling reminder for *Take vitamins at 9am*…"
    validate_markdown_v2(res1)

    res2 = describe_tool("schedule_reminder", {}, private=False)
    assert res2 == "Scheduling reminder…"
    validate_markdown_v2(res2)

    res3 = describe_tool("list_reminders", {"include_sent": False}, private=False)
    assert res3 == "Listing reminders…"
    validate_markdown_v2(res3)

    res4 = describe_tool("cancel_reminder", {"reminder_id": 42}, private=False)
    assert res4 == "Cancelling reminder…"
    validate_markdown_v2(res4)


def test_describe_tool_calories() -> None:
    """Calorie tools produce human progress phrases."""
    res1 = describe_tool(
        "log_meal",
        {"description": "Oatmeal with blueberries", "estimated_calories": 350},
        private=False,
    )
    assert res1 == "Logging meal: *Oatmeal with blueberries*…"
    validate_markdown_v2(res1)

    assert describe_tool("log_meal", {}, private=False) == "Logging meal…"
    assert (
        describe_tool("get_calorie_summary", {"days": 7}, private=False)
        == "Checking calorie summary…"
    )
    assert (
        describe_tool(
            "edit_meal",
            {"entry_id": 1, "description": "Greek yogurt"},
            private=False,
        )
        == "Updating meal: *Greek yogurt*…"
    )
    assert (
        describe_tool("delete_meal", {"entry_id": 1}, private=False)
        == "Deleting meal entry…"
    )
    assert (
        describe_tool("set_calorie_goal", {"daily_calories": 2500}, private=False)
        == "Setting calorie goal to *2500* kcal…"
    )


def test_describe_tool_workouts() -> None:
    """Workout tools produce human progress phrases."""
    assert (
        describe_tool(
            "set_training_program",
            {"program_name": "Hypertrophy Block A"},
            private=False,
        )
        == "Setting training program to *Hypertrophy Block A*…"
    )
    assert (
        describe_tool("get_todays_training", {}, private=False)
        == "Checking today's training…"
    )
    assert (
        describe_tool(
            "log_training",
            {"session_type": "Upper Body Strength"},
            private=False,
        )
        == "Logging training for *Upper Body Strength*…"
    )
    assert (
        describe_tool(
            "advance_training_cycle",
            {"program_name": "PPL"},
            private=False,
        )
        == "Advancing training cycle for *PPL*…"
    )
    assert (
        describe_tool(
            "get_training_history",
            {"exercise_name": "bench_press"},
            private=False,
        )
        == "Checking training history for *bench\\_press*…"
    )
    assert (
        describe_tool("get_training_metrics", {}, private=False)
        == "Fetching training metrics…"
    )
    assert (
        describe_tool("update_training_metrics", {}, private=False)
        == "Updating training metrics…"
    )
    assert (
        describe_tool("delete_workout", {"session_id": 3}, private=False)
        == "Deleting workout…"
    )
    # Legacy workout tools
    assert (
        describe_tool("log_workout", {"split_day": "Legs"}, private=False)
        == "Logging workout for *Legs*…"
    )
    assert (
        describe_tool("get_last_workout", {"split_day": "Push"}, private=False)
        == "Checking last workout for *Push*…"
    )
    assert (
        describe_tool("set_workout_split", {"split_name": "Upper/Lower"}, private=False)
        == "Setting workout split to *Upper/Lower*…"
    )
    assert (
        describe_tool("get_todays_workout", {}, private=False)
        == "Checking today's workout…"
    )


def test_describe_tool_declarative_db() -> None:
    """Declarative DB tools produce human progress phrases."""
    assert (
        describe_tool("create_custom_table", {"table_name": "books"}, private=False)
        == "Creating table *books*…"
    )
    assert (
        describe_tool("delete_custom_table", {"table_name": "books"}, private=False)
        == "Deleting table *books*…"
    )
    assert (
        describe_tool(
            "create_query_template",
            {"template_name": "find_unread"},
            private=False,
        )
        == "Creating query template *find\\_unread*…"
    )
    assert (
        describe_tool(
            "execute_query_template",
            {"template_name": "find_unread"},
            private=False,
        )
        == "Running query template *find\\_unread*…"
    )
    assert (
        describe_tool("list_custom_tables_and_templates", {}, private=False)
        == "Listing custom tables and queries…"
    )
    assert (
        describe_tool("set_custom_instruction_override", {}, private=False)
        == "Updating database instructions…"
    )
    assert (
        describe_tool("delete_custom_instruction_override", {}, private=False)
        == "Resetting database instructions…"
    )


def test_describe_tool_sandbox() -> None:
    """Sandbox tools produce human progress phrases."""
    assert (
        describe_tool("sandbox_run_command", {"command": "pytest -v"}, private=False)
        == "Running *pytest \\-v* in sandbox…"
    )
    assert (
        describe_tool("sandbox_execute_code", {"language": "python"}, private=False)
        == "Executing *python* code in sandbox…"
    )
    assert (
        describe_tool("sandbox_write_file", {"path": "test.py"}, private=False)
        == "Writing file *test\\.py* in sandbox…"
    )
    assert (
        describe_tool("sandbox_read_file", {"path": "test.py"}, private=False)
        == "Reading file *test\\.py* in sandbox…"
    )
    assert (
        describe_tool("sandbox_list_files", {"path": "src/"}, private=False)
        == "Listing files in *src/* in sandbox…"
    )
    assert (
        describe_tool(
            "sandbox_send_file_to_user",
            {"path": "report.pdf"},
            private=False,
        )
        == "Sending file *report\\.pdf* from sandbox…"
    )
    assert (
        describe_tool("sandbox_view_image", {"path": "photo.png"}, private=False)
        == "Viewing image *photo\\.png* from sandbox…"
    )


def test_describe_tool_weather() -> None:
    """Weather tools produce human progress phrases."""
    assert (
        describe_tool("get_current_weather", {"location": "London"}, private=False)
        == "Checking current weather for *London*…"
    )
    assert (
        describe_tool("get_weather_forecast", {"location": "Tokyo"}, private=False)
        == "Fetching weather forecast for *Tokyo*…"
    )


def test_describe_tool_memory() -> None:
    """Memory tools produce human progress phrases."""
    assert (
        describe_tool(
            "save_memory",
            {"text": "User prefers vegetarian meals"},
            private=False,
        )
        == "Saving memory: *User prefers vegetarian meals*…"
    )
    assert (
        describe_tool("search_memory", {"query": "preferences"}, private=False)
        == "Searching memory for *preferences*…"
    )
    assert (
        describe_tool("get_all_memories", {}, private=False)
        == "Retrieving all memories…"
    )
    assert describe_tool("get_memory", {}, private=False) == "Retrieving memory…"
    assert describe_tool("update_memory", {}, private=False) == "Updating memory…"
    assert describe_tool("delete_memory", {}, private=False) == "Deleting memory…"
    assert describe_tool("preload_memory", {}, private=False) == "Loading memories…"


def test_describe_tool_skill_loading() -> None:
    """Skill loader tool produces human progress phrase."""
    assert (
        describe_tool("load_skill", {"name": "explore_repo"}, private=False)
        == "Loading skill *explore\\_repo*…"
    )
    assert describe_tool("load_skill", {}, private=False) == "Loading skill…"


def test_describe_tool_gmail_api() -> None:
    assert (
        describe_tool("gmail_search_messages", {"query": "invoices"}, private=False)
        == "Searching Gmail for *invoices*…"
    )
    assert (
        describe_tool("gmail_search_messages", {"query": "invoices"}, private=True)
        == "Searching Gmail…"
    )
    assert describe_tool("gmail_get_thread", {}, private=False) == "Reading email…"
    assert describe_tool("gmail_create_draft", {}, private=False) == "Drafting email…"
    assert describe_tool("gmail_list_drafts", {}, private=False) == "Checking drafts…"
    assert (
        describe_tool("gmail_modify_thread_labels", {}, private=False)
        == "Updating Gmail labels…"
    )
    assert (
        describe_tool("gmail_custom_action", {}, private=False)
        == "Working with Gmail (custom action)…"
    )


def test_describe_tool_private_never_interpolates_args() -> None:
    """When private=True, no argument interpolation ever occurs."""
    args: dict[str, Any] = {
        "text": "secret medical content",
        "query": "private search terms",
        "command": "secret shell cmd",
    }
    # Private tools
    assert (
        describe_tool("get_health_summary", {"days": 7}, private=True)
        == "Fetching health summary…"
    )
    assert (
        describe_tool("send_text_to_speech", {"text": "hello"}, private=True)
        == "Generating speech…"
    )
    # Other tools with private=True
    assert describe_tool("brave_search", args, private=True) == "Searching the web…"
    assert (
        describe_tool("sandbox_run_command", args, private=True)
        == "Running command in sandbox…"
    )
    assert describe_tool("save_memory", args, private=True) == "Saving to memory…"
    assert (
        describe_tool("log_meal", {"description": "private food"}, private=True)
        == "Logging meal…"
    )


def test_describe_tool_markdownv2_escaping() -> None:
    """Salient args containing special Markdown characters are escaped."""
    raw_query = "protein_rich *food* [quick] & vitamins_d3!"
    result = describe_tool("brave_search", {"query": raw_query}, private=False)
    expected = (
        "Searching the web for "
        "*protein\\_rich \\*food\\* \\[quick\\] & vitamins\\_d3\\!*…"
    )
    assert result == expected
    validate_markdown_v2(result)


def test_describe_tool_arg_truncation() -> None:
    """Arguments longer than 60 chars are truncated to <= 60 chars with '…'."""
    long_query = "a" * 80
    result = describe_tool("brave_search", {"query": long_query}, private=False)
    # 59 chars of 'a' + '…'
    expected_inner = "a" * 59 + "…"
    assert result == f"Searching the web for *{expected_inner}*…"
    validate_markdown_v2(result)


def test_describe_tool_empty_whitespace_arg_fallback() -> None:
    """Whitespace-only args fall back to no-arg labels and avoid empty bold '**'."""
    # Test multiple tools with whitespace args
    assert (
        describe_tool("brave_search", {"query": "   "}, private=False)
        == "Searching the web…"
    )
    assert (
        describe_tool("schedule_reminder", {"message": ""}, private=False)
        == "Scheduling reminder…"
    )
    assert (
        describe_tool("log_meal", {"description": "\t \n"}, private=False)
        == "Logging meal…"
    )
    assert (
        describe_tool("sandbox_run_command", {"command": "   "}, private=False)
        == "Running command in sandbox…"
    )
    assert (
        describe_tool("get_current_weather", {"location": " "}, private=False)
        == "Checking current weather…"
    )
    assert (
        describe_tool("save_memory", {"text": " "}, private=False)
        == "Saving to memory…"
    )
    assert describe_tool("load_skill", {"name": " "}, private=False) == "Loading skill…"


def test_describe_tool_unknown_fallback() -> None:
    """Unknown tools humanize symbol into generic phrase without raw symbol alone."""
    result = describe_tool("log_calories", {"some_arg": "val"}, private=False)
    assert result == "Working on log calories…"
    validate_markdown_v2(result)

    result2 = describe_tool("fetch_spotify_playlist", {}, private=False)
    assert result2 == "Working on fetch spotify playlist…"
    validate_markdown_v2(result2)

    # Dotted and hyphenated MCP tool symbols
    result_mcp1 = describe_tool("zepto-search-products", {}, private=False)
    assert result_mcp1 == "Working on zepto search products…"
    validate_markdown_v2(result_mcp1)

    result_mcp2 = describe_tool("some.mcp.tool-v2", {}, private=False)
    assert result_mcp2 == "Working on some mcp tool v2…"
    validate_markdown_v2(result_mcp2)


ALL_KNOWN_TOOLS = [
    ("brave_search", {"query": "test query. with - hyphen [brackets]"}),
    ("exa_search", {"query": "python 3.13"}),
    ("schedule_reminder", {"message": "Call dentist at 3:00pm!"}),
    ("list_reminders", {}),
    ("cancel_reminder", {"reminder_id": 10}),
    ("log_meal", {"description": "Chicken & rice (200g) [lunch]"}),
    ("get_calorie_summary", {"days": 7}),
    ("edit_meal", {"description": "Protein shake #2"}),
    ("delete_meal", {"entry_id": 1}),
    ("set_calorie_goal", {"daily_calories": 2500}),
    ("set_training_program", {"program_name": "PPL_Hypertrophy (v2)"}),
    ("get_todays_training", {}),
    ("log_training", {"session_type": "Chest + Triceps!"}),
    ("advance_training_cycle", {"program_name": "Phase-1"}),
    ("get_training_history", {"exercise_name": "Squat (Barbell)"}),
    ("get_training_metrics", {}),
    ("update_training_metrics", {}),
    ("delete_workout", {"session_id": 5}),
    ("log_workout", {"split_day": "Push - Day A"}),
    ("get_last_workout", {"split_day": "Pull"}),
    ("set_workout_split", {"split_name": "Upper/Lower"}),
    ("get_todays_workout", {}),
    ("create_custom_table", {"table_name": "user_habits_v2"}),
    ("delete_custom_table", {"table_name": "old_habits"}),
    ("create_query_template", {"template_name": "get_pending_tasks"}),
    ("execute_query_template", {"template_name": "get_pending_tasks"}),
    ("list_custom_tables_and_templates", {}),
    ("set_custom_instruction_override", {}),
    ("delete_custom_instruction_override", {}),
    ("sandbox_run_command", {"command": "cat /tmp/test.py | grep -E 'def '"}),
    ("sandbox_execute_code", {"language": "python3"}),
    ("sandbox_write_file", {"path": "src/blacki/main.py"}),
    ("sandbox_read_file", {"path": "config.json"}),
    ("sandbox_list_files", {"path": "src/"}),
    ("sandbox_send_file_to_user", {"path": "results.csv"}),
    ("sandbox_view_image", {"path": "photo.png"}),
    ("get_current_weather", {"location": "San Francisco, CA"}),
    ("get_weather_forecast", {"location": "New York, NY"}),
    ("get_health_summary", {"days": 7}),
    ("save_memory", {"text": "User loves 100% dark chocolate (85%+)"}),
    ("search_memory", {"query": "chocolate preferences"}),
    ("get_all_memories", {}),
    ("get_memory", {"memory_id": 1}),
    ("update_memory", {"memory_id": 1}),
    ("delete_memory", {"memory_id": 1}),
    ("preload_memory", {}),
    ("send_text_to_speech", {"text": "Hello!"}),
    ("load_skill", {"name": "github-pr-helper"}),
    ("mcp.tool-with.dots-and-hyphens", {"arg": "val"}),
    ("custom_unknown_tool", {"arg": "val"}),
]


@pytest.mark.parametrize("tool_name,args", ALL_KNOWN_TOOLS)
def test_all_tools_emit_strictly_valid_markdown_v2(
    tool_name: str, args: dict[str, Any]
) -> None:
    """Regression test: every tool and branch produces strictly valid MarkdownV2."""
    # 1. With complex arguments
    complex_args = {
        k: (f"special._*[]()~`>#+-=|{{}}.! {v}" if isinstance(v, str) else v)
        for k, v in args.items()
    }
    label1 = describe_tool(tool_name, complex_args, private=False)
    validate_markdown_v2(label1)

    # 2. With empty / whitespace arguments
    empty_args = dict.fromkeys(args, "   ")
    label2 = describe_tool(tool_name, empty_args, private=False)
    validate_markdown_v2(label2)

    # 3. With no arguments
    label3 = describe_tool(tool_name, {}, private=False)
    validate_markdown_v2(label3)

    # 4. With private=True
    label4 = describe_tool(tool_name, complex_args, private=True)
    validate_markdown_v2(label4)


def test_elapsed_collapse_text_is_valid_markdown_v2() -> None:
    """Assert final elapsed collapse text is strictly valid MarkdownV2."""
    validate_markdown_v2("✓ Worked for 2m 5s")


def test_is_private_tool_helper() -> None:
    """is_private_tool correctly identifies private tools."""
    tool_health = _DummyTool("get_health_summary")
    tool_tts = _DummyTool("send_text_to_speech")
    tool_zepto = _DummyTool("zepto_create_cart")
    tool_search = _DummyTool("brave_search")

    assert is_private_tool(tool_health)  # type: ignore[arg-type]
    assert is_private_tool(tool_tts)  # type: ignore[arg-type]
    assert is_private_tool(tool_zepto)  # type: ignore[arg-type]
    assert not is_private_tool(tool_search)  # type: ignore[arg-type]
