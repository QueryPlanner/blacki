"""Human-readable tool progress descriptions for Telegram status updates."""

from __future__ import annotations

from typing import Any

from blacki.telegram.formatting import escape_markdown_plain

_MAX_ARG_LENGTH = 60


def _format_salient_arg(value: Any) -> str:
    """Format and escape a salient argument for MarkdownV2 insertion.

    Truncates values longer than 60 characters with '…' (U+2026). Escapes all
    MarkdownV2 control characters using ``escape_markdown_plain``.
    Returns an empty string if the value is empty or whitespace-only.

    Args:
        value: The argument value to format.

    Returns:
        MarkdownV2-escaped and length-bounded string.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if len(text) > _MAX_ARG_LENGTH:
        text = text[: _MAX_ARG_LENGTH - 1] + "…"
    return escape_markdown_plain(text)


def _humanize_symbol(symbol: str) -> str:
    """Convert an internal symbol into human words (e.g. 'mcp.tool-name')."""
    cleaned = (
        symbol.replace("_", " ").replace("-", " ").replace(".", " ").strip().lower()
    )
    normalized = " ".join(cleaned.split())
    return escape_markdown_plain(normalized)


def describe_tool(tool_name: str, args: dict[str, Any], *, private: bool) -> str:
    """Generate a human-readable present-participle progress phrase for a tool call.

    Produces strictly valid Telegram MarkdownV2 formatted strings with bold emphasis
    markers where salient arguments are present.

    Args:
        tool_name: Name of the tool being executed.
        args: Keyword arguments passed to the tool.
        private: Whether the tool is user-private (if True, arguments are never
            interpolated into the output string).

    Returns:
        A MarkdownV2-ready status description ending in '…'.
    """
    can_interpolate = not private and bool(args)

    # 1. Search tools
    if tool_name in ("brave_search", "exa_search"):
        if can_interpolate and (query := _format_salient_arg(args.get("query"))):
            return f"Searching the web for *{query}*…"
        return "Searching the web…"

    # 2. Reminders
    if tool_name == "schedule_reminder":
        if can_interpolate and (message := _format_salient_arg(args.get("message"))):
            return f"Scheduling reminder for *{message}*…"
        return "Scheduling reminder…"
    if tool_name == "list_reminders":
        return "Listing reminders…"
    if tool_name == "cancel_reminder":
        return "Cancelling reminder…"

    # 3. Calories
    if tool_name == "log_meal":
        if can_interpolate and (desc := _format_salient_arg(args.get("description"))):
            return f"Logging meal: *{desc}*…"
        return "Logging meal…"
    if tool_name == "get_calorie_summary":
        return "Checking calorie summary…"
    if tool_name == "edit_meal":
        if can_interpolate and (desc := _format_salient_arg(args.get("description"))):
            return f"Updating meal: *{desc}*…"
        return "Updating meal…"
    if tool_name == "delete_meal":
        return "Deleting meal entry…"
    if tool_name == "set_calorie_goal":
        if can_interpolate and (
            goal := _format_salient_arg(args.get("daily_calories"))
        ):
            return f"Setting calorie goal to *{goal}* kcal…"
        return "Setting calorie goal…"

    # 4. Workouts
    if tool_name == "set_training_program":
        if can_interpolate and (prog := _format_salient_arg(args.get("program_name"))):
            return f"Setting training program to *{prog}*…"
        return "Setting training program…"
    if tool_name == "get_todays_training":
        return "Checking today's training…"
    if tool_name == "log_training":
        if can_interpolate and (stype := _format_salient_arg(args.get("session_type"))):
            return f"Logging training for *{stype}*…"
        return "Logging training…"
    if tool_name == "advance_training_cycle":
        if can_interpolate and (prog := _format_salient_arg(args.get("program_name"))):
            return f"Advancing training cycle for *{prog}*…"
        return "Advancing training cycle…"
    if tool_name == "get_training_history":
        if can_interpolate and (
            ex := _format_salient_arg(
                args.get("exercise_name") or args.get("program_name")
            )
        ):
            return f"Checking training history for *{ex}*…"
        return "Checking training history…"
    if tool_name == "get_training_metrics":
        return "Fetching training metrics…"
    if tool_name == "update_training_metrics":
        return "Updating training metrics…"
    if tool_name == "delete_workout":
        return "Deleting workout…"
    # Legacy workout tools
    if tool_name == "log_workout":
        if can_interpolate and (split := _format_salient_arg(args.get("split_day"))):
            return f"Logging workout for *{split}*…"
        return "Logging workout…"
    if tool_name == "get_last_workout":
        if can_interpolate and (split := _format_salient_arg(args.get("split_day"))):
            return f"Checking last workout for *{split}*…"
        return "Checking last workout…"
    if tool_name == "set_workout_split":
        if can_interpolate and (split := _format_salient_arg(args.get("split_name"))):
            return f"Setting workout split to *{split}*…"
        return "Setting workout split…"
    if tool_name == "get_todays_workout":
        return "Checking today's workout…"

    # 5. Declarative DB
    if tool_name == "create_custom_table":
        if can_interpolate and (table := _format_salient_arg(args.get("table_name"))):
            return f"Creating table *{table}*…"
        return "Creating table…"
    if tool_name == "delete_custom_table":
        if can_interpolate and (table := _format_salient_arg(args.get("table_name"))):
            return f"Deleting table *{table}*…"
        return "Deleting table…"
    if tool_name == "create_query_template":
        if can_interpolate and (tmpl := _format_salient_arg(args.get("template_name"))):
            return f"Creating query template *{tmpl}*…"
        return "Creating query template…"
    if tool_name == "execute_query_template":
        if can_interpolate and (tmpl := _format_salient_arg(args.get("template_name"))):
            return f"Running query template *{tmpl}*…"
        return "Running query template…"
    if tool_name == "list_custom_tables_and_templates":
        return "Listing custom tables and queries…"
    if tool_name == "set_custom_instruction_override":
        return "Updating database instructions…"
    if tool_name == "delete_custom_instruction_override":
        return "Resetting database instructions…"

    # 6. Sandbox
    if tool_name == "sandbox_run_command":
        if can_interpolate and (cmd := _format_salient_arg(args.get("command"))):
            return f"Running *{cmd}* in sandbox…"
        return "Running command in sandbox…"
    if tool_name == "sandbox_execute_code":
        if can_interpolate and (lang := _format_salient_arg(args.get("language"))):
            return f"Executing *{lang}* code in sandbox…"
        return "Executing code in sandbox…"
    if tool_name == "sandbox_write_file":
        if can_interpolate and (path := _format_salient_arg(args.get("path"))):
            return f"Writing file *{path}* in sandbox…"
        return "Writing file in sandbox…"
    if tool_name == "sandbox_read_file":
        if can_interpolate and (path := _format_salient_arg(args.get("path"))):
            return f"Reading file *{path}* in sandbox…"
        return "Reading file in sandbox…"
    if tool_name == "sandbox_list_files":
        if can_interpolate and (path := _format_salient_arg(args.get("path"))):
            return f"Listing files in *{path}* in sandbox…"
        return "Listing files in sandbox…"
    if tool_name == "sandbox_send_file_to_user":
        if can_interpolate and (path := _format_salient_arg(args.get("path"))):
            return f"Sending file *{path}* from sandbox…"
        return "Sending file from sandbox…"
    if tool_name == "sandbox_view_image":
        if can_interpolate and (path := _format_salient_arg(args.get("path"))):
            return f"Viewing image *{path}* from sandbox…"
        return "Viewing image from sandbox…"

    # 7. Weather
    if tool_name == "get_current_weather":
        if can_interpolate and (loc := _format_salient_arg(args.get("location"))):
            return f"Checking current weather for *{loc}*…"
        return "Checking current weather…"
    if tool_name == "get_weather_forecast":
        if can_interpolate and (loc := _format_salient_arg(args.get("location"))):
            return f"Fetching weather forecast for *{loc}*…"
        return "Fetching weather forecast…"

    # 8. Health (Private)
    if tool_name == "get_health_summary":
        return "Fetching health summary…"

    # 9. Memory
    if tool_name == "save_memory":
        if can_interpolate and (text := _format_salient_arg(args.get("text"))):
            return f"Saving memory: *{text}*…"
        return "Saving to memory…"
    if tool_name == "search_memory":
        if can_interpolate and (query := _format_salient_arg(args.get("query"))):
            return f"Searching memory for *{query}*…"
        return "Searching memory…"
    if tool_name == "get_all_memories":
        return "Retrieving all memories…"
    if tool_name == "get_memory":
        return "Retrieving memory…"
    if tool_name == "update_memory":
        return "Updating memory…"
    if tool_name == "delete_memory":
        return "Deleting memory…"
    if tool_name == "preload_memory":
        return "Loading memories…"

    # 10. TTS (Private)
    if tool_name == "send_text_to_speech":
        return "Generating speech…"

    # 11. Skills and connected services
    if tool_name == "load_skill":
        if can_interpolate and (name := _format_salient_arg(args.get("name"))):
            return f"Loading skill *{name}*…"
        return "Loading skill…"

    if tool_name.startswith("gmail_"):
        action = tool_name.removeprefix("gmail_")
        if action == "search_messages":
            if can_interpolate and (query := _format_salient_arg(args.get("query"))):
                return f"Searching Gmail for *{query}*…"
            return "Searching Gmail…"
        if action in ("get_thread", "get_message"):
            return "Reading email…"
        if action == "create_draft":
            return "Drafting email…"
        if action in ("list_drafts", "get_draft"):
            return "Checking drafts…"
        if "label" in action:
            return "Updating Gmail labels…"
        return f"Working with Gmail ({_humanize_symbol(action)})…"

    # 12. Unknown tool fallback
    humanized = _humanize_symbol(tool_name)
    return f"Working on {humanized}…"
