"""Domain policies and ADK policy plugins."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types as genai_types

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools import BaseTool, ToolContext

NUTRITION_POLICY = """\
<nutrition_policy>
Use nutrition tracking only when the user explicitly logs, edits, deletes, or
summarizes intake, or clearly describes something they consumed. A food name
by itself is ambiguous: ask whether they want information or a log. General
nutrition questions must not create or change records.

For an intake log, estimate calories and macros from the stated portion when
reasonable and disclose material uncertainty. Ask about a missing portion only
when it would make the estimate misleading. Preserve an explicit or relative
meal date in the tool call; never replace an invalid date with today. Use only
breakfast, lunch, dinner, or snack as meal types.

After log_meal, edit_meal, or delete_meal succeeds locally, report the local
result. Do not mention a pending background export in the ordinary confirmation.
A successful log or edit remains saved in Blacki even when remote export is
pending, failed, not_enabled, or authorization_required; a successful delete
remains deleted locally in those states. If export failed, tell the user they
can ask you to retry failed meal exports. If authorization is required, tell
the user to reconnect Google Health. Never claim Google Health accepted a
change unless the status says synced. Do not repeat a meal mutation because
remote export failed or is still pending. Use get_meal_sync_status only when
the user asks about export state, and use retry_meal_sync only for an explicit
retry request.

After both nutrition permissions are granted in a private Telegram chat, Blacki
queues eligible existing meals once for that Google account, then exports new
meal logs, edits, and deletions. Missing nutrition values are omitted rather
than invented.
</nutrition_policy>"""


WORKOUT_POLICY = """\
<workout_policy>
The training-program API is canonical: use set_training_program,
get_todays_training, log_training, advance_training_cycle,
get_training_history, get_training_metrics, and update_training_metrics.
Respect the active program's progression and deload rules.

Record completed only when the user says the session was finished; use partial,
planned, or skipped when that is what they report. Logging never advances the
cycle by implication. Set advance_day=true only when the user explicitly asks
to move the pointer in the same request; otherwise keep it false. Use
advance_training_cycle only for a separate explicit pointer-change request.

For an incomplete log, ask only for required details that cannot be inferred,
such as session type or missing resistance set reps/weight. Omit optional
metrics rather than inventing them. Normalize exercise names to lowercase.
</workout_policy>"""


LEGACY_WORKOUT_POLICY = """\
Legacy split tools are fallback-only. Use them only when they are exposed and
the user explicitly requests an existing simple weekly split after no active
training program is available."""


REMINDER_POLICY = """\
<reminder_policy>
Create, change, or cancel a reminder only when explicitly requested. Listing
or discussing a possible schedule is read-only. Ask for a missing required
time instead of guessing it, and use the shared temporal context for its date.
</reminder_policy>"""


HEALTH_POLICY = """\
<google_health_policy>
Google Health is a read-only wellness summary source. Use
get_health_summary only for the authenticated user's private Telegram data.
Meal export is a separate, optional write capability that applies only after
the user grants both Google Health nutrition permissions. Never request Apple
ID credentials, Fitbit credentials, raw provider payloads, ECG data, medication
or clinical records, or another user's health information. Omit missing
metrics; never infer, diagnose, or present wellness observations as medical
advice. The Apple Health import path is user-configured and may be incomplete,
so describe the source as Google Health and explain that absence does not prove
absence in Apple Health. Keep local Blacki save status distinct from remote
Google Health sync status.
</google_health_policy>"""


DOMAIN_PATTERNS = {
    "nutrition": re.compile(
        r"\b(?:ate|eaten|eating|drank|drink|food|meal|breakfast|lunch|dinner|"
        r"snack|calorie|calories|kcal|macro|macros|nutrition|protein|carbs?|fat|"
        r"google\s+health\s+(?:meal\s+)?(?:sync|export)|backfill|retry)\b",
        re.IGNORECASE,
    ),
    "workout": re.compile(
        r"\b(?:workout|training|exercise|gym|sets?|reps?|bench|squat|deadlift|"
        r"ruck|zone\s*2|vo2|mobility|deload|mesocycle)\b",
        re.IGNORECASE,
    ),
    "reminder": re.compile(
        r"\b(?:remind|reminder|schedule|alarm|notify|notification)\b",
        re.IGNORECASE,
    ),
    "search": re.compile(
        r"\b(?:latest|current|news|recent|today|as of|verify|verified|search|"
        r"look up|source|sources|citation|citations)\b",
        re.IGNORECASE,
    ),
    "health": re.compile(
        r"\b(?:health|steps?|distance|active\s+(?:minutes?|zone)|sleep|"
        r"resting\s+heart|heart\s+rate|body\s*fat|weight|fitbit|google\s+health|"
        r"health\s+summary)\b",
        re.IGNORECASE,
    ),
}

DOMAIN_TOOL_NAMES = {
    "nutrition": frozenset(
        {
            "log_meal",
            "get_calorie_summary",
            "edit_meal",
            "delete_meal",
            "set_calorie_goal",
            "get_meal_sync_status",
            "retry_meal_sync",
        }
    ),
    "workout": frozenset(
        {
            "set_training_program",
            "get_todays_training",
            "log_training",
            "advance_training_cycle",
            "get_training_history",
            "get_training_metrics",
            "update_training_metrics",
            "log_workout",
            "get_last_workout",
            "set_workout_split",
            "get_todays_workout",
        }
    ),
    "reminder": frozenset({"schedule_reminder", "list_reminders", "cancel_reminder"}),
    "search": frozenset({"exa_search", "brave_search"}),
    "health": frozenset({"get_health_summary"}),
}

LEGACY_WORKOUT_TOOL_NAMES = frozenset(
    {"log_workout", "get_last_workout", "set_workout_split", "get_todays_workout"}
)

SEARCH_CONFLICTING_TOOL_NAMES = frozenset(
    {
        "sandbox_run_command",
        "sandbox_write_file",
        "sandbox_read_file",
        "sandbox_list_files",
        "sandbox_send_file_to_user",
        "sandbox_execute_code",
        "sandbox_view_image",
        "McpSkillToolset",
    }
)
SEARCH_STATUS_STATE_KEY = "temp:blacki_search_status"
SEARCH_PRIMARY_STATE_KEY = "temp:blacki_search_primary"
SEARCH_RESULT_STATE_KEY = "temp:blacki_search_result"


def select_domain_policy_names(
    user_text: str, available_tool_names: set[str] | frozenset[str]
) -> tuple[str, ...]:
    """Select request-relevant domains that also have enabled tools."""
    selected = []
    for domain in ("nutrition", "workout", "reminder", "search", "health"):
        if (
            DOMAIN_PATTERNS[domain].search(user_text)
            and DOMAIN_TOOL_NAMES[domain] & available_tool_names
        ):
            selected.append(domain)
    return tuple(selected)


def build_domain_instruction(
    user_text: str, available_tool_names: set[str] | frozenset[str]
) -> str:
    """Build only the domain policies relevant to the current request."""
    blocks = []
    for domain in select_domain_policy_names(user_text, available_tool_names):
        if domain == "nutrition":
            blocks.append(NUTRITION_POLICY)
        elif domain == "workout":
            workout_policy = WORKOUT_POLICY
            if LEGACY_WORKOUT_TOOL_NAMES & available_tool_names:
                workout_policy = workout_policy.replace(
                    "</workout_policy>",
                    f"\n\n{LEGACY_WORKOUT_POLICY}\n</workout_policy>",
                )
            blocks.append(workout_policy)
        elif domain == "reminder":
            blocks.append(REMINDER_POLICY)
        elif domain == "search":  # pragma: no branch - search is the final domain
            blocks.append(_build_search_policy(available_tool_names))
        elif domain == "health":  # pragma: no branch - health is the final domain
            blocks.append(HEALTH_POLICY)
    return "\n\n".join(blocks)


def _build_search_policy(available_tool_names: set[str] | frozenset[str]) -> str:
    enabled = DOMAIN_TOOL_NAMES["search"] & available_tool_names
    if enabled == {"exa_search", "brave_search"}:
        selection = (
            "Use exa_search first with five results unless the user requests another "
            "count. Refine once only if results are empty or irrelevant, then use "
            "brave_search as fallback."
        )
    elif "exa_search" in enabled:
        selection = (
            "Use exa_search with five results unless the user requests another count."
        )
    else:
        selection = (
            "Use brave_search with five results unless the user requests another count."
        )
    return f"""\
<search_policy>
{selection} Use exactly one primary search call. Only when it returns an error
or zero results, make one fallback call; never return to the primary provider.
Prefer a dedicated current-data tool when available. Never use sandbox or
browser automation for ordinary web search. A successful result ends tool use.
Use original result URLs as citations and disclose search failures.
</search_policy>"""


class DomainPolicyPlugin(BasePlugin):
    """Append request-relevant policies for currently enabled tools."""

    def __init__(self, name: str = "domain_policy") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        user_content = callback_context.user_content
        if user_content is None:
            return

        user_text = " ".join(
            part.text for part in (user_content.parts or []) if part.text
        ).strip()
        if not user_text:
            return

        instruction = build_domain_instruction(
            user_text, frozenset(llm_request.tools_dict)
        )
        if instruction:
            llm_request.append_instructions([instruction])

        if "search" in select_domain_policy_names(
            user_text, frozenset(llm_request.tools_dict)
        ):
            _apply_search_tool_budget(callback_context, llm_request)

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, object],
        tool_context: ToolContext,
    ) -> dict[str, object] | None:
        """Prevent repeated or out-of-order external search executions."""
        _ = tool_args
        if tool.name not in DOMAIN_TOOL_NAMES["search"]:
            return None

        status = tool_context.state.get(SEARCH_STATUS_STATE_KEY)
        primary = tool_context.state.get(SEARCH_PRIMARY_STATE_KEY)
        if status == "complete":
            cached_result = tool_context.state.get(SEARCH_RESULT_STATE_KEY)
            if isinstance(cached_result, dict):
                return cached_result
            return {"status": "error", "error": "Search already completed."}
        if status == "primary_failed" and tool.name != "brave_search":
            return {"status": "error", "error": "Use the fallback search provider."}
        if status is None and primary and tool.name != primary:
            return {"status": "error", "error": "Use the primary search provider."}
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, object],
        tool_context: ToolContext,
        result: dict[str, object],
    ) -> None:
        """Record whether the current search should stop or use one fallback."""
        _ = tool_args
        if tool.name not in DOMAIN_TOOL_NAMES["search"]:
            return

        succeeded = result.get("status") == "success" and bool(result.get("results"))
        if succeeded or tool.name == "brave_search":
            tool_context.state[SEARCH_STATUS_STATE_KEY] = "complete"
            tool_context.state[SEARCH_RESULT_STATE_KEY] = result
        else:
            tool_context.state[SEARCH_STATUS_STATE_KEY] = "primary_failed"


class ResponsePolicyPlugin(BasePlugin):
    """Remove marked thought text from eligible final responses."""

    def __init__(self, name: str = "response_policy") -> None:
        super().__init__(name=name)

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        _ = callback_context
        if llm_response.partial or llm_response.content is None:
            return

        parts = llm_response.content.parts or []
        if any(part.function_call or part.function_response for part in parts):
            return
        answer_parts = [part for part in parts if part.text and not part.thought]
        if len(answer_parts) != 1:
            return

        llm_response.content.parts = [
            part for part in parts if not (part.text and part.thought)
        ]


def _apply_search_tool_budget(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Expose one search provider at a time and hide unrelated heavy tools."""
    available = frozenset(llm_request.tools_dict)
    hidden = set(SEARCH_CONFLICTING_TOOL_NAMES & available)
    status = callback_context.state.get(SEARCH_STATUS_STATE_KEY)
    if status is None:
        primary = "exa_search" if "exa_search" in available else "brave_search"
        callback_context.state[SEARCH_PRIMARY_STATE_KEY] = primary

    if status == "complete":
        hidden.update(DOMAIN_TOOL_NAMES["search"] & available)
    elif status == "primary_failed":
        hidden.add("exa_search")
    elif {"exa_search", "brave_search"} <= available:
        hidden.add("brave_search")

    _hide_tools(llm_request, hidden)


def _hide_tools(llm_request: LlmRequest, hidden_names: set[str]) -> None:
    """Hide function declarations while retaining safe ADK execution lookups."""
    if not hidden_names:
        return

    if not llm_request.config.tools:
        return

    retained_tools: list[Any] = []
    for tool in llm_request.config.tools:
        if not isinstance(tool, genai_types.Tool):
            retained_tools.append(tool)
            continue
        declarations = tool.function_declarations
        if declarations is not None:
            tool.function_declarations = [
                declaration
                for declaration in declarations
                if declaration.name not in hidden_names
            ]
            if not tool.function_declarations:
                continue
        retained_tools.append(tool)
    llm_request.config.tools = retained_tools
