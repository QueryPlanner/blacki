"""The Mini App component catalog: every part Blacki may place on a page.

One definition drives three things: the compiler's validation, the reference
text Blacki reads (``reference_text``), and the manifest the browser renderer
is checked against. Parts carry content, data bindings and limits only. There
are no style fields; ``tone`` is the only visual choice and the user's theme
decides what each tone looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TONES = ("default", "hot", "inverse", "info", "success", "warning")
CALLOUT_TONES = ("info", "success", "warning")
TARGET_KINDS = (
    "calories_kcal",
    "protein_g",
    "steps",
    "sleep_minutes",
    "water_ml",
    "weekly_training_minutes",
)
BUILTIN_PAGES = ("today", "health", "food", "training", "files")
AGENT_HOME_PAGE = "foryou"


@dataclass(frozen=True)
class Param:
    """One setting of a component, function or action.

    ``kind`` selects validation: text, rich, number, int, bool, enum, tone,
    value (literal or expression), query (a Query statement), field (a field
    of the ``of`` query), predicate (a row condition over the ``of`` query),
    var (a ``$page_value``), action, children, child, texts (list of text),
    page (a page id), sets (list of [kg, reps]).
    """

    name: str
    kind: str
    required: bool = False
    default: Any = None
    options: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    max_len: int = 160
    max_items: int = 12
    of: str = "data"
    allowed: tuple[str, ...] = ()  # component types allowed for children
    doc: str = ""


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    group: str
    doc: str
    params: tuple[Param, ...] = ()
    example: str = ""
    implicit_source: str | None = None
    implicit_params: dict[str, str] = field(default_factory=dict)

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)


def _p(name: str, kind: str, **kw: Any) -> Param:
    return Param(name, kind, **kw)


_CHILDREN = _p("children", "children", required=True, doc="parts to show")

COMPONENTS: dict[str, ComponentSpec] = {
    spec.name: spec
    for spec in (
        # ---- layout
        ComponentSpec(
            "Page",
            "Layout",
            "The whole page. Every page and card has one, as root.",
            (
                _p("title", "text", required=True, max_len=60),
                _CHILDREN,
                _p("subtitle", "text", max_len=200),
            ),
            'root = Page("Recovery", [intro, chart], subtitle: "Last 14 nights")',
        ),
        ComponentSpec(
            "Section",
            "Layout",
            "A titled block of parts.",
            (_p("title", "text", required=True, max_len=60), _CHILDREN),
            'sleepBlock = Section("Sleep", [stages, avgSleep])',
        ),
        ComponentSpec(
            "Stack",
            "Layout",
            "Parts one under another.",
            (_CHILDREN,),
            "group = Stack([header, chart])",
        ),
        ComponentSpec(
            "Grid",
            "Layout",
            "Parts side by side; becomes one column on narrow screens.",
            (_CHILDREN, _p("columns", "int", default=2, min=1, max=3)),
            "stats = Grid([hrvStat, rhrStat], columns: 2)",
        ),
        ComponentSpec(
            "Tabs",
            "Layout",
            "Switch between views. Children must be Tab parts.",
            (_p("tabs", "children", required=True, allowed=("Tab",), max_items=5),),
            'views = Tabs([Tab("HRV", [hrvBars]), Tab("Heart", [rhrLine])])',
        ),
        ComponentSpec(
            "Tab",
            "Layout",
            "One tab inside Tabs.",
            (_p("label", "text", required=True, max_len=24), _CHILDREN),
        ),
        # ---- text
        ComponentSpec(
            "Header",
            "Text",
            "A title with one optional sentence.",
            (
                _p("title", "text", required=True, max_len=80),
                _p("text", "text", max_len=200),
            ),
            'intro = Header("Short nights and HRV", "Move the cutoff to explore.")',
        ),
        ComponentSpec(
            "Text",
            "Text",
            "A paragraph. Only **bold** formatting is allowed.",
            (_p("text", "rich", required=True, max_len=800),),
            'why = Text("Deep sleep was **76 min**, a little under your average.")',
        ),
        ComponentSpec(
            "Callout",
            "Text",
            "A short note that stands out.",
            (
                _p("text", "rich", required=True, max_len=240),
                _p("tone", "enum", default="info", options=CALLOUT_TONES),
            ),
            'note = Callout("Two nights are missing.", tone: "warning")',
        ),
        ComponentSpec(
            "Empty",
            "Text",
            "What to show when there is nothing yet.",
            (
                _p("title", "text", required=True, max_len=60),
                _p("text", "text", required=True, max_len=200),
                _p("action", "child", allowed=("Button",)),
            ),
            'none = Empty("No water yet", "Tell Blacki how much you drank.")',
        ),
        ComponentSpec(
            "Steps",
            "Text",
            "A numbered sequence: how something works, or what to do.",
            (_p("items", "texts", required=True, max_items=8, max_len=200),),
            'how = Steps(["Wind down at 23:00", "Lights out by 23:30", "Same wake time"])',  # noqa: E501
        ),
        ComponentSpec(
            "Compare",
            "Text",
            "Two or three columns side by side. Children are Side parts.",
            (_p("sides", "children", required=True, allowed=("Side",), max_items=3),),
            'cmp = Compare([Side("Before", ["6:10 asleep"]), Side("After", ["7:20 asleep"])])',  # noqa: E501
        ),
        ComponentSpec(
            "Side",
            "Text",
            "One column inside Compare.",
            (
                _p("title", "text", required=True, max_len=40),
                _p("items", "texts", required=True, max_items=6, max_len=120),
            ),
        ),
        # ---- data
        ComponentSpec(
            "Stat",
            "Data",
            "One number or short value with a label.",
            (
                _p("label", "text", required=True, max_len=60),
                _p("value", "value", required=True),
                _p("unit", "text", max_len=10),
                _p("tone", "enum", default="default", options=TONES),
                _p("note", "text", max_len=120),
            ),
            'avgDeep = Stat("Average deep sleep", Avg(sleep.deep), unit: "min", tone: "hot")',  # noqa: E501
        ),
        ComponentSpec(
            "StatRow",
            "Data",
            "Two to four Stat parts in a row.",
            (_p("stats", "children", required=True, allowed=("Stat",), max_items=4),),
            "row = StatRow([avgDeep, shortNights])",
        ),
        ComponentSpec(
            "KeyBars",
            "Data",
            "A series as key-shaped bars. Rows matching highlight use the accent.",
            (
                _p("data", "query", required=True),
                _p("x", "field", required=True),
                _p("y", "field", required=True),
                _p("highlight", "predicate"),
                _p("height", "int", default=180, min=80, max=320),
            ),
            'bars = KeyBars(sleep, "date", "asleep", highlight: asleep >= $cutoff)',
        ),
        ComponentSpec(
            "StackedBars",
            "Data",
            "Bars split into parts. Children are Series parts.",
            (
                _p("data", "query", required=True),
                _p("x", "field", required=True),
                _p(
                    "series",
                    "children",
                    required=True,
                    allowed=("Series",),
                    max_items=4,
                ),
            ),
            'stages = StackedBars(sleep, "date", [Series("deep", "Deep"), Series("rem", "REM")])',  # noqa: E501
        ),
        ComponentSpec(
            "Series",
            "Data",
            "One part of StackedBars.",
            (
                _p("field", "text", required=True, max_len=40),
                _p("label", "text", required=True, max_len=24),
            ),
        ),
        ComponentSpec(
            "Line",
            "Data",
            "A trend over time.",
            (
                _p("data", "query", required=True),
                _p("x", "field", required=True),
                _p("y", "field", required=True),
                _p("min", "number"),
                _p("max", "number"),
            ),
            'weight = Line(body, "date", "kg")',
        ),
        ComponentSpec(
            "Ring",
            "Data",
            "Progress toward a goal, 0 to 100.",
            (
                _p("value", "value", required=True),
                _p("label", "text", required=True, max_len=80),
            ),
            'hit = Ring(Percent(food, protein >= $goal), "of days reached the goal")',
        ),
        ComponentSpec(
            "RangeBar",
            "Data",
            "A value shown inside a normal or target band.",
            (
                _p("label", "text", required=True, max_len=60),
                _p("value", "value", required=True),
                _p("min", "number", required=True),
                _p("max", "number", required=True),
                _p("low", "number", required=True),
                _p("high", "number", required=True),
                _p("unit", "text", max_len=10),
            ),
            'spo2Band = RangeBar("Blood oxygen", Latest(spo2.value), 85, 100, 95, 100, unit: "%")',  # noqa: E501
        ),
        ComponentSpec(
            "Table",
            "Data",
            "Rows and columns for exact values. Children are Column parts.",
            (
                _p("data", "query", required=True),
                _p(
                    "columns",
                    "children",
                    required=True,
                    allowed=("Column",),
                    max_items=6,
                ),
                _p("limit", "int", default=7, min=1, max=50),
            ),
            'nights = Table(sleep, [Column("date", "Night", format: "date"), Column("asleep", "Asleep", format: "minutes")])',  # noqa: E501
        ),
        ComponentSpec(
            "Column",
            "Data",
            "One column of a Table.",
            (
                _p("field", "text", required=True, max_len=40),
                _p("label", "text", required=True, max_len=24),
                _p(
                    "format",
                    "enum",
                    default="text",
                    options=("text", "number", "date", "minutes"),
                ),
            ),
        ),
        ComponentSpec(
            "List",
            "Data",
            "Records as rows with a key marker.",
            (
                _p("data", "query", required=True),
                _p("title", "field", required=True),
                _p("value", "field"),
                _p("subtitle", "field"),
                _p("unit", "text", max_len=10),
                _p("limit", "int", default=6, min=1, max=20),
            ),
            'recent = List(sessions, "date", value: "minutes", unit: "min")',
        ),
        # ---- inputs
        ComponentSpec(
            "Slider",
            "Inputs",
            "Pick a number in a range. Changes a page value only.",
            (
                _p("var", "var", required=True),
                _p("label", "text", required=True, max_len=40),
                _p("min", "number", required=True),
                _p("max", "number", required=True),
                _p("step", "number", default=1),
                _p("unit", "text", max_len=10),
            ),
            'cut = Slider($cutoff, "Short night under", 300, 480, step: 15, unit: "min")',  # noqa: E501
        ),
        ComponentSpec(
            "Choice",
            "Inputs",
            "Pick one of a few options. Changes a page value only.",
            (
                _p("var", "var", required=True),
                _p("options", "texts", required=True, max_items=5, max_len=24),
            ),
            'view = Choice($view, ["Week", "Month"])',
        ),
        ComponentSpec(
            "Toggle",
            "Inputs",
            "On or off. Changes a page value only.",
            (
                _p("var", "var", required=True),
                _p("label", "text", required=True, max_len=40),
            ),
            'weekends = Toggle($weekends, "Include weekends")',
        ),
        ComponentSpec(
            "Stepper",
            "Inputs",
            "Nudge a number up or down. Changes a page value only.",
            (
                _p("var", "var", required=True),
                _p("label", "text", required=True, max_len=40),
                _p("step", "number", default=1),
                _p("unit", "text", max_len=10),
            ),
            'glasses = Stepper($glasses, "Glasses of water", unit: "glasses")',
        ),
        # ---- actions
        ComponentSpec(
            "Button",
            "Actions",
            "Runs one action when tapped. Saving actions only run on this tap.",
            (
                _p("label", "text", required=True, max_len=28),
                _p("action", "action", required=True),
                _p("style", "enum", default="primary", options=("primary", "quiet")),
            ),
            'save = Button("Save goal", @SaveTarget("protein_g", $goal))',
        ),
        ComponentSpec(
            "Link",
            "Actions",
            "A link to another page in the app.",
            (
                _p("label", "text", required=True, max_len=40),
                _p("page", "page", required=True),
            ),
            'more = Link("See all sleep", "health")',
        ),
        # ---- built-in widgets (fixed behaviour; Blacki only places them)
        ComponentSpec(
            "OctaveWeek",
            "Widgets",
            "The last 7 days as piano keys; tap a day for its numbers.",
            (),
            "week = OctaveWeek()",
            implicit_source="health.days",
            implicit_params={"days": "7"},
        ),
        ComponentSpec(
            "SleepStages",
            "Widgets",
            "Nightly sleep split into stages; tap a night.",
            (_p("nights", "int", default=14, min=7, max=30),),
            "stages = SleepStages(nights: 14)",
            implicit_source="health.sleep",
            implicit_params={"days": "nights"},
        ),
        ComponentSpec(
            "MealLog",
            "Widgets",
            "Logged meals grouped by day, with edit and delete (delete asks first).",
            (_p("days", "int", default=3, min=1, max=14),),
            "meals = MealLog(days: 3)",
            implicit_source="food.meals",
            implicit_params={"days": "days"},
        ),
        ComponentSpec(
            "SetLogger",
            "Widgets",
            "Log sets with weight and rep steppers; Save writes the workout.",
            (
                _p("exercise", "text", required=True, max_len=60),
                _p("sets", "sets", required=True, max_items=10),
            ),
            'squat = SetLogger("Back squat", [[60, 8], [70, 6], [75, 5]])',
        ),
        ComponentSpec(
            "RestTimer",
            "Widgets",
            "A countdown between sets with an adjustable length.",
            (_p("seconds", "int", default=90, min=30, max=300),),
            "rest = RestTimer(seconds: 90)",
        ),
        ComponentSpec(
            "FileList",
            "Widgets",
            "Your stored files with search and delete (asks first).",
            (
                _p(
                    "kind",
                    "enum",
                    default="all",
                    options=("all", "images", "documents", "audio"),
                ),
            ),
            'files = FileList(kind: "all")',
            implicit_source="files.list",
        ),
        ComponentSpec(
            "CardFeed",
            "Widgets",
            "Cards Blacki made from your chats, newest first.",
            (),
            "feed = CardFeed()",
            implicit_source="cards.recent",
        ),
        ComponentSpec(
            "PinnedCards",
            "Widgets",
            "Cards you pinned, shown on this page.",
            (),
            "pinned = PinnedCards()",
            implicit_source="cards.pinned",
        ),
    )
}


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    doc: str
    signature: str


FUNCTIONS: dict[str, FunctionSpec] = {
    f.name: f
    for f in (
        FunctionSpec(
            "Query",
            "Name a data source. Only as a whole statement.",
            'sleep = Query("health.sleep", {days: 14})',
        ),
        FunctionSpec(
            "Count",
            "Rows, optionally only those matching a condition.",
            "Count(sleep)  or  Count(sleep, asleep < $cutoff)",
        ),
        FunctionSpec(
            "Avg",
            "Average of a field.",
            "Avg(sleep.deep)  or  Avg(sleep.deep, asleep >= 360)",
        ),
        FunctionSpec("Sum", "Total of a field.", "Sum(food.kcal)"),
        FunctionSpec("Min", "Smallest value of a field.", "Min(hrv.value)"),
        FunctionSpec("Max", "Largest value of a field.", "Max(hrv.value)"),
        FunctionSpec("Latest", "The most recent row's value.", "Latest(hrv.value)"),
        FunctionSpec(
            "Percent",
            "Percent of rows matching a condition, 0 to 100.",
            "Percent(food, protein >= $goal)",
        ),
        FunctionSpec("Round", "Round a number.", "Round(Avg(sleep.asleep) / 60, 1)"),
        FunctionSpec(
            "Or",
            "The first value that isn't empty.",
            "Or(Latest(targets.protein_g), 140)",
        ),
    )
}

REDUCERS = {"Count", "Avg", "Sum", "Min", "Max", "Latest", "Percent"}


@dataclass(frozen=True)
class ActionSpec:
    name: str
    doc: str
    params: tuple[Param, ...]
    saves: bool = False
    example: str = ""


ACTIONS: dict[str, ActionSpec] = {
    a.name: a
    for a in (
        ActionSpec(
            "Set",
            "Set a page value.",
            (_p("var", "var", required=True), _p("value", "value", required=True)),
            example='@Set($view, "Month")',
        ),
        ActionSpec(
            "Navigate",
            "Open another page.",
            (_p("page", "page", required=True),),
            example='@Navigate("health")',
        ),
        ActionSpec(
            "AskBlacki",
            "Send a message to Blacki in the Telegram chat after the user confirms.",
            (_p("text", "text", required=True, max_len=300),),
            example='@AskBlacki("Why was Monday\'s HRV low?")',
        ),
        ActionSpec("Pin", "Pin this card to Today (cards only).", ()),
        ActionSpec(
            "SaveTarget",
            "Save a daily target to the user's settings.",
            (
                _p("kind", "enum", required=True, options=TARGET_KINDS),
                _p("value", "value", required=True),
            ),
            saves=True,
            example='@SaveTarget("protein_g", $goal)',
        ),
        ActionSpec(
            "Log",
            "Add a row through one of the user's saved INSERT query templates.",
            (
                _p("template", "text", required=True, max_len=64),
                _p("values", "values", required=True),
            ),
            saves=True,
            example='@Log("log_water", {ml: $ml})',
        ),
    )
}


def manifest() -> dict[str, Any]:
    """A machine-readable view of the catalog for the browser renderer."""
    return {
        "components": sorted(COMPONENTS),
        "actions": sorted(ACTIONS),
        "functions": sorted(FUNCTIONS),
        "tones": list(TONES),
    }


def _param_doc(p: Param) -> str:
    bits = [p.kind]
    if p.options:
        bits = ["|".join(p.options)]
    if p.min is not None or p.max is not None:
        bits.append(
            f"{p.min:g}..{p.max:g}" if p.min is not None and p.max is not None else ""
        )
    if p.allowed:
        bits = [f"list of {'/'.join(p.allowed)}"]
    if p.default is not None:
        bits.append(f"default {p.default}")
    return f"{p.name}{'' if p.required else '?'}: {', '.join(b for b in bits if b)}"


def reference_text() -> str:
    """The catalog as compact text for Blacki."""
    lines = ["COMPONENTS (positional settings in order; '?' = optional, give by name)"]
    for spec in COMPONENTS.values():
        params = "; ".join(_param_doc(p) for p in spec.params) or "no settings"
        lines.append(f"- {spec.name} [{spec.group}] {spec.doc} :: {params}")
        if spec.example:
            lines.append(f"    e.g. {spec.example}")
    lines.append("FUNCTIONS")
    for f in FUNCTIONS.values():
        lines.append(f"- {f.name}: {f.doc}  e.g. {f.signature}")
    lines.append("ACTIONS (write with @; saving actions only run when the user taps)")
    for a in ACTIONS.values():
        params = "; ".join(_param_doc(p) for p in a.params) or "no settings"
        lines.append(f"- @{a.name}: {a.doc} :: {params}")
    lines.append(f"TONES: {', '.join(TONES)}")
    return "\n".join(lines)
