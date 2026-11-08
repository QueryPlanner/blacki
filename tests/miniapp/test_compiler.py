from __future__ import annotations

import pytest

from blacki.miniapp.catalog import BUILTIN_PAGES
from blacki.miniapp.compiler import CompileError, compile_program
from blacki.miniapp.lang import Literal, Statement, parse
from blacki.miniapp.store import builtin_source


def test_compile_all_builtin_pages() -> None:
    for page_id in BUILTIN_PAGES:
        src = builtin_source(page_id)
        prog = parse(src)
        ir = compile_program(prog)
        assert ir["version"] == 1
        assert ir["title"]
        assert ir["root"]["type"] == "Page"


def test_compile_valid_page_and_card() -> None:
    src = """
    $goal = 120
    $dynamic_init = Or(Round(1.234, 1), 100)
    $flag = true
    $txt_var = "initial"
    $opt = "A"
    sleep_q = Query("health.sleep", {days: 14})
    items = [sec1, oct1, bars, tbl, btn, sl, stp, tog, ch, stat1]
    root = Page("Recovery", items, subtitle: "Subtitle")
    sec1 = Section("Sleep Stages", [stages, card_grid])
    card_grid = Grid([Text("Hello **world**")])
    stages = SleepStages(nights: 7)
    oct1 = OctaveWeek()
    bars = StackedBars(sleep_q, "date", [ser1])
    ser1 = Series("deep", "Deep")
    tbl = Table(sleep_q, [col1], limit: 10)
    col1 = Column("date", "Date")
    btn = Button("Click", @AskBlacki("Why is HRV low?"))
    sl = Slider($goal, "Goal", 50, 250, step: 5, unit: "m")
    stp = Stepper($goal, "Step Goal", step: 10, unit: "m")
    tog = Toggle($flag, "Enable")
    ch = Choice($opt, ["A", "B"])
    stat1 = Stat(
        "Avg Sleep", Avg(sleep_q.asleep), unit: "m", tone: "default", note: "note"
    )
    """
    prog = parse(src)
    ir = compile_program(prog)
    assert ir["version"] == 1
    assert ir["title"] == "Recovery"
    assert ir["root"]["type"] == "Page"
    assert "sleep_q" in ir["queries"]
    assert "goal" in ir["vars"]
    assert "dynamic_init" in ir["vars"]

    # Compile card with @Pin action
    card_src = """
    root = Page("Card", [btn])
    btn = Button("Pin", @Pin())
    """
    card_ir = compile_program(parse(card_src), kind="card")
    assert card_ir["title"] == "Card"


def test_compile_more_components_and_actions() -> None:
    src = """
    $val = 50
    targets = Query("targets")
    items = [
        callout, cmp, empty_p, rest, set_log, step_p, set_btn,  # noqa: E501
        log_btn, sav_btn, rbar, ln, lk,
    ]
    root = Page("More", items)
    callout = Callout("Notice", tone: "info")
    cmp = Compare([side1, side2])
    side1 = Side("Before", ["A", "B"])
    side2 = Side("After", ["C", "D"])
    empty_p = Empty("Empty", "No data", action: Button("Go", @Navigate("health")))
    rest = RestTimer(seconds: 60)
    set_log = SetLogger("Bench Press", [[60, 8], [70.5, 6]])
    step_p = Steps(["Step 1", "Step 2"])
    set_btn = Button("Set", @Set($val, 100))
    log_btn = Button("Log", @Log("log_water", {ml: 250}))
    sav_btn = Button("Save", @SaveTarget("protein_g", $val))
    rbar = RangeBar("HRV", 65, 20, 120, 40, 80, unit: "ms")
    ln = Line(targets, "calories_kcal", "protein_g", min: 0, max: 500)
    lk = Link("Health", "health")
    """
    ir = compile_program(parse(src))
    assert ir["root"]["type"] == "Page"


def test_compile_unused_lines_and_keep_all_queries() -> None:
    src = """
    unused_sec = Section("Unused", [Text("Hello")])
    $unused_var = 10
    unused_q = Query("health.days")
    root = Page("Title", [Text("Content")])
    """
    prog = parse(src)
    ir = compile_program(prog, keep_all_queries=False)
    assert "unused_sec" in ir["unused"]
    assert "$unused_var" in ir["unused"]
    assert "unused_q" in ir["unused"]
    assert "unused_q" not in ir["queries"]

    # With keep_all_queries=True
    ir2 = compile_program(prog, keep_all_queries=True)
    assert "unused_q" in ir2["queries"]


def test_compile_root_errors() -> None:
    # No root
    with pytest.raises(CompileError, match="there is no root line"):
        compile_program(parse("x = 1"))

    # Root not Page
    with pytest.raises(CompileError, match=r"root must be a Page\(\.\.\.\)"):
        compile_program(parse('root = Text("Hi")'))


def test_declare_var_errors() -> None:
    # Mentions another var
    with pytest.raises(CompileError, match=r"\$b can't start from another page value"):
        compile_program(parse('$a = 10\n$b = $a + 5\nroot = Page("T", [Text("Hi")])'))

    # BinaryOp mentioning var
    with pytest.raises(CompileError, match=r"\$c can't start from another page value"):
        compile_program(parse('$a = 10\n$c = -($a)\nroot = Page("T", [Text("Hi")])'))

    # Call mentioning var
    with pytest.raises(CompileError, match=r"\$d can't start from another page value"):
        compile_program(
            parse('$a = 10\n$d = Round($a)\nroot = Page("T", [Text("Hi")])')
        )

    # Invalid literal type (e.g. None/null)
    with pytest.raises(
        CompileError, match=r"\$a must start as a number, text, true or false"
    ):
        prog = parse('$a = 1\nroot = Page("T", [Text("Hi")])')
        prog.statements["$a"] = Statement(
            name="a", expr=Literal(line=1, value=None), line=1, is_var=True
        )
        compile_program(prog)

    # Text too long (> 120)
    with pytest.raises(CompileError, match=r"\$a text is too long"):
        long_str = "x" * 125
        compile_program(parse(f'$a = "{long_str}"\nroot = Page("T", [Text("Hi")])'))


def test_declare_query_errors() -> None:
    # Query with kwargs
    with pytest.raises(CompileError, match="write queries as"):
        compile_program(
            parse('q = Query(source: "health.sleep")\nroot = Page("T", [Text("Hi")])')
        )

    # Unknown or non-public source
    with pytest.raises(CompileError, match="unknown data source 'unknown.src'"):
        compile_program(
            parse('q = Query("unknown.src")\nroot = Page("T", [Text("Hi")])')
        )

    # Query settings not object
    with pytest.raises(CompileError, match="query settings must be an object"):
        compile_program(
            parse('q = Query("health.sleep", 14)\nroot = Page("T", [Text("Hi")])')
        )

    # Invalid param in settings
    with pytest.raises(CompileError, match="days must be a whole number 1..90"):
        compile_program(
            parse(
                'q = Query("health.sleep", {days: 999})\nroot = Page("T", [Text("Hi")])'
            )
        )

    # Max queries exceeded (> 8)
    queries = "\n".join(f'q{i} = Query("health.days")' for i in range(9))
    with pytest.raises(CompileError, match="at most 8 data sources"):
        compile_program(parse(f'{queries}\nroot = Page("T", [Text("Hi")])'))


def test_component_errors() -> None:
    # Ref not defined
    with pytest.raises(CompileError, match="'missing' is not defined"):
        compile_program(parse('root = Page("T", [missing])'))

    # Circular ref
    with pytest.raises(CompileError, match="'a' refers to itself"):
        compile_program(parse('a = b\nb = a\nroot = Page("T", [a])'))

    # Expected a part but got number
    with pytest.raises(CompileError, match="expected a part such as Stat"):
        compile_program(parse('sec = 123\nroot = Page("T", [sec])'))

    # Unknown component with hint
    with pytest.raises(CompileError, match="unknown part 'Statx'; did you mean Stat?"):
        compile_program(parse('root = Page("T", [Statx()])'))

    # Component not allowed in context (e.g. Stat inside Tabs where Tab is required)
    with pytest.raises(CompileError, match="Stat can't go here; expected Tab"):
        compile_program(parse('root = Page("T", [Tabs([Stat("S", 1)])])'))

    # Too many args
    with pytest.raises(CompileError, match="Text takes at most 1 settings"):
        compile_program(parse('root = Page("T", [Text("Hi", "extra")])'))

    # Unknown setting (kwarg)
    with pytest.raises(CompileError, match="Text has no setting 'bad_setting'"):
        compile_program(parse('root = Page("T", [Text("Hi", bad_setting: 1)])'))

    # Duplicate setting
    with pytest.raises(CompileError, match="Text setting 'text' is given twice"):
        compile_program(parse('root = Page("T", [Text("Hi", text: "Duplicate")])'))

    # Missing required setting
    with pytest.raises(CompileError, match="Text needs 'text'"):
        compile_program(parse('root = Page("T", [Text()])'))


def test_param_validation_errors() -> None:
    # text / rich errors
    with pytest.raises(CompileError, match="Text text must be text in quotes"):
        compile_program(parse('root = Page("T", [Text(123)])'))

    with pytest.raises(CompileError, match="Header title is longer than 80 characters"):
        long_txt = "a" * 90
        compile_program(parse(f'root = Page("T", [Header("{long_txt}")])'))

    with pytest.raises(
        CompileError, match=r"can't contain < or >; only \*\*bold\*\* is allowed"
    ):
        compile_program(parse('root = Page("T", [Text("<script>")])'))

    # number / int errors
    with pytest.raises(CompileError, match="must be a number"):
        compile_program(
            parse('$g = 100\nroot = Page("T", [Slider($g, "Goal", "bad", 200)])')
        )

    with pytest.raises(CompileError, match="MealLog days must be between 1 and 14"):
        compile_program(parse('root = Page("T", [MealLog(days: 99)])'))

    # bool error (using Choice or Toggle)
    with pytest.raises(CompileError, match="Empty text must be text in quotes"):
        compile_program(parse('root = Page("T", [Empty("T", 123)])'))

    # enum error
    with pytest.raises(CompileError, match="must be one of info, success, warning"):
        compile_program(
            parse('root = Page("T", [Callout("Hi", tone: "invalid_tone")])')
        )

    # texts error
    with pytest.raises(CompileError, match="must be a list of text"):
        compile_program(parse('root = Page("T", [Steps(123)])'))

    with pytest.raises(CompileError, match="needs 1 to 8 items"):
        compile_program(parse('root = Page("T", [Steps([])])'))

    with pytest.raises(CompileError, match="items must be short text"):
        compile_program(parse('root = Page("T", [Steps([123])])'))

    # page id error
    with pytest.raises(CompileError, match='must be a page id like "health"'):
        compile_program(
            parse('root = Page("T", [Button("Nav", @Navigate("BAD PAGE ID!"))])')
        )

    # sets error
    with pytest.raises(
        CompileError, match=r"must be a list like \[\[60, 8\], \[70, 6\]\]"
    ):
        compile_program(parse('root = Page("T", [SetLogger("Bench", 123)])'))

    with pytest.raises(
        CompileError,
        match=r"items must be \[kg, reps\] with kg 0\.\.500 and reps 1\.\.100",
    ):
        compile_program(parse('root = Page("T", [SetLogger("Bench", [[600, 8]])])'))

    # query error
    with pytest.raises(CompileError, match="must name a Query"):
        compile_program(
            parse('root = Page("T", [Table("not_a_query", [Column("date", "A")])])')
        )

    # inline Query call as query ref
    inline_q_src = """
    q = Query("health.sleep", {days: 7})
    root = Page("T", [Table(q, [Column("date", "Date")])])
    """
    assert compile_program(parse(inline_q_src))["root"]["type"] == "Page"

    # field error
    with pytest.raises(CompileError, match="is not in this data"):
        compile_program(
            parse(
                'q = Query("health.sleep")\n'
                'root = Page("T", [Table(q, [Column("nonexistent", "Bad")])])'
            )
        )

    # var error
    with pytest.raises(CompileError, match=r"must be a page value like \$goal"):
        compile_program(parse('root = Page("T", [Slider(100, "Goal", 50, 200)])'))

    # children error
    with pytest.raises(CompileError, match=r"must be a list like \[a, b\]"):
        compile_program(parse('root = Page("T", [Section("S", 123)])'))

    with pytest.raises(CompileError, match="holds at most 12 parts"):
        too_many = ", ".join('Text("x")' for _ in range(15))
        compile_program(parse(f'root = Page("T", [{too_many}])'))


def test_expression_and_function_errors() -> None:
    # Text value too long
    with pytest.raises(CompileError, match="text value is too long"):
        long_val = "x" * 205
        compile_program(
            parse(f'$v = Or("{long_val}", "fb")\nroot = Page("T", [Text("Hi")])')
        )

    # Undefined var in component value
    with pytest.raises(CompileError, match=r"\$undeclared is not declared"):
        compile_program(parse('root = Page("T", [Stat("S", $undeclared + 1)])'))

    # Undefined ref in component value
    with pytest.raises(CompileError, match="'missing' is not defined"):
        compile_program(parse('root = Page("T", [Stat("S", missing + 1)])'))

    # Reading Query as bare identifier
    with pytest.raises(CompileError, match="q is data; use Count"):
        compile_program(
            parse('q = Query("health.days")\nroot = Page("T", [Stat("S", q)])')
        )

    # Circular ref in expr
    with pytest.raises(CompileError, match="'x' refers to itself"):
        compile_program(parse('x = y\ny = x\nroot = Page("T", [Stat("S", x)])'))

    # Bare FieldAccess
    with pytest.raises(CompileError, match="use Latest"):
        compile_program(
            parse('q = Query("health.days")\nroot = Page("T", [Stat("S", q.steps)])')
        )

    # Component used as value
    with pytest.raises(CompileError, match="Text is a part, not a value"):
        compile_program(parse('root = Page("T", [Stat("S", Text("Hi"))])'))

    # Unknown function with hint
    with pytest.raises(
        CompileError, match="unknown function 'Rounds'; did you mean Round?"
    ):
        compile_program(parse('root = Page("T", [Stat("S", Rounds(1))])'))

    # Function with kwargs
    with pytest.raises(CompileError, match="Round takes no named settings"):
        compile_program(parse('root = Page("T", [Stat("S", Round(1, digits: 2))])'))

    # Round args and digits error
    with pytest.raises(CompileError, match="write Round"):
        compile_program(parse('root = Page("T", [Stat("S", Round())])'))

    with pytest.raises(CompileError, match="Round digits must be 0 to 3"):
        compile_program(parse('root = Page("T", [Stat("S", Round(1.23, 5))])'))

    # Or args error
    with pytest.raises(CompileError, match="write Or"):
        compile_program(parse('root = Page("T", [Stat("S", Or(1))])'))

    # Reducer errors
    with pytest.raises(CompileError, match="Avg takes data and an optional condition"):
        compile_program(parse('root = Page("T", [Stat("S", Avg())])'))

    with pytest.raises(CompileError, match="Avg needs data like"):
        compile_program(parse('root = Page("T", [Stat("S", Avg(123))])'))

    with pytest.raises(CompileError, match="'not_q' is not a Query"):
        compile_program(
            parse('not_q = 123\nroot = Page("T", [Stat("S", Avg(not_q.steps))])')
        )

    with pytest.raises(CompileError, match="Avg needs a field"):
        compile_program(
            parse('q = Query("health.days")\nroot = Page("T", [Stat("S", Avg(q))])')
        )

    with pytest.raises(CompileError, match="'nonexistent' is not in q"):
        compile_program(
            parse(
                'q = Query("health.days")\n'
                'root = Page("T", [Stat("S", Avg(q.nonexistent))])'
            )
        )

    with pytest.raises(CompileError, match="Percent needs a condition"):
        compile_program(
            parse('q = Query("health.days")\nroot = Page("T", [Stat("S", Percent(q))])')
        )


def test_action_validation_errors() -> None:
    # Not an action (missing @)
    with pytest.raises(CompileError, match=r"an action starts with @"):
        compile_program(parse('root = Page("T", [Button("B", Navigate("health"))])'))

    # Unknown action with hint
    with pytest.raises(
        CompileError, match="unknown action @Navigates; did you mean @Navigate?"
    ):
        compile_program(parse('root = Page("T", [Button("B", @Navigates("health"))])'))

    # Pin on page instead of card
    with pytest.raises(CompileError, match="@Pin only works on cards"):
        compile_program(parse('root = Page("T", [Button("B", @Pin())])'))

    # Too many args
    with pytest.raises(CompileError, match="@Navigate takes at most 1 settings"):
        compile_program(
            parse('root = Page("T", [Button("B", @Navigate("health", "extra"))])')
        )

    # Missing required arg
    with pytest.raises(CompileError, match="@Navigate needs 'page'"):
        compile_program(parse('root = Page("T", [Button("B", @Navigate())])'))

    # @Set with undeclared var
    with pytest.raises(CompileError, match="@Set var must be a declared \\$value"):
        compile_program(parse('root = Page("T", [Button("B", @Set($undeclared, 10))])'))

    # @Log values error
    with pytest.raises(
        CompileError, match=r"@Log values must be an object like \{ml: \$ml\}"
    ):
        compile_program(parse('root = Page("T", [Button("B", @Log("template", 123))])'))

    with pytest.raises(
        CompileError, match="@Log column 'BAD-COL' must be a lowercase name"
    ):
        compile_program(
            parse('root = Page("T", [Button("B", @Log("template", {"BAD-COL": 1}))])')
        )

    # @AskBlacki text error
    with pytest.raises(CompileError, match="@AskBlacki text must be literal text"):
        compile_program(parse('root = Page("T", [Button("B", @AskBlacki(""))])'))

    # Template name not lowercase name
    with pytest.raises(CompileError, match="@Log template must be a lowercase name"):
        compile_program(
            parse('root = Page("T", [Button("B", @Log("BAD TEMPLATE", {ml: 1}))])')
        )


def test_implicit_source_overflow_and_max_nodes() -> None:
    # Fill up 8 queries, then add a component with implicit_source
    queries = "\n".join(f'q{i} = Query("health.days")' for i in range(8))
    src = f"""
    {queries}
    root = Page("T", [OctaveWeek(), Text("Hi")])
    """
    with pytest.raises(CompileError, match="at most 8 data sources"):
        compile_program(parse(src))


def test_compiler_additional_branches() -> None:
    import blacki.miniapp.compiler as comp
    from blacki.miniapp.catalog import ActionSpec, ComponentSpec, Param
    from blacki.miniapp.compiler import _Ctx, _deref
    from blacki.miniapp.lang import Ref

    bad_lines = "\n".join(f"root{i} = Text({i})" for i in range(30))
    with pytest.raises(CompileError) as exc_info:
        compile_program(parse(f'root = Page("T", [Text("Hi")])\n{bad_lines}'))
    assert len(exc_info.value.errors) == 25

    # 93-94: _BailError on root
    original_components = comp.COMPONENTS
    try:
        comp.COMPONENTS = {k: v for k, v in original_components.items() if k != "Page"}
        with pytest.raises(CompileError, match="unknown part 'Page'"):
            compile_program(parse('root = Page("T", [Text("Hi")])'))
    finally:
        comp.COMPONENTS = original_components

    # 218: negative number literal in _literal_value
    ir_neg = compile_program(
        parse('root = Page("T", [Slider($x, "S", -50, 50)])\n$x = 0')
    )
    assert ir_neg["root"]["props"]["children"][0]["props"]["min"] == -50

    # 232-236: deref chain loop (> 50)

    deref_chain = "\n".join(f"a{i} = a{i + 1}" for i in range(60))
    deref_chain += "\na60 = a0"
    prog_deref = parse(deref_chain)
    ctx_deref = _Ctx(prog_deref, "page")
    assert isinstance(_deref(ctx_deref, Ref(line=1, name="a0")), Ref)

    # 272-273: depth > MAX_DEPTH
    nested = 'Text("deep")'
    for _ in range(9):
        nested = f"Stack([{nested}])"
    with pytest.raises(CompileError, match="parts are nested too deeply"):
        compile_program(parse(f'root = Page("T", [{nested}])'))

    # 276-277: nodes > MAX_NODES (300)
    texts: list[str] = []
    for i in range(10):
        t_items = ", ".join(f'Text("t{i}_{j}")' for j in range(4))
        texts.append(f"Stack([{t_items}])")
    inner_str = ", ".join(texts)
    outer_stacks = [f"Stack([{inner_str}])" for _ in range(10)]
    root_stacks = ", ".join(outer_stacks)
    with pytest.raises(CompileError, match="a page can hold at most 300 parts"):
        compile_program(parse(f'root = Page("T", [{root_stacks}])'))

    # 320: implicit_source validation error (e.g. testing invalid params
    # via mock/custom ComponentSpec)
    import blacki.miniapp.compiler as comp

    dummy_spec = ComponentSpec(
        "DummyBadImplicit",
        "Data",
        "doc",
        (),
        implicit_source="health.sleep",
        implicit_params={"days": "999"},
    )
    original_components = comp.COMPONENTS
    try:
        comp.COMPONENTS = {**original_components, "DummyBadImplicit": dummy_spec}
        with pytest.raises(
            CompileError, match="health.sleep days must be a whole number 1..90"
        ):
            compile_program(parse('root = Page("T", [DummyBadImplicit()])'))
    finally:
        comp.COMPONENTS = original_components

    # 388-392: bool parameter validation failure
    dummy_bool = ComponentSpec(
        "DummyBool", "Control", "doc", (Param("flag", "bool", required=True),)
    )
    try:
        comp.COMPONENTS = {**original_components, "DummyBool": dummy_bool}
        with pytest.raises(CompileError, match="DummyBool flag must be true or false"):
            compile_program(parse('root = Page("T", [DummyBool(123)])'))
        ir_bool = compile_program(parse('root = Page("T", [DummyBool(true)])'))
        assert ir_bool["root"]["props"]["children"][0]["props"]["flag"] is True
    finally:
        comp.COMPONENTS = original_components

    # 463: field shorthand Ref: x: date
    # In KeyBars: KeyBars(sleep, "date", "asleep") vs KeyBars(sleep, date, asleep)
    ir_shorthand = compile_program(
        parse('q = Query("health.sleep")\nroot = Page("T", [KeyBars(q, date, asleep)])')
    )
    assert ir_shorthand["root"]["props"]["children"][0]["props"]["x"] == "date"

    # 465-466: field value not string
    with pytest.raises(
        CompileError, match='KeyBars x must be a field name in quotes, like "date"'
    ):
        compile_program(
            parse(
                'q = Query("health.sleep")\n'
                'root = Page("T", [KeyBars(q, 123, "asleep")])'
            )
        )

    # 469-470: field not in query fields
    with pytest.raises(CompileError, match="KeyBars x 'bad_field' is not in this data"):
        compile_program(
            parse(
                'q = Query("health.sleep")\n'
                'root = Page("T", [KeyBars(q, "bad_field", "asleep")])'
            )
        )

    # 474: predicate when p.of not in query_fields (e.g. data param failed)
    dummy_pred = ComponentSpec(
        "DummyPred",
        "Data",
        "doc",
        (
            Param("cond", "predicate", of="data", required=True),
            Param("data", "query", required=True),
        ),
    )
    try:
        comp.COMPONENTS = {**original_components, "DummyPred": dummy_pred}
        with pytest.raises(CompileError, match="DummyPred data must name a Query"):
            compile_program(
                parse(
                    'root = Page("T", [DummyPred(1 == 1, not_a_query)])\n'
                    "not_a_query = 123"
                )
            )
    finally:
        comp.COMPONENTS = original_components

    # 512: unsupported param kind
    dummy_unk = ComponentSpec(
        "DummyUnk", "Data", "doc", (Param("unknown", "unknown_kind", required=True),)
    )
    try:
        comp.COMPONENTS = {**original_components, "DummyUnk": dummy_unk}
        with pytest.raises(
            CompileError, match="DummyUnk unknown has an unsupported kind"
        ):
            compile_program(parse('root = Page("T", [DummyUnk(1)])'))
    finally:
        comp.COMPONENTS = original_components

    # 523-524: _query_ref with inline Query (success and failure)
    ir_inline_q = compile_program(
        parse('root = Page("T", [KeyBars(Query("health.sleep"), "date", "asleep")])')
    )
    assert "_q1" in ir_inline_q["queries"]

    with pytest.raises(CompileError):
        compile_program(
            parse('root = Page("T", [KeyBars(Query("unknown.src"), "x", "y")])')
        )

    # 546: Negate in _expr
    ir_neg_expr = compile_program(
        parse(
            'q = Query("health.days")\n'
            "$x = -(Latest(q.steps))\n"
            'root = Page("T", [Stat("S", $x)])'
        )
    )
    assert ir_neg_expr["vars"]["x"]["init"]["k"] == "neg"

    # 564: allow_any_field in Ref when scope is None
    dummy_any = ComponentSpec(
        "DummyAny",
        "Data",
        "doc",
        (
            Param("cond", "predicate", of="data", required=True),
            Param("data", "query", required=True),
        ),
    )
    try:
        comp.COMPONENTS = {**original_components, "DummyAny": dummy_any}
        ir_any = compile_program(
            parse(
                'q = Query("tracker", {template: "water"})\n'
                'root = Page("T", [DummyAny(arbitrary_field > 10, q)])'
            )
        )
        assert (
            ir_any["root"]["props"]["children"][0]["props"]["cond"]["l"]["f"]
            == "arbitrary_field"
        )
    finally:
        comp.COMPONENTS = original_components

    # 592-593: _expr when node cannot be used as value (e.g. ListExpr in expr position)
    with pytest.raises(CompileError, match="this can't be used as a value"):
        compile_program(parse('root = Page("T", [Stat("S", [1, 2])])'))

    # 676: unreachable in reducer/function, pragma covered or fallback
    # 676: function not Round, Or, or Reducer
    from blacki.miniapp.catalog import FunctionSpec

    dummy_fn = FunctionSpec("DummyFn", "doc", "sig")
    orig_functions = comp.FUNCTIONS
    try:
        comp.FUNCTIONS = {**orig_functions, "DummyFn": dummy_fn}
        with pytest.raises(CompileError, match="DummyFn can't be used here"):
            compile_program(parse('root = Page("T", [Stat("S", DummyFn())])'))
    finally:
        comp.FUNCTIONS = orig_functions

    # 706: action kwargs
    ir_act_kw = compile_program(
        parse('root = Page("T", [Button("B", @Navigate(page: "health"))])')
    )
    assert (
        ir_act_kw["root"]["props"]["children"][0]["props"]["action"]["a"] == "Navigate"
    )

    # 713: action optional param omitted (when p.required is False)

    dummy_act = ActionSpec("DummyAct", "doc", (Param("opt", "text", required=False),))
    original_actions = comp.ACTIONS
    try:
        comp.ACTIONS = {**original_actions, "DummyAct": dummy_act}
        ir_act = compile_program(parse('root = Page("T", [Button("B", @DummyAct())])'))
        assert ir_act["root"]["props"]["children"][0]["props"]["action"]["args"] == {}
    finally:
        comp.ACTIONS = original_actions
