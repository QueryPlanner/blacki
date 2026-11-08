from __future__ import annotations

import inspect
import sys

import pytest

from blacki.miniapp.compiler import compile_program
from blacki.miniapp.lang import (
    MAX_EXPRESSION_OPERATORS,
    MAX_SOURCE_BYTES,
    BinaryOp,
    Call,
    FieldAccess,
    LangError,
    ListExpr,
    Literal,
    Negate,
    ObjectExpr,
    Program,
    Ref,
    Statement,
    Token,
    VarRef,
    _Parser,
    format_expr,
    parse,
    tokenize,
)


def test_program_manipulation_and_formatting() -> None:
    p = Program()
    st1 = Statement("title", Literal(1, "Hello"), line=1)
    st2 = Statement("count", Literal(2, 42), line=2, is_var=True)
    p.set(st1)
    p.set(st2)
    assert p.order == ["title", "$count"]
    assert p.statements["title"] == st1
    assert p.statements["$count"] == st2

    # Replace existing
    st1_new = Statement("title", Literal(3, "World"), line=3)
    p.set(st1_new)
    assert p.order == ["title", "$count"]
    assert p.statements["title"] == st1_new

    # to_source
    src = p.to_source()
    assert 'title = "World"' in src
    assert "$count = 42" in src

    # remove
    assert p.remove("title") is True
    assert p.remove("nonexistent") is False
    assert "title" not in p.statements
    assert p.order == ["$count"]


def test_format_expr_all_variants() -> None:
    assert format_expr(Literal(1, None)) == "null"
    assert format_expr(Literal(1, True)) == "true"
    assert format_expr(Literal(1, False)) == "false"
    assert format_expr(Literal(1, 'hello\n"world\\')) == r'"hello\n\"world\\"'
    assert format_expr(Literal(1, 3.14)) == "3.14"

    assert format_expr(ListExpr(1, (Literal(1, 1), Literal(1, 2)))) == "[1, 2]"
    assert (
        format_expr(ObjectExpr(1, (("a", Literal(1, 1)), ("b", Literal(1, 2)))))
        == "{a: 1, b: 2}"
    )
    assert format_expr(Ref(1, "myRef")) == "myRef"
    assert format_expr(VarRef(1, "myVar")) == "$myVar"

    # Call normal & action
    c1 = Call(1, "Func", (Literal(1, 1),), (("k", Literal(1, "v")),), is_action=False)
    assert format_expr(c1) == 'Func(1, k: "v")'
    c2 = Call(1, "Action", (), (("opt", Literal(1, True)),), is_action=True)
    assert format_expr(c2) == "@Action(opt: true)"

    # FieldAccess
    fa = FieldAccess(1, Ref(1, "row"), "field")
    assert format_expr(fa) == "row.field"

    # BinaryOp & Negate with wrapping
    b1 = BinaryOp(1, "+", Literal(1, 1), Literal(1, 2))
    assert format_expr(b1) == "1 + 2"
    b2 = BinaryOp(1, "*", b1, Literal(1, 3))
    assert format_expr(b2) == "(1 + 2) * 3"
    neg = Negate(1, b1)
    assert format_expr(neg) == "-(1 + 2)"
    neg_lit = Negate(1, Literal(1, 5))
    assert format_expr(neg_lit) == "-5"


def test_tokenize_errors() -> None:
    with pytest.raises(LangError, match="larger than 16 KB"):
        tokenize("x = 1\n" + " " * (MAX_SOURCE_BYTES + 10))

    with pytest.raises(LangError, match="text is missing its closing quote"):
        tokenize('x = "unclosed\n')

    with pytest.raises(LangError, match="text is missing its closing quote"):
        tokenize('x = "unclosed')

    with pytest.raises(LangError, match=r"unknown escape \\z"):
        tokenize(r'x = "bad \z"')

    with pytest.raises(LangError, match="bad number 1.2.3"):
        tokenize("x = 1.2.3")

    with pytest.raises(LangError, match="bad number 1."):
        tokenize("x = 1.")

    with pytest.raises(LangError, match="unexpected character"):
        tokenize("x = %")

    with pytest.raises(LangError, match="a bracket is never closed"):
        tokenize("x = [1, 2")


def test_tokenize_and_parse_valid() -> None:
    src = """
    # Full valid program test
    $goal = 120
    root = Page("My Page", [card1], subtitle: "A note")
    card1 = Card("Card Title", [t1])
    t1 = Text("Hello \\n \\t \\\" \\\\ world")
    calc = -row.val + (2 * 3 - 4 / 2)
    cmp1 = (a <= b) != (c >= d)
    cmp2 = (e < f) == (g > h)
    act = @RunAction(a, b: 2)
    obj = {x: 1, "y": 2}
    empty_obj = {}
    lst = [1, 2]
    empty_lst = []
    call_no_args = Query()
    bool_t = true
    bool_f = false
    null_v = null
    float_num = 12_3.45_6
    use_var = $goal
    """
    prog = parse(src)
    assert "$goal" in prog.statements
    assert "root" in prog.statements
    assert "card1" in prog.statements
    assert "t1" in prog.statements
    assert "calc" in prog.statements
    assert "cmp1" in prog.statements
    assert "cmp2" in prog.statements
    assert "act" in prog.statements
    assert "obj" in prog.statements
    assert "empty_obj" in prog.statements
    assert "lst" in prog.statements
    assert "empty_lst" in prog.statements
    assert "call_no_args" in prog.statements
    assert "bool_t" in prog.statements
    assert "bool_f" in prog.statements
    assert "null_v" in prog.statements
    assert "float_num" in prog.statements
    assert "use_var" in prog.statements


def test_parse_eof_and_newline_handling() -> None:
    # Program ending with statement and immediately EOF (no trailing newline)
    prog = parse("x = 1")
    assert "x" in prog.statements

    # Empty source
    empty = parse("")
    assert len(empty.statements) == 0

    # Source with newline tokens in program loop
    # We pass synthetic tokens with leading newline to hit lines 291-292
    tokens = [
        Token("newline", "\n", 1),
        Token("name", "x", 2),
        Token("op", "=", 2),
        Token("number", "1", 2),
        Token("eof", "", 2),
    ]
    prog_nl = _Parser(tokens).program()
    assert "x" in prog_nl.statements


def test_parse_statement_errors() -> None:
    with pytest.raises(LangError, match="a line must start with a name"):
        parse("123 = 456")

    with pytest.raises(LangError, match="expected '=' but found"):
        parse("x + 1")

    with pytest.raises(LangError, match="unexpected 'extra' after the expression"):
        parse("x = 1 extra")

    with pytest.raises(LangError, match="more than 200 statements"):
        too_many = "\n".join(f"st_{i} = {i}" for i in range(205))
        parse(too_many)


def test_parse_expression_errors() -> None:
    # Nested too deeply
    deep = "1"
    for _ in range(40):
        deep = f"({deep})"
    with pytest.raises(LangError, match="expression is nested too deeply"):
        parse(f"x = {deep}")

    with pytest.raises(LangError, match="expected a field name after '.'"):
        parse("x = a.123")

    with pytest.raises(LangError, match=r"expected a page value name after '\$'"):
        parse("x = $123")

    with pytest.raises(LangError, match="expected an action name after '@'"):
        parse("x = @123")

    with pytest.raises(LangError, match="object keys must be names"):
        parse("x = {1: 2}")

    with pytest.raises(LangError, match="setting 'a' is given twice"):
        parse("x = Func(a: 1, a: 2)")

    with pytest.raises(LangError, match="put named settings after the others"):
        parse("x = Func(a: 1, 2)")

    with pytest.raises(LangError, match="unexpected"):
        parse("x = =")

    with pytest.raises(LangError, match="unexpected"):
        parse("x = +")


def test_parser_synthetic_tokens() -> None:
    # 1. Non-op token (e.g. newline or eof) in primary() hitting line 376->411  # noqa: E501
    tokens1 = [
        Token("name", "x", 1),
        Token("op", "=", 1),
        Token("newline", "\n", 1),
        Token("eof", "", 1),
    ]
    with pytest.raises(LangError, match="unexpected end of line"):
        _Parser(tokens1).program()

    # 2. op token '$' followed by non-name token (e.g. number) hitting line 380
    tokens2 = [
        Token("name", "x", 1),
        Token("op", "=", 1),
        Token("op", "$", 1),
        Token("number", "123", 1),
        Token("eof", "", 1),
    ]
    with pytest.raises(LangError, match=r"expected a page value name after '\$'"):
        _Parser(tokens2).program()

    # 3. op token '@' followed by non-name token hitting line 385
    tokens3 = [
        Token("name", "x", 1),
        Token("op", "=", 1),
        Token("op", "@", 1),
        Token("number", "123", 1),
        Token("eof", "", 1),
    ]
    with pytest.raises(LangError, match="expected an action name after '@'"):
        _Parser(tokens3).program()


def _page_with_expression(expr: str) -> str:
    return f'x = {expr}\nroot = Page("T", [Text("Hi")])'


@pytest.mark.parametrize(
    "expr",
    [
        "- " * 200 + "1",
        " + ".join(["1"] * 201),
        "- " * MAX_EXPRESSION_OPERATORS + "1",  # exactly at the budget
    ],
    ids=["unary-200", "sum-200", "unary-at-budget"],
)
def test_operator_chains_within_budget_compile(expr: str) -> None:
    """Chains a real page could write must parse, format and compile."""
    program = parse(_page_with_expression(expr))
    assert program.to_source()
    assert "root" in compile_program(program)


@pytest.mark.parametrize(
    "expr",
    [
        "- " * (MAX_EXPRESSION_OPERATORS + 1) + "1",
        " + ".join(["1"] * (MAX_EXPRESSION_OPERATORS + 2)),
        "a" + ".b" * (MAX_EXPRESSION_OPERATORS + 1),
        "- " * 2000 + "1",
        " + ".join(["1"] * 3000),
    ],
    ids=["unary-over", "sum-over", "field-over", "unary-2000", "sum-3000"],
)
def test_operator_chains_over_budget_raise_lang_error(expr: str) -> None:
    """Oversized chains fail as LangError, never as an uncaught RecursionError."""
    with pytest.raises(
        LangError,
        match=f"an expression can have at most {MAX_EXPRESSION_OPERATORS} operators",
    ):
        parse(f"x = {expr}")


def _stack_depth() -> int:
    return len(inspect.stack(0))


def test_parse_reports_real_recursion_failure_as_lang_error() -> None:
    """Force a genuine RecursionError inside parse and check it is converted.

    30 nested parentheses stay under the expression nesting check, so the
    parser recurses about 9 frames per level and hits the lowered limit.
    """
    nested = "1"
    for _ in range(30):
        nested = f"({nested})"
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(_stack_depth() + 60)
    try:
        with pytest.raises(LangError, match="expression is nested too deeply") as info:
            parse(f"x = {nested}")
    finally:
        sys.setrecursionlimit(original_limit)
    assert isinstance(info.value.__cause__, RecursionError)
