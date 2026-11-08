"""Validate a parsed program against the catalog and compile it to render IR.

The IR is plain JSON the browser draws. Expressions become small trees the
browser's interpreter evaluates (no ``eval``). Every problem is collected
with its line so Blacki can fix several at once.
"""

from __future__ import annotations

import contextlib
import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from .catalog import ACTIONS, COMPONENTS, FUNCTIONS, REDUCERS, Param
from .lang import (
    MAX_DEPTH,
    BinaryOp,
    Call,
    Expr,
    FieldAccess,
    LangError,
    ListExpr,
    Literal,
    Negate,
    ObjectExpr,
    Program,
    Ref,
    VarRef,
)
from .sources import SOURCES, validate_params

MAX_QUERIES = 8
MAX_NODES = 300
PAGE_ID = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
_BOLD_ONLY = re.compile(r"[<>]")


class CompileError(ValueError):
    """All problems found in one program."""

    def __init__(self, errors: list[LangError]) -> None:
        self.errors = errors
        super().__init__("; ".join(str(e) for e in errors))


@dataclass
class _Ctx:
    program: Program
    kind: str  # page or card
    queries: dict[str, dict[str, Any]] = field(default_factory=dict)
    vars: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[LangError] = field(default_factory=list)
    nodes: int = 0
    resolving: list[str] = field(default_factory=list)
    anon: int = 0
    used: set[str] = field(default_factory=set)

    def err(self, message: str, line: int | None) -> None:
        if len(self.errors) < 25:
            self.errors.append(LangError(message, line))


class _BailError(Exception):
    """Abort the current parameter after recording an error."""


def compile_program(
    program: Program, *, kind: str = "page", keep_all_queries: bool = False
) -> dict[str, Any]:
    """Compile a program to IR; raises CompileError listing every problem."""
    ctx = _Ctx(program, kind)
    for key in program.order:
        st = program.statements[key]
        if not st.is_var and isinstance(st.expr, Call) and st.expr.name == "Query":
            _declare_query(ctx, st.name, st.expr)
    for key in program.order:
        st = program.statements[key]
        if st.is_var:
            _declare_var(ctx, st.name, st.expr, st.line)
    root_st = program.statements.get("root")
    root: dict[str, Any] | None = None
    title = ""
    if root_st is None:
        ctx.err('there is no root line; add  root = Page("Title", [ ... ])', None)
    elif not (isinstance(root_st.expr, Call) and root_st.expr.name == "Page"):
        ctx.err("root must be a Page(...)", root_st.line)
    else:
        try:
            root = _component(ctx, root_st.expr, depth=0)
            title = str(root["props"].get("title", ""))
        except _BailError:
            root = None
    unused = _check_unused(ctx) if root is not None else []
    if ctx.errors:
        raise CompileError(ctx.errors)
    # Data nobody reads is not fetched.
    queries = {
        name: q
        for name, q in ctx.queries.items()
        if keep_all_queries or name.startswith("_") or name in ctx.used
    }
    return {
        "version": 1,
        "title": title,
        "root": root,
        "vars": ctx.vars,
        "queries": queries,
        "unused": unused,
    }


def _check_unused(ctx: _Ctx) -> list[str]:
    """Validate lines nothing on the page uses, and report them by name."""
    unused: list[str] = []
    for key in list(ctx.program.order):
        st = ctx.program.statements[key]
        if key == "root" or key in ctx.used or st.is_var and f"${st.name}" in ctx.used:
            continue
        if st.is_var:
            unused.append(f"${st.name}")
            continue
        unused.append(st.name)
        expr = st.expr
        if isinstance(expr, Call) and expr.name in COMPONENTS and not expr.is_action:
            with contextlib.suppress(_BailError):
                _component(ctx, expr, depth=0)
    return unused


# ---------------------------------------------------------------- declarations


def _declare_var(ctx: _Ctx, name: str, expr: Expr, line: int) -> None:
    value = _literal_value(expr)
    if value is _NOT_LITERAL:
        # A starting value computed from data, e.g. Or(Latest(targets.protein_g), 140).
        if _mentions_var(expr):
            ctx.err(f"${name} can't start from another page value", line)
            return
        try:
            init = _expr(ctx, expr, scope=None)
        except _BailError:
            return
        ctx.vars[name] = {"type": "number", "value": None, "init": init}
        return
    if not isinstance(value, (int, float, str, bool)) or value is None:
        ctx.err(f"${name} must start as a number, text, true or false", line)
        return
    if isinstance(value, str) and len(value) > 120:
        ctx.err(f"${name} text is too long", line)
        return
    kind = (
        "bool"
        if isinstance(value, bool)
        else "number"
        if isinstance(value, (int, float))
        else "text"
    )
    ctx.vars[name] = {"type": kind, "value": value}


def _declare_query(ctx: _Ctx, name: str, call: Call) -> str | None:
    if call.kwargs or not 1 <= len(call.args) <= 2:
        ctx.err('write queries as  name = Query("source", {setting: value})', call.line)
        return None
    source = _literal_value(call.args[0])
    if (
        not isinstance(source, str)
        or source not in SOURCES
        or not SOURCES[source].public
    ):
        public = ", ".join(n for n, s in SOURCES.items() if s.public)
        ctx.err(f"unknown data source {source!r}; available: {public}", call.line)
        return None
    raw: dict[str, Any] = {}
    if len(call.args) == 2:
        obj = call.args[1]
        if not isinstance(obj, ObjectExpr):
            ctx.err("query settings must be an object like {days: 14}", call.line)
            return None
        for key, value_expr in obj.items:
            raw[key] = _literal_value(value_expr)
    params, errors = validate_params(source, raw)
    for message in errors:
        ctx.err(message, call.line)
    if errors:
        return None
    if len(ctx.queries) >= MAX_QUERIES:
        ctx.err(f"a page can read at most {MAX_QUERIES} data sources", call.line)
        return None
    ctx.queries[name] = {"source": source, "params": params}
    return name


def _mentions_var(expr: Any) -> bool:
    if isinstance(expr, VarRef):
        return True
    if isinstance(expr, (BinaryOp,)):
        return _mentions_var(expr.left) or _mentions_var(expr.right)
    if isinstance(expr, Negate):
        return _mentions_var(expr.operand)
    if isinstance(expr, Call):
        return any(_mentions_var(a) for a in expr.args)
    return False


def _literal_value(expr: Expr) -> Any:
    if isinstance(expr, Literal):
        return expr.value
    if (
        isinstance(expr, Negate)
        and isinstance(expr.operand, Literal)
        and isinstance(expr.operand.value, (int, float))
        and not isinstance(expr.operand.value, bool)
    ):
        return -expr.operand.value
    return _NOT_LITERAL


_NOT_LITERAL = object()


# ---------------------------------------------------------------- components


def _deref(ctx: _Ctx, expr: Expr) -> Expr:
    """Follow a bare reference to the statement it names."""
    seen = 0
    while isinstance(expr, Ref) and expr.name in ctx.program.statements:
        ctx.used.add(expr.name)
        expr = ctx.program.statements[expr.name].expr
        seen += 1
        if seen > 50:
            break
    return expr


def _component(
    ctx: _Ctx, expr: Expr, depth: int, allowed: tuple[str, ...] = ()
) -> dict[str, Any]:
    line = expr.line
    if isinstance(expr, Ref):
        if expr.name not in ctx.program.statements:
            ctx.err(f"{expr.name!r} is not defined", line)
            raise _BailError
        if expr.name in ctx.resolving:
            ctx.err(f"{expr.name!r} refers to itself", line)
            raise _BailError
        ctx.used.add(expr.name)
        ctx.resolving.append(expr.name)
        try:
            return _component(
                ctx, ctx.program.statements[expr.name].expr, depth, allowed
            )
        finally:
            ctx.resolving.pop()
    if not isinstance(expr, Call) or expr.is_action:
        ctx.err("expected a part such as Stat(...) or a name that holds one", line)
        raise _BailError
    spec = COMPONENTS.get(expr.name)
    if spec is None:
        hint = difflib.get_close_matches(expr.name, list(COMPONENTS), n=1)
        extra = f"; did you mean {hint[0]}?" if hint else ""
        ctx.err(f"unknown part {expr.name!r}{extra}", line)
        raise _BailError
    if allowed and expr.name not in allowed:
        ctx.err(f"{expr.name} can't go here; expected {' or '.join(allowed)}", line)
        raise _BailError
    if depth > MAX_DEPTH:
        ctx.err("parts are nested too deeply", line)
        raise _BailError
    ctx.nodes += 1
    if ctx.nodes > MAX_NODES:
        ctx.err(f"a page can hold at most {MAX_NODES} parts", line)
        raise _BailError

    given: dict[str, Expr] = {}
    if len(expr.args) > len(spec.params):
        ctx.err(f"{spec.name} takes at most {len(spec.params)} settings", line)
    for param, arg in zip(spec.params, expr.args, strict=False):
        given[param.name] = arg
    for key, value in expr.kwargs:
        if spec.param(key) is None:
            names = ", ".join(p.name for p in spec.params) or "none"
            ctx.err(f"{spec.name} has no setting {key!r} (settings: {names})", line)
            continue
        if key in given:
            ctx.err(f"{spec.name} setting {key!r} is given twice", line)
            continue
        given[key] = value

    props: dict[str, Any] = {}
    query_fields: dict[str, tuple[str, ...] | None] = {}
    # Data settings first, so fields and conditions can be checked against them.
    ordered = sorted(spec.params, key=lambda p: 0 if p.kind == "query" else 1)
    for param in ordered:
        if param.name not in given:
            if param.required:
                ctx.err(f"{spec.name} needs {param.name!r}", line)
            elif param.default is not None:
                props[param.name] = param.default
            continue
        try:
            props[param.name] = _param(
                ctx, spec.name, param, given[param.name], depth, query_fields
            )
        except _BailError:
            continue

    if spec.name == "Pin":  # pragma: no cover - Pin is an action, kept for clarity
        pass
    if spec.implicit_source:
        params: dict[str, Any] = {}
        for key, source in spec.implicit_params.items():
            params[key] = int(source) if source.isdigit() else props.get(source)
        valid, errors = validate_params(spec.implicit_source, params)
        for message in errors:
            ctx.err(message, line)
        if len(ctx.queries) >= MAX_QUERIES:
            ctx.err(f"a page can read at most {MAX_QUERIES} data sources", line)
        else:
            ctx.anon += 1
            qname = f"_{spec.name.lower()}{ctx.anon}"
            ctx.queries[qname] = {"source": spec.implicit_source, "params": valid}
            props["data"] = qname
    _check_child_fields(ctx, spec.name, props, query_fields, line)
    return {"type": spec.name, "line": line, "props": props}


def _check_child_fields(
    ctx: _Ctx, name: str, props: dict[str, Any], fields: dict[str, Any], line: int
) -> None:
    known = fields.get("data")
    if not known:
        return
    key = {"StackedBars": "series", "Table": "columns"}.get(name)
    for child in props.get(key, []) if key else []:
        f = child["props"].get("field")
        if f not in known:
            ctx.err(
                f"{child['type']} field {f!r} is not in this data ({', '.join(known)})",
                child["line"],
            )


def _param(
    ctx: _Ctx,
    owner: str,
    p: Param,
    expr: Expr,
    depth: int,
    query_fields: dict[str, Any],
) -> Any:
    line = expr.line
    where = f"{owner} {p.name}"
    kind = p.kind
    if kind in {"text", "rich"}:
        value = _literal_value(_deref(ctx, expr))
        if not isinstance(value, str):
            ctx.err(f"{where} must be text in quotes", line)
            raise _BailError
        if len(value) > p.max_len:
            ctx.err(f"{where} is longer than {p.max_len} characters", line)
            raise _BailError
        if kind == "rich" and _BOLD_ONLY.search(value):
            ctx.err(f"{where} can't contain < or >; only **bold** is allowed", line)
            raise _BailError
        return value
    if kind in {"number", "int"}:
        value = _literal_value(_deref(ctx, expr))
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if kind == "int":
            ok = ok and isinstance(value, int)
        if not ok:
            ctx.err(
                f"{where} must be a {'whole ' if kind == 'int' else ''}number", line
            )
            raise _BailError
        if (p.min is not None and value < p.min) or (
            p.max is not None and value > p.max
        ):
            ctx.err(f"{where} must be between {p.min:g} and {p.max:g}", line)
            raise _BailError
        return value
    if kind == "bool":
        value = _literal_value(_deref(ctx, expr))
        if not isinstance(value, bool):
            ctx.err(f"{where} must be true or false", line)
            raise _BailError
        return value
    if kind == "enum":
        value = _literal_value(_deref(ctx, expr))
        if value not in p.options:
            ctx.err(f"{where} must be one of {', '.join(p.options)}", line)
            raise _BailError
        return value
    if kind == "texts":
        target = _deref(ctx, expr)
        if not isinstance(target, ListExpr):
            ctx.err(f'{where} must be a list of text like ["a", "b"]', line)
            raise _BailError
        if not target.items or len(target.items) > p.max_items:
            ctx.err(f"{where} needs 1 to {p.max_items} items", line)
            raise _BailError
        out = []
        for item in target.items:
            value = _literal_value(item)
            if (
                not isinstance(value, str)
                or len(value) > p.max_len
                or _BOLD_ONLY.search(value)
            ):
                ctx.err(f"{where} items must be short text without < or >", item.line)
                raise _BailError
            out.append(value)
        return out
    if kind == "page":
        value = _literal_value(_deref(ctx, expr))
        if not isinstance(value, str) or not PAGE_ID.match(value):
            ctx.err(f'{where} must be a page id like "health"', line)
            raise _BailError
        return value
    if kind == "sets":
        target = _deref(ctx, expr)
        if (
            not isinstance(target, ListExpr)
            or not 1 <= len(target.items) <= p.max_items
        ):
            ctx.err(f"{where} must be a list like [[60, 8], [70, 6]]", line)
            raise _BailError
        out_sets = []
        for item in target.items:
            pair = (
                [_literal_value(x) for x in item.items]
                if isinstance(item, ListExpr)
                else []
            )
            if (
                len(pair) != 2
                or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in pair
                )
                or not 0 <= pair[0] <= 500
                or not 1 <= pair[1] <= 100
            ):
                ctx.err(
                    f"{where} items must be [kg, reps] with kg 0..500 and reps 1..100",
                    item.line,
                )
                raise _BailError
            out_sets.append(pair)
        return out_sets
    if kind == "query":
        name = _query_ref(ctx, expr, where)
        query_fields[p.name] = SOURCES[ctx.queries[name]["source"]].fields
        return name
    if kind == "field":
        value = _literal_value(_deref(ctx, expr))
        if isinstance(expr, Ref) and expr.name not in ctx.program.statements:
            value = expr.name  # allow  x: date  as shorthand for "date"
        if not isinstance(value, str):
            ctx.err(f'{where} must be a field name in quotes, like "date"', line)
            raise _BailError
        known = query_fields.get(p.of)
        if known and value not in known:
            ctx.err(f"{where} {value!r} is not in this data ({', '.join(known)})", line)
            raise _BailError
        return value
    if kind == "predicate":
        if p.of not in query_fields:
            raise _BailError
        return _expr(
            ctx,
            expr,
            scope=query_fields[p.of],
            allow_any_field=query_fields[p.of] is None,
        )
    if kind == "value":
        return _expr(ctx, expr, scope=None)
    if kind == "var":
        if not isinstance(expr, VarRef) or expr.name not in ctx.vars:
            ctx.err(
                f"{where} must be a page value like $goal, declared with  $goal = 140",
                line,
            )
            raise _BailError
        ctx.used.add(f"${expr.name}")
        return expr.name
    if kind == "action":
        return _action(ctx, expr)
    if kind in {"children", "child"}:
        target = _deref(ctx, expr)
        if kind == "child":
            return _component(ctx, target, depth + 1, p.allowed)
        if not isinstance(target, ListExpr):
            ctx.err(f"{where} must be a list like [a, b]", line)
            raise _BailError
        if len(target.items) > p.max_items:
            ctx.err(f"{where} holds at most {p.max_items} parts", line)
            raise _BailError
        children = []
        for item in target.items:
            try:
                children.append(_component(ctx, item, depth + 1, p.allowed))
            except _BailError:
                continue
        return children
    ctx.err(f"{where} has an unsupported kind {kind}", line)  # pragma: no cover
    raise _BailError


def _query_ref(ctx: _Ctx, expr: Expr, where: str) -> str:
    if isinstance(expr, Ref) and expr.name in ctx.queries:
        ctx.used.add(expr.name)
        return expr.name
    if isinstance(expr, Call) and expr.name == "Query":
        ctx.anon += 1
        name = _declare_query(ctx, f"_q{ctx.anon}", expr)
        if name:
            return name
        raise _BailError
    ctx.err(
        f'{where} must name a Query, e.g.  sleep = Query("health.sleep")', expr.line
    )
    raise _BailError


# ---------------------------------------------------------------- expressions


def _expr(
    ctx: _Ctx, expr: Expr, scope: tuple[str, ...] | None, allow_any_field: bool = False
) -> dict[str, Any]:
    line = expr.line
    if isinstance(expr, Literal) and (
        expr.value is None or isinstance(expr.value, (int, float, str, bool))
    ):
        if isinstance(expr.value, str) and len(expr.value) > 200:
            ctx.err("text value is too long", line)
            raise _BailError
        return {"k": "lit", "v": expr.value}
    if isinstance(expr, Negate):
        return {"k": "neg", "a": _expr(ctx, expr.operand, scope, allow_any_field)}
    if isinstance(expr, VarRef):
        if expr.name not in ctx.vars:
            ctx.err(f"${expr.name} is not declared; add  ${expr.name} = <value>", line)
            raise _BailError
        ctx.used.add(f"${expr.name}")
        return {"k": "var", "n": expr.name}
    if isinstance(expr, BinaryOp):
        return {
            "k": "bin",
            "o": expr.op,
            "l": _expr(ctx, expr.left, scope, allow_any_field),
            "r": _expr(ctx, expr.right, scope, allow_any_field),
        }
    if isinstance(expr, Ref):
        if scope is not None and expr.name in scope:
            return {"k": "row", "f": expr.name}
        if allow_any_field and expr.name not in ctx.program.statements:
            return {"k": "row", "f": expr.name}
        st = ctx.program.statements.get(expr.name)
        if st is None:
            fields = f" (row fields here: {', '.join(scope)})" if scope else ""
            ctx.err(f"{expr.name!r} is not defined{fields}", line)
            raise _BailError
        if expr.name in ctx.queries:
            ctx.err(
                f"{expr.name} is data; use Count({expr.name}) or Avg({expr.name}.field)",  # noqa: E501
                line,
            )
            raise _BailError
        if expr.name in ctx.resolving:
            ctx.err(f"{expr.name!r} refers to itself", line)
            raise _BailError
        ctx.used.add(expr.name)
        ctx.resolving.append(expr.name)
        try:
            return _expr(ctx, st.expr, scope, allow_any_field)
        finally:
            ctx.resolving.pop()
    if isinstance(expr, FieldAccess):
        ctx.err(
            "use Latest(data.field), Avg(data.field) or similar to read a field", line
        )
        raise _BailError
    if isinstance(expr, Call) and not expr.is_action:
        return _function(ctx, expr, scope, allow_any_field)
    ctx.err("this can't be used as a value", line)
    raise _BailError


def _function(
    ctx: _Ctx, call: Call, scope: tuple[str, ...] | None, allow_any_field: bool
) -> dict[str, Any]:
    line = call.line
    if call.name not in FUNCTIONS or call.name == "Query":
        if call.name in COMPONENTS:
            ctx.err(f"{call.name} is a part, not a value", line)
        else:
            hint = difflib.get_close_matches(call.name, list(FUNCTIONS), n=1)
            ctx.err(
                f"unknown function {call.name!r}"
                + (f"; did you mean {hint[0]}?" if hint else ""),
                line,
            )
        raise _BailError
    if call.kwargs:
        ctx.err(f"{call.name} takes no named settings", line)
        raise _BailError
    if call.name == "Round":
        if not 1 <= len(call.args) <= 2:
            ctx.err("write Round(value) or Round(value, digits)", line)
            raise _BailError
        digits = _literal_value(call.args[1]) if len(call.args) == 2 else 0
        if not isinstance(digits, int) or not 0 <= digits <= 3:
            ctx.err("Round digits must be 0 to 3", line)
            raise _BailError
        return {
            "k": "fn",
            "f": "round",
            "a": _expr(ctx, call.args[0], scope, allow_any_field),
            "d": digits,
        }
    if call.name == "Or":
        if len(call.args) < 2 or len(call.args) > 4:
            ctx.err("write Or(value, fallback)", line)
            raise _BailError
        return {
            "k": "fn",
            "f": "or",
            "a": [_expr(ctx, a, scope, allow_any_field) for a in call.args],
        }
    if call.name in REDUCERS:
        if not 1 <= len(call.args) <= 2:
            ctx.err(f"{call.name} takes data and an optional condition", line)
            raise _BailError
        target = call.args[0]
        field_name: str | None = None
        if isinstance(target, FieldAccess) and isinstance(target.base, Ref):
            qname, field_name = target.base.name, target.name
        elif isinstance(target, Ref):
            qname = target.name
        else:
            ctx.err(f"{call.name} needs data like  sleep  or  sleep.deep", line)
            raise _BailError
        if qname not in ctx.queries:
            ctx.err(f"{qname!r} is not a Query", line)
            raise _BailError
        ctx.used.add(qname)
        fields = SOURCES[ctx.queries[qname]["source"]].fields
        needs_field = call.name not in {"Count", "Percent"}
        if needs_field and not field_name:
            ctx.err(f"{call.name} needs a field, e.g. {call.name}({qname}.value)", line)
            raise _BailError
        if field_name and fields and field_name not in fields:
            ctx.err(f"{field_name!r} is not in {qname} ({', '.join(fields)})", line)
            raise _BailError
        if call.name == "Percent" and len(call.args) != 2:
            ctx.err(
                "Percent needs a condition, e.g. Percent(food, protein >= $goal)", line
            )
            raise _BailError
        node: dict[str, Any] = {"k": "fn", "f": call.name.lower(), "q": qname}
        if field_name:
            node["field"] = field_name
        if len(call.args) == 2:
            node["where"] = _expr(
                ctx, call.args[1], fields, allow_any_field=fields is None
            )
        return node
    ctx.err(f"{call.name} can't be used here", line)  # pragma: no cover
    raise _BailError


# ---------------------------------------------------------------- actions


def _action(ctx: _Ctx, expr: Expr) -> dict[str, Any]:
    line = expr.line
    if not isinstance(expr, Call) or not expr.is_action:
        ctx.err('an action starts with @, e.g. @Navigate("health")', line)
        raise _BailError
    spec = ACTIONS.get(expr.name)
    if spec is None:
        hint = difflib.get_close_matches(expr.name, list(ACTIONS), n=1)
        ctx.err(
            f"unknown action @{expr.name}"
            + (f"; did you mean @{hint[0]}?" if hint else ""),
            line,
        )
        raise _BailError
    if expr.name == "Pin" and ctx.kind != "card":
        ctx.err("@Pin only works on cards", line)
        raise _BailError
    given: dict[str, Expr] = dict(
        zip((p.name for p in spec.params), expr.args, strict=False)
    )
    if len(expr.args) > len(spec.params):
        ctx.err(f"@{spec.name} takes at most {len(spec.params)} settings", line)
        raise _BailError
    for key, value in expr.kwargs:
        given[key] = value
    args: dict[str, Any] = {}
    for p in spec.params:
        if p.name not in given:
            if p.required:
                ctx.err(f"@{spec.name} needs {p.name!r}", line)
                raise _BailError
            continue
        value_expr = given[p.name]
        if p.kind == "var":
            if not isinstance(value_expr, VarRef) or value_expr.name not in ctx.vars:
                ctx.err(f"@{spec.name} {p.name} must be a declared $value", line)
                raise _BailError
            ctx.used.add(f"${value_expr.name}")
            args[p.name] = value_expr.name
        elif p.kind == "value":
            args[p.name] = _expr(ctx, value_expr, scope=None)
        elif p.kind == "values":
            if (
                not isinstance(value_expr, ObjectExpr)
                or not 1 <= len(value_expr.items) <= 12
            ):
                ctx.err(
                    f"@{spec.name} {p.name} must be an object like {{ml: $ml}}", line
                )
                raise _BailError
            args[p.name] = {}
            for key, item in value_expr.items:
                if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
                    ctx.err(
                        f"@{spec.name} column {key!r} must be a lowercase name", line
                    )
                    raise _BailError
                args[p.name][key] = _expr(ctx, item, scope=None)
        elif p.kind == "text" and spec.name == "AskBlacki":
            value = _literal_value(
                value_expr
            )  # literal only: no data or values spliced in
            if not isinstance(value, str) or not 1 <= len(value) <= p.max_len:
                ctx.err(
                    f"@AskBlacki text must be literal text up to {p.max_len} characters",  # noqa: E501
                    line,
                )
                raise _BailError
            args[p.name] = value
        elif p.kind == "text":
            value = _literal_value(value_expr)
            if not isinstance(value, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}", value
            ):
                ctx.err(f"@{spec.name} {p.name} must be a lowercase name", line)
                raise _BailError
            args[p.name] = value
        else:
            args[p.name] = _param(ctx, f"@{spec.name}", p, value_expr, 0, {})
    return {"a": spec.name, "args": args, "saves": spec.saves}
