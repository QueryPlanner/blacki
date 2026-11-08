"""Parser for the Blacki UI language.

The language is a small, line-oriented notation Blacki writes to describe a
page. Each statement is ``name = expression``; ``root`` is the entry point.
Page values are declared as ``$name = literal``. Statements may span lines
while brackets are open, and ``#`` starts a comment.

Nothing here executes code: the parser only builds an AST that the compiler
validates against the component catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_SOURCE_BYTES = 16 * 1024
MAX_STATEMENTS = 200
MAX_DEPTH = 8
# Operators (unary -, binary ops, field access) per statement. Bounds the
# recursion the parser and compiler use on left-deep chains like 1+1+1+...
MAX_EXPRESSION_OPERATORS = 250


class LangError(ValueError):
    """A parse or validation error tied to a source line."""

    def __init__(self, message: str, line: int | None = None) -> None:
        self.message = message
        self.line = line
        super().__init__(f"line {line}: {message}" if line else message)


# ---------------------------------------------------------------- AST nodes


@dataclass(frozen=True)
class Node:
    line: int


@dataclass(frozen=True)
class Literal(Node):
    value: str | int | float | bool | None


@dataclass(frozen=True)
class ListExpr(Node):
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class ObjectExpr(Node):
    items: tuple[tuple[str, Expr], ...]


@dataclass(frozen=True)
class Ref(Node):
    """A bare identifier: another statement, or a row field in predicates."""

    name: str


@dataclass(frozen=True)
class VarRef(Node):
    name: str


@dataclass(frozen=True)
class Call(Node):
    name: str
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...]
    is_action: bool = False


@dataclass(frozen=True)
class FieldAccess(Node):
    base: Expr
    name: str


@dataclass(frozen=True)
class BinaryOp(Node):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Negate(Node):
    operand: Expr


Expr = (
    Literal
    | ListExpr
    | ObjectExpr
    | Ref
    | VarRef
    | Call
    | FieldAccess
    | BinaryOp
    | Negate
)


@dataclass(frozen=True)
class Statement:
    name: str
    expr: Expr
    line: int
    is_var: bool = False


@dataclass
class Program:
    statements: dict[str, Statement] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def set(self, statement: Statement) -> None:
        """Add or replace a statement; the last definition wins."""
        key = f"${statement.name}" if statement.is_var else statement.name
        if key not in self.statements:
            self.order.append(key)
        self.statements[key] = statement

    def remove(self, name: str) -> bool:
        if name not in self.statements:
            return False
        del self.statements[name]
        self.order.remove(name)
        return True

    def to_source(self) -> str:
        """Render the program back to canonical source text."""
        return "\n".join(
            f"{key} = {format_expr(self.statements[key].expr)}" for key in self.order
        )


# ---------------------------------------------------------------- tokenizer


@dataclass(frozen=True)
class Token:
    kind: str  # name, number, string, op, newline, eof
    value: str
    line: int


_OPERATORS = (
    "<=",
    ">=",
    "==",
    "!=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "=",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ",",
    ":",
    ".",
    "$",
    "@",
)
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def tokenize(source: str) -> list[Token]:
    """Split source into tokens; newlines inside brackets are ignored."""
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise LangError(f"code is larger than {MAX_SOURCE_BYTES // 1024} KB")
    tokens: list[Token] = []
    depth = 0
    line = 1
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == "\n":
            if depth == 0 and tokens and tokens[-1].kind != "newline":
                tokens.append(Token("newline", "\n", line))
            line += 1
            i += 1
            continue
        if ch in " \t\r":
            i += 1
            continue
        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
            continue
        if ch == '"':
            i += 1
            out: list[str] = []
            while True:
                if i >= n or source[i] == "\n":
                    raise LangError("text is missing its closing quote", line)
                c = source[i]
                if c == "\\" and i + 1 < n:
                    esc = source[i + 1]
                    if esc not in _ESCAPES:
                        raise LangError(f"unknown escape \\{esc}", line)
                    out.append(_ESCAPES[esc])
                    i += 2
                    continue
                if c == '"':
                    i += 1
                    break
                out.append(c)
                i += 1
            tokens.append(Token("string", "".join(out), line))
            continue
        if ch.isdigit():
            start = i
            while i < n and (source[i].isdigit() or source[i] in "._"):
                i += 1
            text = source[start:i].replace("_", "")
            if text.count(".") > 1 or text.endswith("."):
                raise LangError(f"bad number {source[start:i]}", line)
            tokens.append(Token("number", text, line))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            tokens.append(Token("name", source[start:i], line))
            continue
        for op in _OPERATORS:
            if source.startswith(op, i):
                if op in "([{":
                    depth += 1
                elif op in ")]}":
                    depth = max(0, depth - 1)
                tokens.append(Token("op", op, line))
                i += len(op)
                break
        else:
            raise LangError(f"unexpected character {ch!r}", line)
    if depth:
        raise LangError("a bracket is never closed", line)
    tokens.append(Token("eof", "", line))
    return tokens


# ---------------------------------------------------------------- parser


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.depth = 0
        self.operators = 0

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def next(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def accept(self, value: str) -> bool:
        tok = self.peek()
        if tok.kind == "op" and tok.value == value:
            self.pos += 1
            return True
        return False

    def expect(self, value: str) -> Token:
        tok = self.next()
        if tok.kind != "op" or tok.value != value:
            shown = tok.value if tok.kind != "newline" else "end of line"
            raise LangError(f"expected {value!r} but found {shown!r}", tok.line)
        return tok

    def program(self) -> Program:
        program = Program()
        while True:
            tok = self.peek()
            if tok.kind == "eof":
                return program
            if tok.kind == "newline":
                self.pos += 1
                continue
            program.set(self.statement())
            if len(program.statements) > MAX_STATEMENTS:
                raise LangError(f"more than {MAX_STATEMENTS} statements", tok.line)

    def spend_operator(self, line: int) -> None:
        """Count an operator against the per-statement budget."""
        self.operators += 1
        if self.operators > MAX_EXPRESSION_OPERATORS:
            raise LangError(
                f"an expression can have at most {MAX_EXPRESSION_OPERATORS} operators",
                line,
            )

    def statement(self) -> Statement:
        self.operators = 0
        first = self.peek()
        is_var = self.accept("$")
        tok = self.next()
        if tok.kind != "name":
            raise LangError(
                "a line must start with a name, like  chart = ...", tok.line
            )
        self.expect("=")
        expr = self.expr()
        end = self.next()
        if end.kind not in {"newline", "eof"}:
            raise LangError(f"unexpected {end.value!r} after the expression", end.line)
        if end.kind == "eof":
            self.pos -= 1
        return Statement(tok.value, expr, first.line, is_var=is_var)

    def expr(self) -> Expr:
        self.depth += 1
        if self.depth > MAX_DEPTH * 4:
            raise LangError("expression is nested too deeply", self.peek().line)
        try:
            return self.comparison()
        finally:
            self.depth -= 1

    def comparison(self) -> Expr:
        left = self.additive()
        tok = self.peek()
        if tok.kind == "op" and tok.value in {"<", "<=", ">", ">=", "==", "!="}:
            self.pos += 1
            self.spend_operator(tok.line)
            return BinaryOp(tok.line, tok.value, left, self.additive())
        return left

    def additive(self) -> Expr:
        left = self.term()
        while self.peek().kind == "op" and self.peek().value in {"+", "-"}:
            tok = self.next()
            self.spend_operator(tok.line)
            left = BinaryOp(tok.line, tok.value, left, self.term())
        return left

    def term(self) -> Expr:
        left = self.unary()
        while self.peek().kind == "op" and self.peek().value in {"*", "/"}:
            tok = self.next()
            self.spend_operator(tok.line)
            left = BinaryOp(tok.line, tok.value, left, self.unary())
        return left

    def unary(self) -> Expr:
        tok = self.peek()
        if tok.kind == "op" and tok.value == "-":
            self.pos += 1
            self.spend_operator(tok.line)
            return Negate(tok.line, self.unary())
        return self.postfix()

    def postfix(self) -> Expr:
        node = self.primary()
        while self.accept("."):
            tok = self.next()
            if tok.kind != "name":
                raise LangError("expected a field name after '.'", tok.line)
            self.spend_operator(tok.line)
            node = FieldAccess(tok.line, node, tok.value)
        return node

    def primary(self) -> Expr:
        tok = self.next()
        if tok.kind == "number":
            value = float(tok.value) if "." in tok.value else int(tok.value)
            return Literal(tok.line, value)
        if tok.kind == "string":
            return Literal(tok.line, tok.value)
        if tok.kind == "name":
            if tok.value in {"true", "false"}:
                return Literal(tok.line, tok.value == "true")
            if tok.value == "null":
                return Literal(tok.line, None)
            if self.peek().kind == "op" and self.peek().value == "(":
                return self.call(tok, is_action=False)
            return Ref(tok.line, tok.value)
        if tok.kind == "op":
            if tok.value == "$":
                name = self.next()
                if name.kind != "name":
                    raise LangError("expected a page value name after '$'", tok.line)
                return VarRef(tok.line, name.value)
            if tok.value == "@":
                name = self.next()
                if name.kind != "name":
                    raise LangError("expected an action name after '@'", tok.line)
                return self.call(name, is_action=True)
            if tok.value == "(":
                inner = self.expr()
                self.expect(")")
                return inner
            if tok.value == "[":
                items: list[Expr] = []
                while not self.accept("]"):
                    items.append(self.expr())
                    if not self.accept(","):
                        self.expect("]")
                        break
                return ListExpr(tok.line, tuple(items))
            if tok.value == "{":
                pairs: list[tuple[str, Expr]] = []
                while not self.accept("}"):
                    key = self.next()
                    if key.kind not in {"name", "string"}:
                        raise LangError("object keys must be names", key.line)
                    self.expect(":")
                    pairs.append((key.value, self.expr()))
                    if not self.accept(","):
                        self.expect("}")
                        break
                return ObjectExpr(tok.line, tuple(pairs))
        shown = "end of line" if tok.kind in {"newline", "eof"} else repr(tok.value)
        raise LangError(f"unexpected {shown}", tok.line)

    def call(self, name: Token, *, is_action: bool) -> Call:
        self.expect("(")
        args: list[Expr] = []
        kwargs: list[tuple[str, Expr]] = []
        while not self.accept(")"):
            tok = self.peek()
            nxt = self.peek(1)
            if tok.kind == "name" and nxt.kind == "op" and nxt.value == ":":
                self.pos += 2
                if any(k == tok.value for k, _ in kwargs):
                    raise LangError(f"setting {tok.value!r} is given twice", tok.line)
                kwargs.append((tok.value, self.expr()))
            else:
                if kwargs:
                    raise LangError("put named settings after the others", tok.line)
                args.append(self.expr())
            if not self.accept(","):
                self.expect(")")
                break
        return Call(name.line, name.value, tuple(args), tuple(kwargs), is_action)


def parse(source: str) -> Program:
    """Parse source text into a program. Raises LangError on bad syntax.

    Recursion is bounded by the depth and operator budgets; a RecursionError
    that still gets through is reported as a LangError, never as a crash.
    """
    try:
        return _Parser(tokenize(source)).program()
    except RecursionError as exc:
        raise LangError("expression is nested too deeply") from exc


# ---------------------------------------------------------------- formatting


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return repr(value)


def format_expr(expr: Expr) -> str:
    """Render an expression as canonical source."""
    if isinstance(expr, Literal):
        return _format_value(expr.value)
    if isinstance(expr, ListExpr):
        return "[" + ", ".join(format_expr(i) for i in expr.items) + "]"
    if isinstance(expr, ObjectExpr):
        return "{" + ", ".join(f"{k}: {format_expr(v)}" for k, v in expr.items) + "}"
    if isinstance(expr, Ref):
        return expr.name
    if isinstance(expr, VarRef):
        return f"${expr.name}"
    if isinstance(expr, Call):
        parts = [format_expr(a) for a in expr.args]
        parts += [f"{k}: {format_expr(v)}" for k, v in expr.kwargs]
        return f"{'@' if expr.is_action else ''}{expr.name}({', '.join(parts)})"
    if isinstance(expr, FieldAccess):
        return f"{format_expr(expr.base)}.{expr.name}"
    if isinstance(expr, BinaryOp):
        return f"{_wrap(expr.left)} {expr.op} {_wrap(expr.right)}"
    if isinstance(expr, Negate):
        return f"-{_wrap(expr.operand)}"
    raise TypeError(f"unknown node {expr!r}")  # pragma: no cover


def _wrap(expr: Expr) -> str:
    text = format_expr(expr)
    return f"({text})" if isinstance(expr, BinaryOp) else text
