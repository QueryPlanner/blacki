#!/usr/bin/env python3
"""Write selected environment variables as a private Docker Compose env file."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_ENVIRONMENT_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def quote_compose_value(value: str) -> str:
    """Quote one value without allowing Compose interpolation or line injection."""
    if any(character in value for character in ("\0", "\r", "\n")):
        raise ValueError("environment values cannot contain NUL or line breaks")

    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
    return f'"{escaped}"'


def serialize_compose_env(
    names: Sequence[str],
    environment: Mapping[str, str],
) -> str:
    """Serialize named variables from an environment mapping."""
    lines: list[str] = []
    for name in names:
        if _ENVIRONMENT_KEY.fullmatch(name) is None:
            raise ValueError(f"invalid environment key: {name}")
        if name not in environment:
            raise KeyError(f"missing environment key: {name}")
        lines.append(f"{name}={quote_compose_value(environment[name])}")
    return "\n".join(lines) + "\n"


def write_compose_env(
    path: Path,
    names: Sequence[str],
    environment: Mapping[str, str],
) -> None:
    """Create a mode-0600 Compose env file without replacing an existing file."""
    contents = serialize_compose_env(names, environment)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(contents)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the requested variables and return a command-line exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        print(
            "usage: write_compose_env.py OUTPUT VARIABLE [VARIABLE ...]",
            file=sys.stderr,
        )
        return 2

    try:
        write_compose_env(Path(arguments[0]), arguments[1:], os.environ)
    except (KeyError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
