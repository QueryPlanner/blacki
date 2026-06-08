"""Safety and validation layer for declarative database identifiers and types."""

from __future__ import annotations

import re

# Strict regex matching letters, numbers, and underscores,
# starting with a letter or underscore.
IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Safe SQLite types allowlist
ALLOWED_TYPES = {"TEXT", "INTEGER", "REAL", "BLOB"}

# Standard SQLite reserved keywords blocklist to avoid syntax overlaps and confusion
RESERVED_KEYWORDS = {
    "SELECT",
    "DROP",
    "ALTER",
    "DELETE",
    "INSERT",
    "PRAGMA",
    "CREATE",
    "TABLE",
    "INDEX",
    "UPDATE",
    "WHERE",
    "AND",
    "OR",
    "JOIN",
    "FROM",
    "LIMIT",
    "ORDER",
    "BY",
    "IN",
    "IS",
    "NULL",
    "NOT",
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "REFERENCES",
    "DEFAULT",
    "CHECK",
    "COLLATE",
    "UNIQUE",
    "EXISTS",
    "INTO",
    "VALUES",
    "SET",
    "ON",
}


def validate_identifier(name: str) -> None:
    """Validate a table, column, or template name.

    Args:
        name: Name to validate.

    Raises:
        ValueError: If name fails format, length, or keyword validation.
    """
    if not name:
        raise ValueError("Identifier cannot be empty")

    if len(name) > 64:
        raise ValueError(f"Identifier '{name}' exceeds maximum length of 64 characters")

    if not IDENTIFIER_REGEX.match(name):
        raise ValueError(
            f"Identifier '{name}' is invalid. "
            "Must start with a letter or underscore and "
            "contain only alphanumeric characters and underscores."
        )

    if name.upper() in RESERVED_KEYWORDS:
        raise ValueError(
            f"Identifier '{name}' is a reserved SQL keyword and cannot be used"
        )


def validate_column_type(col_type: str) -> None:
    """Validate that the column type is in the strict allowlist.

    Args:
        col_type: Type string to validate.

    Raises:
        ValueError: If the type is not allowed.
    """
    cleaned_type = col_type.strip().upper()
    if cleaned_type not in ALLOWED_TYPES:
        raise ValueError(
            f"Type '{col_type}' is not allowed. "
            f"Must be one of: {', '.join(sorted(ALLOWED_TYPES))}"
        )
