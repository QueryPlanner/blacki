"""Safety and validation layer for declarative database identifiers and types."""

from __future__ import annotations

import re
import unicodedata

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

MAX_USER_PREFERENCES_LENGTH = 1_000
MAX_USER_PREFERENCE_VALUE_LENGTH = 200
MAX_SCHEMA_METADATA_LENGTH = 500
ALLOWED_USER_PREFERENCE_KEYS = frozenset(
    {"language", "response_style", "tone", "units"}
)
DISALLOWED_PREFERENCE_PATTERNS = (
    re.compile(
        r"ignore\s+(?:all\s+)?(?:previous|system|developer).*instructions?", re.I
    ),
    re.compile(
        r"(?:bypass|disable|override).*\b(?:safety|privacy|permissions?)\b", re.I
    ),
    re.compile(r"\b(?:system|developer)\s+(?:message|prompt|instructions?)\b", re.I),
    re.compile(r"\b(?:call|use|enable|disable)\b.*\btools?\b", re.I),
)


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


def parse_user_preferences(preferences: str) -> dict[str, str]:
    """Parse allow-listed ``key: value`` style preferences.

    The values are data, not free-form instructions. Rejecting unknown keys and
    instruction-like content prevents stored preferences from changing safety
    policy or tool permissions.
    """
    normalized = unicodedata.normalize("NFKC", preferences).strip()
    if not normalized:
        raise ValueError("Preferences cannot be empty")
    if len(normalized) > MAX_USER_PREFERENCES_LENGTH:
        raise ValueError(
            f"Preferences exceed the {MAX_USER_PREFERENCES_LENGTH}-character limit"
        )
    if any(ord(char) < 32 and char not in "\n\t" for char in normalized):
        raise ValueError("Preferences contain unsupported control characters")
    if any(pattern.search(normalized) for pattern in DISALLOWED_PREFERENCE_PATTERNS):
        raise ValueError("Preferences cannot change instructions or tool permissions")

    parsed: dict[str, str] = {}
    for line in normalized.splitlines():
        if not line.strip():
            continue
        key_text, separator, value_text = line.partition(":")
        if not separator:
            raise ValueError("Each preference must use the format 'key: value'")
        key = key_text.strip().lower().replace(" ", "_")
        value = " ".join(value_text.split())
        if key not in ALLOWED_USER_PREFERENCE_KEYS:
            allowed = ", ".join(sorted(ALLOWED_USER_PREFERENCE_KEYS))
            raise ValueError(f"Preference key '{key}' is not allowed; use: {allowed}")
        if not value:
            raise ValueError(f"Preference '{key}' needs a value")
        if len(value) > MAX_USER_PREFERENCE_VALUE_LENGTH:
            raise ValueError(
                f"Preference '{key}' exceeds the "
                f"{MAX_USER_PREFERENCE_VALUE_LENGTH}-character limit"
            )
        if key in parsed:
            raise ValueError(f"Preference '{key}' is duplicated")
        parsed[key] = value
    return parsed


def sanitize_schema_metadata(value: object) -> str:
    """Normalize and length-bound user-controlled schema display metadata."""
    normalized = unicodedata.normalize("NFKC", str(value))
    printable = "".join(
        char for char in normalized if ord(char) >= 32 or char in "\n\t"
    )
    return printable[:MAX_SCHEMA_METADATA_LENGTH]
