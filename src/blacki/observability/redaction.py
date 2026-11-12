"""Shared secret redaction for local logs and dashboard data.

Blacki keeps full model and tool content in local files.  Credentials must not
be written to those files, so the same patterns are used by the logging filter,
the dashboard sanitizers, and the private capture reader.
"""

from __future__ import annotations

import logging
import re

BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
TELEGRAM_TOKEN_RE = re.compile(r"\b[0-9]{6,12}:[A-Za-z0-9_-]{20,}\b")
PROVIDER_SECRET_RE = re.compile(
    r"\b(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{35}|"
    r"ya29\.[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b"
)
KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"client[_-]?secret|password|passwd|secret|authorization|cookie|"
    r"telegram[_-]?bot[_-]?token|bot[_-]?token|api[_-]?token|token)\b\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
)


def redact_secrets(text: str) -> str:
    """Replace known credential shapes with a stable placeholder."""
    text = BEARER_RE.sub("Bearer [REDACTED]", text)
    text = TELEGRAM_TOKEN_RE.sub("[REDACTED]", text)
    text = PROVIDER_SECRET_RE.sub("[REDACTED]", text)
    return KEY_VALUE_SECRET_RE.sub(r"\1[REDACTED]", text)


class RedactionFilter(logging.Filter):
    """Redact a log record's rendered message before a handler writes it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive against bad log args
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True
