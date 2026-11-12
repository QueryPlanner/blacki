"""Shared secret redaction for dashboard data.

Blacki keeps full model and tool content in local files.  The dashboard masks
common credential shapes before rendering private data in the browser.  The
same patterns are intentionally not applied to logs or trace files, which are
operator-only raw records.
"""

from __future__ import annotations

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
