"""Private wiki deployment settings."""

import os


def wiki_configured() -> bool:
    """Enable memory unless the operator explicitly disables it."""
    return os.getenv("BLACKI_WIKI_ENABLED", "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
