"""Agent implementation public package interface."""

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    """Load the runtime application only when the public export is used."""
    if name == "app":
        from .agent import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
