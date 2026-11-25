"""Server-verified private Telegram identity for reminder creation."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_private_conversation: ContextVar[str | None] = ContextVar(
    "reminder_private_conversation", default=None
)


@contextmanager
def reminder_turn(private_conversation: str | None) -> Iterator[None]:
    token = _private_conversation.set(private_conversation)
    try:
        yield
    finally:
        _private_conversation.reset(token)


def is_private_reminder_owner(user_id: str) -> bool:
    return _private_conversation.get() == user_id
