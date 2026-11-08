"""Telegram Mini App sign-in: initData verification and short-lived sessions.

Follows Telegram's documented check: the secret key is
HMAC-SHA256(key="WebAppData", msg=bot_token), and the hash is
HMAC-SHA256(secret_key, data_check_string) over every field except ``hash``,
sorted by key and joined with newlines.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl

MAX_SESSIONS = 200


class InitDataError(ValueError):
    """The launch data is missing, forged, expired or malformed."""


def verify_init_data(init_data: str, bot_token: str, *, max_age_seconds: int) -> int:
    """Return the verified Telegram user id from Mini App launch data."""
    if not init_data or len(init_data) > 4096:
        raise InitDataError("missing launch data")
    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise InitDataError("malformed launch data") from exc
    fields: dict[str, str] = {}
    for key, value in pairs:
        if key in fields:
            raise InitDataError("duplicate launch data field")
        fields[key] = value
    received = fields.pop("hash", "")
    if not received:
        raise InitDataError("launch data is not signed")
    check_string = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, received):
        raise InitDataError("launch data signature does not match")
    try:
        auth_date = int(fields.get("auth_date", ""))
    except ValueError as exc:
        raise InitDataError("launch data has no date") from exc
    now = int(time.time())
    if auth_date > now + 60 or now - auth_date > max_age_seconds:
        raise InitDataError("launch data expired; reopen the app")
    try:
        user = json.loads(fields.get("user", ""))
        user_id = int(user["id"])
    except (ValueError, KeyError, TypeError) as exc:
        raise InitDataError("launch data has no user") from exc
    if user_id <= 0:
        raise InitDataError("launch data has no user")
    return user_id


@dataclass
class _Session:
    user_id: int
    expires_at: float


class SessionStore:
    """In-memory bearer sessions. Single-process only; tokens are stored hashed."""

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, _Session] = {}

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def issue(self, user_id: int) -> str:
        now = time.monotonic()
        self._sessions = {k: s for k, s in self._sessions.items() if s.expires_at > now}
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].expires_at)
            del self._sessions[oldest]
        token = secrets.token_urlsafe(32)
        self._sessions[self._key(token)] = _Session(user_id, now + self.ttl_seconds)
        return token

    def resolve(self, token: str) -> int | None:
        if not token or len(token) > 128:
            return None
        session = self._sessions.get(self._key(token))
        if session is None or session.expires_at <= time.monotonic():
            return None
        return session.user_id
