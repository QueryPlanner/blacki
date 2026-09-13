"""In-memory, user-bound browser takeover lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit

from opensandbox.models.execd import RunCommandOpts

from blacki.sandbox.manager import get_sandbox_manager

from .config import BrowserTakeoverConfig

_START_COMMAND = (
    'agent-browser open "$BLACKI_BROWSER_TAKEOVER_URL" '
    '&& agent-browser stream enable --port "$BLACKI_BROWSER_STREAM_PORT"'
)
_STOP_COMMAND = "agent-browser stream disable"


class BrowserTakeoverError(RuntimeError):
    """A secret-free browser takeover failure."""


@dataclass
class _TakeoverSession:
    session_id: str
    owner_key: str
    sandbox: Any = field(repr=False)
    upstream_url: str = field(repr=False)
    upstream_headers: dict[str, str] = field(repr=False)
    expires_at: float
    completed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    browser_token_digest: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class BrowserTakeoverLease:
    """Internal lease returned to the tool without exposing tokens to the model."""

    session_id: str
    takeover_url: str = field(repr=False)
    expires_in_seconds: int


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _websocket_url(endpoint: str) -> str:
    if "://" not in endpoint:
        return f"ws://{endpoint}"
    parsed = urlsplit(endpoint)
    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    return parsed._replace(scheme=scheme).geturl()


def _validated_login_url(login_url: str) -> str:
    parsed = urlsplit(login_url.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise BrowserTakeoverError("Browser takeover requires an HTTPS login URL")
    if parsed.username or parsed.password:
        raise BrowserTakeoverError("Login URLs cannot contain credentials")
    return parsed.geturl()


class BrowserTakeoverService:
    """Create one-time links and proxy only the matching browser stream."""

    def __init__(self, config: BrowserTakeoverConfig) -> None:
        config.validate()
        self.config = config
        self._sessions: dict[str, _TakeoverSession] = {}
        self._takeover_tokens: dict[str, str] = {}
        self._browser_tokens: dict[str, str] = {}
        self._owners: dict[str, str] = {}
        self._starting_owners: set[str] = set()
        self._lock = asyncio.Lock()

    async def create(self, *, login_url: str, state: Any) -> BrowserTakeoverLease:
        """Start streaming the session browser and mint a single-use link."""
        safe_url = _validated_login_url(login_url)
        owner_key = self._owner_key(state)
        async with self._lock:
            active_id = self._owners.get(owner_key)
            active = self._sessions.get(active_id or "")
            if active is not None and not self._expired(active):
                raise BrowserTakeoverError(
                    "A browser takeover is already active for this conversation"
                )
            if active is not None:
                self._remove_session_locked(active)
            if owner_key in self._starting_owners:
                raise BrowserTakeoverError(
                    "A browser takeover is already starting for this conversation"
                )
            self._starting_owners.add(owner_key)

        manager = get_sandbox_manager()
        try:
            result = await manager.get_or_create_sandbox(state)
            sandbox = result.get("sandbox")
            if sandbox is None:
                raise BrowserTakeoverError("The browser sandbox is unavailable")

            opts = RunCommandOpts(
                timeout=timedelta(seconds=60),
                envs={
                    "BLACKI_BROWSER_TAKEOVER_URL": safe_url,
                    "BLACKI_BROWSER_STREAM_PORT": str(self.config.stream_port),
                },
            )
            execution = await sandbox.commands.run(_START_COMMAND, opts=opts)
            if execution.error:
                raise BrowserTakeoverError("Agent Browser takeover could not start")
            endpoint = await sandbox.get_endpoint(self.config.stream_port)
        except BrowserTakeoverError:
            async with self._lock:
                self._starting_owners.discard(owner_key)
            raise
        except asyncio.CancelledError:
            async with self._lock:
                self._starting_owners.discard(owner_key)
            raise
        except Exception as exc:
            async with self._lock:
                self._starting_owners.discard(owner_key)
            raise BrowserTakeoverError(
                "Agent Browser takeover could not start"
            ) from exc

        session_id = secrets.token_urlsafe(24)
        takeover_token = secrets.token_urlsafe(32)
        session = _TakeoverSession(
            session_id=session_id,
            owner_key=owner_key,
            sandbox=sandbox,
            upstream_url=_websocket_url(endpoint.endpoint),
            upstream_headers=dict(endpoint.headers),
            expires_at=time.monotonic() + self.config.ttl_seconds,
        )
        async with self._lock:
            self._starting_owners.discard(owner_key)
            self._sessions[session_id] = session
            self._takeover_tokens[_digest(takeover_token)] = session_id
            self._owners[owner_key] = session_id

        return BrowserTakeoverLease(
            session_id=session_id,
            takeover_url=f"{self.config.public_url}#{takeover_token}",
            expires_in_seconds=self.config.ttl_seconds,
        )

    async def redeem(self, takeover_token: str) -> str | None:
        """Consume a link token and return a cookie token exactly once."""
        if not takeover_token or len(takeover_token) > 256:
            return None
        async with self._lock:
            session_id = self._takeover_tokens.pop(_digest(takeover_token), None)
            session = self._sessions.get(session_id or "")
            if session is None or self._expired(session):
                if session is not None:
                    self._remove_session_locked(session)
                return None
            browser_token = secrets.token_urlsafe(32)
            digest = _digest(browser_token)
            session.browser_token_digest = digest
            self._browser_tokens[digest] = session.session_id
            return browser_token

    async def authorize(self, browser_token: str | None) -> _TakeoverSession | None:
        """Resolve an active browser cookie without exposing its value."""
        if not browser_token:
            return None
        async with self._lock:
            session_id = self._browser_tokens.get(_digest(browser_token))
            session = self._sessions.get(session_id or "")
            if session is None or self._expired(session):
                if session is not None:
                    self._remove_session_locked(session)
                return None
            return session

    async def complete(self, browser_token: str | None) -> bool:
        """End human control and wake the waiting agent tool."""
        session = await self.authorize(browser_token)
        if session is None:
            return False
        session.completed.set()
        return True

    async def wait(self, lease: BrowserTakeoverLease) -> bool:
        """Wait for the user to return control or for the lease to expire."""
        async with self._lock:
            session = self._sessions.get(lease.session_id)
        if session is None:
            return False
        remaining = max(0.0, session.expires_at - time.monotonic())
        try:
            await asyncio.wait_for(session.completed.wait(), timeout=remaining)
            return True
        except TimeoutError:
            return False

    async def close(self, lease: BrowserTakeoverLease) -> None:
        """Disable streaming and erase every in-memory capability."""
        async with self._lock:
            session = self._sessions.get(lease.session_id)
            if session is not None:
                self._remove_session_locked(session)
        if session is None:
            return
        try:
            await session.sandbox.commands.run(
                _STOP_COMMAND,
                opts=RunCommandOpts(timeout=timedelta(seconds=10)),
            )
        except Exception:
            return

    async def close_all(self) -> None:
        """Invalidate all links during application shutdown."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._takeover_tokens.clear()
            self._browser_tokens.clear()
            self._owners.clear()
            self._starting_owners.clear()
        for session in sessions:
            session.completed.set()
        stopped: set[int] = set()
        for session in sessions:
            sandbox_key = id(session.sandbox)
            if sandbox_key in stopped:
                continue
            stopped.add(sandbox_key)
            try:
                await session.sandbox.commands.run(
                    _STOP_COMMAND,
                    opts=RunCommandOpts(timeout=timedelta(seconds=10)),
                )
            except Exception:  # noqa: S112 - shutdown stays silent and secret-free
                continue

    @staticmethod
    def _owner_key(state: Any) -> str:
        chat_type = state.get("telegram_chat_type")
        chat_id = state.get("telegram_chat_id")
        sender_id = state.get("temp:telegram_sender_user_id")
        if chat_type != "private" or not chat_id or not sender_id:
            raise BrowserTakeoverError(
                "Browser takeover is available only in an authenticated "
                "private Telegram chat"
            )
        if str(chat_id) != str(sender_id):
            raise BrowserTakeoverError("Telegram browser takeover identity mismatch")
        thread_id = state.get("telegram_thread_id", "")
        return f"{chat_id}:{thread_id}"

    @staticmethod
    def _expired(session: _TakeoverSession) -> bool:
        return time.monotonic() >= session.expires_at

    def _remove_session_locked(self, session: _TakeoverSession) -> None:
        self._sessions.pop(session.session_id, None)
        self._owners.pop(session.owner_key, None)
        stale_takeover = [
            token
            for token, session_id in self._takeover_tokens.items()
            if session_id == session.session_id
        ]
        for token in stale_takeover:
            self._takeover_tokens.pop(token, None)
        if session.browser_token_digest:
            self._browser_tokens.pop(session.browser_token_digest, None)
        session.completed.set()


_service: BrowserTakeoverService | None = None


def get_browser_takeover_service() -> BrowserTakeoverService | None:
    """Return the configured process-wide takeover service."""
    global _service
    if _service is not None:
        return _service
    config = BrowserTakeoverConfig.from_environment()
    if config is None:
        return None
    _service = BrowserTakeoverService(config)
    return _service


async def reset_browser_takeover_service() -> None:
    """Close and clear the process-wide service."""
    global _service
    service = _service
    _service = None
    if service is not None:
        await service.close_all()
