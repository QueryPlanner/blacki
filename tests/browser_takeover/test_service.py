"""Tests for one-time browser takeover leases."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from blacki.browser_takeover.config import BrowserTakeoverConfig
from blacki.browser_takeover.service import (
    BrowserTakeoverError,
    BrowserTakeoverLease,
    BrowserTakeoverService,
    _websocket_url,
    get_browser_takeover_service,
    reset_browser_takeover_service,
)


class FakeCommands:
    def __init__(self, error: object | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, object]] = []

    async def run(self, command: str, *, opts: object) -> SimpleNamespace:
        self.calls.append((command, opts))
        return SimpleNamespace(error=self.error)


class FakeSandbox:
    def __init__(self, error: object | None = None) -> None:
        self.commands = FakeCommands(error)

    async def get_endpoint(self, port: int) -> SimpleNamespace:
        assert port == 9223
        return SimpleNamespace(
            endpoint="https://sandbox-proxy.test/route",
            headers={"X-Sandbox-Route": "private"},
        )


@pytest.fixture
def config() -> BrowserTakeoverConfig:
    return BrowserTakeoverConfig(
        "https://blacki.example.ts.net/browser-takeover",
        ttl_seconds=60,
    )


@pytest.fixture
def state() -> dict[str, str]:
    return {
        "telegram_chat_type": "private",
        "telegram_chat_id": "42",
        "temp:telegram_sender_user_id": "42",
    }


async def _create(
    service: BrowserTakeoverService,
    state: dict[str, str],
    sandbox: FakeSandbox | None = None,
) -> tuple[BrowserTakeoverLease, FakeSandbox]:
    sandbox = sandbox or FakeSandbox()
    manager = SimpleNamespace(
        get_or_create_sandbox=AsyncMock(
            return_value={"sandbox": sandbox, "error": None}
        )
    )
    with patch(
        "blacki.browser_takeover.service.get_sandbox_manager",
        return_value=manager,
    ):
        lease = await service.create(
            login_url="https://accounts.example.test/login",
            state=state,
        )
    return lease, sandbox


async def test_create_redeem_complete_wait_and_close(
    config: BrowserTakeoverConfig,
    state: dict[str, str],
) -> None:
    service = BrowserTakeoverService(config)
    lease, sandbox = await _create(service, state)
    link_token = lease.takeover_url.rsplit("#", 1)[1]

    command, opts = sandbox.commands.calls[0]
    assert "accounts.example.test" not in command
    assert opts.envs["BLACKI_BROWSER_TAKEOVER_URL"].endswith("/login")
    assert lease.takeover_url.startswith(config.public_url + "#")

    browser_token = await service.redeem(link_token)
    assert browser_token is not None
    assert await service.redeem("") is None
    assert await service.redeem(link_token) is None
    assert await service.authorize(None) is None
    session = await service.authorize(browser_token)
    assert session is not None
    assert session.upstream_url == "wss://sandbox-proxy.test/route"
    assert session.upstream_headers == {"X-Sandbox-Route": "private"}
    assert await service.complete(browser_token) is True
    assert await service.wait(lease) is True

    await service.close(lease)
    await service.close(lease)

    assert sandbox.commands.calls[-1][0] == "agent-browser stream disable"
    assert await service.authorize(browser_token) is None
    assert await service.complete(browser_token) is False
    assert await service.wait(lease) is False


async def test_wait_expires_and_close_all_wakes_waiter(
    config: BrowserTakeoverConfig,
    state: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = BrowserTakeoverService(config)
    lease, _ = await _create(service, state)
    monkeypatch.setattr("blacki.browser_takeover.service.time.monotonic", lambda: 10e9)

    assert await service.wait(lease) is False
    assert await service.redeem(lease.takeover_url.rsplit("#", 1)[1]) is None

    lease2, _ = await _create(service, state)
    waiter = asyncio.create_task(service.wait(lease2))
    await service.close_all()

    assert await waiter is True


@pytest.mark.parametrize(
    "bad_state",
    [
        {},
        {
            "telegram_chat_type": "group",
            "telegram_chat_id": "42",
            "temp:telegram_sender_user_id": "42",
        },
        {
            "telegram_chat_type": "private",
            "telegram_chat_id": "42",
            "temp:telegram_sender_user_id": "7",
        },
    ],
)
async def test_create_requires_matching_private_telegram_identity(
    config: BrowserTakeoverConfig,
    bad_state: dict[str, str],
) -> None:
    service = BrowserTakeoverService(config)

    with pytest.raises(BrowserTakeoverError):
        await service.create(login_url="https://example.test/login", state=bad_state)


@pytest.mark.parametrize(
    "url",
    ["http://example.test/login", "not-a-url", "https://user:pass@example.test"],
)
async def test_create_rejects_unsafe_login_url(
    config: BrowserTakeoverConfig,
    state: dict[str, str],
    url: str,
) -> None:
    service = BrowserTakeoverService(config)

    with pytest.raises(BrowserTakeoverError):
        await service.create(login_url=url, state=state)


async def test_create_handles_missing_or_failed_sandbox(
    config: BrowserTakeoverConfig,
    state: dict[str, str],
) -> None:
    service = BrowserTakeoverService(config)
    unavailable = SimpleNamespace(
        get_or_create_sandbox=AsyncMock(
            return_value={"sandbox": None, "error": "unavailable"}
        )
    )
    with (
        patch(
            "blacki.browser_takeover.service.get_sandbox_manager",
            return_value=unavailable,
        ),
        pytest.raises(BrowserTakeoverError, match="unavailable"),
    ):
        await service.create(login_url="https://example.test/login", state=state)

    with pytest.raises(BrowserTakeoverError, match="could not start"):
        await _create(service, state, FakeSandbox(error=object()))


async def test_cancelled_start_releases_owner_reservation(
    config: BrowserTakeoverConfig,
    state: dict[str, str],
) -> None:
    service = BrowserTakeoverService(config)
    cancelled = SimpleNamespace(
        get_or_create_sandbox=AsyncMock(side_effect=asyncio.CancelledError)
    )
    with (
        patch(
            "blacki.browser_takeover.service.get_sandbox_manager",
            return_value=cancelled,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await service.create(login_url="https://example.test/login", state=state)

    lease, _ = await _create(service, state)
    await service.close(lease)


async def test_only_one_takeover_can_run_per_conversation(
    config: BrowserTakeoverConfig,
    state: dict[str, str],
) -> None:
    service = BrowserTakeoverService(config)
    lease, _ = await _create(service, state)

    with pytest.raises(BrowserTakeoverError, match="already active"):
        await _create(service, state)

    await service.close(lease)


def test_websocket_endpoint_normalization() -> None:
    assert _websocket_url("sandbox:9223") == "ws://sandbox:9223"
    assert _websocket_url("http://sandbox/route") == "ws://sandbox/route"
    assert _websocket_url("wss://sandbox/route") == "wss://sandbox/route"


async def test_process_service_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    await reset_browser_takeover_service()
    monkeypatch.delenv("BROWSER_TAKEOVER_PUBLIC_URL", raising=False)
    assert get_browser_takeover_service() is None

    monkeypatch.setenv(
        "BROWSER_TAKEOVER_PUBLIC_URL",
        "https://blacki.example.ts.net/browser-takeover",
    )
    first = get_browser_takeover_service()
    assert first is not None
    assert get_browser_takeover_service() is first

    await reset_browser_takeover_service()
