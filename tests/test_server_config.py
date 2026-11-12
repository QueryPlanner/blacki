# mypy: disable-error-code="no-untyped-def"
"""Tests for server configuration."""

import asyncio
import contextlib
import json
import socket
import sys
from collections.abc import AsyncIterator, Generator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import aiosqlite
import pytest
from fastapi import FastAPI


@pytest.fixture
def mock_dependencies() -> Generator[MagicMock]:
    """Mock external dependencies to prevent side effects during import."""
    with (
        patch("google.adk.cli.fast_api.get_fast_api_app") as mock_get_app,
        patch("blacki.utils.initialize_environment") as mock_init_env,
        patch("blacki.observability.setup.configure_otel_resource"),
        patch("blacki.observability.setup.setup_logging"),
        patch("blacki.observability.setup.setup_tracing"),
    ):
        mock_env = MagicMock()
        mock_env.session_uri = None
        mock_env.allow_origins_list = ["*"]
        mock_env.serve_web_interface = True
        mock_env.reload_agents = False
        mock_env.sqlite_path = None
        mock_env.telegram_access_code = None
        mock_env.agent_dir = "src"

        mock_env.host = "127.0.0.1"
        mock_env.port = 8080

        mock_init_env.return_value = mock_env
        mock_get_app.return_value = FastAPI()

        yield mock_get_app


def test_server_session_service_uri_is_none(mock_dependencies: MagicMock) -> None:
    """Verify session_service_uri is None for default SQLite sessions."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    mock_dependencies.assert_called_once()
    call_kwargs = mock_dependencies.call_args[1]

    assert call_kwargs["session_service_uri"] is None
    assert call_kwargs["lifespan"] is server.lifespan
    assert server.env.agent_dir == server.AGENT_DIR


def test_server_configures_observability_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resource, logging, and tracing setup run in that order at startup."""
    sys.modules.pop("blacki.server", None)
    calls: list[str] = []
    mock_env = MagicMock()
    mock_env.agent_name = "order-test"
    mock_env.log_level = "DEBUG"
    mock_env.allow_origins_list = ["*"]
    mock_env.serve_web_interface = False
    mock_env.reload_agents = False
    mock_env.sqlite_path = None

    def record_resource(*, agent_name: str) -> None:
        _ = agent_name
        calls.append("resource")

    def record_logging(*, log_level: str) -> None:
        _ = log_level
        calls.append("logging")

    def record_tracing() -> None:
        calls.append("tracing")

    with (
        patch("google.adk.cli.fast_api.get_fast_api_app", return_value=FastAPI()),
        patch("blacki.utils.initialize_environment", return_value=mock_env),
        patch(
            "blacki.observability.setup.configure_otel_resource",
            side_effect=record_resource,
        ),
        patch(
            "blacki.observability.setup.setup_logging",
            side_effect=record_logging,
        ),
        patch(
            "blacki.observability.setup.setup_tracing",
            side_effect=record_tracing,
        ),
    ):
        import blacki.server as server

    assert calls == ["resource", "logging", "tracing"]
    monkeypatch.setitem(sys.modules, "blacki.server", server)


def test_server_always_mounts_dashboard(
    mock_dependencies: MagicMock,
) -> None:
    """The private dashboard is mounted without a feature flag."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    assert _has_route(server.app, "/dashboard")


def _has_route(app: FastAPI, path: str) -> bool:
    """Find a path in FastAPI's direct or included router entries."""
    pending = list(app.routes)
    while pending:
        route = pending.pop()
        if getattr(route, "path", None) == path:
            return True
        pending.extend(getattr(route, "routes", ()))
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            pending.extend(getattr(original_router, "routes", ()))
    return False


@pytest.mark.asyncio
async def test_server_lifespan_closes_search_clients(
    mock_dependencies: MagicMock,
) -> None:
    """Verify both managed search clients are closed during shutdown."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    container = MagicMock()
    container.initialize_all_storages = AsyncMock()
    init_container = AsyncMock(return_value=container)
    close_container = AsyncMock()
    close_brave = AsyncMock()
    close_exa = AsyncMock()
    close_notify = AsyncMock()
    tracer_provider = MagicMock()
    log_warning = MagicMock()

    with (
        patch.object(server, "init_container", new=init_container),
        patch.object(server, "close_container", new=close_container),
        patch.object(server, "_start_telegram_bot", new=AsyncMock()),
        # The Mini App is always on; these tests check shutdown, not its port.
        patch.object(server, "_start_miniapp", new=AsyncMock()),
        patch.object(server, "_stop_telegram_bot", new=AsyncMock()),
        patch.object(server, "_stop_reminder_scheduler", new=AsyncMock()),
        patch.object(
            server.validation,
            "validate_configuration",
            return_value=["test warning"],
        ),
        patch.object(server.logger, "warning", new=log_warning),
        patch(
            "blacki.tools.brave_search.close_shared_brave_search_client",
            new=close_brave,
        ),
        patch("blacki.tools.search.close_shared_exa_search_client", new=close_exa),
        patch(
            "blacki.telegram.progress_callbacks.close_shared_notify_client",
            new=close_notify,
        ),
        patch.object(server, "_tracer_provider", tracer_provider),
        patch.object(server, "shutdown_tracing") as shutdown_tracing,
    ):
        async with server.lifespan(server.app):
            pass

    container.initialize_all_storages.assert_awaited_once()
    close_container.assert_awaited_once()
    close_brave.assert_awaited_once()
    close_exa.assert_awaited_once()
    close_notify.assert_awaited_once()
    shutdown_tracing.assert_called_once_with(tracer_provider)
    log_warning.assert_called_once_with("test warning")


@pytest.mark.asyncio
async def test_lifespan_cleans_up_after_validation_failure(
    mock_dependencies: MagicMock,
) -> None:
    """A failed startup should still release every initialized resource."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    container = MagicMock()
    container.initialize_all_storages = AsyncMock()
    close_container = AsyncMock()
    stop_bot = AsyncMock()
    stop_scheduler = AsyncMock()
    close_brave = AsyncMock()
    close_exa = AsyncMock()
    close_notify = AsyncMock()

    with (
        patch.object(
            server,
            "init_container",
            new=AsyncMock(return_value=container),
        ),
        patch.object(server, "close_container", new=close_container),
        patch.object(server, "_stop_telegram_bot", new=stop_bot),
        patch.object(server, "_stop_reminder_scheduler", new=stop_scheduler),
        patch.object(
            server.validation,
            "validate_configuration",
            side_effect=server.ConfigurationError("invalid"),
        ),
        patch(
            "blacki.tools.brave_search.close_shared_brave_search_client",
            new=close_brave,
        ),
        patch("blacki.tools.search.close_shared_exa_search_client", new=close_exa),
        patch(
            "blacki.telegram.progress_callbacks.close_shared_notify_client",
            new=close_notify,
        ),
        pytest.raises(server.ConfigurationError, match="invalid"),
    ):
        async with server.lifespan(server.app):
            pass

    stop_scheduler.assert_awaited_once()
    stop_bot.assert_awaited_once()
    close_container.assert_awaited_once()
    close_brave.assert_awaited_once()
    close_exa.assert_awaited_once()
    close_notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_tolerates_container_closed_during_runtime(
    mock_dependencies: MagicMock,
) -> None:
    """Cleanup should skip duplicate container closure but close shared clients."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    container = MagicMock()
    container.initialize_all_storages = AsyncMock()
    close_container = AsyncMock()
    close_brave = AsyncMock()
    close_exa = AsyncMock()
    close_notify = AsyncMock()

    with (
        patch.object(
            server,
            "init_container",
            new=AsyncMock(return_value=container),
        ),
        patch.object(server, "close_container", new=close_container),
        patch.object(server, "_start_telegram_bot", new=AsyncMock()),
        # The Mini App is always on; these tests check shutdown, not its port.
        patch.object(server, "_start_miniapp", new=AsyncMock()),
        patch.object(server, "_stop_telegram_bot", new=AsyncMock()),
        patch.object(server, "_stop_reminder_scheduler", new=AsyncMock()),
        patch.object(server.validation, "validate_configuration", return_value=[]),
        patch(
            "blacki.tools.brave_search.close_shared_brave_search_client",
            new=close_brave,
        ),
        patch("blacki.tools.search.close_shared_exa_search_client", new=close_exa),
        patch(
            "blacki.telegram.progress_callbacks.close_shared_notify_client",
            new=close_notify,
        ),
    ):
        async with server.lifespan(server.app):
            server._container = None

    close_container.assert_not_awaited()
    close_brave.assert_awaited_once()
    close_exa.assert_awaited_once()
    close_notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_telegram_bot_skips_when_unconfigured(
    mock_dependencies: MagicMock,
) -> None:
    """Telegram startup should be a no-op without complete configuration."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    cast(Any, server.env).is_telegram_configured = False

    await server._start_telegram_bot()

    assert server._telegram_bot is None


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduler_fails", [False, True])
async def test_start_telegram_bot_starts_polling_and_tolerates_scheduler_failure(
    mock_dependencies: MagicMock,
    scheduler_fails: bool,
) -> None:
    """Telegram polling stays available if the optional scheduler cannot start."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server.env.telegram_enabled = True
    cast(Any, server.env).is_telegram_configured = True
    server.env.telegram_bot_token = "test-token"  # noqa: S105 - inert fixture value
    bot = MagicMock()
    bot.start_polling = AsyncMock()
    scheduler_error = RuntimeError("scheduler failed") if scheduler_fails else None
    start_scheduler = AsyncMock(side_effect=scheduler_error)
    telegram_agent = MagicMock()
    telegram_app = MagicMock()

    with (
        patch.object(
            server,
            "create_adk_runtime",
            return_value=MagicMock(),
        ) as create_runtime,
        patch(
            "blacki.agent.create_agent",
            return_value=telegram_agent,
        ) as create_agent,
        patch(
            "blacki.agent.create_app",
            return_value=telegram_app,
        ) as create_app,
        patch("blacki.telegram.bot.TelegramBot", return_value=bot),
        patch.object(
            server,
            "_start_reminder_scheduler",
            new=start_scheduler,
        ),
    ):
        await server._start_telegram_bot()

    assert server._telegram_bot is bot
    create_agent.assert_called_once_with(include_user_scoped_tools=True)
    create_app.assert_called_once_with(telegram_agent)
    create_runtime.assert_called_once_with(server.env, agent_app=telegram_app)
    bot.start_polling.assert_awaited_once()
    start_scheduler.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_telegram_bot_propagates_polling_failure(
    mock_dependencies: MagicMock,
) -> None:
    """A polling failure must fail application startup visibly."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server.env.telegram_enabled = True
    cast(Any, server.env).is_telegram_configured = True
    server.env.telegram_bot_token = "test-token"  # noqa: S105 - inert fixture value
    bot = MagicMock()
    bot.start_polling = AsyncMock(side_effect=RuntimeError("polling failed"))

    with (
        patch.object(server, "create_adk_runtime", return_value=MagicMock()),
        patch("blacki.telegram.bot.TelegramBot", return_value=bot),
        pytest.raises(RuntimeError, match="polling failed"),
    ):
        await server._start_telegram_bot()


@pytest.mark.asyncio
async def test_start_reminder_scheduler_handles_container_and_callback_states(
    mock_dependencies: MagicMock,
) -> None:
    """Scheduler startup should require storage and attach Telegram when present."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server._container = None
    await server._start_reminder_scheduler()

    scheduler = MagicMock()
    scheduler.start = AsyncMock()
    server._container = MagicMock()
    server._telegram_bot = None
    with patch("blacki.reminders.get_scheduler", return_value=scheduler):
        await server._start_reminder_scheduler()
    scheduler.set_callback.assert_not_called()

    telegram_bot = MagicMock()
    server._telegram_bot = telegram_bot
    with patch("blacki.reminders.get_scheduler", return_value=scheduler):
        await server._start_reminder_scheduler()

    scheduler.set_callback.assert_called_once_with(
        telegram_bot.handle_scheduled_reminder
    )
    assert scheduler.start.await_count == 2


@pytest.mark.asyncio
async def test_stop_telegram_bot_handles_success_and_failure(
    mock_dependencies: MagicMock,
) -> None:
    """Telegram shutdown failures should not prevent remaining cleanup."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server._telegram_bot = None
    await server._stop_telegram_bot()

    bot = MagicMock()
    bot.stop = AsyncMock(side_effect=[None, RuntimeError("stop failed")])
    server._telegram_bot = bot
    await server._stop_telegram_bot()
    await server._stop_telegram_bot()

    assert bot.stop.await_count == 2


@pytest.mark.asyncio
async def test_stop_reminder_scheduler_handles_all_states(
    mock_dependencies: MagicMock,
) -> None:
    """Scheduler cleanup should tolerate idle, absent, and failed schedulers."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    idle_scheduler = MagicMock()
    idle_scheduler._running = False
    idle_scheduler.stop = AsyncMock()
    running_scheduler = MagicMock()
    running_scheduler._running = True
    running_scheduler.stop = AsyncMock()

    with patch(
        "blacki.reminders.get_scheduler",
        side_effect=[
            idle_scheduler,
            running_scheduler,
            RuntimeError("not initialized"),
            ValueError("stop failed"),
        ],
    ):
        await server._stop_reminder_scheduler()
        await server._stop_reminder_scheduler()
        await server._stop_reminder_scheduler()
        await server._stop_reminder_scheduler()

    idle_scheduler.stop.assert_not_awaited()
    running_scheduler.stop.assert_awaited_once()


def test_main_runs_uvicorn_with_configured_address(
    mock_dependencies: MagicMock,
) -> None:
    """The command entry point should forward the validated host and port."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    with patch.object(server.uvicorn, "run") as run:
        server.main()

    run.assert_called_once_with(
        server.app,
        host="127.0.0.1",
        port=8080,
    )


@pytest.mark.asyncio
async def test_liveness_is_process_only(mock_dependencies: MagicMock) -> None:
    """Liveness must not inspect or initialize external dependencies."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    assert await server.live() == {"status": "alive"}


@pytest.mark.asyncio
async def test_readiness_reports_startup_and_health_alias(
    mock_dependencies: MagicMock,
) -> None:
    """Readiness and its compatibility alias return 503 before startup."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server._container = None
    ready_response = await server.ready()
    health_response = await server.health()

    assert ready_response.status_code == 503
    assert health_response.status_code == ready_response.status_code
    assert json.loads(ready_response.body) == {
        "status": "starting",
        "checks": {"database": "starting"},
    }
    assert health_response.body == ready_response.body


@pytest.mark.asyncio
async def test_readiness_reports_healthy_database(
    mock_dependencies: MagicMock,
) -> None:
    """Readiness returns 200 after the initialized database answers."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    cursor = MagicMock()
    cursor.fetchone = AsyncMock(return_value=(1,))
    execute_context = MagicMock()
    execute_context.__aenter__ = AsyncMock(return_value=cursor)
    execute_context.__aexit__ = AsyncMock(return_value=None)
    container = MagicMock()
    container.conn.execute.return_value = execute_context
    server._container = container

    response = await server.ready()

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "status": "ready",
        "checks": {"database": "healthy"},
    }
    container.conn.execute.assert_called_once_with("SELECT 1")
    cursor.fetchone.assert_awaited_once()


@pytest.mark.asyncio
async def test_readiness_reports_degraded_database(
    mock_dependencies: MagicMock,
) -> None:
    """Readiness returns 503 without leaking database exception details."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    execute_context = MagicMock()
    execute_context.__aenter__ = AsyncMock(side_effect=RuntimeError("db secret"))
    container = MagicMock()
    container.conn.execute.return_value = execute_context
    server._container = container

    response = await server.ready()

    assert response.status_code == 503
    assert json.loads(response.body) == {
        "status": "degraded",
        "checks": {"database": "unhealthy"},
    }
    assert b"db secret" not in response.body


@pytest.mark.asyncio
async def test_google_health_callback_requires_configuration(
    mock_dependencies: MagicMock,
) -> None:
    """The callback fails closed when the optional connector is disabled."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server._google_health_service = None
    response = await server.google_health_callback(state="state", code="code")

    assert response.status_code == 503
    assert b"not configured" in response.body


@pytest.mark.asyncio
async def test_google_health_callback_completes_and_notifies(
    mock_dependencies: MagicMock,
) -> None:
    """The callback consumes OAuth through the service and notifies Telegram."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server
    from blacki.health.service import OAuthCompletion

    service = MagicMock()
    service.complete_authorization = AsyncMock(
        return_value=OAuthCompletion("telegram-chat-42", connected=True)
    )
    bot = MagicMock()
    bot.notify_health_connection = AsyncMock()
    server._google_health_service = service
    server._telegram_bot = bot
    try:
        response = await server.google_health_callback(
            state="state", code="code", error=None
        )
    finally:
        server._google_health_service = None
        server._telegram_bot = None

    assert response.status_code == 200
    assert b"connected" in response.body
    service.complete_authorization.assert_awaited_once_with(
        state="state", code="code", error=None
    )
    bot.notify_health_connection.assert_awaited_once_with(
        "telegram-chat-42", connected=True
    )


@pytest.mark.asyncio
async def test_google_health_callback_handles_cancel_and_safe_errors(
    mock_dependencies: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cancellation and provider/state errors never expose payloads."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server
    from blacki.health.client import GoogleHealthApiError
    from blacki.health.service import GoogleHealthOAuthError, OAuthCompletion

    service = MagicMock()
    server._google_health_service = service
    try:
        service.complete_authorization = AsyncMock(
            return_value=OAuthCompletion("telegram-chat-42", connected=False)
        )
        with patch.object(server, "_schedule_google_health_backfill") as schedule:
            response = await server.google_health_callback(
                state="state", code=None, error="access_denied"
            )
        assert response.status_code == 200
        assert b"cancelled" in response.body
        schedule.assert_not_called()

        service.complete_authorization = AsyncMock(
            side_effect=GoogleHealthOAuthError("state secret")
        )
        response = await server.google_health_callback(state="bad", code="code")
        assert response.status_code == 400
        assert b"state secret" not in response.body

        service.complete_authorization = AsyncMock(
            side_effect=GoogleHealthApiError(
                "provider error", status_code=400, error_code="ACCOUNT_NOT_LINKED"
            )
        )
        with caplog.at_level("ERROR", logger="blacki.server"):
            response = await server.google_health_callback(state="bad", code="code")
        assert response.status_code == 502
        assert b"provider error" not in response.body
        assert "status_code=400 error_code=ACCOUNT_NOT_LINKED" in caplog.text

        service.complete_authorization = AsyncMock(
            side_effect=RuntimeError("unexpected secret")
        )
        response = await server.google_health_callback(state="bad", code="code")
        assert response.status_code == 500
        assert b"unexpected secret" not in response.body

        from blacki.health.service import OAuthCompletion

        service.complete_authorization = AsyncMock(
            return_value=OAuthCompletion("telegram-chat-42", connected=True)
        )
        bot = MagicMock()
        bot.notify_health_connection = AsyncMock(side_effect=RuntimeError("notify"))
        server._telegram_bot = bot
        with patch.object(server, "_schedule_google_health_backfill") as schedule:
            response = await server.google_health_callback(state="state", code="code")
        assert response.status_code == 200
        schedule.assert_called_once_with("telegram-chat-42")
    finally:
        server._google_health_service = None
        server._telegram_bot = None


@pytest.mark.asyncio
async def test_gmail_callback_requires_started_connector(
    mock_dependencies: MagicMock,
) -> None:
    """The callback does not create a second unscoped connector."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server._gmail_oauth_service = None
    response = await server.gmail_callback(state="state", code="code")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_gmail_start_and_stop_are_optional_and_safe(
    mock_dependencies: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Gmail startup and shutdown must tolerate optional configuration failures."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    from cryptography.fernet import Fernet

    import blacki.server as server
    from blacki.gmail import GmailConfig, GmailConfigurationError

    server._gmail_oauth_service = None
    server._container = None
    await server._start_gmail()

    server._container = MagicMock()
    with patch(
        "blacki.gmail.GmailConfig.from_environment",
        return_value=None,
    ):
        await server._start_gmail()
    with patch(
        "blacki.gmail.GmailConfig.from_environment",
        side_effect=GmailConfigurationError("configuration secret"),
    ):
        await server._start_gmail()
    assert server.__dict__["_gmail_oauth_service"] is None

    config = GmailConfig(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.test/integrations/gmail/callback",
        token_encryption_key=Fernet.generate_key().decode(),
    )
    service = MagicMock()
    service.close = AsyncMock()
    with (
        patch("blacki.gmail.GmailConfig.from_environment", return_value=config),
        patch("blacki.gmail.GmailOAuthService", return_value=service) as constructor,
    ):
        await server._start_gmail()
    constructor.assert_called_once_with(config, server._container.gmail_storage)
    assert server._gmail_oauth_service is not None
    await server._stop_gmail()
    service.close.assert_awaited_once()
    assert server._gmail_oauth_service is None

    failing_service = MagicMock()
    failing_service.close = AsyncMock(side_effect=RuntimeError("close secret"))
    server._gmail_oauth_service = failing_service
    with caplog.at_level("ERROR", logger="blacki.server"):
        await server._stop_gmail()
    assert server._gmail_oauth_service is None
    assert "close secret" not in caplog.text
    await server._stop_gmail()


@pytest.mark.asyncio
async def test_gmail_callback_validates_state_and_errors(
    mock_dependencies: MagicMock,
) -> None:
    """Callback failures return safe pages without provider details."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server
    from blacki.gmail import GmailApiError, GmailCredentialError, GmailOAuthError

    service = MagicMock()
    service.complete_authorization = AsyncMock(side_effect=GmailOAuthError("expired"))
    server._gmail_oauth_service = service
    try:
        response = await server.gmail_callback(state=None, code="code")
        assert response.status_code == 400

        response = await server.gmail_callback(state="bad-state", code="code")
        assert response.status_code == 400
        assert b"expired" not in response.body

        service.complete_authorization = AsyncMock(
            side_effect=GmailCredentialError("credential secret")
        )
        response = await server.gmail_callback(state="state", code="code")
        assert response.status_code == 400
        assert b"credential secret" not in response.body

        service.complete_authorization = AsyncMock(
            side_effect=GmailApiError("provider", status_code=503)
        )
        response = await server.gmail_callback(state="state", code="code")
        assert response.status_code == 502

        service.complete_authorization = AsyncMock(
            side_effect=RuntimeError("unexpected disk error")
        )
        response = await server.gmail_callback(state="state", code="code")
        assert response.status_code == 500

        from blacki.gmail import GmailOAuthCompletion

        service.complete_authorization = AsyncMock(
            return_value=GmailOAuthCompletion("telegram-chat-42", connected=True)
        )
        server._telegram_bot = None
        response = await server.gmail_callback(state="state", code="code")
        assert response.status_code == 200
    finally:
        server._gmail_oauth_service = None


@pytest.mark.asyncio
async def test_gmail_callback_notifies_owning_telegram_user(
    mock_dependencies: MagicMock,
) -> None:
    """Successful and cancelled callbacks notify only the bound user."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server
    from blacki.gmail import GmailOAuthCompletion

    service = MagicMock()
    service.complete_authorization = AsyncMock(
        return_value=GmailOAuthCompletion("telegram-chat-42", connected=True)
    )
    bot = MagicMock()
    bot.notify_gmail_connected = AsyncMock()
    server._gmail_oauth_service = service
    server._telegram_bot = bot
    try:
        response = await server.gmail_callback(state="state", code="code")
        assert response.status_code == 200
        assert b"Gmail connected" in response.body
        bot.notify_gmail_connected.assert_awaited_once_with(
            "telegram-chat-42", connected=True
        )

        service.complete_authorization = AsyncMock(
            return_value=GmailOAuthCompletion("telegram-chat-42", connected=False)
        )
        bot.notify_gmail_connected = AsyncMock(
            side_effect=RuntimeError("telegram fail")
        )
        response = await server.gmail_callback(
            state="state",
            code=None,
            error="access_denied",
        )
        assert response.status_code == 200
        assert b"cancelled" in response.body
    finally:
        server._gmail_oauth_service = None
        server._telegram_bot = None


@pytest.mark.asyncio
async def test_google_health_start_and_stop_are_optional(
    mock_dependencies: MagicMock,
) -> None:
    """Startup tolerates absent, invalid, failing, and valid optional config."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    from cryptography.fernet import Fernet

    import blacki.server as server
    from blacki.health.config import GoogleHealthConfigurationError
    from blacki.health.scheduler import GoogleHealthScheduler

    server._container = None
    await server._start_google_health()

    server._container = MagicMock()
    with patch(
        "blacki.health.config.GoogleHealthConfig.from_environment",
        return_value=None,
    ):
        await server._start_google_health()
    assert server._google_health_service is None

    with patch(
        "blacki.health.config.GoogleHealthConfig.from_environment",
        side_effect=GoogleHealthConfigurationError("bad config"),
    ):
        await server._start_google_health()
    assert server._google_health_service is None

    config_values = {
        "client_id": "id",
        "client_secret": "secret",
        "redirect_uri": "https://example.test/callback",
        "token_encryption_key": Fernet.generate_key().decode(),
    }
    with (
        patch(
            "blacki.health.config.GoogleHealthConfig.from_environment",
            return_value=__import__(
                "blacki.health.config", fromlist=["GoogleHealthConfig"]
            ).GoogleHealthConfig(**config_values),
        ),
        patch.object(GoogleHealthScheduler, "start", new=AsyncMock()),
    ):
        await server._start_google_health()
    assert getattr(server, "_google_health_service", None) is not None
    assert getattr(server, "_google_health_scheduler", None) is not None
    await server._stop_google_health()
    assert getattr(server, "_google_health_service", object()) is None

    with (
        patch(
            "blacki.health.config.GoogleHealthConfig.from_environment",
            return_value=__import__(
                "blacki.health.config", fromlist=["GoogleHealthConfig"]
            ).GoogleHealthConfig(**config_values),
        ),
        patch.object(
            GoogleHealthScheduler,
            "start",
            new=AsyncMock(side_effect=RuntimeError("scheduler")),
        ),
    ):
        await server._start_google_health()
    assert server._google_health_service is None
    server._container = None


@pytest.mark.asyncio
async def test_google_health_backfill_schedule_runs_global_and_user_tasks(
    mock_dependencies: MagicMock,
) -> None:
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server
    from blacki.container import AppContainer

    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    container = AppContainer(conn=conn)
    await container.initialize_all_storages()
    worker = MagicMock()
    server._container = container
    server._google_health_export_worker = worker
    run_all = AsyncMock(return_value=[])
    run_user = AsyncMock(return_value=None)
    try:
        with (
            patch(
                "blacki.health.nutrition_backfill.NutritionBackfillCoordinator.run_all_eligible",
                new=run_all,
            ),
            patch(
                "blacki.health.nutrition_backfill.NutritionBackfillCoordinator.run_user",
                new=run_user,
            ),
        ):
            server._schedule_google_health_backfill()
            server._schedule_google_health_backfill("telegram-chat-42")
            tasks = list(server._google_health_backfill_tasks)
            await asyncio.gather(*tasks)

        run_all.assert_awaited_once()
        run_user.assert_awaited_once_with("telegram-chat-42")
        server._google_health_export_worker = None
        server._schedule_google_health_backfill()
        assert not server._google_health_backfill_tasks
    finally:
        server._google_health_backfill_tasks.clear()
        server._container = None
        server._google_health_export_worker = None
        await container.close()


@pytest.mark.asyncio
async def test_google_health_backfill_task_swallows_unexpected_errors(
    mock_dependencies: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server
    from blacki.container import AppContainer

    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    container = AppContainer(conn=conn)
    await container.initialize_all_storages()
    server._container = container
    server._google_health_export_worker = MagicMock()
    try:
        with (
            patch(
                "blacki.health.nutrition_backfill.NutritionBackfillCoordinator.run_all_eligible",
                new=AsyncMock(side_effect=RuntimeError("backfill failure")),
            ),
            caplog.at_level("ERROR", logger="blacki.server"),
        ):
            server._schedule_google_health_backfill()
            await asyncio.gather(*list(server._google_health_backfill_tasks))
        assert "Google Health nutrition backfill task failed" in caplog.text
    finally:
        server._google_health_backfill_tasks.clear()
        server._container = None
        server._google_health_export_worker = None
        await container.close()


@pytest.mark.asyncio
async def test_google_health_backfill_task_preserves_cancellation(
    mock_dependencies: MagicMock,
) -> None:
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server
    from blacki.container import AppContainer

    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    container = AppContainer(conn=conn)
    await container.initialize_all_storages()
    server._container = container
    server._google_health_export_worker = MagicMock()
    try:
        with patch(
            "blacki.health.nutrition_backfill.NutritionBackfillCoordinator.run_all_eligible",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            server._schedule_google_health_backfill()
            task = next(iter(server._google_health_backfill_tasks))
            await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
    finally:
        server._google_health_backfill_tasks.clear()
        server._container = None
        server._google_health_export_worker = None
        await container.close()


@pytest.mark.asyncio
async def test_google_health_stop_cancels_backfill_tasks(
    mock_dependencies: MagicMock,
) -> None:
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    task = asyncio.create_task(asyncio.sleep(60))
    server._google_health_backfill_tasks.add(task)
    server._container = None
    await server._stop_google_health()

    assert task.cancelled()


@pytest.mark.asyncio
async def test_google_health_stop_suppresses_scheduler_and_client_errors(
    mock_dependencies: MagicMock,
) -> None:
    """Shutdown clears health globals even when optional resources fail."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    scheduler = MagicMock()
    scheduler.stop = AsyncMock(side_effect=RuntimeError("scheduler"))
    service = MagicMock()
    service.close = AsyncMock(side_effect=RuntimeError("client"))
    server._google_health_scheduler = scheduler
    server._google_health_service = service
    await server._stop_google_health()
    assert getattr(server, "_google_health_scheduler", object()) is None
    assert getattr(server, "_google_health_service", object()) is None


@pytest.mark.asyncio
async def test_google_health_export_worker_start_failure_rolls_back(
    mock_dependencies: MagicMock,
) -> None:
    """A failing export worker must not leave the scheduler/service dangling."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    from cryptography.fernet import Fernet

    import blacki.server as server
    from blacki.health.config import GoogleHealthConfig
    from blacki.health.nutrition_worker import NutritionExportWorker
    from blacki.health.scheduler import GoogleHealthScheduler

    server._container = MagicMock()
    config = GoogleHealthConfig(
        client_id="id",
        client_secret="secret",
        redirect_uri="https://example.test/callback",
        token_encryption_key=Fernet.generate_key().decode(),
    )
    with (
        patch(
            "blacki.health.config.GoogleHealthConfig.from_environment",
            return_value=config,
        ),
        patch.object(GoogleHealthScheduler, "start", new=AsyncMock()),
        patch.object(GoogleHealthScheduler, "stop", new=AsyncMock()),
        patch.object(
            NutritionExportWorker,
            "start",
            new=AsyncMock(side_effect=RuntimeError("worker boot failure")),
        ),
        patch.object(NutritionExportWorker, "close", new=AsyncMock()),
    ):
        await server._start_google_health()

    assert server._google_health_service is None
    assert server._google_health_scheduler is None
    assert server._google_health_export_worker is None
    server._container = None


@pytest.mark.asyncio
async def test_google_health_stop_without_container_skips_clearing_worker_ref(
    mock_dependencies: MagicMock,
) -> None:
    """Shutdown must not crash if the container was already torn down."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server._container = None
    worker = MagicMock()
    worker.stop = AsyncMock()
    worker.close = AsyncMock()
    server._google_health_export_worker = worker

    await server._stop_google_health()

    worker.stop.assert_awaited_once()
    assert getattr(server, "_google_health_export_worker", object()) is None


@pytest.mark.asyncio
async def test_google_health_stop_suppresses_export_worker_errors(
    mock_dependencies: MagicMock,
) -> None:
    """Shutdown clears the export worker global even if it fails to stop."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server._container = MagicMock()
    worker = MagicMock()
    worker.stop = AsyncMock(side_effect=RuntimeError("worker"))
    server._google_health_export_worker = worker

    await server._stop_google_health()

    assert getattr(server, "_google_health_export_worker", object()) is None
    server._container = None


@pytest.fixture
async def miniapp_server(
    mock_dependencies: MagicMock,
) -> AsyncIterator[Any]:
    """Import the server module and reset its Mini App globals around each test."""
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    server._telegram_bot = None
    server._miniapp_server = None
    server._miniapp_task = None
    yield server
    leftover_task = cast(Any, server)._miniapp_task
    if leftover_task is not None and not leftover_task.done():
        leftover_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await leftover_task
    server._telegram_bot = None
    server._miniapp_server = None
    server._miniapp_task = None


@pytest.mark.asyncio
async def test_start_miniapp_skips_invalid_configuration(
    miniapp_server: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed Mini App port must disable the Mini App, not the main server."""
    monkeypatch.setenv("MINIAPP_PORT", "not-a-port")

    await miniapp_server._start_miniapp()

    assert miniapp_server._miniapp_task is None
    assert miniapp_server._miniapp_server is None
    assert "Mini App configuration is invalid" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("telegram_running", [True, False])
async def test_start_miniapp_serves_own_port_and_attaches_running_bot(
    miniapp_server: Any, monkeypatch: pytest.MonkeyPatch, telegram_running: bool
) -> None:
    """The Mini App runs on its own port; a running bot learns its launch URL."""
    from blacki.telegram.bot import TelegramBot

    monkeypatch.setenv("MINIAPP_PORT", "8091")
    monkeypatch.setenv("MINIAPP_PUBLIC_URL", "https://app.example.test")
    cast(Any, miniapp_server.env).log_level = "INFO"
    bot = create_autospec(TelegramBot, instance=True, spec_set=True)
    miniapp_server._telegram_bot = bot if telegram_running else None

    with patch.object(miniapp_server.uvicorn, "Server", autospec=True) as server_cls:
        await miniapp_server._start_miniapp()

        if telegram_running:
            bot.attach_miniapp.assert_called_once_with(
                "https://app.example.test/miniapp/"
            )
        else:
            bot.attach_miniapp.assert_not_called()
        server_cls.assert_called_once()
        config = server_cls.call_args.args[0]
        assert config.port == 8091
        assert config.host == "127.0.0.1"
        assert config.lifespan == "off"
        assert miniapp_server._miniapp_server is server_cls.return_value
        assert miniapp_server._miniapp_task is not None

        await miniapp_server._stop_miniapp()

    assert server_cls.return_value.should_exit is True
    assert miniapp_server._miniapp_task is None
    assert miniapp_server._miniapp_server is None


@pytest.mark.asyncio
async def test_start_miniapp_port_in_use_keeps_main_loop_alive(
    miniapp_server: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A Mini App port already in use disables only the Mini App.

    Uses a real listening socket and the real uvicorn path (no patching).
    Before the fix, uvicorn's sys.exit(1) escaped the task and killed the loop.
    """
    from blacki.telegram.bot import TelegramBot

    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    try:
        monkeypatch.setenv("MINIAPP_HOST", "127.0.0.1")
        monkeypatch.setenv("MINIAPP_PORT", str(occupied.getsockname()[1]))
        cast(Any, miniapp_server.env).log_level = "INFO"
        bot = create_autospec(TelegramBot, instance=True, spec_set=True)
        miniapp_server._telegram_bot = bot

        await miniapp_server._start_miniapp()
        # Stand-in for the main loop (polling, scheduler): it must keep running.
        heartbeats = 0
        for _ in range(10):
            heartbeats += 1
            await asyncio.sleep(0.02)

        assert heartbeats == 10
        assert miniapp_server._miniapp_task is None
        assert miniapp_server._miniapp_server is None
        bot.attach_miniapp.assert_not_called()
        assert "Mini App could not bind 127.0.0.1" in caplog.text
    finally:
        occupied.close()


@pytest.mark.asyncio
async def test_serve_miniapp_swallows_uvicorn_exit_and_closes_socket(
    miniapp_server: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """uvicorn exits the process on startup failure; the wrapper must absorb it."""
    fake_server = create_autospec(
        miniapp_server.uvicorn.Server, instance=True, spec_set=True
    )
    fake_server.serve.side_effect = SystemExit(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    await miniapp_server._serve_miniapp(fake_server, sock)

    assert sock.fileno() == -1
    assert "Mini App server stopped" in caplog.text


@pytest.mark.asyncio
async def test_stop_miniapp_without_server_is_noop(miniapp_server: Any) -> None:
    """Shutdown must be safe when the Mini App was never started."""
    await miniapp_server._stop_miniapp()

    assert miniapp_server._miniapp_server is None
    assert miniapp_server._miniapp_task is None


@pytest.mark.asyncio
async def test_stop_miniapp_tolerates_task_that_raised(
    miniapp_server: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A Mini App server that crashed must be logged without blocking shutdown."""

    async def crash() -> None:
        raise RuntimeError("socket closed")

    miniapp_server._miniapp_task = asyncio.create_task(crash())
    await asyncio.sleep(0)

    await miniapp_server._stop_miniapp()

    assert "Mini App server stopped with an error" in caplog.text
    assert miniapp_server._miniapp_task is None


@pytest.mark.asyncio
async def test_stop_miniapp_cancels_task_that_was_already_cancelled(
    miniapp_server: Any,
) -> None:
    """A task cancelled before shutdown is cancelled again without raising."""
    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    miniapp_server._miniapp_task = task

    await miniapp_server._stop_miniapp()

    assert task.cancelled()
    assert miniapp_server._miniapp_task is None


@pytest.mark.asyncio
async def test_stop_miniapp_cancels_task_that_ignores_shutdown(
    miniapp_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Mini App server that does not stop within the timeout is cancelled."""
    task = asyncio.create_task(asyncio.sleep(60))
    miniapp_server._miniapp_task = task
    timeout_expired = create_autospec(asyncio.wait_for, side_effect=TimeoutError)
    monkeypatch.setattr(miniapp_server.asyncio, "wait_for", timeout_expired)

    await miniapp_server._stop_miniapp()

    timeout_expired.assert_called_once()
    assert timeout_expired.call_args.kwargs == {"timeout": 5}
    assert task.cancelling() > 0
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert miniapp_server._miniapp_task is None
