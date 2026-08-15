# mypy: disable-error-code="no-untyped-def"
"""Tests for server configuration."""

import json
import sys
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI


@pytest.fixture
def mock_dependencies() -> Generator[MagicMock]:
    """Mock external dependencies to prevent side effects during import."""
    with (
        patch("google.adk.cli.fast_api.get_fast_api_app") as mock_get_app,
        patch("blacki.utils.initialize_environment") as mock_init_env,
        patch("blacki.utils.configure_otel_resource"),
        patch("openinference.instrumentation.google_adk.GoogleADKInstrumentor"),
        patch("blacki.utils.setup_logging"),
        patch("blacki.utils.setup_tracing"),
    ):
        mock_env = MagicMock()
        mock_env.session_uri = None
        mock_env.allow_origins_list = ["*"]
        mock_env.serve_web_interface = True
        mock_env.reload_agents = False
        mock_env.sqlite_path = None
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


@pytest.mark.parametrize(
    "setting",
    ["ZEPTO_MCP_ENABLED", "KOKORO_TTS_BASE_URL"],
)
def test_server_skips_openinference_in_private_tool_mode(
    mock_dependencies: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    """Private tools must take the no-content-instrumentation startup branch."""
    value = "true" if setting == "ZEPTO_MCP_ENABLED" else "http://kokoro.internal"
    monkeypatch.setenv(setting, value)
    if "blacki.server" in sys.modules:
        del sys.modules["blacki.server"]

    import blacki.server as server

    assert server.private_tool_secure_mode is True


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
    log_warning = MagicMock()

    with (
        patch.object(server, "init_container", new=init_container),
        patch.object(server, "close_container", new=close_container),
        patch.object(server, "_start_telegram_bot", new=AsyncMock()),
        patch.object(server, "_stop_telegram_bot", new=AsyncMock()),
        patch.object(server, "_stop_reminder_scheduler", new=AsyncMock()),
        patch.object(
            server.validation,
            "validate_configuration",
            return_value=["test warning"],
        ),
        patch.object(server.logger, "warning", new=log_warning),
        patch("blacki.tools.close_shared_brave_search_client", new=close_brave),
        patch("blacki.search.close_shared_exa_search_client", new=close_exa),
        patch("blacki.callbacks.close_shared_notify_client", new=close_notify),
    ):
        async with server.lifespan(server.app):
            pass

    container.initialize_all_storages.assert_awaited_once()
    close_container.assert_awaited_once()
    close_brave.assert_awaited_once()
    close_exa.assert_awaited_once()
    close_notify.assert_awaited_once()
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
        patch("blacki.tools.close_shared_brave_search_client", new=close_brave),
        patch("blacki.search.close_shared_exa_search_client", new=close_exa),
        patch("blacki.callbacks.close_shared_notify_client", new=close_notify),
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
        patch.object(server, "_stop_telegram_bot", new=AsyncMock()),
        patch.object(server, "_stop_reminder_scheduler", new=AsyncMock()),
        patch.object(server.validation, "validate_configuration", return_value=[]),
        patch("blacki.tools.close_shared_brave_search_client", new=close_brave),
        patch("blacki.search.close_shared_exa_search_client", new=close_exa),
        patch("blacki.callbacks.close_shared_notify_client", new=close_notify),
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
    server.env.telegram_tool_notifications = False
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
    server.env.telegram_tool_notifications = False
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
