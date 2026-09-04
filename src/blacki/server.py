"""FastAPI server module.

This module provides a FastAPI server for ADK agents with comprehensive observability
features using custom OpenTelemetry setup. Includes an optional ADK web interface for
interactive agent testing.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from google.adk.cli.fast_api import get_fast_api_app
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from .adk_runtime import create_adk_runtime
from .container import AppContainer, close_container, init_container
from .dashboard.routes import create_dashboard_router
from .privacy import configure_private_tool_privacy
from .utils import (
    ConfigurationError,
    ServerEnv,
    configure_otel_resource,
    initialize_environment,
    setup_logging,
    setup_tracing,
    validation,
)

logger = logging.getLogger(__name__)

env = initialize_environment(ServerEnv)
private_tool_secure_mode = configure_private_tool_privacy()

configure_otel_resource(
    agent_name=env.agent_name,
)

if not private_tool_secure_mode:
    GoogleADKInstrumentor().instrument()

setup_logging(log_level=env.log_level)
setup_tracing()

_telegram_bot = None
_container: AppContainer | None = None
_google_health_service = None
_google_health_scheduler = None
_google_health_export_worker: Any = None
_gmail_oauth_service: Any = None
_google_health_backfill_tasks: set[asyncio.Task[None]] = set()


def _schedule_google_health_backfill(telegram_user_id: str | None = None) -> None:
    """Queue historical meal enrollment without delaying request handling."""
    if not isinstance(_container, AppContainer):
        return
    if _google_health_export_worker is None:
        return

    from .health.nutrition_backfill import NutritionBackfillCoordinator

    coordinator = NutritionBackfillCoordinator(
        _container.google_health_storage,
        _container.calorie_storage,
        wake=_google_health_export_worker.wake,
    )

    async def run() -> None:
        try:
            if telegram_user_id is None:
                await coordinator.run_all_eligible()
            else:
                await coordinator.run_user(telegram_user_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Google Health nutrition backfill task failed")

    task = asyncio.create_task(
        run(),
        name=(
            "google_health_nutrition_backfill"
            if telegram_user_id is None
            else "google_health_nutrition_backfill_user"
        ),
    )
    _google_health_backfill_tasks.add(task)
    task.add_done_callback(_google_health_backfill_tasks.discard)


async def _start_google_health() -> None:
    """Initialize the optional Google Health connector and its schedulers."""
    global \
        _google_health_scheduler, \
        _google_health_service, \
        _google_health_export_worker

    if _container is None:
        logger.info("Google Health connector not started (no container)")
        return

    from .health.config import GoogleHealthConfig, GoogleHealthConfigurationError
    from .health.nutrition_worker import NutritionExportWorker
    from .health.scheduler import GoogleHealthScheduler
    from .health.service import GoogleHealthService

    try:
        config = GoogleHealthConfig.from_environment()
    except GoogleHealthConfigurationError:
        logger.exception("Google Health configuration is invalid; connector disabled")
        return
    if config is None:
        logger.info("Google Health connector not configured")
        return

    service = GoogleHealthService(config, _container.google_health_storage)
    scheduler = GoogleHealthScheduler(service)
    try:
        await scheduler.start()
    except Exception:
        logger.exception("Google Health scheduler failed to start")
        await service.close()
        return

    export_worker = NutritionExportWorker(config, _container.google_health_storage)
    try:
        await export_worker.start()
    except Exception:
        logger.exception("Google Health nutrition export worker failed to start")
        await scheduler.stop()
        await export_worker.close()
        await service.close()
        return

    _google_health_service = service
    _google_health_scheduler = scheduler
    _google_health_export_worker = export_worker
    _container.nutrition_export_worker = export_worker
    _schedule_google_health_backfill()
    logger.info("Google Health connector initialized")


async def _start_telegram_bot() -> None:
    """Initialize and start the Telegram bot."""
    global _telegram_bot

    if not env.is_telegram_configured:
        logger.info(
            "Telegram bot not configured "
            "(TELEGRAM_ENABLED=false or missing TELEGRAM_BOT_TOKEN)"
        )
        return

    try:
        from .agent import create_agent, create_app
        from .telegram import TelegramConfig
        from .telegram.bot import TelegramBot

        logger.info("Telegram configuration detected, initializing bot...")
        telegram_config = TelegramConfig.model_validate(
            {
                "TELEGRAM_ENABLED": env.telegram_enabled,
                "TELEGRAM_BOT_TOKEN": env.telegram_bot_token,
                "TELEGRAM_ACCESS_CODE": env.telegram_access_code,
            }
        )
        telegram_app = create_app(
            create_agent(include_user_scoped_tools=True),
        )
        adk_runtime = create_adk_runtime(env, agent_app=telegram_app)
        _telegram_bot = TelegramBot(
            telegram_config,
            adk_runtime,
            google_health_service=_google_health_service,
            gmail_oauth_service=_gmail_oauth_service,
        )
        logger.info("Telegram bot instance created")

        await _telegram_bot.start_polling()

        try:
            await _start_reminder_scheduler()
        except Exception:
            logger.exception(
                "Failed to start reminder scheduler — continuing without it"
            )
    except Exception:
        logger.exception("Failed to start Telegram bot")
        raise


async def _start_gmail() -> None:
    """Initialize the optional Gmail OAuth service on shared storage."""
    global _gmail_oauth_service

    if _container is None:
        logger.info("Gmail connector not started (no container)")
        return

    from .gmail import GmailConfig, GmailConfigurationError, GmailOAuthService

    try:
        config = GmailConfig.from_environment()
    except GmailConfigurationError as exc:
        logger.error(
            "Gmail configuration is invalid; connector disabled (%s)",
            type(exc).__name__,
        )
        return
    if config is None:
        logger.info("Gmail connector not configured")
        return

    _gmail_oauth_service = GmailOAuthService(config, _container.gmail_storage)
    logger.info("Gmail API connector initialized")


async def _stop_gmail() -> None:
    """Close the optional Gmail OAuth service."""
    global _gmail_oauth_service

    if _gmail_oauth_service is not None:
        try:
            await _gmail_oauth_service.close()
        except Exception as exc:
            logger.error(
                "Error closing Gmail connector (%s)",
                type(exc).__name__,
            )
    _gmail_oauth_service = None


async def _start_reminder_scheduler() -> None:
    """Start the reminder scheduler if storage is initialized."""
    if _container is None:
        logger.info("Reminder scheduler not started (no container)")
        return

    from .reminders import get_scheduler

    scheduler = get_scheduler()
    if _telegram_bot is not None:
        scheduler.set_callback(_telegram_bot.handle_scheduled_reminder)
    await scheduler.start()
    logger.info("Reminder scheduler started")


async def _stop_telegram_bot() -> None:
    """Stop the Telegram bot."""
    if _telegram_bot:
        logger.info("Stopping Telegram bot...")
        try:
            await _telegram_bot.stop()
            logger.info("Telegram bot stopped")
        except Exception:
            logger.exception("Error stopping Telegram bot")


async def _stop_reminder_scheduler() -> None:
    """Stop the reminder scheduler if running."""
    try:
        from .reminders import get_scheduler

        scheduler = get_scheduler()
        if scheduler._running:
            await scheduler.stop()
            logger.info("Reminder scheduler stopped")
    except RuntimeError:
        logger.debug("Scheduler not initialized, nothing to stop")
    except Exception:
        logger.exception("Error stopping reminder scheduler")


async def _stop_google_health() -> None:
    """Stop the optional health schedulers and close their HTTP clients."""
    global \
        _google_health_scheduler, \
        _google_health_service, \
        _google_health_export_worker

    backfill_tasks = list(_google_health_backfill_tasks)
    for task in backfill_tasks:
        task.cancel()
    if backfill_tasks:
        await asyncio.gather(*backfill_tasks, return_exceptions=True)
    _google_health_backfill_tasks.clear()

    if _google_health_export_worker is not None:
        if _container is not None:
            _container.nutrition_export_worker = None
        try:
            await _google_health_export_worker.stop()
            await _google_health_export_worker.close()
        except Exception:
            logger.exception("Error stopping Google Health nutrition export worker")
    if _google_health_scheduler is not None:
        try:
            await _google_health_scheduler.stop()
        except Exception:
            logger.exception("Error stopping Google Health scheduler")
    if _google_health_service is not None:
        try:
            await _google_health_service.close()
        except Exception:
            logger.exception("Error closing Google Health client")
    _google_health_scheduler = None
    _google_health_service = None
    _google_health_export_worker = None


AGENT_DIR = os.getenv("AGENT_DIR", str(Path(__file__).resolve().parent.parent))

DEFAULT_SQLITE_PATH = str(Path(AGENT_DIR) / ".adk" / "tools.db")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage long-lived application resources.

    Telegram polling needs to be initialized during startup and shut down
    explicitly during application teardown. Running this in the lifespan
    hook keeps the bot lifecycle aligned with the FastAPI app lifecycle.
    """
    global _container

    sqlite_path = env.sqlite_path or DEFAULT_SQLITE_PATH
    _container = await init_container(sqlite_path)
    try:
        await _container.initialize_all_storages()
        await _start_google_health()
        await _start_gmail()

        logger.info("Validating configuration...")
        try:
            warnings = validation.validate_configuration(
                env.telegram_enabled, env.telegram_bot_token
            )
            for warning in warnings:
                logger.warning(warning)
            logger.info("Configuration validated successfully")
        except ConfigurationError:
            logger.exception("Configuration validation failed")
            raise

        await _start_telegram_bot()
        yield
    finally:
        await _stop_google_health()
        await _stop_gmail()
        await _stop_reminder_scheduler()
        await _stop_telegram_bot()

        if _container is not None:
            from .user_files import reset_user_file_service

            reset_user_file_service()
            await close_container()
            _container = None

        from .tools.brave_search import close_shared_brave_search_client

        await close_shared_brave_search_client()

        from .tools.search import close_shared_exa_search_client

        await close_shared_exa_search_client()

        from .callbacks import close_shared_notify_client

        await close_shared_notify_client()


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=None,
    artifact_service_uri=None,
    memory_service_uri="mem0://",
    allow_origins=env.allow_origins_list,
    web=env.serve_web_interface,
    reload_agents=env.reload_agents,
    lifespan=lifespan,
)

app.include_router(create_dashboard_router(env))


@app.get(
    "/integrations/google-health/callback",
    response_class=HTMLResponse,
)
async def google_health_callback(
    state: str | None = None,
    code: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Complete Google OAuth without returning provider data to the browser."""
    if _google_health_service is None:
        return HTMLResponse(
            "<h1>Google Health is not configured</h1>",
            status_code=503,
        )

    from .health.client import GoogleHealthApiError
    from .health.service import GoogleHealthOAuthError

    try:
        completion = await _google_health_service.complete_authorization(
            state=state or "",
            code=code,
            error=error,
        )
    except GoogleHealthOAuthError:
        return HTMLResponse(
            "<h1>Google Health authorization could not be completed</h1>",
            status_code=400,
        )
    except GoogleHealthApiError as exc:
        logger.exception(
            "Google Health provider rejected OAuth completion: "
            "status_code=%s error_code=%s",
            exc.status_code,
            exc.error_code,
        )
        return HTMLResponse(
            "<h1>Google Health authorization could not be completed</h1>",
            status_code=502,
        )
    except Exception:
        logger.exception("Unexpected Google Health OAuth callback failure")
        return HTMLResponse(
            "<h1>Google Health authorization could not be completed</h1>",
            status_code=500,
        )

    if _telegram_bot is not None:
        try:
            await _telegram_bot.notify_health_connection(
                completion.telegram_user_id,
                connected=completion.connected,
            )
        except Exception:
            logger.exception("Failed to notify Telegram after Google Health OAuth")

    if completion.connected:
        _schedule_google_health_backfill(completion.telegram_user_id)

    title = (
        "Google Health connected"
        if completion.connected
        else "Google Health authorization cancelled"
    )
    return HTMLResponse(f"<h1>{title}</h1><p>You can return to Telegram.</p>")


@app.get(
    "/integrations/gmail/callback",
    response_class=HTMLResponse,
)
async def gmail_callback(
    state: str | None = None,
    code: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Complete user-bound Gmail OAuth without returning provider data."""
    if _gmail_oauth_service is None:
        return HTMLResponse(
            "<h1>Gmail is not configured</h1>",
            status_code=503,
        )
    if not state:
        return HTMLResponse(
            "<h1>Gmail authorization failed</h1><p>Missing OAuth state.</p>",
            status_code=400,
        )

    from .gmail import GmailApiError, GmailCredentialError, GmailOAuthError

    try:
        completion = await _gmail_oauth_service.complete_authorization(
            state=state,
            code=code,
            error=error,
        )
    except GmailOAuthError:
        logger.warning("Gmail OAuth callback validation failed")
        return HTMLResponse(
            "<h1>Gmail authorization failed</h1>",
            status_code=400,
        )
    except GmailCredentialError:
        logger.warning("Gmail OAuth callback credential validation failed")
        return HTMLResponse(
            "<h1>Gmail authorization failed</h1>",
            status_code=400,
        )
    except GmailApiError:
        logger.exception("Gmail provider rejected OAuth completion")
        return HTMLResponse(
            "<h1>Gmail authorization could not be completed</h1>",
            status_code=502,
        )
    except Exception:
        logger.exception("Unexpected Gmail OAuth callback failure")
        return HTMLResponse(
            "<h1>Gmail authorization could not be completed</h1>",
            status_code=500,
        )

    if _telegram_bot is not None:
        try:
            await _telegram_bot.notify_gmail_connected(
                completion.telegram_user_id,
                connected=completion.connected,
            )
        except Exception:
            logger.exception("Failed to notify Telegram after Gmail OAuth")

    title = (
        "Gmail connected" if completion.connected else "Gmail authorization cancelled"
    )
    return HTMLResponse(f"<h1>{title}</h1><p>You can return to Telegram.</p>")


@app.get("/live")
async def live() -> dict[str, str]:
    """Report only that the application process and event loop are alive."""
    return {"status": "alive"}


async def _readiness_response() -> JSONResponse:
    """Check already-initialized dependencies required to serve tool requests."""
    if _container is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "checks": {"database": "starting"},
            },
        )

    try:
        async with _container.conn.execute("SELECT 1") as cursor:
            await cursor.fetchone()
    except Exception:
        logger.exception("Readiness database check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "checks": {"database": "unhealthy"},
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "checks": {"database": "healthy"},
        },
    )


@app.get("/ready")
async def ready() -> JSONResponse:
    """Report whether critical startup resources are initialized and healthy."""
    return await _readiness_response()


@app.get("/health")
async def health() -> JSONResponse:
    """Compatibility alias with exactly the same semantics as readiness."""
    return await _readiness_response()


def main() -> None:
    """Run the FastAPI server.

    Starts the ADK agent server. Features include:
    - Environment variable loading and validation via Pydantic
    - Custom OpenTelemetry setup for resource attributes
    - Optional ADK web interface for interactive agent testing
    - Session and memory persistence
    - CORS configuration

    Environment Variables:
        AGENT_DIR: Path to agent source directory (default: auto-detect from __file__)
        AGENT_NAME: Unique service identifier (required)
        LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        SERVE_WEB_INTERFACE: Whether to serve the web interface (true/false)
        RELOAD_AGENTS: Whether to reload agents on file changes (true/false)
        AGENT_ENGINE: Agent Engine instance for session and memory
        SQLITE_PATH: Path to SQLite database (default: {AGENT_DIR}/.adk/tools.db)
        OPENROUTER_API_KEY: Key for LiteLLM/OpenRouter
        ALLOW_ORIGINS: JSON array string of allowed CORS origins
        HOST: Server host (default: 127.0.0.1, set to 0.0.0.0 for containers)
        PORT: Server port (default: 8080)
    """
    uvicorn.run(
        app,
        host=env.host,
        port=env.port,
    )

    return


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    main()
