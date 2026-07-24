"""FastAPI server module.

This module provides a FastAPI server for ADK agents with comprehensive observability
features using custom OpenTelemetry setup. Includes an optional ADK web interface for
interactive agent testing.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.adk.cli.fast_api import get_fast_api_app
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from .adk_runtime import create_adk_runtime
from .container import AppContainer, close_container, init_container
from .utils import (
    ConfigurationError,
    ServerEnv,
    configure_otel_resource,
    initialize_environment,
    setup_logging,
    setup_tracing,
    validation,
)
from .utils.privacy import route_data_redaction_enabled

logger = logging.getLogger(__name__)

env = initialize_environment(ServerEnv)

configure_otel_resource(
    agent_name=env.agent_name,
)

_route_data_redaction = route_data_redaction_enabled()
GoogleADKInstrumentor().instrument(
    config=TraceConfig(
        hide_inputs=True if _route_data_redaction else None,
        hide_outputs=True if _route_data_redaction else None,
    )
)

setup_logging(log_level=env.log_level)
setup_tracing()

_telegram_bot = None
_container: AppContainer | None = None


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
        from .telegram import TelegramConfig
        from .telegram.bot import TelegramBot

        logger.info("Telegram configuration detected, initializing bot...")
        telegram_config = TelegramConfig.model_validate(
            {
                "TELEGRAM_ENABLED": env.telegram_enabled,
                "TELEGRAM_BOT_TOKEN": env.telegram_bot_token,
                "TELEGRAM_TOOL_NOTIFICATIONS": env.telegram_tool_notifications,
            }
        )
        adk_runtime = create_adk_runtime(env)
        _telegram_bot = TelegramBot(telegram_config, adk_runtime)
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


AGENT_DIR = os.getenv("AGENT_DIR", str(Path(__file__).resolve().parent.parent))

DEFAULT_SQLITE_PATH = str(Path(AGENT_DIR) / ".adk" / "tools.db")

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=None,
    artifact_service_uri=None,
    memory_service_uri="mem0://",
    allow_origins=env.allow_origins_list,
    web=env.serve_web_interface,
    reload_agents=env.reload_agents,
)


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
        await _stop_reminder_scheduler()
        await _stop_telegram_bot()

        if _container is not None:
            await close_container()
            _container = None

        from .tools import close_shared_brave_search_client

        await close_shared_brave_search_client()

        from .search import close_shared_exa_search_client

        await close_shared_exa_search_client()

        from .routes import close_shared_routes_client

        await close_shared_routes_client()

        from .callbacks import close_shared_notify_client

        await close_shared_notify_client()


app.router.lifespan_context = lifespan


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
