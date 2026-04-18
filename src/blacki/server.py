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

import asyncpg  # type: ignore[import-untyped]
import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from .adk_runtime import (
    build_session_db_kwargs,
    build_session_service_uri,
    create_adk_runtime,
)
from .utils import (
    ServerEnv,
    configure_otel_resource,
    initialize_environment,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Load and validate environment configuration
env = initialize_environment(ServerEnv)

# Configure OpenTelemetry
configure_otel_resource(
    agent_name=env.agent_name,
)

# Initialize Langfuse/OpenInference instrumentation
GoogleADKInstrumentor().instrument()

# Configure logging
setup_logging(log_level=env.log_level)

# Telegram bot instance (initialized on startup)
_telegram_bot = None
_reminder_pool: asyncpg.Pool | None = None


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

        await _start_reminder_scheduler()
    except Exception:
        logger.exception("Failed to start Telegram bot")
        raise


async def _start_reminder_scheduler() -> None:
    """Start the reminder scheduler if storage is initialized."""
    if _reminder_pool is None:
        logger.info("Reminder scheduler not started (no database pool)")
        return

    try:
        from .reminders import get_scheduler

        scheduler = get_scheduler()
        if _telegram_bot is not None and _telegram_bot._api is not None:
            scheduler.set_api(_telegram_bot._api)
        await scheduler.start()
        logger.info("Reminder scheduler started")
    except Exception:
        logger.exception("Failed to start reminder scheduler")


async def _stop_telegram_bot() -> None:
    """Stop the Telegram bot."""
    if _telegram_bot:
        logger.info("Stopping Telegram bot...")
        try:
            await _telegram_bot.stop()
            logger.info("Telegram bot stopped")
        except Exception:
            logger.exception("Error stopping Telegram bot")


async def _init_reminder_pool(database_url: str) -> asyncpg.Pool:
    """Initialize the Postgres pool for reminder storage."""
    pool = await asyncpg.create_pool(
        database_url,
        min_size=1,
        max_size=5,
    )

    from .reminders import init_reminder_storage

    await init_reminder_storage(pool)
    logger.info("Reminder storage initialized with Postgres pool")
    return pool


async def _stop_reminder_scheduler() -> None:
    """Stop the reminder scheduler if running."""
    try:
        from .reminders import get_scheduler

        scheduler = get_scheduler()
        if scheduler._running:
            await scheduler.stop()
            logger.info("Reminder scheduler stopped")
    except RuntimeError:
        pass
    except Exception:
        logger.exception("Error stopping reminder scheduler")


async def _close_reminder_pool() -> None:
    """Close the reminder Postgres pool."""
    global _reminder_pool
    if _reminder_pool is not None:
        await _reminder_pool.close()
        _reminder_pool = None
        logger.info("Reminder Postgres pool closed")


# Use .resolve() to handle symlinks and ensure absolute path across environments
AGENT_DIR = os.getenv("AGENT_DIR", str(Path(__file__).resolve().parent.parent))

session_uri = build_session_service_uri(env)
session_db_kwargs = build_session_db_kwargs(env)

# ADK fastapi app will set up OTel using resource attributes from env vars
app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=session_uri,
    session_db_kwargs=session_db_kwargs,
    artifact_service_uri=None,  # Explicitly None as GCP bucket not used
    # Memory service does not yet support Postgres scheme in ADK
    memory_service_uri=None,
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
    global _reminder_pool

    if env.database_url:
        _reminder_pool = await _init_reminder_pool(env.database_url)

    await _start_telegram_bot()
    try:
        yield
    finally:
        await _stop_reminder_scheduler()
        await _close_reminder_pool()
        await _stop_telegram_bot()


app.router.lifespan_context = lifespan


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for container orchestration.

    Returns:
        dict with status key indicating service health.
    """
    return {"status": "ok"}


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
        DATABASE_URL: Postgres URL for session and memory
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


if __name__ == "__main__":
    main()
