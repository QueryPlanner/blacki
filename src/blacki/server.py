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
from typing import Any

import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.service_registry import get_service_registry
from google.adk.memory.base_memory_service import BaseMemoryService
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from .adk_runtime import (
    build_session_db_kwargs,
    build_session_service_uri,
    create_adk_runtime,
)
from .container import AppContainer, close_container, init_container
from .utils import (
    ConfigurationError,
    ServerEnv,
    configure_otel_resource,
    initialize_environment,
    setup_logging,
    validation,
)

logger = logging.getLogger(__name__)

env = initialize_environment(ServerEnv)

configure_otel_resource(
    agent_name=env.agent_name,
)

GoogleADKInstrumentor().instrument()

setup_logging(log_level=env.log_level)

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

session_uri = build_session_service_uri(env)
session_db_kwargs = build_session_db_kwargs(env)


def _create_mem0_memory_service(uri: str, **kwargs: Any) -> BaseMemoryService:
    """Factory for mem0:// URI scheme.

    Returns Mem0MemoryService if client is available, InMemoryMemoryService otherwise.
    """
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

    from blacki.memory.config import get_memory_client

    client = get_memory_client()
    if client is None:
        logger.info("Mem0 client not available, using in-memory memory service")
        return InMemoryMemoryService()

    from blacki.memory.mem0_memory_service import Mem0MemoryService

    logger.info("Mem0 memory service initialized")
    return Mem0MemoryService(client)


get_service_registry().register_memory_service("mem0", _create_mem0_memory_service)

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=session_uri,
    session_db_kwargs=session_db_kwargs,
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

    if env.database_url:
        _container = await init_container(env.database_url)
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
    try:
        yield
    finally:
        await _stop_reminder_scheduler()
        await _stop_telegram_bot()

        if _container is not None:
            await close_container()
            _container = None

        from .tools import close_shared_brave_search_client

        await close_shared_brave_search_client()

        from .callbacks import close_shared_notify_client

        await close_shared_notify_client()


app.router.lifespan_context = lifespan


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint for container orchestration.

    Returns:
        dict with status key indicating service health.
    """
    from blacki.memory.config import get_memory_client

    checks: dict[str, str] = {}

    if _container is not None:
        try:
            await _container.pool.fetchval("SELECT 1")
            checks["database"] = "healthy"
        except Exception:
            checks["database"] = "unhealthy"

    client = get_memory_client()
    checks["memory_service"] = "healthy" if client else "unavailable"

    all_ok = all(v in ("healthy", "unavailable") for v in checks.values())
    status = "ok" if all_ok else "degraded"

    return {"status": status, "checks": checks}


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
