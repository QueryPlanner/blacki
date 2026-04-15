"""FastAPI server module.

This module provides a FastAPI server for ADK agents with comprehensive observability
features using custom OpenTelemetry setup. Includes an optional ADK web interface for
interactive agent testing.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

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


async def _start_telegram_bot() -> None:
    """Initialize and start the Telegram bot."""
    global _telegram_bot

    if not env.is_telegram_configured:
        logger.info("Telegram bot not configured, skipping initialization")
        return

    try:
        from .agent import model
        from .telegram import TelegramConfig
        from .telegram.bot import TelegramBot

        telegram_config = TelegramConfig.model_validate(
            {
                "TELEGRAM_ENABLED": env.telegram_enabled,
                "TELEGRAM_BOT_TOKEN": env.telegram_bot_token,
            }
        )
        _telegram_bot = TelegramBot(telegram_config, model)
        logger.info("Telegram bot initialized")

        logger.info("Starting Telegram bot polling...")
        await _telegram_bot.start_polling()
    except Exception:
        logger.exception("Failed to start Telegram bot")


async def _stop_telegram_bot() -> None:
    """Stop the Telegram bot."""
    if _telegram_bot:
        logger.info("Stopping Telegram bot...")
        await _telegram_bot.stop()


# Use .resolve() to handle symlinks and ensure absolute path across environments
AGENT_DIR = os.getenv("AGENT_DIR", str(Path(__file__).resolve().parent.parent))

# Handle database URL conversion for asyncpg
session_uri = env.session_uri
if session_uri and session_uri.startswith("postgresql://"):
    session_uri = session_uri.replace("postgresql://", "postgresql+asyncpg://", 1)

# Define engine/pool settings for asyncpg connections
session_db_kwargs = {
    "pool_pre_ping": env.db_pool_pre_ping,
    "pool_recycle": env.db_pool_recycle,
    "pool_size": env.db_pool_size,
    "max_overflow": env.db_max_overflow,
    "pool_timeout": env.db_pool_timeout,
}

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
    await _start_telegram_bot()
    try:
        yield
    finally:
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
