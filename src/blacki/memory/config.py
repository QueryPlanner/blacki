"""Mem0 OSS memory configuration and client factory."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from mem0 import Memory

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "openrouter/openai/gpt-oss-20b"
DEFAULT_EMBEDDER_PROVIDER = "gemini"
DEFAULT_EMBEDDER_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDER_DIMS = 768
DEFAULT_COLLECTION_NAME = "blacki_memories_gemini_768"
DEFAULT_QDRANT_PATH = "./data/qdrant"

_memory_client: Memory | None = None
_memory_client_error: str | None = None


def _get_env_value(name: str) -> str:
    """Return a stripped environment value or an empty string."""
    return os.getenv(name, "").strip()


def _mask_hostname(hostname: str) -> str:
    """Mask a hostname enough for logs and tool responses."""
    if len(hostname) <= 12:
        return hostname
    return f"{hostname[:6]}...{hostname[-6:]}"


def _get_qdrant_hostname() -> str:
    """Return the configured Qdrant Cloud hostname, if any."""
    qdrant_url = _get_env_value("MEM0_QDRANT_URL")
    if not qdrant_url:
        return ""
    return urlparse(qdrant_url).hostname or ""


def _describe_memory_init_error(error: Exception) -> str:
    """Translate backend initialization failures into actionable user text."""
    error_message = str(error)
    qdrant_hostname = _get_qdrant_hostname()

    qdrant_host_does_not_resolve = (
        "nodename nor servname provided" in error_message
        or "Name or service not known" in error_message
    )
    if qdrant_host_does_not_resolve and qdrant_hostname:
        masked_hostname = _mask_hostname(qdrant_hostname)
        return (
            "Memory backend could not reach Qdrant. "
            f"MEM0_QDRANT_URL host '{masked_hostname}' does not resolve. "
            "Use the Qdrant Cloud cluster endpoint from Cluster Details, "
            "for example https://xyz-example.eu-central.aws.cloud.qdrant.io."
        )

    return f"Memory backend failed to initialize: {error_message}"


def _get_embedding_dims() -> int:
    """Get the configured embedding vector size."""
    dims = _get_env_value("MEM0_EMBEDDER_DIMS")
    if not dims:
        return DEFAULT_EMBEDDER_DIMS

    try:
        return int(dims)
    except ValueError:
        logger.warning("Invalid MEM0_EMBEDDER_DIMS: %s", dims)
        return DEFAULT_EMBEDDER_DIMS


def _get_llm_config() -> dict[str, Any]:
    """Build LLM configuration for OSS Memory from environment."""
    llm_model = _get_env_value("MEM0_LLM_MODEL")
    openrouter_key = _get_env_value("OPENROUTER_API_KEY")
    if not llm_model:
        if openrouter_key:
            llm_model = DEFAULT_LLM_MODEL
        else:
            return {}

    llm_provider = _get_env_value("MEM0_LLM_PROVIDER")
    if not llm_provider:
        llm_provider = "litellm" if openrouter_key else "openai"

    llm_config: dict[str, Any] = {"model": llm_model}

    api_key = _get_env_value("MEM0_LLM_API_KEY")
    if not api_key:
        api_key = openrouter_key
    if api_key:
        llm_config["api_key"] = api_key

    temperature = _get_env_value("MEM0_LLM_TEMPERATURE")
    if temperature:
        try:
            llm_config["temperature"] = float(temperature)
        except ValueError:
            logger.warning("Invalid MEM0_LLM_TEMPERATURE: %s", temperature)

    max_tokens = _get_env_value("MEM0_LLM_MAX_TOKENS")
    if max_tokens:
        try:
            llm_config["max_tokens"] = int(max_tokens)
        except ValueError:
            logger.warning("Invalid MEM0_LLM_MAX_TOKENS: %s", max_tokens)

    return {
        "provider": llm_provider,
        "config": llm_config,
    }


def _get_embedder_config() -> dict[str, Any]:
    """Build embedder configuration for OSS Memory from environment."""
    embedder_provider = _get_env_value("MEM0_EMBEDDER_PROVIDER")
    if not embedder_provider:
        embedder_provider = DEFAULT_EMBEDDER_PROVIDER

    embedder_model = _get_env_value("MEM0_EMBEDDER_MODEL")
    if not embedder_model:
        embedder_model = DEFAULT_EMBEDDER_MODEL

    embedder_config: dict[str, Any] = {
        "model": embedder_model,
        "embedding_dims": _get_embedding_dims(),
    }

    api_key = _get_env_value("MEM0_EMBEDDER_API_KEY")
    if not api_key:
        api_key = _get_env_value("GOOGLE_API_KEY")
    if api_key:
        embedder_config["api_key"] = api_key

    return {
        "provider": embedder_provider,
        "config": embedder_config,
    }


def _get_vector_store_config() -> dict[str, Any]:
    """Build vector store configuration for OSS Memory from environment."""
    collection_name = _get_env_value("MEM0_COLLECTION_NAME")
    if not collection_name:
        collection_name = DEFAULT_COLLECTION_NAME

    qdrant_config: dict[str, Any] = {
        "collection_name": collection_name,
        "embedding_model_dims": _get_embedding_dims(),
    }

    qdrant_url = _get_env_value("MEM0_QDRANT_URL")
    qdrant_api_key = _get_env_value("MEM0_QDRANT_API_KEY")
    qdrant_cloud_configured = bool(qdrant_url or qdrant_api_key)

    if qdrant_cloud_configured:
        if not qdrant_url or not qdrant_api_key:
            missing = "MEM0_QDRANT_URL" if not qdrant_url else "MEM0_QDRANT_API_KEY"
            raise ValueError(
                f"MEM0_QDRANT_URL and MEM0_QDRANT_API_KEY must be set together. "
                f"Missing: {missing}."
            )

        qdrant_config["url"] = qdrant_url
        qdrant_config["api_key"] = qdrant_api_key
        return {
            "provider": "qdrant",
            "config": qdrant_config,
        }

    qdrant_host = _get_env_value("MEM0_QDRANT_HOST")
    qdrant_port = _get_env_value("MEM0_QDRANT_PORT")
    qdrant_server_configured = bool(qdrant_host or qdrant_port)

    if qdrant_server_configured:
        if not qdrant_host or not qdrant_port:
            missing = "MEM0_QDRANT_HOST" if not qdrant_host else "MEM0_QDRANT_PORT"
            raise ValueError(
                f"MEM0_QDRANT_HOST and MEM0_QDRANT_PORT must be set together. "
                f"Missing: {missing}."
            )

        qdrant_config["host"] = qdrant_host
        qdrant_config["port"] = int(qdrant_port)
        return {
            "provider": "qdrant",
            "config": qdrant_config,
        }

    qdrant_path = _get_env_value("MEM0_QDRANT_PATH")
    if not qdrant_path:
        qdrant_path = DEFAULT_QDRANT_PATH
    qdrant_config["path"] = qdrant_path
    return {
        "provider": "qdrant",
        "config": qdrant_config,
    }


def _build_oss_config() -> dict[str, Any]:
    """Build full configuration dict for OSS Memory."""
    config: dict[str, Any] = {}

    llm_config = _get_llm_config()
    if llm_config:
        config["llm"] = llm_config

    config["embedder"] = _get_embedder_config()
    config["vector_store"] = _get_vector_store_config()

    return config


def get_memory_client() -> Memory | None:
    """Get or create the Mem0 client instance.

    Returns an OSS Memory client. Returns None if mem0ai is not installed or
    the configured memory backend cannot be initialized.
    """
    global _memory_client, _memory_client_error

    if _memory_client is not None:
        return _memory_client

    try:
        from mem0 import Memory

        config = _build_oss_config()
        _memory_client = Memory.from_config(config)
        _memory_client_error = None
        logger.info("Initialized Mem0 OSS client")
        return _memory_client
    except ImportError:
        _memory_client_error = "mem0ai is not installed, memory tools are disabled."
        logger.warning(_memory_client_error)
        return None
    except Exception as e:
        _memory_client_error = _describe_memory_init_error(e)
        logger.exception("Failed to initialize Mem0 OSS client: %s", e)
        return None


def get_memory_client_error() -> str | None:
    """Return the last Mem0 initialization error, if any."""
    return _memory_client_error


def get_default_user_id() -> str:
    """Get the default user ID from environment."""
    return os.getenv("MEM0_USER_ID", "default").strip()


def get_search_limit() -> int:
    """Get the default search result limit from environment."""
    limit = os.getenv("MEM0_SEARCH_LIMIT", "5").strip()
    try:
        return int(limit)
    except ValueError:
        return 5


def reset_memory_client() -> None:
    """Reset the cached memory client (for testing)."""
    global _memory_client, _memory_client_error
    _memory_client = None
    _memory_client_error = None
