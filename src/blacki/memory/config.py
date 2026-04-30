"""Mem0 memory configuration and client factory."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mem0 import Memory, MemoryClient

logger = logging.getLogger(__name__)

_memory_client: MemoryClient | Memory | None = None
_is_cloud_client: bool = False


def _get_llm_config() -> dict[str, Any]:
    """Build LLM configuration for OSS Memory from environment."""
    llm_model = os.getenv("MEM0_LLM_MODEL", "").strip()
    if not llm_model:
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if openrouter_key:
            llm_model = "openrouter/google/gemini-2.0-flash-001"
        else:
            return {}

    config: dict[str, Any] = {"model": llm_model}

    api_key = os.getenv("MEM0_LLM_API_KEY", "").strip()
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        config["api_key"] = api_key

    temperature = os.getenv("MEM0_LLM_TEMPERATURE", "").strip()
    if temperature:
        try:
            config["temperature"] = float(temperature)
        except ValueError:
            logger.warning("Invalid MEM0_LLM_TEMPERATURE: %s", temperature)

    max_tokens = os.getenv("MEM0_LLM_MAX_TOKENS", "").strip()
    if max_tokens:
        try:
            config["max_tokens"] = int(max_tokens)
        except ValueError:
            logger.warning("Invalid MEM0_LLM_MAX_TOKENS: %s", max_tokens)

    return config


def _get_embedder_config() -> dict[str, Any]:
    """Build embedder configuration for OSS Memory from environment."""
    embedder_model = os.getenv("MEM0_EMBEDDER_MODEL", "").strip()
    if not embedder_model:
        embedder_model = "BAAI/bge-small-en-v1.5"

    config: dict[str, Any] = {"model": embedder_model}

    dims = os.getenv("MEM0_EMBEDDER_DIMS", "").strip()
    if dims:
        try:
            config["embedding_dims"] = int(dims)
        except ValueError:
            logger.warning("Invalid MEM0_EMBEDDER_DIMS: %s", dims)

    return config


def _get_vector_store_config() -> dict[str, Any]:
    """Build vector store configuration for OSS Memory from environment."""
    qdrant_host = os.getenv("MEM0_QDRANT_HOST", "").strip()
    qdrant_port = os.getenv("MEM0_QDRANT_PORT", "").strip()

    if qdrant_host and qdrant_port:
        return {
            "type": "qdrant",
            "config": {
                "host": qdrant_host,
                "port": int(qdrant_port),
            },
        }

    qdrant_path = os.getenv("MEM0_QDRANT_PATH", "./data/qdrant").strip()
    return {
        "type": "qdrant",
        "config": {
            "path": qdrant_path,
        },
    }


def _build_oss_config() -> dict[str, Any]:
    """Build full configuration dict for OSS Memory."""
    config: dict[str, Any] = {}

    llm_config = _get_llm_config()
    if llm_config:
        config["llm"] = {"config": llm_config}

    config["embedder"] = {"config": _get_embedder_config()}
    config["vector_store"] = _get_vector_store_config()

    collection_name = os.getenv("MEM0_COLLECTION_NAME", "").strip()
    if collection_name:
        config["vector_store"]["config"]["collection_name"] = collection_name

    return config


def get_memory_client() -> MemoryClient | Memory | None:
    """Get or create the Mem0 client instance.

    Returns MemoryClient (cloud) if MEM0_API_KEY is set, otherwise Memory (OSS).
    Returns None if mem0ai is not installed.
    """
    global _memory_client, _is_cloud_client

    if _memory_client is not None:
        return _memory_client

    api_key = os.getenv("MEM0_API_KEY", "").strip()

    if api_key:
        try:
            from mem0 import MemoryClient

            _memory_client = MemoryClient(api_key=api_key)
            _is_cloud_client = True
            logger.info("Initialized Mem0 Cloud client")
            return _memory_client
        except ImportError:
            logger.warning("mem0ai not installed, memory tools disabled")
            return None

    try:
        from mem0 import Memory

        config = _build_oss_config()
        _memory_client = Memory.from_config(config)
        _is_cloud_client = False
        logger.info("Initialized Mem0 OSS client with config: %s", config)
        return _memory_client
    except ImportError:
        logger.warning("mem0ai not installed, memory tools disabled")
        return None


def is_cloud_client() -> bool:
    """Check if the current client is the cloud MemoryClient."""
    return _is_cloud_client


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
    global _memory_client, _is_cloud_client
    _memory_client = None
    _is_cloud_client = False
