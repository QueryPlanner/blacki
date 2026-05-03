"""Tests for Memory config module."""

from unittest.mock import MagicMock, patch

import pytest

from blacki.memory.config import (
    _build_oss_config,
    _get_embedder_config,
    _get_llm_config,
    _get_vector_store_config,
    get_default_user_id,
    get_memory_client,
    get_memory_client_error,
    get_search_limit,
    reset_memory_client,
)


class TestGetLlmConfig:
    """Tests for _get_llm_config function."""

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    def test_returns_empty_when_no_model_or_openrouter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return empty dict when no model or openrouter key."""
        monkeypatch.delenv("MEM0_LLM_MODEL", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = _get_llm_config()

        assert result == {}

    def test_uses_litellm_openrouter_default_when_no_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use LiteLLM and OpenRouter default when no MEM0_LLM_MODEL."""
        monkeypatch.delenv("MEM0_LLM_MODEL", raising=False)
        monkeypatch.delenv("MEM0_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        result = _get_llm_config()

        assert result["provider"] == "litellm"
        assert result["config"]["model"] == "openrouter/openai/gpt-oss-20b"
        assert result["config"]["api_key"] == "test_key"

    def test_uses_custom_model_and_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use custom model and provider when set."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "custom-model")
        monkeypatch.setenv("MEM0_LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = _get_llm_config()

        assert result["provider"] == "openai"
        assert result["config"]["model"] == "custom-model"

    def test_uses_mem0_llm_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should prefer MEM0_LLM_API_KEY over OPENROUTER_API_KEY."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_API_KEY", "mem0_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")

        result = _get_llm_config()

        assert result["config"]["api_key"] == "mem0_key"

    def test_includes_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should include temperature when set."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_TEMPERATURE", "0.5")

        result = _get_llm_config()

        assert result["config"]["temperature"] == 0.5

    def test_invalid_temperature_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning for invalid temperature."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_TEMPERATURE", "invalid")

        result = _get_llm_config()

        assert "temperature" not in result["config"]
        assert "Invalid MEM0_LLM_TEMPERATURE" in caplog.text

    def test_includes_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should include max_tokens when set."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_MAX_TOKENS", "2000")

        result = _get_llm_config()

        assert result["config"]["max_tokens"] == 2000

    def test_invalid_max_tokens_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning for invalid max_tokens."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_MAX_TOKENS", "invalid")

        result = _get_llm_config()

        assert "max_tokens" not in result["config"]
        assert "Invalid MEM0_LLM_MAX_TOKENS" in caplog.text


class TestGetEmbedderConfig:
    """Tests for _get_embedder_config function."""

    def test_uses_gemini_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use Gemini embedding defaults when not set."""
        monkeypatch.delenv("MEM0_EMBEDDER_PROVIDER", raising=False)
        monkeypatch.delenv("MEM0_EMBEDDER_MODEL", raising=False)
        monkeypatch.delenv("MEM0_EMBEDDER_DIMS", raising=False)

        result = _get_embedder_config()

        assert result["provider"] == "gemini"
        assert result["config"]["model"] == "gemini-embedding-001"
        assert result["config"]["embedding_dims"] == 768

    def test_uses_google_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should pass GOOGLE_API_KEY to the Gemini embedder."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.delenv("MEM0_EMBEDDER_API_KEY", raising=False)

        result = _get_embedder_config()

        assert result["config"]["api_key"] == "google_key"

    def test_uses_mem0_embedder_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should prefer MEM0_EMBEDDER_API_KEY over GOOGLE_API_KEY."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("MEM0_EMBEDDER_API_KEY", "embedder_key")

        result = _get_embedder_config()

        assert result["config"]["api_key"] == "embedder_key"

    def test_uses_custom_provider_and_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use custom embedder provider and model when set."""
        monkeypatch.setenv("MEM0_EMBEDDER_PROVIDER", "openai")
        monkeypatch.setenv("MEM0_EMBEDDER_MODEL", "text-embedding-3-small")

        result = _get_embedder_config()

        assert result["provider"] == "openai"
        assert result["config"]["model"] == "text-embedding-3-small"

    def test_includes_dims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should include embedding_dims when set."""
        monkeypatch.setenv("MEM0_EMBEDDER_DIMS", "512")

        result = _get_embedder_config()

        assert result["config"]["embedding_dims"] == 512

    def test_invalid_dims_logs_warning_and_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning and fall back for invalid dimensions."""
        monkeypatch.setenv("MEM0_EMBEDDER_DIMS", "invalid")

        result = _get_embedder_config()

        assert result["config"]["embedding_dims"] == 768
        assert "Invalid MEM0_EMBEDDER_DIMS" in caplog.text


class TestGetVectorStoreConfig:
    """Tests for _get_vector_store_config function."""

    def test_uses_local_qdrant_only_when_cloud_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use local Qdrant path when Cloud URL/API key are absent."""
        monkeypatch.delenv("MEM0_QDRANT_URL", raising=False)
        monkeypatch.delenv("MEM0_QDRANT_API_KEY", raising=False)
        monkeypatch.delenv("MEM0_QDRANT_HOST", raising=False)
        monkeypatch.delenv("MEM0_QDRANT_PORT", raising=False)

        result = _get_vector_store_config()

        assert result["provider"] == "qdrant"
        assert result["config"]["path"] == "./data/qdrant"
        assert result["config"]["collection_name"] == "blacki_memories_gemini_768"
        assert result["config"]["embedding_model_dims"] == 768

    def test_uses_qdrant_cloud_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use Qdrant Cloud when URL and API key are set."""
        monkeypatch.setenv("MEM0_QDRANT_URL", "https://cluster.qdrant.io")
        monkeypatch.setenv("MEM0_QDRANT_API_KEY", "qdrant_key")

        result = _get_vector_store_config()

        assert result["provider"] == "qdrant"
        assert result["config"]["url"] == "https://cluster.qdrant.io"
        assert result["config"]["api_key"] == "qdrant_key"
        assert "path" not in result["config"]

    def test_requires_qdrant_cloud_url_and_api_key_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should reject partial Qdrant Cloud configuration."""
        monkeypatch.setenv("MEM0_QDRANT_URL", "https://cluster.qdrant.io")
        monkeypatch.delenv("MEM0_QDRANT_API_KEY", raising=False)

        with pytest.raises(ValueError, match="MEM0_QDRANT_URL"):
            _get_vector_store_config()

    def test_uses_remote_qdrant_when_host_port_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use remote Qdrant server when host and port are set."""
        monkeypatch.delenv("MEM0_QDRANT_URL", raising=False)
        monkeypatch.delenv("MEM0_QDRANT_API_KEY", raising=False)
        monkeypatch.setenv("MEM0_QDRANT_HOST", "localhost")
        monkeypatch.setenv("MEM0_QDRANT_PORT", "6333")

        result = _get_vector_store_config()

        assert result["provider"] == "qdrant"
        assert result["config"]["host"] == "localhost"
        assert result["config"]["port"] == 6333

    def test_qdrant_dimensions_match_embedder_dimensions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use the embedder dimensions for the vector store."""
        monkeypatch.setenv("MEM0_EMBEDDER_DIMS", "1536")

        result = _get_vector_store_config()

        assert result["config"]["embedding_model_dims"] == 1536


class TestBuildOssConfig:
    """Tests for _build_oss_config function."""

    def test_builds_full_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should build full OSS config from environment."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
        monkeypatch.setenv("MEM0_COLLECTION_NAME", "test_collection")
        monkeypatch.setenv("MEM0_QDRANT_URL", "https://cluster.qdrant.io")
        monkeypatch.setenv("MEM0_QDRANT_API_KEY", "qdrant_key")

        result = _build_oss_config()

        assert result["llm"]["provider"] == "litellm"
        assert result["embedder"]["provider"] == "gemini"
        assert result["vector_store"]["provider"] == "qdrant"
        assert result["vector_store"]["config"]["collection_name"] == "test_collection"


class TestGetMemoryClient:
    """Tests for get_memory_client function."""

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    def test_returns_oss_client_even_with_mem0_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use OSS Memory instead of Mem0 Platform."""
        monkeypatch.setenv("MEM0_API_KEY", "platform_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        mock_client = MagicMock()
        mock_memory = MagicMock()
        mock_memory.from_config.return_value = mock_client
        with patch("mem0.Memory", mock_memory):
            result = get_memory_client()

        assert result is mock_client

    def test_returns_oss_client_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return Memory when MEM0_API_KEY is not set."""
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        mock_client = MagicMock()
        mock_memory = MagicMock()
        mock_memory.from_config.return_value = mock_client
        with patch("mem0.Memory", mock_memory):
            result = get_memory_client()

        assert result is mock_client

    def test_returns_none_on_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return None when OSS memory cannot initialize."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        mock_memory = MagicMock()
        mock_memory.from_config.side_effect = ValueError("bad config")
        with patch("mem0.Memory", mock_memory):
            result = get_memory_client()

        assert result is None
        assert (
            get_memory_client_error()
            == "Memory backend failed to initialize: bad config"
        )

    def test_describes_qdrant_dns_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should explain Qdrant host resolution failures."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
        monkeypatch.setenv(
            "MEM0_QDRANT_URL",
            "https://abcdef1234567890.qdrant.io",
        )
        monkeypatch.setenv("MEM0_QDRANT_API_KEY", "qdrant_key")

        mock_memory = MagicMock()
        mock_memory.from_config.side_effect = RuntimeError(
            "[Errno 8] nodename nor servname provided, or not known"
        )
        with patch("mem0.Memory", mock_memory):
            result = get_memory_client()

        error = get_memory_client_error()

        assert result is None
        assert error is not None
        assert "MEM0_QDRANT_URL host" in error
        assert "does not resolve" in error

    def test_caches_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should cache the client instance."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        mock_client = MagicMock()
        mock_memory = MagicMock()
        mock_memory.from_config.return_value = mock_client
        with patch("mem0.Memory", mock_memory):
            client1 = get_memory_client()
            client2 = get_memory_client()

        mock_memory.from_config.assert_called_once()
        assert client1 is client2


class TestGetDefaultUserId:
    """Tests for get_default_user_id function."""

    def test_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return 'default' when MEM0_USER_ID is not set."""
        monkeypatch.delenv("MEM0_USER_ID", raising=False)

        assert get_default_user_id() == "default"

    def test_returns_custom_user_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return custom user ID when set."""
        monkeypatch.setenv("MEM0_USER_ID", "custom_user")

        assert get_default_user_id() == "custom_user"


class TestGetSearchLimit:
    """Tests for get_search_limit function."""

    def test_returns_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return 5 when MEM0_SEARCH_LIMIT is not set."""
        monkeypatch.delenv("MEM0_SEARCH_LIMIT", raising=False)

        assert get_search_limit() == 5

    def test_returns_custom_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return custom limit when set."""
        monkeypatch.setenv("MEM0_SEARCH_LIMIT", "10")

        assert get_search_limit() == 10

    def test_returns_default_on_invalid_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return 5 when MEM0_SEARCH_LIMIT is invalid."""
        monkeypatch.setenv("MEM0_SEARCH_LIMIT", "invalid")

        assert get_search_limit() == 5


class TestResetMemoryClient:
    """Tests for reset_memory_client function."""

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    def test_resets_client_and_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should reset client to None and clear any error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        mock_client = MagicMock()
        mock_memory = MagicMock()
        mock_memory.from_config.return_value = mock_client
        with patch("mem0.Memory", mock_memory):
            client = get_memory_client()
            assert client is not None

        reset_memory_client()

        assert get_memory_client_error() is None
