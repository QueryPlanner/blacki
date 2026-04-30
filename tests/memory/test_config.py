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
    get_search_limit,
    is_cloud_client,
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

    def test_uses_openrouter_default_when_no_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use openrouter default model when no MEM0_LLM_MODEL."""
        monkeypatch.delenv("MEM0_LLM_MODEL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        result = _get_llm_config()

        assert result["model"] == "openrouter/google/gemini-2.0-flash-001"
        assert result["api_key"] == "test_key"

    def test_uses_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use custom model when MEM0_LLM_MODEL is set."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "custom-model")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = _get_llm_config()

        assert result["model"] == "custom-model"

    def test_uses_mem0_llm_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should prefer MEM0_LLM_API_KEY over OPENROUTER_API_KEY."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_API_KEY", "mem0_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")

        result = _get_llm_config()

        assert result["api_key"] == "mem0_key"

    def test_includes_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should include temperature when set."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_TEMPERATURE", "0.5")

        result = _get_llm_config()

        assert result["temperature"] == 0.5

    def test_invalid_temperature_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning for invalid temperature."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_TEMPERATURE", "invalid")

        result = _get_llm_config()

        assert "temperature" not in result
        assert "Invalid MEM0_LLM_TEMPERATURE" in caplog.text

    def test_includes_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should include max_tokens when set."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_MAX_TOKENS", "2000")

        result = _get_llm_config()

        assert result["max_tokens"] == 2000

    def test_invalid_max_tokens_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning for invalid max_tokens."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("MEM0_LLM_MAX_TOKENS", "invalid")

        result = _get_llm_config()

        assert "max_tokens" not in result
        assert "Invalid MEM0_LLM_MAX_TOKENS" in caplog.text


class TestGetEmbedderConfig:
    """Tests for _get_embedder_config function."""

    def test_uses_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use default model when not set."""
        monkeypatch.delenv("MEM0_EMBEDDER_MODEL", raising=False)

        result = _get_embedder_config()

        assert result["model"] == "BAAI/bge-small-en-v1.5"

    def test_uses_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use custom model when set."""
        monkeypatch.setenv("MEM0_EMBEDDER_MODEL", "custom-embedder")

        result = _get_embedder_config()

        assert result["model"] == "custom-embedder"

    def test_includes_dims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should include embedding_dims when set."""
        monkeypatch.setenv("MEM0_EMBEDDER_DIMS", "512")

        result = _get_embedder_config()

        assert result["embedding_dims"] == 512

    def test_invalid_dims_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning for invalid dims."""
        monkeypatch.setenv("MEM0_EMBEDDER_DIMS", "invalid")

        result = _get_embedder_config()

        assert "embedding_dims" not in result
        assert "Invalid MEM0_EMBEDDER_DIMS" in caplog.text


class TestGetVectorStoreConfig:
    """Tests for _get_vector_store_config function."""

    def test_uses_embedded_mode_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use embedded Qdrant by default."""
        monkeypatch.delenv("MEM0_QDRANT_HOST", raising=False)
        monkeypatch.delenv("MEM0_QDRANT_PORT", raising=False)

        result = _get_vector_store_config()

        assert result["type"] == "qdrant"
        assert "path" in result["config"]

    def test_uses_remote_qdrant_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use remote Qdrant when host and port are set."""
        monkeypatch.setenv("MEM0_QDRANT_HOST", "localhost")
        monkeypatch.setenv("MEM0_QDRANT_PORT", "6333")

        result = _get_vector_store_config()

        assert result["type"] == "qdrant"
        assert result["config"]["host"] == "localhost"
        assert result["config"]["port"] == 6333


class TestBuildOssConfig:
    """Tests for _build_oss_config function."""

    def test_builds_full_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should build full config from environment."""
        monkeypatch.setenv("MEM0_LLM_MODEL", "test-model")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
        monkeypatch.setenv("MEM0_COLLECTION_NAME", "test_collection")

        result = _build_oss_config()

        assert "llm" in result
        assert "embedder" in result
        assert "vector_store" in result


class TestGetMemoryClient:
    """Tests for get_memory_client function."""

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    def test_returns_cloud_client_with_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return MemoryClient when MEM0_API_KEY is set."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")

        mock_client = MagicMock()
        with patch("mem0.MemoryClient", return_value=mock_client) as mock_class:
            result = get_memory_client()

            mock_class.assert_called_once_with(api_key="test_key")
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

    def test_returns_none_on_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return None when mem0ai is not installed."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")

        with patch("mem0.MemoryClient", side_effect=ImportError("mem0 not found")):
            result = get_memory_client()

            assert result is None

    def test_caches_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should cache the client instance."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")

        mock_client = MagicMock()
        with patch("mem0.MemoryClient", return_value=mock_client) as mock_class:
            client1 = get_memory_client()
            client2 = get_memory_client()

            mock_class.assert_called_once()
            assert client1 is client2


class TestIsCloudClient:
    """Tests for is_cloud_client function."""

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the memory client before each test."""
        reset_memory_client()

    def test_returns_true_for_cloud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return True when using cloud client."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")

        mock_client = MagicMock()
        with patch("mem0.MemoryClient", return_value=mock_client):
            get_memory_client()
            assert is_cloud_client() is True

    def test_returns_false_for_oss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return False when using OSS client."""
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

        mock_client = MagicMock()
        mock_memory = MagicMock()
        mock_memory.from_config.return_value = mock_client
        with patch("mem0.Memory", mock_memory):
            get_memory_client()
            assert is_cloud_client() is False


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

    def test_resets_client_and_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should reset client to None and is_cloud_client to False."""
        monkeypatch.setenv("MEM0_API_KEY", "test_key")

        mock_client = MagicMock()
        with patch("mem0.MemoryClient", return_value=mock_client):
            client = get_memory_client()
            assert client is not None
            assert is_cloud_client() is True

        reset_memory_client()

        assert is_cloud_client() is False
