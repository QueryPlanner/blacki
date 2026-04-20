"""Tests for OpenSandbox config module."""

import os
from datetime import timedelta
from unittest.mock import patch

import pytest

from blacki.sandbox.config import (
    SANDBOX_STATE_KEY,
    SandboxConfig,
    load_sandbox_config,
)


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SandboxConfig()

        assert config.enabled is False
        assert config.domain == "localhost:9090"
        assert config.api_key is None
        assert config.timeout_minutes == 30
        assert config.memory_limit == "512Mi"
        assert config.cpu_limit == "0.5"
        assert config.image == "opensandbox/code-interpreter:v1.0.2"

    def test_timeout_property(self) -> None:
        """Test timeout property returns timedelta."""
        config = SandboxConfig(timeout_minutes=45)

        assert config.timeout == timedelta(minutes=45)

    def test_resource_property(self) -> None:
        """Test resource property returns dict."""
        config = SandboxConfig(cpu_limit="1", memory_limit="1Gi")

        assert config.resource == {"cpu": "1", "memory": "1Gi"}

    def test_validate_memory_limit_valid(self) -> None:
        """Test valid memory limit values."""
        valid_limits = ["512Mi", "1Gi", "256M", "2G"]
        for limit in valid_limits:
            config = SandboxConfig(memory_limit=limit)
            assert config.memory_limit == limit

    def test_validate_memory_limit_invalid(self) -> None:
        """Test invalid memory limit raises error."""
        with pytest.raises(ValueError, match="memory_limit must end with"):
            SandboxConfig(memory_limit="invalid")

    def test_validate_cpu_limit_valid(self) -> None:
        """Test valid CPU limit values."""
        valid_limits = ["0.5", "1", "2", "0.25"]
        for limit in valid_limits:
            config = SandboxConfig(cpu_limit=limit)
            assert config.cpu_limit == limit

    def test_validate_cpu_limit_invalid(self) -> None:
        """Test invalid CPU limit raises error."""
        with pytest.raises(ValueError, match="cpu_limit must be a number"):
            SandboxConfig(cpu_limit="invalid")

    def test_timeout_minutes_bounds(self) -> None:
        """Test timeout minutes bounds."""
        with pytest.raises(ValueError):
            SandboxConfig(timeout_minutes=0)

        with pytest.raises(ValueError):
            SandboxConfig(timeout_minutes=1441)


class TestLoadSandboxConfig:
    """Tests for load_sandbox_config."""

    def test_load_from_environment(self) -> None:
        """Test loading config from environment variables."""
        env = {
            "SANDBOX_ENABLED": "true",
            "SANDBOX_DOMAIN": "sandbox.example.com:8080",
            "SANDBOX_API_KEY": "test-key",
            "SANDBOX_TIMEOUT_MINUTES": "60",
            "SANDBOX_MEMORY_LIMIT": "1Gi",
            "SANDBOX_CPU_LIMIT": "1",
            "SANDBOX_IMAGE": "custom/image:latest",
        }

        with patch.dict(os.environ, env, clear=False):
            config = load_sandbox_config()

        assert config.enabled is True
        assert config.domain == "sandbox.example.com:8080"
        assert config.api_key == "test-key"
        assert config.timeout_minutes == 60
        assert config.memory_limit == "1Gi"
        assert config.cpu_limit == "1"
        assert config.image == "custom/image:latest"

    def test_enabled_variations(self) -> None:
        """Test various SANDBOX_ENABLED values."""
        true_values = ["true", "True", "TRUE", "1", "yes"]
        for val in true_values:
            with patch.dict(os.environ, {"SANDBOX_ENABLED": val}, clear=False):
                config = load_sandbox_config()
                assert config.enabled is True, f"Failed for {val}"

        false_values = ["false", "False", "FALSE", "0", "no", ""]
        for val in false_values:
            with patch.dict(os.environ, {"SANDBOX_ENABLED": val}, clear=False):
                config = load_sandbox_config()
                assert config.enabled is False, f"Failed for {val}"

    def test_empty_api_key_becomes_none(self) -> None:
        """Test empty API key becomes None."""
        with patch.dict(os.environ, {"SANDBOX_API_KEY": ""}, clear=False):
            config = load_sandbox_config()
            assert config.api_key is None


class TestConstants:
    """Tests for module constants."""

    def test_sandbox_state_key(self) -> None:
        """Test sandbox state key constant."""
        assert SANDBOX_STATE_KEY == "__sandbox_id__"
