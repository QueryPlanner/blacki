"""Tests for the tool registry."""

from pathlib import Path
from unittest.mock import patch

from blacki.registry import ToolConfig, build_tool_config_from_env, build_tools


class TestToolConfig:
    """Tests for ToolConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have None defaults for all optional fields."""
        config = ToolConfig()

        assert config.brave_search_api_key is None
        assert config.database_url is None
        assert config.sandbox_enabled is False
        assert config.skills_dir is None

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        skills_path = Path("/tmp/skills")
        config = ToolConfig(
            brave_search_api_key="test-key",
            database_url="postgres://localhost/test",
            sandbox_enabled=True,
            skills_dir=skills_path,
        )

        assert config.brave_search_api_key == "test-key"
        assert config.database_url == "postgres://localhost/test"
        assert config.sandbox_enabled is True
        assert config.skills_dir == skills_path


class TestBuildTools:
    """Tests for build_tools function."""

    def test_empty_config_returns_memory_tools(self) -> None:
        """Should return memory tools even with empty config."""
        config = ToolConfig()
        tools = build_tools(config)

        assert len(tools) == 8

    def test_brave_search_tools_added(self) -> None:
        """Should add Brave Search tools when API key provided."""
        config = ToolConfig(brave_search_api_key="test-key")

        tools = build_tools(config)

        assert len(tools) == 9

    def test_database_tools_added(self) -> None:
        """Should add database-backed tools when database URL provided."""
        config = ToolConfig(database_url="postgres://localhost/test")

        tools = build_tools(config)

        assert len(tools) > 7

    def test_sandbox_tools_added(self) -> None:
        """Should add sandbox tools when enabled."""
        config = ToolConfig(sandbox_enabled=True)

        tools = build_tools(config)

        assert len(tools) == 14

    def test_weather_tools_disabled(self) -> None:
        """Should not add weather tools when disabled."""
        config = ToolConfig(weather_enabled=False)

        tools = build_tools(config)

        assert len(tools) == 6

    def test_all_tools_with_full_config(self) -> None:
        """Should include all tools with full configuration."""
        config = ToolConfig(
            brave_search_api_key="test-key",
            database_url="postgres://localhost/test",
            sandbox_enabled=True,
            skills_dir=Path(__file__).parent.parent / "src" / "blacki" / "skills",
        )

        tools = build_tools(config)

        assert len(tools) > 15

    def test_build_brave_search_tools_import_error(self) -> None:
        """Should handle ImportError gracefully."""
        with (
            patch.dict("sys.modules", {"blacki.tools": None}),
            patch("blacki.tools.brave_search", side_effect=ImportError("test")),
        ):
            config = ToolConfig(brave_search_api_key="test-key")
            tools = build_tools(config)

            assert len(tools) == 8


class TestBuildToolConfigFromEnv:
    """Tests for build_tool_config_from_env function."""

    def test_empty_env(self) -> None:
        """Should return default config with empty env."""
        with patch.dict("os.environ", {}, clear=True):
            config = build_tool_config_from_env()

            assert config.brave_search_api_key is None
            assert config.database_url is None
            assert config.sandbox_enabled is False
            assert config.skills_dir is not None

    def test_brave_search_api_key_from_env(self) -> None:
        """Should read BRAVE_SEARCH_API_KEY from env."""
        with patch.dict(
            "os.environ", {"BRAVE_SEARCH_API_KEY": "test-api-key"}, clear=False
        ):
            config = build_tool_config_from_env()

            assert config.brave_search_api_key == "test-api-key"

    def test_brave_search_api_key_stripped(self) -> None:
        """Should strip whitespace from BRAVE_SEARCH_API_KEY."""
        with patch.dict(
            "os.environ", {"BRAVE_SEARCH_API_KEY": "  test-key  "}, clear=False
        ):
            config = build_tool_config_from_env()

            assert config.brave_search_api_key == "test-key"

    def test_brave_search_api_key_empty_string_becomes_none(self) -> None:
        """Should convert empty string to None."""
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
            config = build_tool_config_from_env()

            assert config.brave_search_api_key is None

    def test_database_url_from_env(self) -> None:
        """Should read DATABASE_URL from env."""
        with patch.dict(
            "os.environ", {"DATABASE_URL": "postgres://localhost/test"}, clear=False
        ):
            config = build_tool_config_from_env()

            assert config.database_url == "postgres://localhost/test"

    def test_sandbox_enabled_from_env_true(self) -> None:
        """Should enable sandbox when SANDBOX_ENABLED is true."""
        for value in ["true", "True", "TRUE", "1", "yes"]:
            with patch.dict("os.environ", {"SANDBOX_ENABLED": value}, clear=False):
                config = build_tool_config_from_env()
                assert config.sandbox_enabled is True, f"Failed for value: {value}"

    def test_sandbox_enabled_from_env_false(self) -> None:
        """Should not enable sandbox when SANDBOX_ENABLED is false."""
        for value in ["false", "False", "FALSE", "0", "no", ""]:
            with patch.dict("os.environ", {"SANDBOX_ENABLED": value}, clear=False):
                config = build_tool_config_from_env()
                assert config.sandbox_enabled is False, f"Failed for value: {value}"

    def test_skills_dir_always_set(self) -> None:
        """Should always set skills_dir to package skills directory."""
        with patch.dict("os.environ", {}, clear=True):
            config = build_tool_config_from_env()

            assert config.skills_dir is not None
            assert config.skills_dir.name == "skills"


class TestBuildBraveSearchTools:
    """Tests for _build_brave_search_tools."""

    def test_returns_tool_when_available(self) -> None:
        """Should return brave_search tool when available."""
        from blacki.registry import _build_brave_search_tools

        tools = _build_brave_search_tools()

        assert len(tools) == 1

    def test_returns_empty_on_import_error(self) -> None:
        """Should return empty list on ImportError."""
        from blacki import registry

        with (
            patch.object(
                registry, "_build_brave_search_tools", side_effect=ImportError("test")
            ),
            patch("blacki.tools.brave_search", side_effect=ImportError("test")),
        ):
            pass


class TestBuildReminderTools:
    """Tests for _build_reminder_tools."""

    def test_returns_tools_when_available(self) -> None:
        """Should return reminder tools when available."""
        from blacki.registry import _build_reminder_tools

        tools = _build_reminder_tools()

        assert len(tools) == 3


class TestBuildCalorieTools:
    """Tests for _build_calorie_tools."""

    def test_returns_tools_when_available(self) -> None:
        """Should return calorie tools when available."""
        from blacki.registry import _build_calorie_tools

        tools = _build_calorie_tools()

        assert len(tools) == 5


class TestBuildWorkoutTools:
    """Tests for _build_workout_tools."""

    def test_returns_tools_when_available(self) -> None:
        """Should return workout tools when available."""
        from blacki.registry import _build_workout_tools

        tools = _build_workout_tools()

        assert len(tools) == 7


class TestBuildSandboxTools:
    """Tests for _build_sandbox_tools."""

    def test_returns_tools_when_available(self) -> None:
        """Should return sandbox tools when available."""
        from blacki.registry import _build_sandbox_tools

        tools = _build_sandbox_tools()

        assert len(tools) == 6


class TestBuildMemoryTools:
    """Tests for _build_memory_tools."""

    def test_returns_tools_when_available(self) -> None:
        """Should return memory tools when available."""
        from blacki.registry import _build_memory_tools

        tools = _build_memory_tools()

        assert len(tools) == 6


class TestBuildSkillTools:
    """Tests for _build_skill_tools."""

    def test_returns_empty_for_nonexistent_dir(self) -> None:
        """Should return empty list for non-existent skills directory."""
        from blacki.registry import _build_skill_tools

        tools = _build_skill_tools(Path("/nonexistent/skills"))

        assert tools == []
