"""Tests for the tool registry."""

from pathlib import Path
from unittest.mock import patch

from blacki.registry import ToolConfig, build_tool_config_from_env, build_tools


class TestToolConfig:
    """Tests for ToolConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have None defaults for all optional fields."""
        config = ToolConfig()

        assert config.exa_api_key is None
        assert config.brave_search_api_key is None
        assert config.sqlite_path is None
        assert config.sandbox_enabled is False
        assert config.skills_dir is None
        assert config.legacy_workout_tools_enabled is False

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        skills_path = Path("/tmp/skills")
        config = ToolConfig(
            exa_api_key="exa-key",
            brave_search_api_key="test-key",
            sqlite_path="/tmp/blacki.db",
            sandbox_enabled=True,
            skills_dir=skills_path,
            legacy_workout_tools_enabled=True,
        )

        assert config.exa_api_key == "exa-key"
        assert config.brave_search_api_key == "test-key"
        assert config.sqlite_path == "/tmp/blacki.db"
        assert config.sandbox_enabled is True
        assert config.skills_dir == skills_path
        assert config.legacy_workout_tools_enabled is True


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

    def test_exa_search_tools_added_first(self) -> None:
        """Should register Exa before Brave when both keys are provided."""
        config = ToolConfig(
            exa_api_key="exa-key",
            brave_search_api_key="brave-key",
        )

        tools = build_tools(config)

        assert [tool.__name__ for tool in tools[:2]] == ["exa_search", "brave_search"]
        assert len(tools) == 10

    def test_database_tools_added(self) -> None:
        """Should add database-backed tools when sqlite path provided."""
        config = ToolConfig(sqlite_path="/tmp/blacki.db")

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
            exa_api_key="exa-key",
            brave_search_api_key="test-key",
            sqlite_path="/tmp/blacki.db",
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

    def test_build_exa_search_tools_import_error(self) -> None:
        """Should omit Exa Search if its module cannot be imported."""
        from blacki.registry import _build_exa_search_tools

        with patch.dict("sys.modules", {"blacki.search": None}):
            tools = _build_exa_search_tools()

        assert tools == []


class TestBuildToolConfigFromEnv:
    """Tests for build_tool_config_from_env function."""

    def test_empty_env(self) -> None:
        """Should return default config with default sqlite path."""
        with patch.dict("os.environ", {}, clear=True):
            config = build_tool_config_from_env()

            assert config.exa_api_key is None
            assert config.brave_search_api_key is None
            assert config.sqlite_path is not None
            assert config.sqlite_path.endswith(".adk/tools.db")
            assert config.sandbox_enabled is False
            assert config.skills_dir is not None

    def test_brave_search_api_key_from_env(self) -> None:
        """Should read BRAVE_SEARCH_API_KEY from env."""
        with patch.dict(
            "os.environ", {"BRAVE_SEARCH_API_KEY": "test-api-key"}, clear=False
        ):
            config = build_tool_config_from_env()

            assert config.brave_search_api_key == "test-api-key"

    def test_exa_api_key_from_env_is_stripped(self) -> None:
        """Should read and strip EXA_API_KEY from the environment."""
        with patch.dict("os.environ", {"EXA_API_KEY": "  exa-key  "}, clear=False):
            config = build_tool_config_from_env()

            assert config.exa_api_key == "exa-key"

    def test_empty_exa_api_key_becomes_none(self) -> None:
        """Should disable Exa for an empty environment value."""
        with patch.dict("os.environ", {"EXA_API_KEY": "   "}, clear=False):
            config = build_tool_config_from_env()

            assert config.exa_api_key is None

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

    def test_sqlite_path_from_env(self) -> None:
        """Should read SQLITE_PATH from env."""
        with patch.dict("os.environ", {"SQLITE_PATH": "/tmp/blacki.db"}, clear=False):
            config = build_tool_config_from_env()

            assert config.sqlite_path == "/tmp/blacki.db"

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

    def test_legacy_workout_tools_flag_from_env(self) -> None:
        """Should hide legacy workout tools unless explicitly enabled."""
        with patch.dict(
            "os.environ", {"LEGACY_WORKOUT_TOOLS_ENABLED": "true"}, clear=False
        ):
            assert build_tool_config_from_env().legacy_workout_tools_enabled is True

        with patch.dict(
            "os.environ", {"LEGACY_WORKOUT_TOOLS_ENABLED": "false"}, clear=False
        ):
            assert build_tool_config_from_env().legacy_workout_tools_enabled is False


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


class TestBuildExaSearchTools:
    """Tests for _build_exa_search_tools."""

    def test_returns_tool_when_available(self) -> None:
        """Should return the Exa Search tool when available."""
        from blacki.registry import _build_exa_search_tools

        tools = _build_exa_search_tools()

        assert len(tools) == 1
        assert tools[0].__name__ == "exa_search"


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
        """Should expose only canonical training tools by default."""
        from blacki.registry import _build_workout_tools

        tools = _build_workout_tools()

        assert len(tools) == 8
        assert "log_training" in {tool.__name__ for tool in tools}
        assert "log_workout" not in {tool.__name__ for tool in tools}

    def test_returns_legacy_tools_only_when_enabled(self) -> None:
        """Should expose weekly split fallbacks only behind the feature flag."""
        from blacki.registry import _build_workout_tools

        tools = _build_workout_tools(include_legacy=True)

        assert len(tools) == 12
        assert "log_workout" in {tool.__name__ for tool in tools}


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

    def test_does_not_register_credentialed_gemini_cli_skill(self) -> None:
        """General sandboxes must not advertise a credential-dependent skill."""
        from blacki.registry import _build_skill_tools

        tools = _build_skill_tools(Path("src/blacki/skills"))

        assert len(tools) == 1
        assert "gemini_cli" not in tools[0]._skills


class TestBuildDeclarativeDbTools:
    """Tests for _build_declarative_db_tools."""

    def test_returns_tools_when_available(self) -> None:
        """Should return declarative database tools when available."""
        from blacki.registry import _build_declarative_db_tools

        tools = _build_declarative_db_tools()

        assert len(tools) == 7
