"""Tests for MCP Skill Toolset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from google.adk.skills.models import Frontmatter, Resources, Skill
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext

from blacki.skills.mcp_skill_toolset import (
    SKILL_STATE_PREFIX,
    McpSkillToolset,
    _LoadSkillResourceTool,
    _LoadSkillTool,
    load_skill_from_dir,
)


@pytest.fixture
def sample_skill() -> Skill:
    """Create a sample skill for testing."""
    return Skill(
        frontmatter=Frontmatter(
            name="test_skill",
            description="A test skill for unit testing",
            metadata={"version": "1.0.0"},
        ),
        instructions="# Test Skill\n\nThis is a test skill.",
        resources=Resources(),
    )


@pytest.fixture
def second_skill() -> Skill:
    """Create a second skill for testing multiple skills."""
    return Skill(
        frontmatter=Frontmatter(
            name="second_skill",
            description="A second test skill",
        ),
        instructions="# Second Skill\n\nThis is another test skill.",
        resources=Resources(),
    )


@pytest.fixture
def mock_mcp_toolset() -> McpToolset:
    """Create a mock MCP toolset."""
    mock_toolset = create_autospec(McpToolset, spec_set=True, instance=True)
    mock_toolset.get_tools = AsyncMock(return_value=[])
    mock_toolset.close = AsyncMock()
    return mock_toolset  # type: ignore[no-any-return]


@pytest.fixture
def mock_tool_context() -> ToolContext:
    """Create a mock tool context with state."""
    mock_context = create_autospec(ToolContext, spec_set=True, instance=True)
    mock_context.state = {}
    return mock_context  # type: ignore[no-any-return]


class TestLoadSkillFromDir:
    """Tests for load_skill_from_dir function."""

    def test_loads_skill_from_valid_directory(self, tmp_path: Path) -> None:
        """Test loading a valid skill from directory."""
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: test_skill
description: Test skill description
version: 1.0.0
---

# Test Skill Instructions

This is the skill body.
"""
        )

        skill = load_skill_from_dir(skill_dir)

        assert skill is not None
        assert skill.frontmatter.name == "test_skill"
        assert skill.frontmatter.description == "Test skill description"
        assert "Test Skill Instructions" in skill.instructions

    def test_returns_none_for_missing_skill_md(self, tmp_path: Path) -> None:
        """Test returns None when SKILL.md is missing."""
        skill_dir = tmp_path / "no_skill"
        skill_dir.mkdir()

        skill = load_skill_from_dir(skill_dir)

        assert skill is None

    def test_returns_none_for_missing_frontmatter(self, tmp_path: Path) -> None:
        """Test returns None when frontmatter is missing."""
        skill_dir = tmp_path / "no_frontmatter"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# No frontmatter here")

        skill = load_skill_from_dir(skill_dir)

        assert skill is None

    def test_uses_directory_name_as_default(self, tmp_path: Path) -> None:
        """Test uses directory name when name not in frontmatter."""
        skill_dir = tmp_path / "my_custom_skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
description: Skill without name
---

# Instructions
"""
        )

        skill = load_skill_from_dir(skill_dir)

        assert skill is not None
        assert skill.frontmatter.name == "my_custom_skill"

    def test_handles_list_metadata(self, tmp_path: Path) -> None:
        """Test handles list values in frontmatter metadata."""
        skill_dir = tmp_path / "list_skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: list_skill
description: Skill with list metadata
tags:
  - tag1
  - tag2
---

# Instructions
"""
        )

        skill = load_skill_from_dir(skill_dir)

        assert skill is not None
        assert "tags" in skill.frontmatter.metadata
        assert "tag1, tag2" in skill.frontmatter.metadata["tags"]

    def test_returns_none_for_invalid_frontmatter_format(self, tmp_path: Path) -> None:
        """Test returns None when frontmatter format is invalid."""
        skill_dir = tmp_path / "invalid_format"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
only one delimiter
"""
        )

        skill = load_skill_from_dir(skill_dir)

        assert skill is None

    def test_returns_none_for_empty_frontmatter(self, tmp_path: Path) -> None:
        """Test returns None when frontmatter is empty."""
        skill_dir = tmp_path / "empty_frontmatter"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
---

# Instructions
"""
        )

        skill = load_skill_from_dir(skill_dir)

        assert skill is None

    def test_handles_numeric_metadata(self, tmp_path: Path) -> None:
        """Test handles numeric values in frontmatter metadata."""
        skill_dir = tmp_path / "numeric_skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: numeric_skill
description: Skill with numeric metadata
count: 42
pi: 3.14
enabled: true
---

# Instructions
"""
        )

        skill = load_skill_from_dir(skill_dir)

        assert skill is not None
        assert skill.frontmatter.metadata["count"] == "42"
        assert skill.frontmatter.metadata["pi"] == "3.14"
        assert skill.frontmatter.metadata["enabled"] == "True"

    def test_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Test handles invalid YAML in frontmatter."""
        skill_dir = tmp_path / "invalid_yaml"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: [invalid
description: broken yaml
---

# Instructions
"""
        )

        skill = load_skill_from_dir(skill_dir)

        assert skill is None

    def test_skips_non_primitive_metadata(self, tmp_path: Path) -> None:
        """Test skips non-primitive values in frontmatter metadata."""
        skill_dir = tmp_path / "non_primitive"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: non_primitive_skill
description: Skill with non-primitive metadata
nested:
  key: value
---

# Instructions
"""
        )

        skill = load_skill_from_dir(skill_dir)

        assert skill is not None
        assert "nested" not in skill.frontmatter.metadata


class TestLoadSkillTool:
    """Tests for _LoadSkillTool."""

    def test_tool_name_and_description(self, sample_skill: Skill) -> None:
        """Test tool has correct name and description."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillTool(toolset)

        assert tool.name == "load_skill"
        assert "instructions" in tool.description.lower()

    def test_get_declaration(self, sample_skill: Skill) -> None:
        """Test _get_declaration returns valid declaration with name param."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillTool(toolset)

        declaration = tool._get_declaration()

        assert declaration is not None
        assert declaration.name == "load_skill"
        props = declaration.parameters_json_schema.get("properties", {})
        assert "name" in props
        assert "required" in declaration.parameters_json_schema

    @pytest.mark.asyncio
    async def test_run_async_returns_skill_data_and_activates(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test run_async returns skill data and activates MCP tools."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillTool(toolset)

        result = await tool.run_async(
            args={"name": "test_skill"}, tool_context=mock_tool_context
        )

        assert result["skill_name"] == "test_skill"
        assert "instructions" in result
        assert "frontmatter" in result
        state_key = f"{SKILL_STATE_PREFIX}test_skill"
        assert mock_tool_context.state[state_key] is True

    @pytest.mark.asyncio
    async def test_missing_name_returns_error(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test missing name returns error with available skills."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillTool(toolset)

        result = await tool.run_async(args={}, tool_context=mock_tool_context)

        assert "error" in result
        assert result["error_code"] == "MISSING_SKILL_NAME"
        assert "available_skills" in result
        assert "test_skill" in result["available_skills"]

    @pytest.mark.asyncio
    async def test_unknown_skill_name_returns_error(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test unknown skill name returns error with available skills."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillTool(toolset)

        result = await tool.run_async(
            args={"name": "unknown_skill"}, tool_context=mock_tool_context
        )

        assert "error" in result
        assert result["error_code"] == "SKILL_NOT_FOUND"
        assert "available_skills" in result
        assert "test_skill" in result["available_skills"]

    @pytest.mark.asyncio
    async def test_loads_correct_skill_with_multiple_skills(
        self, sample_skill: Skill, second_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test loads correct skill when multiple skills exist."""
        toolset = McpSkillToolset(
            skills=[
                (sample_skill, MagicMock(spec=McpToolset)),
                (second_skill, MagicMock(spec=McpToolset)),
            ]
        )
        tool = _LoadSkillTool(toolset)

        result = await tool.run_async(
            args={"name": "second_skill"}, tool_context=mock_tool_context
        )

        assert result["skill_name"] == "second_skill"
        assert "Second Skill" in result["instructions"]


class TestLoadSkillResourceTool:
    """Tests for _LoadSkillResourceTool."""

    def test_tool_name_and_description(self, sample_skill: Skill) -> None:
        """Test tool has correct name and description."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        assert tool.name == "load_skill_resource"
        assert "resource" in tool.description.lower()

    def test_get_declaration(self, sample_skill: Skill) -> None:
        """Test _get_declaration returns valid declaration."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        declaration = tool._get_declaration()

        assert declaration is not None
        assert declaration.name == "load_skill_resource"
        assert declaration.parameters_json_schema is not None
        props = declaration.parameters_json_schema.get("properties", {})
        assert "name" in props
        assert "path" in props

    @pytest.mark.asyncio
    async def test_missing_name_returns_error(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test missing name returns error."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        result = await tool.run_async(
            args={"path": "references/test.md"}, tool_context=mock_tool_context
        )

        assert "error" in result
        assert result["error_code"] == "MISSING_SKILL_NAME"

    @pytest.mark.asyncio
    async def test_missing_path_returns_error(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test missing path returns error."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        result = await tool.run_async(
            args={"name": "test_skill"}, tool_context=mock_tool_context
        )

        assert "error" in result
        assert result["error_code"] == "MISSING_RESOURCE_PATH"

    @pytest.mark.asyncio
    async def test_unknown_skill_name_returns_error(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test unknown skill name returns error."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        result = await tool.run_async(
            args={"name": "unknown", "path": "references/test.md"},
            tool_context=mock_tool_context,
        )

        assert "error" in result
        assert result["error_code"] == "SKILL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_invalid_path_returns_error(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test invalid path prefix returns error."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        result = await tool.run_async(
            args={"name": "test_skill", "path": "invalid/path.txt"},
            tool_context=mock_tool_context,
        )

        assert "error" in result
        assert result["error_code"] == "INVALID_RESOURCE_PATH"

    @pytest.mark.asyncio
    async def test_missing_reference_returns_error(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test missing reference returns error."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        result = await tool.run_async(
            args={"name": "test_skill", "path": "references/missing.md"},
            tool_context=mock_tool_context,
        )

        assert "error" in result
        assert result["error_code"] == "RESOURCE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_missing_asset_returns_error(
        self, sample_skill: Skill, mock_tool_context: ToolContext
    ) -> None:
        """Test missing asset returns error."""
        toolset = McpSkillToolset(skills=[(sample_skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        result = await tool.run_async(
            args={"name": "test_skill", "path": "assets/missing.txt"},
            tool_context=mock_tool_context,
        )

        assert "error" in result
        assert result["error_code"] == "RESOURCE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_loads_existing_asset(self, mock_tool_context: ToolContext) -> None:
        """Test loading an existing asset."""
        skill = Skill(
            frontmatter=Frontmatter(
                name="test_skill",
                description="Test skill",
            ),
            instructions="Test instructions",
            resources=Resources(
                assets={"template.txt": "Hello, World!"},
            ),
        )
        toolset = McpSkillToolset(skills=[(skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        result = await tool.run_async(
            args={"name": "test_skill", "path": "assets/template.txt"},
            tool_context=mock_tool_context,
        )

        assert result["skill_name"] == "test_skill"
        assert result["path"] == "assets/template.txt"
        assert result["content"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_loads_existing_reference(
        self, mock_tool_context: ToolContext
    ) -> None:
        """Test loading an existing reference."""
        skill = Skill(
            frontmatter=Frontmatter(
                name="test_skill",
                description="Test skill",
            ),
            instructions="Test instructions",
            resources=Resources(
                references={"guide.md": "# Guide\n\nThis is a guide."},
            ),
        )
        toolset = McpSkillToolset(skills=[(skill, MagicMock(spec=McpToolset))])
        tool = _LoadSkillResourceTool(toolset)

        result = await tool.run_async(
            args={"name": "test_skill", "path": "references/guide.md"},
            tool_context=mock_tool_context,
        )

        assert result["skill_name"] == "test_skill"
        assert result["path"] == "references/guide.md"
        assert "# Guide" in result["content"]


class TestMcpSkillToolset:
    """Tests for McpSkillToolset."""

    @pytest.mark.asyncio
    async def test_get_tools_returns_base_tools(
        self, sample_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test get_tools returns base tools (load_skill, load_skill_resource)."""
        toolset = McpSkillToolset(skills=[(sample_skill, mock_mcp_toolset)])

        tools = await toolset.get_tools()

        tool_names = [t.name for t in tools]
        assert "load_skill" in tool_names
        assert "load_skill_resource" in tool_names
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_get_tools_with_activation(
        self, sample_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test get_tools includes MCP tools after activation."""
        mock_mcp_tool = MagicMock(spec=BaseTool)
        mock_mcp_tool.name = "mcp_tool_1"
        mock_mcp_toolset.get_tools = AsyncMock(return_value=[mock_mcp_tool])

        toolset = McpSkillToolset(skills=[(sample_skill, mock_mcp_toolset)])

        mock_readonly_context = MagicMock()
        mock_readonly_context.state = {f"{SKILL_STATE_PREFIX}test_skill": True}

        tools = await toolset.get_tools(readonly_context=mock_readonly_context)

        tool_names = [t.name for t in tools]
        assert "mcp_tool_1" in tool_names

    @pytest.mark.asyncio
    async def test_get_tools_handles_mcp_failure(
        self, sample_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test get_tools handles MCP toolset failure gracefully."""
        mock_mcp_toolset.get_tools = AsyncMock(side_effect=RuntimeError("MCP failed"))

        toolset = McpSkillToolset(skills=[(sample_skill, mock_mcp_toolset)])

        mock_readonly_context = MagicMock()
        mock_readonly_context.state = {f"{SKILL_STATE_PREFIX}test_skill": True}

        tools = await toolset.get_tools(readonly_context=mock_readonly_context)

        tool_names = [t.name for t in tools]
        assert "load_skill" in tool_names
        assert "mcp_tool_1" not in tool_names

    @pytest.mark.asyncio
    async def test_close_calls_mcp_close(
        self, sample_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test close calls MCP toolset close."""
        toolset = McpSkillToolset(skills=[(sample_skill, mock_mcp_toolset)])

        await toolset.close()

        mock_mcp_toolset.close.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_close_skips_none_mcp_toolsets(self, sample_skill: Skill) -> None:
        """Test close handles skills without MCP toolsets."""
        toolset = McpSkillToolset(skills=[(sample_skill, None)])

        await toolset.close()

    def test_list_skills_returns_frontmatter(
        self, sample_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test _list_skills returns skill frontmatter."""
        toolset = McpSkillToolset(skills=[(sample_skill, mock_mcp_toolset)])

        skills = toolset._list_skills()

        assert len(skills) == 1
        assert skills[0].name == "test_skill"

    def test_get_skill_by_name(
        self, sample_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test _get_skill returns skill by name."""
        toolset = McpSkillToolset(skills=[(sample_skill, mock_mcp_toolset)])

        skill = toolset._get_skill("test_skill")

        assert skill is not None
        assert skill.frontmatter.name == "test_skill"

    def test_get_skill_unknown_name_returns_none(
        self, sample_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test _get_skill returns None for unknown name."""
        toolset = McpSkillToolset(skills=[(sample_skill, mock_mcp_toolset)])

        skill = toolset._get_skill("unknown_skill")

        assert skill is None

    @pytest.mark.asyncio
    async def test_process_llm_request_injects_instructions(
        self, sample_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test process_llm_request injects skill instructions."""
        toolset = McpSkillToolset(skills=[(sample_skill, mock_mcp_toolset)])

        mock_llm_request = MagicMock()
        mock_llm_request.append_instructions = MagicMock()

        mock_tool_context = create_autospec(ToolContext, spec_set=True, instance=True)

        await toolset.process_llm_request(
            tool_context=mock_tool_context,
            llm_request=mock_llm_request,
        )

        mock_llm_request.append_instructions.assert_called_once()
        injected_instruction = mock_llm_request.append_instructions.call_args[0][0][0]
        assert "test_skill" in injected_instruction
        assert "load_skill" in injected_instruction

    @pytest.mark.asyncio
    async def test_multiple_skills(
        self, sample_skill: Skill, second_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test toolset with multiple skills."""
        toolset = McpSkillToolset(
            skills=[
                (sample_skill, mock_mcp_toolset),
                (second_skill, None),
            ]
        )

        skills = toolset._list_skills()
        assert len(skills) == 2
        skill_names = [s.name for s in skills]
        assert "test_skill" in skill_names
        assert "second_skill" in skill_names

    @pytest.mark.asyncio
    async def test_skill_without_mcp_toolset(self, sample_skill: Skill) -> None:
        """Test skill without MCP toolset (None)."""
        toolset = McpSkillToolset(skills=[(sample_skill, None)])

        tools = await toolset.get_tools()

        tool_names = [t.name for t in tools]
        assert "load_skill" in tool_names
        assert "load_skill_resource" in tool_names
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_activation_does_not_add_tools_for_skill_without_mcp(
        self, sample_skill: Skill
    ) -> None:
        """Test activation for skill without MCP toolset doesn't add tools."""
        toolset = McpSkillToolset(skills=[(sample_skill, None)])

        mock_readonly_context = MagicMock()
        mock_readonly_context.state = {f"{SKILL_STATE_PREFIX}test_skill": True}

        tools = await toolset.get_tools(readonly_context=mock_readonly_context)

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "load_skill" in tool_names
        assert "load_skill_resource" in tool_names

    @pytest.mark.asyncio
    async def test_close_calls_all_mcp_toolsets(
        self, sample_skill: Skill, second_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test close calls close on all MCP toolsets."""
        mock_mcp_toolset_2 = create_autospec(McpToolset, spec_set=True, instance=True)
        mock_mcp_toolset_2.get_tools = AsyncMock(return_value=[])
        mock_mcp_toolset_2.close = AsyncMock()

        toolset = McpSkillToolset(
            skills=[
                (sample_skill, mock_mcp_toolset),
                (second_skill, mock_mcp_toolset_2),
            ]
        )

        await toolset.close()

        mock_mcp_toolset.close.assert_awaited_once()  # type: ignore[attr-defined]
        mock_mcp_toolset_2.close.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_multiple_skills_activation(
        self, sample_skill: Skill, second_skill: Skill, mock_mcp_toolset: McpToolset
    ) -> None:
        """Test MCP tools are added for each activated skill."""
        mock_mcp_tool_1 = MagicMock(spec=BaseTool)
        mock_mcp_tool_1.name = "mcp_tool_1"
        mock_mcp_toolset_1 = create_autospec(McpToolset, spec_set=True, instance=True)
        mock_mcp_toolset_1.get_tools = AsyncMock(return_value=[mock_mcp_tool_1])
        mock_mcp_toolset_1.close = AsyncMock()

        mock_mcp_tool_2 = MagicMock(spec=BaseTool)
        mock_mcp_tool_2.name = "mcp_tool_2"
        mock_mcp_toolset_2 = create_autospec(McpToolset, spec_set=True, instance=True)
        mock_mcp_toolset_2.get_tools = AsyncMock(return_value=[mock_mcp_tool_2])
        mock_mcp_toolset_2.close = AsyncMock()

        toolset = McpSkillToolset(
            skills=[
                (sample_skill, mock_mcp_toolset_1),
                (second_skill, mock_mcp_toolset_2),
            ]
        )

        mock_readonly_context = MagicMock()
        mock_readonly_context.state = {
            f"{SKILL_STATE_PREFIX}test_skill": True,
            f"{SKILL_STATE_PREFIX}second_skill": True,
        }

        tools = await toolset.get_tools(readonly_context=mock_readonly_context)

        tool_names = [t.name for t in tools]
        assert "mcp_tool_1" in tool_names
        assert "mcp_tool_2" in tool_names
