"""MCP Skill Toolset with auto-activation.

This module provides a toolset that combines ADK Skills with MCP tools,
supporting automatic activation when load_skill is called. MCP tools are
automatically exposed after the skill instructions are loaded.

This implementation follows the native ADK SkillToolset pattern, supporting
multiple skills per toolset with optional MCP toolsets for each skill.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.skills import format_skills_as_xml
from google.adk.skills.models import Frontmatter, Resources, Skill
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

SKILL_STATE_PREFIX = "mcp_skill_activated_"


def load_skill_from_dir(skill_path: Path) -> Skill | None:
    """Load a Skill from a directory containing SKILL.md.

    Parses the YAML frontmatter and markdown body from SKILL.md,
    creating a Skill object with frontmatter and instructions.

    Args:
        skill_path: Path to the skill directory containing SKILL.md

    Returns:
        Skill object if SKILL.md exists and is valid, None otherwise
    """
    skill_md_path = skill_path / "SKILL.md"
    if not skill_md_path.exists():
        logger.warning("SKILL.md not found at %s", skill_md_path)
        return None

    try:
        content = skill_md_path.read_text(encoding="utf-8")

        if not content.startswith("---"):
            logger.warning("SKILL.md at %s missing frontmatter", skill_md_path)
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            logger.warning(
                "SKILL.md at %s has invalid frontmatter format", skill_md_path
            )
            return None

        frontmatter_yaml = parts[1].strip()
        instructions = parts[2].strip()

        frontmatter_data = yaml.safe_load(frontmatter_yaml)
        if not frontmatter_data:
            logger.warning("SKILL.md at %s has empty frontmatter", skill_md_path)
            return None

        name = frontmatter_data.get("name", skill_path.name)
        description = frontmatter_data.get("description", "")

        metadata: dict[str, str] = {}
        for k, v in frontmatter_data.items():
            if k in (
                "name",
                "description",
                "license",
                "compatibility",
                "allowed_tools",
            ):
                continue
            if isinstance(v, list):
                metadata[k] = ", ".join(str(item) for item in v)
            elif isinstance(v, (str, int, float, bool)):
                metadata[k] = str(v)

        frontmatter = Frontmatter(
            name=name,
            description=description,
            metadata=metadata,
        )

        return Skill(
            frontmatter=frontmatter,
            instructions=instructions,
            resources=Resources(),
        )

    except Exception:
        logger.exception("Failed to load skill from %s", skill_path)
        return None


class _LoadSkillTool(BaseTool):
    """Tool to load a skill's instructions.

    Follows the native ADK LoadSkillTool pattern with instance-level
    skill lookup via the parent toolset.
    """

    def __init__(self, toolset: McpSkillToolset) -> None:
        super().__init__(
            name="load_skill",
            description="Loads the SKILL.md instructions for a given skill.",
        )
        self._toolset = toolset

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        available_skills = list(self._toolset._skills.keys())
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            f"The name of the skill to load. "
                            f"Available: {available_skills}"
                        ),
                    },
                },
                "required": ["name"],
            },
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        skill_name = args.get("name")

        if not skill_name:
            return {
                "error": "Skill name is required.",
                "available_skills": list(self._toolset._skills.keys()),
                "error_code": "MISSING_SKILL_NAME",
            }

        skill = self._toolset._get_skill(skill_name)
        if not skill:
            return {
                "error": f"Skill '{skill_name}' not found.",
                "available_skills": list(self._toolset._skills.keys()),
                "error_code": "SKILL_NOT_FOUND",
            }

        await self._toolset._activate_skill(skill_name, tool_context)

        return {
            "skill_name": skill_name,
            "instructions": skill.instructions,
            "frontmatter": skill.frontmatter.model_dump(),
        }


class _LoadSkillResourceTool(BaseTool):
    """Tool to load resources (references or assets) from a skill.

    Follows the native ADK LoadSkillResourceTool pattern with instance-level
    skill lookup via the parent toolset.
    """

    def __init__(self, toolset: McpSkillToolset) -> None:
        super().__init__(
            name="load_skill_resource",
            description=(
                "Loads a resource file (from references/ or assets/) from within a"
                " skill."
            ),
        )
        self._toolset = toolset

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        available_skills = list(self._toolset._skills.keys())
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            f"The name of the skill. Available: {available_skills}"
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "The relative path to the resource (e.g.,"
                            " 'references/my_doc.md' or 'assets/template.txt')."
                        ),
                    },
                },
                "required": ["name", "path"],
            },
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        skill_name = args.get("name")
        resource_path = args.get("path")

        if not skill_name:
            return {
                "error": "Skill name is required.",
                "available_skills": list(self._toolset._skills.keys()),
                "error_code": "MISSING_SKILL_NAME",
            }

        if not resource_path:
            return {
                "error": "Resource path is required.",
                "error_code": "MISSING_RESOURCE_PATH",
            }

        skill = self._toolset._get_skill(skill_name)
        if not skill:
            return {
                "error": f"Skill '{skill_name}' not found.",
                "available_skills": list(self._toolset._skills.keys()),
                "error_code": "SKILL_NOT_FOUND",
            }

        content = None
        if resource_path.startswith("references/"):
            ref_name = resource_path[len("references/") :]
            content = skill.resources.get_reference(ref_name)
        elif resource_path.startswith("assets/"):
            asset_name = resource_path[len("assets/") :]
            content = skill.resources.get_asset(asset_name)
        else:
            return {
                "error": "Path must start with 'references/' or 'assets/'.",
                "error_code": "INVALID_RESOURCE_PATH",
            }

        if content is None:
            return {
                "error": (
                    f"Resource '{resource_path}' not found in skill '{skill_name}'."
                ),
                "error_code": "RESOURCE_NOT_FOUND",
            }

        return {
            "skill_name": skill_name,
            "path": resource_path,
            "content": content,
        }


class McpSkillToolset(BaseToolset):
    """A toolset for managing and interacting with agent skills with optional MCP tools.

    This toolset follows the native ADK SkillToolset pattern while adding
    optional MCP toolset integration. Each skill can have an associated
    McpToolset that gets activated when the skill is loaded.

    Usage:
        notion_skill = load_skill_from_dir(Path("skills/notion"))
        github_skill = load_skill_from_dir(Path("skills/github"))
        pure_skill = load_skill_from_dir(Path("skills/templates"))

        toolset = McpSkillToolset(
            skills=[
                (notion_skill, notion_mcp),      # Skill WITH MCP tools
                (github_skill, github_mcp),      # Skill WITH MCP tools
                (pure_skill, None),              # Skill WITHOUT MCP tools
            ]
        )

        agent = LlmAgent(tools=[toolset], ...)

    Attributes:
        _skills: Dictionary mapping skill names to Skill objects.
        _mcp_toolsets: Dictionary mapping skill names to optional McpToolset objects.
    """

    def __init__(
        self,
        *,
        skills: list[tuple[Skill, McpToolset | None]],
        tool_name_prefix: str | None = None,
    ) -> None:
        """Initialize the toolset with skills and optional MCP toolsets.

        Args:
            skills: List of (Skill, McpToolset | None) tuples. Each tuple
                contains a Skill and an optional McpToolset. If McpToolset
                is None, the skill has no MCP tools.
            tool_name_prefix: Optional prefix for tool names.
        """
        super().__init__(tool_name_prefix=tool_name_prefix)
        self._skills: dict[str, Skill] = {}
        self._mcp_toolsets: dict[str, McpToolset | None] = {}

        for skill, mcp_toolset in skills:
            self._skills[skill.frontmatter.name] = skill
            self._mcp_toolsets[skill.frontmatter.name] = mcp_toolset

        self._load_skill_tool = _LoadSkillTool(self)
        self._load_resource_tool = _LoadSkillResourceTool(self)

    def _get_skill(self, name: str) -> Skill | None:
        """Retrieves a skill by name."""
        return self._skills.get(name)

    def _list_skills(self) -> list[Frontmatter]:
        """Lists the frontmatter of all available skills."""
        return [s.frontmatter for s in self._skills.values()]

    async def _activate_skill(self, name: str, tool_context: ToolContext) -> None:
        """Activate MCP tools for a skill (if any MCP toolset exists).

        Sets the activation state for the skill, which causes get_tools()
        to include MCP tools for this skill in subsequent calls.

        Args:
            name: The name of the skill to activate.
            tool_context: The tool context containing state.
        """
        state_key = f"{SKILL_STATE_PREFIX}{name}"
        tool_context.state[state_key] = True

    async def get_tools(
        self, readonly_context: ReadonlyContext | None = None
    ) -> list[BaseTool]:
        """Returns the list of tools in this toolset.

        Includes load_skill and load_skill_resource tools, plus MCP tools
        for any activated skills that have MCP toolsets.

        Args:
            readonly_context: Context used to check activation state.

        Returns:
            List of tools available under the specified context.
        """
        tools: list[BaseTool] = [
            self._load_skill_tool,
            self._load_resource_tool,
        ]

        for skill_name in self._skills:
            state_key = f"{SKILL_STATE_PREFIX}{skill_name}"
            is_activated = readonly_context is not None and bool(
                readonly_context.state.get(state_key)
            )

            mcp_toolset = self._mcp_toolsets.get(skill_name)
            if is_activated and mcp_toolset:
                try:
                    mcp_tools = await mcp_toolset.get_tools(readonly_context)
                    tools.extend(mcp_tools)
                except Exception:
                    logger.exception(
                        "Failed to get MCP tools for skill '%s'", skill_name
                    )

        return tools

    async def process_llm_request(
        self,
        *,
        tool_context: ToolContext,
        llm_request: LlmRequest,
    ) -> None:
        """Adds available skills to the system instruction."""
        skill_frontmatters = self._list_skills()
        skills_xml = format_skills_as_xml(skill_frontmatters)

        skill_si = f"""
You can use specialized 'skills' to help you with complex tasks.
Each skill has a name and a description listed below:
{skills_xml}

Skills are folders of instructions and resources that extend your
capabilities for specialized tasks. Each skill folder contains:
- **SKILL.md** (required): Main instruction file with metadata and
  detailed markdown instructions.
- **references/** (Optional): Additional documentation or examples.
- **assets/** (Optional): Templates, scripts or other resources.

This is very important:

1. If a skill seems relevant to the current user query, you MUST call
   `load_skill(name="<SKILL_NAME>")` to read its full instructions
   before proceeding.
2. Once you have read the instructions, follow them exactly as
   documented before replying to the user.
3. Use `load_skill_resource(name="<SKILL_NAME>", path="...")` to view
   files within the skill's directory (e.g., `references/*`, `assets/*`).
"""

        llm_request.append_instructions([skill_si])

    async def close(self) -> None:
        """Closes all MCP toolsets."""
        for mcp_toolset in self._mcp_toolsets.values():
            if mcp_toolset:
                await mcp_toolset.close()
