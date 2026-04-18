"""MCP Skills module for Blacki agent.

This module provides MCP skill toolsets that combine ADK Skills with MCP tools,
supporting auto-activation when load_skill is called.
"""

from .mcp_skill_toolset import (
    McpSkillToolset,
    load_skill_from_dir,
)

__all__ = [
    "McpSkillToolset",
    "load_skill_from_dir",
]
