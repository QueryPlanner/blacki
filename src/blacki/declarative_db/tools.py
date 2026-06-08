"""Agent-facing ADK tools for declarative databases."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.tools import ToolContext

from blacki.declarative_db.storage import get_declarative_db_storage

logger = logging.getLogger(__name__)


async def create_custom_table(
    table_name: str,
    columns: list[dict[str, Any]],
    tool_context: ToolContext,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a new custom physical table and register its metadata.

    The physical SQLite table is securely created with the specified columns and
    types, and is scoped and isolated per user.

    Args:
        table_name: The name of the table to create (lowercase, alphanumeric).
        columns: A list of dicts specifying column definitions. Each column dict
                 can have:
                 - "name" (str): Column name (alphanumeric, lowercase).
                 - "type" (str): Column type. Restricted strictly to:
                                 "TEXT", "INTEGER", "REAL", "BLOB".
                 - "primary_key" (bool): Whether column is primary key.
                 - "not_null" (bool): Whether column cannot be null.
                 - "default" (any): Default column value.
        tool_context: ADK tool context.
        description: A short optional description of the table's purpose.

    Returns:
        A status dictionary confirming creation or detailing validation errors.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "User not identified in tool context.",
        }

    try:
        storage = get_declarative_db_storage()
        await storage.create_custom_table(
            user_id=str(user_id),
            table_name=table_name,
            columns=columns,
            description=description,
        )
        return {
            "status": "success",
            "message": (
                f"Table '{table_name}' was successfully created with "
                f"{len(columns)} columns."
            ),
        }
    except Exception as e:
        logger.exception("Failed to create custom table %s", table_name)
        return {
            "status": "error",
            "message": f"Failed to create table: {e}",
        }


async def delete_custom_table(
    table_name: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Physically drop a custom table and delete all its query templates.

    Args:
        table_name: The logical name of the table to drop.
        tool_context: ADK tool context.

    Returns:
        A status dictionary.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "User not identified in tool context.",
        }

    try:
        storage = get_declarative_db_storage()
        deleted = await storage.delete_custom_table(str(user_id), table_name)
        if deleted:
            return {
                "status": "success",
                "message": (
                    f"Table '{table_name}' and all its templates "
                    "were successfully deleted."
                ),
            }
        return {
            "status": "error",
            "message": f"Table '{table_name}' was not found.",
        }
    except Exception as e:
        logger.exception("Failed to delete custom table %s", table_name)
        return {
            "status": "error",
            "message": f"Failed to delete table: {e}",
        }


async def create_query_template(
    template_name: str,
    table_name: str,
    query_type: str,
    tool_context: ToolContext,
    select_columns: list[str] | None = None,
    filter_columns: list[str] | None = None,
    order_by_column: str | None = None,
    order_by_direction: str | None = None,
    limit_val: int | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Save a query template against a custom user table.

    Rather than compiling raw string statements, templates define structure.
    The storage manager compiles them securely into parameterized queries
    during runtime execution.

    Args:
        template_name: The name of this query template (unique per user).
        table_name: The logical table this query executes against.
        query_type: The action of the query. Must be one of:
                    "SELECT", "INSERT", "UPDATE", "DELETE".
        tool_context: ADK tool context.
        select_columns: List of columns to return (for SELECT). If empty, returns all.
        filter_columns: List of columns to bind parameters to inside the WHERE clause
                        (e.g., ["id"] maps to WHERE "id" = ?). All must be
                        passed in parameters during execution.
        order_by_column: Name of column to sort on (for SELECT).
        order_by_direction: Sorting direction (for SELECT). Must be "ASC" or "DESC".
        limit_val: Optional integer pagination constraint (for SELECT).
        description: A brief optional description of what the query template does.

    Returns:
        A status dictionary.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "User not identified in tool context.",
        }

    try:
        storage = get_declarative_db_storage()
        await storage.create_query_template(
            user_id=str(user_id),
            template_name=template_name,
            table_name=table_name,
            query_type=query_type,
            select_columns=select_columns,
            filter_columns=filter_columns,
            order_by_column=order_by_column,
            order_by_direction=order_by_direction,
            limit_val=limit_val,
            description=description,
        )
        return {
            "status": "success",
            "message": (
                f"Query template '{template_name}' was successfully "
                f"registered for table '{table_name}'."
            ),
        }
    except Exception as e:
        logger.exception("Failed to create query template %s", template_name)
        return {
            "status": "error",
            "message": f"Failed to register query template: {e}",
        }


async def execute_query_template(
    template_name: str,
    parameters: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Execute a saved query template securely with inputs.

    Args:
        template_name: Name of the template to run.
        parameters: A dictionary of key-value bindings. Every key must map
                    exactly to a valid column defined for the underlying table.
        tool_context: ADK tool context.

    Returns:
        A result dictionary containing status and output records (for SELECT) or
        affected rows count / last inserted ID (for write queries).
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "User not identified in tool context.",
        }

    try:
        storage = get_declarative_db_storage()
        res = await storage.execute_query_template(
            user_id=str(user_id),
            template_name=template_name,
            parameters=parameters,
        )
        return {
            "status": "success",
            "result": res,
        }
    except Exception as e:
        logger.exception("Failed to execute query template %s", template_name)
        return {
            "status": "error",
            "message": f"Execution failed: {e}",
        }


async def list_custom_tables_and_templates(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """List all custom tables, columns, and registered templates for the active user.

    Args:
        tool_context: ADK tool context.

    Returns:
        A dictionary mapping table names to their schema structures and query templates.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "User not identified in tool context.",
        }

    try:
        storage = get_declarative_db_storage()
        schemas = await storage.list_custom_tables_and_templates(str(user_id))
        return {
            "status": "success",
            "tables": schemas,
        }
    except Exception as e:
        logger.exception("Failed to list schemas")
        return {
            "status": "error",
            "message": f"Failed to fetch custom schemas: {e}",
        }


async def set_custom_instruction_override(
    instructions: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Persist a custom instruction override for the user.

    This instructions block is injected into system instruction prompts to guide
    the model's persona, handling, or workflows dynamically, safely isolated from
    the system prompt codebase.

    Args:
        instructions: Raw text instructions to guide the model.
        tool_context: ADK tool context.

    Returns:
        A status dictionary.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "User not identified in tool context.",
        }

    try:
        storage = get_declarative_db_storage()
        await storage.set_custom_instruction_override(str(user_id), instructions)
        return {
            "status": "success",
            "message": "Custom instructions successfully saved.",
        }
    except Exception as e:
        logger.exception("Failed to set instruction override")
        return {
            "status": "error",
            "message": f"Failed to save instructions: {e}",
        }


async def delete_custom_instruction_override(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Clear custom instruction override for the user.

    Args:
        tool_context: ADK tool context.

    Returns:
        A status dictionary.
    """
    user_id = getattr(tool_context, "user_id", None) or tool_context.state.get(
        "user_id"
    )
    if not user_id:
        return {
            "status": "error",
            "message": "User not identified in tool context.",
        }

    try:
        storage = get_declarative_db_storage()
        deleted = await storage.delete_custom_instruction_override(str(user_id))
        if deleted:
            return {
                "status": "success",
                "message": "Custom instructions successfully deleted.",
            }
        return {
            "status": "success",
            "message": "No custom instructions existed to delete.",
        }
    except Exception as e:
        logger.exception("Failed to delete instruction override")
        return {
            "status": "error",
            "message": f"Failed to clear instructions: {e}",
        }
