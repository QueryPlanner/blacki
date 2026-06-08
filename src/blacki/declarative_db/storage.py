"""SQLite storage implementation for declarative tables and query templates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from blacki.declarative_db.validation import validate_column_type, validate_identifier
from blacki.storage.base import SqlStorage
from blacki.utils.timezone import now_utc

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SqliteDeclarativeDbStorage(SqlStorage):
    """Storage implementation for custom schemas, query templates, and overrides."""

    def __init__(self, conn: aiosqlite.Connection, lock: asyncio.Lock) -> None:
        super().__init__(conn, lock)

    async def _create_tables(self) -> None:
        # Create metadata table for tracking tables
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS custom_tables (
                user_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                physical_name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, table_name)
            )
        """)

        # Create metadata table for column definitions
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS custom_table_columns (
                user_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                column_type TEXT NOT NULL,
                is_primary_key INTEGER NOT NULL DEFAULT 0,
                is_not_null INTEGER NOT NULL DEFAULT 0,
                default_value TEXT,
                PRIMARY KEY (user_id, table_name, column_name),
                FOREIGN KEY (user_id, table_name)
                    REFERENCES custom_tables(user_id, table_name) ON DELETE CASCADE
            )
        """)

        # Create metadata table for query templates
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_query_templates (
                user_id TEXT NOT NULL,
                template_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                query_type TEXT NOT NULL,
                description TEXT,
                select_columns TEXT,     -- JSON list of column names
                filter_columns TEXT,     -- JSON list of column names
                order_by_column TEXT,
                order_by_direction TEXT, -- ASC, DESC
                limit_val INTEGER,
                PRIMARY KEY (user_id, template_name),
                FOREIGN KEY (user_id, table_name)
                    REFERENCES custom_tables(user_id, table_name) ON DELETE CASCADE
            )
        """)

        # Create metadata table for user-scoped instructions
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS custom_instruction_overrides (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                instructions TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
        """)

    def _get_physical_name(self, user_id: str, table_name: str) -> str:
        """Derive a safe physical table name scoped by hashed user_id."""
        user_hash = hashlib.md5(
            user_id.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:16]
        return f"usr_{user_hash}_{table_name}"

    async def create_custom_table(
        self,
        user_id: str,
        table_name: str,
        columns: list[dict[str, Any]],
        description: str | None = None,
    ) -> None:
        """Create a safe custom physical table and store its metadata.

        Args:
            user_id: The scoping user ID.
            table_name: Logical table name.
            columns: List of columns containing name, type, primary_key,
                     not_null, and default keys.
            description: Optional table description.
        """
        validate_identifier(table_name)
        physical_name = self._get_physical_name(user_id, table_name)

        if not columns:
            raise ValueError("Table must define at least one column")

        # Validate columns and prepare dynamic DDL components
        validated_cols: list[dict[str, Any]] = []
        ddl_parts: list[str] = []

        for col in columns:
            col_name = col.get("name", "").strip()
            col_type = col.get("type", "").strip().upper()
            is_pk = int(bool(col.get("primary_key")))
            is_nn = int(bool(col.get("not_null")))
            default_val = col.get("default")

            validate_identifier(col_name)
            validate_column_type(col_type)

            validated_cols.append(
                {
                    "name": col_name,
                    "type": col_type,
                    "is_pk": is_pk,
                    "is_nn": is_nn,
                    "default": str(default_val) if default_val is not None else None,
                }
            )

            # Form column DDL segment
            part = f'"{col_name}" {col_type}'
            if is_pk:
                part += " PRIMARY KEY"
            if is_nn:
                part += " NOT NULL"
            if default_val is not None:
                # Basic escaping of default string values if string-like
                if isinstance(default_val, str):
                    escaped_default = default_val.replace("'", "''")
                    part += f" DEFAULT '{escaped_default}'"
                else:
                    part += f" DEFAULT {default_val}"

            ddl_parts.append(part)

        # Build physical CREATE TABLE string
        create_ddl = (
            f'CREATE TABLE IF NOT EXISTS "{physical_name}" (\n    '
            + ",\n    ".join(ddl_parts)
            + "\n)"
        )

        now = now_utc().isoformat(timespec="seconds")

        async with self._lock:
            # Check table counts guardrail
            existing_count_row = await self._fetch_one(
                "SELECT COUNT(*) as count FROM custom_tables WHERE user_id = ?",
                (user_id,),
            )
            existing_count = existing_count_row["count"] if existing_count_row else 0
            if existing_count >= 5:
                raise ValueError("Limit of 5 custom tables per user reached")

            if len(columns) > 15:
                raise ValueError("Limit of 15 columns per custom table reached")

            # Initialize physical table and save metadata within one serial transaction
            await self._conn.execute("BEGIN")
            try:
                # Physically create table
                await self._conn.execute(create_ddl)

                # Save table metadata
                await self._conn.execute(
                    """
                    INSERT INTO custom_tables (
                        user_id, table_name, physical_name, description, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (user_id, table_name) DO UPDATE SET
                        description = excluded.description,
                        physical_name = excluded.physical_name
                    """,
                    (user_id, table_name, physical_name, description, now),
                )

                # Delete any old metadata columns
                await self._conn.execute(
                    "DELETE FROM custom_table_columns "
                    "WHERE user_id = ? AND table_name = ?",
                    (user_id, table_name),
                )

                # Save column metadata
                for c in validated_cols:
                    await self._conn.execute(
                        """
                        INSERT INTO custom_table_columns (
                            user_id, table_name, column_name, column_type,
                            is_primary_key, is_not_null, default_value
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            table_name,
                            c["name"],
                            c["type"],
                            c["is_pk"],
                            c["is_nn"],
                            c["default"],
                        ),
                    )

                await self._conn.execute("COMMIT")
                logger.info(
                    "Successfully created table %s with %d columns",
                    table_name,
                    len(validated_cols),
                )
            except Exception as e:
                await self._conn.execute("ROLLBACK")
                logger.exception("Failed to create custom table %s", table_name)
                raise e

    async def delete_custom_table(self, user_id: str, table_name: str) -> bool:
        """Physically drop a table and delete its metadata records."""
        validate_identifier(table_name)
        physical_name = self._get_physical_name(user_id, table_name)

        async with self._lock:
            # Check if table metadata exists first
            row = await self._fetch_one(
                "SELECT table_name FROM custom_tables "
                "WHERE user_id = ? AND table_name = ?",
                (user_id, table_name),
            )
            if not row:
                return False

            await self._conn.execute("BEGIN")
            try:
                # Drop physical table
                await self._conn.execute(f'DROP TABLE IF EXISTS "{physical_name}"')

                # Clear metadata (ON DELETE CASCADE propagates to columns & templates)
                await self._conn.execute(
                    "DELETE FROM custom_tables WHERE user_id = ? AND table_name = ?",
                    (user_id, table_name),
                )

                await self._conn.execute("COMMIT")
                logger.info("Successfully dropped table %s", table_name)
                return True
            except Exception as e:
                await self._conn.execute("ROLLBACK")
                logger.exception("Failed to drop custom table %s", table_name)
                raise e

    async def create_query_template(
        self,
        user_id: str,
        template_name: str,
        table_name: str,
        query_type: str,
        select_columns: list[str] | None = None,
        filter_columns: list[str] | None = None,
        order_by_column: str | None = None,
        order_by_direction: str | None = None,
        limit_val: int | None = None,
        description: str | None = None,
    ) -> None:
        """Create or save a query template against a logical custom table."""
        validate_identifier(template_name)
        validate_identifier(table_name)
        query_type = query_type.strip().upper()
        if query_type not in {"SELECT", "INSERT", "UPDATE", "DELETE"}:
            raise ValueError("Query type must be SELECT, INSERT, UPDATE, or DELETE")

        # Validate identifiers inside lists
        if select_columns:
            for s in select_columns:
                validate_identifier(s)
        if filter_columns:
            for f in filter_columns:
                validate_identifier(f)
        if order_by_column:
            validate_identifier(order_by_column)
        if order_by_direction:
            order_by_dir_upper = order_by_direction.strip().upper()
            if order_by_dir_upper not in {"ASC", "DESC"}:
                raise ValueError("Sorting direction must be ASC or DESC")
            order_by_direction = order_by_dir_upper

        async with self._lock:
            # Check templates limit (10 per user)
            existing_count_row = await self._fetch_one(
                "SELECT COUNT(*) as count FROM saved_query_templates WHERE user_id = ?",
                (user_id,),
            )
            existing_count = existing_count_row["count"] if existing_count_row else 0
            if existing_count >= 10:
                raise ValueError("Limit of 10 saved query templates per user reached")

            # Check if custom table exists
            table_row = await self._fetch_one(
                "SELECT table_name FROM custom_tables "
                "WHERE user_id = ? AND table_name = ?",
                (user_id, table_name),
            )
            if not table_row:
                raise ValueError(f"Table '{table_name}' does not exist for this user")

            # Validate column existence against metadata
            allowed_cols_rows = await self._fetch_all(
                "SELECT column_name FROM custom_table_columns "
                "WHERE user_id = ? AND table_name = ?",
                (user_id, table_name),
            )
            allowed_cols = {r["column_name"] for r in allowed_cols_rows}

            for col_list, label in [
                (select_columns, "select_columns"),
                (filter_columns, "filter_columns"),
            ]:
                if col_list:
                    for c in col_list:
                        if c not in allowed_cols:
                            raise ValueError(
                                f"Column '{c}' in {label} does not "
                                f"exist in table '{table_name}'"
                            )

            if order_by_column and order_by_column not in allowed_cols:
                raise ValueError(
                    f"Column '{order_by_column}' in order_by_column "
                    f"does not exist in table '{table_name}'"
                )

            # Insert metadata into saved templates
            await self._conn.execute(
                """
                INSERT INTO saved_query_templates (
                    user_id, template_name, table_name, query_type, description,
                    select_columns, filter_columns, order_by_column,
                    order_by_direction, limit_val
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, template_name) DO UPDATE SET
                    table_name = excluded.table_name,
                    query_type = excluded.query_type,
                    description = excluded.description,
                    select_columns = excluded.select_columns,
                    filter_columns = excluded.filter_columns,
                    order_by_column = excluded.order_by_column,
                    order_by_direction = excluded.order_by_direction,
                    limit_val = excluded.limit_val
                """,
                (
                    user_id,
                    template_name,
                    table_name,
                    query_type,
                    description,
                    json.dumps(select_columns) if select_columns is not None else None,
                    json.dumps(filter_columns) if filter_columns is not None else None,
                    order_by_column,
                    order_by_direction,
                    limit_val,
                ),
            )
            logger.info(
                "Saved query template %s against table %s", template_name, table_name
            )

    async def execute_query_template(
        self,
        user_id: str,
        template_name: str,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]] | int:
        """Securely build and run a saved template query with validation."""
        validate_identifier(template_name)

        # 1. Fetch template from DB
        template = await self._fetch_one(
            "SELECT * FROM saved_query_templates "
            "WHERE user_id = ? AND template_name = ?",
            (user_id, template_name),
        )
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        table_name = template["table_name"]
        query_type = template["query_type"]

        # Fetch physical table name
        table_meta = await self._fetch_one(
            "SELECT physical_name FROM custom_tables "
            "WHERE user_id = ? AND table_name = ?",
            (user_id, table_name),
        )
        if not table_meta:
            raise ValueError(f"Underlying table '{table_name}' does not exist")
        physical_name = table_meta["physical_name"]

        # Fetch valid columns
        valid_cols_rows = await self._fetch_all(
            "SELECT column_name FROM custom_table_columns "
            "WHERE user_id = ? AND table_name = ?",
            (user_id, table_name),
        )
        valid_cols = {r["column_name"] for r in valid_cols_rows}

        # 2. Strict Parameter Validation
        # Verify that all keys supplied in parameters are valid matching column names
        for key in parameters:
            validate_identifier(key)
            if key not in valid_cols:
                raise ValueError(
                    f"Parameter key '{key}' is not a valid "
                    f"column for table '{table_name}'"
                )

        # Parse column lists from JSON
        select_cols = (
            json.loads(template["select_columns"]) if template["select_columns"] else []
        )
        filter_cols = (
            json.loads(template["filter_columns"]) if template["filter_columns"] else []
        )

        # 3. Secure Query Assembly
        sql = ""
        bindings: tuple[Any, ...] = ()

        if query_type == "SELECT":
            # Form SELECT segment
            cols_str = ", ".join(f'"{c}"' for c in select_cols) if select_cols else "*"

            # Validated & quoted identifiers: SQL safe.
            sql = f'SELECT {cols_str} FROM "{physical_name}"'  # noqa: S608

            # Form WHERE segment
            if filter_cols:
                where_segments = []
                where_bindings = []
                for c in filter_cols:
                    if c in parameters:
                        where_segments.append(f'"{c}" = ?')
                        where_bindings.append(parameters[c])
                    else:
                        raise ValueError(f"Missing required filter parameter '{c}'")
                sql += " WHERE " + " AND ".join(where_segments)
                bindings = tuple(where_bindings)

            # Form ORDER BY segment
            if template["order_by_column"]:
                ob_col = template["order_by_column"]
                ob_dir = template["order_by_direction"] or "ASC"
                sql += f' ORDER BY "{ob_col}" {ob_dir}'

            # Form LIMIT segment
            if template["limit_val"] is not None:
                sql += f" LIMIT {int(template['limit_val'])}"

            # Run SELECT
            return await self._fetch_all(sql, bindings)

        elif query_type == "INSERT":
            # Form INSERT parameters from direct parameters matching columns
            insert_keys = [k for k in parameters if k in valid_cols]
            if not insert_keys:
                raise ValueError("Must provide at least one valid parameter to insert")

            cols_str = ", ".join(f'"{k}"' for k in insert_keys)
            placeholders = ", ".join("?" for _ in insert_keys)
            sql = f'INSERT INTO "{physical_name}" ({cols_str}) VALUES ({placeholders})'  # noqa: S608
            bindings = tuple(parameters[k] for k in insert_keys)

            # Run write insert
            return await self._execute(sql, bindings)

        elif query_type == "UPDATE":
            # Form UPDATE columns from params not in filter_columns
            update_keys = [
                k for k in parameters if k in valid_cols and k not in filter_cols
            ]
            if not update_keys:
                raise ValueError("No columns provided to update")

            set_segments = [f'"{k}" = ?' for k in update_keys]
            set_bindings = [parameters[k] for k in update_keys]

            sql = f'UPDATE "{physical_name}" SET ' + ", ".join(set_segments)  # noqa: S608

            # Form WHERE segment
            if filter_cols:
                where_segments = []
                where_bindings = []
                for c in filter_cols:
                    if c in parameters:
                        where_segments.append(f'"{c}" = ?')
                        where_bindings.append(parameters[c])
                    else:
                        raise ValueError(
                            f"Missing required update filter parameter '{c}'"
                        )
                sql += " WHERE " + " AND ".join(where_segments)
                bindings = tuple(set_bindings + where_bindings)
            else:
                bindings = tuple(set_bindings)

            # Run write update
            async with self._lock:
                cursor = await self._conn.execute(sql, bindings)
                return cursor.rowcount

        elif query_type == "DELETE":
            sql = f'DELETE FROM "{physical_name}"'  # noqa: S608
            if filter_cols:
                where_segments = []
                where_bindings = []
                for c in filter_cols:
                    if c in parameters:
                        where_segments.append(f'"{c}" = ?')
                        where_bindings.append(parameters[c])
                    else:
                        raise ValueError(
                            f"Missing required delete filter parameter '{c}'"
                        )
                sql += " WHERE " + " AND ".join(where_segments)
                bindings = tuple(where_bindings)

            # Run write delete
            async with self._lock:
                cursor = await self._conn.execute(sql, bindings)
                return cursor.rowcount

        else:
            raise ValueError(f"Unsupported query type: {query_type}")

    async def set_custom_instruction_override(
        self, user_id: str, instructions: str
    ) -> None:
        """Update or insert a custom instruction override."""
        now = now_utc().isoformat(timespec="seconds")
        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO custom_instruction_overrides (
                    user_id, key, instructions, updated_at
                ) VALUES (?, 'custom_instructions', ?, ?)
                ON CONFLICT (user_id, key) DO UPDATE SET
                    instructions = excluded.instructions,
                    updated_at = excluded.updated_at
                """,
                (user_id, instructions, now),
            )
            logger.info("Updated custom instructions for user %s", user_id)

    async def get_custom_instruction_override(self, user_id: str) -> str | None:
        """Retrieve custom instruction override for user."""
        row = await self._fetch_one(
            "SELECT instructions FROM custom_instruction_overrides "
            "WHERE user_id = ? AND key = 'custom_instructions'",
            (user_id,),
        )
        return row["instructions"] if row else None

    async def delete_custom_instruction_override(self, user_id: str) -> bool:
        """Clear custom instruction override for user."""
        async with self._lock:
            cursor = await self._conn.execute(
                "DELETE FROM custom_instruction_overrides "
                "WHERE user_id = ? AND key = 'custom_instructions'",
                (user_id,),
            )
            return cursor.rowcount > 0

    async def list_custom_tables_and_templates(self, user_id: str) -> dict[str, Any]:
        """List tables, columns, and saved templates for a given user."""
        tables = await self._fetch_all(
            "SELECT table_name, description, created_at FROM custom_tables "
            "WHERE user_id = ? ORDER BY table_name ASC",
            (user_id,),
        )
        columns = await self._fetch_all(
            "SELECT table_name, column_name, column_type, is_primary_key, "
            "is_not_null, default_value FROM custom_table_columns "
            "WHERE user_id = ? ORDER BY table_name ASC, column_name ASC",
            (user_id,),
        )
        templates = await self._fetch_all(
            "SELECT template_name, table_name, query_type, description, "
            "select_columns, filter_columns, order_by_column, "
            "order_by_direction, limit_val FROM saved_query_templates "
            "WHERE user_id = ? ORDER BY template_name ASC",
            (user_id,),
        )

        # Structure response
        structured_tables: dict[str, Any] = {}
        for t in tables:
            t_name = t["table_name"]
            structured_tables[t_name] = {
                "description": t["description"],
                "created_at": t["created_at"],
                "columns": [],
                "templates": [],
            }

        for col in columns:
            t_name = col["table_name"]
            if t_name in structured_tables:
                structured_tables[t_name]["columns"].append(
                    {
                        "name": col["column_name"],
                        "type": col["column_type"],
                        "primary_key": bool(col["is_primary_key"]),
                        "not_null": bool(col["is_not_null"]),
                        "default": col["default_value"],
                    }
                )

        for tmpl in templates:
            t_name = tmpl["table_name"]
            if t_name in structured_tables:
                structured_tables[t_name]["templates"].append(
                    {
                        "name": tmpl["template_name"],
                        "type": tmpl["query_type"],
                        "description": tmpl["description"],
                        "select_columns": json.loads(tmpl["select_columns"])
                        if tmpl["select_columns"]
                        else None,
                        "filter_columns": json.loads(tmpl["filter_columns"])
                        if tmpl["filter_columns"]
                        else None,
                        "order_by": (
                            f"{tmpl['order_by_column']} {tmpl['order_by_direction']}"
                            if tmpl["order_by_column"]
                            else None
                        ),
                        "limit": tmpl["limit_val"],
                    }
                )

        return structured_tables

    async def get_schema_instructions_xml(self, user_id: str) -> str:
        """Compile schemas and overrides into instructions XML."""
        schema_data = await self.list_custom_tables_and_templates(user_id)
        override = await self.get_custom_instruction_override(user_id)

        blocks: list[str] = []

        if override:
            blocks.append(
                f"<custom_instruction_overrides>\n{override}\n</custom_instruction_overrides>"
            )

        if schema_data:
            schema_lines = [
                "You have access to the following custom user-defined database tables:"
            ]
            for t_name, t_val in schema_data.items():
                schema_lines.append(f"\nTable: {t_name}")
                if t_val["description"]:
                    schema_lines.append(f"  Description: {t_val['description']}")
                schema_lines.append("  Columns:")
                for col in t_val["columns"]:
                    pk_label = " PRIMARY KEY" if col["primary_key"] else ""
                    nn_label = " NOT NULL" if col["not_null"] else ""
                    def_label = (
                        f" DEFAULT {col['default']}"
                        if col["default"] is not None
                        else ""
                    )
                    col_info = (
                        f"    - {col['name']} ({col['type']})"
                        f"{pk_label}{nn_label}{def_label}"
                    )
                    schema_lines.append(col_info)

                if t_val["templates"]:
                    schema_lines.append("  Saved Query Templates:")
                    for tmpl in t_val["templates"]:
                        desc = (
                            f" ({tmpl['description']})" if tmpl["description"] else ""
                        )
                        schema_lines.append(f"    - Template: {tmpl['name']}{desc}")
                        schema_lines.append(f"      Operation: {tmpl['type']}")
                        if tmpl["select_columns"]:
                            cols_joined = ", ".join(tmpl["select_columns"])
                            schema_lines.append(f"      Returns Columns: {cols_joined}")
                        if tmpl["filter_columns"]:
                            params_joined = ", ".join(tmpl["filter_columns"])
                            schema_lines.append(
                                f"      Required Parameters: {params_joined}"
                            )
                        if tmpl["order_by"]:
                            schema_lines.append(f"      Sorted By: {tmpl['order_by']}")
                        if tmpl["limit"]:
                            schema_lines.append(f"      Limit: {tmpl['limit']}")

            schema_str = "\n".join(schema_lines)
            blocks.append(
                f"<custom_database_schemas_and_templates>\n{schema_str}\n</custom_database_schemas_and_templates>"
            )

        return "\n\n".join(blocks) if blocks else ""


_storage: SqliteDeclarativeDbStorage | None = None


def get_declarative_db_storage() -> SqliteDeclarativeDbStorage:
    """Return the process-wide singleton SqliteDeclarativeDbStorage instance."""
    from blacki.container import get_container

    container = get_container()
    storage = container.declarative_db_storage
    if not storage.is_initialized:
        raise RuntimeError(
            "Declarative DB storage not initialized. Call storage.initialize() first."
        )
    return storage
