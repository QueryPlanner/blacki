# mypy: disable-error-code="no-untyped-def"
"""Unit and integration tests for the declarative database tool template system."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from blacki.container import set_container_from_connection
from blacki.declarative_db.plugin import DeclarativeDbPlugin
from blacki.declarative_db.storage import (
    SqliteDeclarativeDbStorage,
)
from blacki.declarative_db.tools import (
    create_custom_table,
    create_query_template,
    delete_custom_instruction_override,
    delete_custom_table,
    execute_query_template,
    list_custom_tables_and_templates,
    set_custom_instruction_override,
)
from blacki.declarative_db.validation import validate_column_type, validate_identifier

# ==============================================================================
# 1. Validation Layer Tests
# ==============================================================================


class TestValidation:
    """Test safe identifier and type validation constraints."""

    def test_valid_identifiers(self) -> None:
        """Should accept valid identifiers."""
        validate_identifier("users")
        validate_identifier("user_todos")
        validate_identifier("_private_table")
        validate_identifier("column1")

    def test_invalid_identifiers_pattern(self) -> None:
        """Should reject names with special characters or starting with numbers."""
        with pytest.raises(ValueError, match="is invalid"):
            validate_identifier("user-todos")
        with pytest.raises(ValueError, match="is invalid"):
            validate_identifier("1todos")
        with pytest.raises(ValueError, match="is invalid"):
            validate_identifier("user todos")

    def test_invalid_identifiers_keywords(self) -> None:
        """Should reject case-insensitive SQL reserved keywords."""
        with pytest.raises(ValueError, match="reserved SQL keyword"):
            validate_identifier("SELECT")
        with pytest.raises(ValueError, match="reserved SQL keyword"):
            validate_identifier("Drop")
        with pytest.raises(ValueError, match="reserved SQL keyword"):
            validate_identifier("table")

    def test_invalid_identifiers_length(self) -> None:
        """Should reject names exceeding 64 characters."""
        long_name = "a" * 65
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_identifier(long_name)

    def test_invalid_identifiers_empty(self) -> None:
        """Should reject empty names."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_identifier("")

    def test_valid_column_types(self) -> None:
        """Should accept allowlisted column types."""
        validate_column_type("TEXT")
        validate_column_type("integer")
        validate_column_type("REAL")
        validate_column_type("blob")

    def test_invalid_column_types(self) -> None:
        """Should reject non-allowlisted column types."""
        with pytest.raises(ValueError, match="is not allowed"):
            validate_column_type("VARCHAR(50)")
        with pytest.raises(ValueError, match="is not allowed"):
            validate_column_type("DATETIME")
        with pytest.raises(ValueError, match="is not allowed"):
            validate_column_type("SERIAL")


# ==============================================================================
# 2. Storage Integration Tests
# ==============================================================================


@pytest.fixture
async def sqlite_conn(tmp_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    """Provide a real, isolated SQLite connection with schema created."""
    db_path = tmp_path / "test_declarative.db"
    from blacki.storage.sqlite import create_connection

    conn = await create_connection(db_path)
    yield conn
    await conn.close()


@pytest.fixture
async def storage(sqlite_conn: aiosqlite.Connection) -> SqliteDeclarativeDbStorage:
    """Provide initialized SqliteDeclarativeDbStorage instance."""
    lock = asyncio.Lock()
    storage = SqliteDeclarativeDbStorage(sqlite_conn, lock)
    await storage.initialize()
    return storage


class TestStorageIntegration:
    """Integration tests for SqliteDeclarativeDbStorage."""

    @pytest.mark.anyio
    async def test_create_and_delete_custom_table(
        self, storage: SqliteDeclarativeDbStorage, sqlite_conn: aiosqlite.Connection
    ) -> None:
        """Should create a physical custom table and metadata, and drop it properly."""
        user_id = "user_123"
        table_name = "items"
        columns: list[dict[str, Any]] = [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "title", "type": "TEXT", "not_null": True},
            {"name": "price", "type": "REAL", "default": 0.0},
        ]

        # 1. Create table
        await storage.create_custom_table(
            user_id=user_id,
            table_name=table_name,
            columns=columns,
            description="Item catalog",
        )

        # Verify physical table structure via PRAGMA
        physical_name = storage._get_physical_name(user_id, table_name)
        async with sqlite_conn.execute(
            f"PRAGMA table_info('{physical_name}')"
        ) as cursor:
            rows = list(await cursor.fetchall())
            assert len(rows) == 3
            assert rows[0]["name"] == "id"
            assert rows[0]["type"] == "INTEGER"
            assert rows[0]["pk"] == 1
            assert rows[1]["name"] == "title"
            assert rows[1]["notnull"] == 1
            assert rows[2]["name"] == "price"
            assert rows[2]["dflt_value"] == "0.0"

        # Verify metadata
        metadata = await storage.list_custom_tables_and_templates(user_id)
        assert table_name in metadata
        assert metadata[table_name]["description"] == "Item catalog"
        assert len(metadata[table_name]["columns"]) == 3

        # 2. Delete table
        deleted = await storage.delete_custom_table(user_id, table_name)
        assert deleted is True

        # Verify metadata is cleared
        metadata = await storage.list_custom_tables_and_templates(user_id)
        assert table_name not in metadata

        # Verify physical table is dropped
        drop_query = (
            "SELECT count(*) FROM sqlite_master "  # noqa: S608
            f"WHERE type='table' AND name='{physical_name}'"  # noqa: S608
        )
        async with sqlite_conn.execute(drop_query) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0

    @pytest.mark.anyio
    async def test_table_limits_guardrails(
        self, storage: SqliteDeclarativeDbStorage
    ) -> None:
        """Should block creating more than 5 tables or tables with > 15 columns."""
        user_id = "user_guard"

        # Table with too many columns
        too_many_cols: list[dict[str, Any]] = [
            {"name": f"col_{i}", "type": "TEXT"} for i in range(16)
        ]
        with pytest.raises(ValueError, match="Limit of 15 columns"):
            await storage.create_custom_table(user_id, "big_table", too_many_cols)

        # Register 5 tables
        cols: list[dict[str, Any]] = [{"name": "id", "type": "INTEGER"}]
        for i in range(5):
            await storage.create_custom_table(user_id, f"table_{i}", cols)

        # Attempting the 6th table should fail
        with pytest.raises(ValueError, match="Limit of 5 custom tables"):
            await storage.create_custom_table(user_id, "table_6", cols)

    @pytest.mark.anyio
    async def test_query_templates_crud_and_validation(
        self, storage: SqliteDeclarativeDbStorage
    ) -> None:
        """Should manage and validate templates, enforcing a limit of 10."""
        user_id = "user_templates"
        table_name = "tasks"
        columns: list[dict[str, Any]] = [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "task_name", "type": "TEXT"},
            {"name": "status", "type": "TEXT"},
        ]

        await storage.create_custom_table(user_id, table_name, columns)

        # 1. Create a SELECT template
        await storage.create_query_template(
            user_id=user_id,
            template_name="get_by_status",
            table_name=table_name,
            query_type="SELECT",
            select_columns=["id", "task_name"],
            filter_columns=["status"],
            order_by_column="id",
            order_by_direction="DESC",
            limit_val=5,
            description="Fetch tasks by status",
        )

        # Validate duplicate/overwrite
        await storage.create_query_template(
            user_id=user_id,
            template_name="get_by_status",
            table_name=table_name,
            query_type="SELECT",
            select_columns=["id", "task_name"],
            filter_columns=["status"],
            description="Fetch tasks by status (updated)",
        )

        # 2. Reject template with invalid query_type
        with pytest.raises(ValueError, match="Query type must be"):
            await storage.create_query_template(
                user_id, "invalid_tmpl", table_name, "DROP"
            )

        # 3. Reject template with invalid sorting direction
        with pytest.raises(ValueError, match="Sorting direction must be"):
            await storage.create_query_template(
                user_id,
                "invalid_tmpl",
                table_name,
                "SELECT",
                order_by_direction="OTHER",
            )

        # 4. Reject template referencing non-existent table
        with pytest.raises(ValueError, match="does not exist"):
            await storage.create_query_template(
                user_id, "tmpl", "non_existent_table", "SELECT"
            )

        # 5. Reject template referencing non-existent column
        with pytest.raises(ValueError, match="does not exist in table"):
            await storage.create_query_template(
                user_id,
                "tmpl",
                table_name,
                "SELECT",
                select_columns=["non_existent_col"],
            )

        # 6. Template limits guardrail (max 10)
        for i in range(1, 10):  # Already have 1, add 9 more
            await storage.create_query_template(
                user_id, f"tmpl_{i}", table_name, "INSERT"
            )

        with pytest.raises(ValueError, match="Limit of 10 saved query templates"):
            await storage.create_query_template(
                user_id, "tmpl_11", table_name, "INSERT"
            )

    @pytest.mark.anyio
    async def test_execute_queries_and_sql_injection_protection(
        self, storage: SqliteDeclarativeDbStorage
    ) -> None:
        """Should run parameterized templates and catch malicious key injections."""
        user_id = "user_exec"
        table_name = "notes"
        columns: list[dict[str, Any]] = [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "content", "type": "TEXT"},
            {"name": "category", "type": "TEXT"},
        ]

        await storage.create_custom_table(user_id, table_name, columns)

        # Create INSERT template
        await storage.create_query_template(
            user_id=user_id,
            template_name="add_note",
            table_name=table_name,
            query_type="INSERT",
        )

        # Create SELECT template
        await storage.create_query_template(
            user_id=user_id,
            template_name="get_notes",
            table_name=table_name,
            query_type="SELECT",
            select_columns=["id", "content"],
            filter_columns=["category"],
        )

        # Create UPDATE template
        await storage.create_query_template(
            user_id=user_id,
            template_name="update_note",
            table_name=table_name,
            query_type="UPDATE",
            filter_columns=["id"],
        )

        # Create DELETE template
        await storage.create_query_template(
            user_id=user_id,
            template_name="delete_notes",
            table_name=table_name,
            query_type="DELETE",
            filter_columns=["category"],
        )

        # 1. Execute INSERT
        row_id1 = await storage.execute_query_template(
            user_id, "add_note", {"content": "Buy milk", "category": "shopping"}
        )
        assert row_id1 == 1

        row_id2 = await storage.execute_query_template(
            user_id, "add_note", {"content": "Work gym", "category": "fitness"}
        )
        assert row_id2 == 2

        # 2. Execute SELECT
        res = await storage.execute_query_template(
            user_id, "get_notes", {"category": "shopping"}
        )
        assert isinstance(res, list)
        assert len(res) == 1
        assert res[0]["content"] == "Buy milk"
        assert "category" not in res[0]  # select_columns only returns id and content

        # 3. Execute UPDATE
        affected_rows = await storage.execute_query_template(
            user_id, "update_note", {"id": 1, "content": "Buy whole milk"}
        )
        assert affected_rows == 1

        # Verify update worked
        res = await storage.execute_query_template(
            user_id, "get_notes", {"category": "shopping"}
        )
        assert isinstance(res, list)
        assert res[0]["content"] == "Buy whole milk"

        # 4. Execute DELETE
        deleted_count = await storage.execute_query_template(
            user_id, "delete_notes", {"category": "shopping"}
        )
        assert deleted_count == 1

        # Verify no items left in shopping category
        res = await storage.execute_query_template(
            user_id, "get_notes", {"category": "shopping"}
        )
        assert isinstance(res, list)
        assert len(res) == 0

        # 5. Security Parameter Key Injection Guard check
        # Attempt to pass a malicious SQL string as a parameter key.
        mal_key = "category OR 1=1; DROP TABLE notes; --"
        with pytest.raises(ValueError, match="is invalid|is not a valid column"):
            await storage.execute_query_template(
                user_id, "get_notes", {mal_key: "shopping"}
            )

    @pytest.mark.anyio
    async def test_custom_instructions_overrides(
        self, storage: SqliteDeclarativeDbStorage
    ) -> None:
        """Should save, get, and delete custom user-scoped prompt overrides."""
        user_id = "user_override"

        # Initially empty
        val = await storage.get_custom_instruction_override(user_id)
        assert val is None

        # Set instructions
        await storage.set_custom_instruction_override(user_id, "Tone: Sarcastic")
        val = await storage.get_custom_instruction_override(user_id)
        assert val == "Tone: Sarcastic"

        # Clear instructions
        deleted = await storage.delete_custom_instruction_override(user_id)
        assert deleted is True
        val = await storage.get_custom_instruction_override(user_id)
        assert val is None

    @pytest.mark.anyio
    async def test_get_schema_instructions_xml(
        self, storage: SqliteDeclarativeDbStorage
    ) -> None:
        """Should compile schemas and overrides into instructions XML."""
        user_id = "user_xml"
        assert await storage.get_schema_instructions_xml(user_id) == ""

        # Set instructions override
        await storage.set_custom_instruction_override(user_id, "Be extra kind.")

        # Create table & template
        await storage.create_custom_table(
            user_id, "logs", [{"name": "id", "type": "INTEGER", "primary_key": True}]
        )
        await storage.create_query_template(user_id, "add_log", "logs", "INSERT")

        xml = await storage.get_schema_instructions_xml(user_id)
        assert "<custom_instruction_overrides>" in xml
        assert "Be extra kind." in xml
        assert "<custom_database_schemas_and_templates>" in xml
        assert "Table: logs" in xml
        assert "id (INTEGER) PRIMARY KEY" in xml
        assert "Template: add_log" in xml


# ==============================================================================
# 3. ADK Plugin Layer Tests
# ==============================================================================


class TestDeclarativeDbPlugin:
    """Test dynamic context injection plugin."""

    @pytest.mark.anyio
    async def test_plugin_appends_instructions(
        self, storage: SqliteDeclarativeDbStorage, sqlite_conn: aiosqlite.Connection
    ) -> None:
        """Should compile active schema XML and safely append to instructions."""
        # Setup container for process-wide lazy instantiation
        container = set_container_from_connection(sqlite_conn)
        container._declarative_db_storage = storage

        plugin = DeclarativeDbPlugin()

        # Build mock callback_context and llm_request
        mock_session = MagicMock()
        mock_session.state = {"user_id": "user_plugin_test"}
        callback_context = MagicMock()
        callback_context.session = mock_session

        llm_request = MagicMock()
        llm_request.append_instructions = MagicMock()

        # Add data to storage to compile
        await storage.set_custom_instruction_override(
            "user_plugin_test", "Always speak in Spanish."
        )

        # Run callback
        await plugin.before_model_callback(
            callback_context=callback_context, llm_request=llm_request
        )

        # Assert compiled schema was appended to LLM instructions
        llm_request.append_instructions.assert_called_once()
        args = llm_request.append_instructions.call_args[0][0]
        assert len(args) == 1
        assert "Spanish" in args[0]
        assert "<custom_instruction_overrides>" in args[0]


# ==============================================================================
# 4. Agent Tools Layer Tests
# ==============================================================================


class TestAgentTools:
    """Test ADK tools proxy functions."""

    @pytest.fixture(autouse=True)
    def setup_mock_container(self) -> Generator[None, None, None]:
        """Mock out get_declarative_db_storage singleton lookup."""
        self.mock_storage = MagicMock()
        patcher = MagicMock()
        patcher.return_value = self.mock_storage

        # Patch get_declarative_db_storage in the tools module
        self.patcher = patch(
            "blacki.declarative_db.tools.get_declarative_db_storage", patcher
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @pytest.mark.anyio
    async def test_tools_extract_user_id_and_proxy_calls(self) -> None:
        """Agent tools should pull user_id and call equivalent storage methods."""
        tool_context = MagicMock()
        tool_context.user_id = None
        tool_context.state = {"user_id": "tool_user"}

        # 1. create_custom_table tool
        self.mock_storage.create_custom_table = AsyncMock()
        res = await create_custom_table(
            "events",
            [{"name": "id", "type": "INTEGER"}],
            tool_context,
            "History of events",
        )
        assert res["status"] == "success"
        self.mock_storage.create_custom_table.assert_called_once_with(
            user_id="tool_user",
            table_name="events",
            columns=[{"name": "id", "type": "INTEGER"}],
            description="History of events",
        )

        # 2. delete_custom_table tool
        self.mock_storage.delete_custom_table = AsyncMock(return_value=True)
        res = await delete_custom_table("events", tool_context)
        assert res["status"] == "success"
        self.mock_storage.delete_custom_table.assert_called_once_with(
            "tool_user", "events"
        )

        # 3. create_query_template tool
        self.mock_storage.create_query_template = AsyncMock()
        res = await create_query_template("add_event", "events", "INSERT", tool_context)
        assert res["status"] == "success"
        self.mock_storage.create_query_template.assert_called_once_with(
            user_id="tool_user",
            template_name="add_event",
            table_name="events",
            query_type="INSERT",
            select_columns=None,
            filter_columns=None,
            order_by_column=None,
            order_by_direction=None,
            limit_val=None,
            description=None,
        )

        # 4. execute_query_template tool
        self.mock_storage.execute_query_template = AsyncMock(return_value=[{"id": 1}])
        res = await execute_query_template("get_events", {"id": 1}, tool_context)
        assert res["status"] == "success"
        assert res["result"] == [{"id": 1}]
        self.mock_storage.execute_query_template.assert_called_once_with(
            user_id="tool_user",
            template_name="get_events",
            parameters={"id": 1},
        )

        # 5. list_custom_tables_and_templates tool
        self.mock_storage.list_custom_tables_and_templates = AsyncMock(return_value={})
        res = await list_custom_tables_and_templates(tool_context)
        assert res["status"] == "success"
        assert res["tables"] == {}
        self.mock_storage.list_custom_tables_and_templates.assert_called_once_with(
            "tool_user"
        )

        # 6. set_custom_instruction_override tool
        self.mock_storage.set_custom_instruction_override = AsyncMock()
        res = await set_custom_instruction_override("Fly high", tool_context)
        assert res["status"] == "success"
        self.mock_storage.set_custom_instruction_override.assert_called_once_with(
            "tool_user", "Fly high"
        )

        # 7. delete_custom_instruction_override tool
        self.mock_storage.delete_custom_instruction_override = AsyncMock(
            return_value=True
        )
        res = await delete_custom_instruction_override(tool_context)
        assert res["status"] == "success"
        self.mock_storage.delete_custom_instruction_override.assert_called_once_with(
            "tool_user"
        )
