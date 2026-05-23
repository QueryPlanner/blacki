# mypy: disable-error-code="no-untyped-def"
"""Unit tests for SqlStorage base class."""

import asyncio
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from blacki.storage.base import SqlStorage


class ConcreteStorage(SqlStorage):
    """Concrete implementation for testing abstract base class."""

    async def _create_tables(self) -> None:
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, value TEXT)"
        )


@pytest.fixture
async def conn():
    """Create an in-memory SQLite connection for testing."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


@pytest.fixture
def lock():
    """Create a lock for write operations."""
    return asyncio.Lock()


@pytest.fixture
async def storage(conn, lock):
    """Create a storage instance with the test connection."""
    storage = ConcreteStorage(conn, lock)
    await storage.initialize()
    yield storage
    await storage.close()


class TestSqlStorageExecute:
    """Tests for _execute method."""

    @pytest.mark.asyncio
    async def test_execute_with_lock(self, storage) -> None:
        """Should execute query with lock by default."""
        rid = await storage._execute(
            "INSERT INTO test_table (value) VALUES (?)", ("test_value",)
        )

        assert rid == 1

        row = await storage._fetch_one("SELECT * FROM test_table WHERE id = ?", (rid,))
        assert row is not None
        assert row["value"] == "test_value"

    @pytest.mark.asyncio
    async def test_execute_without_lock(self, storage) -> None:
        """Should execute query without lock when use_lock=False."""
        rid = await storage._execute(
            "INSERT INTO test_table (value) VALUES (?)",
            ("test_value",),
            use_lock=False,
        )

        assert rid == 1

        row = await storage._fetch_one("SELECT * FROM test_table WHERE id = ?", (rid,))
        assert row is not None
        assert row["value"] == "test_value"

    @pytest.mark.asyncio
    async def test_execute_raises_runtime_error_when_lastrowid_none(
        self, conn, lock
    ) -> None:
        """Should raise RuntimeError when lastrowid is None after insert."""
        from unittest.mock import MagicMock

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = None

        async_cm = MagicMock()
        async_cm.__aenter__ = AsyncMock(return_value=mock_cursor)
        async_cm.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute.return_value = async_cm

        storage = ConcreteStorage(mock_conn, lock)

        with pytest.raises(RuntimeError, match="Failed to get lastrowid after insert"):
            await storage._execute(
                "INSERT INTO test_table (value) VALUES (?)", ("test",)
            )

    @pytest.mark.asyncio
    async def test_execute_without_lock_raises_runtime_error_when_lastrowid_none(
        self, conn, lock
    ) -> None:
        """Should raise RuntimeError when lastrowid is None with use_lock=False."""
        from unittest.mock import MagicMock

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = None

        async_cm = MagicMock()
        async_cm.__aenter__ = AsyncMock(return_value=mock_cursor)
        async_cm.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute.return_value = async_cm

        storage = ConcreteStorage(mock_conn, lock)

        with pytest.raises(RuntimeError, match="Failed to get lastrowid after insert"):
            await storage._execute(
                "INSERT INTO test_table (value) VALUES (?)",
                ("test",),
                use_lock=False,
            )


class TestSqlStorageExecuteMany:
    """Tests for _execute_many method."""

    @pytest.mark.asyncio
    async def test_execute_many_with_lock(self, storage) -> None:
        """Should execute many with lock by default."""
        params_list = [("value1",), ("value2",), ("value3",)]
        await storage._execute_many(
            "INSERT INTO test_table (value) VALUES (?)", params_list
        )

        rows = await storage._fetch_all("SELECT * FROM test_table ORDER BY id")
        assert len(rows) == 3
        assert rows[0]["value"] == "value1"
        assert rows[1]["value"] == "value2"
        assert rows[2]["value"] == "value3"

    @pytest.mark.asyncio
    async def test_execute_many_without_lock(self, storage) -> None:
        """Should execute many without lock when use_lock=False."""
        params_list = [("value1",), ("value2",)]
        await storage._execute_many(
            "INSERT INTO test_table (value) VALUES (?)",
            params_list,
            use_lock=False,
        )

        rows = await storage._fetch_all("SELECT * FROM test_table ORDER BY id")
        assert len(rows) == 2


class TestSqlStorageFetchVal:
    """Tests for _fetch_val method."""

    @pytest.mark.asyncio
    async def test_fetch_val_returns_value(self, storage) -> None:
        """Should return single value."""
        await storage._execute(
            "INSERT INTO test_table (value) VALUES (?)", ("test_value",)
        )

        result = await storage._fetch_val("SELECT value FROM test_table WHERE id = 1")

        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_fetch_val_returns_none_when_no_row(self, storage) -> None:
        """Should return None when no row found."""
        result = await storage._fetch_val("SELECT value FROM test_table WHERE id = 999")

        assert result is None


class TestSqlStorageConn:
    """Tests for conn property."""

    @pytest.mark.asyncio
    async def test_conn_returns_underlying_connection(self, storage, conn) -> None:
        """Should return the underlying connection."""
        result = storage.conn

        assert result is conn
