# mypy: disable-error-code="no-untyped-def"
"""Unit tests for SQLite connection management."""

from pathlib import Path

import pytest

from blacki.storage.sqlite import close_connection, create_connection


class TestCreateConnection:
    """Tests for create_connection function."""

    @pytest.mark.asyncio
    async def test_create_connection_creates_file(self, tmp_path: Path) -> None:
        """Should create database file and parent directories."""
        db_path = tmp_path / "subdir" / "test.db"

        conn = await create_connection(db_path)

        assert db_path.exists()
        assert db_path.parent.exists()

        await conn.close()

    @pytest.mark.asyncio
    async def test_create_connection_sets_row_factory(self, tmp_path: Path) -> None:
        """Should set row_factory to aiosqlite.Row."""
        db_path = tmp_path / "test.db"

        conn = await create_connection(db_path)

        assert conn.row_factory is not None

        await conn.close()

    @pytest.mark.asyncio
    async def test_create_connection_configures_pragmas(self, tmp_path: Path) -> None:
        """Should configure WAL mode and other pragmas."""
        db_path = tmp_path / "test.db"

        conn = await create_connection(db_path)

        async with conn.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0].lower() == "wal"

        async with conn.execute("PRAGMA foreign_keys") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1

        await conn.close()


class TestCloseConnection:
    """Tests for close_connection function."""

    @pytest.mark.asyncio
    async def test_close_connection_closes_connection(self, tmp_path: Path) -> None:
        """Should close the SQLite connection."""
        db_path = tmp_path / "test.db"
        conn = await create_connection(db_path)

        await close_connection(conn)

        with pytest.raises(ValueError, match="no active connection"):
            await conn.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_close_connection_with_memory_db(self) -> None:
        """Should close an in-memory connection."""
        import aiosqlite

        conn = await aiosqlite.connect(":memory:", isolation_level=None)

        await close_connection(conn)

        with pytest.raises(ValueError, match="no active connection"):
            await conn.execute("SELECT 1")
