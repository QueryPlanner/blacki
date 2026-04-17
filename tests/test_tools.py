"""Unit tests for custom tools."""

import logging
import uuid
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import mock classes from conftest
from conftest import MockState, MockToolContext
from google.adk.tools import ToolContext
from pydantic import BaseModel

from blacki.tools import (
    _pending_schemas,
    _pending_tasks,
    _serialize_browser_output,
    browser_get_task_status,
    browser_list_profiles,
    browser_stop_session,
    browser_task,
    example_tool,
)


@pytest.fixture(autouse=True)
def clear_pending_tasks() -> None:
    """Clear pending task state before each test."""
    _pending_tasks.clear()
    _pending_schemas.clear()


class TestExampleTool:
    """Tests for the example_tool function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    def test_example_tool_returns_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool returns success status and message."""
        caplog.set_level(logging.INFO)

        state = MockState({"user_id": "test_user", "session_key": "value"})
        tool_context = cast(ToolContext, MockToolContext(state=state))

        result = example_tool(tool_context)

        assert result["status"] == "success"
        assert result["message"] == "Successfully used example_tool."

    def test_example_tool_logs_state_keys(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool logs session state keys."""
        caplog.set_level(logging.INFO)

        state = MockState({"key1": "value1", "key2": "value2"})
        tool_context = cast(ToolContext, MockToolContext(state=state))

        example_tool(tool_context)

        assert "Session state keys:" in caplog.text
        assert "Successfully used example_tool." in caplog.text

        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert len(info_records) == 2

    def test_example_tool_with_empty_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool handles empty state correctly."""
        caplog.set_level(logging.INFO)

        result = example_tool(self._tool_context())

        assert result["status"] == "success"
        assert result["message"] == "Successfully used example_tool."

        assert "Session state keys:" in caplog.text


class _SampleOutputModel(BaseModel):
    """Minimal model for browser_task structured-output tests."""

    title: str


class TestSerializeBrowserOutput:
    """Tests for ``_serialize_browser_output`` (Browser Use result shaping)."""

    def test_serializes_list_of_models(self) -> None:
        """Lists of Pydantic models become JSON-ready dict lists."""
        rows = [_SampleOutputModel(title="a"), _SampleOutputModel(title="b")]
        assert _serialize_browser_output(rows) == [
            {"title": "a"},
            {"title": "b"},
        ]

    def test_serializes_dict_with_nested_model(self) -> None:
        """Dict values that are models are serialized recursively."""
        payload = {"item": _SampleOutputModel(title="x"), "n": 1}
        assert _serialize_browser_output(payload) == {"item": {"title": "x"}, "n": 1}


class TestBrowserTask:
    """Tests for Browser Use Cloud ``browser_task``."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_browser_task_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return a clear error payload."""
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await browser_task("open example.com", tool_context)

        assert result["status"] == "error"
        assert result["task_id"] is None
        assert "BROWSER_USE_API_KEY" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_browser_task_empty_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace-only task should be rejected."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        result = await browser_task("  \n  ", tool_context)

        assert result["status"] == "error"
        assert "non-empty" in (result.get("error") or "").lower()

    @pytest.mark.asyncio
    async def test_browser_task_invalid_output_schema_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject unsupported ``output_schema`` types."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        result = await browser_task(
            "do something",
            tool_context,
            output_schema="not-valid",
        )

        assert result["status"] == "error"
        assert "output_schema" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_browser_task_returns_pending_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """browser_task should return immediately with pending status."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_session = MagicMock()
        mock_session.live_url = "https://live.example/session"

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=mock_session)

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_task("open example.com", tool_context)

        assert result["status"] == "pending"
        assert result["task_id"] == str(tid)
        assert result["session_id"] == str(sid)
        assert result["live_preview_url"] == "https://live.example/session"
        assert "browser_get_task_status" in result.get("message", "")

        assert str(tid) in _pending_tasks
        assert _pending_tasks[str(tid)]["session_id"] == str(sid)

    @pytest.mark.asyncio
    async def test_browser_task_passes_keep_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """keep_alive=True should be passed to create."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("open example.com", tool_context, keep_alive=True)

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("keepAlive") is True

    @pytest.mark.asyncio
    async def test_browser_task_passes_session_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """session_id should be passed to create for follow-up tasks."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("continue task", tool_context, session_id=str(sid))

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("session_id") == str(sid)

    @pytest.mark.asyncio
    async def test_browser_task_passes_profile_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """profile_id should be passed via session_settings."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()
        profile_uuid = str(uuid.uuid4())

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("login to site", tool_context, profile_id=profile_uuid)

        _args, kwargs = mock_client.tasks.create.call_args
        session_settings = kwargs.get("session_settings")
        assert session_settings is not None
        assert str(session_settings.profile_id) == profile_uuid

    @pytest.mark.asyncio
    async def test_browser_task_passes_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """model parameter should override default LLM."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task(
                "open example.com", tool_context, model="claude-sonnet-4.6"
            )

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("llm") == "claude-sonnet-4.6"

    @pytest.mark.asyncio
    async def test_browser_task_passes_json_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dict output_schema should be sent as structured_output."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("extract x", tool_context, output_schema=schema)

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("structured_output") is not None
        assert "integer" in kwargs["structured_output"]

    @pytest.mark.asyncio
    async def test_browser_task_stores_pydantic_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pydantic model should be stored in _pending_schemas for later polling."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task(
                "get title",
                tool_context,
                output_schema=_SampleOutputModel,
            )

        assert str(tid) in _pending_schemas
        assert _pending_schemas[str(tid)] == _SampleOutputModel


class TestBrowserGetTaskStatus:
    """Tests for ``browser_get_task_status``."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_get_status_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return error."""
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await browser_get_task_status("task-123", tool_context)

        assert result["status"] == "error"
        assert "BROWSER_USE_API_KEY" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_get_status_unknown_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown task_id should return error."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        result = await browser_get_task_status("unknown-task-id", tool_context)

        assert result["status"] == "error"
        assert "Unknown task_id" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_get_status_returns_running_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Timeout during poll should return running status."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        tid = str(uuid.uuid4())
        _pending_tasks[tid] = {
            "api_key": "bu_test",
            "session_id": "session-123",
            "live_preview_url": "https://live.example/session",
            "status": "pending",
        }
        _pending_schemas[tid] = None

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._poll_task_output",
                side_effect=TimeoutError("poll timeout"),
            ),
        ):
            result = await browser_get_task_status(tid, tool_context)

        assert result["status"] == "running"
        assert result["task_id"] == tid
        assert result["output"] is None
        assert "still running" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_get_status_returns_finished_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finished task should return output and clean up pending state."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        tid = str(uuid.uuid4())
        _pending_tasks[tid] = {
            "api_key": "bu_test",
            "session_id": "session-123",
            "live_preview_url": "https://live.example/session",
            "status": "pending",
        }
        _pending_schemas[tid] = None

        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.is_success = True

        mock_result = MagicMock()
        mock_result.task = mock_task
        mock_result.output = '{"ok": true}'

        mock_client = MagicMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._poll_task_output",
                new=AsyncMock(return_value=mock_result),
            ),
        ):
            result = await browser_get_task_status(tid, tool_context)

        assert result["status"] == "finished"
        assert result["output"] == {"ok": True}
        assert result["is_success"] is True
        assert tid not in _pending_tasks
        assert tid not in _pending_schemas

    @pytest.mark.asyncio
    async def test_get_status_uses_stored_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Poll should use stored Pydantic schema for typed output."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        tid = str(uuid.uuid4())
        _pending_tasks[tid] = {
            "api_key": "bu_test",
            "session_id": "session-123",
            "live_preview_url": None,
            "status": "pending",
        }
        _pending_schemas[tid] = _SampleOutputModel

        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.is_success = True

        mock_result = MagicMock()
        mock_result.task = mock_task
        mock_result.output = _SampleOutputModel(title="test-title")

        mock_client = MagicMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._poll_task_output",
                new=AsyncMock(return_value=mock_result),
            ) as mock_poll,
        ):
            result = await browser_get_task_status(tid, tool_context)

        mock_poll.assert_awaited_once()
        call_args = mock_poll.call_args
        assert call_args[0][2] == _SampleOutputModel
        assert result["output"] == {"title": "test-title"}


class TestBrowserStopSession:
    """Tests for ``browser_stop_session``."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_stop_session_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return error."""
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await browser_stop_session("session-123", tool_context)

        assert result["status"] == "error"
        assert "BROWSER_USE_API_KEY" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_stop_session_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: delete session and return success."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.sessions.delete = AsyncMock(return_value=None)

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_stop_session("session-123", tool_context)

        assert result["status"] == "success"
        assert result["session_id"] == "session-123"
        mock_client.sessions.delete.assert_awaited_once_with("session-123")


class TestBrowserListProfiles:
    """Tests for ``browser_list_profiles``."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_list_profiles_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return error with empty profiles."""
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await browser_list_profiles(tool_context)

        assert result["status"] == "error"
        assert result["profiles"] == []
        assert "BROWSER_USE_API_KEY" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_list_profiles_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: return list of profiles."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        mock_profile = MagicMock()
        mock_profile.id = uuid.uuid4()
        mock_profile.name = "test-profile"

        mock_response = MagicMock()
        mock_response.profiles = [mock_profile]

        mock_client = MagicMock()
        mock_client.profiles.list = AsyncMock(return_value=mock_response)

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_list_profiles(tool_context)

        assert result["status"] == "success"
        assert len(result["profiles"]) == 1
        assert result["profiles"][0]["name"] == "test-profile"

    @pytest.mark.asyncio
    async def test_list_profiles_with_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Query parameter should be passed to API."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        mock_response = MagicMock()
        mock_response.profiles = []

        mock_client = MagicMock()
        mock_client.profiles.list = AsyncMock(return_value=mock_response)

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_list_profiles(tool_context, query="my-profile")

        mock_client.profiles.list.assert_awaited_once_with(query="my-profile")
