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

from blacki.tools import _serialize_browser_output, browser_task, example_tool


class TestExampleTool:
    """Tests for the example_tool function."""

    def test_example_tool_returns_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool returns success status and message."""
        # Setup logging to capture INFO level
        caplog.set_level(logging.INFO)

        # Create mock tool context with state
        state = MockState({"user_id": "test_user", "session_key": "value"})
        tool_context = MockToolContext(state=state)

        # Execute tool
        result = example_tool(tool_context)  # type: ignore

        # Verify return value
        assert result["status"] == "success"
        assert result["message"] == "Successfully used example_tool."

    def test_example_tool_logs_state_keys(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool logs session state keys."""
        # Setup logging to capture INFO level
        caplog.set_level(logging.INFO)

        # Create mock tool context with state
        state = MockState({"key1": "value1", "key2": "value2"})
        tool_context = MockToolContext(state=state)

        # Execute tool
        example_tool(tool_context)  # type: ignore

        # Verify logging
        assert "Session state keys:" in caplog.text
        assert "Successfully used example_tool." in caplog.text

        # Verify INFO level was used
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert len(info_records) == 2

    def test_example_tool_with_empty_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool handles empty state correctly."""
        # Setup logging
        caplog.set_level(logging.INFO)

        # Create mock tool context with empty state
        state = MockState({})
        tool_context = MockToolContext(state=state)

        # Execute tool
        result = example_tool(tool_context)  # type: ignore

        # Verify success even with empty state
        assert result["status"] == "success"
        assert result["message"] == "Successfully used example_tool."

        # Verify logging occurred
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
        assert result["output"] is None
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
    async def test_browser_task_success_with_mocked_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: create task, fetch live URL, then poll for completion."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()
        mock_task = MagicMock()
        mock_task.session_id = sid
        mock_task.id = tid
        mock_task.status.value = "finished"
        mock_task.is_success = True
        mock_result = MagicMock()
        mock_result.task = mock_task
        mock_result.output = '{"ok": true}'

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_session = MagicMock()
        mock_session.live_url = "https://live.example/session"

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=mock_session)
        mock_client.close = AsyncMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._async_poll_output",
                new=AsyncMock(return_value=mock_result),
            ) as mock_poll,
        ):
            result = await browser_task("open example.com", tool_context)

        assert result["status"] == "finished"
        assert result["output"] == {"ok": True}
        assert result["live_preview_url"] == "https://live.example/session"
        assert result["session_id"] == str(sid)
        assert result["task_id"] == str(tid)
        assert result["is_success"] is True
        mock_client.tasks.create.assert_awaited_once()
        mock_client.sessions.get.assert_awaited_once_with(str(sid))
        mock_poll.assert_awaited_once_with(
            mock_client.tasks,
            str(tid),
            None,
        )

    @pytest.mark.asyncio
    async def test_browser_task_passes_json_schema_to_create(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dict ``output_schema`` should be sent as ``structured_output`` on create."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        sid = uuid.uuid4()
        tid = uuid.uuid4()
        mock_task = MagicMock()
        mock_task.session_id = sid
        mock_task.id = tid
        mock_task.status.value = "finished"
        mock_task.is_success = True
        mock_result = MagicMock()
        mock_result.task = mock_task
        mock_result.output = '{"x": 1}'

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))
        mock_client.close = AsyncMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._async_poll_output",
                new=AsyncMock(return_value=mock_result),
            ),
        ):
            await browser_task(
                "extract x",
                tool_context,
                output_schema=schema,
            )

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("structured_output") is not None
        assert "integer" in kwargs["structured_output"]

    @pytest.mark.asyncio
    async def test_browser_task_passes_pydantic_schema_to_poll(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pydantic model is passed to the poll step for typed output parsing."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()
        mock_task = MagicMock()
        mock_task.session_id = sid
        mock_task.id = tid
        mock_task.status.value = "finished"
        mock_task.is_success = True
        mock_result = MagicMock()
        mock_result.task = mock_task
        mock_result.output = _SampleOutputModel(title="hi")

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))
        mock_client.close = AsyncMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._async_poll_output",
                new=AsyncMock(return_value=mock_result),
            ) as mock_poll,
        ):
            result = await browser_task(
                "get title",
                tool_context,
                output_schema=_SampleOutputModel,
            )

        _args, kwargs = mock_client.tasks.create.call_args
        assert "structured_output" in kwargs
        assert _SampleOutputModel.__name__ in kwargs["structured_output"]
        mock_poll.assert_awaited_once_with(
            mock_client.tasks,
            str(tid),
            _SampleOutputModel,
        )
        assert result["output"] == {"title": "hi"}
