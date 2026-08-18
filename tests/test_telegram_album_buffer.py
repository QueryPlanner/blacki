# mypy: ignore-errors
"""Unit tests for the standalone Telegram album-buffering mechanics.

Full end-to-end album flows (photos arriving via polling, debounce/flush
timing, turn processing) are covered as TelegramBot integration tests in
test_telegram_bot.py. This file only covers AlbumBuffer's own buffering
state machine in isolation.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from blacki.telegram.album_buffer import AlbumBuffer, _BufferedAlbum
from blacki.telegram.types import ChatType, Message


@pytest.fixture
def on_flush() -> MagicMock:
    return MagicMock()


@pytest.fixture
def buffer(on_flush: MagicMock) -> AlbumBuffer:
    return AlbumBuffer(on_flush=on_flush)


@pytest.mark.asyncio
async def test_cleanup_album_buffer_branches(buffer: AlbumBuffer) -> None:
    """Test cleanup() when handles are None or future is done/None."""
    loop = asyncio.get_running_loop()
    done_future: asyncio.Future[None] = loop.create_future()
    done_future.set_result(None)

    album1 = _BufferedAlbum(
        chat_id=123,
        message_thread_id=None,
        media_group_id="clean-1",
        chat_type=ChatType.PRIVATE,
        messages=[],
        debounce_handle=None,
        max_wait_task=None,
        future=done_future,
    )
    buffer._buffers[(123, None, "clean-1")] = album1
    buffer.cleanup(album1)
    assert (123, None, "clean-1") not in buffer._buffers

    album2 = _BufferedAlbum(
        chat_id=123,
        message_thread_id=None,
        media_group_id="clean-2",
        chat_type=ChatType.PRIVATE,
        messages=[],
        debounce_handle=None,
        max_wait_task=None,
        future=None,
    )
    buffer._buffers[(123, None, "clean-2")] = album2
    buffer.cleanup(album2)
    assert (123, None, "clean-2") not in buffer._buffers


@pytest.mark.asyncio
async def test_buffer_album_message_branches(buffer: AlbumBuffer) -> None:
    """Test add_message() when debounce_handle or future is None."""
    msg = Message.model_validate(
        {
            "message_id": 1,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": 123, "type": "private"},
            "media_group_id": "branch-buf",
            "photo": [
                {
                    "file_id": "p1",
                    "file_unique_id": "u1",
                    "width": 10,
                    "height": 10,
                }
            ],
        }
    )

    album = _BufferedAlbum(
        chat_id=123,
        message_thread_id=None,
        media_group_id="branch-buf",
        chat_type=ChatType.PRIVATE,
        messages=[],
        debounce_handle=None,
        future=None,
    )
    buffer._buffers[(123, None, "branch-buf")] = album

    await buffer.add_message(msg, 1)
    assert len(album.messages) == 1
    assert album.debounce_handle is not None
    album.debounce_handle.cancel()


@pytest.mark.asyncio
async def test_album_max_wait_triggers_flush(buffer: AlbumBuffer) -> None:
    """Test _max_wait completing its sleep and triggering flush."""
    album = _BufferedAlbum(
        chat_id=123,
        message_thread_id=None,
        media_group_id="max-wait-flush",
        chat_type=ChatType.PRIVATE,
        messages=[],
    )
    buffer._buffers[(123, None, "max-wait-flush")] = album

    with patch("blacki.telegram.album_buffer._ALBUM_MAX_WAIT_SECONDS", 0.01):
        await buffer._max_wait(album)

    assert album.processed is True
    assert (123, None, "max-wait-flush") not in buffer._buffers


@pytest.mark.asyncio
async def test_flush_album_branches(buffer: AlbumBuffer) -> None:
    """Test _flush with an already-processed album and None handles."""
    album_processed = _BufferedAlbum(
        chat_id=123,
        message_thread_id=None,
        media_group_id="flushed-already",
        chat_type=ChatType.PRIVATE,
        messages=[],
        processed=True,
    )
    buffer._flush(album_processed)

    album_none_handles = _BufferedAlbum(
        chat_id=123,
        message_thread_id=None,
        media_group_id="none-handles",
        chat_type=ChatType.PRIVATE,
        messages=[],
        debounce_handle=None,
        max_wait_task=None,
    )
    buffer._flush(album_none_handles)
    assert album_none_handles.processed is True
