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


def _album_message(message_id: int, group_id: str = "ordered") -> Message:
    """Build one small photo update for buffer state-machine tests."""
    return Message.model_validate(
        {
            "message_id": message_id,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": 123, "type": "private"},
            "media_group_id": group_id,
            "photo": [
                {
                    "file_id": f"photo-{message_id}",
                    "file_unique_id": f"unique-{message_id}",
                    "width": 10,
                    "height": 10,
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_buffer_sorts_and_deduplicates_updates(
    buffer: AlbumBuffer, on_flush: MagicMock
) -> None:
    """Polling retries and task scheduling cannot reorder or duplicate images."""
    messages = [_album_message(30), _album_message(10), _album_message(30)]
    tasks = [
        asyncio.create_task(buffer.add_message(message, seq))
        for seq, message in enumerate(messages, start=1)
    ]
    await asyncio.sleep(0)

    album = buffer._buffers[(123, None, "ordered")]
    assert [message.message_id for message in album.messages] == [10, 30]

    buffer._flush(album)
    assert album.future is not None
    album.future.set_result(None)
    await asyncio.gather(*tasks)
    assert on_flush.call_count == 1
    flushed = on_flush.call_args.args[0]
    assert [message.message_id for message in flushed.messages] == [10, 30]
    buffer.mark_completed(flushed)

    # A duplicate arriving after the turn was scheduled is ignored as well.
    await buffer.add_message(_album_message(10), 4)
    assert on_flush.call_count == 1

    # A genuinely late member reuses the completed key and starts a fresh
    # bounded buffer, while the duplicate above does not.
    late_task = asyncio.create_task(buffer.add_message(_album_message(20), 5))
    await asyncio.sleep(0)
    late_album = buffer._buffers[(123, None, "ordered")]
    buffer._flush(late_album)
    assert late_album.future is not None
    late_album.future.set_result(None)
    await late_task
    assert on_flush.call_count == 2


@pytest.mark.asyncio
async def test_flushed_album_can_be_retried_after_cancellation(
    buffer: AlbumBuffer, on_flush: MagicMock
) -> None:
    """Cancellation clears in-flight IDs so Telegram can retry the album."""
    task = asyncio.create_task(buffer.add_message(_album_message(1, "retry"), 1))
    await asyncio.sleep(0)
    album = buffer._buffers[(123, None, "retry")]
    buffer._flush(album)
    await buffer.add_message(_album_message(1, "retry"), 2)
    assert on_flush.call_count == 1
    buffer.mark_retryable(album)
    assert album.future is not None
    album.future.set_result(None)
    await task

    retry_task = asyncio.create_task(buffer.add_message(_album_message(1, "retry"), 2))
    await asyncio.sleep(0)
    retry_album = buffer._buffers[(123, None, "retry")]
    assert retry_album is not album
    buffer._flush(retry_album)
    assert retry_album.future is not None
    retry_album.future.set_result(None)
    await retry_task
    assert on_flush.call_count == 2


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

    # A duplicate in an existing album with no future returns immediately.
    await buffer.add_message(msg, 2)
    assert len(album.messages) == 1


@pytest.mark.asyncio
async def test_add_message_cleans_up_on_watchdog_schedule_failure(
    buffer: AlbumBuffer,
) -> None:
    """If scheduling the max-wait watchdog raises, the album must not be
    left orphaned in self._buffers with no timer to ever flush it
    (Github issue #163, item 2)."""
    msg = Message.model_validate(
        {
            "message_id": 1,
            "date": "2024-01-01T00:00:00Z",
            "chat": {"id": 123, "type": "private"},
            "media_group_id": "schedule-fail",
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

    def _fail_to_schedule(coro):
        coro.close()  # avoid a "coroutine was never awaited" warning
        raise RuntimeError("event loop is shutting down")

    with (
        patch(
            "blacki.telegram.album_buffer.asyncio.create_task",
            side_effect=_fail_to_schedule,
        ),
        pytest.raises(RuntimeError, match="event loop is shutting down"),
    ):
        await buffer.add_message(msg, 1)

    assert (123, None, "schedule-fail") not in buffer._buffers


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

    with patch("blacki.telegram.album_buffer._MAX_COMPLETED_ALBUMS", 1):
        for group_id in ("evict-1", "evict-2"):
            album = _BufferedAlbum(
                chat_id=123,
                message_thread_id=None,
                media_group_id=group_id,
                chat_type=ChatType.PRIVATE,
                messages=[_album_message(1, group_id)],
            )
            buffer._buffers[(123, None, group_id)] = album
            buffer._flush(album)
            buffer.mark_completed(album)
        assert list(buffer._completed_message_ids) == [(123, None, "evict-2")]


def test_flush_is_idempotent_against_debounce_max_wait_race(
    buffer: AlbumBuffer, on_flush: MagicMock
) -> None:
    """Both the debounce callback and the max-wait task call _flush on the
    same album; this pins _flush's check-then-set as a single synchronous
    call with no ``await`` in between, so calling it twice back-to-back
    (as if both timers fired in the same event-loop iteration) must flush
    exactly once (Github issue #163, item 4)."""
    album = _BufferedAlbum(
        chat_id=123,
        message_thread_id=None,
        media_group_id="race",
        chat_type=ChatType.PRIVATE,
        messages=[],
    )
    buffer._buffers[(123, None, "race")] = album

    buffer._on_debounce_expired(album)
    buffer._flush(album)

    assert on_flush.call_count == 1
