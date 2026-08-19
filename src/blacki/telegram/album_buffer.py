"""Buffering for Telegram media-group (photo album) messages.

Telegram delivers a multi-photo album as several separate messages that
share a ``media_group_id``, arriving in quick succession rather than as one
message. This buffers those messages, waits for a short debounce window (or
a max-wait cap, in case delivery stalls) for the rest of the album to
arrive, then hands the complete album to a caller-supplied flush callback.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from .types import ChatType, Message

logger = logging.getLogger(__name__)

_ALBUM_DEBOUNCE_SECONDS = 0.5
_ALBUM_MAX_WAIT_SECONDS = 2.0


@dataclass(slots=True)
class _BufferedAlbum:
    """In-memory buffer for a Telegram media group (photo album)."""

    chat_id: int
    message_thread_id: int | None
    media_group_id: str
    chat_type: ChatType | None
    messages: list[Message]
    debounce_handle: asyncio.TimerHandle | None = None
    max_wait_task: asyncio.Task[None] | None = None
    future: asyncio.Future[None] | None = None
    processed: bool = False
    created_seq: int = 0


class AlbumBuffer:
    """Buffers album messages and flushes complete albums via a callback.

    Owns no turn-processing or session logic: ``on_flush`` is called
    synchronously with the completed ``_BufferedAlbum`` once its debounce
    window elapses (or the max-wait cap is hit), and the caller decides how
    to process it.
    """

    def __init__(self, on_flush: Callable[[_BufferedAlbum], None]) -> None:
        self._on_flush = on_flush
        self._buffers: dict[tuple[int, int | None, str], _BufferedAlbum] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()

    def get_active(
        self, chat_id: int, message_thread_id: int | None
    ) -> list[_BufferedAlbum]:
        """Return active (not yet flushed) album buffers for a conversation."""
        return [
            album
            for (cid, tid, _), album in self._buffers.items()
            if cid == chat_id and tid == message_thread_id and not album.processed
        ]

    async def add_message(self, message: Message, seq: int) -> None:
        """Buffer an incoming album message; wait until its album is flushed."""
        chat_id = message.chat.id
        message_thread_id = message.message_thread_id
        media_group_id = cast(str, message.media_group_id)
        key = (chat_id, message_thread_id, media_group_id)

        album = self._buffers.get(key)
        if album is None:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[None] = loop.create_future()
            album = _BufferedAlbum(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                media_group_id=media_group_id,
                chat_type=message.chat.type,
                messages=[message],
                future=future,
                created_seq=seq,
            )
            self._buffers[key] = album

            max_wait_task = asyncio.create_task(self._max_wait(album))
            self._background_tasks.add(max_wait_task)
            max_wait_task.add_done_callback(self._background_tasks.discard)
            album.max_wait_task = max_wait_task
        else:
            album.messages.append(message)
            if album.debounce_handle is not None:
                album.debounce_handle.cancel()

        loop = asyncio.get_running_loop()
        album.debounce_handle = loop.call_later(
            _ALBUM_DEBOUNCE_SECONDS,
            self._on_debounce_expired,
            album,
        )

        try:
            if album.future is not None:
                await asyncio.shield(album.future)
        except asyncio.CancelledError:
            raise

    async def _max_wait(self, album: _BufferedAlbum) -> None:
        """Flush the album after the maximum wait timeout."""
        try:
            await asyncio.sleep(_ALBUM_MAX_WAIT_SECONDS)
            self._flush(album)
        except asyncio.CancelledError:
            pass

    def _on_debounce_expired(self, album: _BufferedAlbum) -> None:
        """Handle debounce timer expiration by flushing the album."""
        self._flush(album)

    def _flush(self, album: _BufferedAlbum) -> None:
        """Mark an album processed, remove it from the buffer, and flush it."""
        if album.processed:
            return
        album.processed = True

        key = (album.chat_id, album.message_thread_id, album.media_group_id)
        self._buffers.pop(key, None)

        if album.debounce_handle is not None:
            album.debounce_handle.cancel()
            album.debounce_handle = None
        if album.max_wait_task is not None:
            album.max_wait_task.cancel()
            album.max_wait_task = None

        self._on_flush(album)

    def cleanup(self, album: _BufferedAlbum) -> None:
        """Cancel timers and tasks and release a buffered album."""
        key = (album.chat_id, album.message_thread_id, album.media_group_id)
        self._buffers.pop(key, None)

        if album.debounce_handle is not None:
            album.debounce_handle.cancel()
            album.debounce_handle = None

        if album.max_wait_task is not None:
            album.max_wait_task.cancel()
            album.max_wait_task = None

        if album.future is not None and not album.future.done():
            album.future.cancel()

    async def shutdown(self) -> None:
        """Cancel and release all currently buffered albums and their tasks."""
        for album in list(self._buffers.values()):
            self.cleanup(album)

        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
