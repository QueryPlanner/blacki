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
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from .types import ChatType, Message

logger = logging.getLogger(__name__)

_ALBUM_DEBOUNCE_SECONDS = 1.0
# This is a safety net for a *stalled* album, not a normal-path flush
# trigger: the debounce timer above already extends correctly on every
# arrival, so a healthy album -- even one delivered slowly -- flushes via
# debounce shortly after its last message. This cap only exists so a
# media group that never fully arrives (e.g. Telegram never delivers the
# last item) can't buffer forever and starve the conversation's next
# turn. 15s tolerates a slow upload of Telegram's max 10-photo media
# group (a few seconds per photo) without leaving a genuinely stuck
# conversation waiting anywhere near that long.
_ALBUM_MAX_WAIT_SECONDS = 15.0
_MAX_COMPLETED_ALBUMS = 256


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
        self._inflight_message_ids: dict[tuple[int, int | None, str], set[int]] = {}
        self._completed_message_ids: OrderedDict[
            tuple[int, int | None, str], set[int]
        ] = OrderedDict()

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
        """Buffer an incoming album message until its album is flushed."""
        chat_id = message.chat.id
        message_thread_id = message.message_thread_id
        media_group_id = cast(str, message.media_group_id)
        key = (chat_id, message_thread_id, media_group_id)

        album = self._buffers.get(key)
        inflight_ids = self._inflight_message_ids.get(key)
        if (
            album is None
            and inflight_ids is not None
            and message.message_id in inflight_ids
        ):
            # The original update is still being processed. Do not create a
            # second turn for a Telegram retry while it is in flight.
            return
        completed_ids = self._completed_message_ids.get(key)
        if album is None and completed_ids is not None:
            if message.message_id in completed_ids:
                # Telegram retries can deliver an already flushed update after
                # the original task has completed. It must not create a second
                # model turn.
                return
            self._completed_message_ids.move_to_end(key)

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

            try:
                max_wait_task = asyncio.create_task(self._max_wait(album))
            except Exception:
                # Scheduling failed before any timer exists to flush this
                # album later, so don't leave it orphaned in self._buffers.
                self._buffers.pop(key, None)
                raise
            self._background_tasks.add(max_wait_task)
            max_wait_task.add_done_callback(self._background_tasks.discard)
            album.max_wait_task = max_wait_task
        else:
            if any(item.message_id == message.message_id for item in album.messages):
                if album.future is not None:
                    await asyncio.shield(album.future)
                return
            album.messages.append(message)
            if album.debounce_handle is not None:
                album.debounce_handle.cancel()

        album.messages.sort(key=lambda item: item.message_id)

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
        """Mark an album processed, remove it from the buffer, and flush it.

        Both the debounce callback and the max-wait task call this method,
        so the check-then-set on ``album.processed`` below is the only guard
        against double-flushing. It is race-free only because this method
        is synchronous with no ``await`` before ``album.processed = True``:
        the event loop cannot interleave the check and the set. Do not add
        an ``await`` before that line, or make this method ``async``,
        without replacing the guard with something that is still atomic
        against both callers.
        """
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

        key_message_ids = {message.message_id for message in album.messages}
        if key_message_ids:
            self._inflight_message_ids[key] = key_message_ids

        self._on_flush(album)

    def mark_completed(self, album: _BufferedAlbum) -> None:
        """Suppress retries after the flushed album finishes successfully."""
        key = (album.chat_id, album.message_thread_id, album.media_group_id)
        message_ids = self._inflight_message_ids.pop(key, None)
        if not message_ids:
            return
        self._completed_message_ids[key] = message_ids
        self._completed_message_ids.move_to_end(key)
        while len(self._completed_message_ids) > _MAX_COMPLETED_ALBUMS:
            self._completed_message_ids.popitem(last=False)

    def mark_retryable(self, album: _BufferedAlbum) -> None:
        """Allow a retry after a flushed album is cancelled or fails."""
        key = (album.chat_id, album.message_thread_id, album.media_group_id)
        self._inflight_message_ids.pop(key, None)

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
