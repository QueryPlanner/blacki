"""Streaming message management for real-time Telegram updates.

Implements time-based throttling (300ms per channel) for streaming updates,
with separate strategies for private chats (sendMessageDraft) and
group/supergroup chats (editMessageText-based streaming).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from blacki.adk_runtime import StreamChunk

from .api import TelegramApiClient, TelegramApiError
from .formatting import escape_markdown, format_for_telegram
from .types import ChatType, ParseMode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DRAFT_UPDATE_INTERVAL_SEC = 0.3
TELEGRAM_MESSAGE_LIMIT = 4096


def _default_id_factory() -> int:
    return secrets.randbits(63) or 1


class _Channel(ABC):
    """Abstract base for streaming message channels.

    Each channel manages a single Telegram message that is created lazily
    on first non-empty content and updated throttled at 300ms intervals.
    """

    def __init__(
        self,
        api: TelegramApiClient,
        chat_id: int,
        message_thread_id: int | None,
        update_interval_sec: float,
    ) -> None:
        self.api = api
        self.chat_id = chat_id
        self.message_thread_id = message_thread_id
        self.update_interval_sec = update_interval_sec
        self._last_write_time = 0.0
        self._full_text = ""
        self._wrote_successfully = False

    async def write(self, text: str, *, is_final: bool) -> None:
        """Write text to the channel with throttling.

        Args:
            text: The text content to write.
            is_final: If True, flush immediately regardless of throttle.
        """
        if not text:
            return

        self._full_text = text

        now = time.monotonic()
        elapsed = now - self._last_write_time

        if elapsed < self.update_interval_sec and not is_final:
            return

        await self._write_throttled(text[:TELEGRAM_MESSAGE_LIMIT], is_final=is_final)
        self._last_write_time = now

    async def finalize(self) -> None:
        """Finalize the channel, handling long messages and fallback."""
        if not self._full_text:
            return

        if len(self._full_text) <= TELEGRAM_MESSAGE_LIMIT:
            if not self._wrote_successfully:
                await self._write_throttled(self._full_text, is_final=True)
            return

        chunks = split_long_message(self._full_text)
        if not chunks:
            return

        if self._wrote_successfully:
            await self._write_throttled(chunks[0], is_final=True)
        else:
            await self._write_throttled(chunks[0], is_final=True)

        for chunk in chunks[1:]:
            try:
                await self.api.send_message(
                    chat_id=self.chat_id,
                    text=chunk,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=self.message_thread_id,
                )
            except TelegramApiError:
                logger.exception("Failed to send additional message chunk")

    @abstractmethod
    async def _write_throttled(self, text: str, *, is_final: bool) -> None:
        """Write text to the channel (throttling already handled)."""
        ...

    async def _send_fallback(self, text: str) -> None:
        """Send a fallback message if all writes failed."""
        if not text:
            return
        try:
            await self.api.send_message(
                chat_id=self.chat_id,
                text=text[:TELEGRAM_MESSAGE_LIMIT],
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=self.message_thread_id,
            )
            logger.info("Sent fallback message after streaming failures")
        except TelegramApiError:
            logger.exception("Failed to send fallback message")


class _DraftChannel(_Channel):
    """Channel for private chats using sendMessageDraft.

    Uses a single draft_id for all updates to the same message.
    Telegram converts the draft to a regular message on the final call.
    """

    def __init__(
        self,
        api: TelegramApiClient,
        chat_id: int,
        message_thread_id: int | None,
        update_interval_sec: float,
        id_factory: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(api, chat_id, message_thread_id, update_interval_sec)
        self._draft_id = (id_factory or _default_id_factory)()

    async def _write_throttled(self, text: str, *, is_final: bool) -> None:
        try:
            await self.api.send_message_draft(
                chat_id=self.chat_id,
                text=text,
                draft_id=self._draft_id,
                message_thread_id=self.message_thread_id,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            self._wrote_successfully = True
        except TelegramApiError as e:
            await self._handle_write_error(e, text, is_final=is_final)

    async def _handle_write_error(
        self, error: TelegramApiError, text: str, *, is_final: bool
    ) -> None:
        if error.error_code == 429 and error.retry_after:
            await asyncio.sleep(error.retry_after)
            try:
                await self.api.send_message_draft(
                    chat_id=self.chat_id,
                    text=text,
                    draft_id=self._draft_id,
                    message_thread_id=self.message_thread_id,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                self._wrote_successfully = True
                return
            except TelegramApiError:
                logger.warning("Retry after rate limit failed")

        if 500 <= (error.error_code or 0) <= 599:
            await asyncio.sleep(1.0)
            try:
                await self.api.send_message_draft(
                    chat_id=self.chat_id,
                    text=text,
                    draft_id=self._draft_id,
                    message_thread_id=self.message_thread_id,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                self._wrote_successfully = True
                return
            except TelegramApiError:
                logger.warning("Retry after server error failed")

        logger.warning("Draft write failed: %s", error)

        if is_final and not self._wrote_successfully:
            await self._send_fallback(text)


class _EditChannel(_Channel):
    """Channel for group/supergroup chats using editMessageText.

    Creates a message on first write, then edits it for subsequent updates.
    """

    def __init__(
        self,
        api: TelegramApiClient,
        chat_id: int,
        message_thread_id: int | None,
        update_interval_sec: float,
    ) -> None:
        super().__init__(api, chat_id, message_thread_id, update_interval_sec)
        self._message_id: int | None = None

    async def _write_throttled(self, text: str, *, is_final: bool) -> None:
        if self._message_id is None:
            try:
                message = await self.api.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    message_thread_id=self.message_thread_id,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                self._message_id = message.message_id
                self._wrote_successfully = True
                return
            except TelegramApiError as e:
                await self._handle_write_error(e, text, is_final=is_final)
                return

        try:
            await self.api.edit_message_text(
                chat_id=self.chat_id,
                message_id=self._message_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            self._wrote_successfully = True
        except TelegramApiError as e:
            await self._handle_write_error(e, text, is_final=is_final)

    async def _handle_write_error(
        self, error: TelegramApiError, text: str, *, is_final: bool
    ) -> None:
        if error.error_code == 429 and error.retry_after:
            await asyncio.sleep(error.retry_after)
            try:
                if self._message_id is None:
                    message = await self.api.send_message(
                        chat_id=self.chat_id,
                        text=text,
                        message_thread_id=self.message_thread_id,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                    self._message_id = message.message_id
                else:
                    await self.api.edit_message_text(
                        chat_id=self.chat_id,
                        message_id=self._message_id,
                        text=text,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                self._wrote_successfully = True
                return
            except TelegramApiError:
                logger.warning("Retry after rate limit failed")

        if 500 <= (error.error_code or 0) <= 599:
            await asyncio.sleep(1.0)
            try:
                if self._message_id is None:
                    message = await self.api.send_message(
                        chat_id=self.chat_id,
                        text=text,
                        message_thread_id=self.message_thread_id,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                    self._message_id = message.message_id
                else:
                    await self.api.edit_message_text(
                        chat_id=self.chat_id,
                        message_id=self._message_id,
                        text=text,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                self._wrote_successfully = True
                return
            except TelegramApiError:
                logger.warning("Retry after server error failed")

        logger.warning("Edit write failed: %s", error)

        if is_final and not self._wrote_successfully:
            await self._send_fallback(text)


class StreamSession:
    """Manages streaming of thoughts and content to Telegram.

    Creates one channel for thoughts and one for content, each backed by
    a single Telegram message. Picks _DraftChannel for private chats and
    _EditChannel for groups/supergroups/channels.
    """

    def __init__(
        self,
        api: TelegramApiClient,
        chat_type: ChatType,
        *,
        update_interval_sec: float = DRAFT_UPDATE_INTERVAL_SEC,
        id_factory: Callable[[], int] | None = None,
    ) -> None:
        self.api = api
        self.chat_type = chat_type
        self.update_interval_sec = update_interval_sec
        self.id_factory = id_factory
        self._thoughts_channel: _Channel | None = None
        self._content_channel: _Channel | None = None

    def _create_channel(self, chat_id: int, message_thread_id: int | None) -> _Channel:
        if self.chat_type == ChatType.PRIVATE:
            return _DraftChannel(
                api=self.api,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                update_interval_sec=self.update_interval_sec,
                id_factory=self.id_factory,
            )
        return _EditChannel(
            api=self.api,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            update_interval_sec=self.update_interval_sec,
        )

    async def run(
        self,
        chunks: AsyncIterator[StreamChunk],
        chat_id: int,
        message_thread_id: int | None = None,
    ) -> tuple[str, str]:
        """Stream chunks to Telegram, returning final (thoughts, content)."""
        final_thoughts = ""
        final_content = ""

        async for chunk in chunks:
            if chunk.thoughts:
                if self._thoughts_channel is None:
                    self._thoughts_channel = self._create_channel(
                        chat_id, message_thread_id
                    )
                formatted = _format_thoughts(chunk.thoughts)
                await self._thoughts_channel.write(
                    formatted, is_final=not chunk.is_partial
                )

            if chunk.content:
                if self._content_channel is None:
                    self._content_channel = self._create_channel(
                        chat_id, message_thread_id
                    )
                formatted = _format_content(chunk.content)
                await self._content_channel.write(
                    formatted, is_final=not chunk.is_partial
                )

            if not chunk.is_partial:
                final_thoughts = chunk.thoughts
                final_content = chunk.content

        if self._thoughts_channel is not None:
            await self._thoughts_channel.finalize()
        if self._content_channel is not None:
            await self._content_channel.finalize()

        if not final_thoughts and not final_content:
            try:
                await self.api.send_message(
                    chat_id=chat_id,
                    text="I apologize, but I couldn't generate a response\\.",
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=message_thread_id,
                )
            except TelegramApiError:
                logger.exception("Failed to send apology message")

        return final_thoughts, final_content


class DraftManager:
    """Legacy adapter for backward compatibility.

    Delegates to StreamSession internally.
    """

    def __init__(
        self,
        api_client: TelegramApiClient,
        *,
        update_interval_sec: float = DRAFT_UPDATE_INTERVAL_SEC,
    ) -> None:
        self.api = api_client
        self.update_interval_sec = update_interval_sec

    async def stream_response(
        self,
        chat_id: int,
        chunks: AsyncIterator[StreamChunk],
        *,
        message_thread_id: int | None = None,
    ) -> tuple[str, str]:
        session = StreamSession(
            api=self.api,
            chat_type=ChatType.PRIVATE,
            update_interval_sec=self.update_interval_sec,
        )
        return await session.run(
            chunks=chunks,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
        )


def _format_thoughts(thoughts: str) -> str:
    return f"_Thinking: {escape_markdown(thoughts)}_"


def _format_content(content: str) -> str:
    return format_for_telegram(content)


def split_long_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split a long message into Telegram-safe chunks.

    Args:
        text: The text to split.
        limit: Maximum characters per chunk (default: 4096).

    Returns:
        List of text chunks, each within the character limit.
    """
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_index = _find_chunk_boundary(remaining, limit)
        chunk = remaining[:split_index].rstrip()
        chunks.append(chunk)
        remaining = remaining[len(chunk) :].lstrip()

    return chunks


def _find_chunk_boundary(text: str, limit: int) -> int:
    for separator in ("\n\n", "\n", " "):
        split_index = text.rfind(separator, 0, limit + 1)
        if split_index > 0:
            return split_index

    return limit
