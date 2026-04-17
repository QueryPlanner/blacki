"""Streaming message management for real-time Telegram updates.

Implements time-based throttling (300ms) for streaming updates using
sendMessage + editMessageText for all chat types. This ensures stable,
persisted final messages across private chats, groups, and supergroups.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from blacki.adk_runtime import StreamChunk

from .api import TelegramApiClient, TelegramApiError
from .formatting import format_for_telegram
from .types import ParseMode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

UPDATE_INTERVAL_SEC = 0.3
TELEGRAM_MESSAGE_LIMIT = 4096
MESSAGE_NOT_MODIFIED_ERROR = "message is not modified"


class StreamSession:
    """Manages streaming of content to Telegram.

    Creates a single message on first content and updates it with throttling.
    Uses sendMessage + editMessageText for all chat types to ensure stable,
    persisted final messages that don't vanish.

    Note: This class is implemented and tested but not yet integrated into
    the Telegram bot. It is available for future streaming support.
    """

    def __init__(
        self,
        api: TelegramApiClient,
        *,
        update_interval_sec: float = UPDATE_INTERVAL_SEC,
    ) -> None:
        self.api = api
        self.update_interval_sec = update_interval_sec
        self._message_id: int | None = None
        self._last_write_time = 0.0
        self._source_text = ""
        self._full_text = ""
        self._last_sent_text = ""
        self._wrote_successfully = False

    async def run(
        self,
        chunks: AsyncIterator[StreamChunk],
        chat_id: int,
        message_thread_id: int | None = None,
    ) -> str:
        """Stream chunks to Telegram, returning final content."""
        final_content = ""

        async for chunk in chunks:
            if chunk.content:
                self._set_stream_text(
                    incoming_text=chunk.content,
                    is_partial=chunk.is_partial,
                )

                now = time.monotonic()
                elapsed = now - self._last_write_time

                if elapsed >= self.update_interval_sec or not chunk.is_partial:
                    await self._write(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        is_final=not chunk.is_partial,
                    )
                    self._last_write_time = now

            if not chunk.is_partial:
                final_content = self._source_text

        if self._full_text:
            await self._finalize(chat_id=chat_id, message_thread_id=message_thread_id)

        if not final_content:
            try:
                await self.api.send_message(
                    chat_id=chat_id,
                    text="I apologize, but I couldn't generate a response\\.",
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=message_thread_id,
                )
            except TelegramApiError:
                logger.exception("Failed to send apology message")

        return final_content

    async def _write(
        self,
        chat_id: int,
        message_thread_id: int | None,
        *,
        is_final: bool,
    ) -> None:
        text = self._full_text[:TELEGRAM_MESSAGE_LIMIT]
        if not text:
            return

        if text == self._last_sent_text:
            return

        if self._message_id is None:
            try:
                message = await self.api.send_message(
                    chat_id=chat_id,
                    text=text,
                    message_thread_id=message_thread_id,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                self._message_id = message.message_id
                self._last_sent_text = text
                self._wrote_successfully = True
                return
            except TelegramApiError as e:
                await self._handle_write_error(
                    e,
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    is_final=is_final,
                )
                return

        try:
            await self.api.edit_message_text(
                chat_id=chat_id,
                message_id=self._message_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            self._last_sent_text = text
            self._wrote_successfully = True
        except TelegramApiError as e:
            await self._handle_write_error(
                e,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                is_final=is_final,
            )

    async def _handle_write_error(
        self,
        error: TelegramApiError,
        chat_id: int,
        message_thread_id: int | None,
        *,
        is_final: bool,
    ) -> None:
        if _is_message_not_modified(error):
            self._last_sent_text = self._full_text[:TELEGRAM_MESSAGE_LIMIT]
            self._wrote_successfully = True
            return

        if error.error_code == 429 and error.retry_after:
            await asyncio.sleep(error.retry_after)
            try:
                if self._message_id is None:
                    message = await self.api.send_message(
                        chat_id=chat_id,
                        text=self._full_text[:TELEGRAM_MESSAGE_LIMIT],
                        message_thread_id=message_thread_id,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                    self._message_id = message.message_id
                else:
                    await self.api.edit_message_text(
                        chat_id=chat_id,
                        message_id=self._message_id,
                        text=self._full_text[:TELEGRAM_MESSAGE_LIMIT],
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                self._last_sent_text = self._full_text[:TELEGRAM_MESSAGE_LIMIT]
                self._wrote_successfully = True
                return
            except TelegramApiError:
                logger.warning("Retry after rate limit failed")

        if 500 <= (error.error_code or 0) <= 599:
            await asyncio.sleep(1.0)
            try:
                if self._message_id is None:
                    message = await self.api.send_message(
                        chat_id=chat_id,
                        text=self._full_text[:TELEGRAM_MESSAGE_LIMIT],
                        message_thread_id=message_thread_id,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                    self._message_id = message.message_id
                else:
                    await self.api.edit_message_text(
                        chat_id=chat_id,
                        message_id=self._message_id,
                        text=self._full_text[:TELEGRAM_MESSAGE_LIMIT],
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                self._last_sent_text = self._full_text[:TELEGRAM_MESSAGE_LIMIT]
                self._wrote_successfully = True
                return
            except TelegramApiError:
                logger.warning("Retry after server error failed")

        logger.warning("Stream write failed: %s", error)

        if is_final and not self._wrote_successfully:
            await self._send_fallback(
                chat_id=chat_id, message_thread_id=message_thread_id
            )

    async def _finalize(
        self,
        chat_id: int,
        message_thread_id: int | None,
    ) -> None:
        if len(self._full_text) <= TELEGRAM_MESSAGE_LIMIT:
            if not self._wrote_successfully:
                await self._write(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    is_final=True,
                )
            return

        chunks = split_long_message(self._full_text)
        if not chunks:
            return

        if self._wrote_successfully and self._message_id is not None:
            await self.api.edit_message_text(
                chat_id=chat_id,
                message_id=self._message_id,
                text=chunks[0],
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        else:
            await self._write(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                is_final=True,
            )

        for chunk in chunks[1:]:
            try:
                await self.api.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=message_thread_id,
                )
            except TelegramApiError:
                logger.exception("Failed to send additional message chunk")

    async def _send_fallback(
        self,
        chat_id: int,
        message_thread_id: int | None,
    ) -> None:
        if not self._full_text:
            return
        try:
            await self.api.send_message(
                chat_id=chat_id,
                text=self._full_text[:TELEGRAM_MESSAGE_LIMIT],
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=message_thread_id,
            )
            logger.info("Sent fallback message after streaming failures")
        except TelegramApiError:
            logger.exception("Failed to send fallback message")

    def _set_stream_text(self, incoming_text: str, *, is_partial: bool) -> None:
        self._source_text = _merge_stream_text(
            current_text=self._source_text,
            incoming_text=incoming_text,
            is_partial=is_partial,
        )
        self._full_text = _format_content(self._source_text)


def _format_content(content: str) -> str:
    return format_for_telegram(content)


def _merge_stream_text(
    current_text: str,
    incoming_text: str,
    *,
    is_partial: bool,
) -> str:
    if not current_text:
        return incoming_text

    if incoming_text.startswith(current_text):
        return incoming_text

    if not is_partial:
        return incoming_text

    if current_text.startswith(incoming_text):
        return current_text

    return current_text + incoming_text


def _is_message_not_modified(error: TelegramApiError) -> bool:
    if error.error_code != 400:
        return False

    error_message = str(error).lower()
    return MESSAGE_NOT_MODIFIED_ERROR in error_message


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
