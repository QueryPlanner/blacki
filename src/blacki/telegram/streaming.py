"""Streaming draft manager for real-time Telegram message updates.

Implements time-based throttling (300ms) for sendMessageDraft API calls,
managing separate drafts for thoughts and content with proper formatting.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator

from blacki.adk_runtime import StreamChunk

from .api import TelegramApiClient, TelegramApiError
from .types import ParseMode

logger = logging.getLogger(__name__)

DRAFT_UPDATE_INTERVAL_MS = 300
DRAFT_UPDATE_INTERVAL_SEC = DRAFT_UPDATE_INTERVAL_MS / 1000.0
TELEGRAM_MESSAGE_LIMIT = 4096


class DraftManager:
    """Manages streaming draft updates with time-based throttling.

    This class coordinates real-time message streaming via Telegram's
    sendMessageDraft API (Bot API 9.5+). It maintains two separate drafts:
    one for thoughts (italicized) and one for content.

    The draft update frequency is throttled to avoid hitting Telegram's
    rate limits while providing a smooth user experience.
    """

    def __init__(
        self,
        api_client: TelegramApiClient,
        *,
        update_interval_sec: float = DRAFT_UPDATE_INTERVAL_SEC,
    ) -> None:
        """Initialize the draft manager.

        Args:
            api_client: The Telegram API client for making requests.
            update_interval_sec: Minimum time between draft updates.
        """
        self.api = api_client
        self.update_interval_sec = update_interval_sec

    async def stream_response(
        self,
        chat_id: int,
        chunks: AsyncIterator[StreamChunk],
        *,
        message_thread_id: int | None = None,
    ) -> tuple[str, str]:
        """Stream ADK chunks to Telegram with two draft messages.

        Creates and updates two draft messages:
        1. Thoughts draft (italicized with "Thinking:" prefix)
        2. Content draft (formatted normally)

        When the stream completes, both drafts are replaced with final messages.

        Args:
            chat_id: The Telegram chat ID to send to.
            chunks: Async iterator of StreamChunk objects from ADK.
            message_thread_id: Optional thread ID for topic groups.

        Returns:
            A tuple of (final_thoughts, final_content) strings.
        """
        thoughts_draft_id = str(uuid.uuid4())
        content_draft_id = str(uuid.uuid4())

        last_update_time = 0.0
        accumulated_thoughts = ""
        accumulated_content = ""
        final_thoughts = ""
        final_content = ""

        async for chunk in chunks:
            accumulated_thoughts = chunk.thoughts
            accumulated_content = chunk.content

            now = time.monotonic()
            elapsed = now - last_update_time

            if elapsed >= self.update_interval_sec:
                await self._update_drafts(
                    chat_id=chat_id,
                    thoughts_draft_id=thoughts_draft_id,
                    content_draft_id=content_draft_id,
                    thoughts=accumulated_thoughts,
                    content=accumulated_content,
                    message_thread_id=message_thread_id,
                )
                last_update_time = now

            if not chunk.is_partial:
                final_thoughts = chunk.thoughts
                final_content = chunk.content

        if accumulated_thoughts or accumulated_content:
            await self._update_drafts(
                chat_id=chat_id,
                thoughts_draft_id=thoughts_draft_id,
                content_draft_id=content_draft_id,
                thoughts=accumulated_thoughts,
                content=accumulated_content,
                message_thread_id=message_thread_id,
            )

        return final_thoughts, final_content

    async def _update_drafts(
        self,
        chat_id: int,
        thoughts_draft_id: str,
        content_draft_id: str,
        thoughts: str,
        content: str,
        *,
        message_thread_id: int | None = None,
    ) -> None:
        """Update both draft messages."""
        tasks = []

        if thoughts:
            tasks.append(
                self._update_draft(
                    chat_id=chat_id,
                    draft_id=thoughts_draft_id,
                    text=_format_thoughts(thoughts),
                    message_thread_id=message_thread_id,
                )
            )

        if content:
            tasks.append(
                self._update_draft(
                    chat_id=chat_id,
                    draft_id=content_draft_id,
                    text=_format_content(content),
                    message_thread_id=message_thread_id,
                )
            )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Failed to update draft: %s", result)

    async def _update_draft(
        self,
        chat_id: int,
        draft_id: str,
        text: str,
        *,
        message_thread_id: int | None = None,
    ) -> None:
        """Update a single draft message."""
        try:
            await self.api.send_message_draft(
                chat_id=chat_id,
                text=text[:TELEGRAM_MESSAGE_LIMIT],
                draft_id=draft_id,
                message_thread_id=message_thread_id,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except TelegramApiError as e:
            logger.warning("Draft update failed (draft_id=%s): %s", draft_id, e)


def _format_thoughts(thoughts: str) -> str:
    """Format thoughts as italicized text with prefix.

    Returns:
        Formatted text like "_Thinking: ..._" for MarkdownV2.
    """
    from .bot import escape_markdown

    escaped = escape_markdown(thoughts)
    return f"_Thinking: {escaped}_"


def _format_content(content: str) -> str:
    """Format content for Telegram MarkdownV2.

    Returns:
        Properly formatted content for Telegram.
    """
    from .bot import format_for_telegram

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
    """Find the most readable split point within the limit.

    Prefers paragraph breaks, then line breaks, then word boundaries.
    """
    for separator in ("\n\n", "\n", " "):
        split_index = text.rfind(separator, 0, limit + 1)
        if split_index > 0:
            return split_index

    return limit
