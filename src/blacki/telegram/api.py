"""Low-level HTTP client for Telegram Bot API using httpx.

Implements direct HTTP calls to the official Telegram Bot API (version 9.5+).
"""

import logging
from typing import Any

import httpx

from .types import BotCommand, Message, ParseMode, TelegramResponse, Update

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"
TELEGRAM_API_TIMEOUT = 30.0


class TelegramApiError(Exception):
    """Exception raised when a Telegram API call fails."""

    def __init__(
        self,
        message: str,
        error_code: int | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after


class TelegramApiClient:
    """Direct HTTP client for Telegram Bot API.

    This client implements the core Telegram Bot API methods using httpx
    for HTTP requests. It provides a clean, dependency-free interface to
    the Telegram Bot API without requiring third-party libraries.
    """

    def __init__(
        self,
        token: str,
        *,
        timeout: float = TELEGRAM_API_TIMEOUT,
    ) -> None:
        """Initialize the Telegram API client.

        Args:
            token: The bot token from @BotFather.
            timeout: HTTP request timeout in seconds.
        """
        self.token = token
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "TelegramApiClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure an HTTP client exists and return it."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_url(self, method: str) -> str:
        """Build the API URL for a given method."""
        return TELEGRAM_API_BASE.format(token=self.token, method=method)

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an API request and return the result.

        Args:
            method: The Telegram API method name.
            params: Optional parameters for the request.

        Returns:
            The parsed JSON response.

        Raises:
            TelegramApiError: If the API returns an error.
            httpx.HTTPError: If the HTTP request fails.
        """
        client = await self._ensure_client()
        url = self._build_url(method)

        response = await client.post(url, json=params)
        response.raise_for_status()

        data = response.json()
        telegram_response = TelegramResponse.model_validate(data)

        if not telegram_response.ok:
            retry_after = None
            if telegram_response.parameters:
                retry_after = telegram_response.parameters.retry_after

            raise TelegramApiError(
                message=telegram_response.description or "Unknown Telegram API error",
                error_code=telegram_response.error_code,
                retry_after=retry_after,
            )

        return telegram_response.result or {}

    async def get_me(self) -> dict[str, Any]:
        """Get information about the bot.

        Returns:
            Bot information including id, username, etc.
        """
        return await self._request("getMe")

    async def get_updates(
        self,
        *,
        offset: int | None = None,
        limit: int | None = None,
        timeout: int = 0,
        allowed_updates: list[str] | None = None,
    ) -> list[Update]:
        """Receive incoming updates using long polling.

        Args:
            offset: Identifier of the first update to be returned.
            limit: Limits the number of updates to be retrieved (1-100).
            timeout: Timeout in seconds for long polling.
            allowed_updates: List of update types to receive.

        Returns:
            A list of Update objects.
        """
        params: dict[str, Any] = {}
        if offset is not None:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        if timeout > 0:
            params["timeout"] = timeout
        if allowed_updates is not None:
            params["allowed_updates"] = allowed_updates

        result = await self._request("getUpdates", params)
        return [Update.model_validate(u) for u in result]

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        message_thread_id: int | None = None,
        parse_mode: ParseMode | None = None,
        disable_notification: bool = False,
        protect_content: bool = False,
    ) -> Message:
        """Send a text message to a chat.

        Args:
            chat_id: Unique identifier for the target chat or username.
            text: Text of the message to be sent (1-4096 characters).
            message_thread_id: Unique identifier for the target message thread.
            parse_mode: Mode for parsing entities in the message text.
            disable_notification: Send silently without notification.
            protect_content: Protect the content from forwarding/saving.

        Returns:
            The sent Message object.
        """
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if message_thread_id is not None:
            params["message_thread_id"] = message_thread_id
        if parse_mode is not None:
            params["parse_mode"] = parse_mode.value
        if disable_notification:
            params["disable_notification"] = True
        if protect_content:
            params["protect_content"] = True

        result = await self._request("sendMessage", params)
        return Message.model_validate(result)

    async def send_message_draft(
        self,
        chat_id: int | str,
        text: str,
        draft_id: str,
        *,
        message_thread_id: int | None = None,
        parse_mode: ParseMode | None = None,
    ) -> Message:
        """Send or update a streaming draft message.

        This method (added in Bot API 9.5, March 2026) enables real-time
        message streaming. Each call with the same draft_id updates the
        existing draft message. When the stream is complete, the draft
        automatically converts to a regular message.

        Args:
            chat_id: Unique identifier for the target chat or username.
            text: Current text content of the draft.
            draft_id: Unique identifier for this draft session.
            message_thread_id: Unique identifier for the target message thread.
            parse_mode: Mode for parsing entities in the message text.

        Returns:
            The draft Message object.
        """
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "draft_id": draft_id,
        }
        if message_thread_id is not None:
            params["message_thread_id"] = message_thread_id
        if parse_mode is not None:
            params["parse_mode"] = parse_mode.value

        result = await self._request("sendMessageDraft", params)
        return Message.model_validate(result)

    async def edit_message_text(
        self,
        chat_id: int | str,
        message_id: int,
        text: str,
        *,
        parse_mode: ParseMode | None = None,
    ) -> Message:
        """Edit text of a previously sent message.

        Args:
            chat_id: Unique identifier for the target chat or username.
            message_id: Identifier of the message to edit.
            text: New text of the message (1-4096 characters).
            parse_mode: Mode for parsing entities in the message text.

        Returns:
            The edited Message object.
        """
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if parse_mode is not None:
            params["parse_mode"] = parse_mode.value

        result = await self._request("editMessageText", params)
        return Message.model_validate(result)

    async def delete_message(
        self,
        chat_id: int | str,
        message_id: int,
    ) -> bool:
        """Delete a message.

        Args:
            chat_id: Unique identifier for the target chat or username.
            message_id: Identifier of the message to delete.

        Returns:
            True if the message was deleted successfully.
        """
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }
        await self._request("deleteMessage", params)
        return True

    async def send_chat_action(
        self,
        chat_id: int | str,
        action: str,
        *,
        message_thread_id: int | None = None,
    ) -> bool:
        """Send a chat action (e.g., typing).

        Args:
            chat_id: Unique identifier for the target chat or username.
            action: Type of action (typing, upload_photo, record_video, etc.).
            message_thread_id: Unique identifier for the target message thread.

        Returns:
            True if the action was sent successfully.
        """
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "action": action,
        }
        if message_thread_id is not None:
            params["message_thread_id"] = message_thread_id

        await self._request("sendChatAction", params)
        return True

    async def set_my_commands(
        self,
        commands: list[BotCommand],
        *,
        scope: dict[str, Any] | None = None,
        language_code: str | None = None,
    ) -> bool:
        """Set the bot's command menu.

        Args:
            commands: List of bot commands to set (1-100 commands).
            scope: Scope to which the commands apply.
            language_code: Language code for the commands.

        Returns:
            True if the commands were set successfully.
        """
        params: dict[str, Any] = {
            "commands": [c.model_dump() for c in commands],
        }
        if scope is not None:
            params["scope"] = scope
        if language_code is not None:
            params["language_code"] = language_code

        await self._request("setMyCommands", params)
        return True

    async def get_my_commands(self) -> list[BotCommand]:
        """Get the current list of bot commands.

        Returns:
            List of BotCommand objects.
        """
        result = await self._request("getMyCommands")
        return [BotCommand.model_validate(c) for c in result]
