"""Pydantic models for Telegram Bot API types.

Based on Telegram Bot API 9.5 (March 2026).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field


class ParseMode(str, Enum):
    """Formatting options for message text."""

    MARKDOWN = "Markdown"
    MARKDOWN_V2 = "MarkdownV2"
    HTML = "HTML"


class ChatType(str, Enum):
    """Type of chat."""

    PRIVATE = "private"
    GROUP = "group"
    SUPERGROUP = "supergroup"
    CHANNEL = "channel"


class Chat(BaseModel):
    """A Telegram chat."""

    id: int
    type: ChatType
    title: str | None = None
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None


class User(BaseModel):
    """A Telegram user."""

    id: int
    is_bot: bool = False
    first_name: str
    last_name: str | None = None
    username: str | None = None
    language_code: str | None = None


class Message(BaseModel):
    """A Telegram message."""

    message_id: int = Field(..., alias="message_id")
    message_thread_id: int | None = Field(None, alias="message_thread_id")
    from_user: User | None = Field(None, alias="from")
    sender_chat: Chat | None = Field(None, alias="sender_chat")
    date: datetime
    chat: Chat
    forward_from: User | None = Field(None, alias="forward_from")
    forward_from_chat: Chat | None = Field(None, alias="forward_from_chat")
    reply_to_message: Message | None = Field(None, alias="reply_to_message")
    text: str | None = None
    entities: list[MessageEntity] | None = None
    caption: str | None = None

    model_config = {"populate_by_name": True}


class MessageEntity(BaseModel):
    """An entity in a message (e.g., bold, italic, URL)."""

    type: str
    offset: int
    length: int
    url: str | None = None
    user: User | None = None
    language: str | None = None


class Update(BaseModel):
    """An incoming update from Telegram."""

    update_id: int = Field(..., alias="update_id")
    message: Message | None = None
    edited_message: Message | None = Field(None, alias="edited_message")
    channel_post: Message | None = Field(None, alias="channel_post")
    edited_channel_post: Message | None = Field(None, alias="edited_channel_post")
    callback_query: CallbackQuery | None = Field(None, alias="callback_query")

    model_config = {"populate_by_name": True}


class CallbackQuery(BaseModel):
    """A callback query from an inline button."""

    id: str
    from_user: User = Field(..., alias="from")
    message: Message | None = None
    chat_instance: str = Field(..., alias="chat_instance")
    data: str | None = None

    model_config = {"populate_by_name": True}


class BotCommand(BaseModel):
    """A bot command for the command menu."""

    command: str = Field(..., max_length=32)
    description: str = Field(..., max_length=256)


class ResponseParameters(BaseModel):
    """Additional response parameters for API errors."""

    migrate_to_chat_id: int | None = Field(None, alias="migrate_to_chat_id")
    retry_after: int | None = Field(None, alias="retry_after")

    model_config = {"populate_by_name": True}


class TelegramResponse(BaseModel):
    """Base response from Telegram API."""

    ok: bool
    result: Any = None
    error_code: int | None = Field(None, alias="error_code")
    description: str | None = None
    parameters: ResponseParameters | None = None

    model_config = {"populate_by_name": True}


class SendMessageResponse(BaseModel):
    """Response from sendMessage/sendMessageDraft."""

    message_id: int = Field(..., alias="message_id")
    chat: Chat
    date: datetime
    text: str | None = None

    model_config = {"populate_by_name": True}


class GetUpdatesResponse(BaseModel):
    """Response from getUpdates."""

    updates: list[Update] = Field(default_factory=list)


class InlineKeyboardButton(BaseModel):
    """A button for an inline keyboard."""

    text: str
    url: str | None = None
    callback_data: str | None = Field(None, max_length=64)


class InlineKeyboardMarkup(BaseModel):
    """An inline keyboard markup."""

    inline_keyboard: list[list[InlineKeyboardButton]]


class ReplyKeyboardMarkup(BaseModel):
    """A reply keyboard markup."""

    keyboard: list[list[KeyboardButton]]
    resize_keyboard: bool | None = Field(None, alias="resize_keyboard")
    one_time_keyboard: bool | None = Field(None, alias="one_time_keyboard")

    model_config = {"populate_by_name": True}


class KeyboardButton(BaseModel):
    """A button for a reply keyboard."""

    text: str
    request_contact: bool | None = Field(None, alias="request_contact")
    request_location: bool | None = Field(None, alias="request_location")

    model_config = {"populate_by_name": True}


class ReplyKeyboardRemove(BaseModel):
    """Remove a reply keyboard."""

    remove_keyboard: bool = True


class ForceReply(BaseModel):
    """Force a reply from the user."""

    force_reply: bool = True
    selective: bool | None = None


ReplyMarkup = Annotated[
    InlineKeyboardMarkup | ReplyKeyboardMarkup | ReplyKeyboardRemove | ForceReply,
    Field(discriminator=None),
]
