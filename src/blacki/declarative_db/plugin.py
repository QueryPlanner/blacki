"""ADK plugin for injecting declarative database instructions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from blacki.declarative_db.storage import get_declarative_db_storage

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest

logger = logging.getLogger(__name__)


def _get_user_id(callback_context: CallbackContext) -> str | None:
    """Return the active user ID without widening the request scope."""
    if not callback_context.session:
        return None
    user_id = callback_context.session.state.get(
        "user_id"
    ) or callback_context.session.state.get("telegram_chat_id")
    return str(user_id) if user_id else None


class DeclarativeDbPlugin(BasePlugin):
    """Append sanitized custom database schemas as capability metadata."""

    def __init__(self, name: str = "declarative_db") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Callback executed before the LLM is invoked."""
        user_id = _get_user_id(callback_context)
        if not user_id:
            logger.debug(
                "before_model_callback: No user_id or telegram_chat_id in session state"
            )
            return

        try:
            storage = get_declarative_db_storage()
            schema_xml = await storage.get_schema_instructions_xml(user_id)
            if schema_xml:
                logger.info(
                    "Injecting custom database metadata for user %s (%d chars)",
                    user_id,
                    len(schema_xml),
                )
                llm_request.append_instructions([schema_xml])
        except Exception:
            logger.exception("Failed to inject user database schema metadata")


class StoredPreferencesPlugin(BasePlugin):
    """Append allow-listed stored preferences after all developer policies."""

    def __init__(self, name: str = "stored_preferences") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Append structured style preferences at the lowest prompt precedence."""
        user_id = _get_user_id(callback_context)
        if not user_id:
            return

        try:
            storage = get_declarative_db_storage()
            preferences_xml = await storage.get_user_preferences_instruction_xml(
                user_id
            )
            if preferences_xml:
                logger.info(
                    "Injecting stored preferences for user %s (%d chars)",
                    user_id,
                    len(preferences_xml),
                )
                llm_request.append_instructions([preferences_xml])
        except Exception:
            logger.exception("Failed to inject stored user preferences")
