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


class DeclarativeDbPlugin(BasePlugin):
    """ADK plugin to dynamically load declarative database schemas.

    Loads both schemas and custom instruction overrides.
    This plugin queries the storage layer using the active session user's ID
    and appends user-defined tables, query templates, and instruction overrides
    to the system instruction context on every LLM call.
    """

    def __init__(self, name: str = "declarative_db") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Callback executed before the LLM is invoked."""
        if not callback_context.session:
            return

        user_id = callback_context.session.state.get(
            "user_id"
        ) or callback_context.session.state.get("telegram_chat_id")
        if not user_id:
            logger.debug(
                "before_model_callback: No user_id or telegram_chat_id in session state"
            )
            return

        try:
            storage = get_declarative_db_storage()
            schema_xml = await storage.get_schema_instructions_xml(str(user_id))
            if schema_xml:
                logger.info(
                    "Injecting custom database instructions for user %s (%d chars)",
                    user_id,
                    len(schema_xml),
                )
                llm_request.append_instructions([schema_xml])
        except Exception:
            logger.exception("Failed to inject user database schemas and instructions")
