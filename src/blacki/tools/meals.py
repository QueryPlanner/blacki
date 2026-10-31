"""Compact meal dispatch with selective, schema-backed action discovery."""

from __future__ import annotations

import inspect
from typing import Any, cast, get_type_hints

from google.adk.tools import BaseTool, FunctionTool, ToolContext
from google.genai import types
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    model_validator,
)

from blacki.tools import calories
from blacki.tools.action_identity import MEAL_ACTION_NAMES

MEAL_ACTIONS = {name: getattr(calories, name) for name in MEAL_ACTION_NAMES}


class _MealCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _empty_arguments_for_null(cls, data: Any) -> Any:
        """Treat an explicit null as no arguments for zero-argument actions."""
        if isinstance(data, dict) and data.get("arguments") is None:
            return {**data, "arguments": {}}
        return data


class _DescribeCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    domain: str
    action: str


def _invalid_arguments(error: ValidationError) -> dict[str, Any]:
    """Return validation guidance without echoing private argument values."""
    return {
        "status": "error",
        "error": "Invalid action arguments. Use describe_action for the schema.",
        "details": error.errors(
            include_input=False, include_context=False, include_url=False
        ),
    }


class MealsTool(BaseTool):
    """Execute only the enabled meal actions using the host's original context."""

    def __init__(self) -> None:
        super().__init__(
            name="meals",
            description=(
                "Track meals, calories, goals and Google Health meal sync. "
                "Use describe_action(domain='meals', action=...) when an action's "
                "parameters are unknown. Pass its parameters in arguments."
            ),
        )
        self._tools = {
            name: FunctionTool(function) for name, function in MEAL_ACTIONS.items()
        }
        self._validators: dict[str, type[BaseModel]] = {}
        for name, function in MEAL_ACTIONS.items():
            hints = get_type_hints(function)
            fields: dict[str, Any] = {
                key: (
                    hints[key],
                    ...
                    if parameter.default is inspect.Parameter.empty
                    else parameter.default,
                )
                for key, parameter in inspect.signature(function).parameters.items()
                if key != "tool_context"
            }
            self._validators[name] = create_model(
                f"{name}Arguments",
                __config__=ConfigDict(extra="forbid", strict=True),
                **fields,
            )

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(self._tools)},
                    "arguments": {"type": "object", "additionalProperties": True},
                },
                "required": ["action", "arguments"],
                "additionalProperties": False,
            },
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any]:
        try:
            call = _MealCall.model_validate(args)
            if call.action not in self._tools:
                return {"status": "error", "error": "Unknown meal action."}
            validated = self._validators[call.action].model_validate(call.arguments)
        except ValidationError as error:
            return _invalid_arguments(error)
        # The generated model forbids context/identity injection. Only ADK can
        # supply tool_context; discovery never changes authorization or state.
        result = await self._tools[call.action].run_async(
            args=validated.model_dump(exclude_unset=True), tool_context=tool_context
        )
        return cast(dict[str, Any], result)


class DescribeActionTool(BaseTool):
    """Expose action documentation only for the catalog registered with this tool."""

    def __init__(self, meals: MealsTool) -> None:
        super().__init__(
            name="describe_action",
            description="Read one enabled domain action's description and parameters.",
        )
        self._meals = meals

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "enum": ["meals"]},
                    "action": {"type": "string"},
                },
                "required": ["domain", "action"],
                "additionalProperties": False,
            },
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any]:
        del tool_context
        try:
            call = _DescribeCall.model_validate(args)
        except ValidationError as error:
            return _invalid_arguments(error)
        if call.domain != "meals" or call.action not in self._meals._tools:
            return {"status": "error", "error": "Unknown domain action."}
        return {
            "status": "success",
            "domain": call.domain,
            "action": call.action,
            "description": self._meals._tools[call.action].description,
            "parameters": self._meals._validators[call.action].model_json_schema(),
        }


def create_meal_tools() -> list[BaseTool]:
    """Build a compact domain tool and its read-only action discovery tool."""
    meals = MealsTool()
    return [meals, DescribeActionTool(meals)]
