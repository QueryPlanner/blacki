"""Controlled local ADK replay; no model or provider API calls."""

import ast
import asyncio
import importlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool, FunctionTool
from google.genai import types

import blacki.prompts.policies as policies
from blacki.container import AppContainer, reset_container_for_tests, set_container
from blacki.models.openrouter_request import build_chat_request
from blacki.prompts.instructions import (
    return_global_instruction,
    return_instruction_root,
)
from blacki.prompts.policies import NUTRITION_POLICY, DomainPolicyPlugin
from blacki.tools.meals import MEAL_ACTIONS, create_meal_tools
from blacki.tools.registry import ToolConfig, build_tools

# Keep the tokenizer out of Blacki's runtime dependencies. Run with
# uv run --with tiktoken==0.12.0 python scripts/benchmark_meal_tools.py
tiktoken = importlib.import_module("tiktoken")
encs = {name: tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")}


def tokens(value: Any) -> dict[str, int]:
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return {name: len(encoding.encode(text)) for name, encoding in encs.items()}


git_path = shutil.which("git")
if git_path is None:
    raise RuntimeError("Git is required to read the benchmark baseline")
base_source = subprocess.run(  # noqa: S603
    [git_path, "show", "4e10c3f:src/blacki/prompts/policies.py"],
    capture_output=True,
    text=True,
    check=True,
).stdout
base_policy = next(
    ast.literal_eval(node.value)
    for node in ast.parse(base_source).body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "NUTRITION_POLICY"
        for target in node.targets
    )
)


class ReplayLlm(BaseLlm):
    steps: list[tuple[str, dict[str, Any]]]
    records: list[dict[str, Any]] = []
    outputs: list[dict[str, int]] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        wire = build_chat_request(
            llm_request, model="google/gemini-2.5-flash"
        ).create_kwargs(stream=False)
        self.records.append(
            {"input": tokens(wire), "tools": tokens(wire.get("tools", []))}
        )
        for content in llm_request.contents:
            for part in content.parts or []:
                if part.function_response:
                    response = part.function_response.response
                    if response is None or response.get("status") != "success":
                        raise RuntimeError("Controlled tool replay failed")
        index = len(self.records) - 1
        if index < len(self.steps):
            name, args = self.steps[index]
            content = types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=f"call-{index}", name=name, args=args
                        )
                    )
                ],
            )
        else:
            content = types.Content(
                role="model",
                parts=[types.Part(text="Completed the requested meal operation.")],
            )
        self.outputs.append(tokens(content.model_dump(exclude_none=True)))
        yield LlmResponse(content=content)


async def run(case: tuple[str, dict[str, Any], str], mode: str) -> dict[str, Any]:
    policies.NUTRITION_POLICY = base_policy if mode == "flat" else NUTRITION_POLICY
    with tempfile.TemporaryDirectory(prefix="blacki-meals-benchmark-") as directory:
        container = await AppContainer.create(Path(directory) / "meals.db")
        await container.initialize_all_storages()
        set_container(container)
        try:
            config = ToolConfig(
                sqlite_path=str(Path(directory) / "meals.db"),
                sandbox_enabled=True,
                exa_api_key="benchmark",
                brave_search_api_key="benchmark",
                skills_dir=Path("src/blacki/skills"),
                r2_files_enabled=True,
                google_health_enabled=True,
                kokoro_tts_base_url="http://example.invalid",
            )
            tools = build_tools(config, include_user_scoped_tools=True)
            if mode == "flat":
                tools = [
                    tool
                    for tool in tools
                    if not isinstance(tool, BaseTool)
                    or tool.name not in {"meals", "describe_action"}
                ]
                tools.extend(MEAL_ACTIONS.values())
            steps: list[tuple[str, dict[str, Any]]] = []
            action, args, message = case
            if mode == "cold":
                steps.append(("describe_action", {"domain": "meals", "action": action}))
            steps.append(
                (action, args)
                if mode == "flat"
                else ("meals", {"action": action, "arguments": args})
            )
            model = ReplayLlm(model="controlled-replay", steps=steps)
            agent = LlmAgent(
                name="blacki",
                model=model,
                instruction=return_instruction_root(),
                tools=tools,
            )
            app = App(
                name="blacki",
                root_agent=agent,
                plugins=[
                    DomainPolicyPlugin(),
                    GlobalInstructionPlugin(return_global_instruction),
                ],
            )
            sessions = InMemorySessionService()
            session = await sessions.create_session(
                app_name="blacki",
                user_id="telegram-chat-42",
                state={"telegram_chat_id": "42", "telegram_chat_type": "private"},
            )
            runner = Runner(app=app, session_service=sessions)
            async for _ in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=message)]
                ),
            ):
                pass
            for name, _args in steps:
                if name == "describe_action":
                    continue
                if action == "log_meal":
                    summary = await container.calorie_storage.get_daily_summary(
                        session.user_id, "2026-10-20"
                    )
                    if summary.entry_count != 1 or summary.total_calories != 95:
                        raise RuntimeError(
                            "Meal replay did not write exactly one expected entry"
                        )
            await runner.close()
            return {
                "calls": len(model.records),
                "initial_schema": model.records[0]["tools"],
                "input_total": {
                    name: sum(row["input"][name] for row in model.records)
                    for name in encs
                },
                "output_total": {
                    name: sum(row[name] for row in model.outputs) for name in encs
                },
            }
        finally:
            reset_container_for_tests()
            await container.close()


async def main() -> None:
    results = {}
    cases: list[tuple[str, dict[str, Any], str]] = [
        (
            "log_meal",
            {"description": "apple", "estimated_calories": 95, "date": "2026-10-20"},
            "Log one apple with 95 calories on 2026-10-20.",
        ),
        (
            "get_calorie_summary",
            {"date": "2026-10-20"},
            "Show my calorie summary for 2026-10-20.",
        ),
    ]
    for case in cases:
        results[case[0]] = {
            mode: await run(case, mode) for mode in ("flat", "direct", "cold")
        }
    # Isolate actual meals declarations using the production adapter.
    subtotals = {}
    tool_groups: list[tuple[str, list[BaseTool]]] = [
        ("flat", [FunctionTool(f) for f in MEAL_ACTIONS.values()]),
        ("compact", create_meal_tools()),
    ]
    for name, tools in tool_groups:
        req = LlmRequest()
        req.append_tools(tools)
        subtotals[name] = tokens(
            build_chat_request(req, model="google/gemini-2.5-flash").tools
        )
    results["meals_schema"] = subtotals
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
