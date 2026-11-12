# Proposal

## Why

Blacki reaches models through OpenRouter's Chat Completions API. On that path:

- **Reasoning is lost between tool calls.** The adapter keeps only the readable summary and drops the encrypted reasoning OpenRouter returns. A live trace shows `gpt-6-luna` restarting its reasoning after every tool call.
- **GPT-6 tool calling depends on undocumented behavior.** OpenAI documents GPT-6 tool calling only on the Responses API. Through OpenRouter's Chat Completions, it works only because OpenRouter appears to convert the request upstream, which it doesn't document.

Blacki will use only OpenAI models from now on, so OpenRouter's main advantage, switching between providers, no longer applies. Calling OpenAI's Responses API directly gives the documented contract: replaying reasoning items unchanged without server-side storage (`store: false`), native reasoning controls, and no OpenRouter fee. ADK's own `OpenAIResponsesLlm` doesn't solve this. It drops reasoning items on replay (google/adk-python#8110) and would need an ADK upgrade from 2.5.0.

## What Changes

- Add an ADK model adapter that calls OpenAI's Responses API through the existing `openai` SDK, translating ADK requests and responses in both directions, with and without streaming.
- Store each model output item (reasoning, function call, message) on its ADK turn. Replay the items unchanged and in order on later requests to the same model. Requests always set `store: false`, so OpenAI keeps no response state and Blacki's session database holds the encrypted reasoning.
- Send reasoning settings as the native Responses `reasoning` object (`effort`, `summary`), not OpenRouter's `extra_body.reasoning`.
- Replace live model metadata from OpenRouter with a static catalog of supported OpenAI models: their reasoning efforts, default effort and token prices. `/model` and `/thinking` use the catalog.
- Record an estimated cost for each model call from token usage and catalog prices, because OpenAI doesn't report cost. Fill the ledger's existing estimated-cost fields.
- **BREAKING:** Remove OpenRouter. This removes the `OpenRouterLlm` adapter, its request, response and ID modules, the `OPENROUTER_API_KEY` setting, `openrouter/` model IDs, the session-affinity header, and OpenRouter wording in Telegram help, config validation, docs and `AGENTS.md`.
- **BREAKING:** Model IDs become plain OpenAI IDs, such as `gpt-6.1-terra`. Saved Telegram selections in the old `openrouter/openai/<id>` form are read as `<id>`. Saved selections for any other OpenRouter model fall back to the default model.
- Replace ADK's default compaction summarizer with one that stores only readable text, runs at low effort, and records its cost, so compaction can't put stray reasoning into history.
- Leave commentary-phase assistant text (tool-call preambles) out of the user-facing reply. It stays in the history.
- Keep encrypted reasoning out of logs and traces, and out of any request that goes to native Gemini.
- **BREAKING (deploy):** the publish workflow writes `OPENAI_API_KEY` from a new GitHub Actions secret instead of `OPENROUTER_API_KEY`.
- Keep native Gemini (`GOOGLE_API_KEY`) as the alternative root model, unchanged.

## Capabilities

### New Capabilities
- `llm-provider`: How Blacki calls its language model. Covers the Responses transport, translating requests and responses, carrying reasoning across tool calls and turns, retention (`store: false`), streaming, errors, and what happens when the model is missing or misconfigured.
- `model-selection`: The catalog of supported models, how a chat chooses its model and reasoning effort through `/model` and `/thinking`, defaults, and how saved legacy selections are interpreted.
- `llm-cost-accounting`: How each model call's token usage and estimated cost are recorded when the provider reports no cost.

### Modified Capabilities
- None. The only existing spec, `private-user-wiki`, is unaffected.

## Impact

- **Code:** `src/blacki/models/` gets a new Responses adapter, the model catalog and changes to the factory, inference profile and capability lookup. The `openrouter*.py` modules are removed. Also `src/blacki/agent.py`, `src/blacki/runtime/adk.py`, `src/blacki/telegram/settings_menu.py`, `src/blacki/telegram/bot.py`, `src/blacki/observability/costs.py`, `src/blacki/utils/config.py`, `src/blacki/utils/validation.py`, `src/blacki/utils/exceptions.py` and `src/blacki/server.py`.
- **Tests:** OpenRouter tests are replaced by Responses adapter tests that mock only the OpenAI HTTP boundary. Model factory, inference, Telegram model and cost tests are updated.
- **Configuration:** `OPENAI_API_KEY` replaces `OPENROUTER_API_KEY`, and `ROOT_AGENT_MODEL` takes an OpenAI model ID. `.env.example`, deployment docs and compose files are updated.
- **Dependencies:** no new runtime dependency (`openai>=2.20,<3` is already present). ADK stays at 2.5.0.
- **Data and privacy:** prompts go to OpenAI directly under OpenAI's data controls instead of OpenRouter's. Encrypted reasoning is added to session events in SQLite, with the same lifetime as the messages next to it. Without zero data retention approved for the OpenAI organization, OpenAI keeps abuse-monitoring logs for up to 30 days. Requesting that approval is an account decision outside this change.
- **Out of scope:** OpenAI server-side compaction, hosted OpenAI tools (web search, code interpreter, file search, MCP), background mode, and changing which Brave search tool Blacki uses.
