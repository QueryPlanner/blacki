# Design

## Context

See `proposal.md` for the motivation. The current state that shapes this design (ADK 2.5.0; checked in code and in an independent review):

- **Model adapter.** The root `LlmAgent` gets its model from `models/factory.py:build_model()`. That returns a native Gemini model name or `OpenRouterLlm`, an ADK `BaseLlm` over `AsyncOpenAI` that calls `chat.completions.create`. The adapter creates a client per call, closes it in `finally`, and keeps credentials in private attributes.
- **History.** ADK rebuilds history from session events on every model call.
  - An event is dropped only when all its parts are invisible, meaning thought-only or empty (`flows/llm_flows/contents.py:317-345`).
  - `_copy_content_for_request` copies every part of a kept event, including `part_metadata` (`contents.py:639-683`).
  - Only function-response events are reordered or merged, so parts keep their order within a model event.
  - Partial events are never persisted. One model response becomes one event.
- **Persistence.** `DatabaseSessionService` stores `event.model_dump(exclude_none=True, mode="json")` (`sessions/schemas/v1.py:233`), so `part_metadata` survives a restart.
- **Function call ids.** For a custom `BaseLlm`, ADK sets `adk-*` function-call and response ids to `None` before the adapter sees them (`contents.py:675-679`). OpenAI's `call_*` ids pass through. `openrouter_request.py` already pairs calls with results by order when ids are missing, and inserts placeholder results for calls that have none (lines 154-164, 314, 433-476).
- **Per-chat overrides.** `TelegramModelOverridePlugin` (`agent.py:56-110`) sets `llm_request.model` and puts reasoning in `http_options.extra_body["reasoning"]` (`models/inference.py`). It drops reasoning for native Gemini.
- **Model capabilities.** `/thinking` gets a model's efforts from `OpenRouterModelCapabilitiesResolver`, through the `ModelCapabilitiesResolver` protocol.
- **Costs.** `observability/costs.py` reads the cost OpenRouter reports. The ledger already has estimated-cost, `cost_kind` and `cost_source` columns.
- **Compaction.** `EventsCompactionConfig` (`agent.py:225`) uses ADK's default `LlmEventSummarizer`. It calls the root model with no config, tools or plugins (`apps/llm_event_summarizer.py:141-153`) and stores the first response's content as `compacted_content`, which is replayed as a model turn.
- **Native Gemini.** `google-genai` forwards `Part.part_metadata` to the Gemini Developer API and raises on Vertex (`genai/models.py:3740, 3826`).
- **Observability.**
  - With Blacki's defaults (`observability/setup.py:59-63`), ADK's experimental span format drops `part_metadata` (`telemetry/_experimental_semconv.py:223-262`) and renders thought text as plain text.
  - The legacy span capture (`telemetry/tracing.py:354, 385, 437`) dumps whole parts, but is on only if `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` is enabled.
  - `observability/lifecycle.py:128-132, 158-160` logs `content.model_dump()` at DEBUG.
- **Instructions and streaming.** `TemporalContextPlugin` changes the instructions once a day. Telegram turns use `StreamingMode.SSE`. `runtime/adk.py:526-553` joins every non-thought text part of a turn into the reply.

## Goals / Non-Goals

**Goals:**
- Replay model output items unchanged and in order to the same model, so reasoning carries across tool calls and, on models that use it, across user turns.
- Keep the adapter's surface the same as `OpenRouterLlm`'s, so the agent, runner and plugins don't change.
- Keep ADK at 2.5.0.

**Non-Goals:**
- OpenAI server-side state (`previous_response_id`, conversations) or OpenAI's compaction.
- Hosted OpenAI tools, background mode, `configuration_update` items, async tools.
- A dynamic model catalog.
- Showing commentary text (see D13) anywhere except the conversation history.

## Decisions

### D1. Our own `BaseLlm`, not ADK's `OpenAIResponsesLlm`
`models/openai_responses.py` defines `OpenAIResponsesLlm(BaseLlm)`, with translation in `openai_request.py` and `openai_response.py`.
- **Alternative:** upgrade ADK to ≥2.14 and subclass its adapter. Rejected:
  - It drops reasoning on replay (google/adk-python#8110), so the part we most need would override its private helpers.
  - The ADK upgrade is pinned together with OTel instrumentation and belongs in its own change.
- Revisit once #8110 is fixed.

### D2. Each ADK part carries the raw item that produced it
Each Responses output item becomes one ADK `Part` with:

`part.part_metadata["blacki_openai"] = {"model": <normalized requested model id>, "item": <raw item>}`

The raw item is `model_dump(exclude_none=True, mode="json")`. The model id stored is the **normalized requested id** (`llm_request.model` after `normalize_model_id`), not `response.model`. OpenAI returns a dated snapshot id there that would never equal the requested id.

| Output item | ADK part |
|---|---|
| `reasoning` | `Part(thought=True, text=<joined summary text or "">)` |
| `message`, `phase` final or absent | `Part(text=<output_text>)`; refusal content becomes visible text |
| `message`, `phase: "commentary"` | `Part(text=<text>)` marked as commentary in metadata (D13) |
| `function_call` | `Part(function_call=FunctionCall(id=call_id, name, args))` |
| other types | Not created; logged once at debug level |

When rebuilding a model turn for the same model, the parts are walked in order and each stored `item` is sent unchanged. Other parts are converted portably: text becomes an assistant message, function calls become `function_call` items, and thoughts are dropped.
- **Alternative:** keep only ids and rebuild items. Rejected, because only raw replay keeps fields we don't model, such as `phase` and future fields.

### D3. Reasoning items with an empty summary
They're stored as thought parts with empty text. ADK keeps them while another part in the event is visible. Responses always follows reasoning with a message or function call, except on a truncated turn. A test pins this using ADK 2.5.0's real contents processor.

### D4. Same-model rule
Stored items are replayed only when the stored model equals the normalized request model, compared as exact ids. Sibling models lose reasoning continuity, but no invalid request can result.

### D5. Requests
- `store=False` always, and `include=["reasoning.encrypted_content"]`.
- `instructions` comes from `system_instruction`, text only.
- Function tools use JSON-schema normalization carried over from `openrouter_request.py`. Images become `input_image` data URLs.
- **Call and result pairing carried over from the OpenRouter adapter:**
  - Real `call_*` ids are used as-is.
  - When ADK has cleared an id to `None`, calls and results are paired by order, and both sides get one synthetic `call_blacki_*` id.
  - A call with no result gets a placeholder `function_call_output`, because Responses rejects a call with no output.
- **Reasoning settings:** the adapter takes `reasoning` out of `extra_body` and sends `reasoning={"effort": e, "summary": "auto"}` when an effort is set, `{"summary": "auto"}` otherwise.
- **Sampling:** `temperature` and `top_p` are sent only when the effort is explicitly `none` **and** the catalog marks the model as accepting them. Otherwise they're dropped with a warning, because reasoning models reject them and still reason by default when no effort is set.
- Other `extra_body` keys pass through the reserved-key filter.
- **Orphan pruning:** before sending, any replayed reasoning item not directly followed by an item from the same response is dropped. This prevents 400 errors when history around it was changed.

### D6. Responses and streaming
- **Non-streamed:** output items become parts as in D2. `usage` becomes `usage_metadata`. Reasoning tokens are already counted inside `output_tokens`.
- **Finish reasons:**
  - `incomplete` with `max_output_tokens` → `MAX_TOKENS`.
  - `incomplete` with `content_filter` → `SAFETY`.
  - Otherwise `STOP`.
- **Streamed:**
  - `response.output_text.delta` → partial text.
  - `response.reasoning_summary_text.delta` → partial thought.
  - Both `response.completed` and `response.incomplete` end the stream: the single final `LlmResponse` is built from `event.response`.
  - `response.failed` and `error` raise.
  - Partial responses carry no metadata.

### D7. A rejected reasoning item downgrades the session
A 400 counts as a reasoning rejection when its `code` is `invalid_encrypted_content`, or its `param` points at an `input[n]` item that is a reasoning item. On that error, the adapter removes all reasoning items and retries once, then records the session id in an in-process set. Later requests in that session skip reasoning replay until the process restarts. This avoids paying a failed call on every request. Only the error code is logged. If the retry fails too, the error is raised.

ADK 2.5.0 passes no session identity into `BaseLlm` calls. Blacki's override plugin therefore writes a non-secret key into `llm_request.config.labels["blacki_session"]`: an HMAC of app, user and session using a process secret. The adapter scopes the downgrade to that key, falling back to the model id when the key is absent, and never sends labels to OpenAI. The plugin removes the label from requests that go to native Gemini.

### D8. Static catalog
`models/catalog.py` has one frozen entry per model:
- `id`, `display_name`
- `efforts`, `default_effort`
- `accepts_sampling_at_none`
- prices per million tokens: `input`, `cached_input`, `cache_write` (optional), `output`

`CatalogCapabilitiesResolver` implements `ModelCapabilitiesResolver`. Entries: `gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.1-sol`, `gpt-6-astra`. Luna and Terra accept `none`; Sol and Astra don't. Prices are copied from OpenAI's model pages at implementation time.

### D9. Cost estimate
`(input − cached) × input_rate + cached × cached_rate + output × output_rate`, written with `cost_kind="estimate"` and `cost_source="catalog_price"`.
- **Cache writes:** priced separately only when usage reports a cache-write count. The adapter reads OpenAI's documented `cache_write_tokens` field if present, and otherwise prices those tokens as normal input.
- **Reasoning tokens:** never added on top of `output`, because they're already counted in it.

### D10. Model ids, root provider and saved selections
- `normalize_model_id` strips `openrouter/openai/` and lowercases.
- **Root is OpenAI:** catalog ids are selectable with `/model`. Any saved selection that isn't a catalog id, including plain `gemini-*`, uses the default model with a warning.
- **Root is native Gemini** (no `OPENAI_API_KEY`): `/model <openai-id>` is refused with "OpenAI is not configured", and saved OpenAI selections use the default.
- Any other `openrouter/...` value in `ROOT_AGENT_MODEL` is a startup error.

### D11. Keeping encrypted data out of logs, traces and Gemini
- **Logs:** the adapter never logs request or response bodies. `observability/lifecycle.py` debug dumps go through a redaction helper that removes `part_metadata` and `thought_signature`.
- **Traces:** experimental span capture already drops `part_metadata`. At startup Blacki forces `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false` and warns if an operator set it to `true`, because the legacy format would dump whole parts. In traces, reasoning summaries appear as plain text wherever content capture is on. That's accepted and documented.
- **Native Gemini:** when the effective model is native Gemini, the override plugin removes `blacki_openai` metadata and empty thought parts from the request contents.

### D12. Compaction uses a Blacki summarizer
`EventsCompactionConfig.summarizer` gets a Blacki summarizer that wraps ADK's default prompt and:
- calls the root model at the catalog's lowest non-`none` effort, with `store=false`;
- keeps only message text in `compacted_content`, dropping reasoning parts and metadata, so no reasoning item from an unrelated request gets into history;
- reports its usage to the cost ledger.

### D13. Commentary is history, not reply
Messages with `phase: "commentary"` are stored and replayed, but `runtime/adk.py` leaves them out of the Telegram reply text and the final answer. Today's reply is built from all text parts, which would mix tool-preamble chatter into the answer.

### D14. Removal is in the same change
The OpenRouter modules, tests, deploy wiring and wording are removed in this change. `tests/tools/test_meals.py` and `scripts/benchmark_meal_tools.py` move to the new request builder. Native Gemini stays as the fallback root model.

## Risks / Trade-offs

- **Replayed reasoning costs input tokens.** The A/B on `gpt-6.1-terra` showed about +75% input on a 13-call task, with no change in correctness. → Cost is recorded per call. Caching offsets part of it.
- **Encrypted reasoning may expire or be rejected.** Nothing documents an expiry. → D7.
- **Changing history drops reasoning.** ADK compaction and the daily instruction change rewrite the start of the history, and newer models may then drop replayed reasoning. → Accepted. Orphan pruning in D5 prevents errors.
- **Sessions grow larger.** → Same lifetime as the messages next to them.
- **Privacy.** Prompts go to OpenAI directly. Without approved ZDR, OpenAI keeps abuse-monitoring logs for up to 30 days. → `store=false` is always set, and ZDR is documented as an account decision.
- **Fixed catalog.** → Adding a model is one entry plus tests.
- **ADK internals may shift on upgrade** (D3 and the id clearing). → Tests pin the pinned ADK's behavior.

## Migration Plan

1. **Before deploy:** add the `OPENAI_API_KEY` GitHub Actions secret. `docker-publish.yml` writes it into the deployed `.env` in place of `OPENROUTER_API_KEY`.
2. **Deploy:** set `ROOT_AGENT_MODEL` to a catalog id. The legacy `openrouter/openai/<id>` form still works.
3. **Saved Telegram selections** are converted when read, so no data migration is needed.
4. **Existing sessions** replay older turns portably, without reasoning.
5. **Rollback:** revert the PR and restore the OpenRouter secret. Verified: the pre-change OpenRouter request builder, run on session events written by the new adapter, builds valid messages and leaves out the empty thought parts and `part_metadata`. No data cleanup is needed.

## Open Questions

- Exact GPT-6 prices and Sol's default effort, copied from OpenAI's model pages during implementation. Doesn't change the design.
