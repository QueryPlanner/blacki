# Tasks

## 1. Pin the ADK behaviors the design depends on

- [x] 1.1 Using ADK 2.5.0's real contents processor (no mocks), prove the following, with SQLite in `tmp_path`:
  - (a) a model event with an empty-text thought part plus parallel function calls keeps every part, in order, in `LlmRequest.contents`;
  - (b) `part_metadata` survives `DatabaseSessionService` save and reload;
  - (c) `adk-*` call and response ids reach the adapter as `None`.

  Verify with `tests/test_openai_responses_history.py`, which fails if any of these breaks.

## 2. Model catalog and id normalization

- [x] 2.1 Add `models/catalog.py` with entries for `gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.1-sol` and `gpt-6-astra`: efforts, default effort, `accepts_sampling_at_none`, and prices (input, cached input, optional cache write, output) copied from OpenAI's model pages, with the source URL in a comment. Add `normalize_model_id`. Verify with parametrized tests for plain ids, `openrouter/openai/<id>`, Gemini ids and unsupported ids.
- [x] 2.2 Add `CatalogCapabilitiesResolver` implementing `ModelCapabilitiesResolver`, and remove the OpenRouter Models API resolver from `models/capabilities.py`. Verify with tests: Luna offers `none`, Astra doesn't, unknown models resolve to `None`.

## 3. Request translation (ADK → Responses)

- [x] 3.1 Add `models/openai_request.py`. It handles:
  - user text and images, `instructions`, and function tools, with schema normalization moved from `openrouter_request.py`;
  - `store=False` and the encrypted-reasoning include;
  - the reserved `extra_body` filter;
  - order-based call/result pairing for `id=None`, with synthetic `call_blacki_*` ids, and placeholder outputs for unanswered calls, carried over from `openrouter_request.py:154-164, 314, 433-476`.

  Verify with request tests for each content kind, for ADK confirmation-flow calls with cleared ids, and that `store` is always `false` with no `previous_response_id`.
- [x] 3.2 Implement D2 and D4: same-model replay of stored items in order, portable fallback otherwise, and D5 orphan pruning. Verify with tests:
  - reasoning item is identical to what was returned and directly before its function calls;
  - an empty-summary reasoning item is replayed;
  - no reasoning is sent when the model differs;
  - an orphaned reasoning item is dropped.
- [x] 3.3 Take `reasoning` out of `extra_body` and send the typed `reasoning` with `summary: "auto"`. Send `temperature` and `top_p` only when effort is explicitly `none` and the catalog allows it. Verify with tests for effort set, effort unset (no sampling parameters) and `none` on Luna.

## 4. Response translation (Responses → ADK)

- [x] 4.1 Add `models/openai_response.py` for non-streamed responses. It covers:
  - one part per output item, with `part_metadata["blacki_openai"]` storing the normalized **requested** model id;
  - refusal content becoming visible text;
  - commentary `phase` recorded in the metadata;
  - the usage mapping, with reasoning inside output;
  - finish reasons `MAX_TOKENS` and `SAFETY`.

  Verify with tests from recorded-shape JSON fixtures, including `incomplete` for both `max_output_tokens` and `content_filter`, a refusal, and a dated `response.model` that differs from the requested id.
- [x] 4.2 Add the stream accumulator:
  - partial text and summary deltas;
  - a single final response from either `response.completed` or `response.incomplete`;
  - `response.failed` and `error` raising typed errors.

  Verify with event-sequence tests asserting the partial count, exactly one final response, and stored items identical to the non-streamed path.

## 5. Adapter, errors and retry

- [x] 5.1 Add `models/openai_responses.py` `OpenAIResponsesLlm(BaseLlm)` over `AsyncOpenAI`: a client per call closed in `finally`, the key in a private attribute, streaming and non-streaming. Verify with tests that mock only the HTTP boundary (an `httpx.MockTransport` client factory) and run the real request and response code.
- [x] 5.2 Map OpenAI SDK errors to typed provider errors (status, code, request id; never key or content) in `utils/exceptions.py`, replacing the OpenRouter errors. Verify with tests for 401, 403, 429 and 500, asserting that messages contain no key or prompt text.
- [x] 5.3 Implement D7: detect a rejection by `code == "invalid_encrypted_content"` or by a `param` pointing at a reasoning item, retry once without reasoning, and add the session to the in-process downgrade set. Verify with tests: retry succeeds; retry fails and raises; the next request in the same session skips reasoning without a failed call; an unrelated 400 doesn't retry.

## 6. Wiring: factory, override plugin, runtime, compaction, privacy

- [x] 6.1 Update `models/factory.py`: an OpenAI catalog model with `OPENAI_API_KEY` gives `OpenAIResponsesLlm`; a native Gemini id keeps its path; a missing key or unsupported id fails at startup with the documented messages. Verify with updated `tests/test_model_factory.py`.
- [x] 6.2 Update `TelegramModelOverridePlugin` in `agent.py:56-110` and `models/inference.py` to implement D10 and the saved-effort fallback:
  - normalize saved selections;
  - apply reasoning only for OpenAI;
  - fall back to the default for non-catalog saved ids, including plain `gemini-*` with an OpenAI root;
  - replace an unsupported saved effort with the model default.

  Verify with override and inference tests for each spec scenario.
- [x] 6.3 In the same plugin, strip `blacki_openai` metadata and empty thought parts from `llm_request.contents` when the effective model is native Gemini (D11). Verify with a test that a session with OpenAI turns, routed to Gemini, builds contents with no `part_metadata`.
- [x] 6.4 Remove the OpenRouter session-affinity header from `runtime/adk.py` `_turn_config`. Verify with a runtime test that `http_options` has no `x-session-id`.
- [x] 6.5 Implement D13 in `runtime/adk.py` `run_user_turn_with_thoughts` and the streaming path: leave commentary-phase text out of the reply. Verify with tests where a commentary message plus a tool call is followed by a final answer, and only the final answer is returned.
- [x] 6.6 Implement D12: a Blacki summarizer passed as `EventsCompactionConfig.summarizer` in `agent.py` that keeps only message text, uses the catalog's lowest non-`none` effort, and records usage. Verify with tests using ADK's real compaction trigger and a mocked HTTP boundary: compacted content has no reasoning parts or metadata, and one ledger row is written.
- [x] 6.7 Implement D11 logging and trace protection:
  - redact `part_metadata` and `thought_signature` in `observability/lifecycle.py` debug dumps;
  - force `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false` with a warning in `observability/setup.py`.

  Verify with tests: debug logs from a replayed turn contain no `encrypted_content`; with experimental capture on, exported span attributes contain no `encrypted_content` or `blacki_openai`; setting the legacy flag to `true` is overridden and warned about.

## 7. Telegram model and thinking UI

- [x] 7.1 Update `/model` in `telegram/settings_menu.py`: catalog validation, a reply listing supported ids for unknown input, refusal when OpenAI is not configured, and usage text "Use /model <model-id>". Verify with updated `tests/test_telegram_model_command.py` and `tests/test_telegram_settings_menu.py`.
- [x] 7.2 Update `/thinking` to use the catalog resolver and refuse a stale effort button. Verify with settings-menu tests for Luna (offers `none`) and Astra (doesn't).
- [x] 7.3 Update `telegram/bot.py` help text, both plain and MarkdownV2 variants, to drop "OpenRouter". Verify with a test that asserts the help text names OpenAI model ids.

## 8. Cost accounting

- [x] 8.1 Replace OpenRouter cost extraction in `observability/costs.py` with the D9 estimate: `cost_kind="estimate"`, `cost_source="catalog_price"`, cache writes priced only when reported, reasoning not added again, and empty cost plus a warning logged once per model when a price is missing. Verify with `tests/test_llm_costs.py` cases for each spec scenario and a streamed call writing exactly one ledger row.
- [x] 8.2 Check that the dashboard renders estimated costs. Verify with dashboard data tests, updated where they assumed provider-reported cost.

## 9. Remove OpenRouter; configuration, deploy and docs

- [x] 9.1 Move the meal-tool request building in `tests/tools/test_meals.py` and `scripts/benchmark_meal_tools.py` to `openai_request.py`. Then delete `models/openrouter.py`, `openrouter_request.py`, `openrouter_response.py`, `openrouter_ids.py` and their tests. Verify that `rg -i openrouter src tests scripts` returns only intentional legacy-id handling and its tests.
- [x] 9.2 Replace `OPENROUTER_API_KEY` with `OPENAI_API_KEY` in:
  - `utils/config.py`, `utils/validation.py` and the `server.py` help text;
  - `.env.example` and `.env.minimal`;
  - `systemd/agent.service`;
  - `.github/workflows/docker-publish.yml` (lines 107 and 180);
  - the sandbox canary in `tests/sandbox/test_manager.py:168`.

  Verify with config, validation, deployment-contract (`tests/test_deployment_contract.py`) and application-layout tests, and `actionlint` on the workflow if it's available.
- [x] 9.3 Update the docs:
  - `docs/architecture.md` and the environment-variable and deployment docs;
  - `docs/troubleshooting.md`, `docs/telegram-setup.md` and `docs/meal-tools-benchmark.md`;
  - the observability cost section;
  - `README.md` and the `CHANGELOG.md` Unreleased entry (marked **BREAKING**);
  - the `brave_search.py:61` docstring and the `AGENTS.md` LLM line;
  - a note on `store: false`, 30-day abuse monitoring and the new GitHub secret.

  Verify that `uv run mkdocs build --strict` passes.
- [x] 9.4 Check the rollback claim: run the pre-change OpenRouter request builder (from git history, in a test-only script, not committed) against session events written by the new adapter. Record in `design.md` whether it tolerates empty thought parts and `part_metadata`. If it doesn't, document the cleanup.

## 10. Integration checks

- [x] 10.1 Run the full local suite: `uv run ruff format`, `uv run ruff check`, `uv run mypy .`, `uv run pytest --cov=src`, with 100% coverage on new and changed code. Rerun from `ruff format` after any fix.
- [x] 10.2 Live smoke test against OpenAI with `gpt-6.1-terra` at effort `high`, through the real runner and `DatabaseSessionService` with streaming on: one turn with at least two tool calls. Confirm the following, logging item types only:
  - later requests carry the earlier reasoning items;
  - a restart mid-session still replays them;
  - the ledger shows an estimated cost;
  - the reply excludes commentary.

## Workflow follow-up

- Before deploy, add the `OPENAI_API_KEY` GitHub Actions secret. This is a repository settings action the owner must take.
- Open the PR with the required headings after local checks pass.
- Archive the change after merge, and sync the three new specs into `openspec/specs/`.
