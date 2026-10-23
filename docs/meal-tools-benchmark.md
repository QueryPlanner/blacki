# Meals tool pilot measurement

Measured on 2026-10-23 against the handlers and OpenRouter request adapter in
this change. These are local tokenizer estimates, not provider usage, bills,
or real-model task success measurements.

## Initial schemas

The seven original meal functions use 820 `cl100k_base` tokens or 850
`o200k_base` tokens after serialization by the production OpenRouter adapter.
The compact `meals` tool plus `describe_action` use 188 or 190 tokens,
respectively. This is a 77.1–77.6% reduction in the initial meals schema.

An all-listed-capabilities configuration has 50 tools before the pilot and
45 afterward. The benchmark payload exercises `build_tools` directly, so it
measures 49 and 44; the running agent adds `PreloadMemoryTool`. It enables
both search providers, sandbox, canonical training, weather, skills, memory,
custom data, reminders, Google Health, durable files, and speech. Gmail
remains unloaded. Its complete schema decreases from
6,885–7,009 tokens to 6,253–6,349 tokens, about 9.2–9.4%. This constructed
configuration is not a measurement of a running deployment.

The generated parameter models also supply discovery schemas. Discovery moves
selected schemas into conversation history; it does not eliminate their cost.

## Controlled workflow replay

A deterministic `BaseLlm` supplied model responses to real ADK `Runner` and
`App` instances. Each run used an isolated SQLite database, the production
root instruction, temporal instruction plugin, domain policy plugin, tool
registry, and `build_chat_request(...).create_kwargs(stream=False)` adapter.
The flat baseline used the original seven handlers and the original nutrition
policy from base commit `4e10c3f`. The compact runs used this change's policy.
The replay verified successful tool responses; the logging case independently
verified exactly one 95-calorie entry in SQLite.

The user messages were "Log one apple with 95 calories on 2026-10-20." and
"Show my calorie summary for 2026-10-20." A direct path called the operation
and then produced a final response. A cold path first called `describe_action`,
then called the operation, then produced a final response. Both compact paths
called the operation through `meals`.

Token counts below sum compact JSON representations of **all input request
payloads**, including tools, instructions, conversation history, and tool
results. They use `cl100k_base`; `o200k_base` showed the same direction.

From the repository root, reproduce the replay and both tokenizer counts with:

```bash
uv run --with tiktoken==0.12.0 python scripts/benchmark_meal_tools.py
```

The tokenizer is an isolated measurement dependency, not an application
dependency. Its first run may download public tokenizer vocabularies. The
harness reads the original policy with `git show 4e10c3f:src/blacki/prompts/policies.py`,
so that base commit must be available locally. It creates and closes temporary
databases and never reads deployment credentials. Temporal context can cause
small count differences on later dates.

| Workflow | Flat input tokens | Compact direct input tokens | Compact cold input tokens |
| --- | ---: | ---: | ---: |
| Log one meal | 19,332 | 18,179 | 28,078 |
| Read calorie summary | 19,324 | 18,171 | 27,635 |
| Model calls per workflow | 2 | 2 | 3 |

The direct path reduces total input by about 6%. Cold discovery increases it
by about 43–45% because it sends the complete context through another model
call. Tokenized model output JSON was 52–61 tokens for the flat paths, 60–69
for direct dispatch, and 96–104 for cold discovery. These output counts describe
the scripted responses, not actual generated or reasoning token usage.

The direct path assumes the model already knows the arguments without needing
discovery in this turn. It does not include an earlier discovery transcript and
is not a measurement of a warmed conversation. The replay does not establish
how often a real model selects either path, actual provider caching or token
framing, latency, or task success rates.

## Validation and rollout boundary

`tests/tools/test_meals.py` exercises all seven actions with real SQLite
storage, local saves with failed remote export and explicit requeue, rejected
arguments without writes, owner isolation, private-chat restrictions,
read-only discovery, privacy, progress, retry identity, and a real runner
continuing completed historical `log_meal` call/response events. The serialized
provider-schema test guards the initial payload reduction independently of
the domain dispatcher.

Run the relevant checks with:

```bash
uv run pytest tests/tools/test_meals.py tests/test_tools_boundary.py
```

The initial schema target is met. This pilot does **not** establish a general
cost reduction. Expansion to other domains requires representative real-model
evaluation of total input/output usage, discovery frequency, repeated actions,
latency, and mutation correctness. A separate discovery call on every ordinary
meal request is not a successful cost outcome.
