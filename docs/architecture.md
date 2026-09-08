# Architecture

Blacki keeps the agent process small and delegates expensive capabilities to
managed APIs when they are enabled.

## Runtime

The container runs:

```text
python -m blacki.server
```

`src/blacki/server.py` creates a FastAPI application with Google ADK's
`get_fast_api_app`, initializes SQLite-backed storage, optionally starts
Telegram long polling, and exposes `/live`, `/ready`, and `/health`.
It also mounts the package-backed private observability dashboard at
`/dashboard` with APIs for aggregate statistics, users, sessions, local logs,
and local traces. LiteLLM-backed requests additionally write content-free
usage and cost records to the local SQLite ledger so the dashboard can show
per-user and per-session spend without parsing prompts or responses.

`src/blacki/agent.py` creates the `LlmAgent`, selects a native Gemini model or a
LiteLLM/OpenRouter model from the environment, registers tools, and assembles
the ADK application plugins.

## Application composition

The application dependency direction is:

```text
server -> agent -> tools
              -> prompts and plugins
              -> models
              -> observability
server -> runtime.adk -> ADK sessions and runners
```

`agent.py` is the ADK composition and discovery entry point. It selects tool
lists from `tools/registry.py`, selects models through `models/factory.py`, and
assembles the prompt and observability plugins. `models/inference.py` owns
request-scoped model settings. `models/capabilities.py` keeps OpenRouter
metadata at the provider boundary.

`runtime/adk.py` owns the shared ADK runner, session versioning, confirmation
handling, and Telegram-facing turn orchestration. `prompts/instructions.py`
contains root and worker instruction text. `prompts/policies.py` contains
domain routing and response-policy plugins. These packages remain separate so
model and prompt code do not import the application composition entry point.

Keep `blacki.agent`, `blacki.server:main`, and `python -m blacki.server` as the
stable discovery and execution interfaces. New internal imports should point to
the owning package rather than recreating root-level modules.

## Request paths

### Telegram

1. The Telegram bot polls Telegram's HTTPS API for updates.
2. A Telegram chat is mapped to the ADK runtime.
3. Text is sent as an ADK text part. Telegram photos and `image/*` documents
   are downloaded, validated, and sent as native inline image parts. Media
   groups become one ordered, separately labelled multimodal turn. Native image
   uploads never require sandbox materialization; R2 cataloging is best-effort.
4. The agent calls the configured model and tools.
5. The response is sent back through Telegram's API. When the private Kokoro
   tool is enabled and selected, it synthesizes a bounded MP3 in memory and
   sends it directly to the same chat or topic through Telegram `sendAudio`.

Long conversations use Google ADK's native token-based event compaction. After
the latest prompt reaches 200,000 tokens, ADK summarizes older conversation
events after the successful turn and retains the latest eight raw events for
immediate context. This is prompt compaction, not a hard session reset: the
same versioned Telegram session continues, the compaction event is persisted,
and `/reset` remains available when a completely new conversation is desired.

For a private chat with the optional Google Health connector, `/connect_health`
creates a short-lived one-time OAuth state and sends a Google authorization URL.
The HTTPS callback consumes the state, exchanges the code, resolves Google's
server-side identity, and stores only an encrypted refresh token plus safe
connection metadata. A bounded background job refreshes and reads recent data,
normalizes it into daily SQLite records, and the Telegram commands and
`get_health_summary` tool read those normalized records. When both nutrition
scopes are granted, a durable coordinator queues existing meals once per Google
account. New meal mutations from eligible private chats also enqueue durable,
account-bound Google `nutrition-log` revisions. The export worker
retries pending work independently of health imports, preserves operation
ordering per meal, and exposes safe pending, synced, failed, and
authorization-required counts. Tokens, raw provider payloads, meal
descriptions, and provider identifiers never enter application logs or traces;
the user's own meal text can still appear in the private Telegram conversation
as the normal result of logging a meal.

Long polling is outbound. It does not require a public webhook, domain, TLS
certificate, or inbound application port.

### HTTP and ADK web interface

FastAPI listens on port 8080 inside the container. Docker Compose maps that port
to `${HOST_BIND_IP:-127.0.0.1}:${HOST_PORT:-8080}` on the host in the
production overlay. It remains loopback-only by default. The web interface is
disabled in the VPS samples and can be reached securely through an SSH tunnel
when enabled.

The dashboard is always available on the local server. Set `HOST_BIND_IP` to
the server's Tailscale IPv4 address for direct tailnet access, or keep the
loopback bind and use Tailscale Serve for tailnet-only HTTPS access from the
repository directory:

```bash
HOST_PORT="$(sed -n 's/^HOST_PORT="\{0,1\}\([0-9][0-9]*\)"\{0,1\}$/\1/p' .env | tail -n 1)"
tailscale serve --bg "localhost:${HOST_PORT:-8080}"
```

The `--bg` flag keeps the Serve configuration across host reboots and service
restarts; it remains tailnet-only. Configure Tailscale ACLs and device
restrictions for the operator devices. Do not use Tailscale Funnel or set
`HOST_BIND_IP=0.0.0.0` because the dashboard contains private user chats.

## Persistence boundaries

Blacki uses different stores for different responsibilities:

| Responsibility | Store | Persists across restart? |
| --- | --- | --- |
| ADK HTTP/web conversation sessions | In-memory ADK session service | No |
| Reminders, calories, workouts, preferences, declarative data | SQLite | Yes |
| Telegram runtime session metadata | SQLite-backed application state | Yes |
| Telegram photo parts in ADK history | SQLite-backed session events | Yes |
| Optional Mem0 memory with local Qdrant | `/app/data` | Yes with the Compose volume |
| Optional Mem0 memory with Qdrant Cloud | Managed Qdrant | Provider-managed |
| Zepto OAuth credentials | `/app/data/credentials/zepto-mcp-remote/` | Yes with the Compose volume |
| Google Health refresh tokens and normalized summaries | SQLite (`tools.db`), tokens encrypted at rest | Yes with the Compose volume |
| Google Health meal export revisions and retry state | SQLite (`tools.db`), payloads account-bound and sent over HTTPS | Yes with the Compose volume |
| Application logs and traces | JSON files under `/app/logs` | Yes with the Compose volume |

Compose maps `.adk_state/`, `data/`, and `logs/` from the host. Back up the
first two as application state.

Native Telegram photos and image documents are capped at 10 MB because ADK
retains inline user parts in session history for later conversation turns. This
bounds session database growth and image replay costs while preserving
multimodal context. Non-image Telegram documents continue to use the temporary
per-session sandbox path.

## Managed integrations

Optional tools follow the project's cloud-first principle:

- models use OpenRouter or Google;
- search can use Exa with Brave as a fallback;
- browser automation can use Browser Use Cloud;
- speech synthesis can use a private Kokoro API reachable over Tailscale;
- grocery shopping can use Zepto's hosted MCP server for one allowlisted,
  shared account;
- vector memory can use Qdrant Cloud; and
- code execution can use an OpenSandbox server; and
- health summaries can use Google Health API after private Telegram OAuth.

Each integration degrades independently when its credentials are absent.
Startup still requires at least one model API key.

Kokoro speech delivery is registered only on the Telegram-specific root agent.
The public ADK runner and delegated task worker cannot call it. Blacki sends
only the requested speech text to the configured Kokoro endpoint, accepts a
bounded MP3 response, keeps it in memory, and serializes synthesis through
Telegram upload so concurrent chats cannot retain multiple audio payloads. It
does not retry Telegram delivery automatically. Configuring the tool disables
content-rich ADK and OpenInference logging, although the local ADK session
database still retains tool calls and arguments as normal conversation history.

Zepto is registered only on a Telegram-specific root-agent runner. The public
ADK HTTP runner and delegated task worker never receive the Zepto toolset.
Unauthorized Telegram identities are rejected before Blacki opens an MCP
connection. ADK confirmation is required only when an order or payment tool
uses `confirmOrder=true`; other individual Zepto tools run directly. Its OAuth
files are plaintext protected by a `0700` directory and `0600` file
permissions; they are not encrypted. Shopping prompts, tool calls, and results
remain in the local ADK session database and are sent to the configured model
as part of normal agent execution.

Google Health summaries are a separate read-only boundary. The connector uses
the current Google Health API, not the legacy Fitbit Web API. It requests the
current read-only activity/fitness, measurements, and sleep scopes plus
`googlehealth.nutrition.readonly` and `googlehealth.nutrition.writeonly` for
optional meal export. It handles missing or partially imported categories as
unavailable. Health commands reject group chats, and the summary tool requires
private Telegram session state. Meal export performs one durable historical
backfill per Google account and keeps local save status separate from remote
sync status. `/disconnect_health`
requires an explicit inline-button confirmation, cancels future meal sync,
retains local calorie logs, and does not purge records already sent to Google;
requests already submitted may still complete.

The import and meal-export jobs share the existing scheduler process but remain
independent: health imports use the configured interval and meal export runs
every minute. Run only one active scheduler process per `tools.db`; the
deployment does not claim cross-process dispatch leases.

### Sandbox credential threat model

Sandbox commands may process untrusted uploaded files, fetched web content, and
nested-agent instructions. Any of those inputs can attempt prompt injection or
run shell commands that inspect the process environment. Blacki therefore treats
every general-purpose sandbox as untrusted:

- model, repository, search, OpenRouter, Google, Telegram, and application
  credentials are never copied into the sandbox environment;
- `SANDBOX_API_KEY` authenticates the host-side OpenSandbox connection only;
- sandbox SDK exception details are not returned to tools or written to logs,
  because provider errors can echo credential material; and
- the Gemini CLI sandbox skill is not registered while no least-privilege
  credential broker exists.

An authenticated capability must be implemented as a separately authorized,
short-lived broker operation. Adding a standing environment variable is not an
acceptable opt-in path.

## Health semantics

`/live` is side-effect-free and process-only. `/ready` checks the already
initialized SQLite connection and returns HTTP 503 during startup or database
failure. `/health` delegates to the same implementation for compatibility.
Optional Mem0 memory is not readiness-critical and probes never lazily
initialize it.

## Security boundary

The default deployment is not a public web application:

- the host port binds to loopback;
- the production Compose overlay disables the web interface and reload by
  default;
- Telegram needs outbound HTTPS only;
- secrets live in an ignored `.env` file; and
- `.dockerignore` excludes secrets and runtime state from image builds.

Public web access, authentication, TLS termination, and reverse-proxy
configuration are intentionally separate architectural decisions. A direct
Tailscale bind or Tailscale Serve can provide private network reachability, but
the application deliberately does not implement HTTP Basic auth, cookies, or
Tailscale identity-header authentication.
