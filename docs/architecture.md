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
Telegram long polling, and exposes `/health`.

`src/blacki/agent.py` creates the `LlmAgent`, selects a native Gemini model or a
LiteLLM/OpenRouter model from the environment, registers tools, and assembles
the ADK application plugins.

## Request paths

### Telegram

1. The Telegram bot polls Telegram's HTTPS API for updates.
2. A Telegram chat is mapped to the ADK runtime.
3. The agent calls the configured model and tools.
4. The response is sent back through Telegram's API.

Long polling is outbound. It does not require a public webhook, domain, TLS
certificate, or inbound application port.

### HTTP and ADK web interface

FastAPI listens on port 8080 inside the container. Docker Compose maps that port
to `127.0.0.1:8080` on the host by default. The web interface is disabled in
the VPS samples and can be reached securely through an SSH tunnel when enabled.

## Persistence boundaries

Blacki uses different stores for different responsibilities:

| Responsibility | Store | Persists across restart? |
| --- | --- | --- |
| ADK HTTP/web conversation sessions | In-memory ADK session service | No |
| Reminders, calories, workouts, preferences, declarative data | SQLite | Yes |
| Telegram runtime session metadata | SQLite-backed application state | Yes |
| Optional Mem0 memory with local Qdrant | `/app/data` | Yes with the Compose volume |
| Optional Mem0 memory with Qdrant Cloud | Managed Qdrant | Provider-managed |
| Application logs and traces | JSON files under `/app/logs` | Yes with the Compose volume |

Compose maps `.adk_state/`, `data/`, and `logs/` from the host. Back up the
first two as application state.

## Managed integrations

Optional tools follow the project's cloud-first principle:

- models use OpenRouter or Google;
- search can use Exa with Brave as a fallback;
- browser automation can use Browser Use Cloud;
- vector memory can use Qdrant Cloud; and
- code execution can use an OpenSandbox server.

Each integration degrades independently when its credentials are absent.
Startup still requires at least one model API key.

## Health semantics

Docker Compose uses a side-effect-free TCP liveness probe. The `/health`
endpoint performs richer application checks and may initialize optional memory
configuration. It returns HTTP 200 with a `degraded` payload when optional
memory is unavailable, so operators should inspect the JSON body rather than
treat the status code as readiness.

## Security boundary

The default deployment is not a public web application:

- the host port binds to loopback;
- Telegram needs outbound HTTPS only;
- secrets live in an ignored `.env` file; and
- `.dockerignore` excludes secrets and runtime state from image builds.

Public web access, authentication, TLS termination, and reverse-proxy
configuration are intentionally separate architectural decisions.
