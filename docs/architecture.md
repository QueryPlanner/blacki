## Architecture (minimal, pragmatic)

### Why this repo exists

Google ADK is useful even without Google Cloud:

- You can run the ADK Dev UI locally on your own infrastructure
- You can use a non-Google model provider via `LiteLlm` and `OpenRouter`
- You can persist sessions in a regular database (Postgres)

### Key choices

- **Entry point**: `python -m blacki.server`
  - Wraps `google.adk.cli.fast_api.get_fast_api_app(...)`
  - Uses in-memory sessions by default for fast response times
  - Configures OpenTelemetry for vendor-neutral tracing (Langfuse auto-config included)
- **Agents directory**: `src/`
  - ADK Dev UI lists *directories* under `agents_dir`.
- **Main Agent**: `src/blacki/agent.py`
  - Contains `root_agent` to keep ADK discovery simple.
- **DB URL normalization**: Handled in `server.py`
  - Converts standard Postgres URLs (e.g. `postgresql://`) to asyncpg-compatible ones (`postgresql+asyncpg://`)

### Session Architecture (Performance-First)

ADK uses **in-memory sessions by default** for maximum performance:

- In-memory sessions: ~5ms load time
- Postgres sessions: ~2-4.6 seconds load time (40% of total response time)

**Why in-memory?** Each ADK turn loads the entire conversation history synchronously. On a cheap VPS, the network round-trip to Postgres adds unacceptable latency before the LLM can even start thinking.

**Postgres is reserved for:**
- Reminders system (persistent scheduled tasks)
- Any other application-level persistence needs

To enable persistent ADK sessions (not recommended for most use cases), set `AGENT_ENGINE` to use Google Agent Engine.

### What ADK uses sessions for

ADK session state stores:

- session rows (IDs + state)
- events (conversation history / tool calls)
- app/user state snapshots

In-memory sessions are lost on restart, but the trade-off is worth it for the latency improvement. The agent remains functional and responsive without the database overhead.
