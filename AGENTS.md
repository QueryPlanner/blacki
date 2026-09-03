# Blacki Architecture and Development Guide

## Philosophy

**Blacki** (also called BlackKey) is a Telegram-first personal assistant built on
Google ADK and designed to run on inexpensive self-hosted infrastructure. The
guiding principle is to keep the agent lightweight and delegate expensive work
to managed services.

### Self-Hosted Agent, Managed Tools

The agent itself runs on a $5-10/month VPS, but tools should use managed cloud services whenever possible:

- **Browser automation** → Browser Use Cloud (not local Playwright/Puppeteer)
- **Search** → Brave Search API (not self-hosted SearXNG)
- **LLM** → OpenRouter/Google (not self-hosted Ollama)

This architecture keeps the server footprint minimal while giving the agent powerful capabilities. Managed services handle:
- Infrastructure maintenance
- Scaling
- Updates and security patches
- Geographic distribution (e.g., Browser Use proxies)

### Why This Matters

Self-hosting everything sounds appealing, but for a personal assistant:
- Browser automation requires headless Chrome, which consumes significant RAM/CPU
- Search indices need constant crawling and updates
- LLM inference needs GPU or is painfully slow

By delegating to managed services, the agent stays fast, cheap, and reliable. The server can be a tiny VPS that costs less than a coffee per month.

### Tool Design Principles

1. **Cloud-first**: Default to managed service APIs
2. **Minimal dependencies**: The server should run on minimal resources
3. **Graceful degradation**: Tools should fail gracefully if API keys are missing
4. **Stateless where possible**: Let managed services handle state

## Architecture

Blacki has a small host-side control plane and several separately authorized
capability boundaries:

- `src/blacki/server.py` starts the FastAPI and ADK runtime, health endpoints,
  storage, and optional Telegram polling.
- `src/blacki/agent.py` builds the root agent, model configuration, plugins, and
  the public/delegated agent variants.
- `src/blacki/registry.py` is the tool factory. It decides which tools are
  exposed for the public runner, the Telegram root agent, and delegated workers.
- `src/blacki/prompt.py` contains global behavior rules. Feature-specific
  operating and safety instructions belong in a loaded skill when possible.
- `src/blacki/telegram/` maps private Telegram chats and topics to ADK sessions.
  Telegram long polling is outbound and does not require a public webhook.
- SQLite stores application data, session metadata, OAuth state, and catalogs.
  Secrets and refresh tokens stay on the Blacki host and are never copied into
  a sandbox.

The root agent may receive private, user-scoped integrations such as Google
Health, Gmail, Zepto, TTS, and durable user files. Public ADK requests and
delegated workers receive only the tools explicitly allowed for their boundary.
Shared session-sandbox access does not grant access to another user's accounts.

### File lifecycle and storage boundaries

The sandbox and Cloudflare R2 solve different problems:

- The OpenSandbox instance is session-scoped, created lazily, and temporary.
  Its default lifetime is currently 30 minutes. It is the working area for
  inspection, parsing, code execution, and intermediate results.
- Cloudflare R2 is optional durable storage for supported Telegram attachments.
  When `R2_FILES_ENABLED=true` and the private bucket is configured, Blacki
  writes the attachment to R2 and records owner-scoped metadata in SQLite.
- A supported Telegram upload may therefore have both an R2 object and a
  sandbox working copy. The copies are independent. If R2 fails, processing
  may continue with a temporary sandbox copy and a warning. If the sandbox is
  unavailable after R2 succeeds, the durable object can be restored later.
- `list_user_files` lists the owner's R2 catalog, `restore_user_file` copies an
  R2 object into the current sandbox, and `delete_user_file` removes an exact
  owner-scoped durable object after confirmation.
- Gmail attachment downloads currently write only to the current session
  sandbox. They do not automatically go to R2, do not enter the durable file
  catalog, and are not sent through Telegram automatically. The Gmail result
  files used to keep large email bodies out of model context follow the same
  temporary sandbox boundary.

Treat every sandbox file as untrusted input. Do not execute, extract, or open
downloaded attachments automatically. R2 credentials and other provider
credentials must remain outside the sandbox. Any future feature that moves a
file from a sandbox to R2 must explicitly define ownership, retention, naming,
deduplication, failure cleanup, and user-facing consent.

### Gmail boundary

Gmail is a direct Gmail REST API connector under `src/blacki/gmail/`, not a
general-purpose shared service. OAuth credentials are stored per private
Telegram user, Gmail tools are registered only for the private root-agent
flow, and delegated workers remain unable to call Gmail. The Gmail skill in
`src/blacki/skills/gmail/SKILL.md` supplies the agent-facing usage and safety
rules; loading the skill is not itself an authorization grant.

## Project Overview

Blacki is a production-ready personal assistant built with the Google Agent
Development Kit (ADK) on self-hosted infrastructure. It provides a clean,
observable foundation that can run on bare metal, a VPS, or a private cloud.

### Key Technologies
*   **Language:** Python 3.13+
*   **Framework:** Google ADK (`google-adk`)
*   **Model Interface:** LiteLLM (supports Google, OpenRouter, etc.)
*   **Server:** FastAPI
*   **Database:** PostgreSQL (via `asyncpg`)
*   **Observability:** OpenTelemetry (OTel) with vendor-neutral OTLP configuration
*   **Infrastructure:** Docker, Docker Compose

## Building and Running

### Prerequisites
*   Python 3.13+
*   [`uv`](https://github.com/astral-sh/uv) (Package Manager)
*   Docker & Docker Compose (for containerized deployment)

### Setup
1.  **Configure Environment:**
    Copy `.env.example` to `.env` and set the required variables:
    *   `AGENT_NAME`: Unique ID for the agent.
    *   `DATABASE_URL`: Postgres connection string.
    *   `OPENROUTER_API_KEY` / `GOOGLE_API_KEY`: LLM API keys.

2.  **Install Dependencies:**
    ```bash
    uv sync
    ```

### Execution Commands

| Task | Command | Description |
| :--- | :--- | :--- |
| **Run Locally** | `uv run python -m blacki.server` | Starts the agent server on localhost:8080. |
| **Run (Script)**| `uv run server` | Alternative command using the project script entry point. |
| **Docker Run** | `docker compose up --build -d` | Builds and starts the agent in a Docker container. |
| **Test** | `uv run pytest` | Runs the test suite. |
| **Lint** | `uv run ruff check` | Runs linter checks. |
| **Format** | `uv run ruff format` | Formats code using Ruff. |
| **Type Check** | `uv run mypy .` | Runs static type checking. |

## Development Conventions

### Required cross-feature impact review

Before implementing or materially changing a feature, inspect the existing
features and data paths that it may touch. The implementation agent must
actively consider at least:

- user identity and authorization boundaries;
- Telegram, ADK session, and delegated-worker behavior;
- sandbox lifetime and whether a file is temporary or durable;
- Cloudflare R2 storage, catalog, retention, restore, and deletion behavior;
- OAuth scopes, token ownership, external APIs, and privacy implications;
- model context size, logs, traces, and whether content can leak across users;
- confirmation requirements for state-changing operations; and
- failure and partial-success behavior when one storage or provider is down.

Do not assume that a new file-producing feature should use the same lifecycle
as Telegram uploads. If the request does not specify whether a new artifact is
sandbox-only, copied to R2, sent to Telegram, or retained elsewhere, complete
the read-only investigation first and ask the user one focused question before
choosing a persistence or delivery policy. Explain the current behavior and
the available options so the user can decide. If the user has already decided,
state that assumption in the implementation summary and verify every affected
boundary.

For every new integration, identify which existing agents and toolsets should
see it, whether it should be skill-gated, and how it interacts with existing
storage and privacy controls. Do not broaden a feature's data retention or
external delivery merely because another feature already does so.

### Code Structure
*   **`src/blacki/`**: Contains the core agent logic.
    *   `agent.py`: Defines the `root_agent` and ADK application configuration.
    *   `server.py`: FastAPI server entry point with OTel instrumentation.
    *   `prompt.py`: Manages agent prompts and instructions.
    *   `tools.py`: Helper tools for the agent.

*   **`tests/`**: Unit and integration tests.

### Code Quality
Before committing anything or creating a Pull Request, you **must** ensure all local checks pass. This includes running formatting, linting, type checks, and tests. The CI pipeline will run these same checks and fail if they are not satisfied:

1.  **Format Code:** `uv run ruff format`
2.  **Lint Code:** `uv run ruff check`
3.  **Type Check:** `uv run mypy .`
4.  **Run Tests:** `uv run pytest --cov=src`

**⚠️ CRITICAL WARNING:** If you modify code to fix an error reported by one of these tools (e.g., adding type hints for `mypy` or `# noqa` comments for `ruff check`), you **MUST** re-run the entire suite of checks starting from `ruff format`. Fixing an error for one tool frequently breaks the rules of another (especially formatting). Do not commit until all checks pass consecutively without any further file modifications.

Ensure all steps pass locally before staging and committing files to avoid CI failures.

### Testing Standards for AI Assistants
When asked to write or modify tests, you **MUST** adhere to the following strict guidelines derived from the ADK philosophy:

1.  **Real Code Over Mocks**:
    *   **Do not mock** internal logic (e.g., `LlmAgent`, `Prompt`, `Tool`). Use the real classes.
    *   **Only mock** external boundaries (e.g., `LiteLLM`, `asyncpg`, `Network APIs`).
    *   **Why?** This ensures we test the integration of components, not just isolated units.

2.  **Pytest Best Practices**:
    *   Use **fixtures** (`conftest.py`) for setup/teardown.
    *   Use **`@pytest.mark.parametrize`** for testing multiple inputs/outputs.
    *   Use **`tmp_path` fixture** for any file system operations.
    *   **Strict Mocking**: Always use `create_autospec(spec_set=True)` to ensure mocks match the actual API.

3.  **Test Coverage**:
    *   Every new feature **must** have a corresponding test.
    *   Tests must cover both the "Happy Path" (success) and "Edge Cases" (failure/errors).

### Deployment
*   **Containerization:** The `Dockerfile` provides a multi-stage build optimized for production.
*   **CI/CD:** GitHub Actions workflows (`.github/workflows/`) handle testing, linting, and publishing Docker images to GHCR.
