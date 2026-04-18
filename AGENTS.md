# Google ADK on Bare Metal

## Philosophy

**Blacki** is a personal assistant agent designed to run on cheap self-hosted infrastructure. The guiding principle: **keep the agent lightweight, delegate heavy lifting to managed services.**

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

## Project Overview

**Google ADK on Bare Metal** is a production-ready template designed for building and deploying AI agents using the Google Agent Development Kit (ADK) on self-hosted infrastructure. It removes cloud provider lock-in by providing a clean, performant, and observable foundation that runs on bare metal, VPS, or private clouds.

### Key Technologies
*   **Language:** Python 3.13+
*   **Framework:** Google ADK (`google-adk`)
*   **Model Interface:** LiteLLM (supports Google, OpenRouter, etc.)
*   **Server:** FastAPI
*   **Database:** PostgreSQL (via `asyncpg`)
*   **Observability:** OpenTelemetry (OTel) with Langfuse support
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