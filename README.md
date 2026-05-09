# Telegram Bot Template - Google ADK on Bare Metal

Deploy your own AI assistant on Telegram with Google ADK — no cloud lock-in, no per-request fees. Run on a **$5/mo VPS** (vs Railway's $20/mo minimum). Use any LLM via OpenRouter. Own your infrastructure.

**Why This Matters**

| Platform | Monthly Cost | Lock-in | LLM Choice |
|----------|-------------|---------|------------|
| Railway | $20+ | High | Limited |
| Vercel AI | $20+ | High | Limited |
| **Blacki (this project)** | **$5** | **None** | **Any via OpenRouter** |

A **production-ready template** for building and deploying Google ADK agents on your own infrastructure (bare metal, VPS, or private cloud) without the complexity or lock-in of heavy cloud providers.

**Philosophy**
We believe you should own your agents. This template is designed to strip away the "cloud magic" and give you a clean, performant, and observable foundation that runs anywhere—from a $5/mo VPS to a Raspberry Pi cluster.

## Key Features

- 🐳 **Deploy Anywhere**: Pre-configured Docker & Compose setup. Runs on Hetzner, DigitalOcean, or your basement server.
- 🛠️ **Automated Setup**: Includes a `setup.sh` script to harden your server (UFW, Fail2Ban) and install dependencies in minutes.
- 🔄 **CI/CD Included**: GitHub Actions workflow builds multi-arch images (AMD64/ARM64) and pushes to GHCR automatically.
- 🔭 **Open Observability**: Built-in OpenTelemetry (OTel) instrumentation. Configure any OTLP-compatible backend (Axiom, Jaeger, Honeycomb, Langfuse, etc.) via standard environment variables.
- 🚀 **Modern Stack**: Python 3.11+, `uv`, `fastapi`, `asyncpg`.
- ⚡ **Fast Response Times**: In-memory sessions for low-latency agent responses.

## Quickstart

### Prerequisites
- Python **3.11+**
- [`uv`](https://github.com/astral-sh/uv)
- An LLM API Key (OpenRouter or Google)

Optional:
- A Postgres connection string (for Reminders system)
- A Telegram bot token (for Telegram bot integration)

### 1) Configure Environment

**Quick Start (Minimal Config)**

Copy `.env.minimal` to `.env` for the minimal required configuration:

```bash
cp .env.minimal .env
```

Edit `.env` and set:
- **`AGENT_NAME`**: Unique ID for your agent.
- **`OPENROUTER_API_KEY`**: Get one at https://openrouter.ai/keys
- **`TELEGRAM_BOT_TOKEN`**: Get from @BotFather on Telegram

**Full Configuration**

For all available options, copy `.env.example` instead:

```bash
cp .env.example .env
```

### 2) Install Dependencies

```bash
uv sync
```

### 3) Run Locally

```bash
uv run python -m blacki.server
```
Visit `http://127.0.0.1:8080`.

## Deployment: It's Just One Command

We've simplified deployment to the absolute basics. No Kubernetes required.

### Option 1: Using the Pre-built Image (Recommended)

Since we include CI/CD, every push to `main` builds a fresh image. On your server:

```bash
# 1. Pull the latest image
docker pull ghcr.io/queryplanner/google-adk-on-bare-metal:main

# 2. Start the service
docker compose up -d
```

### Option 2: Build Yourself

```bash
git pull
docker compose up --build -d
```

👉 **[Read the Full Deployment Guide](docs/DEPLOYMENT.md)**

## Upgrading

1. Pull the latest changes:
   ```bash
   git pull
   ```

2. Check `.env.example` for new configuration options:
   ```bash
   git diff HEAD~1 .env.example
   ```

3. If new keys are required, add them to your `.env` file.

4. Restart the server:
   ```bash
   docker compose restart
   ```

For breaking changes, see [CHANGELOG.md](CHANGELOG.md).

## Observability

The template comes pre-wired with **OpenTelemetry** using standard OTLP environment variables. Configure your preferred backend (Axiom, Jaeger, Honeycomb, Langfuse, etc.) by setting `OTEL_EXPORTER_OTLP_*` variables in your `.env`. You are not locked into any specific observability vendor.

## Documentation

- [Development Guide](docs/development.md)
- [Architecture](docs/architecture.md)
- [Observability Setup](docs/base-infra/observability.md)
