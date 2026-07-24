# Blacki

Blacki is a self-hosted personal assistant built with
[Google ADK](https://google.github.io/adk-docs/). It runs on your own Linux
server, talks to an LLM through OpenRouter or Google AI Studio, and can expose
the assistant through Telegram long polling or the ADK web interface.

Blacki keeps the runtime small and delegates expensive capabilities to managed
services when configured. You own the server and data volumes; model and
optional service providers may still charge for API usage.

## Start here

| Goal | Guide |
| --- | --- |
| Put Blacki on an Ubuntu or Debian VPS | [First VPS deployment](docs/DEPLOYMENT.md) |
| Create and connect a Telegram bot | [Telegram setup](docs/telegram-setup.md) |
| Work on Blacki locally | [Development](docs/development.md) |
| Understand every setting | [Configuration](docs/base-infra/environment-variables.md) |
| Run upgrades and backups | [Day-two operations](docs/operations.md) |

## First run with Docker Compose

Install Docker Engine with the Compose plugin, then:

```bash
git clone https://github.com/QueryPlanner/blacki.git
cd blacki
cp .env.minimal .env
chmod 600 .env
```

Replace every `replace-me` value in `.env`, then validate and start Blacki:

```bash
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

The HTTP port binds to `127.0.0.1` by default. Telegram polling needs outbound
HTTPS only, so a Telegram deployment does not need a public application port.
See the [deployment guide](docs/DEPLOYMENT.md) for verification, secure browser
access, and upgrade notes.

## What persists

| Data | Host path | Notes |
| --- | --- | --- |
| Tools, reminders, preferences, and Telegram state | `.adk_state/` | SQLite |
| Optional local Mem0/Qdrant data | `data/` | Used only when local memory is configured |
| Application JSON logs and traces | `logs/` | Monitor and rotate these files |
| ADK HTTP/web sessions | Not persisted | In-memory and reset on restart |

Back up `.adk_state/` and `data/`. Keep `.env` private and never commit it.

## Local Python

Blacki supports Python 3.11 through 3.13 and uses
[uv](https://docs.astral.sh/uv/):

```bash
cp .env.example .env
uv sync
uv run python -m blacki.server
```

For the complete workflow and quality checks, see
[Development](docs/development.md).

## Documentation

The documentation site uses Material for MkDocs:

```bash
uv sync --group docs
uv run mkdocs serve
```

Open `http://127.0.0.1:8000`. CI builds the site in strict mode for every
documentation or deployment change.

## License

See [LICENSE](LICENSE).
