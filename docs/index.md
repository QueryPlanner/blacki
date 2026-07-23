# Blacki

Run a personal Google ADK assistant on a Linux server you control.

Blacki packages the agent, Telegram integration, SQLite-backed tools, optional
memory, and local observability into one Docker Compose service. The supported
VPS path is intentionally narrow: Ubuntu or Debian, Docker Engine, and the
Docker Compose plugin.

## Choose your path

<div class="grid cards" markdown>

-   **Deploy to a VPS**

    ---

    Start from a clean server, configure one model provider, and bring up a
    private-by-default Telegram assistant.

    [First VPS deployment](DEPLOYMENT.md)

-   **Connect Telegram**

    ---

    Create a bot with BotFather, configure long polling, and verify the two
    supported commands.

    [Telegram setup](telegram-setup.md)

-   **Develop locally**

    ---

    Install with uv, run the server or Docker Compose, execute tests, and
    preview the documentation.

    [Development](development.md)

-   **Fix a problem**

    ---

    Diagnose startup, model, Telegram, health, port, persistence, and
    permission failures.

    [Troubleshooting](troubleshooting.md)

</div>

## Deployment contract

- Docker Compose is the supported VPS runtime.
- The host port binds to `127.0.0.1` unless you explicitly change it.
- Telegram uses outbound long polling and needs no inbound application port.
- `AGENT_NAME` and at least one model API key are required.
- SQLite and optional local memory data are stored in host-mounted directories.
- HTTP and ADK web sessions are in memory and reset on restart.
- Logs and traces are written to local JSON files; remote OTLP export is not
  currently configured by environment variables alone.

## Next steps

After the first deployment, read [Day-two operations](operations.md) for
backups, upgrades, rollback, and log retention. The
[Architecture](architecture.md) guide explains why the runtime is designed this
way.
