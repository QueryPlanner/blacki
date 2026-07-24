# Day-two operations

## Check status

```bash
docker compose -f compose.yaml -f compose.prod.yaml ps
docker compose -f compose.yaml -f compose.prod.yaml logs --tail=100 agent
curl --fail http://127.0.0.1:8080/ready
```

The container health status is the application readiness check. SQLite is
critical; optional memory is not. Use `/live` only to distinguish an alive
process from a dependency-readiness failure.

## Follow logs

```bash
docker compose -f compose.yaml -f compose.prod.yaml logs --follow agent
```

Blacki also appends application JSON logs and traces to `logs/` on the host.
Those files are separate from Docker's stdout logs and are not rotated by
Docker log-driver settings. Monitor their size:

```bash
du -sh logs
```

Define a host retention or log-rotation policy before long-running production
use. Archive or clear old files only during a maintenance window after keeping
anything required for incident analysis.

## Restart

```bash
docker compose -f compose.yaml -f compose.prod.yaml restart agent
```

HTTP and ADK web sessions are in memory and reset. SQLite-backed tool state and
mounted local memory remain.

## Back up

The state directories are:

- `.adk_state/` for SQLite-backed application data; and
- `data/` for optional local Mem0/Qdrant data.

For a consistent filesystem copy, briefly stop the service:

```bash
docker compose -f compose.yaml -f compose.prod.yaml stop agent
tar -czf blacki-state-backup.tgz .adk_state data
docker compose -f compose.yaml -f compose.prod.yaml start agent
```

Store the archive away from the VPS. Back up `.env` separately in an encrypted
secrets system; it contains credentials and is intentionally excluded from the
state archive.

If Mem0 uses Qdrant Cloud, follow the provider's backup and retention
procedures instead of assuming `data/` contains the remote vectors.

## Upgrade a source-built deployment

1. Read `CHANGELOG.md` and compare `.env.example` for new settings.
2. Back up state.
3. Pull with fast-forward-only and rebuild:

   ```bash
   git pull --ff-only
   docker compose -f compose.yaml -f compose.prod.yaml up --build -d
   docker compose -f compose.yaml -f compose.prod.yaml ps
   ```

4. Inspect logs and the health payload.

Compose recreates the container without deleting the mounted state
directories.

## Upgrade a prebuilt-image deployment

Only use this path when the configured image tag and registry access are
verified:

```bash
docker compose -f compose.yaml -f compose.prod.yaml pull
docker compose -f compose.yaml -f compose.prod.yaml up --no-build -d
docker compose -f compose.yaml -f compose.prod.yaml ps
```

Pin a release tag for reproducible upgrades. A floating branch tag trades
convenience for weaker rollback guarantees.

## Automated deployment safeguards

Pull requests that change application source or deployment inputs build the
production image and start it with `compose.smoke.yaml`. The smoke service uses
disposable container state, publishes no host port, and forces Telegram off so
validation cannot poll the production bot or execute scheduled reminders. It
does exercise the image entrypoint, ADK instrumentation, application startup,
SQLite initialization, and `/ready`. Model-provider and Telegram credentials
remain separate integration checks.

The `main` deployment workflow repeats that isolated startup check against the
exact image digest before stopping the active service. If the active deployment
is healthy, the workflow preserves its revision, environment, and image. A
failed promoted `/ready` check automatically restores that captured deployment.
The successful image digest is persisted as `IMAGE` in `.env` so later Compose
commands keep using the verified artifact.

Deployment credentials are serialized without Compose interpolation and staged
in unique, private transfer directories. The workflow removes those temporary
files after both successful and failed deployment attempts.

Automatic rollback is unavailable when no healthy service is running at the
start of deployment. The workflow reports that condition explicitly instead of
treating an unhealthy container as a safe rollback target.

## Roll back

For a source build, switch to a known release tag or commit and rebuild. Restore
the state archive only if the application data format changed or state was
damaged; code rollback and data rollback are separate decisions.

For a prebuilt image, restore the previous image tag in `.env` and run:

```bash
docker compose -f compose.yaml -f compose.prod.yaml pull
docker compose -f compose.yaml -f compose.prod.yaml up --no-build -d
```

## Remove containers without removing state

```bash
docker compose -f compose.yaml -f compose.prod.yaml down
```

This removes the service container and network. The bind-mounted
`.adk_state/`, `data/`, and `logs/` directories remain on the host.
