# Documentation map

The documentation is organized around the lifecycle of a Blacki installation.

## Get started

- [First VPS deployment](DEPLOYMENT.md) is the supported Docker Compose path.
- [Telegram setup](telegram-setup.md) connects the primary chat interface.
- [Local development](development.md) covers uv, tests, and MkDocs.

## Operate

- [Configuration](base-infra/environment-variables.md) lists deployment and
  application settings.
- [Day-two operations](operations.md) covers status, backups, upgrades, and
  rollback.
- [Troubleshooting](troubleshooting.md) maps common symptoms to checks.
- [Observability](base-infra/observability.md) explains local logs and traces.

## Understand

- [Architecture](architecture.md) describes runtime and persistence boundaries.
- [Docker Compose](base-infra/docker-compose-workflow.md) explains the service
  contract.
- [Docker image](base-infra/dockerfile-strategy.md) explains the multi-stage
  build and non-root runtime.

The MkDocs navigation in `mkdocs.yml` is the source of truth for the published
information architecture.
