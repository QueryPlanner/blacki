# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-17

### Added
- Migrate Telegram bot to official Bot API with streaming support via `sendMessageDraft` (#13).
- Add `run_user_turn_streaming()` to ADK runtime for real-time response streaming.
- Add `DraftManager` for throttled draft message updates (300ms interval).
- Add `TelegramApiClient` with direct HTTP calls using `httpx`.
- Add Pydantic models for Telegram Bot API types.

### Changed
- Replace `python-telegram-bot` with native `httpx`-based API client.
- Stream thoughts and content separately as italicized draft messages.

### Removed
- Remove `python-telegram-bot` dependency.

## [0.2.0] - 2026-03-28

### Added
- Integrate mem0ai memory with LiteLLM and OpenRouter support (#29).

### Changed
- Refactor mem0 integration into a dedicated package layout (#31).
- Upgrade google-adk to 1.25.1 (#27).

### Fixed
- Use dynamic image name in Docker Compose and deploy flow (#28).

## [0.1.1] - 2026-02-20

### Fixed
- Prevent overlapping template replacements in `init_template.py`.
- Mandated explicit sequence for CI checks in `AGENTS.md`.

### Added
- Added `.gemini/skills/google-adk/SKILL.md`.

## [0.1.0] - 2026-02-08

### Added
- **Automated Server Setup:** Introduced `setup.sh` to automate system updates, Docker installation, firewall (UFW) configuration, and Fail2Ban setup.
- **Production Deployment Workflow:** Enhanced GitHub Actions (`docker-publish.yml`) to securely inject environment variables (DB credentials, API keys) into the production server.
- **Configurable Connection Pooling:** Exposed PostgreSQL connection pool settings (`DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, etc.) via environment variables.
- **Observability Configuration:** Added support for configuring `ROOT_AGENT_MODEL` and Langfuse keys (`LANGFUSE_PUBLIC_KEY`, etc.) via deployment secrets.
- **Testing Standards:** Added `AGENTS.md` with strict guidelines for AI assistants, enforcing real-code testing over mocking internal logic.
- **Documentation:** Updated `README.md` and `docs/DEPLOYMENT.md` with comprehensive deployment guides.

### Changed
- Refactored `agent.py` to dynamically load `LiteLlm` for OpenRouter models.
- Standardized CI checks (`ruff`, `mypy`, `pytest`) to run before every build.

### Fixed
- Resolved `ValueError: Missing key inputs argument` by ensuring API keys are properly injected into the container environment.
- Addressed interactive prompt issues in `setup.sh` by setting `DEBIAN_FRONTEND=noninteractive`.

[Unreleased]: https://github.com/QueryPlanner/google-adk-on-bare-metal/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/QueryPlanner/google-adk-on-bare-metal/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/QueryPlanner/google-adk-on-bare-metal/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/QueryPlanner/google-adk-on-bare-metal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/QueryPlanner/google-adk-on-bare-metal/releases/tag/v0.1.0
