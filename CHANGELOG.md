# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add a tested Material for MkDocs documentation site
- Add a safe first-VPS deployment and operations contract
- Add private Kokoro speech delivery for Telegram replies

### Changed
- Make Docker Compose private, persistent, and health-checked by default
- Enable the same-privilege ADK task worker by default
- Restrict owner infrastructure deployment to the QueryPlanner repository

## [0.1.0] - 2026-05-07

### Added
- Improve DX with startup validation and docs (#70)
- Send images as photos for better UX (#66)
- Add date parameter to log_meal and edit_meal tools (#62)
- Add Mem0 secrets to deploy workflow (#57)
- Add mem0 memory integration with CRUD tools (#54)
- Sandbox file transfer and deepseek support (#51)
- Add calorie and workout tracking tools (#49)
- Add GH_TOKEN for GHCR pull auth (#46)
- Add explore-repo skill, remove mcp (#44)
- Add entrypoint for code interpreter sandbox (#43)
- Telegram intermediate responses with thinking filtering (#42)

### Changed
- Make AGENT_NAME configurable via environment variable (#65)
- Refactor architectural debt - global singletons, duplicate patterns, and module-level initialization (#59)
- Refactor vendor-neutral OTLP observability config (#58)
- Use in-memory ADK sessions for speed (#48)

### Fixed
- Secure deployment workflow (#74)
- Use read_bytes for binary file support (#64)

### Removed
- Remove tables for Telegram compatibility (#53)
- Remove agent-browser from docker image (#52)
- Remove GH_TOKEN from deployment workflow (#45)

[Unreleased]: https://github.com/QueryPlanner/blacki/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/QueryPlanner/blacki/releases/tag/v0.1.0
