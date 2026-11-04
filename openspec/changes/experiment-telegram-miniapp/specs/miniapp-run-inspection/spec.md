# Spec delta

## Purpose

Show each user understandable task progress and recorded usage without exposing the operator dashboard or private tool payloads.

## ADDED Requirements

### Requirement: Safe owned-run progress

The app SHALL show the user's active and recent runs with status, elapsed time, and safe tool-progress labels. It SHALL distinguish complete, failed, superseded, and interrupted runs. Raw prompts, private tool arguments, provider credentials, and other users' records SHALL NOT be exposed through this view.

#### Scenario: A run uses a private integration
- **WHEN** the user inspects its progress
- **THEN** they see a safe action label and outcome without credential or private-input payloads

### Requirement: Honest usage and recovery

Usage views SHALL distinguish provider-reported, estimated, and unavailable costs. Reopening the app SHALL reconnect to a still-running task or show its recorded terminal state. A server restart SHALL NOT silently replay an interrupted state-changing operation.

#### Scenario: A run has no reported price
- **WHEN** the user opens its cost card
- **THEN** unavailable cost is labelled unavailable rather than shown as zero

#### Scenario: The server restarts during a task
- **WHEN** the user reopens that task
- **THEN** it is marked interrupted and any retry is an explicit new action
