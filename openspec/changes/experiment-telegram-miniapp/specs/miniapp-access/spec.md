# Spec delta

## Purpose

Allow authorized Telegram users to open Blacki's Mini App and act on their own records without exposing operator or sandbox administration.

## ADDED Requirements

### Requirement: Verified Telegram access

The system SHALL verify signed Telegram launch data, its freshness, and existing Blacki access policy before granting live API access. Client-supplied identity or record identifiers SHALL NOT grant ownership.

#### Scenario: A valid existing user opens the app
- **WHEN** an authorized user supplies valid launch data within the configured freshness window
- **THEN** the app receives a short-lived session scoped to that user
- **AND** forwarded links to another user's records remain inaccessible

#### Scenario: Launch data is invalid or stale
- **WHEN** launch verification fails or Blacki access has been revoked
- **THEN** live data and mutations are denied with a recoverable reopen or access-required message
- **AND** fixture mode is never used as an authentication fallback

### Requirement: Consistent ownership and private routing

The app SHALL resolve the authenticated user to each feature's existing ownership identity. Its public routes SHALL expose only Mini App assets and scoped APIs; operator views, public ADK endpoints, provider credentials, and sandbox administration SHALL remain outside that route set.

#### Scenario: The same user opens meals and files
- **WHEN** the user switches between existing meals and durable files
- **THEN** both screens show that user's existing data despite different backend ownership keys
- **AND** no client request can choose another user's identity

### Requirement: Safe repeated and conflicting saves

Each mutation SHALL have an operation identity that makes retries idempotent within its advertised validity window. Expired operations SHALL be rejected rather than replayed as new. Record edits SHALL detect intervening Blacki changes and require reconciliation instead of silently overwriting them. Destructive actions SHALL identify the target and require explicit confirmation.

#### Scenario: A save response is lost
- **WHEN** the client retries the same operation after a network interruption
- **THEN** the result identifies the single committed mutation without creating a duplicate

#### Scenario: Chat changes an open record
- **WHEN** a user submits an editor based on an older record revision
- **THEN** the app shows the conflict and current record without losing the user's draft

### Requirement: Isolated demonstration data

The comparison prototype SHALL clearly label fixture data, support deterministic reset, and prevent fixture operations from contacting live integrations or changing production records.

#### Scenario: A demo reminder is created
- **WHEN** the user saves a reminder in fixture mode
- **THEN** only the resettable demo data changes and no Telegram notification is scheduled
