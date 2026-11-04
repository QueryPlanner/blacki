# Spec delta

## Purpose

Offer conversation inside the Mini App with useful controls and results while sharing Blacki's existing private conversation safely across entry points.

## ADDED Requirements

### Requirement: Shared private conversation

In-app assistant SHALL use the same private Blacki conversation and authorized toolset as the user's Telegram chat. Visible messages SHALL identify their origin without duplicating turns. Internal reasoning, system instructions, and private tool payloads SHALL NOT be rendered as conversation content.

#### Scenario: The user continues a Telegram conversation in the app
- **WHEN** the user opens In-app assistant and sends a follow-up
- **THEN** the existing private conversation supplies context without copying messages into a new session

### Requirement: Cross-channel turn coordination

New agent messages from either entry point SHALL follow one conversation's existing supersession policy. Direct form saves and experience switches SHALL NOT cancel an agent turn. Supersession SHALL be visible, and already committed tool effects SHALL remain committed.

#### Scenario: A Telegram message arrives during an in-app response
- **WHEN** the newer message supersedes the active turn
- **THEN** the previous response is labelled superseded and the new turn starts after cancellation cleanup

#### Scenario: A form is saved while Blacki is working
- **WHEN** the user saves a meal correction
- **THEN** the direct edit completes without starting another agent turn or cancelling the current one

### Requirement: Structured interactive answers

Both experiences SHALL support focused record editors, choice buttons, labelled sliders, tables, and filterable charts through a fixed validated component vocabulary. Telegram companion SHALL open these in the Mini App; In-app assistant SHALL render them within or alongside the conversation. Every action SHALL recheck current authorization and record state.

#### Scenario: Blacki offers a meal correction control
- **WHEN** a valid answer references an owned meal
- **THEN** the control opens or renders the shared meal editor instead of asking the user to describe the correction in another message

#### Scenario: A result contains unsupported executable content
- **WHEN** the result fails component validation
- **THEN** the app safely renders a text fallback without executing the content

### Requirement: Explicit response delivery

Closing the Mini App SHALL detach its UI without cancelling the server's agent turn. App-origin responses SHALL remain available in the shared conversation without automatically posting duplicate replies to Telegram. Telegram notifications for app-origin work SHALL require a user-enabled preference.

#### Scenario: An in-app response finishes after the app closes
- **WHEN** the user later reopens In-app assistant
- **THEN** the completed reply is available once in the shared conversation
- **AND** Telegram receives no duplicate unless the user requested a completion notification
