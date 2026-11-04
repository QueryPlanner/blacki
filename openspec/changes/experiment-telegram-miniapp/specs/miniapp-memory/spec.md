# Spec delta

## Purpose

Let users inspect and correct the personal memories available to Blacki through direct, owner-scoped controls.

## ADDED Requirements

### Requirement: Browse and search owned memories

The app SHALL provide memory cards, bounded browsing, semantic search when supported, and a clear unavailable state when memory is not configured. It SHALL NOT imply that a partial page is the user's complete memory collection.

#### Scenario: The user searches an existing fact
- **WHEN** the user submits a memory search
- **THEN** matching owned memories appear with edit controls and any available timestamps

### Requirement: Correct or delete an exact memory

The user SHALL be able to edit a selected memory and confirm deletion of an exact memory. Stale selections or ownership failures SHALL leave other memories unchanged. Successful changes SHALL be reflected when the memory is next retrieved by Blacki.

#### Scenario: A memory disappears while its editor is open
- **WHEN** the user attempts to save or delete that memory
- **THEN** the app reports that it is no longer available and does not target another memory
