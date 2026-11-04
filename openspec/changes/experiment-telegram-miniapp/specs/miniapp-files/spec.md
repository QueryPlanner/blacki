# Spec delta

## Purpose

Give the user a direct browser for their existing durable attachments without changing file retention or granting access to another user's sandbox.

## ADDED Requirements

### Requirement: Searchable owned-file catalog

The app SHALL provide filename search, type filters, list or card views, and selection controls for available durable files. It SHALL show names, sizes, availability, and retention metadata when known. File content and names SHALL be treated as untrusted input.

#### Scenario: The user filters images
- **WHEN** the user selects the image filter
- **THEN** only their available image entries appear, and unsafe content is not executed as a preview

### Requirement: Explicit restore to the current conversation

The app SHALL allow selected catalog files to be restored into the authenticated user's current Blacki conversation sandbox through an explicit action. A failed restore SHALL leave the durable source unchanged and identify any partial success.

#### Scenario: Durable storage works but the sandbox is unavailable
- **WHEN** the user requests restore
- **THEN** the app reports the unavailable working copy and keeps the durable object available for a later retry

### Requirement: Confirmed deletion and unchanged custody

Deletion SHALL name the selected durable files and require explicit confirmation. Browsing or viewing a result SHALL NOT automatically copy temporary sandbox or Gmail files into durable storage. This experiment SHALL NOT add generated-file export or download retention policies.

#### Scenario: The user cancels deletion
- **WHEN** the confirmation is dismissed
- **THEN** no durable object or catalog entry is removed

### Requirement: Safe preview and explicit download

The app SHALL support bounded safe previews and explicit downloads of existing owned durable files. It SHALL NOT execute uploaded HTML or SVG, reveal storage credentials, or retain a second durable server copy for a preview or download.

#### Scenario: The user downloads a stored document
- **WHEN** an authenticated user selects download on an available owned object
- **THEN** its bytes are delivered through an authorized request with a safe filename and unchanged server retention
