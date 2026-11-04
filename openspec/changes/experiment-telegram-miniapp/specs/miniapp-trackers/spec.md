# Spec delta

## Purpose

Turn Blacki's custom data tables into usable forms and record views while preserving validated, user-scoped operations.

## ADDED Requirements

### Requirement: Typed tracker forms

The app SHALL show owned trackers as searchable tables and typed record forms. Supported field metadata SHALL include labels, required state, numeric precision, date interpretation, and enum choices where applicable. Unsupported fields SHALL be identified rather than rendered as misleading editable controls.

#### Scenario: A tracker has a status field
- **WHEN** its metadata declares a set of allowed status values
- **THEN** the editor provides those choices and rejects other values

### Requirement: Scoped structured operations

Tracker edits SHALL use validated fields and owned query definitions. The UI SHALL NOT accept raw SQL, database paths, executable markup, or another user's table identity. Row updates SHALL target a stable identifier and confirm destructive actions.

#### Scenario: A tracker has no stable row identifier
- **WHEN** the user attempts a row edit
- **THEN** the app disables unsafe mutation and explains the missing identifier without issuing a broad update

### Requirement: Existing trackers remain usable

Existing tables SHALL remain queryable when optional UI metadata is absent. The app SHALL show safe inferred read-only fields and provide an explicit metadata configuration path without dropping or silently converting stored values.

#### Scenario: An existing tracker has an unsupported binary field
- **WHEN** the user opens it
- **THEN** supported columns remain visible and the binary field is labelled unsupported for editing
