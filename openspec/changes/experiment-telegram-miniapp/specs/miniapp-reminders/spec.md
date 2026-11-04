# Spec delta

## Purpose

Let users manage reminders through a calendar, recurrence choices, and quick actions while preserving correct ownership and schedule semantics.

## ADDED Requirements

### Requirement: Calendar and recurrence editing

The app SHALL provide an agenda or calendar, date/time pickers, weekday choices, and a human-readable preview of upcoming occurrences in the configured scheduling timezone. Dates SHALL retain their intended meaning across daylight-saving transitions.

#### Scenario: The user creates a repeating reminder
- **WHEN** the user chooses weekdays and a local delivery time
- **THEN** the preview shows the timezone and next occurrences before the reminder is saved

### Requirement: Owner-scoped edits and cancellation

The user SHALL be able to edit a reminder's instruction and schedule and cancel a selected reminder with confirmation. All schedule mutations SHALL verify ownership and update the active scheduler consistently.

#### Scenario: A forwarded link names another user's reminder
- **WHEN** a user attempts to edit or cancel that reminder
- **THEN** no schedule changes and the app returns an unavailable or unauthorized state

### Requirement: Snooze does not change recurrence

Snoozing SHALL defer a selected occurrence using quick duration choices or an exact time. It SHALL NOT silently replace a recurring reminder's underlying schedule. The app SHALL distinguish a deferred occurrence from the next regular occurrence.

#### Scenario: A recurring reminder is snoozed
- **WHEN** the user defers the current occurrence by fifteen minutes
- **THEN** the deferred delivery is shown separately and later regular occurrences retain their original schedule
