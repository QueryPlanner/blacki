# Spec delta

## Purpose

Support a real gym session with direct set entry, an optional rest timer, and comparisons to the user's existing training history.

## ADDED Requirements

### Requirement: Program and comparable-history views

The app SHALL show the active training day, prescribed exercises, and previous comparable sessions. Exercise and date filters SHALL update charts and tables without requiring a conversation turn.

#### Scenario: The user starts today's workout
- **WHEN** the user opens a training day with previous comparable records
- **THEN** the exercise cards show previous weights and reps alongside new set controls

### Requirement: Incremental set editing

The user SHALL be able to enter weight and reps using steppers and exact fields, mark warm-up sets, edit or remove a set, and record partial or completed workouts. Saving a set SHALL update its intended workout without appending duplicate exercises or advancing the training cycle implicitly.

#### Scenario: The user corrects the second set
- **WHEN** the user edits and resaves an already logged set
- **THEN** that set changes once while other sets and the current cycle position remain unchanged

### Requirement: Honest rest-timer behavior

The app SHALL offer rest presets and an adjustable timer with a visible remaining duration. Returning after suspension SHALL recalculate from the deadline. The app SHALL NOT promise an alarm while Telegram is closed or the operating system suspends execution.

#### Scenario: The user briefly leaves Telegram
- **WHEN** the user returns after the timer deadline
- **THEN** the timer shows that rest has finished without restarting or extending it
