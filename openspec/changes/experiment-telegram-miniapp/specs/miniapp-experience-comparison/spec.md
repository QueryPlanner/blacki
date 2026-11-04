# Spec delta

## Purpose

Let the user compare Telegram conversation with in-app conversation while keeping useful feature controls and data consistent across both experiences.

## ADDED Requirements

### Requirement: Two selectable experiences

The app SHALL offer Telegram companion and In-app assistant experiences. Both SHALL provide the same feature screens, theme, and data; only In-app assistant SHALL include an in-app conversation composer. Switching SHALL preserve the selected record and handle unsaved edits explicitly.

#### Scenario: The user switches during a meal edit
- **WHEN** the user switches experiences with unsaved meal changes
- **THEN** the app offers to keep the draft, save it, or discard it before navigation
- **AND** switching never creates another meal or agent conversation

### Requirement: Equivalent comparison tasks

The prototype SHALL provide resettable tasks to correct a meal, log workout sets, and reschedule a reminder in either experience. Both SHALL have working interactive controls; a static mockup or plain conversation screen SHALL NOT count as a completed experiment.

#### Scenario: The user repeats a comparison task
- **WHEN** the user resets a task and opens the other experience
- **THEN** the starting fixture records and available feature controls match the first run
- **AND** simulated assistant responses are labelled as demo responses

### Requirement: Explicit preference without forced selection

The user SHALL be able to keep both experiences, choose a default, or defer the choice. The app SHALL NOT randomly assign an experience or send experiment analytics to an external service.

#### Scenario: The user has no preference yet
- **WHEN** the user finishes a trial without choosing a default
- **THEN** both experiences remain available and no preference is inferred from silence
