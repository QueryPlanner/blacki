# Spec delta

## Purpose

Make Blacki's existing model and reasoning preferences easier to select without losing exact model identifiers or misrepresenting provider capabilities.

## ADDED Requirements

### Requirement: Searchable exact model selection

The app SHALL show the active model, support searchable model choices when metadata is available, accept exact identifiers, and offer user-managed favourites. It SHALL distinguish a configured selection from provider availability verified on an actual request.

#### Scenario: Model discovery is temporarily unavailable
- **WHEN** the model catalog cannot load
- **THEN** the current selection remains visible and exact-ID entry remains available

### Requirement: Supported discrete reasoning choices

Reasoning controls SHALL show only the selected model's supported discrete choices and an inherited default. A model change SHALL reset incompatible explicit reasoning choices. The app SHALL NOT represent reasoning as an invented continuous percentage.

#### Scenario: A user changes models
- **WHEN** the new model does not support the previous reasoning choice
- **THEN** the preference returns to the inherited default and the displayed choices update

### Requirement: Turn-stable preferences

Saved model preferences SHALL apply to subsequent agent turns in both experiences. An active turn SHALL retain its original preference snapshot. Reset controls SHALL describe exactly which preferences they clear.

#### Scenario: The model changes during a response
- **WHEN** the user saves a new model while an agent turn is active
- **THEN** that response continues with its original model and the next turn uses the new selection
