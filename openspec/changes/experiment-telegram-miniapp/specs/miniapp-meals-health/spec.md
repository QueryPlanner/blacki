# Spec delta

## Purpose

Make meals, nutrition goals, and available health trends easy to inspect and correct through direct controls shared by both Mini App experiences.

## ADDED Requirements

### Requirement: Editable meal cards and numeric controls

The app SHALL show daily totals, meal cards, and date navigation. Meal editors SHALL expose description, date, meal type, calories, and available macros, with a calorie slider and exact numeric fields. Unknown nutrient values SHALL remain unknown until explicitly supplied.

#### Scenario: A user corrects an estimated meal
- **WHEN** the user opens a meal from chat or the daily screen, adjusts calories, and saves
- **THEN** one existing meal is updated and the daily totals refresh
- **AND** unspecified macros are not guessed or silently scaled

### Requirement: Visible user-controlled goals

The app SHALL let the user inspect and change supported nutrition goals through a labelled slider and exact input, starting from existing saved values. It SHALL distinguish logged values from goals and SHALL NOT invent a personalised goal when no saved value exists.

#### Scenario: The goal is not configured
- **WHEN** the user opens daily nutrition without an existing goal
- **THEN** the app shows an unset state with an explicit setup control

### Requirement: Truthful local and remote states

Meal writes SHALL preserve existing nutrition-export behavior. The app SHALL distinguish local save success from pending, failed, or completed remote sync. Health charts SHALL show source freshness and unavailable categories without treating missing values as zero.

#### Scenario: Google Health export is unavailable
- **WHEN** a meal saves locally while export fails
- **THEN** the meal remains saved once and the sync state is visible without asking the user to log it again

#### Scenario: A health category has not been imported
- **WHEN** the user selects a sleep or activity chart with no records
- **THEN** the chart shows unavailable data rather than fabricated measurements
