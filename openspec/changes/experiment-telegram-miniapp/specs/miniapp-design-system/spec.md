# Spec delta

## Purpose

Give Blacki a playful, rounded visual language and accessible interactive controls that work comfortably inside Telegram on a phone.

## ADDED Requirements

### Requirement: Shared toybox appearance

Both experiences SHALL use the same kids-inspired theme with warm cream backgrounds, colourful rounded cards, pill-shaped choices, chunky buttons, and friendly illustrative details. Body text and important state labels SHALL remain readable; colour alone SHALL NOT communicate state.

#### Scenario: The user moves between conversation and training
- **WHEN** the user navigates between an assistant card and the training workspace
- **THEN** typography, colour roles, corner shapes, and button behaviour remain consistent

### Requirement: Controls match their values

The app SHALL use sliders for bounded approximate quantities, steppers for discrete counts, choice controls for enumerations, and text or numeric fields for exact edits. Every slider SHALL show its value and unit and provide an exact-value alternative. Changing a control SHALL update a draft until save or an explicitly labelled autosave commits it.

#### Scenario: The user adjusts a quantity
- **WHEN** a slider is moved or its associated exact field is edited
- **THEN** the displayed value and both controls agree within the stated precision
- **AND** dragging does not create one backend mutation per intermediate value

### Requirement: Touch and accessibility support

Primary touch targets SHALL be at least 44 by 44 CSS pixels. Controls SHALL have visible labels, keyboard operation, focus indication, and screen-reader state. Ordinary text SHALL meet WCAG AA contrast, and reduced-motion preferences SHALL disable decorative movement.

#### Scenario: A user cannot drag precisely
- **WHEN** the user operates a control with a keyboard or assistive technology
- **THEN** they can select and save the same value without dragging

### Requirement: Telegram viewport and theme adaptation

The layout SHALL respect Telegram safe areas, back navigation, keyboard visibility, and light or dark presentation. Loading, empty, error, disconnected, and disabled-integration states SHALL have explicit text and appropriate recovery controls.

#### Scenario: The keyboard opens on a small phone
- **WHEN** the user edits a field near the bottom of the viewport
- **THEN** the active field and save action remain reachable without overlapping Telegram controls
