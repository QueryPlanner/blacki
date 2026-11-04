# Proposal

## Why

Blacki already tracks meals, training, reminders, files, memories, and model preferences, but editing and inspecting these records through messages takes unnecessary turns. The user wants to try both Telegram-based conversation and conversation inside a Mini App, with useful interactive controls and a playful DaisyUI design, before choosing a preferred experience.

## What Changes

- Add one Mini App with two user-selectable experiences. **Telegram companion** keeps conversation in Telegram and opens focused screens from bot replies. **In-app assistant** includes conversation and interactive result cards inside the Mini App. Both expose the same feature screens and operate on the same owned records.
- Establish a shared, mobile-first DaisyUI design with rounded cards, tactile buttons, pill choices, sliders, steppers, date controls, and friendly illustrations. A kids-inspired toybox aesthetic is a requested visual direction; controls must still be legible and precise.
- Build a fixture-only comparison first, then connect the shared screens to authenticated Blacki services. Test the same meal, training, and reminder tasks in both experiences before choosing a default.
- Cover the full discussed scope: meals and health, training, reminders, durable files, model settings, memory, custom trackers, run inspection, and interactive tables/charts/forms.
- Add authenticated Mini App APIs and a shared conversation coordinator. Preserve existing ownership, confirmation, nutrition-sync, sandbox, and private integration boundaries.
- Keep implementation and deployment separate from this planning change. Production network exposure and any new artifact-export retention remain later decisions; they do not block fixture prototypes.

## Capabilities

### New Capabilities

- `miniapp-access`: Launch, authentication, identity mapping, API isolation, and safe record mutations.
- `miniapp-experience-comparison`: Switchable companion and in-app experiences, comparable tasks, and explicit user preference.
- `miniapp-design-system`: Playful DaisyUI theme, interactive controls, accessibility, and Telegram layout adaptation.
- `miniapp-meals-health`: Meal and nutrition editing, goal controls, health trends, and truthful sync states.
- `miniapp-training`: Program browsing, set logging, rest timer, and comparable training history.
- `miniapp-reminders`: Calendar, recurrence controls, editing, cancellation, and snoozing.
- `miniapp-files`: Owner-scoped catalog browsing, selection, sandbox restore, and confirmed deletion.
- `miniapp-model-settings`: Searchable model selection, favourites, and supported reasoning choices.
- `miniapp-memory`: Browsing, searching, correcting, and deleting owned memories.
- `miniapp-trackers`: Schema-backed forms and tables for custom records.
- `miniapp-run-inspection`: Content-safe user activity, live progress, and usage/cost inspection.
- `miniapp-assistant`: In-app conversation, shared session coordination, and validated interactive results.

### Modified Capabilities

None. The project currently has no canonical OpenSpec capability specs. Existing backend behavior is evidence for the design, not a claim that these new UI capabilities already exist.

## Impact

The future implementation touches `src/blacki/server.py`, Telegram transport/types and turn sequencing, `runtime/adk.py`, shared feature services/storage, and new Mini App assets and APIs. It reuses the pinned Tailwind CSS 4.1.18 and DaisyUI 5.5.5 build tooling. No new Node server, separate agent, database service, browser on the host, or external analytics service is required. Feature code, runtime settings, and dependencies are unchanged by this proposal.
