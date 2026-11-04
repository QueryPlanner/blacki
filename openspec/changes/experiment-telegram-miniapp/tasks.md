# Tasks

All tasks are implementation work for a later apply request. This planning change does not complete any task. Groups 1 and 2 produce the first usable comparison; later groups connect the same UI to live Blacki services.

## 1. Shared visual foundation

- [ ] 1.1 Add isolated Mini App assets and a disabled feature flag using the existing static build; verify the pinned build succeeds and dashboard assets/appearance remain unchanged.
- [ ] 1.2 Implement the cream-and-colourful `blacki-playroom` theme, rounded cards, tactile buttons, pill choices, and reduced-motion treatment; verify light/dark contrast, focus visibility, and 44px primary touch targets in a browser.
- [ ] 1.3 Implement shared labelled sliders with exact inputs, steppers, choices, dialogs, and draft/save state; verify keyboard edits and slider edits stay synchronized and dragging emits no live write requests.
- [ ] 1.4 Add deterministic fixture adapters and reset controls with explicit demo labels; verify all prototype actions remain isolated from production data and provider APIs, and document fixture startup/reset commands.

## 2. Both interactive prototypes

- [ ] 2.1 Implement the two-experience switch, common navigation, record deep links, and draft preservation; verify switching modes retains the selected record without creating a new session or duplicate record.
- [ ] 2.2 Build the fixture meal/health panels with calorie and goal sliders, exact macro fields, meal-type choices, and charts; verify the meal-correction trial in both experiences, including unknown nutrient and unset-goal states.
- [ ] 2.3 Build fixture training cards with set steppers, warm-up toggles, completed-set actions, and a rest slider/timer; verify three-set entry, correction, and suspension/deadline recovery in both experiences.
- [ ] 2.4 Build the fixture reminder calendar, date/time controls, weekday choices, recurrence preview, and snooze buttons; verify equivalent reschedule/snooze trials and visible timezone labels in both experiences.
- [ ] 2.5 Add representative file, model, memory, tracker, activity, and interactive-result panels plus labelled simulated assistant responses; verify each destination has usable controls and loading/empty/error states rather than a static placeholder.
- [ ] 2.6 Document and perform the four design.md comparison trials on phone and desktop, in both mode orders; record the user's actual feedback without inventing a preference or requiring a default before live work.

## 3. Live access and shared mutation boundary

- [ ] 3.1 Add verified Telegram launch-data authentication and short-lived in-memory bearer sessions; verify valid, forged, stale, future-dated, revoked-access, and expired-session scenarios with route tests and document configured lifetimes.
- [ ] 3.2 Add a server-owned identity resolver for private conversation records and raw-sender file ownership; verify cross-user/deep-link denial and existing-user data compatibility for every feature API.
- [ ] 3.3 Add the separate Mini App router, request bounds, CSP, and recovery states; verify only the allowlisted routes can be exposed and existing dashboard/ADK/sandbox boundaries remain protected.
- [ ] 3.4 Add owner-scoped operation receipts, expiry rejection, local record revisions, and resource coordination shared by tools and HTTP routes; verify duplicate requests, changed payloads, lost responses, stale edits, and uncertain provider outcomes.
- [ ] 3.5 Add Mini App button/menu transport support behind the feature flag; verify Telegram payload serialization, disabled-mode behavior, current command compatibility, and contextual record navigation.
- [ ] 3.6 Document additive migrations, API authentication, single-process assumptions, and rollback; verify a test database with existing user records upgrades without changing ownership or deleting data.

## 4. Live meals and health

- [ ] 4.1 Connect meal read/edit/delete and supported goal controls through shared services; verify API and tool edits update one owned record and refresh totals without inventing nutrient values.
- [ ] 4.2 Preserve the existing meal/export transaction with revision and operation checks; verify successful local save during provider failure, retries without duplicate meals, and current sync-state rendering.
- [ ] 4.3 Connect normalized health trends, source dates, and unavailable categories; verify missing data is not plotted as zero and document provider authorization and refresh behavior.

## 5. Live training

- [ ] 5.1 Extract an owner-scoped workout mutation service for both tools and routes; verify explicit session/exercise/set identities and incremental edits without duplicate appends or implicit cycle advancement.
- [ ] 5.2 Connect active-day, set-entry, partial/completed status, and comparison panels; verify cross-channel stale edits, warm-up state, and comparison against the correct previous session.
- [ ] 5.3 Verify rest presets, exact duration, and elapsed-time recovery using a controlled clock and a suspended browser; document that a closed mobile WebView cannot promise an alarm.

## 6. Live reminders

- [ ] 6.1 Add owner-scoped schedule/instruction updates that reconcile storage and the scheduler; verify foreign IDs, concurrent edits, and failed scheduler updates cannot silently change the wrong reminder.
- [ ] 6.2 Implement occurrence-aware snooze records or overrides; verify snoozing one occurrence preserves the base recurrence, does not duplicate due dispatch, and shows the deferred and next regular times distinctly.
- [ ] 6.3 Connect agenda/calendar, recurring previews, and confirmed cancellation; verify one-shot and recurring schedules across normal days, clock changes, and the configured timezone, and document these semantics.

## 7. Live files, model settings, and memory

- [ ] 7.1 Connect the R2 catalog, search/filter/selection controls, and canonical-session restore; verify raw-sender ownership, expired objects, sandbox unavailability, and partial multi-file restore without source loss.
- [ ] 7.2 Add bounded safe preview and explicit existing-file download, plus exact-target deletion confirmation; verify MIME/filename handling, executable-content rejection, authorization, and unchanged retention, and document file custody.
- [ ] 7.3 Connect model discovery, favourites, exact-ID fallback, and supported reasoning choices; verify model changes reset incompatible reasoning and active turns retain their original profile, and document next-turn behavior.
- [ ] 7.4 Extract shared memory operations and connect browse/search/edit/delete; verify owner checks, stale snapshots, missing memories, provider-unavailable states, and bounded pagination, and document external-writer limitations.

## 8. Custom trackers and structured result controls

- [ ] 8.1 Add additive tracker UI metadata and stable-row identification; verify existing tables still query correctly, unsupported types are read-only, and metadata changes do not silently convert stored values.
- [ ] 8.2 Connect typed forms, choices, filters, sorting, and confirmed deletion through validated owned query operations; verify wrong-owner tables, missing row keys, arbitrary SQL, and invalid values are rejected, and document supported fields.
- [ ] 8.3 Define and validate the versioned editor/choice/quantity/table/chart envelope; verify limits, record ownership, escaping, and a safe text fallback for invalid or executable content.
- [ ] 8.4 Render the same result components as contextual Mini App panels and in-app assistant cards; verify chart filters, expired snapshots, and state-changing actions use shared APIs, and document temporary result lifetime without adding artifact exports.

## 9. Shared conversation and run inspection

- [ ] 9.1 Extract the existing turn sequencing into one coordinator used by Telegram and app chat; verify cross-channel supersession waits for cleanup, direct form saves do not cancel turns, and session reset/version behavior remains consistent.
- [ ] 9.2 Add owned conversation history, app-origin turns, authorized fetch streaming, and origin labels; verify continuity with Telegram history, no duplicate turns, and exclusion of internal reasoning/prompts/private tool payloads.
- [ ] 9.3 Add reconnectable run metadata and interrupted-state recovery; verify closing the app leaves work running, reconnect reuses the run, and restart never silently replays external effects.
- [ ] 9.4 Add safe activity, duration, and usage/cost projections from the run registry and existing ledger; verify cross-user isolation and distinct reported/estimated/unavailable cost states without raw operator-log access.
- [ ] 9.5 Add explicit completion-notification preference and delivery routing; verify app replies do not automatically duplicate into Telegram and notifications require opt-in, and document session/delivery behavior.

## 10. Integrated trials and rollout readiness

- [ ] 10.1 Exercise both experiences end to end with concurrent Telegram and Mini App actions, reconnects, provider failures, stale links, and repeated saves; verify all capability scenarios with real internal services and mocks only at external boundaries.
- [ ] 10.2 Run browser interaction checks on a developer machine or isolated sandbox for keyboard/touch controls, Telegram viewport/keyboard changes, reduced motion, and light/dark themes; document the results without adding a browser to the production Blacki host.
- [ ] 10.3 Configure a controlled live HTTPS trial only after the hostname and public-versus-tailnet deployment choice are supplied; verify route isolation, bot launch controls, and feature-flag rollback before exposing live data.
- [ ] 10.4 Run the same comparison tasks with live owned records, capture the user's preference or decision to keep both, and update the default only on explicit choice; verify switching and existing Telegram workflows remain available.
- [ ] 10.5 Complete OpenSpec verification and the repository's format/lint/type/coverage checks after implementation; verify the built behavior matches these specs and the documentation contains the tested trial and rollback instructions.

## Workflow follow-up

- Keep this change active during implementation; planning completion is not implementation completion.
- Archive and sync the completed capabilities only after implementation verification and the user's review.
