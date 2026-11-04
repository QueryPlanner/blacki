# Design

## Context

See [proposal.md](proposal.md) for the requested scope. The code inspected is based on main commit `9e94275`, with the OpenSpec setup available in the working branch. There are no existing canonical capability specs.

| Existing code | Consequence for this design |
| --- | --- |
| `telegram/bot.py` maps private chats to `telegram-chat-{id}` and owns cancellation/turn locks | Both conversation entry points need one coordinator and canonical identity. |
| `telegram/types.py` has URL and callback buttons but no Mini App button model | Launch and menu-button support is an explicit transport addition. |
| `calories/service.py` changes meals and nutrition-export revisions together | Both UI variants must reuse this service. Direct SQL would break sync behavior. |
| `workouts/storage.py` supports exercise edits; `tools/workouts.py` can append today's exercises | Incremental set saves need a shared workout mutation service and explicit identities. |
| `reminders/storage.py` reschedules by reminder ID without an owner parameter | Web-facing edits need a new owner-scoped service, not direct exposure of that scheduler method. |
| R2 file ownership uses the raw Telegram sender; most domain records use conversation identities | A server-owned identity resolver must choose each feature's key. |
| The dashboard has all-user observability and relies on loopback/Tailscale access | It stays an operator tool; the Mini App gets separate routes and projections. |
| `package.json` pins DaisyUI 5.5.5 and Tailwind CSS 4.1.18 | Reuse the existing static-asset build. Current online component docs must be checked against the pinned package. |

## Goals / Non-Goals

**Goals:** Make the two experiences directly comparable; provide working controls in both; connect one set of UI components to the existing services; keep a small server footprint; protect existing record and integration semantics.

**Non-Goals:** This change does not choose the user's preferred experience in advance, replace the operator dashboard, create a second agent, permit arbitrary model-authored JavaScript, add payments, or publish the app. Generated-file exports, automatic sandbox-to-R2 persistence, background mobile alarms, and redesigning browser takeover are outside the experiment.

## Decisions

### 1. One app, two conversation placements

Use an explicit experience switch with the labels **Telegram companion** and **In-app assistant**. They share route panels, controls, fixture definitions, backend adapters, and records. The switch is a presentation preference, not a new account or session identifier.

- Companion opens a home screen from the bot menu. Contextual bot buttons open an exact editor or result panel. Conversation continues in the existing Telegram chat.
- In-app assistant adds an Assistant destination with streamed replies and interactive cards. The same editors can expand from a reply or open as a focused panel.
- Both retain Home, Training, Reminders, Files, and More navigation. More contains model settings, memory, trackers, and activity. Assistant is prominent only in the in-app experience.
- Use explicit navigation and visible mode labels. Preserve drafts when switching, with save/discard choices. A deep link selects a route or record but never grants access.

Two independent apps were rejected because feature and design differences would bias the comparison and double maintenance. A randomized A/B rollout was rejected because this is a personal preference experiment.

### 2. Interactive fixture prototype comes first

The first implementation milestone is a usable visual comparison with deterministic fixtures, not live accounts. Both modes must expose working meal, training, and reminder controls, and the other modules must have navigable representative screens before connecting providers.

The fixture adapter uses its own resettable state and clearly labels every screen and simulated assistant reply. It cannot call real API adapters. Live mode requires verified Telegram launch data and never falls back to fixtures after an authentication error.

Use the same trial checklist in each mode:

| Trial | User action | What to compare |
| --- | --- | --- |
| Correct a meal | Open the same lunch, drag calories, enter an exact value, save | Ability to find the record, precision, taps, clarity of save state |
| Log a workout | Enter three sets, correct one, run a short rest timer | Reachability of controls, interruptions, accidental duplicates |
| Change a reminder | Pick another date, inspect recurrence, snooze one occurrence | Schedule understanding and confidence in the result |
| Explore a result | Filter a chart and open a referenced record | Whether conversation placement helps or obscures the controls |

Record the user's comments manually. Persist only an explicitly selected experience preference. No external analytics or inferred preference is needed. The user can retain both modes after the trial.

### 3. A shared DaisyUI toybox theme

Create an app-local `blacki-playroom` theme. Preserve the operator dashboard's styling. Start with the following proposed tokens, then adjust contrast and appearance during visual review:

| Token | Proposed value or treatment |
| --- | --- |
| Page background | Warm cream `#FFF8EC` |
| Main text | Dark navy `#253047` |
| Primary action | Deep purple `#7041D9`, light label |
| Supporting card colours | Lavender `#E9DEFF`, mint `#DDF4E5`, peach `#FFE1CC`, yellow `#FFF0B5` |
| Card shape | About 28px corners, generous spacing, subtle dark outline |
| Inputs | About 18px corners and an obvious focus ring |
| Buttons | Pill or rounded rectangle, substantial height, small pressed-state movement |
| Typography | Rounded system font stack with readable system fallbacks; no required remote font service |
| Illustration | Small friendly shapes or category drawings, always secondary to labels |

Dark presentation keeps the colourful accents but uses dark backgrounds and tested text pairs. Supporting colours are not automatically valid action backgrounds; contrast checks determine the actual foreground colour. Provide reduced-motion styles instead of making animation essential.

Build reusable control functions and panels using the existing static JavaScript approach and Tailwind/DaisyUI build. A framework migration is unnecessary for this experiment. Component styling must work with the pinned DaisyUI version; do not silently upgrade it to match newer documentation.

| Feature | Concrete controls |
| --- | --- |
| Home | Large action tiles, nutrition progress, today's training card, upcoming-reminder cards |
| Meals and health | Meal cards, date chips, calorie range plus exact input, macro fields, meal-type pills, goal controls, range-filtered charts |
| Training | Exercise cards, weight/reps steppers, warm-up toggle, completed-set buttons, rest presets, adjustable rest slider and exact seconds |
| Reminders | Calendar/agenda switch, date/time controls, weekday chips, recurrence preview, quick snooze buttons, confirmation dialog |
| Files | Search field, type chips, grid/list switch, selection boxes, restore action, safe preview/download controls, delete confirmation |
| Models | Searchable choices, favourite buttons, exact-ID input, supported reasoning radio buttons or segmented choices |
| Memory | Searchable cards, text editor, exact-target delete dialog |
| Trackers | Typed fields, enum chips, numeric steppers, table filters and sort controls |
| Activity | Status timeline, expandable safe tool summaries, duration and cost cards |
| Assistant results | Choice buttons, bounded sliders, shared editors, sortable tables, chart filters, explicit action buttons |

A slider does not save each drag event. It changes a visible draft and synchronizes with an exact input. Enumerations such as reasoning effort use discrete controls. Unknown health or nutrient values stay unknown. Forms use explicit Save initially; any later autosave must be visibly labelled and acknowledge committed state.

### 4. Separate authenticated HTTP boundary

Serve `/miniapp` static assets and `/api/miniapp/*` from the existing FastAPI application. Add a Mini App router rather than relaxing dashboard protections or exposing the public ADK runner.

Verify Telegram `initData` server-side, including signature, timestamp age, and Blacki access policy. Resolve the signed Telegram sender to the canonical private conversation and the existing per-feature owner keys. Start with private-chat data; reject ambiguous group or topic targeting.

Issue a short-lived opaque bearer session held in JavaScript memory and use authorized `fetch` for data and streaming. Do not put tokens in URLs or persist them in browser storage. This avoids relying on third-party cookies inside Telegram Web. Expired sessions require a fresh verified launch. Recheck revoked access before mutations. The app must never receive provider tokens or sandbox routing headers.

Use a CSP suitable for the Telegram bridge and locally built assets, with no arbitrary inline execution or remote model-authored code. Keep authorization errors recoverable without echoing private payloads. Add bounded input, pagination, and per-user request limits.

The implementation defaults to a disabled feature flag and loopback serving. Live Telegram trials require a configured HTTPS address reachable by the client. Public HTTPS versus tailnet-only access is a deployment setting decided before live exposure; the same auth and route isolation are required either way.

### 5. Shared services and reliable mutations

Feature routes call owner-aware services; tools become adapters to those same services. Avoid constructing synthetic ADK tool contexts or sending ordinary form edits through an LLM.

- Reuse `MealService.mutate()` with verified private identity so local meals and remote-export revisions remain one transaction. Add owner/revision checks without separating the existing export transaction.
- Introduce shared training and reminder mutation services. Workout set identities prevent whole-workout append behavior on every save. Reminder edits update storage and scheduler together. Snooze uses an occurrence override/follow-up record; it does not replace a recurring rule.
- Use the existing inference profile and capability resolver for model settings. Snapshot preferences at turn start.
- Extract memory and tracker adapters with the existing ownership checks. External memory edits are serialized within Blacki and checked against the last fetched value; the provider may not offer atomic compare-and-swap against independent external writers.
- Add revision checks for local records and resource-scoped coordination for provider operations. A stale editor keeps its draft and offers reload/reapply rather than silent overwrite.
- Issue owner-scoped operation IDs with a defined validity window before live writes. Keep mutation receipts and original outcomes until expiry; reject expired operations instead of treating missing receipts as new. Commit local mutations and receipts together. An uncertain external outcome must be shown and reconciled before retrying a non-idempotent action.

These checks must cover both tools and HTTP routes. UI-level disabling alone cannot protect against concurrent chat updates or repeated requests.

### 6. Shared conversation and explicit delivery

Move the existing per-conversation sequencing into one coordinator used by Telegram and Mini App chat. Preserve the current cancel-and-replace behavior for newer agent messages, including cancellation cleanup before the replacement turn starts. A direct record save, view switch, reconnect, or closed WebView is not a new agent message.

Use the existing private ADK runner, canonical session locator, and version/reset behavior. Show only user messages, public assistant replies, and safe progress, with origin labels. Never expose internal reasoning, prompts, or raw private tool payloads as transcript.

Create a small owner-scoped run registry for reconnectable status. Stream through authorized fetch, retaining the run ID and a bounded progress cursor. Closing the UI unsubscribes without cancelling the server task. A process restart marks unfinished runs interrupted; it does not silently resume external effects. Keep the existing single-process/scheduler deployment assumption explicit.

Telegram-origin turns retain Telegram delivery. App-origin turns are visible in the shared conversation and app, without automatic duplicate Telegram replies. A separate explicit completion-notification preference can opt into a bot notification. Do not replay historical Telegram messages when the user opens the app.

### 7. Fixed interactive-result vocabulary

Define a versioned result envelope containing a panel type, title, bounded data or owned-record references, and allowlisted actions. Initial types are `record-editor`, `choice`, `quantity`, `table`, and `chart`. Validate it on the server and render it with shared components. Invalid payloads fall back to escaped text.

Actions refer to server-issued resources and are reauthorized at execution time. The model does not author endpoint URLs, SQL, HTML, or JavaScript. Companion mode sends a Mini App link; in-app mode presents the same panel within or alongside the reply. Charts use current owned aggregates and visible source ranges.

Dynamic result snapshots are bounded and temporary. If data cannot be reconstructed after expiry or a restart, show an unavailable state and offer explicit regeneration. Viewing a panel creates no durable artifact file. Exporting generated files is a separate retention decision.

### 8. File custody and integration failure

The file browser lists the existing sender-owned R2 catalog and restores selected files into the canonical conversation sandbox. Safe image/text previews and explicit downloads are read-only deliveries of an existing durable object. Use authenticated fetch and temporary client buffers, bounded content types/sizes, escaped filenames, and no executable SVG/HTML previews. Do not add a second durable server copy or expose bucket credentials.

Deleting durable data still requires exact-target confirmation. A sandbox failure leaves R2 data intact; a partial multi-file restore shows which copies succeeded. Temporary sandbox files and Gmail attachments are not automatically promoted into the catalog. Existing private Gmail/Health OAuth remains external consent; Telegram identity does not replace provider authorization.

## Risks / Trade-offs

- A playful theme can reduce legibility. Use restrained decorative areas, tested text pairs, exact inputs, and visible focus states.
- Two entry points can race on the same records. Centralize mutations and sequencing and test cross-channel conflicts.
- Progress history can leak private inputs. Store content-free run metadata and expose scoped projections, not raw operator logs.
- Provider writes can complete after a connection fails. Expose uncertain outcomes and reconcile rather than blindly replaying.
- Static JavaScript can become hard to maintain as screens grow. Keep domain panels and controls modular; revisit a framework only after the trial reveals a concrete need.
- A stopped mobile WebView cannot reliably run alarms or maintain streams. Recalculate timers and reconnect from server state on return.

## Migration Plan

1. Ship the fixture comparison behind the disabled Mini App flag and complete both-mode UI trials.
2. Add the authenticated router, shared identity/mutation services, and additive SQLite migrations. Preserve existing records and owner keys.
3. Connect meals, training, reminders, then the remaining feature panels. Add conversation coordination and validated result rendering before live in-app chat.
4. Configure a reachable HTTPS route and bot launch controls only for a controlled live trial. Confirm public or tailnet exposure before changing reverse-proxy configuration.
5. Run the same task checklist in both modes. Keep both available until the user explicitly chooses a default.
6. Roll back by disabling Mini App routes and bot launch controls. Existing Telegram conversation and data remain usable; do not drop additive tables or domain records during rollback.

## Open Questions

- Which exact colour balance, illustrations, and button depth the user prefers can be settled using the working visual trial.
- Whether the user eventually keeps one default or both experiences is the experiment's outcome, not a prerequisite for implementation.
- The live hostname and network exposure are deployment inputs. They must be supplied before live launch; they do not change the component or authentication design.

## References

- Telegram Mini Apps: https://core.telegram.org/bots/webapps
- DaisyUI buttons: https://daisyui.com/components/button/
- DaisyUI range inputs: https://daisyui.com/components/range/
- DaisyUI themes: https://daisyui.com/docs/themes/
- Existing feature boundaries: `AGENTS.md`, `docs/architecture.md`, `docs/telegram-setup.md`, and `docs/browser-takeover.md`.
