import { FixtureAdapter, localDateValue } from "./fixtures.js";
import { button, escapeHtml } from "./controls.js";
import { comparisonDialog, formatView } from "./views.js";

const navItems = [
  { id: "home", label: "Home", glyph: "⌂", colour: "butter" },
  { id: "meals", label: "Meals", glyph: "◒", colour: "peach" },
  { id: "training", label: "Training", glyph: "↗", colour: "mint" },
  { id: "reminders", label: "Reminders", glyph: "◷", colour: "sky" },
  { id: "files", label: "Files", glyph: "▤", colour: "lavender" },
  { id: "more", label: "More", glyph: "•••", colour: "butter" },
  { id: "assistant", label: "Assistant", glyph: "✳", colour: "mint" },
];
const allowedScreens = new Set(navItems.map((item) => item.id).concat(["models", "memory", "trackers", "activity"]));
const fixtures = new FixtureAdapter();
const params = new URLSearchParams(window.location.search);

const state = {
  screen: allowedScreens.has(params.get("screen")) ? params.get("screen") : "home",
  selectedRecord: params.get("record") || null,
  mode: "companion",
  preference: "undecided",
  notifyOnCompletion: false,
  theme: window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "blacki-playroom-dark" : "blacki-playroom",
  drafts: {},
  pendingAction: null,
  fileSearch: "",
  modelSearch: "",
  memorySearch: "",
  trackerSearch: "",
  trackerSort: "asc",
  selectedTracker: null,
};

const main = document.getElementById("main-content");
const desktopNav = document.getElementById("desktop-nav");
const mobileNav = document.getElementById("mobile-nav");
const decisionDialog = document.getElementById("decision-dialog");
const actionDialog = document.getElementById("action-dialog");
const toastRegion = document.getElementById("toast-region");
let toastTimer;
let timerInterval;

function isDirty() { return Object.keys(state.drafts).length > 0; }

function navMarkup(item) {
  const currentScreen = ["models", "memory", "trackers", "activity"].includes(state.screen) ? "more" : state.screen;
  return `<a class="nav-link" href="#${item.id}" data-action="navigate" data-screen="${item.id}" aria-current="${currentScreen === item.id ? "page" : "false"}" data-colour="${item.colour}"><span class="nav-glyph">${item.glyph}</span><span>${escapeHtml(item.label)}</span></a>`;
}

function renderNavigation() {
  const items = state.mode === "assistant" ? navItems : navItems.filter((item) => item.id !== "assistant");
  desktopNav.innerHTML = items.map((item) => navMarkup(item)).join("");
  mobileNav.innerHTML = items.map((item) => navMarkup(item, true)).join("");
  document.querySelectorAll("[data-mode-choice]").forEach((control) => {
    control.setAttribute("aria-pressed", control.dataset.modeChoice === state.mode ? "true" : "false");
  });
  document.documentElement.dataset.theme = state.theme;
}

function updateLocation() {
  const url = new URL(window.location.href);
  if (state.screen === "home") url.searchParams.delete("screen");
  else url.searchParams.set("screen", state.screen);
  if (state.selectedRecord) url.searchParams.set("record", state.selectedRecord);
  else url.searchParams.delete("record");
  window.history.replaceState({}, "", url);
}

function render({ focusMain = false } = {}) {
  renderNavigation();
  main.innerHTML = formatView(state, fixtures.data);
  updateLocation();
  if (focusMain) main.focus({ preventScroll: true });
  updateTimerDisplay();
  updateTelegramBackButton();
}

function navigate(screen, { record = null, update = true } = {}) {
  if (!allowedScreens.has(screen)) return;
  if (screen === "assistant" && state.mode !== "assistant") return;
  state.screen = screen;
  if (record) state.selectedRecord = record;
  if (update) render({ focusMain: true });
}

function showToast(message) {
  toastRegion.innerHTML = `<div class="toast-message">${escapeHtml(message)}</div>`;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { toastRegion.replaceChildren(); }, 3200);
}

function showDialog(dialog, markup) {
  dialog.innerHTML = markup;
  if (!dialog.open) dialog.showModal();
}

function closeDialog(dialog) {
  if (dialog.open) dialog.close();
}

function compareDialog() {
  showDialog(decisionDialog, comparisonDialog(state));
}

function requestMode(mode) {
  if (mode === state.mode) return;
  captureDrafts();
  if (!isDirty()) { setMode(mode); return; }
  const nextMode = mode;
  showDialog(decisionDialog, `<div class="dialog-inner"><div class="heading-kicker">UNSAVED DEMO CHANGES</div><h2 id="decision-title">Keep this little draft?</h2><p>Switching modes will keep the same demo records. Choose what to do with your unsaved edits.</p><div class="dialog-actions dialog-stack"><button class="play-button soft" type="button" data-action="mode-keep" data-mode="${nextMode}">Keep draft and switch</button><button class="play-button" type="button" data-action="mode-save" data-mode="${nextMode}">Save and switch</button><button class="play-button danger" type="button" data-action="mode-discard" data-mode="${nextMode}">Discard and switch</button></div></div>`);
}

function setMode(mode) {
  state.mode = mode;
  if (mode === "companion" && state.screen === "assistant") state.screen = "home";
  closeDialog(decisionDialog);
  render({ focusMain: true });
}

function captureMealDraft(form = document.getElementById("meal-form")) {
  if (!form) return;
  const id = form.dataset.mealId;
  const existing = fixtures.meal(id);
  const value = (name) => form.querySelector(`[name="${name}"]`)?.value ?? "";
  const exactCalories = document.querySelector('[data-exact="meal-calories"]')?.value;
  const previous = state.drafts.meal;
  const numberOrExisting = (name, key) => {
    const raw = value(name).trim();
    if (raw === "") return existing?.[key] ?? null;
    const parsed = Number(raw);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : existing?.[key] ?? null;
  };
  state.drafts.meal = {
    id,
    isNew: previous?.isNew ?? false,
    title: value("title"),
    type: value("type"),
    date: value("date") || existing?.date || localDateValue(),
    calories: exactCalories?.trim() ? Number(exactCalories) : Number(existing?.calories ?? 0),
    protein: numberOrExisting("protein", "protein"),
    carbs: numberOrExisting("carbs", "carbs"),
    fat: numberOrExisting("fat", "fat"),
  };
}

function captureReminderDraft() {
  const form = document.getElementById("reminder-form");
  if (!form) return;
  state.drafts.reminder = {
    ...fixtures.data.reminder,
    title: form.elements.title.value,
    date: form.elements.date.value,
    time: form.elements.time.value,
    repeat: [...document.querySelectorAll(".weekday-chip[aria-pressed=true]")].map((button) => Number(button.dataset.day)),
  };
}

function captureSetDraft(row) {
  if (!row) return;
  const setId = row.dataset.setId;
  const existing = findSet(setId);
  if (!existing) return;
  const draft = state.drafts.sets?.[setId] ?? { ...existing.set, exerciseId: row.dataset.exerciseId };
  draft.exerciseId = row.dataset.exerciseId;
  for (const field of ["weight", "reps"]) {
    const input = row.querySelector(`[data-set-field="${field}"]`);
    if (input && Number.isFinite(Number(input.value))) draft[field] = Number(input.value);
  }
  const warmup = row.querySelector('[data-set-field="warmup"]');
  if (warmup) draft.warmup = warmup.checked;
  state.drafts.sets ??= {};
  state.drafts.sets[setId] = draft;
}

function captureDrafts() {
  if (state.drafts.meal) captureMealDraft();
  if (state.drafts.reminder) captureReminderDraft();
  if (state.drafts.sets) document.querySelectorAll(".set-row").forEach((row) => {
    if (state.drafts.sets[row.dataset.setId]) captureSetDraft(row);
  });
  const goal = document.querySelector('[data-exact="calorie-goal"]');
  if (goal && state.drafts.goal != null) state.drafts.goal = Number(goal.value);
  const modelId = document.getElementById("exact-model-id");
  if (modelId && state.drafts.exactModelId != null) {
    state.drafts.exactModelId = modelId.value.trim();
    state.drafts.model = modelId.value.trim();
  }
  const fileSearch = document.getElementById("file-search");
  if (fileSearch) state.fileSearch = fileSearch.value;
  const modelSearch = document.getElementById("model-search");
  if (modelSearch) state.modelSearch = modelSearch.value;
  const memorySearch = document.getElementById("memory-search");
  if (memorySearch) state.memorySearch = memorySearch.value;
  const trackerSearch = document.getElementById("tracker-search");
  if (trackerSearch) state.trackerSearch = trackerSearch.value;
}

function commitDrafts() {
  const drafts = state.drafts;
  if (drafts.meal) {
    const { id, isNew, ...meal } = drafts.meal;
    if (isNew) fixtures.data.meals.unshift({ id, ...meal, time: "Just now" });
    else fixtures.updateMeal(id, meal);
  }
  if (drafts.goal != null) {
    fixtures.data.calorieGoal = Number(drafts.goal);
    fixtures.data.goalConfigured = true;
  }
  if (drafts.sets) {
    for (const [setId, patch] of Object.entries(drafts.sets)) {
      const { exerciseId } = patch;
      const { exercise, set } = findSet(setId, exerciseId);
      if (set && exercise) {
        const { exerciseId: _exerciseId, ...setPatch } = patch;
        fixtures.updateSet(exercise.id, setId, setPatch);
      }
    }
  }
  if (drafts.reminder) fixtures.updateReminder(drafts.reminder);
  if (drafts.restSeconds != null) fixtures.data.restSeconds = Number(drafts.restSeconds);
  if (drafts.assistantCalories != null) fixtures.updateMeal("meal-lunch", { calories: Number(drafts.assistantCalories) });
  if (drafts.model) {
    fixtures.data.activeModel = drafts.model;
    if (!fixtures.data.models.find((item) => item.id === drafts.model)) {
      fixtures.data.models.push({ id: drafts.model, label: drafts.model, provider: "Custom ID", reasoning: ["default"], favorite: false });
    }
  }
  const activeModel = fixtures.data.models.find((item) => item.id === fixtures.data.activeModel);
  const requestedReasoning = drafts.reasoning ?? fixtures.data.reasoning;
  fixtures.data.reasoning = activeModel?.reasoning.includes(requestedReasoning) ? requestedReasoning : "default";
  state.drafts = {};
}

function findSet(setId, knownExerciseId = null) {
  for (const workout of fixtures.data.workouts) {
    for (const exercise of workout.exercises) {
      if (knownExerciseId && exercise.id !== knownExerciseId) continue;
      const set = exercise.sets.find((item) => item.id === setId);
      if (set) return { exercise, set };
    }
  }
  return { exercise: null, set: null };
}

function updateRange(target) {
  const rangeId = target.dataset.range || target.dataset.exact;
  if (!rangeId) return;
  const range = document.querySelector(`[data-range="${rangeId}"]`);
  const exact = document.querySelector(`[data-exact="${rangeId}"]`);
  if (!range || !exact) return;
  const sourceValue = Number(target.value);
  if (!Number.isFinite(sourceValue)) return;
  const min = Number(range.min);
  const max = Number(range.max);
  const clamped = Math.min(Math.max(sourceValue, min), max);
  const step = Number(range.step) || 1;
  const snapped = Math.round((clamped - min) / step) * step + min;
  range.value = String(snapped);
  exact.value = String(snapped);
  if (rangeId === "meal-calories") captureMealDraft();
  if (rangeId === "calorie-goal") state.drafts.goal = snapped;
  if (rangeId === "assistant-calories") state.drafts.assistantCalories = snapped;
  if (rangeId === "rest-seconds") state.drafts.restSeconds = snapped;
}

function updateSearch(target) {
  const query = target.value.trim().toLowerCase();
  if (target.id === "file-search") {
    state.fileSearch = target.value;
    document.querySelectorAll(".file-row").forEach((row) => { row.hidden = !row.dataset.fileName.includes(query); });
  } else if (target.id === "model-search") {
    state.modelSearch = target.value;
    document.querySelectorAll(".model-row").forEach((row) => { row.hidden = !row.dataset.modelSearch.includes(query); });
  } else if (target.id === "memory-search") {
    state.memorySearch = target.value;
    document.querySelectorAll(".memory-card").forEach((row) => { row.hidden = !row.dataset.memoryText.includes(query); });
  } else if (target.id === "tracker-search") {
    state.trackerSearch = target.value;
    document.querySelectorAll(".play-table tbody tr").forEach((row) => { row.hidden = !row.textContent.toLowerCase().includes(query); });
  }
}

function updateReminderPreview() {
  const draft = state.drafts.reminder;
  const data = draft ?? fixtures.data.reminder;
  const selected = data.repeat.map((day) => ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][day]).join(", ") || "Does not repeat";
  const preview = document.getElementById("recurrence-preview");
  if (preview) preview.textContent = `${selected} · ${data.time} (${data.timezone})`;
}

function updateTimerDisplay() {
  const face = document.querySelector(".timer-face");
  if (!face) return;
  const deadline = fixtures.data.restDeadline;
  const total = deadline ? Math.max(0, Math.ceil((deadline - Date.now()) / 1000)) : fixtures.data.restSeconds;
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  face.innerHTML = `<strong>${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}</strong><small>${deadline == null ? "Ready when you are" : total ? "Resting" : "Rest is finished"}</small>`;
}

function updateTelegramBackButton() {
  const backButton = window.Telegram?.WebApp?.BackButton;
  if (!backButton) return;
  if (state.screen === "home") backButton.hide();
  else backButton.show();
}

function showProfile() {
  const prefs = [
    ["both", "Keep both modes available"],
    ["companion", "Open Telegram companion first"],
    ["assistant", "Open In-app assistant first"],
    ["deferred", "Decide later"],
  ];
  const active = state.preference;
  const choices = prefs.map(([value, label]) => `<button class="pill-button preference-choice ${active === value ? "selected" : ""}" type="button" data-action="set-preference" data-value="${value}" aria-pressed="${active === value}">${escapeHtml(label)}</button>`).join("");
  showDialog(actionDialog, `<div class="dialog-inner"><div class="heading-kicker">YOUR SETTINGS</div><h2 id="action-title">Set your Blacki vibe.</h2><p>Nothing is chosen for you. Pick a starting preference or leave both ways open.</p><div class="preference-list">${choices}</div><label class="switch-line notification-choice"><input type="checkbox" data-notify-choice ${state.notifyOnCompletion ? "checked" : ""} />Preview a Telegram alert after an in-app reply</label><small class="field-note">The alert is simulated. This demo never sends a Telegram message.</small><div class="dialog-actions"><button class="play-button soft" type="button" data-action="close-dialog">Done</button></div></div>`);
}

function showConfirmation(title, copy, confirmLabel, action, payload = {}) {
  state.pendingAction = { action, ...payload };
  showDialog(actionDialog, `<div class="dialog-inner"><div class="heading-kicker">PLEASE CONFIRM</div><h2 id="action-title">${escapeHtml(title)}</h2><p>${escapeHtml(copy)}</p><div class="dialog-actions"><button class="play-button soft" type="button" data-action="close-dialog">Keep it</button><button class="play-button danger" type="button" data-action="confirm-action">${escapeHtml(confirmLabel)}</button></div></div>`);
}

function handleConfirmation() {
  const pending = state.pendingAction;
  state.pendingAction = null;
  closeDialog(actionDialog);
  if (!pending) return;
  if (pending.action === "reset") {
    fixtures.reset();
    state.drafts = {};
    state.screen = "home";
    state.selectedRecord = null;
    state.fileSearch = "";
    render();
    showToast("Demo reset. All your sample records are back.");
  } else if (pending.action === "delete-files") {
    fixtures.data.files = fixtures.data.files.filter((file) => !fixtures.data.selectedFiles.includes(file.id));
    fixtures.data.selectedFiles = [];
    render();
    showToast("Selected demo files deleted.");
  } else if (pending.action === "delete-memory") {
    fixtures.data.memories = fixtures.data.memories.filter((memory) => memory.id !== pending.id);
    render();
    showToast("Demo memory removed.");
  } else if (pending.action === "cancel-reminder") {
    fixtures.data.reminder.enabled = false;
    render();
    showToast("Demo reminder paused.");
  }
}

function confirmSwitch(action, mode) {
  closeDialog(decisionDialog);
  if (action === "mode-save") { commitDrafts(); }
  if (action === "mode-discard") { state.drafts = {}; }
  setMode(mode);
  if (action === "mode-save") showToast("Demo changes saved before switching.");
  if (action === "mode-discard") showToast("Demo changes discarded.");
}

function setRangeById(id, value) {
  const range = document.querySelector(`[data-range="${id}"]`);
  const exact = document.querySelector(`[data-exact="${id}"]`);
  if (!range || !exact) return;
  const min = Number(range.min); const max = Number(range.max);
  const clamped = Math.min(Math.max(Number(value), min), max);
  range.value = String(clamped); exact.value = String(clamped);
  updateRange(range);
}

function saveMeal() {
  const form = document.getElementById("meal-form");
  if (!form || !form.reportValidity()) return;
  captureMealDraft(form);
  const draft = state.drafts.meal;
  if (!draft.title.trim() || !Number.isFinite(draft.calories) || draft.calories <= 0) {
    showToast("Add a meal name and a valid calorie value first."); return;
  }
  commitDrafts();
  render();
  showToast("Meal saved in the demo. Today's totals are up to date.");
}

function saveReminder() {
  const form = document.getElementById("reminder-form");
  if (!form || !form.reportValidity()) return;
  captureReminderDraft();
  if (!state.drafts.reminder.title.trim()) { showToast("Add a reminder before saving."); return; }
  commitDrafts();
  render();
  showToast("Reminder saved in the demo.");
}

function addSet(exerciseId) {
  const exercise = fixtures.data.workouts[0].exercises.find((item) => item.id === exerciseId);
  if (!exercise) return;
  const id = `${exerciseId}-set-${exercise.sets.length + 1}`;
  exercise.sets.push({ id, weight: exercise.sets.at(-1)?.weight ?? 0, reps: exercise.sets.at(-1)?.reps ?? 10, warmup: false, complete: false });
  render();
  showToast("A new demo set is ready to edit.");
}

function saveAssistantMeal() {
  const calories = Number(document.querySelector('[data-exact="assistant-calories"]')?.value);
  if (!Number.isFinite(calories) || calories < 0) { showToast("Enter a calorie value first."); return; }
  fixtures.updateMeal("meal-lunch", { calories });
  delete state.drafts.assistantCalories;
  render();
  showToast("Lunch updated in the demo.");
}

function handleAction(target) {
  const action = target.dataset.action;
  if (!action) return;
  if (action === "navigate") { captureDrafts(); navigate(target.dataset.screen); return; }
  if (action === "close-dialog") { closeDialog(decisionDialog); closeDialog(actionDialog); state.pendingAction = null; return; }
  if (action === "compare") { compareDialog(); return; }
  if (action === "switch-mode") { requestMode(state.mode === "companion" ? "assistant" : "companion"); return; }
  if (action === "mode-keep" || action === "mode-save" || action === "mode-discard") { confirmSwitch(action, target.dataset.mode); return; }
  if (action === "start-meal-trial") { closeDialog(decisionDialog); state.screen = "meals"; state.selectedRecord = "meal-lunch"; render({ focusMain: true }); return; }
  if (action === "select-meal") {
    if (state.drafts.meal && state.drafts.meal.id !== target.dataset.id) { showToast("Save this meal draft before opening another one."); return; }
    state.selectedRecord = target.dataset.id; render(); return;
  }
  if (action === "meal-today") {
    const form = document.getElementById("meal-form");
    if (!form) { showToast("Choose a meal first."); return; }
    form.elements.date.value = localDateValue();
    captureMealDraft(form);
    showToast("Date set to today. Save the meal to keep this change.");
    return;
  }
  if (action === "save-meal") { saveMeal(); return; }
  if (action === "add-meal") {
    const id = `meal-new-${Date.now()}`;
    state.selectedRecord = id;
    state.drafts.meal = { id, isNew: true, title: "", type: "snack", calories: 100, protein: null, carbs: null, fat: null };
    render();
    document.getElementById("meal-title")?.focus();
    return;
  }
  if (action === "edit-goal") { navigate("meals"); return; }
  if (action === "save-goal") {
    const raw = Number(document.querySelector('[data-exact="calorie-goal"]')?.value ?? document.getElementById("goal-setup-value")?.value);
    if (!Number.isFinite(raw) || raw < 1000 || raw > 3500) { showToast("Choose a goal between 1,000 and 3,500 kcal."); return; }
    state.drafts.goal = raw;
    commitDrafts(); render(); showToast("Goal saved in the demo."); return;
  }
  if (action === "toggle-set") {
    const row = target.closest(".set-row"); captureSetDraft(row);
    const setId = row?.dataset.setId;
    const draft = state.drafts.sets?.[setId];
    if (draft) draft.complete = !draft.complete;
    render(); return;
  }
  if (action === "step") {
    const row = target.closest(".set-row");
    const input = row?.querySelector(`[data-step-value="${target.dataset.step}"]`);
    if (!row || !input) return;
    const next = Math.min(Math.max(Number(input.value) + Number(target.dataset.delta), Number(input.min)), Number(input.max));
    input.value = String(Math.round(next * 10) / 10);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    return;
  }
  if (action === "add-set") { addSet(target.dataset.exerciseId); return; }
  if (action === "save-workout") {
    captureDrafts();
    commitDrafts(); render(); showToast("Workout sets saved in the demo."); return;
  }
  if (action === "complete-workout" || action === "finish-workout") {
    captureDrafts(); commitDrafts();
    fixtures.data.workouts[0].status = "Completed";
    fixtures.data.activity.unshift({ title: "Workout completed", detail: "Upper body A was marked complete.", time: "Just now", status: "Completed" });
    render(); showToast("Workout marked complete in the demo."); return;
  }
  if (action === "rest-preset") { state.drafts.restSeconds = Number(target.dataset.value); render(); return; }
  if (action === "toggle-timer") {
    if (fixtures.data.restDeadline) fixtures.data.restDeadline = null;
    else {
      if (state.drafts.restSeconds != null) fixtures.data.restSeconds = Number(state.drafts.restSeconds);
      delete state.drafts.restSeconds;
      fixtures.data.restDeadline = Date.now() + Number(fixtures.data.restSeconds) * 1000;
    }
    updateTimerDisplay();
    if (!timerInterval) timerInterval = window.setInterval(updateTimerDisplay, 1000);
    return;
  }
  if (action === "reminder-day") {
    const form = document.getElementById("reminder-form");
    if (form) form.elements.date.value = target.dataset.date;
    captureReminderDraft(); updateReminderPreview();
    return;
  }
  if (action === "toggle-weekday") {
    const current = state.drafts.reminder?.repeat ?? fixtures.data.reminder.repeat;
    const day = Number(target.dataset.day);
    const repeat = current.includes(day) ? current.filter((item) => item !== day) : [...current, day].sort();
    target.setAttribute("aria-pressed", repeat.includes(day) ? "true" : "false");
    captureReminderDraft();
    state.drafts.reminder.repeat = repeat;
    updateReminderPreview();
    return;
  }
  if (action === "save-reminder") { saveReminder(); return; }
  if (action === "snooze") {
    const minutes = Number(target.dataset.value);
    fixtures.updateReminder({ snoozedUntil: new Date(Date.now() + minutes * 60000).toISOString() });
    render(); showToast(`This occurrence is snoozed for ${minutes} minutes. The regular schedule stays unchanged.`); return;
  }
  if (action === "cancel-reminder") { showConfirmation("Pause this reminder?", `The demo reminder “${fixtures.data.reminder.title}” will be paused.`, "Pause reminder", "cancel-reminder"); return; }
  if (action === "file-filter") { fixtures.data.fileFilter = target.dataset.value; render(); return; }
  if (action === "toggle-file-view") {
    fixtures.data.fileLayout = fixtures.data.fileLayout === "list" ? "grid" : "list";
    render(); return;
  }
  if (action === "restore-files") {
    const selected = fixtures.data.files.filter((file) => fixtures.data.selectedFiles.includes(file.id));
    showToast(selected.length ? `${selected.length} demo file${selected.length === 1 ? "" : "s"} restored to this demo conversation.` : "Select at least one available demo file."); return;
  }
  if (action === "preview-file") { previewSelectedFile(); return; }
  if (action === "download-file") { showToast("Downloads are disabled for demo files. No real file was requested."); return; }
  if (action === "delete-files") {
    const names = fixtures.data.files.filter((file) => fixtures.data.selectedFiles.includes(file.id)).map((file) => file.name);
    if (!names.length) { showToast("Select the exact demo files you want to delete."); return; }
    showConfirmation("Delete these demo files?", names.join(", "), "Delete selected", "delete-files"); return;
  }
  if (action === "choose-model") { state.drafts.model = target.dataset.id; render(); return; }
  if (action === "favorite-model") {
    const model = fixtures.data.models.find((item) => item.id === target.dataset.id);
    if (model) model.favorite = !model.favorite;
    render(); return;
  }
  if (action === "reasoning") { state.drafts.reasoning = target.dataset.value; render(); return; }
  if (action === "save-model") {
    const exact = document.getElementById("exact-model-id")?.value.trim();
    if (exact) state.drafts.model = exact;
    if (!state.drafts.model) { showToast("Choose a model or enter an exact model ID."); return; }
    commitDrafts(); render(); showToast("Model preference saved for the next demo turn."); return;
  }
  if (action === "edit-memory") { openMemoryEditor(target.dataset.id); return; }
  if (action === "save-memory") { saveMemory(); return; }
  if (action === "delete-memory") {
    const memory = fixtures.data.memories.find((item) => item.id === target.dataset.id);
    if (!memory) { showToast("That demo memory is no longer available."); return; }
    showConfirmation("Delete this memory?", memory.text, "Delete memory", "delete-memory", { id: memory.id }); return;
  }
  if (action === "choose-tracker") { state.selectedTracker = target.dataset.id; render(); return; }
  if (action === "chart-range") { fixtures.data.chartRange = target.dataset.value; render(); return; }
  if (action === "sort-tracker") { state.trackerSort = state.trackerSort === "asc" ? "desc" : "asc"; const tracker = fixtures.data.trackers.find((item) => item.id === state.selectedTracker) ?? fixtures.data.trackers[0]; tracker.rows.reverse(); render(); return; }
  if (action === "add-tracker-row") { openTrackerRowDialog(); return; }
  if (action === "save-tracker") { saveTrackerRow(); return; }
  if (action === "save-new-tracker") { saveNewTrackerRow(); return; }
  if (action === "save-assistant-meal") { saveAssistantMeal(); return; }
  if (action === "reset") { showConfirmation("Reset all demo records?", "Meals, workouts, reminders, files, and chat return to the starting examples.", "Reset demo", "reset"); return; }
  if (action === "set-preference") {
    state.preference = target.dataset.value;
    if (state.preference === "companion") setMode("companion");
    if (state.preference === "assistant") setMode("assistant");
    showProfile();
    return;
  }
  if (action === "confirm-action") { handleConfirmation(); return; }
}

function previewSelectedFile() {
  const file = fixtures.data.files.find((item) => fixtures.data.selectedFiles.includes(item.id));
  if (!file) { showToast("Select a demo file first."); return; }
  if (!file.available) { showToast("That demo file is unavailable. The source is unchanged."); return; }
  const content = file.kind === "Text" ? "A tiny note for a calmer weekend.\n\nThis content is a safe, escaped demo preview." : "Image previews are represented as a safe placeholder in this demo.";
  showDialog(actionDialog, `<div class="dialog-inner"><div class="heading-kicker">SAFE DEMO PREVIEW</div><h2 id="action-title">${escapeHtml(file.name)}</h2><p>${escapeHtml(file.kind)} · ${formatBytes(file.size)} · No copy is saved.</p><pre class="safe-preview">${escapeHtml(content)}</pre><div class="dialog-actions"><button class="play-button soft" type="button" data-action="close-dialog">Close preview</button></div></div>`);
}

function formatBytes(bytes) { return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`; }

function openMemoryEditor(id) {
  const memory = fixtures.data.memories.find((item) => item.id === id);
  if (!memory) { showToast("That demo memory is no longer available."); return; }
  state.pendingAction = { action: "edit-memory", id };
  showDialog(actionDialog, `<div class="dialog-inner"><div class="heading-kicker">EDIT MEMORY</div><h2 id="action-title">Make it more accurate.</h2><p>This exact demo memory will be changed. No external memory provider is connected.</p><div class="field"><label for="memory-editor">Memory text</label><textarea class="textarea" id="memory-editor" maxlength="800">${escapeHtml(memory.text)}</textarea></div><div class="dialog-actions"><button class="play-button soft" type="button" data-action="close-dialog">Cancel</button><button class="play-button" type="button" data-action="save-memory">Save memory</button></div></div>`);
}

function saveMemory() {
  const id = state.pendingAction?.id;
  const memory = fixtures.data.memories.find((item) => item.id === id);
  const text = document.getElementById("memory-editor")?.value.trim();
  if (!memory || !text) { showToast("That memory is unavailable or empty."); closeDialog(actionDialog); return; }
  memory.text = text;
  memory.updated = "Edited just now";
  state.pendingAction = null;
  closeDialog(actionDialog); render(); showToast("Demo memory updated.");
}

function selectedTracker() {
  return fixtures.data.trackers.find((item) => item.id === state.selectedTracker) ?? fixtures.data.trackers[0];
}

function openTrackerRowDialog() {
  const tracker = selectedTracker();
  if (!tracker) return;
  state.pendingAction = { action: "add-tracker-row", trackerId: tracker.id };
  const fields = tracker.fields.map((field, index) => `<div class="field"><label for="new-tracker-${index}">${escapeHtml(field)}</label><input class="input" id="new-tracker-${index}" data-new-tracker-field="${index}" maxlength="240" /></div>`).join("");
  showDialog(actionDialog, `<div class="dialog-inner"><div class="heading-kicker">NEW DEMO ROW</div><h2 id="action-title">Add to ${escapeHtml(tracker.name)}.</h2><p>This row stays in the resettable fixture data.</p><div class="tracker-fields">${fields}</div><div class="dialog-actions"><button class="btn play-button soft" type="button" data-action="close-dialog">Cancel</button><button class="btn play-button" type="button" data-action="save-new-tracker">Add row</button></div></div>`);
}

function saveTrackerRow() {
  const form = document.getElementById("tracker-form");
  if (!form) return;
  const tracker = fixtures.data.trackers.find((item) => item.id === form.dataset.trackerId);
  if (!tracker || !tracker.rows.length) { showToast("That demo tracker is unavailable."); return; }
  const values = tracker.fields.map((_, index) => form.querySelector(`[data-tracker-field="${index}"]`)?.value ?? "");
  tracker.rows[0] = values;
  render();
  showToast("Tracker row saved in the demo.");
}

function saveNewTrackerRow() {
  const trackerId = state.pendingAction?.trackerId;
  const tracker = fixtures.data.trackers.find((item) => item.id === trackerId);
  if (!tracker) { closeDialog(actionDialog); showToast("That demo tracker is no longer available."); return; }
  const row = tracker.fields.map((_, index) => document.querySelector(`[data-new-tracker-field="${index}"]`)?.value.trim() ?? "");
  if (row.some((value) => !value)) { showToast("Fill in each field before adding this demo row."); return; }
  tracker.rows.push(row);
  state.pendingAction = null;
  closeDialog(actionDialog);
  render();
  showToast("Demo row added.");
}

function showProfileFromButton() {
  showProfile();
}

document.addEventListener("click", (event) => {
  const modeChoice = event.target.closest("[data-mode-choice]");
  if (modeChoice) { requestMode(modeChoice.dataset.modeChoice); return; }
  const target = event.target.closest("button, a");
  if (!target) return;
  if (target.matches("a[data-action='navigate']")) event.preventDefault();
  if (target.id === "theme-toggle") {
    state.theme = state.theme === "blacki-playroom" ? "blacki-playroom-dark" : "blacki-playroom";
    renderNavigation();
    const tg = window.Telegram?.WebApp;
    tg?.setHeaderColor?.(state.theme === "blacki-playroom-dark" ? "#202334" : "#fff8ec");
    tg?.setBackgroundColor?.(state.theme === "blacki-playroom-dark" ? "#202334" : "#fff8ec");
    return;
  }
  if (target.id === "profile-button") { showProfileFromButton(); return; }
  if (target.id === "reset-button") { showConfirmation("Reset all demo records?", "Meals, workouts, reminders, files, and chat return to the starting examples.", "Reset demo", "reset"); return; }
  if (target.dataset.action) handleAction(target);
});

document.addEventListener("input", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement)) return;
  if (target.matches("[data-range], [data-exact]")) updateRange(target);
  if (target.id === "file-search" || target.id === "model-search" || target.id === "memory-search" || target.id === "tracker-search") updateSearch(target);
  if (target.closest("#meal-form")) captureMealDraft();
  if (target.closest("#reminder-form")) { captureReminderDraft(); updateReminderPreview(); }
  if (target.closest(".set-row")) captureSetDraft(target.closest(".set-row"));
  if (target.id === "exact-model-id") {
    state.drafts.exactModelId = target.value.trim();
    state.drafts.model = target.value.trim();
  }
  if (target.id === "goal-setup-value") {
    const value = Number(target.value);
    if (Number.isFinite(value)) state.drafts.goal = value;
  }
  if (target.matches("[data-notify-choice]")) state.notifyOnCompletion = target.checked;
});

document.addEventListener("change", (event) => {
  const target = event.target;
  if (target.matches("[data-file-select]")) {
    const id = target.dataset.fileSelect;
    const selected = fixtures.data.selectedFiles;
    fixtures.data.selectedFiles = target.checked ? [...new Set([...selected, id])] : selected.filter((item) => item !== id);
    render();
  }
  if (target.matches("[data-set-field='warmup']")) captureSetDraft(target.closest(".set-row"));
  if (target.matches("[data-set-field='weight'], [data-set-field='reps']")) captureSetDraft(target.closest(".set-row"));
  if (target.id === "meal-calories-exact") captureMealDraft();
});

document.addEventListener("submit", (event) => {
  if (event.target.id === "chat-form") {
    event.preventDefault();
    const input = document.getElementById("chat-input");
    const message = input?.value.trim();
    if (!message) return;
    fixtures.data.chat.push({ role: "user", text: message, origin: "You · this app" });
    fixtures.data.chat.push({ role: "assistant", text: demoReply(message), origin: "Demo response" });
    render();
    document.getElementById("chat-input")?.focus();
    document.getElementById("chat-scroll")?.scrollTo({ top: 999999, behavior: "smooth" });
    if (state.notifyOnCompletion) showToast("A Telegram alert would appear here. No message was sent.");
    return;
  }
  if (event.target.id === "meal-form") { event.preventDefault(); saveMeal(); return; }
  if (event.target.id === "reminder-form") { event.preventDefault(); saveReminder(); }
});

function demoReply(message) {
  const lower = message.toLowerCase();
  if (lower.includes("meal") || lower.includes("calorie")) return "I can open the meal card for you. Adjust the slider or enter an exact number, then save it when it looks right.";
  if (lower.includes("workout") || lower.includes("set")) return "Your workout is ready. Set changes stay in the demo until you save them.";
  if (lower.includes("remind")) return "Open Reminders to choose a day and see the local-time preview before saving.";
  return "This is a labelled demo reply. Try asking about a meal, a workout set, or a reminder.";
}

function initializeTelegram() {
  const telegram = window.Telegram?.WebApp;
  if (!telegram) return;
  telegram.ready();
  telegram.expand();
  if (telegram.colorScheme === "dark") state.theme = "blacki-playroom-dark";
  telegram.onEvent?.("themeChanged", () => {
    state.theme = telegram.colorScheme === "dark" ? "blacki-playroom-dark" : "blacki-playroom";
    renderNavigation();
  });
  telegram.BackButton?.onClick?.(() => navigate("home"));
  telegram.setHeaderColor?.(state.theme === "blacki-playroom-dark" ? "#202334" : "#fff8ec");
  telegram.setBackgroundColor?.(state.theme === "blacki-playroom-dark" ? "#202334" : "#fff8ec");
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") { closeDialog(decisionDialog); closeDialog(actionDialog); state.pendingAction = null; }
});

initializeTelegram();
render();
