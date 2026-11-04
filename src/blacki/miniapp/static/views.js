import { badge, button, emptyState, escapeHtml, exactRange, icon, pill, sectionHeading, stepper } from "./controls.js";
import { localDateValue } from "./fixtures.js";

const weekNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const weekdays = ["S", "M", "T", "W", "T", "F", "S"];
const COLOURS = ["#7041d9", "#4e9a68", "#e9815d", "#d4a320", "#5584b9", "#ca6f9c", "#6e81a6"];

const heading = (kicker, title, description, actions = "") => `
  <header class="page-heading">
    <div><div class="heading-kicker">${escapeHtml(kicker)}</div><h1>${escapeHtml(title)}</h1><p>${escapeHtml(description)}</p></div>
    ${actions ? `<div class="heading-actions">${actions}</div>` : ""}
  </header>`;

const card = (content, classes = "") => `<section class="play-card ${classes}">${content}</section>`;

function totals(data) {
  return data.meals
    .filter((meal) => meal.date === localDateValue())
    .reduce((sum, meal) => sum + Number(meal.calories || 0), 0);
}

function macroBox(label, value, unit) {
  return `<div class="macro-box"><span>${escapeHtml(label)}</span><strong>${value == null ? `<span class="unknown-value">Not logged</span>` : `${Number(value)} ${escapeHtml(unit)}`}</strong></div>`;
}

function mealForm(meal) {
  const selected = meal ?? { id: "meal-lunch", title: "", type: "lunch", date: localDateValue(), calories: 0, protein: null, carbs: null, fat: null };
  const options = ["breakfast", "lunch", "dinner", "snack"].map((type) => `<option value="${type}" ${selected.type === type ? "selected" : ""}>${type[0].toUpperCase()}${type.slice(1)}</option>`).join("");
  return `
    <form id="meal-form" data-meal-id="${escapeHtml(selected.id)}">
      <div class="field-grid">
        <div class="field"><label for="meal-title">Meal</label><input class="input" id="meal-title" name="title" maxlength="160" value="${escapeHtml(selected.title)}" required /></div>
        <div class="field"><label for="meal-type">Meal type</label><select class="select" id="meal-type" name="type">${options}</select></div>
        <div class="field"><label for="meal-date">Date</label><input class="input" id="meal-date" name="date" type="date" value="${escapeHtml(selected.date || localDateValue())}" required /></div>
      </div>
      <div class="field-gap"></div>
      ${exactRange({ id: "meal-calories", label: "Calories", value: selected.calories, min: 0, max: 1200, step: 1, unit: "kcal", help: "A quick estimate, with an exact field." })}
      <div class="field-grid macro-edit-grid">
        ${["protein", "carbs", "fat"].map((key) => `<div class="field"><label for="meal-${key}">${key[0].toUpperCase()}${key.slice(1)} <small>${selected[key] == null ? "Not logged" : "grams"}</small></label><input class="input" id="meal-${key}" name="${key}" type="number" min="0" step="0.1" inputmode="decimal" placeholder="Leave blank to keep unknown" value="${selected[key] == null ? "" : Number(selected[key])}" /></div>`).join("")}
      </div>
      <div class="form-footer"><span class="field-note">Unknown nutrition stays unknown until you add it.</span>${button("Save meal", "save-meal")}</div>
    </form>`;
}

function calorieChart(data, range = "Week", maxScale = 2200) {
  const values = range === "Today" ? [totals(data)] : data.weekCalories;
  const max = Math.max(maxScale, ...values);
  const bars = values.map((value, index) => {
    const height = Math.max(5, Math.round((value / max) * 100));
    const x = values.length === 1 ? 162 : 25 + index * 46;
    const y = 120 - height;
    const label = range === "Today" ? "Today" : weekNames[(index + 1) % 7];
    return `<g><rect class="chart-bar chart-fill-${index}" x="${x}" y="${y}" width="25" height="${height}" rx="8" /><text class="chart-label" x="${x + 12}" y="139" text-anchor="middle">${label}</text></g>`;
  }).join("");
  const label = range === "Today" ? "Calories logged today" : "Calories logged during the last seven days";
  return `<svg class="week-chart" viewBox="0 0 350 148" role="img" aria-label="${label}">${bars}</svg>`;
}

function homeView(state, data) {
  const count = totals(data);
  const goal = data.goalConfigured ? Number(data.calorieGoal) : null;
  const remaining = goal == null ? null : Math.max(goal - count, 0);
  const modeCopy = state.mode === "assistant"
    ? "Chat with Blacki here, then tap a control when you want to change something."
    : "Your conversation stays in Telegram. Open a screen from a bot link or use these quick actions.";
  return `${heading("Home · your day, your way", "Little steps, big energy.", "A cheerful spot to check in, make a small change, and get on with your day.")}
    <div class="panel-grid">
      <section class="hero-card">
        <div class="heading-kicker">${state.mode === "assistant" ? "IN-APP ASSISTANT" : "TELEGRAM COMPANION"}</div>
        <h2>Make today a good one.</h2>
        <p>${escapeHtml(modeCopy)}</p>
        <div class="hero-actions">${button("Open today's meals", "navigate", { tone: "" , attrs: 'data-screen="meals"' })}${button("Compare the two modes", "compare", { tone: "soft" })}</div>
      </section>
      ${card(`${sectionHeading("Food so far", "Your logged meals, no guesses added.", "◒", "butter")}
        <div class="summary-number">${count.toLocaleString()} <small>${goal == null ? "/ no goal yet" : `/ ${goal.toLocaleString()} kcal`}</small></div>
        ${goal == null ? `<p class="card-subtitle">No calorie goal is set. Add one when you want to.</p>` : `<progress class="meter-progress" value="${Math.min(count, goal)}" max="${goal}" aria-label="${count} of ${goal} calories logged"></progress><div class="summary-meta"><span>${remaining ? `${remaining.toLocaleString()} kcal left` : "Goal reached"}</span><span>Goal, not a limit</span></div>`}
        <div class="home-footer">${button(goal == null ? "Set a goal" : "Adjust goal", "edit-goal", { small: true, tone: "soft" })}</div>`, "home-summary tint-butter")}
      <div class="quick-row">
        <button class="quick-tile tint-mint" type="button" data-action="navigate" data-screen="training"><span class="card-icon">↗</span><span><strong>Training</strong><br /><small>2 moves · 1 set done</small></span></button>
        <button class="quick-tile tint-peach" type="button" data-action="navigate" data-screen="reminders"><span class="card-icon">◷</span><span><strong>Next reminder</strong><br /><small>Take a little walk · 6:30 pm</small></span></button>
        <button class="quick-tile tint-sky" type="button" data-action="navigate" data-screen="files"><span class="card-icon">▤</span><span><strong>Your files</strong><br /><small>${data.files.length} demo items</small></span></button>
        <button class="quick-tile tint-lavender" type="button" data-action="navigate" data-screen="more"><span class="card-icon">✳</span><span><strong>More tools</strong><br /><small>Memory, models, trackers</small></span></button>
      </div>
      ${card(`${sectionHeading("This week's rhythm", "A quick view of what you logged.", "▥", "lavender")}${calorieChart(data)}<div class="chart-legend"><span class="legend-dot"></span>Calories logged · sample data</div>`, "section-card tint-lavender")}
      ${card(`${sectionHeading("Your next little win", "Tuesday, then another when you're ready.", "✓", "mint")}<div class="data-row"><div><strong>Finish a workout set</strong><small>Seated cable row · 2 sets to go</small></div>${badge("In progress", "info")}</div><div class="data-row"><div><strong>Try the other Blacki mode</strong><small>Same records, another way to talk</small></div>${button("Switch", "switch-mode", { small: true, tone: "soft" })}</div><div class="home-footer">${button("Open training", "navigate", { small: true, tone: "soft", attrs: 'data-screen="training"' })}</div>`, "section-card tint-mint")}
    </div>`;
}

function mealsView(state, data) {
  const selectedId = state.selectedRecord && data.meals.some((meal) => meal.id === state.selectedRecord) ? state.selectedRecord : "meal-lunch";
  const selected = data.meals.find((meal) => meal.id === selectedId) ?? state.drafts.meal ?? data.meals[0];
  const selectedDraft = state.drafts.meal?.id === selected?.id ? state.drafts.meal : selected;
  const goal = state.drafts.goal ?? (data.goalConfigured ? Number(data.calorieGoal) : 1500);
  const rows = data.meals.map((meal) => `<button class="meal-select-row ${meal.id === selectedId ? "selected" : ""}" type="button" data-action="select-meal" data-id="${escapeHtml(meal.id)}" aria-pressed="${meal.id === selectedId}"><span class="meal-select-icon">${meal.type === "lunch" ? "☼" : meal.type === "breakfast" ? "◉" : "•"}</span><span class="meal-select-copy"><strong>${escapeHtml(meal.title)}</strong><small>${escapeHtml(meal.date || "Date unknown")} · ${escapeHtml(meal.type)} · ${escapeHtml(meal.time)}</small></span><span class="data-value">${Number(meal.calories)} kcal</span></button>`).join("");
  return `${heading("Meals & health", "A little more in control.", "Correct a meal directly. Exact fields are here if sliders feel fiddly.", button("Today", "meal-today", { small: true, tone: "soft" }))}
    <div class="panel-grid">
      ${card(`${sectionHeading("Today on your plate", `${data.meals.filter((meal) => meal.date === localDateValue()).length} logged meals · totals update when you save.`, "◒", "butter")}<div class="total-line"><strong>${totals(data).toLocaleString()} <small>kcal logged</small></strong>${data.goalConfigured ? badge(`${data.calorieGoal} kcal goal`, "info") : badge("No goal set", "neutral")}</div><div class="meal-list">${rows}</div><div class="home-footer">${button("Add a meal", "add-meal", { tone: "soft" })}</div>`, "section-two-thirds tint-butter")}
      ${card(`${sectionHeading("Daily goal", "A goal is guidance, not a score.", "◎", "lavender")}${data.goalConfigured || state.drafts.goal != null ? exactRange({ id: "calorie-goal", label: "Calorie goal", value: goal, min: 1000, max: 3500, step: 1, unit: "kcal per day", help: "Set the number that makes sense for you." }) : `<div class="field goal-setup"><label for="goal-setup-value">Your daily goal <small>Nothing is filled in for you.</small></label><input class="input" id="goal-setup-value" type="number" min="1000" max="3500" step="1" inputmode="numeric" placeholder="Type a number" /></div>`}<div class="home-footer">${button(data.goalConfigured ? "Save goal" : "Set this goal", "save-goal")}</div>`, "section-third tint-lavender")}
      ${card(`${sectionHeading(selectedDraft ? `Edit ${escapeHtml(selectedDraft.type)}` : "Add meal", "Changes stay local in this demo.", "✎", "peach")}${mealForm(selectedDraft)}`, "section-two-thirds tint-peach")}
      ${card(`${sectionHeading("Meal details", "Only values you entered are shown.", "◌", "mint")}${selectedDraft ? `<div class="macro-grid">${macroBox("Protein", selectedDraft.protein, "g")}${macroBox("Carbs", selectedDraft.carbs, "g")}${macroBox("Fat", selectedDraft.fat, "g")}</div><div class="item-note">${selectedDraft.protein == null || selectedDraft.carbs == null || selectedDraft.fat == null ? "Some nutrition values are unknown. They stay unknown until you add them." : "All shown values came from this saved meal."}</div>` : emptyState("Choose a meal to see its details.")}`, "section-third tint-mint")}
      ${card(`${sectionHeading("Seven day view", "Missing health records stay empty, not zero.", "▥", "sky")}<div class="button-row chart-filters">${pill("Week", "chart-range", "Week", data.chartRange === "Week")}${pill("Today", "chart-range", "Today", data.chartRange === "Today")}</div>${calorieChart(data, data.chartRange)}<div class="chart-legend"><span class="legend-dot"></span>Logged calories · sample history</div><div class="health-unavailable"><span class="unavailable-dot">?</span><span><strong>Sleep & activity</strong><small>Not connected in this demo</small></span></div>`, "section-wide tint-sky")}
    </div>`;
}

function workoutView(state, data) {
  const workout = data.workouts.find((item) => item.id === "workout-today");
  const exerciseCards = workout.exercises.map((exercise) => `
    <article class="item-card exercise-card">
      <div class="item-card-top"><div><h3>${escapeHtml(exercise.name)}</h3><p>Previous: ${exercise.id === "exercise-row" ? "40 kg × 10" : "14 kg × 10"}</p></div>${badge(`${exercise.sets.filter((set) => set.complete).length}/${exercise.sets.length} sets`, "info")}</div>
      <div class="set-list">${exercise.sets.map((savedSet, index) => { const set = state.drafts.sets?.[savedSet.id] ?? savedSet; return `<div class="set-row" data-set-id="${escapeHtml(set.id)}" data-exercise-id="${escapeHtml(exercise.id)}">
        <span class="set-index">Set ${index + 1}</span>
        <div class="set-controls"><span class="set-unit">kg</span>${stepper({ id: `${set.id}-weight`, field: "weight", label: `Set ${index + 1} weight in kilograms`, value: set.weight, min: 0, max: 500, step: 0.5 })}<span class="set-unit">reps</span>${stepper({ id: `${set.id}-reps`, field: "reps", label: `Set ${index + 1} repetitions`, value: set.reps, min: 0, max: 100, step: 1 })}</div>
        <div class="set-actions"><label class="switch-line"><input type="checkbox" data-set-field="warmup" ${set.warmup ? "checked" : ""} />Warm-up</label><button class="complete-set ${set.complete ? "is-complete" : ""}" type="button" data-action="toggle-set" aria-pressed="${set.complete}">${set.complete ? "Done" : "Finish"}</button></div>
      </div>`; }).join("")}</div>
      <div class="exercise-footer">${button("Save set changes", "save-workout", { small: true })}${button("Add set", "add-set", { small: true, tone: "soft", attrs: `data-exercise-id="${escapeHtml(exercise.id)}"` })}</div>
    </article>`).join("");
  const deadline = data.restDeadline ? Math.max(0, Math.ceil((data.restDeadline - Date.now()) / 1000)) : null;
  const restSeconds = state.drafts.restSeconds ?? Number(data.restSeconds);
  const timerValue = deadline ?? restSeconds;
  const minutes = Math.floor(timerValue / 60);
  const seconds = timerValue % 60;
  return `${heading("Training", "Move at your own pace.", "Your sets, your progress. Correct a row without logging it twice.", button("Finish workout", "finish-workout", { small: true }))}
    <div class="panel-grid">
      ${card(`${sectionHeading(workout.title, "Wednesday · Day 2 of your plan", "↗", "mint")}<div class="workout-meta"><span>${badge(workout.status, "info")}</span><span class="subtle-label">Previous comparable session · Monday</span></div>${exerciseCards}<div class="home-footer">${button("Mark workout complete", "complete-workout", { tone: "secondary" })}</div>`, "section-two-thirds tint-mint")}
      ${card(`${sectionHeading("Rest timer", "It keeps time while this app is open.", "◷", "peach")}<div class="timer-face" aria-live="polite"><strong>${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}</strong><small>${deadline == null ? "Ready when you are" : timerValue ? "Resting" : "Rest is finished"}</small></div><div class="button-row timer-presets">${[60, 90, 120, 180].map((value) => pill(`${value / 60} min`, "rest-preset", String(value), Number(restSeconds) === value)).join("")}</div>${exactRange({ id: "rest-seconds", label: "Rest duration", value: restSeconds, min: 15, max: 300, step: 1, unit: "seconds", help: "Pick a preset or set the exact time." })}<div class="home-footer">${button(deadline ? "Stop timer" : "Start timer", "toggle-timer", { tone: "soft" })}<p class="timer-note">A closed Telegram window may pause updates. The time is recalculated when you come back.</p></div>`, "section-third tint-peach")}
      ${card(`${sectionHeading("Recent progress", "Previous session · same workout", "▥", "lavender")}${calorieChart({ weekCalories: [30, 44, 38, 58, 50, 65, 72] }, "Week", 72)}<div class="chart-legend"><span class="legend-dot"></span>Training effort · example view</div><div class="data-row"><div><strong>Seated cable row</strong><small>Last time · 40 kg × 10 reps</small></div><span class="data-value">+5 kg</span></div>`, "section-wide tint-lavender")}
    </div>`;
}

function reminderView(state, data) {
  const reminder = state.drafts.reminder ?? data.reminder;
  const base = new Date(`${reminder.date}T12:00:00`);
  const strip = Array.from({ length: 7 }, (_, index) => {
    const date = new Date(base); date.setDate(base.getDate() + index);
    const day = date.getDay();
    const current = date.toISOString().slice(0, 10) === reminder.date;
    return `<button class="day-choice ${current ? "selected" : ""}" type="button" data-action="reminder-day" data-date="${date.toISOString().slice(0, 10)}" aria-pressed="${current}"><strong>${day}</strong><small>${weekNames[day]} ${date.getDate()}</small></button>`;
  }).join("");
  const repeat = reminder.repeat.map((day) => weekNames[day]).join(", ") || "Does not repeat";
  const snooze = reminder.snoozedUntil ? `<div class="snooze-status"><strong>Snoozed occurrence</strong><span>${escapeHtml(new Date(reminder.snoozedUntil).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }))}</span><small>Next regular time stays ${escapeHtml(reminder.time)} · ${escapeHtml(reminder.timezone)}</small></div>` : "";
  return `${heading("Reminders", "A nudge, when you need it.", "See the local time and next dates before you save a change.")}
    <div class="panel-grid">
      ${card(`${sectionHeading("Your agenda", "November · Asia/Kolkata", "◷", "sky")}<div class="day-strip" role="group" aria-label="Choose a date">${strip}</div><div class="reminder-card"><div class="item-card-top"><div><h3>${escapeHtml(reminder.title)}</h3><p>Today · ${escapeHtml(reminder.time)} · ${escapeHtml(reminder.timezone)}</p></div>${badge(reminder.enabled ? "On" : "Paused", reminder.enabled ? "success" : "neutral")}</div>${snooze}<div class="button-row snooze-actions"><span class="field-label">Snooze this occurrence</span>${[5, 15, 30].map((minutes) => pill(`${minutes} min`, "snooze", String(minutes))).join("")}</div></div>`, "section-two-thirds tint-sky")}
      ${card(`${sectionHeading("Change reminder", "The recurring schedule stays separate from snooze.", "✎", "butter")}<form id="reminder-form"><div class="field"><label for="reminder-title">What should Blacki remind you?</label><input class="input" id="reminder-title" name="title" maxlength="120" value="${escapeHtml(reminder.title)}" required /></div><div class="field-grid reminder-datetime"><div class="field"><label for="reminder-date">Date</label><input class="input" id="reminder-date" name="date" type="date" value="${escapeHtml(reminder.date)}" required /></div><div class="field"><label for="reminder-time">Local time</label><input class="input" id="reminder-time" name="time" type="time" value="${escapeHtml(reminder.time)}" required /></div></div><div class="field"><span class="field-label">Repeat on <small>Preview uses ${escapeHtml(reminder.timezone)}</small></span><div class="weekday-row" role="group" aria-label="Repeat weekdays">${weekdays.map((label, day) => `<button class="weekday-chip" type="button" data-action="toggle-weekday" data-day="${day}" aria-pressed="${reminder.repeat.includes(day)}">${label}</button>`).join("")}</div></div><div class="recurrence-preview" id="recurrence-preview">${escapeHtml(repeat)} · ${escapeHtml(reminder.time)} (${escapeHtml(reminder.timezone)})</div><div class="form-footer">${button("Save reminder", "save-reminder")}</div></form>`, "section-third tint-butter")}
      ${card(`${sectionHeading("Trial task", "Try the same change in each mode.", "✳", "lavender")}<ol class="trial-list"><li class="trial-item"><span class="trial-number">1</span><span><strong>Choose a different day</strong><small>Watch the recurrence preview update.</small></span>${badge("Ready", "neutral")}</li><li class="trial-item"><span class="trial-number">2</span><span><strong>Snooze once</strong><small>Check that the regular schedule stays put.</small></span>${badge("Ready", "neutral")}</li></ol>`, "section-wide tint-lavender")}
    </div>`;
}

function filesView(state, data) {
  const matching = data.files.filter((file) => (data.fileFilter === "All files" || file.kind === data.fileFilter) && (!state.fileSearch || file.name.toLowerCase().includes(state.fileSearch.toLowerCase())));
  const files = matching.length ? matching.map((file) => `<div class="file-row" data-file-name="${escapeHtml(file.name.toLowerCase())}"><input type="checkbox" data-file-select="${escapeHtml(file.id)}" aria-label="Select ${escapeHtml(file.name)}" ${data.selectedFiles.includes(file.id) ? "checked" : ""} /><span class="file-icon" aria-hidden="true">${file.kind === "Image" ? "▧" : file.kind === "Text" ? "T" : "▤"}</span><span class="file-name"><strong>${escapeHtml(file.name)}</strong><small>${escapeHtml(file.kind)} · ${formatBytes(file.size)} · ${escapeHtml(file.expires)}</small></span><span>${file.available ? badge("Available", "success") : badge("Unavailable", "warn")}</span></div>`).join("") : emptyState("No files match this search.");
  return `${heading("Files", "Your things, in one place.", "Browse the durable file catalog. Preview and restore actions are simulated here.", button(data.fileLayout === "list" ? "Card view" : "List view", "toggle-file-view", { small: true, tone: "soft" }))}
    <div class="panel-grid">
      ${card(`${sectionHeading("File shelf", "Demo files are separate from your real attachments.", "▤", "sky")}<div class="filter-row"><label class="visually-hidden" for="file-search">Search your files</label><input class="input" id="file-search" type="search" placeholder="Find a file" value="${escapeHtml(state.fileSearch || "")}" />${["All files", "Image", "Text", "Document"].map((kind) => pill(kind, "file-filter", kind, data.fileFilter === kind)).join("")}</div><div class="file-list ${data.fileLayout === "grid" ? "grid-view" : ""}">${files}</div><div class="file-actions"><span class="field-note">${data.selectedFiles.length} selected</span><div class="button-row">${button("Restore to conversation", "restore-files", { small: true, tone: "soft" })}${button("Delete selected", "delete-files", { small: true, tone: "danger" })}</div></div>`, "section-two-thirds tint-sky")}
      ${card(`${sectionHeading("Safe preview", "Nothing opens as executable content.", "◫", "mint")}<div class="preview-frame"><div class="preview-paper"><div class="preview-lines"></div><strong>File preview</strong><small>Choose a file to see a safe text or image preview.</small></div></div><div class="button-row home-footer">${button("Preview selected", "preview-file", { small: true, tone: "soft" })}${button("Download existing file", "download-file", { small: true, tone: "soft" })}</div><p class="card-subtitle">This demo does not connect to R2 or download real files.</p>`, "section-third tint-mint")}
      ${card(`${sectionHeading("How files work", "Your originals stay where they are.", "!", "butter")}<ul class="plain-list"><li>Temporary Gmail and sandbox files are not copied here.</li><li>Restore creates a working copy for the current conversation.</li><li>Deleting a durable file always needs a confirmation.</li></ul>`, "section-wide tint-butter")}
    </div>`;
}

function formatBytes(bytes) {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`;
}

function modelsView(state, data) {
  const visible = data.models.filter((model) => `${model.label} ${model.id} ${model.provider}`.toLowerCase().includes((state.modelSearch || "").toLowerCase()));
  const activeModel = state.drafts.model ?? data.activeModel;
  const model = data.models.find((item) => item.id === activeModel);
  const reasoning = model?.reasoning ?? ["default"];
  return `${heading("More · model settings", "Choose your thinking buddy.", "Search names or enter an exact model ID. Saved changes would apply to the next turn.")}
    <div class="panel-grid">
      ${card(`${sectionHeading("Model choices", "Provider availability is not checked in this demo.", "✳", "lavender")}<div class="field"><label for="model-search">Search models</label><input class="input" id="model-search" type="search" placeholder="Try Gemini, GPT, Claude…" value="${escapeHtml(state.modelSearch || "")}" /></div><div class="model-list">${visible.map((option) => `<div class="model-row" data-model-search="${escapeHtml(`${option.label} ${option.id} ${option.provider}`.toLowerCase())}"><button type="button" class="model-option ${option.id === activeModel ? "active" : ""}" data-action="choose-model" data-id="${escapeHtml(option.id)}" aria-pressed="${option.id === activeModel}"><span class="model-avatar">${escapeHtml(option.provider.slice(0, 1))}</span><span class="model-copy"><strong>${escapeHtml(option.label)}</strong><small>${escapeHtml(option.id)} · ${escapeHtml(option.provider)}</small></span><span class="model-check">${option.id === activeModel ? "✓" : ""}</span></button><button class="favorite" type="button" data-action="favorite-model" data-id="${escapeHtml(option.id)}" aria-label="${option.favorite ? "Remove" : "Add"} ${escapeHtml(option.label)} ${option.favorite ? "from" : "to"} favourites">${option.favorite ? "★" : "☆"}</button></div>`).join("") || emptyState("No models matched. Use the exact ID below if you know it.")}</div><div class="field exact-model"><label for="exact-model-id">Exact model ID</label><input class="input" id="exact-model-id" maxlength="160" placeholder="provider/model-id" value="${escapeHtml(state.drafts.exactModelId || "")}" /><small class="field-note">Custom IDs are kept as typed in this demo.</small></div><div class="form-footer">${button("Save model", "save-model")}</div>`, "section-two-thirds tint-lavender")}
      ${card(`${sectionHeading("Reasoning level", "Choose a supported option.", "◌", "mint")}<p class="card-subtitle">Current selection · ${escapeHtml(model?.label ?? activeModel)}</p><div class="reasoning-options">${reasoning.map((choice) => pill(choice === "default" ? "Use model default" : choice[0].toUpperCase() + choice.slice(1), "reasoning", choice, (state.drafts.reasoning ?? data.reasoning) === choice)).join("")}</div><div class="item-note">This is a set of choices, not a made-up percentage.</div>${button("Save reasoning", "save-model", { small: true, tone: "soft" })}`, "section-third tint-mint")}
      ${card(`${sectionHeading("Favourites", "Your quick picks.", "★", "butter")}<div class="model-favourites">${data.models.filter((item) => item.favorite).map((item) => `<div class="data-row"><div><strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(item.id)}</small></div>${badge("Favourite", "info")}</div>`).join("") || emptyState("No favourites yet. Tap a star to add one.")}</div>`, "section-wide tint-butter")}
    </div>`;
}

function memoryView(state, data) {
  const memorySearch = (state.memorySearch || "").toLowerCase();
  const memories = data.memories.filter((memory) => memory.text.toLowerCase().includes(memorySearch));
  return `${heading("More · memory", "What Blacki remembers.", "Review a note, make it more accurate, or remove the exact item.")}
    <div class="panel-grid">
      ${card(`${sectionHeading("Your memories", "A small window into saved personal notes.", "◉", "peach")}<div class="field"><label for="memory-search">Search memories</label><input class="input" id="memory-search" type="search" placeholder="Try gym, food, coffee…" value="${escapeHtml(state.memorySearch || "")}" /></div><div class="memory-list">${memories.map((memory) => `<article class="memory-card" data-memory-text="${escapeHtml(memory.text.toLowerCase())}"><div class="memory-text">${escapeHtml(memory.text)}</div><div class="memory-footer"><small>${escapeHtml(memory.updated)}</small><div class="button-row">${button("Edit", "edit-memory", { small: true, tone: "soft", attrs: `data-id="${escapeHtml(memory.id)}"` })}${button("Delete", "delete-memory", { small: true, tone: "danger", attrs: `data-id="${escapeHtml(memory.id)}"` })}</div></div></article>`).join("") || emptyState("No saved memory matched that search.")}</div>`, "section-two-thirds tint-peach")}
      ${card(`${sectionHeading("A note about memory", "This is a demo collection.", "i", "lavender")}<p>Blacki uses the notes it can retrieve when they help answer a question. The real provider may return a limited page, so a page of results is not always the whole collection.</p><div class="error-state">External memory updates may take a moment to show up here. This demo does not connect to a memory provider.</div>`, "section-third tint-lavender")}
    </div>`;
}

function trackersView(state, data) {
  const selectedId = state.selectedTracker || data.trackers[0]?.id;
  const tracker = data.trackers.find((item) => item.id === selectedId) ?? data.trackers[0];
  const rows = tracker.rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("");
  const firstRow = tracker.rows[0] ?? tracker.fields.map(() => "");
  const formFields = tracker.fields.map((field, index) => {
    const value = firstRow[index] ?? "";
    const label = escapeHtml(field);
    const inputId = `tracker-field-${index}`;
    if (/status|mood/i.test(field)) {
      const choices = [...new Set(tracker.rows.map((row) => row[index]).filter(Boolean))];
      return `<div class="field"><label for="${inputId}">${label}</label><select class="select" id="${inputId}" name="field_${index}" data-tracker-field="${index}">${choices.map((choice) => `<option value="${escapeHtml(choice)}" ${choice === value ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select></div>`;
    }
    const numeric = value !== "" && Number.isFinite(Number(value));
    return `<div class="field"><label for="${inputId}">${label}</label><input class="input" id="${inputId}" name="field_${index}" data-tracker-field="${index}" ${numeric ? `type="number" min="0" step="1" value="${Number(value)}"` : `type="text" maxlength="240" value="${escapeHtml(value)}"`} /></div>`;
  }).join("");
  return `${heading("More · trackers", "Your own little tables.", "A typed view for custom records. Demo rows are editable only in the prototype.")}
    <div class="panel-grid">
      ${card(`${sectionHeading("Trackers", "Pick one to inspect its fields.", "▦", "sky")}<div class="tracker-picker">${data.trackers.map((item) => `<button class="tracker-option ${selectedId === item.id ? "active" : ""}" type="button" data-action="choose-tracker" data-id="${escapeHtml(item.id)}" aria-pressed="${selectedId === item.id}"><span class="tracker-badge">▦</span><span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.description)}</small></span></button>`).join("")}</div>`, "section-third tint-sky")}
      ${card(`${sectionHeading(tracker.name, `${tracker.rows.length} sample records`, "▤", "mint")}<div class="filter-row"><label class="visually-hidden" for="tracker-search">Search records</label><input class="input" id="tracker-search" type="search" placeholder="Filter rows" value="${escapeHtml(state.trackerSearch || "")}" /><button class="pill-button" type="button" data-action="sort-tracker">Sort ${state.trackerSort === "desc" ? "↓" : "↑"}</button>${button("Add row", "add-tracker-row", { small: true, tone: "soft" })}</div><div class="table-wrap"><table class="play-table"><caption class="visually-hidden">${escapeHtml(tracker.name)} demo records</caption><thead><tr>${tracker.fields.map((field) => `<th scope="col">${escapeHtml(field)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div><p class="field-note">Existing tables without field labels would stay read-only until configured.</p>`, "section-two-thirds tint-mint")}
      ${card(`${sectionHeading("Edit a row", "A typed form for the selected tracker.", "✎", "butter")}<form id="tracker-form" data-tracker-id="${escapeHtml(tracker.id)}"><div class="tracker-fields">${formFields}</div><div class="form-footer">${button("Save row", "save-tracker")}</div></form>`, "section-third tint-butter")}
    </div>`;
}

function activityView(state, data) {
  return `${heading("More · activity", "See what just happened.", "Progress labels are useful. Private prompts and raw tool details stay out of this view.")}
    <div class="panel-grid">
      ${card(`${sectionHeading("Recent activity", "Safe summaries from this demo.", "◷", "lavender")}<div class="timeline-list">${data.activity.map((item, index) => `<article class="timeline-item" data-depth="${index === 1 ? "1" : "0"}"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.detail)} · ${escapeHtml(item.time)}</small><div class="timeline-badge">${badge(item.status, item.status === "Saved" ? "success" : item.status === "Unavailable" ? "warn" : "info")}</div></article>`).join("")}</div>`, "section-two-thirds tint-lavender")}
      ${card(`${sectionHeading("Usage", "No invented prices.", "◎", "butter")}<div class="cost-card"><span class="cost-number">—</span><div><strong>Cost unavailable</strong><small>No provider price was reported in this demo.</small></div></div><div class="data-row"><div><strong>Tasks completed</strong><small>Sample activity only</small></div><span class="data-value">2</span></div><div class="data-row"><div><strong>Server state</strong><small>Fixture mode · no live runs</small></div>${badge("Demo", "neutral")}</div>`, "section-third tint-butter")}
      ${card(`${sectionHeading("Your private activity stays private", "This is not the operator dashboard.", "!", "peach")}<p>The demo shows only friendly task labels. It never opens Blacki's raw logs, another user's run, provider credentials, prompts, or private tool arguments.</p>`, "section-wide tint-peach")}
    </div>`;
}

function moreView() {
  const links = [
    ["Models", "Choose an exact model ID and a supported reasoning level.", "◌", "models", "lavender"],
    ["Memory", "Review personal notes Blacki can use.", "◉", "memory", "peach"],
    ["Trackers", "Browse structured records in safe typed forms.", "▦", "trackers", "mint"],
    ["Activity", "Check progress and usage without private logs.", "◷", "activity", "butter"],
  ];
  return `${heading("More tools", "A few useful extras.", "Open settings and records directly, without another chat turn.")}
    <div class="panel-grid more-grid">${links.map(([title, text, glyph, screen, colour]) => `<button class="more-tile tint-${colour}" type="button" data-action="navigate" data-screen="${screen}"><span class="card-icon">${glyph}</span><span><strong>${title}</strong><small>${text}</small></span><span class="more-arrow" aria-hidden="true">→</span></button>`).join("")}
      ${card(`${sectionHeading("Compare the experiences", "One set of screens, two ways to chat.", "✳", "lavender")}<p>Switch between Telegram companion and In-app assistant at any time. Your demo records stay the same either way.</p><div class="button-row">${button("Read the trial checklist", "compare", { tone: "soft" })}${button("Reset demo data", "reset", { tone: "soft" })}</div>`, "section-wide tint-lavender")}
    </div>`;
}

function assistantView(state, data) {
  const bubbles = data.chat.map((message) => `<div class="chat-bubble ${message.role === "user" ? "user" : ""}"><span class="chat-origin">${message.role === "user" ? "You · this app" : escapeHtml(message.origin || "Demo response")}</span>${escapeHtml(message.text)}</div>`).join("");
  const lunch = data.meals.find((meal) => meal.id === "meal-lunch");
  return `${heading("In-app assistant", "Chat, then tap to act.", "A demo conversation with the same controls you can open from the other screens.")}
    <div class="panel-grid">
      ${card(`${sectionHeading("Blacki", "Replies below are labelled demo responses.", "✳", "lavender")}<div class="chat-panel"><div class="chat-scroll" id="chat-scroll" role="log" aria-label="Demo conversation" aria-live="polite">${bubbles}</div><form class="chat-compose" id="chat-form"><label class="visually-hidden" for="chat-input">Message Blacki</label><input class="input" id="chat-input" name="message" maxlength="500" placeholder="Ask about your day…" autocomplete="off" required /><button class="play-button" type="submit" aria-label="Send demo message">Send</button></form></div>`, "section-two-thirds tint-lavender")}
      ${card(`${sectionHeading("Tap a result", "No need to describe the edit in another message.", "◒", "butter")}<div class="assistant-result"><div class="item-card-top"><div><h3>Lunch calories</h3><p>${escapeHtml(lunch.title)} · ${Number(lunch.calories)} kcal</p></div>${badge("Demo card", "neutral")}</div>${exactRange({ id: "assistant-calories", label: "Update lunch calories", value: lunch.calories, min: 0, max: 1200, step: 1, unit: "kcal" })}<div class="assistant-card-footer">${button("Save lunch", "save-assistant-meal")}</div></div>`, "section-third tint-butter")}
      ${card(`${sectionHeading("The same tools, either way", "Open a screen from chat or from the menu.", "↗", "mint")}<div class="button-row">${button("Edit a meal", "navigate", { tone: "soft", attrs: 'data-screen="meals"' })}${button("Log a set", "navigate", { tone: "soft", attrs: 'data-screen="training"' })}${button("Move a reminder", "navigate", { tone: "soft", attrs: 'data-screen="reminders"' })}</div>`, "section-wide tint-mint")}
    </div>`;
}

function comparisonDialog(state) {
  return `<div class="dialog-inner"><div class="heading-kicker">A FAIR LITTLE TEST</div><h2 id="decision-title">Try both on the same tasks.</h2><p>Start with fresh demo data, complete the same task in each mode, then reset before you switch the order. Nothing here is sent to analytics.</p><ol class="trial-list"><li class="trial-item"><span class="trial-number">1</span><span><strong>Correct lunch</strong><small>Drag calories, type an exact number, save.</small></span>${badge("Meals", "info")}</li><li class="trial-item"><span class="trial-number">2</span><span><strong>Log three workout sets</strong><small>Change one set and use a short rest timer.</small></span>${badge("Training", "success")}</li><li class="trial-item"><span class="trial-number">3</span><span><strong>Reschedule a reminder</strong><small>Try a new day, then snooze just once.</small></span>${badge("Reminders", "warn")}</li><li class="trial-item"><span class="trial-number">4</span><span><strong>Filter a result</strong><small>Change the chart range and open a record.</small></span>${badge("Results", "neutral")}</li></ol><p class="field-note">The app remembers no preference unless you choose one. You can keep both or decide later.</p><div class="dialog-actions"><button class="play-button soft" type="button" data-action="close-dialog">Back to the app</button><button class="play-button" type="button" data-action="start-meal-trial">Start with a meal</button></div></div>`;
}

function formatView(state, data) {
  switch (state.screen) {
    case "meals": return mealsView(state, data);
    case "training": return workoutView(state, data);
    case "reminders": return reminderView(state, data);
    case "files": return filesView(state, data);
    case "models": return modelsView(state, data);
    case "memory": return memoryView(state, data);
    case "trackers": return trackersView(state, data);
    case "activity": return activityView(state, data);
    case "more": return moreView();
    case "assistant": return assistantView(state, data);
    default: return homeView(state, data);
  }
}

export { comparisonDialog, formatView };
