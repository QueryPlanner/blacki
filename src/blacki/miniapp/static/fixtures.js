export function localDateValue(date = new Date()) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export const createFixtures = () => {
  const today = localDateValue();
  return {
  meals: [
    { id: "meal-breakfast", title: "Oats, banana & peanut butter", type: "breakfast", time: "8:15 AM", calories: 465, protein: 22, carbs: 61, fat: 15, date: today },
    { id: "meal-lunch", title: "Chicken rice bowl", type: "lunch", time: "1:10 PM", calories: 630, protein: 42, carbs: 72, fat: 17, date: today },
    { id: "meal-snack", title: "Greek yoghurt", type: "snack", time: "4:30 PM", calories: 145, protein: null, carbs: null, fat: null, date: today },
  ],
  calorieGoal: 1900,
  goalConfigured: true,
  mealTypes: ["breakfast", "lunch", "dinner", "snack"],
  weekCalories: [1480, 1760, 1320, 1940, 1540, 1760, 1240],
  workouts: [
    { id: "workout-today", title: "Upper body A", status: "In progress", date: "Today", exercises: [
      { id: "exercise-row", name: "Seated cable row", sets: [
        { id: "set-row-1", weight: 35, reps: 12, warmup: true, complete: true },
        { id: "set-row-2", weight: 45, reps: 10, warmup: false, complete: false },
        { id: "set-row-3", weight: 45, reps: 10, warmup: false, complete: false },
      ] },
      { id: "exercise-press", name: "Incline dumbbell press", sets: [
        { id: "set-press-1", weight: 16, reps: 10, warmup: false, complete: false },
        { id: "set-press-2", weight: 16, reps: 10, warmup: false, complete: false },
      ] },
    ] },
    { id: "workout-previous", title: "Upper body A", status: "Completed", date: "Monday", exercises: [] },
  ],
  restSeconds: 90,
  restDeadline: null,
  reminder: { id: "reminder-water", title: "Take a little walk", date: today, time: "18:30", timezone: "Asia/Kolkata", repeat: [1, 3, 5], enabled: true, snoozedUntil: null },
  files: [
    { id: "file-notes", name: "weekend-notes.txt", kind: "Text", mime: "text/plain", size: 2480, available: true, expires: "Kept in your files" },
    { id: "file-receipt", name: "lunch-receipt.jpg", kind: "Image", mime: "image/jpeg", size: 184320, available: true, expires: "Kept in your files" },
    { id: "file-plan", name: "upper-body-plan.pdf", kind: "Document", mime: "application/pdf", size: 492100, available: false, expires: "Storage unavailable in demo" },
  ],
  fileFilter: "All files",
  fileLayout: "list",
  selectedFiles: [],
  models: [
    { id: "google/gemini-2.5-flash", label: "Gemini 2.5 Flash", provider: "Google", reasoning: ["default", "low", "medium", "high"], favorite: true },
    { id: "openai/gpt-5-mini", label: "GPT-5 mini", provider: "OpenAI", reasoning: ["default", "low", "medium", "high"], favorite: false },
    { id: "anthropic/claude-sonnet-4", label: "Claude Sonnet", provider: "Anthropic", reasoning: ["default", "low", "medium"], favorite: false },
  ],
  activeModel: "google/gemini-2.5-flash",
  reasoning: "default",
  memories: [
    { id: "memory-gym", text: "I usually train at the gym five or six days a week.", updated: "Updated recently" },
    { id: "memory-food", text: "I prefer meal suggestions with plenty of protein.", updated: "Saved earlier" },
    { id: "memory-coffee", text: "I like unsweetened iced coffee and matcha.", updated: "Saved earlier" },
  ],
  trackers: [
    { id: "tracker-reading", name: "Reading log", description: "Books and notes", fields: ["Book", "Status", "Pages"], rows: [["The Design of Everyday Things", "Reading", "146"], ["The Pragmatic Programmer", "Next", "0"], ["Atomic Habits", "Finished", "320"]] },
    { id: "tracker-mood", name: "Mood check-ins", description: "A small daily pulse", fields: ["Day", "Mood", "Note"], rows: [["Monday", "Good", "Got outside"], ["Tuesday", "Okay", "Busy afternoon"]] },
  ],
  activity: [
    { title: "Meal totals refreshed", detail: "Your lunch edit is reflected in today's total.", time: "Just now", status: "Saved" },
    { title: "Workout set logged", detail: "One set is complete. Two are ready to edit.", time: "12 min ago", status: "In progress" },
    { title: "Health sync", detail: "No provider status is connected in this demo.", time: "Today", status: "Unavailable" },
  ],
  chartRange: "Week",
  chat: [
    { role: "assistant", text: "You have logged 1,240 calories so far. Want to adjust lunch or add a snack?", origin: "Demo response" },
  ],
  saved: {},
  };
};

export class FixtureAdapter {
  constructor() { this.reset(); }

  reset() {
    this.data = createFixtures();
    return this.data;
  }

  meal(id) { return this.data.meals.find((meal) => meal.id === id) ?? null; }

  updateMeal(id, patch) {
    const meal = this.meal(id);
    if (!meal) return false;
    Object.assign(meal, patch);
    return true;
  }

  updateSet(exerciseId, setId, patch) {
    for (const workout of this.data.workouts) {
      const exercise = workout.exercises.find((item) => item.id === exerciseId);
      const set = exercise?.sets.find((item) => item.id === setId);
      if (set) { Object.assign(set, patch); return true; }
    }
    return false;
  }

  updateReminder(patch) {
    Object.assign(this.data.reminder, patch);
  }
}
