// Renders compiled Blacki UI (IR) into DOM. Text is always set as text; styles
// only through the CSSOM (no inline style attributes, no HTML strings).
import { evaluate, formatValue, initialVars } from "./expr.js";

export function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "css") Object.entries(value).forEach(([k, v]) => (k.startsWith("--") ? el.style.setProperty(k, v) : (el.style[k] = v)));
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "text") el.textContent = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  kids.flat().forEach((k) => k && el.append(k));
  return el;
}

// Only **bold** formatting, built from text nodes.
function rich(text) {
  return String(text).split(/(\*\*[^*]+\*\*)/).map((part) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? h("strong", {}, part.slice(2, -2)) : part);
}

const D = (s) => new Date(String(s).length === 10 ? `${s}T12:00:00` : String(s).replace(" ", "T"));
export const shortDate = (s) => { const d = D(s); return Number.isNaN(d.getTime()) ? String(s ?? "") : d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }); };
const weekday = (s) => D(s).toLocaleDateString("en-GB", { weekday: "short" });
const hm = (m) => (m == null ? "–" : `${Math.floor(m / 60)}:${String(Math.round(m % 60)).padStart(2, "0")}`);
const fmtCell = (fmt, v) => (v == null || v === "" ? "–" : fmt === "date" ? shortDate(v) : fmt === "minutes" ? hm(v) : fmt === "number" ? formatValue(v) : String(v));
const looksDate = (v) => typeof v === "string" && /^\d{4}-\d{2}-\d{2}/.test(v);
const label = (v) => (looksDate(v) ? shortDate(v) : formatValue(v));

const STATUS_TEXT = {
  not_connected: "Connect Google Health in Telegram with /connect_health to see this.",
  archive_consent_required: "Reconnect Google Health in Telegram to allow the archive.",
  unavailable: "This data isn't available right now.",
  error: "This data couldn't be loaded.",
};

export class RenderContext {
  constructor({ ir, data, kind, pageId, cardId, api, ui }) {
    this.ir = ir;
    this.kind = kind;
    this.pageId = pageId;
    this.cardId = cardId;
    this.api = api;
    this.ui = ui;
    this.env = { data: data || {}, vars: {} };
    this.env.vars = initialVars(ir.vars, this.env);
    this.watchers = [];
  }
  watch(fn) { this.watchers.push(fn); fn(); }
  set(name, value) { this.env.vars[name] = value; this.watchers.forEach((fn) => fn()); }
  value(node, row) { return evaluate(node, this.env, row); }
  rows(q) { const r = this.env.data[q]; return r && Array.isArray(r.rows) ? r.rows : []; }
  status(q) { const r = this.env.data[q]; return r ? r.status : "error"; }
  get builtin() { return this.kind === "builtin"; }
}

export function renderPage(ctx) {
  return render(ctx.ir.root, ctx);
}

function render(node, ctx) {
  const fn = PARTS[node.type];
  if (!fn) return h("p", { class: "note" }, `This part isn't supported yet: ${node.type}`);
  try {
    return fn(node.props || {}, ctx);
  } catch (err) {
    console.error(err);
    return h("p", { class: "note" }, "This part couldn't be shown.");
  }
}
const kids = (list, ctx) => (list || []).map((n) => render(n, ctx));

function dataNote(ctx, q) {
  const status = ctx.status(q);
  if (status === "ok") return h("p", { class: "note" }, "Nothing here yet for this period.");
  const msg = ctx.env.data[q]?.error || STATUS_TEXT[status] || STATUS_TEXT.error;
  return h("div", { class: "r-callout" }, msg);
}

const toneClass = { hot: "hot", inverse: "inv" };
function figbox(tone) {
  const box = h("div", { class: `figbox ${toneClass[tone] || ""}` });
  if (["info", "success", "warning"].includes(tone)) {
    box.style.background = `var(--color-${tone})`;
    box.style.color = `var(--color-${tone}-content)`;
  }
  return box;
}

async function runAction(action, ctx, button) {
  const args = {};
  for (const [key, value] of Object.entries(action.args || {})) {
    if (value && typeof value === "object" && "k" in value) args[key] = ctx.value(value);
    else if (value && typeof value === "object" && !Array.isArray(value)) {
      args[key] = Object.fromEntries(Object.entries(value).map(([k, v]) => [k, v && typeof v === "object" && "k" in v ? ctx.value(v) : v]));
    } else args[key] = value;
  }
  switch (action.a) {
    case "Set": ctx.set(args.var, args.value); return;
    case "Navigate": ctx.ui.navigate(args.page); return;
    case "AskBlacki": {
      const ok = await ctx.ui.confirm(`Send this to Blacki in Telegram?\n\n“${args.text}”`);
      if (!ok) return;
      break;
    }
    case "Pin": args.pinned = true; break;
    default: break;
  }
  if (button) button.disabled = true;
  try {
    const result = await ctx.api.action({ action: action.a, page: ctx.pageId, card: ctx.cardId || "", args });
    ctx.ui.toast(result.message || "Done");
    ctx.ui.haptic("success");
  } catch (err) {
    ctx.ui.toast(err.message || "That didn't work");
    ctx.ui.haptic("error");
  } finally {
    if (button) button.disabled = false;
  }
}

function bars(rows, { x, y, height = 180, highlight, ctx, onPick }) {
  const box = h("div", { class: "kb" });
  box.style.height = `${height}px`;
  const values = rows.map((r) => (typeof r[y] === "number" ? r[y] : 0));
  const max = Math.max(...values, 1);
  const els = rows.map((row, i) => {
    const el = h(onPick ? "button" : "i", { title: `${label(row[x])}: ${formatValue(row[y])}`, type: onPick ? "button" : null });
    el.style.height = `${Math.max(2, (values[i] / max) * 100)}%`;
    if (onPick) el.addEventListener("click", () => onPick(i));
    box.append(el);
    return el;
  });
  if (highlight) ctx.watch(() => els.forEach((el, i) => el.classList.toggle("hi", !!ctx.value(highlight, rows[i]))));
  else els.forEach((el) => el.classList.add("hi"));
  return box;
}
const axis = (rows, x) => h("div", { class: "axis" }, h("span", {}, label(rows[0]?.[x])), h("span", {}, label(rows[rows.length - 1]?.[x])));

const PARTS = {
  Page: (p, ctx) => h("div", { class: "page" },
    h("div", { class: "page-head wrap" }, h("h1", {}, p.title), p.subtitle && h("p", {}, p.subtitle)),
    h("div", { class: "wrap page-body" }, kids(p.children, ctx))),
  Section: (p, ctx) => h("section", { class: "r-section" }, h("h2", {}, p.title), kids(p.children, ctx)),
  Stack: (p, ctx) => h("div", { class: "r-stack" }, kids(p.children, ctx)),
  Grid: (p, ctx) => { const g = h("div", { class: `r-grid cols-${p.columns || 2}` }, kids(p.children, ctx)); return g; },
  Tabs: (p, ctx) => {
    const panels = p.tabs.map((t) => h("div", { class: "r-stack tab-panel" }, kids(t.props.children, ctx)));
    const bar = h("div", { class: "tabs", role: "tablist" });
    const show = (i) => { panels.forEach((el, j) => { el.hidden = i !== j; }); [...bar.children].forEach((b, j) => b.setAttribute("aria-selected", String(i === j))); };
    p.tabs.forEach((t, i) => bar.append(h("button", { role: "tab", type: "button", onclick: () => show(i) }, t.props.label)));
    show(0);
    return h("div", { class: "r-tabs" }, bar, panels);
  },
  Tab: (p, ctx) => h("div", { class: "r-stack" }, kids(p.children, ctx)),
  Header: (p) => h("div", { class: "r-header" }, h("h3", {}, p.title), p.text && h("p", { class: "sub" }, p.text)),
  Text: (p) => h("p", { class: "r-text" }, rich(p.text)),
  Callout: (p) => h("div", { class: `r-callout ${p.tone === "info" ? "" : p.tone}` }, rich(p.text)),
  Empty: (p, ctx) => h("div", { class: "r-empty" }, h("b", {}, p.title), h("p", {}, p.text), p.action && render(p.action, ctx)),
  Steps: (p) => h("ol", { class: "r-steps" }, p.items.map((t) => h("li", {}, t))),
  Compare: (p, ctx) => h("div", { class: "r-compare" }, kids(p.sides, ctx)),
  Side: (p) => h("div", { class: "r-side" }, h("b", {}, p.title), h("ul", {}, p.items.map((t) => h("li", {}, t)))),
  Stat: (p, ctx) => {
    const value = h("b");
    const box = figbox(p.tone);
    box.append(value, h("span", {}, p.label));
    if (p.note) box.append(h("span", { class: "fig-note" }, p.note));
    ctx.watch(() => {
      const v = ctx.value(p.value);
      value.textContent = formatValue(v);
      if (p.unit && v != null) value.append(h("small", {}, p.unit));
    });
    return box;
  },
  StatRow: (p, ctx) => h("div", { class: "figs" }, kids(p.stats, ctx)),
  KeyBars: (p, ctx) => {
    const rows = ctx.rows(p.data);
    if (!rows.length) return dataNote(ctx, p.data);
    return h("div", { class: "chart" }, bars(rows, { x: p.x, y: p.y, height: p.height, highlight: p.highlight, ctx }), axis(rows, p.x));
  },
  StackedBars: (p, ctx) => {
    const rows = ctx.rows(p.data);
    if (!rows.length) return dataNote(ctx, p.data);
    const series = p.series.map((s) => s.props);
    return stacked(rows, p.x, series);
  },
  Line: (p, ctx) => {
    const rows = ctx.rows(p.data).filter((r) => typeof r[p.y] === "number");
    if (!rows.length) return dataNote(ctx, p.data);
    const vals = rows.map((r) => r[p.y]);
    const lo = p.min ?? Math.min(...vals);
    const hi = p.max ?? Math.max(...vals);
    const span = hi - lo || 1;
    const pts = vals.map((v, i) => `${(i / Math.max(1, vals.length - 1)) * 100},${100 - ((v - lo) / span) * 100}`);
    const last = pts[pts.length - 1].split(",");
    const chart = svg("svg", { class: "line", viewBox: "0 0 100 100", preserveAspectRatio: "none", role: "img", "aria-label": `${vals.length} values, latest ${formatValue(vals[vals.length - 1])}` },
      svg("polyline", { points: pts.join(" ") }), svg("circle", { cx: last[0], cy: last[1], r: 1.6 }));
    return h("div", { class: "chart" }, h("div", { class: "line-range" }, h("span", {}, formatValue(hi)), h("span", {}, formatValue(lo))), chart, axis(rows, p.x));
  },
  Ring: (p, ctx) => {
    const r = 48;
    const c = 2 * Math.PI * r;
    const val = svg("circle", { class: "val", cx: 60, cy: 60, r });
    const text = svg("text", { x: 60, y: 60 });
    const ring = svg("svg", { class: "ring", viewBox: "0 0 120 120", role: "img" }, svg("circle", { class: "track", cx: 60, cy: 60, r }), val, text);
    ctx.watch(() => {
      const v = Math.max(0, Math.min(100, Number(ctx.value(p.value)) || 0));
      val.setAttribute("stroke-dasharray", `${(v / 100) * c} ${c}`);
      text.textContent = `${Math.round(v)}%`;
      ring.setAttribute("aria-label", `${Math.round(v)}% ${p.label}`);
    });
    return h("div", { class: "r-ring" }, ring, h("span", {}, p.label));
  },
  RangeBar: (p, ctx) => {
    const span = p.max - p.min || 1;
    const band = h("i", { class: "band" });
    band.style.left = `${((p.low - p.min) / span) * 100}%`;
    band.style.width = `${((p.high - p.low) / span) * 100}%`;
    const mark = h("i", { class: "mark" });
    const value = h("b");
    const note = h("span", { class: "sub" });
    ctx.watch(() => {
      const v = ctx.value(p.value);
      value.textContent = v == null ? "–" : `${formatValue(v)}${p.unit ? ` ${p.unit}` : ""}`;
      mark.hidden = v == null;
      if (v != null) mark.style.left = `${Math.max(0, Math.min(100, ((v - p.min) / span) * 100))}%`;
      note.textContent = v == null ? "No reading" : v < p.low ? "Below the usual range" : v > p.high ? "Above the usual range" : "Within the usual range";
    });
    return h("div", { class: "r-range" }, h("div", { class: "r-range-top" }, h("span", {}, p.label), value),
      h("div", { class: "track" }, band, mark), h("div", { class: "axis" }, h("span", {}, `${p.min}`), note, h("span", {}, `${p.max}`)));
  },
  Table: (p, ctx) => {
    const rows = ctx.rows(p.data).slice(-(p.limit || 7)).reverse();
    if (!rows.length) return dataNote(ctx, p.data);
    const cols = p.columns.map((c) => c.props);
    return h("div", { class: "table-wrap" }, h("table", { class: "r-table" },
      h("thead", {}, h("tr", {}, cols.map((c) => h("th", { class: ["number", "minutes"].includes(c.format) ? "n" : "" }, c.label)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, cols.map((c) => h("td", { class: ["number", "minutes"].includes(c.format) ? "n" : "" }, fmtCell(c.format, r[c.field]))))))));
  },
  List: (p, ctx) => {
    const rows = ctx.rows(p.data).slice(-(p.limit || 6)).reverse();
    if (!rows.length) return dataNote(ctx, p.data);
    const values = rows.map((r) => (typeof r[p.value] === "number" ? r[p.value] : 0));
    const max = Math.max(...values, 1);
    return h("div", { class: "rows" }, rows.map((r, i) => {
      const key = h("i", { class: "key" });
      key.style.height = `${16 + (values[i] / max) * 36}px`;
      return h("div", { class: "row-item" }, key,
        h("div", {}, h("b", {}, label(r[p.title])), p.subtitle && h("small", {}, label(r[p.subtitle]))),
        p.value ? h("span", { class: "num" }, `${formatValue(r[p.value])}${p.unit ? ` ${p.unit}` : ""}`) : h("span"));
    }));
  },
  Slider: (p, ctx) => {
    const out = h("output");
    const input = h("input", { type: "range", min: p.min, max: p.max, step: p.step || 1, "aria-label": p.label });
    input.value = ctx.env.vars[p.var];
    const paint = () => { out.textContent = `${formatValue(Number(input.value))}${p.unit ? ` ${p.unit}` : ""}`; };
    input.addEventListener("input", () => { paint(); ctx.set(p.var, Number(input.value)); });
    paint();
    return h("div", { class: "slider" }, h("label", {}, h("span", {}, p.label), out), input);
  },
  Choice: (p, ctx) => {
    const row = h("div", { class: "chips" });
    const paint = () => [...row.children].forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === String(ctx.env.vars[p.var]))));
    p.options.forEach((o) => row.append(h("button", { type: "button", "data-v": o, onclick: () => { ctx.set(p.var, o); paint(); } }, o)));
    paint();
    return row;
  },
  Toggle: (p, ctx) => {
    const input = h("input", { type: "checkbox", role: "switch" });
    input.checked = !!ctx.env.vars[p.var];
    input.addEventListener("change", () => ctx.set(p.var, input.checked));
    return h("label", { class: "r-toggle" }, p.label, input);
  },
  Stepper: (p, ctx) => {
    const out = h("output");
    const paint = () => { out.textContent = `${formatValue(ctx.env.vars[p.var])}${p.unit ? ` ${p.unit}` : ""}`; };
    const step = (d) => { ctx.set(p.var, Math.max(0, (Number(ctx.env.vars[p.var]) || 0) + d)); paint(); };
    paint();
    return h("div", { class: "r-stepper" }, h("b", {}, p.label), h("div", { class: "step" },
      h("button", { type: "button", "aria-label": `Less ${p.label}`, onclick: () => step(-(p.step || 1)) }, "−"), out,
      h("button", { type: "button", "aria-label": `More ${p.label}`, onclick: () => step(p.step || 1) }, "+")));
  },
  Button: (p, ctx) => {
    const btn = h("button", { class: `btn ${p.style === "quiet" ? "quiet" : ""}`, type: "button" }, p.label);
    btn.addEventListener("click", () => runAction(p.action, ctx, btn));
    return btn;
  },
  Link: (p, ctx) => h("a", { class: "r-link", href: `#page/${p.page}`, onclick: (e) => { e.preventDefault(); ctx.ui.navigate(p.page); } }, p.label),
  OctaveWeek: (p, ctx) => octave(ctx.rows(p.data), ctx, p.data),
  SleepStages: (p, ctx) => sleepStages(ctx.rows(p.data), ctx, p.data),
  MealLog: (p, ctx) => mealLog(p, ctx),
  SetLogger: (p, ctx) => setLogger(p, ctx),
  RestTimer: (p, ctx) => restTimer(p, ctx),
  FileList: (p, ctx) => fileList(p, ctx),
  CardFeed: (p, ctx) => cardFeed(p, ctx),
  PinnedCards: (p, ctx) => pinnedCards(ctx),
};

function stacked(rows, x, series, onPick) {
  const colours = ["var(--accent)", "var(--color-secondary)", "var(--surface-2)", "color-mix(in oklab, var(--ink) 35%, transparent)"];
  const total = (r) => series.reduce((t, s) => t + (Number(r[s.field]) || 0), 0);
  const max = Math.max(...rows.map(total), 1);
  const chart = h("div", { class: "kb" });
  chart.style.height = "220px";
  const cols = rows.map((r, i) => {
    const col = h(onPick ? "button" : "div", { class: "stack-col", type: onPick ? "button" : null, title: label(r[x]) });
    col.style.height = `${(total(r) / max) * 100}%`;
    series.forEach((s, j) => { const seg = h("i"); seg.style.flex = String(Number(r[s.field]) || 0); seg.style.background = colours[j]; col.append(seg); });
    if (onPick) col.addEventListener("click", () => onPick(i));
    chart.append(col);
    return col;
  });
  const legend = h("div", { class: "legend" }, series.map((s, j) => { const sw = h("i"); sw.style.background = colours[j]; return h("span", {}, sw, s.label); }));
  const wrap = h("div", { class: "chart" }, chart, axis(rows, x), legend);
  wrap.cols = cols;
  return wrap;
}

function octave(rows, ctx, q) {
  const week = rows.slice(-7);
  if (!week.length) return dataNote(ctx, q);
  const title = h("h2");
  const readout = h("div", { class: "readout" });
  const keys = h("div", { class: "octave", role: "group", "aria-label": "Pick a day" });
  const lineFor = (d) => {
    const s = d.asleep ?? 0;
    if (!d.asleep) return "no sleep recorded";
    if (s >= 480) return "a full night";
    if (s < 330) return "a short night";
    return "an ordinary night";
  };
  const pick = (i) => {
    [...keys.querySelectorAll(".white")].forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)));
    const d = week[i];
    title.textContent = `${D(d.date).toLocaleDateString("en-GB", { weekday: "long" })}, ${lineFor(d)}`;
    readout.replaceChildren(...[[hm(d.asleep), "asleep"], [d.deep != null ? `${d.deep} min` : "–", "deep"], [d.hrv != null ? `${Math.round(d.hrv)} ms` : "–", "HRV"], [d.resting_hr ?? "–", "resting bpm"], [d.steps != null ? d.steps.toLocaleString("en-GB") : "–", "steps"]]
      .map(([v, k]) => h("div", {}, h("b", {}, String(v)), k)));
    ctx.ui.haptic("select");
  };
  week.forEach((d, i) => {
    const fill = h("i", { class: "fill" });
    fill.style.height = `${Math.min(1, (d.asleep || 0) / 540) * 100}%`;
    keys.append(h("button", { class: "white key", type: "button", "aria-label": `${weekday(d.date)} ${shortDate(d.date)}`, onclick: () => pick(i) },
      fill, h("span", { class: "d" }, weekday(d.date)[0]), h("span", { class: "d" }, String(D(d.date).getDate()))));
  });
  [0, 1, 3, 4, 5].forEach((slot) => { const k = h("i", { class: "black key" }); k.style.setProperty("--slot", String(slot + 1)); keys.append(k); });
  let latest = week.length - 1;
  while (latest > 0 && !week[latest].asleep && week[latest].hrv == null) latest -= 1;
  pick(latest);
  return h("div", { class: "stage" }, title, keys, readout, h("p", { class: "stage-note" }, "Grey on each key is hours asleep, out of 9."));
}

function sleepStages(rows, ctx, q) {
  if (!rows.length) return dataNote(ctx, q);
  const detail = h("div", { class: "figs" });
  const series = [{ field: "deep", label: "Deep" }, { field: "rem", label: "REM" }, { field: "light", label: "Light" }, { field: "awake", label: "Awake" }];
  let chart;
  const pick = (i) => {
    chart.cols.forEach((c, j) => c.setAttribute("aria-pressed", String(i === j)));
    const n = rows[i];
    detail.replaceChildren(...[[hm(n.asleep), `Asleep, ${weekday(n.date)} ${shortDate(n.date)}`, "hot"], [n.bed, `In bed, up at ${n.wake}`], [`${n.deep} min`, "Deep"], [`${n.rem} min`, "REM"]]
      .map(([v, l, tone]) => { const b = figbox(tone); b.append(h("b", {}, String(v)), h("span", {}, l)); return b; }));
  };
  chart = stacked(rows, "date", series, pick);
  pick(rows.length - 1);
  return h("div", { class: "r-stack" }, chart, detail);
}

function mealLog(p, ctx) {
  const rows = ctx.rows(p.data);
  if (!rows.length) return dataNote(ctx, p.data);
  const byDay = {};
  rows.forEach((m) => (byDay[m.date] ||= []).push(m));
  const out = h("div", { class: "r-stack" });
  Object.keys(byDay).sort().reverse().forEach((day) => {
    const meals = byDay[day];
    const kcal = meals.reduce((t, m) => t + (m.kcal || 0), 0);
    const protein = Math.round(meals.reduce((t, m) => t + (m.protein || 0), 0));
    out.append(h("div", { class: "daylog" }, h("h3", {}, `${weekday(day)} ${shortDate(day)}`, h("span", {}, `${kcal} kcal, ${protein} g protein`)),
      h("div", { class: "rows" }, meals.map((m) => mealRow(m, ctx)))));
  });
  return out;
}

function mealRow(m, ctx) {
  const key = h("i", { class: "key" });
  key.style.height = `${Math.max(18, Math.min(56, (m.kcal || 0) / 14))}px`;
  const name = String(m.name || "Meal");
  const short = name.length > 80 ? `${name.slice(0, name.lastIndexOf(" ", 80))}…` : name;
  const from = m.source === "google_health" ? "from Google Health" : null;
  const row = h("div", { class: "row-item" }, key,
    h("div", {}, h("b", {}, short), h("small", {}, [m.time, m.protein != null ? `${Math.round(m.protein)} g protein` : null, from].filter(Boolean).join(", "))),
    h("span", { class: "num" }, `${m.kcal} kcal`));
  if (!ctx.builtin || m.id == null) return row;
  const editor = h("div", { class: "meal-edit", hidden: true });
  const kcal = h("input", { type: "number", min: 0, max: 10000, inputmode: "numeric", "aria-label": "Calories" });
  const protein = h("input", { type: "number", min: 0, max: 1000, step: "0.1", inputmode: "decimal", "aria-label": "Protein grams" });
  kcal.value = m.kcal ?? "";
  protein.value = m.protein ?? "";
  const save = h("button", { class: "btn", type: "button" }, "Save");
  save.addEventListener("click", async () => {
    const changes = {};
    if (kcal.value !== "" && Number(kcal.value) !== m.kcal) changes.kcal = Math.round(Number(kcal.value));
    if (protein.value !== "" && Number(protein.value) !== m.protein) changes.protein = Number(protein.value);
    if (!Object.keys(changes).length) { editor.hidden = true; return; }
    save.disabled = true;
    try {
      const r = await ctx.api.action({ action: "meal.edit", page: ctx.pageId, args: { id: m.id, changes } });
      ctx.ui.toast(r.message);
      ctx.ui.reload();
    } catch (err) { ctx.ui.toast(err.message); } finally { save.disabled = false; }
  });
  const del = h("button", { class: "btn quiet danger", type: "button" }, "Delete");
  del.addEventListener("click", async () => {
    if (!(await ctx.ui.confirm(`Delete “${short}” (${m.kcal} kcal) from ${shortDate(m.date)}? This can't be undone.`))) return;
    try {
      const r = await ctx.api.action({ action: "meal.delete", page: ctx.pageId, args: { id: m.id } });
      ctx.ui.toast(r.message);
      ctx.ui.reload();
    } catch (err) { ctx.ui.toast(err.message); }
  });
  editor.append(h("label", {}, "kcal", kcal), h("label", {}, "protein g", protein), h("div", { class: "row-actions" }, save, del));
  const toggle = h("button", { class: "chip-btn", type: "button", "aria-expanded": "false" }, "Edit");
  toggle.addEventListener("click", () => { editor.hidden = !editor.hidden; toggle.setAttribute("aria-expanded", String(!editor.hidden)); });
  return h("div", { class: "meal-wrap" }, row, h("div", { class: "row-tools" }, toggle), editor);
}

function stepper(value, unit, delta, onSet) {
  const out = h("output", {}, String(value), h("small", {}, unit));
  const set = (d) => { value = Math.max(0, +(value + d).toFixed(1)); out.firstChild.textContent = String(value); onSet(value); };
  return h("div", { class: "step" }, h("button", { type: "button", "aria-label": `Less ${unit}`, onclick: () => set(-delta) }, "−"), out, h("button", { type: "button", "aria-label": `More ${unit}`, onclick: () => set(delta) }, "+"));
}

function setLogger(p, ctx) {
  const sets = p.sets.map(([kg, reps]) => ({ kg, reps, done: false }));
  const name = h("input", { class: "ex-name", type: "text", maxlength: 60, "aria-label": "Exercise" });
  name.value = p.exercise;
  const list = h("div");
  const paint = () => {
    list.replaceChildren(...sets.map((s, i) => {
      const done = h("button", { class: "done", type: "button", "aria-pressed": String(s.done) }, "Done");
      done.addEventListener("click", () => { s.done = !s.done; done.setAttribute("aria-pressed", String(s.done)); if (s.done) ctx.ui.emit("rest"); ctx.ui.haptic("select"); });
      return h("div", { class: "set" }, h("span", {}, String(i + 1)), stepper(s.kg, "kg", 2.5, (v) => { s.kg = v; }), stepper(s.reps, "reps", 1, (v) => { s.reps = Math.max(1, Math.round(v)); }), done);
    }));
  };
  paint();
  const add = h("button", { class: "chip-btn", type: "button" }, "Add set");
  add.addEventListener("click", () => { if (sets.length < 10) { const last = sets[sets.length - 1] || { kg: 20, reps: 8 }; sets.push({ kg: last.kg, reps: last.reps, done: false }); paint(); } });
  const save = h("button", { class: "btn", type: "button" }, "Save workout");
  save.addEventListener("click", async () => {
    const done = sets.filter((s) => s.done);
    if (!done.length) { ctx.ui.toast("Tick Done on the sets you finished first"); return; }
    if (!ctx.builtin) { ctx.ui.toast("Saving workouts works on the Training page"); return; }
    save.disabled = true;
    try {
      const r = await ctx.api.action({ action: "workout.save", page: ctx.pageId, args: { exercise: name.value.trim(), sets: done.map((s) => ({ kg: s.kg, reps: s.reps })) } });
      ctx.ui.toast(r.message);
      sets.forEach((s) => { s.done = false; });
      paint();
    } catch (err) { ctx.ui.toast(err.message); } finally { save.disabled = false; }
  });
  return h("div", { class: "ex" }, h("div", { class: "ex-head" }, name, h("span", {}, "Sets save only when you tap Save")), list, h("div", { class: "row-actions" }, add, save));
}

function restTimer(p, ctx) {
  let length = p.seconds;
  let left = 0;
  let timer = null;
  const clock = h("b");
  const paint = () => { const s = left || length; clock.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
  const start = () => {
    clearInterval(timer);
    left = length;
    paint();
    timer = setInterval(() => { left -= 1; paint(); if (left <= 0) { clearInterval(timer); left = 0; paint(); ctx.ui.toast("Rest over. Next set."); ctx.ui.haptic("warning"); } }, 1000);
  };
  ctx.ui.on("rest", start);
  const out = h("output");
  const input = h("input", { type: "range", min: 30, max: 300, step: 15, "aria-label": "Rest length" });
  input.value = length;
  const paintLen = () => { out.textContent = `${length} s`; };
  input.addEventListener("input", () => { length = Number(input.value); paintLen(); if (!left) paint(); });
  paint();
  paintLen();
  return h("div", { class: "rest" }, h("span", {}, "Rest timer, starts when you tick a set"), clock,
    h("div", { class: "slider" }, h("label", {}, h("span", {}, "Rest length"), out), input),
    h("button", { class: "btn quiet", type: "button", onclick: start }, "Start now"));
}

function fileList(p, ctx) {
  const all = ctx.rows(p.data);
  if (!all.length) return ctx.status(p.data) === "ok" ? h("div", { class: "r-empty" }, h("b", {}, "No files yet"), h("p", {}, "Send a photo, PDF or voice note to Blacki in Telegram and it lands here.")) : dataNote(ctx, p.data);
  const kinds = { all: () => true, images: (f) => String(f.mime || "").startsWith("image/"), documents: (f) => /pdf|text|word|sheet|document/.test(String(f.mime || "")), audio: (f) => String(f.mime || "").startsWith("audio/") };
  let kind = p.kind || "all";
  let q = "";
  const size = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
  const list = h("div", { class: "rows" });
  const paint = () => {
    const shown = all.filter(kinds[kind]).filter((f) => String(f.name).toLowerCase().includes(q));
    if (!shown.length) { list.replaceChildren(h("p", { class: "note" }, "No files match.")); return; }
    list.replaceChildren(...shown.map((f) => {
      const del = h("button", { class: "chip-btn", type: "button" }, "Delete");
      del.addEventListener("click", async () => {
        if (!ctx.builtin) { ctx.ui.toast("Delete files from the Files page"); return; }
        if (!(await ctx.ui.confirm(`Delete “${f.name}” (${size(f.size)}) for good? This removes the stored copy.`))) return;
        try { const r = await ctx.api.action({ action: "file.delete", page: ctx.pageId, args: { id: f.id } }); ctx.ui.toast(r.message); ctx.ui.reload(); } catch (err) { ctx.ui.toast(err.message); }
      });
      const key = h("i", { class: "key" });
      key.style.height = "40px";
      return h("div", { class: "row-item" }, key, h("div", {}, h("b", {}, f.name), h("small", {}, `${f.mime || f.kind}, ${size(f.size)}, sent ${shortDate(String(f.uploaded).slice(0, 10))}`)), ctx.builtin ? del : h("span"));
    }));
  };
  const chips = h("div", { class: "chips" }, Object.keys(kinds).map((k) => h("button", { type: "button", "aria-pressed": String(k === kind), onclick: (e) => { kind = k; [...chips.children].forEach((c) => c.setAttribute("aria-pressed", String(c === e.currentTarget))); paint(); } }, k[0].toUpperCase() + k.slice(1))));
  const search = h("input", { class: "search", type: "search", placeholder: "Search your files", "aria-label": "Search your files" });
  search.addEventListener("input", () => { q = search.value.toLowerCase(); paint(); });
  paint();
  const total = all.reduce((t, f) => t + (f.size || 0), 0);
  const a = figbox("hot"); a.append(h("b", {}, String(all.length)), h("span", {}, all.length === 1 ? "File kept" : "Files kept"));
  const b = figbox(); b.append(h("b", {}, size(total)), h("span", {}, "Space used"));
  return h("div", { class: "r-stack" }, h("div", { class: "figs" }, a, b), search, chips, list);
}

function cardFeed(p, ctx) {
  const cards = ctx.rows(p.data);
  if (!cards.length) return h("div", { class: "r-empty" }, h("b", {}, "No cards yet"), h("p", {}, "Ask Blacki something in Telegram, like “how's my sleep this fortnight?”, and the answer can come back as a card."));
  return h("div", { class: "rows" }, cards.map((c) => {
    const open = h("button", { class: "chip-btn", type: "button", onclick: () => ctx.ui.openCard(c.id) }, "Open");
    const remove = h("button", { class: "chip-btn", type: "button" }, "Remove");
    remove.addEventListener("click", async () => {
      if (!(await ctx.ui.confirm(`Remove the card “${c.title}”?`))) return;
      try { await ctx.api.action({ action: "card.delete", page: ctx.pageId, args: { id: c.id } }); ctx.ui.reload(); } catch (err) { ctx.ui.toast(err.message); }
    });
    const key = h("i", { class: "key" });
    key.style.height = "36px";
    if (c.pinned) key.classList.add("pinned");
    return h("div", { class: "row-item" }, key, h("div", {}, h("b", {}, c.title), h("small", {}, `${c.pinned ? "Pinned, " : ""}${shortDate(String(c.created).slice(0, 10))}`)), h("div", { class: "chips no-gap" }, open, remove));
  }));
}

function pinnedCards(ctx) {
  const cards = ctx.env.data._pinned_cards || [];
  if (!cards.length) return h("div");
  return h("section", { class: "r-section" }, h("h2", {}, "Pinned"), cards.map((c) => {
    const sub = new RenderContext({ ir: c.ir, data: c.data, kind: "card", pageId: "", cardId: c.id, api: ctx.api, ui: ctx.ui });
    const unpin = h("button", { class: "chip-btn", type: "button" }, "Unpin");
    unpin.addEventListener("click", async () => { try { await ctx.api.action({ action: "Pin", card: c.id, args: { pinned: false } }); ctx.ui.reload(); } catch (err) { ctx.ui.toast(err.message); } });
    const body = render(c.ir.root, sub);
    body.classList.add("embedded");
    return h("div", { class: "pinned-card" }, h("div", { class: "pinned-head" }, h("b", {}, c.title), unpin), body);
  }));
}
