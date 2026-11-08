// Blacki Mini App shell: Telegram sign-in, navigation, page loading and UI helpers.
import { RenderContext, h, renderPage, shortDate } from "./render.js";
import { THEMES, applyTheme, localTheme, randomTheme, themeCss } from "./theme.js";

const tg = window.Telegram?.WebApp;
const main = document.getElementById("main");
const topnav = document.getElementById("topnav");
const tabbar = document.getElementById("tabbar");
const toastEl = document.getElementById("toast");
let token = null;
let pages = [];
let theme = localTheme() || { name: "blacki" };
const listeners = {};

applyTheme(theme);

// ---------------------------------------------------------------- api

async function call(method, path, body) {
  const res = await fetch(`/miniapp/api/${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch { /* empty body */ }
  if (res.status === 401 && path !== "session") { token = null; await signIn(); return call(method, path, body); }
  if (!res.ok) throw new Error(data.message || data.error || `Request failed (${res.status})`);
  return data;
}
const api = { action: (body) => call("POST", "action", body) };

async function signIn() {
  const initData = tg?.initData || "";
  const body = initData ? { init_data: initData } : { dev: true };
  const result = await call("POST", "session", body);
  token = result.token;
  if (result.dev) document.body.classList.add("dev");
}

// ---------------------------------------------------------------- ui helpers

const ui = {
  toast(text) {
    toastEl.textContent = text;
    toastEl.classList.add("on");
    clearTimeout(ui._t);
    ui._t = setTimeout(() => toastEl.classList.remove("on"), 2600);
  },
  confirm(text) {
    return new Promise((resolve) => {
      if (tg?.showConfirm && tg.platform !== "unknown") tg.showConfirm(text, (ok) => resolve(!!ok));
      else resolve(window.confirm(text));
    });
  },
  haptic(kind) {
    try {
      const hf = tg?.HapticFeedback;
      if (!hf) return;
      if (kind === "select") hf.selectionChanged();
      else hf.notificationOccurred(kind);
    } catch { /* not supported */ }
  },
  navigate(pageId) { location.hash = `#page/${pageId}`; },
  openCard(cardId) { location.hash = `#card/${cardId}`; },
  reload() { route(); },
  on(event, fn) { (listeners[event] ||= []).push(fn); },
  emit(event) { (listeners[event] || []).forEach((fn) => fn()); },
};

// ---------------------------------------------------------------- navigation

const PRIMARY = ["today", "health", "food", "training", "foryou"];
function drawNav(current) {
  const link = (p) => h("a", { href: `#page/${p.id}`, "aria-current": p.id === current ? "page" : null }, p.title);
  topnav.replaceChildren(...pages.map(link));
  const primary = PRIMARY.map((id) => pages.find((p) => p.id === id)).filter(Boolean);
  const moreActive = !PRIMARY.includes(current);
  const more = h("a", { href: "#more", "aria-current": moreActive ? "page" : null, onclick: (e) => { e.preventDefault(); openMore(); } }, "More");
  tabbar.replaceChildren(...primary.map((p) => { const a = link(p); a.textContent = p.id === "foryou" ? "For you" : p.title; return a; }), more);
}

function openMore() {
  const extra = pages.filter((p) => !PRIMARY.includes(p.id));
  sheet("More", [
    h("div", { class: "rows" }, extra.map((p) => h("a", { class: "row-item link-row", href: `#page/${p.id}`, onclick: () => closeSheet() }, h("i", { class: "key" }), h("div", {}, h("b", {}, p.title), h("small", {}, p.kind === "builtin" ? "Built in" : "Made by Blacki")), h("span", { class: "num" }, "›")))),
    h("button", { class: "btn", type: "button", onclick: () => { closeSheet(); openThemes(); } }, "Theme"),
  ]);
}

let openSheet = null;
function sheet(title, content) {
  closeSheet();
  const d = h("dialog", { class: "sheet" }, h("div", { class: "sheet-top" }, h("h2", {}, title), h("button", { class: "btn ghost", type: "button", onclick: () => closeSheet() }, "Done")), h("div", { class: "sheet-body" }, content));
  d.addEventListener("close", () => { d.remove(); if (openSheet === d) openSheet = null; });
  document.body.append(d);
  d.showModal();
  openSheet = d;
  return d;
}
function closeSheet() { if (openSheet) openSheet.close(); }

function openThemes() {
  const note = h("p", { class: "gen" });
  const list = h("ul", { class: "themes" });
  const paint = () => [...list.querySelectorAll("button")].forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.name === theme.name)));
  const choose = async (next) => {
    theme = next;
    applyTheme(theme);
    paint();
    try { await api.action({ action: "prefs.theme", args: { theme } }); } catch { ui.toast("Theme kept on this device only"); }
  };
  THEMES.forEach((name) => {
    const dots = h("span", { class: "dots", "aria-hidden": "true" }, h("i"), h("i"), h("i"), h("i"));
    list.append(h("li", {}, h("button", { type: "button", "data-name": name, "data-theme": name, onclick: () => { note.textContent = ""; choose({ name }); } }, dots, h("span", {}, name))));
  });
  const random = h("button", { class: "btn", type: "button", onclick: () => { const t = randomTheme(); note.textContent = `Random ${t.vars.dark ? "dark" : "light"} theme, corners ${t.vars.radius.box}. Press again for another.`; choose(t); } }, "Random theme");
  const copy = h("button", { class: "btn ghost", type: "button", onclick: async () => { try { await navigator.clipboard.writeText(themeCss(theme)); ui.toast("Theme CSS copied"); } catch { ui.toast("Copy isn't available here"); } } }, "Copy CSS");
  paint();
  sheet("Theme", [h("div", { class: "row-actions" }, random, copy), note, list]);
}

// ---------------------------------------------------------------- pages

function showError(message) {
  main.replaceChildren(h("div", { class: "wrap page-head" }, h("h1", {}, "Hmm."), h("p", {}, message)));
}

async function showPage(pageId) {
  drawNav(pageId);
  main.replaceChildren(h("div", { class: "wrap loading" }, h("p", { class: "note" }, "Loading…")));
  const page = await call("GET", `page/${encodeURIComponent(pageId)}`);
  const ctx = new RenderContext({ ir: page.ir, data: page.data, kind: page.kind, pageId: page.id, api, ui });
  const view = renderPage(ctx);
  if (page.kind === "agent") view.prepend(h("div", { class: "wrap made-by" }, h("span", {}, `Made by Blacki · version ${page.version}`)));
  main.replaceChildren(view);
  window.scrollTo(0, 0);
}

async function showCard(cardId) {
  drawNav("foryou");
  const card = await call("GET", `card/${encodeURIComponent(cardId)}`);
  const ctx = new RenderContext({ ir: card.ir, data: card.data, kind: "card", pageId: "", cardId: card.id, api, ui });
  const pin = h("button", { class: "chip-btn", type: "button" }, card.pinned ? "Unpin" : "Pin to Today");
  pin.addEventListener("click", async () => {
    try { const r = await api.action({ action: "Pin", card: card.id, args: { pinned: !card.pinned } }); ui.toast(r.message); route(); } catch (err) { ui.toast(err.message); }
  });
  const head = h("div", { class: "wrap made-by" }, h("span", {}, card.pinned ? "Card by Blacki · pinned" : `Card by Blacki · until ${shortDate(String(card.expires_at).slice(0, 10))}`), pin);
  const view = renderPage(ctx);
  view.prepend(head);
  main.replaceChildren(view);
  window.scrollTo(0, 0);
}

async function route() {
  const hash = location.hash || "#page/today";
  try {
    const card = hash.match(/^#card\/([A-Za-z0-9_-]{1,40})$/);
    if (card) { await showCard(card[1]); return; }
    const page = hash.match(/^#page\/([a-z][a-z0-9-]{1,31})$/);
    await showPage(page ? page[1] : "today");
  } catch (err) {
    showError(err.message || "Something went wrong.");
  }
  if (tg?.BackButton) {
    const atHome = !location.hash || location.hash === "#page/today";
    atHome ? tg.BackButton.hide() : tg.BackButton.show();
  }
}

async function start() {
  if (tg) {
    tg.ready();
    tg.expand?.();
    tg.BackButton?.onClick(() => history.back());
  }
  try {
    await signIn();
    const nav = await call("GET", "nav");
    pages = nav.pages;
    if (nav.prefs?.theme) { theme = nav.prefs.theme; applyTheme(theme); }
  } catch (err) {
    showError(err.message || "Couldn't sign in. Close the app and open it again from Telegram.");
    return;
  }
  const params = new URLSearchParams(location.search);
  const startCard = params.get("card") || tg?.initDataUnsafe?.start_param;
  if (startCard && /^[A-Za-z0-9_-]{1,40}$/.test(startCard) && !location.hash) location.hash = `#card/${startCard}`;
  document.getElementById("theme-open").addEventListener("click", openThemes);
  window.addEventListener("hashchange", route);
  route();
}

start();
