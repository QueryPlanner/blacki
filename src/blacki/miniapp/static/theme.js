// Theme picker: daisyUI 5 themes, Blacki's own theme, and a random generator.
// The choice is saved to the user's Blacki settings; Blacki itself can't change it.
export const THEMES = ["blacki", "light", "dark", "cupcake", "bumblebee", "emerald", "corporate", "synthwave", "retro", "cyberpunk", "valentine", "halloween", "garden", "forest", "aqua", "lofi", "pastel", "fantasy", "wireframe", "black", "luxury", "dracula", "cmyk", "autumn", "business", "acid", "lemonade", "night", "coffee", "winter", "dim", "nord", "sunset", "caramellatte", "abyss", "silk"];
const VARS = ["base-100", "base-200", "base-300", "base-content", "primary", "primary-content", "secondary", "secondary-content", "accent", "accent-content", "neutral", "neutral-content"];
const root = document.documentElement;
const local = {
  get() { try { return JSON.parse(localStorage.getItem("blacki-theme")); } catch { return null; } },
  set(v) { try { localStorage.setItem("blacki-theme", JSON.stringify(v)); } catch { /* storage unavailable */ } },
};

function clearInline() {
  VARS.forEach((v) => root.style.removeProperty(`--color-${v}`));
  ["box", "field", "selector"].forEach((v) => root.style.removeProperty(`--radius-${v}`));
}

export function applyTheme(theme) {
  clearInline();
  if (theme && theme.vars) {
    root.dataset.theme = theme.vars.dark ? "dark" : "light";
    Object.entries(theme.vars.colors || {}).forEach(([k, v]) => { if (VARS.includes(k) && /^oklch\([\d.% ]+\)$/.test(v)) root.style.setProperty(`--color-${k}`, v); });
    Object.entries(theme.vars.radius || {}).forEach(([k, v]) => { if (["box", "field", "selector"].includes(k) && /^[\d.]+rem$/.test(v)) root.style.setProperty(`--radius-${k}`, v); });
  } else {
    root.dataset.theme = THEMES.includes(theme?.name) ? theme.name : "blacki";
  }
  const bg = getComputedStyle(root).getPropertyValue("--color-base-100").trim();
  const tg = window.Telegram?.WebApp;
  if (tg && bg) { try { tg.setHeaderColor?.("bg_color"); tg.setBackgroundColor?.("bg_color"); } catch { /* older clients */ } }
  local.set(theme);
}

const rnd = (a, b) => a + Math.random() * (b - a);
const ok = (l, c, hue) => `oklch(${l.toFixed(1)}% ${c.toFixed(3)} ${hue.toFixed(0)})`;
const onFor = (l) => (l > 66 ? 18 : 97);

export function randomTheme() {
  const dark = Math.random() < 0.5;
  const baseHue = rnd(0, 360);
  const baseC = rnd(0, 0.03);
  const ph = rnd(0, 360);
  const sh = (ph + rnd(90, 150)) % 360;
  const ah = (ph + rnd(180, 240)) % 360;
  const pl = dark ? rnd(70, 88) : rnd(45, 62);
  const sl = dark ? rnd(65, 85) : rnd(50, 70);
  const al = rnd(60, 85);
  const b1 = dark ? rnd(14, 22) : rnd(96, 100);
  const step = dark ? 4 : -2.5;
  const colors = {
    "base-100": ok(b1, baseC, baseHue), "base-200": ok(b1 + step, baseC, baseHue), "base-300": ok(b1 + step * 2.2, baseC, baseHue),
    "base-content": ok(dark ? 94 : 22, baseC, baseHue),
    primary: ok(pl, rnd(0.12, 0.22), ph), "primary-content": ok(onFor(pl), 0.02, ph),
    secondary: ok(sl, rnd(0.1, 0.2), sh), "secondary-content": ok(onFor(sl), 0.02, sh),
    accent: ok(al, rnd(0.1, 0.18), ah), "accent-content": ok(onFor(al), 0.02, ah),
    neutral: ok(dark ? 30 : 25, baseC, baseHue), "neutral-content": ok(95, baseC, baseHue),
  };
  const r = [["0rem", "0rem", "0rem"], [".25rem", ".25rem", ".25rem"], [".5rem", ".5rem", ".25rem"], ["1rem", ".5rem", ".5rem"], ["1.5rem", "1rem", "1rem"], ["2rem", "2rem", "1rem"]][Math.floor(rnd(0, 6))];
  return { name: "random", vars: { dark, colors, radius: { box: r[0], field: r[1], selector: r[2] } } };
}

export function themeCss(theme) {
  if (!theme?.vars) return `/* daisyUI theme: ${theme?.name || "blacki"} */`;
  const lines = Object.entries(theme.vars.colors).map(([k, v]) => `  --color-${k}: ${v};`)
    .concat(Object.entries(theme.vars.radius).map(([k, v]) => `  --radius-${k}: ${v};`));
  return `[data-theme="my-random"] {\n  color-scheme: ${theme.vars.dark ? "dark" : "light"};\n${lines.join("\n")}\n}`;
}

export function localTheme() { return local.get(); }
