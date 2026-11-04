export const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));

export const icon = (glyph, colour = "lavender") => `<span class="card-icon" data-colour="${escapeHtml(colour)}" aria-hidden="true">${escapeHtml(glyph)}</span>`;

export const button = (label, action, options = {}) => {
  const size = options.small ? " small" : "";
  const tone = options.tone ? ` ${options.tone}` : "";
  const attrs = options.attrs ?? "";
  return `<button class="btn play-button${tone}${size}" type="button" data-action="${escapeHtml(action)}" ${attrs}>${label}</button>`;
};

export const pill = (label, action, value, selected = false) => `<button class="pill-button" type="button" data-action="${escapeHtml(action)}" data-value="${escapeHtml(value)}" aria-pressed="${selected ? "true" : "false"}">${escapeHtml(label)}</button>`;

export const exactRange = ({ id, label, value, min, max, step = 1, unit = "", help = "Use the slider or type an exact value." }) => `
  <div class="field range-field">
    <div class="field-label"><label for="${escapeHtml(id)}">${escapeHtml(label)}</label><small>${escapeHtml(help)}</small></div>
    <div class="range-wrap">
      <input class="range range-primary" id="${escapeHtml(id)}" data-range="${escapeHtml(id)}" type="range" min="${min}" max="${max}" step="${step}" value="${Number(value)}" aria-label="${escapeHtml(label)}" />
      <label class="range-value" for="${escapeHtml(id)}-exact"><span class="visually-hidden">Exact ${escapeHtml(label)} </span><input class="exact-input" id="${escapeHtml(id)}-exact" data-exact="${escapeHtml(id)}" type="number" min="${min}" max="${max}" step="${step}" value="${Number(value)}" aria-label="Exact ${escapeHtml(label)}" /></label>
    </div>
    <p class="range-help">${escapeHtml(unit)} · Changes stay in this draft until you save.</p>
  </div>`;

export const stepper = ({ label, value, min = 0, max = 999, step = 1, id, field = "" }) => `
  <div class="stepper" role="group" aria-label="${escapeHtml(label)}">
    <button type="button" data-action="step" data-step="${escapeHtml(id)}" data-delta="-${step}" aria-label="Decrease ${escapeHtml(label)}">−</button>
    <input type="number" data-step-value="${escapeHtml(id)}" data-set-field="${escapeHtml(field)}" min="${min}" max="${max}" step="${step}" value="${Number(value)}" aria-label="${escapeHtml(label)}" />
    <button type="button" data-action="step" data-step="${escapeHtml(id)}" data-delta="${step}" aria-label="Increase ${escapeHtml(label)}">+</button>
  </div>`;

export const sectionHeading = (title, subtitle = "", glyph = "✳", colour = "lavender") => `
  <div class="section-header"><div><h2>${escapeHtml(title)}</h2>${subtitle ? `<p class="card-subtitle">${escapeHtml(subtitle)}</p>` : ""}</div>${icon(glyph, colour)}</div>`;

export const badge = (label, tone = "neutral") => `<span class="status-pill ${escapeHtml(tone)}">${escapeHtml(label)}</span>`;

export const emptyState = (message) => `<div class="empty-state">${escapeHtml(message)}</div>`;
