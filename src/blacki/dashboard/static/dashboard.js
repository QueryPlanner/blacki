"use strict";

const root = document.getElementById("view");
const refresh = document.getElementById("refresh");
const pageSize = 50;
const maxPageSize = 200;
const views = {
  sessions: ["Sessions", "Read conversations, inspect captured model context, and browse a user’s wiki."],
  overview: ["Overview", "Review activity, token use, and reported costs in the local files."],
  traces: ["Traces", "Inspect recorded spans, timing, and trace status."],
  logs: ["Logs", "Search the application log by message or severity."],
};
const state = {
  view: "sessions", user: "", session: "", pane: "transcript", wikiSelection: {},
  userSearch: "", userLimit: pageSize, sessionLimit: pageSize, window: "all",
  traces: { search: "", status: "", limit: 50 }, logs: { search: "", level: "", limit: 50 },
};
let currentView;
let currentDetail;
const numbers = new Intl.NumberFormat("en-IN");
const dates = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata", day: "numeric", month: "short", year: "numeric",
});
const clocks = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});
const date = (value) => value ? dates.format(new Date(value)) : "Date unavailable";
const clock = (value) => value ? clocks.format(new Date(value)) : "";
const dateTime = (value) => value ? `${date(value)} · ${clock(value)} IST` : "Date unavailable";
const count = (value) => numbers.format(value ?? 0);
const formatCost = (value) => value == null ? "Unavailable" : "$" + value.toFixed(4);
const formatPercent = (value) => value == null ? "Unavailable" : (value * 100).toFixed(1) + "%";
const formatMs = (value) => value == null ? "Unavailable" : count(Math.round(value)) + " ms";

function costValue(reported, estimated) {
  if (reported != null) return formatCost(reported);
  return estimated == null ? "Unavailable" : formatCost(estimated) + " estimated";
}

function costDescription(item) {
  const month = costValue(item.monthly_cost_usd, item.monthly_estimated_cost_usd);
  const total = costValue(item.cumulative_cost_usd, item.cumulative_estimated_cost_usd);
  return "Cost this month " + month + " · Cumulative " + total;
}

const plural = (n, word) => `${count(n)} ${word}${n === 1 ? "" : "s"}`;

function el(tag, classes = "", value) {
  const node = document.createElement(tag);
  if (classes) node.className = classes;
  if (value !== undefined && value !== null) node.textContent = value;
  return node;
}

function action(label, classes, onClick) {
  const node = el("button", classes, label);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

function labelControl(label, control) {
  const node = el("label", "field-label", label);
  node.append(control);
  return node;
}

function options(items, selected) {
  const node = el("select", "field");
  items.forEach(([value, label]) => node.append(new Option(label, value, false, value === selected)));
  return node;
}

function notice(value, role = "") {
  const node = el("p", "empty", value);
  if (role) node.setAttribute("role", role);
  return node;
}

function warnings(parent, data) {
  for (const warning of data.warnings ?? []) parent.append(el("p", "warning", warning));
}

function error(target) {
  target.replaceChildren(notice("Couldn’t read the local data. Try refreshing the dashboard.", "alert"));
}

async function api(endpoint, params, signal) {
  const query = new URLSearchParams(params);
  const response = await fetch(`/dashboard/api/${endpoint}?${query}`, { signal, cache: "no-store" });
  if (response.status === 403) return { status: "forbidden" };
  if (!response.ok) throw new Error(`Local response ${response.status}`);
  const data = await response.json();
  signal.throwIfAborted();
  return data;
}

async function loadVisibleRange(endpoint, params, limit, signal) {
  const result = await api(endpoint, { ...params, offset: 0, limit: Math.min(limit, maxPageSize) }, signal);
  // Refresh the range the operator has already loaded, in API-sized pages.
  while (result.items.length < Math.min(limit, result.total)) {
    const offset = result.items.length;
    const next = await api(endpoint, { ...params, offset, limit: Math.min(limit - offset, maxPageSize) }, signal);
    if (!next.items.length) break;
    result.items.push(...next.items);
    result.total = next.total;
  }
  return result;
}

function dateNode(value, full = false) {
  const node = el("time", "", full ? dateTime(value) : clock(value));
  if (value) node.dateTime = value;
  return node;
}

function facts(items, classes, itemClass = "") {
  const list = el("dl", classes);
  for (const [label, value] of items) {
    const item = el("div", itemClass);
    item.append(el("dt", "", label), el("dd", "", value));
    list.append(item);
  }
  return list;
}

function sessionTitle(row) {
  return row.version == null ? "Session" : `Session v${row.version}`;
}

function ownerName(owner) {
  return owner.display_name || owner.username || owner.user_id;
}

async function renderSessions(signal) {
  const rail = el("aside", "session-rail");
  const panel = el("div", "inspector");
  const bench = el("div", "workbench");
  bench.append(rail, panel);
  const users = await loadVisibleRange("users", { search: state.userSearch }, state.userLimit, signal);
  root.replaceChildren(bench);
  const listedUsers = [...users.items];
  const searchForm = el("form", "owner-search");
  const search = el("input", "field");
  search.type = "search";
  search.maxLength = 128;
  search.value = state.userSearch;
  search.placeholder = "Name or ID";
  const apply = el("button", "control", "Find");
  apply.type = "submit";
  searchForm.append(labelControl("Find user", search), apply);
  searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    state.userSearch = search.value.trim();
    state.user = "";
    state.session = "";
    state.userLimit = pageSize;
    state.sessionLimit = pageSize;
    state.wikiSelection = {};
    render();
  });
  rail.append(searchForm);
  if (!listedUsers.length) {
    panel.append(notice(state.userSearch ? "No users match this search." : "No local ADK sessions are available."));
    warnings(panel, users);
    return;
  }
  const chooser = options([], "");
  const railHead = el("div", "rail-head");
  const totalLabel = el("span");
  const list = el("div", "session-list");
  let sessionPage;
  let sessionRows = [];
  let selectedOwner;

  function setUserOptions() {
    const entries = listedUsers.map((u) => [u.user_id, ownerName(u)]);
    const visible = listedUsers.some((u) => u.user_id === state.user);
    if (!visible) entries.unshift(["", "Choose a user"]);
    const next = options(entries, visible ? state.user : "");
    if (!visible) next.options[0].disabled = true;
    chooser.replaceChildren(...next.options);
    chooser.value = visible ? state.user : "";
  }

  function ownerMore(offset) {
    const button = action("More users", "control more", async () => {
      button.disabled = true;
      try {
        const result = await api("users", { search: state.userSearch, offset, limit: pageSize }, signal);
        listedUsers.push(...result.items);
        state.userLimit = Math.max(state.userLimit, listedUsers.length);
        setUserOptions();
        if (!selectedOwner && chooser.value) await loadSessions();
        if (offset + result.items.length < result.total) {
          button.replaceWith(ownerMore(offset + result.items.length));
        } else {
          button.remove();
        }
      } catch (reason) {
        if (reason.name !== "AbortError") button.textContent = "Retry loading users";
        button.disabled = false;
      }
    });
    return button;
  }

  if (!state.user) state.user = listedUsers[0].user_id;
  setUserOptions();
  const ownerCost = el("p", "rail-note");
  rail.append(labelControl("User", chooser), ownerCost);
  if (listedUsers.length < users.total) rail.append(ownerMore(listedUsers.length));
  railHead.append(el("h2", "", "Session history"), totalLabel);
  rail.append(railHead, list);
  const more = action("More sessions", "control more", () => loadSessions(true));
  more.hidden = true;
  const railWarnings = el("div");
  rail.append(more, railWarnings, el("p", "rail-note", "Telegram /reset keeps older sessions. Times shown in IST."));

  async function loadSessions(append = false) {
    currentDetail?.abort();
    const controller = new AbortController();
    currentDetail = controller;
    const detailSignal = AbortSignal.any([signal, controller.signal]);
    const offset = append ? sessionRows.length : 0;
    selectedOwner = listedUsers.find((user) => user.user_id === state.user);
    if (!selectedOwner) {
      ownerCost.textContent = "";
      list.replaceChildren();
      panel.replaceChildren(notice("The selected user is outside the loaded results. Choose a user to continue."));
      return;
    }
    ownerCost.textContent = costDescription(selectedOwner);
    if (!append) {
      list.replaceChildren(notice("Reading sessions…", "status"));
      panel.replaceChildren();
    }
    more.disabled = true;
    try {
      const params = { user_id: state.user };
      sessionPage = append
        ? await api("sessions", { ...params, offset, limit: pageSize }, detailSignal)
        : await loadVisibleRange("sessions", params, state.sessionLimit, detailSignal);
      sessionRows = append ? [...sessionRows, ...sessionPage.items] : sessionPage.items;
      state.sessionLimit = Math.max(state.sessionLimit, sessionRows.length);
      if (!state.session) state.session = sessionRows[0]?.session_id || "";
      renderRail();
      await renderDetail(detailSignal);
    } catch (reason) {
      if (reason.name !== "AbortError") error(list);
    } finally {
      more.disabled = false;
    }
  }

  function renderRail() {
    totalLabel.textContent = plural(sessionPage.total, "session");
    list.replaceChildren();
    for (const row of sessionRows) {
      const key = action("", "session-key", () => {
        state.session = row.session_id;
        renderRail();
        list.querySelector("[aria-current]")?.focus({ preventScroll: true });
        selectDetail();
      });
      if (row.session_id === state.session) key.setAttribute("aria-current", "true");
      const title = el("span", "key-title", sessionTitle(row));
      if (row.is_current) title.append(el("span", "current-label", "Current"));
      key.append(title, el("span", "key-meta", `${date(row.updated_at)} · ${plural(row.message_count, "message")}`));
      list.append(key);
    }
    more.hidden = sessionRows.length >= sessionPage.total;
    if (!sessionRows.length) list.append(notice("No sessions for this user."));
    railWarnings.replaceChildren();
    warnings(railWarnings, sessionPage);
  }

  chooser.addEventListener("change", () => {
    state.user = chooser.value;
    state.session = "";
    state.sessionLimit = pageSize;
    state.wikiSelection = {};
    more.hidden = true;
    loadSessions();
  });

  function selectDetail() {
    currentDetail?.abort();
    const controller = new AbortController();
    currentDetail = controller;
    return renderDetail(AbortSignal.any([signal, controller.signal]));
  }

  async function renderDetail(detailSignal) {
    const row = sessionRows.find((item) => item.session_id === state.session);
    if (!row) {
      const message = state.session
        ? "The selected session is outside the loaded history. Choose a session to continue."
        : "Select a session to read its transcript.";
      panel.replaceChildren(notice(message));
      return;
    }
    const header = el("div", "session-heading");
    const line = el("div", "session-line");
    const title = el("div");
    title.append(el("h2", "", sessionTitle(row)), el("p", "", "Last active " + dateTime(row.updated_at)));
    title.append(el("p", "", costDescription(row)));
    line.append(title, facts([
      ["Messages", count(row.message_count)],
      ["Invocations", count(row.invocation_count)],
      ["Tokens", count(row.tokens)],
    ], "session-facts"));
    const tabs = el("nav", "detail-tabs");
    tabs.setAttribute("aria-label", "Session and user details");
    const body = el("div", "inspector-body");
    for (const [id, text] of [["transcript", "Transcript"], ["wiki", "User wiki"]]) {
      const tab = action(text, "", () => {
        state.pane = id;
        selectDetail();
        panel.querySelector("[aria-current]")?.focus({ preventScroll: true });
      });
      if (id === state.pane) tab.setAttribute("aria-current", "true");
      tabs.append(tab);
    }
    header.append(line, tabs);
    body.append(notice(state.pane === "wiki" ? "Reading the user wiki…" : "Reading the transcript…", "status"));
    panel.replaceChildren(header, body);
    try {
      if (state.pane === "wiki") {
        await renderWiki(body, selectedOwner, state.wikiSelection, detailSignal);
      } else {
        const params = { user_id: state.user, session_id: state.session };
        const transcript = await api("session", params, detailSignal);
        renderTranscript(body, transcript, params, detailSignal);
      }
    } catch (reason) {
      if (reason.name !== "AbortError") error(body);
    }
  }

  await loadSessions();
}

const privateStatus = {
  disabled: "Private views require an enabled dashboard host.",
  forbidden: "This host is not enabled for private dashboard views.",
  trace_missing: "The local trace file is unavailable.",
  session_missing: "This saved session is no longer available.",
  capture_missing: "This request capture is no longer available.",
  missing: "No wiki database was found in local storage.",
  empty: "This user has no saved wiki pages.",
  wiki_disabled: "Wiki memory is disabled.",
  unsupported_user: "Wiki data is available for direct Telegram users.",
  user_missing: "This user has no local sessions.",
  page_missing: "This wiki page is no longer available.",
  wiki_changed: "The wiki changed. Reload its pages to continue.",
  unavailable: "The local data could not be read.",
};

function requestContext(params, signal) {
  const context = el("details", "context");
  const summary = el("summary");
  const mark = el("span", "context-mark");
  mark.setAttribute("aria-hidden", "true");
  const caret = el("span", "caret");
  caret.setAttribute("aria-hidden", "true");
  const title = el("span", "context-title", "System prompt & tools");
  const subtitle = el("span", "context-subtitle", "Inspect the context recorded for a model request");
  title.append(subtitle);
  summary.append(mark, title, caret);
  const contents = el("div", "context-body");
  context.append(summary, contents);
  let loading;

  context.addEventListener("toggle", () => {
    loading?.abort();
    if (!context.open) {
      contents.replaceChildren();
      return;
    }
    loading = new AbortController();
    const openSignal = AbortSignal.any([signal, loading.signal]);
    loadCaptures(contents, subtitle, params, openSignal);
  });
  return context;
}

async function loadCaptures(contents, subtitle, params, signal) {
  contents.replaceChildren(notice("Reading request captures…", "status"));
  try {
    const first = await api("context", params, signal);
    if (first.status !== "available" || !first.items.length) {
      contents.replaceChildren(el("p", "help", privateStatus[first.status]
        || "The local traces have no recorded system prompt for this session."));
      if (first.limited_scan) contents.append(el("p", "warning", "The scan skipped some trace records."));
      return;
    }
    const captures = [...first.items];
    subtitle.textContent = plural(first.total, "captured model request");
    const label = (capture, index) => dateTime(capture.timestamp) + " · Request " + (first.total - index);
    const request = options(captures.map((capture, index) => [String(index), label(capture, index)]), "0");
    const captureBody = el("div", "capture-body");
    let selectedSection = "prompt";
    let detailRequest;
    async function showCapture() {
      detailRequest?.abort();
      detailRequest = new AbortController();
      const captureSignal = AbortSignal.any([signal, detailRequest.signal]);
      const selected = captures[Number(request.value)];
      captureBody.replaceChildren(notice("Reading model context…", "status"));
      try {
        const response = await api("context", {
          ...params, trace_id: selected.trace_id, span_id: selected.span_id,
        }, captureSignal);
        if (response.status === "available") {
          renderCapture(captureBody, response.capture, selectedSection, (section) => {
            selectedSection = section;
          });
        } else {
          captureBody.replaceChildren(notice(privateStatus[response.status]));
        }
      } catch (reason) {
        if (reason.name !== "AbortError") error(captureBody);
      }
    }
    let nextOffset = captures.length;
    const more = action("More captured requests", "control more", async () => {
      more.disabled = true;
      try {
        const next = await api("context", { ...params, offset: nextOffset }, signal);
        if (next.status !== "available") {
          more.textContent = privateStatus[next.status] || "Captures unavailable";
          return;
        }
        const keys = new Set(captures.map((c) => c.trace_id + c.span_id));
        for (const item of next.items) {
          if (keys.has(item.trace_id + item.span_id)) continue;
          const index = captures.length;
          captures.push(item);
          request.append(new Option(label(item, index), String(index)));
        }
        nextOffset += next.items.length;
        more.hidden = nextOffset >= next.total || !next.items.length;
      } catch (reason) {
        if (reason.name !== "AbortError") more.textContent = "Retry loading requests";
      } finally {
        more.disabled = false;
      }
    });
    more.hidden = nextOffset >= first.total;
    contents.replaceChildren(labelControl("Captured model request", request), more, captureBody);
    contents.append(el("p", "help", "Captured trace text may already be incomplete. The dashboard filters credential patterns."));
    if (first.limited_scan) contents.append(el("p", "warning", "The scan skipped some trace records."));
    request.addEventListener("change", showCapture);
    await showCapture();
  } catch (reason) {
    if (reason.name !== "AbortError") error(contents);
  }
}

function renderTranscript(body, transcript, params, signal) {
  const contextPanel = requestContext(params, signal);
  const heading = el("div", "thread-title");
  heading.append(el("h3", "", "Conversation"), el("span", "", "Oldest first"));
  body.replaceChildren(contextPanel, heading);
  if (!transcript.messages.length) {
    body.append(notice("This session has no saved conversation events."));
  } else {
    const timeline = el("ol", "message-list");
    for (const message of transcript.messages) timeline.append(renderMessage(message));
    body.append(timeline);
  }
  warnings(body, transcript);
}

function renderCapture(container, capture, selectedSection, onSelect) {
  const controls = el("nav", "detail-tabs capture-tabs");
  controls.setAttribute("aria-label", "Captured request sections");
  const output = el("div", "capture-output");
  const buttons = new Map();

  function showSection(section) {
    onSelect(section);
    for (const [id, button] of buttons) {
      if (id === section) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    }
    if (section === "prompt") {
      output.replaceChildren(el("pre", "prompt", capture.system_prompt));
      return;
    }
    const toolList = el("div", "tool-list");
    for (const tool of capture.tools) {
      const item = el("details", "tool");
      const title = el("summary", "", tool.name);
      const caret = el("span", "caret");
      caret.setAttribute("aria-hidden", "true");
      title.append(caret);
      item.append(title, el("p", "message-text", tool.description));
      toolList.append(item);
    }
    output.replaceChildren();
    if (capture.tools.length) output.append(toolList);
    else output.append(el("p", "help", "This request did not record tool descriptions."));
    if (capture.partial_tools) output.append(el("p", "warning", "Some tool declarations were missing or could not be read."));
  }

  for (const [id, label] of [["prompt", "System prompt"], ["tools", `Tool descriptions (${count(capture.tools.length)})`]]) {
    const button = action(label, "", () => showSection(id));
    buttons.set(id, button);
    controls.append(button);
  }
  container.replaceChildren();
  if (capture.model) container.append(el("p", "help", capture.model));
  container.append(controls, output);
  showSection(selectedSection);
}

function renderMessage(message) {
  const isText = typeof message.text === "string";
  const role = isText ? (message.role === "human" ? "human" : "model") : "meta";
  const marker = el("span", "message-icon", { human: "U", model: "B", meta: "·" }[role]);
  marker.setAttribute("aria-hidden", "true");
  const item = el("li", `message ${role}`);
  const details = el("div");
  const header = el("div", "message-head");
  const labels = { human: "User", model: "Blacki", meta: message.type === "tool" ? "Tool event" : "Attachment" };
  const content = isText ? message.text : message.type === "tool"
    ? `${message.name} · ${message.status}` : message.mime_type;
  header.append(el("b", "", labels[role]), dateNode(message.timestamp));
  details.append(header, el("p", "message-text", content));
  item.append(marker, details);
  return item;
}

async function renderWiki(container, owner, selection, signal, pageFocus = false) {
  const data = await api("wiki", { user_id: owner.user_id, ...selection }, signal);
  const wiki = el("div", "wiki");
  wiki.append(el("h3", "", "User wiki"), el("p", "muted", ownerName(owner)));
  let pageRequest;
  function changePage(nextSelection) {
    pageRequest?.abort();
    pageRequest = new AbortController();
    state.wikiSelection = nextSelection;
    const pageSignal = AbortSignal.any([signal, pageRequest.signal]);
    renderWiki(container, owner, nextSelection, pageSignal, true).catch((reason) => {
      if (reason.name !== "AbortError") error(container);
    });
  }
  if (data.status !== "available") {
    wiki.append(el("p", "help", privateStatus[data.status] || "Wiki data is unavailable."));
    if (["page_missing", "wiki_changed"].includes(data.status)) {
      wiki.append(action("Reload wiki pages", "control", () => changePage({})));
    }
    container.replaceChildren(wiki);
    return;
  }

  const offset = selection.offset || 0;
  const claimOffset = selection.claim_offset || 0;
  const layout = el("div", "wiki-split");
  const pages = el("nav", "wiki-pages");
  pages.setAttribute("aria-label", "Wiki pages");
  const reader = el("div", "wiki-page");
  data.pages.forEach((path, index) => {
    const page = offset + index;
    const button = action(path, "page-key", () => changePage({ page, offset, revision: data.revision }));
    if (page === data.page) button.setAttribute("aria-current", "page");
    pages.append(button);
  });
  const pageNavigation = el("div", "pagination");
  if (offset > 0) pageNavigation.append(action("Previous pages", "control", () => changePage({
    offset: Math.max(0, offset - 50), revision: data.revision,
  })));
  if (data.more_pages) pageNavigation.append(action("More pages", "control", () => changePage({
    offset: offset + 50, revision: data.revision,
  })));
  pages.append(pageNavigation);
  const list = el("ul");
  for (const claim of data.claims) {
    const item = el("li", "", claim.text);
    if (claim.disputed) item.append(el("small", "", " · Disputed"));
    list.append(item);
  }
  reader.append(el("h4", "", data.pages[data.page - offset]), el("p", "muted", "Revision " + count(data.revision)), list);
  const claimNavigation = el("div", "pagination");
  const showClaims = (claim_offset) => changePage({ page: data.page, offset, revision: data.revision, claim_offset });
  if (claimOffset > 0) claimNavigation.append(action("Previous claims", "control", () => showClaims(Math.max(0, claimOffset - 50))));
  if (data.more_claims) claimNavigation.append(action("More claims", "control", () => showClaims(claimOffset + 50)));
  reader.append(claimNavigation);
  layout.append(pages, reader);
  wiki.append(layout);
  container.replaceChildren(wiki);
  if (pageFocus) pages.querySelector("[aria-current]")?.focus();
}

function chart(activity) {
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Message count by time bucket: ${activity.map((bucket) => count(bucket.messages)).join(", ")}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("viewBox", `0 0 ${activity.length * 10} 110`);
  const max = Math.max(1, ...activity.map((bucket) => bucket.messages));
  activity.forEach((bucket, index) => {
    const bar = document.createElementNS(namespace, "rect");
    const height = bucket.messages / max * 103;
    bar.setAttribute("x", String(index * 10));
    bar.setAttribute("width", "8");
    bar.setAttribute("y", String(110 - height));
    bar.setAttribute("height", String(height));
    const title = document.createElementNS(namespace, "title");
    title.textContent = `${dateTime(bucket.timestamp)} · ${plural(bucket.messages, "message")}`;
    bar.append(title);
    svg.append(bar);
  });
  return svg;
}

async function renderOverview(signal) {
  const picker = options([["all", "All local activity"], ["24h", "Last 24 hours"], ["7d", "Last 7 days"], ["30d", "Last 30 days"]], state.window);
  picker.addEventListener("change", () => {
    state.window = picker.value;
    render();
  });
  const toolbar = el("div", "report-toolbar");
  toolbar.append(labelControl("Time window", picker));
  root.replaceChildren(toolbar, notice("Reading activity…", "status"));
  const data = await api("overview", { window: state.window }, signal);
  const stats = data.stats;
  root.lastChild.remove();
  root.append(facts([
    ["Sessions", count(stats.sessions)], ["Messages", count(stats.messages)],
    ["Traces", count(stats.traces)], ["Tokens", count(stats.tokens)],
  ], "metrics", "metric"));
  const activity = el("div", "report");
  activity.append(el("h2", "", "Conversation activity"));
  if (data.activity?.some((bucket) => bucket.messages > 0)) {
    const caption = el("div", "chart-caption");
    caption.append(el("span", "", date(data.activity[0].timestamp)), el("span", "", date(data.activity.at(-1).timestamp)));
    activity.append(chart(data.activity), caption);
  } else {
    activity.append(el("p", "muted", "No messages in this time window."));
  }
  const details = el("div", "report");
  details.append(el("h2", "", "Usage and health"), facts([
    ["Active users", count(stats.users)],
    ["Average trace latency", formatMs(stats.avg_latency_ms)],
    ["95th-percentile trace latency", formatMs(stats.p95_latency_ms)],
    ["Request error rate", formatPercent(stats.error_rate)],
    ["Trace errors", count(stats.trace_errors)],
    ["Reported model cost", formatCost(stats.cost?.window_usd)],
    ["Estimated model cost", formatCost(stats.cost?.estimated_window_usd)],
    ["Log errors", count(stats.log_errors)],
    ["Monthly model cost", costValue(stats.monthly_cost_usd, stats.monthly_estimated_cost_usd)],
    ["Cumulative model cost", costValue(stats.cumulative_cost_usd, stats.cumulative_estimated_cost_usd)],
    ["Average monthly cost per user", formatCost(stats.average_user_monthly_cost_usd)],
    ["Provider-reported cost coverage", formatPercent(stats.cost_coverage)],
  ], "report-stats"));
  const sources = el("div", "report");
  sources.append(el("h2", "", "Local sources"), facts([
    ["Application log records", count(stats.log_records)],
    ["Trace spans", count(stats.trace_spans)],
    ["Conversation requests", count(stats.requests)],
  ], "report-stats"));
  root.append(activity, details, sources);
  warnings(root, data);
}

function filterBar(view, signal, update) {
  const toolbar = el("form", "report-toolbar");
  const search = el("input", "field");
  search.type = "search";
  search.maxLength = 128;
  search.placeholder = view === "traces" ? "Find a trace" : "Find a log entry";
  search.value = state[view].search;
  const field = view === "traces" ? "status" : "level";
  const values = view === "traces" ? ["OK", "ERROR", "UNSET"] : ["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"];
  const picker = options([["", "All"], ...values.map((value) => [value, value])], state[view][field]);
  const submit = el("button", "control", "Apply filters");
  submit.type = "submit";
  toolbar.append(labelControl("Search", search), labelControl(view === "traces" ? "Status" : "Level", picker), submit);
  toolbar.addEventListener("submit", (event) => {
    event.preventDefault();
    if (signal.aborted) return;
    state[view] = { search: search.value.trim(), [field]: picker.value, limit: 50 };
    update();
  });
  return toolbar;
}

async function renderRecords(view, signal) {
  const target = el("div");
  const detail = el("div", "trace-detail");
  let request;
  async function update() {
    request?.abort();
    request = new AbortController();
    const localSignal = AbortSignal.any([request.signal, signal]);
    target.replaceChildren(notice("Reading local records…", "status"));
    detail.replaceChildren();
    try {
      const data = await api(view, state[view], localSignal);
      target.replaceChildren(el("p", "result-count", `Showing ${count(data.items.length)} of ${count(data.total)} matching records`));
      if (!data.items.length) target.append(notice("No records match these filters."));
      for (const entry of data.items) {
        if (view === "logs") {
          target.append(renderLog(entry));
        } else {
          const node = action("", "trace-link", () => openTrace(detail, entry.trace_id, localSignal));
          node.append(el("strong", "", entry.name), el("span", "", `${entry.status} · ${plural(entry.span_count, "span")}`), el("span", "", dateTime(entry.start_time) + " · " + formatMs(entry.duration_ms)));
          target.append(node);
        }
      }
      if (data.items.length < data.total && state[view].limit < 200) {
        target.append(action("Show more records", "control more", () => {
          state[view].limit = Math.min(state[view].limit + 50, 200);
          update();
        }));
      }
      warnings(target, data);
    } catch (reason) {
      if (reason.name !== "AbortError") error(target);
    }
  }
  root.replaceChildren(filterBar(view, signal, update), detail, target);
  await update();
}

function renderLog(entry) {
  const item = el("details", "log-entry");
  const title = el("summary");
  const level = el("span", "level", entry.level || "LOG");
  if (["ERROR", "CRITICAL"].includes(entry.level)) level.setAttribute("data-error", "");
  const message = typeof entry.message === "string" ? entry.message : "Structured log entry";
  title.append(level, el("strong", "", message.length > 150 ? `${message.slice(0, 149)}…` : message), dateNode(entry.timestamp, true));
  const body = el("div", "entry-body");
  body.append(el("p", "muted", entry.logger), el("p", "message-text", message));
  const fields = entry.attributes || entry.fields || entry.metadata || entry.context;
  if (fields) body.append(el("pre", "prompt", JSON.stringify(fields, null, 2)));
  item.append(title, body);
  return item;
}

let traceRequest;
async function openTrace(target, id, signal) {
  traceRequest?.abort();
  traceRequest = new AbortController();
  const traceSignal = AbortSignal.any([traceRequest.signal, signal]);
  target.replaceChildren(notice("Reading trace spans…", "status"));
  try {
    const trace = await api("trace", { trace_id: id }, traceSignal);
    const report = el("div", "report");
    const title = el("h2", "", trace.name);
    title.tabIndex = -1;
    const list = el("ul", "trace-rows");
    const ids = new Set(trace.spans.map((span) => span.span_id));
    for (const span of trace.spans) {
      const row = el("li", ids.has(span.parent_span_id) ? "trace-child" : "");
      const expandable = el("details", "log-entry");
      const summary = el("summary");
      summary.append(el("strong", "", span.name), el("span", "level", span.status + " · " + formatMs(span.duration_ms)));
      const body = el("div", "entry-body");
      body.append(el("p", "muted", span.span_id), el("p", "muted", dateTime(span.start_time) + " · Cost " + costValue(span.cost_usd, span.estimated_cost_usd)));
      if (Object.keys(span.attributes || {}).length) body.append(el("pre", "prompt", JSON.stringify(span.attributes, null, 2)));
      if (span.events?.length) body.append(el("pre", "prompt", JSON.stringify(span.events, null, 2)));
      expandable.append(summary, body);
      row.append(expandable);
      list.append(row);
    }
    report.append(title, el("p", "help", trace.status + " · " + dateTime(trace.start_time) + " · " + plural(trace.span_count, "span")),
      el("p", "help", "Trace " + trace.trace_id + " · Cost " + costValue(trace.cost_usd, trace.estimated_cost_usd)), list);
    report.append(action("Close trace", "control", () => target.replaceChildren()));
    warnings(report, trace);
    target.replaceChildren(report);
    title.focus();
  } catch (reason) {
    if (reason.name !== "AbortError") error(target);
  }
}

async function render() {
  currentView?.abort();
  currentDetail?.abort();
  const controller = new AbortController();
  currentView = controller;
  const hash = location.hash.slice(1);
  if (Object.hasOwn(views, hash)) state.view = hash;
  const view = state.view;
  const [title, description] = views[view];
  document.getElementById("view-title").textContent = title;
  document.getElementById("view-description").textContent = description;
  for (const link of document.querySelectorAll(".main-nav a")) {
    if (link.hash === `#${view}`) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  refresh.disabled = true;
  root.setAttribute("aria-busy", "true");
  root.replaceChildren(notice("Reading local files…", "status"));
  try {
    if (view === "sessions") await renderSessions(controller.signal);
    else if (view === "overview") await renderOverview(controller.signal);
    else await renderRecords(view, controller.signal);
    controller.signal.throwIfAborted();
    document.getElementById("last-updated").textContent = "Updated " + clock(new Date()) + " IST";
  } catch (reason) {
    if (reason.name !== "AbortError") error(root);
  } finally {
    if (currentView === controller) {
      refresh.disabled = false;
      root.setAttribute("aria-busy", "false");
    }
  }
}

window.addEventListener("hashchange", () => {
  if (location.hash !== "#main") render();
});
refresh.addEventListener("click", render);
const theme = document.getElementById("theme-toggle");
theme.addEventListener("click", () => {
  const light = document.documentElement.dataset.theme !== "light";
  document.documentElement.dataset.theme = light ? "light" : "blacki";
  theme.setAttribute("aria-pressed", String(light));
});
const autoRefresh = document.getElementById("auto-refresh");
window.setInterval(() => {
  if (!autoRefresh.checked || document.hidden || refresh.disabled || window.scrollY > 80) return;
  if (root.contains(document.activeElement) || root.querySelector("details[open], .trace-rows")) return;
  if (state.view === "sessions" && state.pane === "wiki") return;
  render();
}, 60_000);
render();
